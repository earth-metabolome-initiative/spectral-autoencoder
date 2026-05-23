//! Inference helper that loads a trained checkpoint and embeds spectra.
//!
//! Wraps the primitives the training loop already uses
//! ([`SpectralAutoencoderConfig::init`] / [`PeakSetAutoencoderConfig::init`],
//! [`Module::load_record`], [`SpectrumVectorizer`] / [`SpectrumTokenizer`],
//! [`ConditioningEncoder`], and the per-row reconstruction-similarity helpers
//! in [`crate::model::reconstruction`]) into a single "load -> embed -> metrics"
//! pipeline. The bin layer is just a clap shell around this type. Downstream
//! library consumers can use it directly without depending on any of the I/O
//! traits in [`crate::embed`].
//!
//! ## Checkpoint layout
//!
//! `from_checkpoint(dir, device)` expects two files in `dir`:
//!
//! - `model-config.json`: a [`SavedSpectrumModelConfig`] (one of `Flat` or
//!   `PeakSet`).
//! - `model.mpk`: the binary Burn record (via [`CompactRecorder`]).
//!
//! The variant on the JSON side decides which autoencoder to build. The
//! record file is loaded into that autoencoder type.
//!
//! ## Output
//!
//! [`embed`](SpectrumEmbedder::embed) returns one [`EmbeddingRow`] per input
//! (or `None` when the input could not be vectorized/tokenized). Each row
//! carries the latent embedding plus three reconstruction-quality signals:
//! linear cosine, modified linear cosine, and log-MSE of the dense
//! reconstruction. The first two come from the differentiable scorers in
//! [`crate::model::reconstruction`]. The third is computed in-line here.

use std::path::{Path, PathBuf};

use burn::data::dataloader::batcher::Batcher;
use burn::module::Module;
use burn::prelude::*;
use burn::record::CompactRecorder;
use burn::tensor::{Tensor, Transaction};
use mass_spectrometry::prelude::{Spectrum, SpectrumAlloc};
use serde::{Deserialize, Serialize};

use crate::batch::{AutoencoderBatch, TokenizedAutoencoderBatch};
use crate::model::reconstruction::{
    normalized_precursor, peak_products, reconstruction_similarity_for_match_mode,
};
use crate::{
    AutoencoderBatcher, AutoencoderSample, ConditioningConfig, ConditioningEncoder, Error,
    PeakSetAutoencoder, PeakSetAutoencoderConfig, Result, SetReconstructionLossConfig,
    SpectralAutoencoder, SpectralAutoencoderConfig, SpectrumTokenizer, SpectrumTokenizerConfig,
    SpectrumVectorizer, SpectrumVectorizerConfig, TokenizedAutoencoderBatcher,
    TokenizedAutoencoderSample,
};

/// Default file name for the JSON model-config under a checkpoint directory.
pub const MODEL_CONFIG_FILE: &str = "model-config.json";
/// Default file name (without extension) for the Burn record under a checkpoint
/// directory. The [`CompactRecorder`] appends `.mpk` itself.
pub const MODEL_RECORD_FILE: &str = "model";

/// On-disk wrapper for the model config. The `variant` tag drives runtime
/// model-type selection in [`SpectrumEmbedder::from_checkpoint`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "variant", content = "config", rename_all = "snake_case")]
pub enum SavedSpectrumModelConfig {
    /// Flat-vector autoencoder.
    Flat(SpectralAutoencoderConfig),
    /// Peak-set transformer autoencoder.
    PeakSet(PeakSetAutoencoderConfig),
}

impl SavedSpectrumModelConfig {
    /// Stable, human-readable name for the variant.
    #[must_use]
    pub const fn variant_name(&self) -> &'static str {
        match self {
            Self::Flat(_) => "flat",
            Self::PeakSet(_) => "peak_set",
        }
    }

    /// Reads a JSON config from disk.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] if the file cannot be read, or [`Error::Json`] if
    /// the file is not valid JSON for this enum.
    pub fn load_json(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Ok(serde_json::from_str(&text)?)
    }

    /// Writes a JSON config to disk.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] if the file cannot be written, or [`Error::Json`]
    /// if serialization fails.
    pub fn save_json(&self, path: &Path) -> Result<()> {
        let text = serde_json::to_string_pretty(self)?;
        std::fs::write(path, text).map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })
    }
}

/// One spectrum's embedder output: latent embedding + reconstruction signals.
///
/// No identifier or precursor metadata is echoed back. The input position
/// is the join key.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingRow {
    /// Latent embedding; length is [`SpectrumEmbedder::latent_width`].
    pub latent: Vec<f32>,
    /// Per-row linear-cosine similarity between target and reconstruction,
    /// clamped to `[0, 1]`. Low values flag inputs the model can't reproduce.
    pub reconstruction_linear_cosine: f32,
    /// Per-row modified-linear-cosine similarity (precursor-shifted match).
    pub reconstruction_modified_linear_cosine: f32,
    /// Per-row mean-squared error on the dense reconstruction. Higher is
    /// further from the input.
    pub reconstruction_log_mse: f32,
}

enum EmbedderVariant<B: Backend> {
    Flat {
        model: Box<SpectralAutoencoder<B>>,
        vectorizer: SpectrumVectorizer,
        vectorizer_config: SpectrumVectorizerConfig,
        loss_config: SetReconstructionLossConfig,
        batcher: AutoencoderBatcher,
        latent_width: usize,
    },
    PeakSet {
        model: Box<PeakSetAutoencoder<B>>,
        tokenizer: SpectrumTokenizer,
        tokenizer_config: SpectrumTokenizerConfig,
        loss_config: SetReconstructionLossConfig,
        batcher: TokenizedAutoencoderBatcher,
        latent_width: usize,
    },
}

/// Loads a trained checkpoint and embeds spectra batches on demand.
///
/// Construct via [`SpectrumEmbedder::from_checkpoint`] (defaults) or
/// [`SpectrumEmbedder::builder`] (settable `skip_errors`). The embedder owns
/// its vectorizer / tokenizer + conditioning encoder, so per-call
/// allocations are amortised across `embed` invocations.
pub struct SpectrumEmbedder<B: Backend> {
    variant: EmbedderVariant<B>,
    conditioning: ConditioningEncoder,
    skip_errors: bool,
    batch_size: usize,
    device: B::Device,
}

/// Default batch size for [`SpectrumEmbedderBuilder`] and
/// [`SpectrumEmbedder::embed_stream`].
pub const DEFAULT_EMBED_BATCH_SIZE: usize = 4096;

/// Fluent builder for [`SpectrumEmbedder`].
///
/// The default behaviour matches [`SpectrumEmbedder::from_checkpoint`]:
/// failed inputs abort the batch with [`Error::InvalidBatch`]. Set
/// `skip_errors(true)` to instead drop failing inputs from the output (in
/// which case the output is a strict subset of the input preserving
/// relative order).
pub struct SpectrumEmbedderBuilder<B: Backend> {
    checkpoint_dir: PathBuf,
    device: B::Device,
    skip_errors: bool,
    batch_size: usize,
}

impl<B: Backend<FloatElem = f32>> SpectrumEmbedderBuilder<B> {
    /// Creates a builder targeting the checkpoint directory.
    #[must_use]
    pub fn new(checkpoint_dir: impl Into<PathBuf>, device: B::Device) -> Self {
        Self {
            checkpoint_dir: checkpoint_dir.into(),
            device,
            skip_errors: false,
            batch_size: DEFAULT_EMBED_BATCH_SIZE,
        }
    }

    /// When `true`, [`SpectrumEmbedder::embed`] silently drops inputs that
    /// cannot be vectorised / tokenised (empty peaks, parse failure). When
    /// `false` (default), the first such failure aborts the entire batch.
    #[inline]
    #[must_use]
    pub fn with_skip_errors(mut self, skip_errors: bool) -> Self {
        self.skip_errors = skip_errors;
        self
    }

    /// Spectra processed per forward pass in
    /// [`SpectrumEmbedder::embed_stream`]. Direct [`SpectrumEmbedder::embed`]
    /// calls bypass this knob. They encode the entire slice they are given.
    ///
    /// Defaults to [`DEFAULT_EMBED_BATCH_SIZE`].
    #[inline]
    #[must_use]
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Builds the embedder by loading the checkpoint.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] or [`Error::Json`] when the config file is
    /// missing or malformed, or surfaces the underlying Burn record-loading
    /// error (wrapped in [`Error::Io`]) if the binary record cannot be
    /// deserialised, or [`Error::InvalidBatch`] when `batch_size == 0`.
    pub fn build(self) -> Result<SpectrumEmbedder<B>> {
        if self.batch_size == 0 {
            return Err(Error::InvalidBatch(
                "SpectrumEmbedderBuilder: batch_size must be greater than zero".into(),
            ));
        }
        let mut embedder =
            SpectrumEmbedder::<B>::from_checkpoint(&self.checkpoint_dir, self.device)?;
        embedder.skip_errors = self.skip_errors;
        embedder.batch_size = self.batch_size;
        Ok(embedder)
    }
}

impl<B: Backend<FloatElem = f32>> SpectrumEmbedder<B> {
    /// Starts a fluent builder targeting `checkpoint_dir`.
    #[must_use]
    pub fn builder(
        checkpoint_dir: impl Into<PathBuf>,
        device: B::Device,
    ) -> SpectrumEmbedderBuilder<B> {
        SpectrumEmbedderBuilder::new(checkpoint_dir, device)
    }

    /// Loads `model-config.json` and `model.mpk` from a training run
    /// directory using defaults (`skip_errors = false`).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] or [`Error::Json`] when the config file is missing
    /// or malformed, or surfaces the underlying Burn record-loading error
    /// (wrapped in [`Error::Io`]) if the binary record cannot be deserialised.
    pub fn from_checkpoint(dir: &Path, device: B::Device) -> Result<Self> {
        let config_path = dir.join(MODEL_CONFIG_FILE);
        let record_path = dir.join(MODEL_RECORD_FILE);
        let saved = SavedSpectrumModelConfig::load_json(&config_path)?;
        let recorder = CompactRecorder::new();
        match saved {
            SavedSpectrumModelConfig::Flat(config) => {
                let model = config.init::<B>(&device);
                let model = model
                    .load_file(record_path.clone(), &recorder, &device)
                    .map_err(|source| Error::Io {
                        path: record_path,
                        source: std::io::Error::other(source.to_string()),
                    })?;
                let vectorizer_config = vectorizer_config_for_flat(&config);
                let vectorizer = SpectrumVectorizer::new(vectorizer_config.clone());
                let latent_width = config.encoder.latent_width;
                let precursor_mz_scale = config.precursor_mz_scale;
                let loss_config = config.loss;
                let conditioning =
                    ConditioningEncoder::new(ConditioningConfig { precursor_mz_scale });
                Ok(Self {
                    variant: EmbedderVariant::Flat {
                        model: Box::new(model),
                        vectorizer,
                        vectorizer_config,
                        loss_config,
                        batcher: AutoencoderBatcher,
                        latent_width,
                    },
                    conditioning,
                    skip_errors: false,
                    batch_size: DEFAULT_EMBED_BATCH_SIZE,
                    device,
                })
            }
            SavedSpectrumModelConfig::PeakSet(config) => {
                let model = config.init::<B>(&device);
                let model = model
                    .load_file(record_path.clone(), &recorder, &device)
                    .map_err(|source| Error::Io {
                        path: record_path,
                        source: std::io::Error::other(source.to_string()),
                    })?;
                let tokenizer_config = tokenizer_config_for_peak_set(&config);
                let tokenizer = SpectrumTokenizer::new(tokenizer_config.clone());
                let latent_width = config.encoder.latent_width;
                let precursor_mz_scale = config.precursor_mz_scale;
                let loss_config = config.loss;
                let conditioning =
                    ConditioningEncoder::new(ConditioningConfig { precursor_mz_scale });
                Ok(Self {
                    variant: EmbedderVariant::PeakSet {
                        model: Box::new(model),
                        tokenizer,
                        tokenizer_config,
                        loss_config,
                        batcher: TokenizedAutoencoderBatcher,
                        latent_width,
                    },
                    conditioning,
                    skip_errors: false,
                    batch_size: DEFAULT_EMBED_BATCH_SIZE,
                    device,
                })
            }
        }
    }

    /// Latent embedding width.
    #[must_use]
    pub const fn latent_width(&self) -> usize {
        match &self.variant {
            EmbedderVariant::Flat { latent_width, .. }
            | EmbedderVariant::PeakSet { latent_width, .. } => *latent_width,
        }
    }

    /// Variant of the loaded model (`"flat"` or `"peak_set"`).
    #[must_use]
    pub const fn variant_name(&self) -> &'static str {
        match &self.variant {
            EmbedderVariant::Flat { .. } => "flat",
            EmbedderVariant::PeakSet { .. } => "peak_set",
        }
    }

    /// Batch size used by [`Self::embed_stream`].
    #[must_use]
    pub const fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Encodes a batch of spectra into latent embeddings plus reconstruction
    /// metrics.
    ///
    /// Output rows are positional: when the embedder was built with the default
    /// `skip_errors = false`, returned `Vec` length equals `inputs.len()` and
    /// row `i` corresponds to input row `i`. With `skip_errors = true`, the
    /// returned `Vec` is a strict subset of the input, preserving relative
    /// order; the caller cannot recover the input position of skipped rows
    /// from the embedder.
    ///
    /// # Errors
    ///
    /// With `skip_errors = false` (default), returns [`Error::InvalidBatch`]
    /// at the first input that fails to vectorise / tokenise. Also returns
    /// an error when the device-side tensor transaction fails to materialise
    /// the output scalars.
    pub fn embed<S>(&mut self, spectra: &[S]) -> Result<Vec<EmbeddingRow>>
    where
        S: SpectrumAlloc<Precision = f32>,
        S::MutationError: Into<Error>,
    {
        if spectra.is_empty() {
            return Ok(Vec::new());
        }
        match &mut self.variant {
            EmbedderVariant::Flat { .. } => self.embed_flat(spectra),
            EmbedderVariant::PeakSet { .. } => self.embed_peak_set(spectra),
        }
    }

    fn embed_flat<S>(&self, spectra: &[S]) -> Result<Vec<EmbeddingRow>>
    where
        S: SpectrumAlloc<Precision = f32>,
        S::MutationError: Into<Error>,
    {
        let EmbedderVariant::Flat {
            model,
            vectorizer,
            vectorizer_config,
            loss_config,
            batcher,
            latent_width,
        } = &self.variant
        else {
            unreachable!("embed_flat dispatched with non-Flat variant");
        };

        let mut samples: Vec<AutoencoderSample> = Vec::with_capacity(spectra.len());
        let mut row_to_sample: Vec<Option<usize>> = Vec::with_capacity(spectra.len());
        for (index, spectrum) in spectra.iter().enumerate() {
            match vectorizer.encode(spectrum) {
                Ok(vector) => {
                    let conditions = self
                        .conditioning
                        .encode_precursor_mz(finite_positive_precursor(spectrum));
                    row_to_sample.push(Some(samples.len()));
                    samples.push(AutoencoderSample {
                        spectrum: vector.values,
                        conditions,
                    });
                }
                Err(_) => {
                    if !self.skip_errors {
                        return Err(Error::InvalidBatch(format!(
                            "input row {index} could not be vectorised (no usable peaks)"
                        )));
                    }
                    row_to_sample.push(None);
                }
            }
        }

        if samples.is_empty() {
            return Ok(Vec::new());
        }

        let batch: AutoencoderBatch<B> = batcher.batch(samples, &self.device);
        let target_spectra = batch.target_spectra.clone();
        let target_conditions = batch.target_conditions.clone();
        let output = model.forward(batch.spectra, batch.conditions);

        let mz_per_peak = vectorizer_config.max_peaks;
        let (linear_cosine_per_row, modified_per_row) = flat_per_row_similarities(
            output.reconstruction.clone(),
            target_spectra.clone(),
            target_conditions,
            output.condition_reconstruction.clone(),
            mz_per_peak,
            *loss_config,
        );
        let log_mse_per_row = per_row_log_mse(output.reconstruction.clone(), target_spectra);

        materialise_rows(
            &row_to_sample,
            output.latent,
            linear_cosine_per_row,
            modified_per_row,
            log_mse_per_row,
            *latent_width,
        )
    }

    fn embed_peak_set<S>(&self, spectra: &[S]) -> Result<Vec<EmbeddingRow>>
    where
        S: SpectrumAlloc<Precision = f32>,
        S::MutationError: Into<Error>,
    {
        let EmbedderVariant::PeakSet {
            model,
            tokenizer,
            tokenizer_config,
            loss_config,
            batcher,
            latent_width,
        } = &self.variant
        else {
            unreachable!("embed_peak_set dispatched with non-PeakSet variant");
        };

        let mut samples: Vec<TokenizedAutoencoderSample> = Vec::with_capacity(spectra.len());
        let mut row_to_sample: Vec<Option<usize>> = Vec::with_capacity(spectra.len());
        for (index, spectrum) in spectra.iter().enumerate() {
            match tokenizer.encode(spectrum) {
                Ok(tokens) => {
                    let conditions = self
                        .conditioning
                        .encode_precursor_mz(finite_positive_precursor(spectrum));
                    row_to_sample.push(Some(samples.len()));
                    samples.push(TokenizedAutoencoderSample {
                        token_features: tokens.features,
                        target_pairs: tokens.target_pairs,
                        peak_mask: tokens.peak_mask,
                        padding_mask: tokens.padding_mask,
                        conditions,
                    });
                }
                Err(_) => {
                    if !self.skip_errors {
                        return Err(Error::InvalidBatch(format!(
                            "input row {index} could not be tokenised (no usable peaks)"
                        )));
                    }
                    row_to_sample.push(None);
                }
            }
        }

        if samples.is_empty() {
            return Ok(Vec::new());
        }

        let batch: TokenizedAutoencoderBatch<B> = batcher.batch(samples, &self.device);
        let target_pairs = batch.target_pairs.clone();
        let target_peak_mask = batch.target_peak_mask.clone();
        let target_conditions = batch.target_conditions.clone();

        let output = model.forward(
            batch.token_features,
            batch.peak_mask,
            batch.padding_mask,
            batch.conditions,
        );

        let max_peaks = tokenizer_config.max_peaks;
        let (linear_cosine_per_row, modified_per_row) = peak_set_per_row_similarities(
            output.reconstruction.clone(),
            target_pairs.clone(),
            target_peak_mask,
            target_conditions,
            output.condition_reconstruction.clone(),
            max_peaks,
            *loss_config,
        );
        let log_mse_per_row =
            per_row_log_mse_from_triples(output.reconstruction.clone(), target_pairs);

        materialise_rows(
            &row_to_sample,
            output.latent,
            linear_cosine_per_row,
            modified_per_row,
            log_mse_per_row,
            *latent_width,
        )
    }

    /// Streams `spectra` through the embedder in chunks of
    /// [`Self::batch_size`] and yields one [`EmbeddingRow`] per
    /// successfully-encoded input in input order. Failed inputs surface per
    /// the embedder's `skip_errors` policy: with the default
    /// `skip_errors = false`, the first failure yields one `Err(...)` and
    /// the iterator terminates; with `skip_errors = true`, failures are
    /// silently dropped from the output.
    ///
    /// The returned iterator borrows `self` for `'_`; multiple concurrent
    /// streams on the same embedder are not possible (the embedder owns
    /// mutable scratch state).
    pub fn embed_stream<I, S>(&mut self, spectra: I) -> EmbedStream<'_, B, S, I::IntoIter>
    where
        I: IntoIterator<Item = S>,
        S: SpectrumAlloc<Precision = f32>,
        S::MutationError: Into<Error>,
    {
        EmbedStream {
            embedder: self,
            source: spectra.into_iter(),
            pending: Vec::new().into_iter(),
            fused: false,
        }
    }
}

/// Iterator returned by [`SpectrumEmbedder::embed_stream`].
pub struct EmbedStream<'a, B, S, I>
where
    B: Backend<FloatElem = f32>,
    S: SpectrumAlloc<Precision = f32>,
    S::MutationError: Into<Error>,
    I: Iterator<Item = S>,
{
    embedder: &'a mut SpectrumEmbedder<B>,
    source: I,
    pending: std::vec::IntoIter<EmbeddingRow>,
    fused: bool,
}

impl<'a, B, S, I> Iterator for EmbedStream<'a, B, S, I>
where
    B: Backend<FloatElem = f32>,
    S: SpectrumAlloc<Precision = f32>,
    S::MutationError: Into<Error>,
    I: Iterator<Item = S>,
{
    type Item = Result<EmbeddingRow>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(row) = self.pending.next() {
            return Some(Ok(row));
        }
        if self.fused {
            return None;
        }
        let mut buffer: Vec<S> = Vec::with_capacity(self.embedder.batch_size);
        for _ in 0..self.embedder.batch_size {
            match self.source.next() {
                Some(spectrum) => buffer.push(spectrum),
                None => break,
            }
        }
        if buffer.is_empty() {
            self.fused = true;
            return None;
        }
        match self.embedder.embed(&buffer) {
            Ok(rows) => {
                self.pending = rows.into_iter();
                let next_row = self.pending.next();
                if next_row.is_none() && self.source.size_hint().1 == Some(0) {
                    // Source exhausted and the batch produced no rows
                    // (all skipped under skip_errors). Stop iterating.
                    self.fused = true;
                }
                next_row.map(Ok)
            }
            Err(error) => {
                self.fused = true;
                Some(Err(error))
            }
        }
    }
}

fn vectorizer_config_for_flat(config: &SpectralAutoencoderConfig) -> SpectrumVectorizerConfig {
    let max_peaks = config.encoder.spectrum_width / 2;
    SpectrumVectorizerConfig {
        max_peaks,
        ..SpectrumVectorizerConfig::default()
    }
}

fn tokenizer_config_for_peak_set(config: &PeakSetAutoencoderConfig) -> SpectrumTokenizerConfig {
    let extra_features = config.encoder.token_feature_width.saturating_sub(3);
    SpectrumTokenizerConfig {
        max_peaks: config.encoder.max_peaks,
        mz_fourier_frequencies: extra_features / 2,
        ..SpectrumTokenizerConfig::default()
    }
}

fn finite_positive_precursor<S>(spectrum: &S) -> Option<f64>
where
    S: Spectrum<Precision = f32>,
{
    let value = spectrum.precursor_mz();
    (value.is_finite() && value > 0.0).then(|| f64::from(value))
}

fn flat_per_row_similarities<B: Backend>(
    reconstruction: Tensor<B, 2>,
    target: Tensor<B, 2>,
    target_conditions: Tensor<B, 2>,
    reconstructed_conditions: Tensor<B, 2>,
    max_peaks: usize,
    config: SetReconstructionLossConfig,
) -> (Tensor<B, 1>, Tensor<B, 1>) {
    let [batch_size, vector_width] = reconstruction.dims();
    debug_assert_eq!(vector_width, max_peaks * 2);
    let reconstruction_triples = reconstruction.reshape([batch_size, max_peaks, 2]);
    let target_triples = target.clone().reshape([batch_size, max_peaks, 2]);
    let target_mask = target
        .reshape([batch_size, max_peaks, 2])
        .narrow(2, 1, 1)
        .reshape([batch_size, max_peaks])
        .greater_elem(0.0)
        .float();
    triples_per_row_similarities(
        reconstruction_triples,
        target_triples,
        target_mask,
        target_conditions,
        reconstructed_conditions,
        config,
    )
}

fn peak_set_per_row_similarities<B: Backend>(
    reconstruction: Tensor<B, 3>,
    target_pairs: Tensor<B, 2>,
    target_mask: Tensor<B, 2>,
    target_conditions: Tensor<B, 2>,
    reconstructed_conditions: Tensor<B, 2>,
    max_peaks: usize,
    config: SetReconstructionLossConfig,
) -> (Tensor<B, 1>, Tensor<B, 1>) {
    let [batch_size, _, _] = reconstruction.dims();
    let target_triples = target_pairs.reshape([batch_size, max_peaks, 2]);
    triples_per_row_similarities(
        reconstruction,
        target_triples,
        target_mask,
        target_conditions,
        reconstructed_conditions,
        config,
    )
}

fn triples_per_row_similarities<B: Backend>(
    reconstruction: Tensor<B, 3>,
    target_triples: Tensor<B, 3>,
    target_mask: Tensor<B, 2>,
    target_conditions: Tensor<B, 2>,
    reconstructed_conditions: Tensor<B, 2>,
    config: SetReconstructionLossConfig,
) -> (Tensor<B, 1>, Tensor<B, 1>) {
    let [batch_size, max_peaks, output_width] = reconstruction.dims();
    let pred_mz = reconstruction
        .clone()
        .narrow(2, 0, 1)
        .reshape([batch_size, max_peaks]);
    let pred_intensity = reconstruction
        .clone()
        .narrow(2, 1, 1)
        .reshape([batch_size, max_peaks]);
    let pred_presence = if output_width >= 3 {
        reconstruction
            .narrow(2, 2, 1)
            .reshape([batch_size, max_peaks])
    } else {
        pred_intensity.ones_like()
    };
    let target_mz = target_triples
        .clone()
        .narrow(2, 0, 1)
        .reshape([batch_size, max_peaks]);
    let target_intensity = target_triples
        .narrow(2, 1, 1)
        .reshape([batch_size, max_peaks]);

    let pred_products = peak_products(
        pred_mz.clone(),
        pred_intensity,
        pred_presence.clamp_min(0.0).clamp_max(1.0),
        config,
    );
    let target_products = peak_products(
        target_mz.clone(),
        target_intensity,
        target_mask.clone(),
        config,
    );
    let target_precursor = normalized_precursor(target_conditions);
    let pred_precursor = normalized_precursor(reconstructed_conditions);

    let linear_cosine = reconstruction_similarity_for_match_mode(
        pred_mz.clone(),
        pred_products.clone(),
        pred_precursor.clone(),
        target_mz.clone(),
        target_products.clone(),
        target_mask.clone(),
        target_precursor.clone(),
        config.normalized_mz_tolerance,
        false,
    );
    let modified = reconstruction_similarity_for_match_mode(
        pred_mz,
        pred_products,
        pred_precursor,
        target_mz,
        target_products,
        target_mask,
        target_precursor,
        config.normalized_mz_tolerance,
        true,
    );
    (linear_cosine, modified)
}

fn per_row_log_mse<B: Backend>(reconstruction: Tensor<B, 2>, target: Tensor<B, 2>) -> Tensor<B, 1> {
    let [batch_size, vector_width] = reconstruction.dims();
    let diff = reconstruction - target;
    diff.powf_scalar(2.0)
        .sum_dim(1)
        .reshape([batch_size])
        .div_scalar(vector_width as f32)
}

fn per_row_log_mse_from_triples<B: Backend>(
    reconstruction: Tensor<B, 3>,
    target_pairs: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let [batch_size, max_peaks, output_width] = reconstruction.dims();
    let target_triples = target_pairs.reshape([batch_size, max_peaks, 2]);
    let predicted_pairs = reconstruction.narrow(2, 0, 2);
    debug_assert!(output_width >= 2);
    let diff = predicted_pairs - target_triples;
    let elements = (max_peaks * 2) as f32;
    diff.powf_scalar(2.0)
        .sum_dim(2)
        .sum_dim(1)
        .reshape([batch_size])
        .div_scalar(elements)
}

fn materialise_rows<B: Backend<FloatElem = f32>>(
    row_to_sample: &[Option<usize>],
    latent: Tensor<B, 2>,
    linear_cosine: Tensor<B, 1>,
    modified: Tensor<B, 1>,
    log_mse: Tensor<B, 1>,
    latent_width: usize,
) -> Result<Vec<EmbeddingRow>> {
    let [latent_rows, _] = latent.dims();
    let bundle = Transaction::default()
        .register(latent)
        .register(linear_cosine)
        .register(modified)
        .register(log_mse)
        .try_execute()
        .map_err(|source| Error::Io {
            path: PathBuf::from("<transaction>"),
            source: std::io::Error::other(source.to_string()),
        })?;
    let mut iter = bundle.into_iter();
    let latent_data = iter.next().ok_or_else(missing_transaction_output)?;
    let linear_data = iter.next().ok_or_else(missing_transaction_output)?;
    let modified_data = iter.next().ok_or_else(missing_transaction_output)?;
    let log_mse_data = iter.next().ok_or_else(missing_transaction_output)?;
    let latent_values = latent_data.as_slice::<f32>().map_err(latent_dtype_error)?;
    let linear_values = linear_data.as_slice::<f32>().map_err(latent_dtype_error)?;
    let modified_values = modified_data
        .as_slice::<f32>()
        .map_err(latent_dtype_error)?;
    let log_mse_values = log_mse_data.as_slice::<f32>().map_err(latent_dtype_error)?;
    debug_assert_eq!(latent_values.len(), latent_rows * latent_width);

    let rows = row_to_sample
        .iter()
        .filter_map(|sample_slot| {
            sample_slot.map(|sample_index| {
                let start = sample_index * latent_width;
                let end = start + latent_width;
                EmbeddingRow {
                    latent: latent_values[start..end].to_vec(),
                    reconstruction_linear_cosine: linear_values[sample_index],
                    reconstruction_modified_linear_cosine: modified_values[sample_index],
                    reconstruction_log_mse: log_mse_values[sample_index],
                }
            })
        })
        .collect();
    Ok(rows)
}

fn missing_transaction_output() -> Error {
    Error::Io {
        path: PathBuf::from("<transaction>"),
        source: std::io::Error::other("embed transaction returned fewer tensors than expected"),
    }
}

fn latent_dtype_error(source: impl std::fmt::Display) -> Error {
    Error::Io {
        path: PathBuf::from("<transaction>"),
        source: std::io::Error::other(format!("transaction tensor not f32: {source}")),
    }
}

#[cfg(all(test, feature = "ndarray"))]
mod tests {
    use super::*;
    use crate::{
        DecoderConfig, EncoderConfig, PeakSetDecoderConfig, PeakSetEncoderConfig,
        SetReconstructionLossConfig,
    };
    use burn::backend::NdArray;
    use burn::backend::ndarray::NdArrayDevice;
    use tempfile::TempDir;

    type B = NdArray<f32, i64>;

    fn save_then_reload_round_trip<F>(config: SavedSpectrumModelConfig, init: F)
    where
        F: FnOnce(&NdArrayDevice) -> Box<dyn FnOnce(&Path)>,
    {
        let dir = TempDir::new().expect("tempdir");
        let config_path = dir.path().join(MODEL_CONFIG_FILE);
        config.save_json(&config_path).expect("save json");

        let loaded = SavedSpectrumModelConfig::load_json(&config_path).expect("load json");
        assert_eq!(loaded.variant_name(), config.variant_name());

        let device = NdArrayDevice::default();
        init(&device)(dir.path());
    }

    #[test]
    fn saved_config_round_trips_for_flat() {
        let encoder = EncoderConfig::builder()
            .with_spectrum_width(8)
            .with_condition_width(2)
            .with_hidden_widths(vec![16, 16])
            .with_latent_width(4)
            .build()
            .expect("encoder config");
        let decoder = DecoderConfig::builder()
            .with_latent_width(4)
            .with_condition_width(0)
            .with_hidden_widths(vec![16])
            .with_spectrum_width(8)
            .with_condition_output_width(2)
            .build()
            .expect("decoder config");
        let model_config = SpectralAutoencoderConfig::builder()
            .with_encoder(encoder)
            .with_decoder(decoder)
            .build()
            .expect("autoencoder config");
        let saved = SavedSpectrumModelConfig::Flat(model_config);
        save_then_reload_round_trip(saved, |_device| {
            Box::new(move |_dir| {
                // Round trip done by save_then_reload_round_trip itself.
            })
        });
    }

    #[test]
    fn embed_returns_empty_for_empty_input() {
        let device = NdArrayDevice::default();
        let encoder_config = EncoderConfig::builder()
            .with_spectrum_width(4)
            .with_condition_width(2)
            .with_hidden_widths(vec![8])
            .with_latent_width(4)
            .build()
            .expect("encoder config");
        let decoder_config = DecoderConfig::builder()
            .with_latent_width(4)
            .with_condition_width(0)
            .with_hidden_widths(vec![8])
            .with_spectrum_width(4)
            .with_condition_output_width(2)
            .build()
            .expect("decoder config");
        let model_config = SpectralAutoencoderConfig::builder()
            .with_encoder(encoder_config)
            .with_decoder(decoder_config)
            .build()
            .expect("autoencoder config");
        let model = model_config.init::<B>(&device);
        let vectorizer_config = SpectrumVectorizerConfig {
            max_peaks: 2,
            ..SpectrumVectorizerConfig::default()
        };
        let mut embedder = SpectrumEmbedder::<B> {
            variant: EmbedderVariant::Flat {
                model: Box::new(model),
                vectorizer: SpectrumVectorizer::new(vectorizer_config.clone()),
                vectorizer_config,
                loss_config: SetReconstructionLossConfig::default(),
                batcher: AutoencoderBatcher,
                latent_width: 4,
            },
            conditioning: ConditioningEncoder::default(),
            skip_errors: false,
            batch_size: DEFAULT_EMBED_BATCH_SIZE,
            device,
        };
        assert_eq!(
            embedder
                .embed::<mass_spectrometry::prelude::GenericSpectrum<f32>>(&[])
                .expect("embed empty"),
            Vec::new()
        );
        assert_eq!(embedder.latent_width(), 4);
        assert_eq!(embedder.variant_name(), "flat");
    }

    #[test]
    fn peak_set_variant_round_trip() {
        let encoder = PeakSetEncoderConfig::builder()
            .with_max_peaks(4)
            .with_token_feature_width(SpectrumTokenizerConfig::default().feature_width())
            .with_condition_width(2)
            .with_token_embedding_width(8)
            .with_attention_heads(2)
            .with_transformer_layers(1)
            .with_transformer_feed_forward_width(16)
            .with_dropout(0.0)
            .with_hidden_widths(vec![8])
            .with_latent_width(4)
            .build()
            .expect("encoder config");
        let decoder = PeakSetDecoderConfig::builder()
            .with_max_peaks(4)
            .with_latent_width(4)
            .with_condition_width(0)
            .with_query_width(8)
            .with_attention_heads(2)
            .with_decoder_layers(1)
            .with_decoder_feed_forward_width(16)
            .with_dropout(0.0)
            .with_condition_output_width(2)
            .build()
            .expect("decoder config");
        let model_config = PeakSetAutoencoderConfig::builder()
            .with_encoder(encoder)
            .with_decoder(decoder)
            .with_loss(SetReconstructionLossConfig::default())
            .build()
            .expect("autoencoder config");
        let saved = SavedSpectrumModelConfig::PeakSet(model_config);
        assert_eq!(saved.variant_name(), "peak_set");
        let dir = TempDir::new().expect("tempdir");
        saved
            .save_json(&dir.path().join(MODEL_CONFIG_FILE))
            .expect("save");
        let reloaded =
            SavedSpectrumModelConfig::load_json(&dir.path().join(MODEL_CONFIG_FILE)).expect("load");
        assert_eq!(reloaded.variant_name(), "peak_set");
    }
}
