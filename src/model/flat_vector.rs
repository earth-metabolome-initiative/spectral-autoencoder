//! Flat vector autoencoder baseline.

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use burn::{
    nn::{Linear, LinearConfig, Relu},
    prelude::*,
    tensor::{Tensor, activation::sigmoid},
};
use serde::{Deserialize, Serialize};

use crate::{
    conditioning::ConditioningConfig,
    model::RegularizationConfig,
    model::auxiliary::{
        AuxiliaryLossConfig, EmbeddingAuxiliaryHeads, EmbeddingAuxiliaryHeadsConfig,
    },
    model::reconstruction::{
        FlatVectorReconstructionOrdering, SetReconstructionLossConfig,
        flat_vector_reconstruction_loss_from_vectors,
    },
    vectorize::SpectrumVectorizerConfig,
};

fn default_precursor_mz_scale() -> f64 {
    ConditioningConfig::default().precursor_mz_scale()
}

/// Default number of training epochs over which the m/z Gaussian gate
/// bandwidth `σ` decays from `mz_tolerance_start` to `mz_tolerance_end`.
/// Chosen as a bounded window so long training runs spend most of their
/// epochs at the tight σ_end for precision; capped to the total epoch count
/// at the call site so short runs still complete the decay.
pub const DEFAULT_MZ_TOLERANCE_DECAY_EPOCHS: usize = 50;

#[cfg(feature = "train")]
use crate::{
    model::auxiliary::{
        apply_latent_noise, weighted_intruder_detection_loss,
        weighted_masked_precursor_reconstruction_output, weighted_precursor_reconstruction_output,
        weighted_similarity_ranking_output,
    },
    model::reconstruction::{
        flat_vector_reconstruction_losses_from_vectors_with_masks, slot_chamfer_magnet_mz,
        vector_element_mask_to_peak_mask, vector_target_mask,
    },
    training::{AutoencoderDiagnostics, AutoencoderLossBreakdown},
};

/// Encoder model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncoderConfig {
    /// Width of the vectorized spectrum.
    pub spectrum_width: usize,
    /// Width of the encoder-side metadata conditioning vector.
    pub condition_width: usize,
    /// Hidden layer widths.
    pub hidden_widths: Vec<usize>,
    /// Latent embedding width.
    pub latent_width: usize,
}

impl EncoderConfig {
    /// Starts a fluent builder. All four fields are required before
    /// [`EncoderConfigBuilder::build`].
    #[must_use]
    pub fn builder() -> EncoderConfigBuilder {
        EncoderConfigBuilder::default()
    }

    /// Creates an initialized encoder.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Encoder<B> {
        let mut input_width = self.spectrum_width + self.condition_width;
        let mut layers = Vec::with_capacity(self.hidden_widths.len());
        for &hidden_width in &self.hidden_widths {
            layers.push(LinearConfig::new(input_width, hidden_width).init(device));
            input_width = hidden_width;
        }

        Encoder {
            layers,
            latent: LinearConfig::new(input_width, self.latent_width).init(device),
            activation: Relu::new(),
        }
    }
}

/// Fluent builder for the flat-vector [`EncoderConfig`].
#[derive(Debug, Clone, Default)]
pub struct EncoderConfigBuilder {
    spectrum_width: Option<usize>,
    condition_width: Option<usize>,
    hidden_widths: Option<Vec<usize>>,
    latent_width: Option<usize>,
}

impl EncoderConfigBuilder {
    /// Creates an empty builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the vectorized-spectrum width (required).
    #[inline]
    #[must_use]
    pub fn with_spectrum_width(mut self, value: usize) -> Self {
        self.spectrum_width = Some(value);
        self
    }

    /// Sets the encoder-side metadata-condition width (required; may be `0`).
    #[inline]
    #[must_use]
    pub fn with_condition_width(mut self, value: usize) -> Self {
        self.condition_width = Some(value);
        self
    }

    /// Sets the hidden-layer widths (required).
    #[inline]
    #[must_use]
    pub fn with_hidden_widths(mut self, value: Vec<usize>) -> Self {
        self.hidden_widths = Some(value);
        self
    }

    /// Sets the latent embedding width (required).
    #[inline]
    #[must_use]
    pub fn with_latent_width(mut self, value: usize) -> Self {
        self.latent_width = Some(value);
        self
    }

    /// Builds the config.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::IncompleteBuilder`] when any required field is
    /// unset.
    pub fn build(self) -> crate::Result<EncoderConfig> {
        let spectrum_width =
            self.spectrum_width
                .ok_or_else(|| crate::Error::IncompleteBuilder {
                    config: "EncoderConfig",
                    field: "spectrum_width",
                })?;
        let condition_width =
            self.condition_width
                .ok_or_else(|| crate::Error::IncompleteBuilder {
                    config: "EncoderConfig",
                    field: "condition_width",
                })?;
        let hidden_widths = self
            .hidden_widths
            .ok_or_else(|| crate::Error::IncompleteBuilder {
                config: "EncoderConfig",
                field: "hidden_widths",
            })?;
        let latent_width = self
            .latent_width
            .ok_or_else(|| crate::Error::IncompleteBuilder {
                config: "EncoderConfig",
                field: "latent_width",
            })?;
        Ok(EncoderConfig {
            spectrum_width,
            condition_width,
            hidden_widths,
            latent_width,
        })
    }
}

/// Decoder model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecoderConfig {
    /// Width of the latent embedding.
    pub latent_width: usize,
    /// Width of the decoder-side metadata conditioning vector.
    ///
    /// The diffusion-oriented default is zero so the decoder reconstructs from
    /// the latent embedding alone.
    pub condition_width: usize,
    /// Hidden layer widths.
    pub hidden_widths: Vec<usize>,
    /// Width of the reconstructed spectrum vector.
    pub spectrum_width: usize,
    /// Width of the reconstructed metadata condition vector.
    pub condition_output_width: usize,
}

impl DecoderConfig {
    /// Starts a fluent builder. All fields are required before
    /// [`DecoderConfigBuilder::build`].
    #[must_use]
    pub fn builder() -> DecoderConfigBuilder {
        DecoderConfigBuilder::default()
    }

    /// Creates an initialized decoder.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Decoder<B> {
        let mut input_width = self.latent_width + self.condition_width;
        let mut layers = Vec::with_capacity(self.hidden_widths.len());
        for &hidden_width in &self.hidden_widths {
            layers.push(LinearConfig::new(input_width, hidden_width).init(device));
            input_width = hidden_width;
        }

        Decoder {
            layers,
            spectrum_output: LinearConfig::new(input_width, self.spectrum_width).init(device),
            condition_output: LinearConfig::new(input_width, self.condition_output_width)
                .init(device),
            activation: Relu::new(),
            condition_width: self.condition_width,
        }
    }
}

/// Fluent builder for the flat-vector [`DecoderConfig`].
#[derive(Debug, Clone, Default)]
pub struct DecoderConfigBuilder {
    latent_width: Option<usize>,
    condition_width: Option<usize>,
    hidden_widths: Option<Vec<usize>>,
    spectrum_width: Option<usize>,
    condition_output_width: Option<usize>,
}

impl DecoderConfigBuilder {
    /// Creates an empty builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the latent embedding width (required).
    #[inline]
    #[must_use]
    pub fn with_latent_width(mut self, value: usize) -> Self {
        self.latent_width = Some(value);
        self
    }

    /// Sets the decoder-side metadata-condition width (required; may be `0`).
    #[inline]
    #[must_use]
    pub fn with_condition_width(mut self, value: usize) -> Self {
        self.condition_width = Some(value);
        self
    }

    /// Sets the hidden-layer widths (required).
    #[inline]
    #[must_use]
    pub fn with_hidden_widths(mut self, value: Vec<usize>) -> Self {
        self.hidden_widths = Some(value);
        self
    }

    /// Sets the reconstructed spectrum width (required).
    #[inline]
    #[must_use]
    pub fn with_spectrum_width(mut self, value: usize) -> Self {
        self.spectrum_width = Some(value);
        self
    }

    /// Sets the reconstructed metadata-condition width (required).
    #[inline]
    #[must_use]
    pub fn with_condition_output_width(mut self, value: usize) -> Self {
        self.condition_output_width = Some(value);
        self
    }

    /// Builds the config.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::IncompleteBuilder`] when any required field is
    /// unset.
    pub fn build(self) -> crate::Result<DecoderConfig> {
        let latent_width = self
            .latent_width
            .ok_or_else(|| crate::Error::IncompleteBuilder {
                config: "DecoderConfig",
                field: "latent_width",
            })?;
        let condition_width =
            self.condition_width
                .ok_or_else(|| crate::Error::IncompleteBuilder {
                    config: "DecoderConfig",
                    field: "condition_width",
                })?;
        let hidden_widths = self
            .hidden_widths
            .ok_or_else(|| crate::Error::IncompleteBuilder {
                config: "DecoderConfig",
                field: "hidden_widths",
            })?;
        let spectrum_width =
            self.spectrum_width
                .ok_or_else(|| crate::Error::IncompleteBuilder {
                    config: "DecoderConfig",
                    field: "spectrum_width",
                })?;
        let condition_output_width =
            self.condition_output_width
                .ok_or_else(|| crate::Error::IncompleteBuilder {
                    config: "DecoderConfig",
                    field: "condition_output_width",
                })?;
        Ok(DecoderConfig {
            latent_width,
            condition_width,
            hidden_widths,
            spectrum_width,
            condition_output_width,
        })
    }
}

/// Full autoencoder configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralAutoencoderConfig {
    /// Encoder configuration.
    pub encoder: EncoderConfig,
    /// Decoder configuration.
    pub decoder: DecoderConfig,
    /// Set reconstruction loss configuration.
    #[serde(default)]
    pub loss: SetReconstructionLossConfig,
    /// How flat-vector peaks are aligned before reconstruction loss.
    #[serde(default)]
    pub reconstruction_ordering: FlatVectorReconstructionOrdering,
    /// Scale used to denormalize reconstructed precursor m/z diagnostics.
    #[serde(default = "default_precursor_mz_scale")]
    pub precursor_mz_scale: f64,
    /// L1/L2 model-parameter regularization.
    #[serde(default)]
    pub regularization: RegularizationConfig,
    /// Auxiliary denoising, intruder, precursor, and similarity-ranking objectives.
    #[serde(default)]
    pub auxiliary: AuxiliaryLossConfig,
    /// Initial value of the m/z Gaussian gate bandwidth `σ`. Defaults to
    /// `loss.normalized_mz_tolerance` when the annealing schedule is unused.
    #[serde(default)]
    pub mz_sigma_start: f64,
    /// Final value of `σ` reached after `mz_sigma_decay_steps` training steps.
    /// Defaults to `loss.normalized_mz_tolerance` so the schedule is a no-op
    /// unless the user opts in.
    #[serde(default)]
    pub mz_sigma_end: f64,
    /// Number of training steps over which `σ` linearly decays from start to
    /// end. `0` disables annealing (σ stays at `mz_sigma_start`).
    #[serde(default)]
    pub mz_sigma_decay_steps: usize,
}

impl SpectralAutoencoderConfig {
    /// Starts a fluent builder. `encoder` and `decoder` are required;
    /// remaining fields fall back to their type's default.
    #[must_use]
    pub fn builder() -> SpectralAutoencoderConfigBuilder {
        SpectralAutoencoderConfigBuilder::default()
    }

    /// Starting flat-vector baseline configuration for the 20M-spectrum run.
    ///
    /// This intentionally remains a compressive baseline over top-N
    /// `(m/z, intensity)` pairs, but uses enough width for the harder
    /// top-128 reconstruction and similarity-ranking setup.
    #[must_use]
    pub fn twenty_million_run() -> Self {
        Self::twenty_million_run_with_peaks(SpectrumVectorizerConfig::default().max_peaks)
    }

    /// Starting flat-vector baseline configuration for a top-N GeMS run.
    #[must_use]
    pub fn twenty_million_run_with_peaks(max_peaks: usize) -> Self {
        Self::symmetric(
            SpectrumVectorizerConfig {
                max_peaks,
                ..SpectrumVectorizerConfig::default()
            }
            .vector_width(),
            ConditioningConfig::default().vector_width(),
            256,
            vec![4096, 2048, 1024, 512],
        )
        .with_reconstruction_ordering(FlatVectorReconstructionOrdering::IntensityDescending)
    }

    /// Creates a symmetric deterministic autoencoder configuration.
    #[must_use]
    pub fn symmetric(
        spectrum_width: usize,
        condition_width: usize,
        latent_width: usize,
        hidden_widths: Vec<usize>,
    ) -> Self {
        Self::symmetric_with_decoder_conditioning(
            spectrum_width,
            condition_width,
            0,
            latent_width,
            hidden_widths,
        )
    }

    /// Creates a symmetric deterministic autoencoder configuration with
    /// explicit encoder and decoder conditioning widths.
    #[must_use]
    pub fn symmetric_with_decoder_conditioning(
        spectrum_width: usize,
        encoder_condition_width: usize,
        decoder_condition_width: usize,
        latent_width: usize,
        hidden_widths: Vec<usize>,
    ) -> Self {
        let mut decoder_hidden = hidden_widths.clone();
        decoder_hidden.reverse();
        Self {
            encoder: EncoderConfig {
                spectrum_width,
                condition_width: encoder_condition_width,
                hidden_widths,
                latent_width,
            },
            decoder: DecoderConfig {
                latent_width,
                condition_width: decoder_condition_width,
                hidden_widths: decoder_hidden,
                spectrum_width,
                condition_output_width: encoder_condition_width,
            },
            loss: SetReconstructionLossConfig::default(),
            reconstruction_ordering: FlatVectorReconstructionOrdering::Slot,
            precursor_mz_scale: ConditioningConfig::default().precursor_mz_scale(),
            regularization: RegularizationConfig::default(),
            auxiliary: AuxiliaryLossConfig::default(),
            mz_sigma_start: 0.0,
            mz_sigma_end: 0.0,
            mz_sigma_decay_steps: 0,
        }
    }

    /// Sets L1/L2 model-parameter regularization.
    #[must_use]
    pub const fn with_regularization(mut self, regularization: RegularizationConfig) -> Self {
        self.regularization = regularization;
        self
    }

    /// Sets auxiliary objective weights.
    #[must_use]
    pub const fn with_auxiliary(mut self, auxiliary: AuxiliaryLossConfig) -> Self {
        self.auxiliary = auxiliary;
        self
    }

    /// Sets how flat-vector peaks are aligned before reconstruction loss.
    #[must_use]
    pub const fn with_reconstruction_ordering(
        mut self,
        ordering: FlatVectorReconstructionOrdering,
    ) -> Self {
        self.reconstruction_ordering = ordering;
        self
    }

    /// Sets the σ annealing schedule for the m/z Gaussian gate. `decay_steps == 0`
    /// disables annealing and σ stays at `start` for the whole run. When the
    /// schedule fields stay at zero (the default), the model falls back to a
    /// constant σ equal to `loss.normalized_mz_tolerance` so existing configs
    /// behave bit-identically to before.
    #[must_use]
    pub const fn with_mz_sigma_schedule(
        mut self,
        start: f64,
        end: f64,
        decay_steps: usize,
    ) -> Self {
        self.mz_sigma_start = start;
        self.mz_sigma_end = end;
        self.mz_sigma_decay_steps = decay_steps;
        self
    }

    /// Creates an initialized autoencoder.
    pub fn init<B: Backend>(&self, device: &B::Device) -> SpectralAutoencoder<B> {
        // Schedule fields default to the static `normalized_mz_tolerance` when
        // the user hasn't opted in (both start and end zero), so the dynamic
        // path degenerates to the previous constant-σ behaviour bit-for-bit.
        let baseline_sigma = self.loss.normalized_mz_tolerance;
        let mz_sigma_start = if self.mz_sigma_start > 0.0 {
            self.mz_sigma_start
        } else {
            baseline_sigma
        };
        let mz_sigma_end = if self.mz_sigma_end > 0.0 {
            self.mz_sigma_end
        } else {
            baseline_sigma
        };
        SpectralAutoencoder {
            encoder: self.encoder.init(device),
            decoder: self.decoder.init(device),
            auxiliary_heads: EmbeddingAuxiliaryHeadsConfig {
                latent_width: self.encoder.latent_width,
                max_peaks: self.encoder.spectrum_width / 2,
                intruder_hidden_width: self.auxiliary.intruder_hidden_width,
            }
            .init(device),
            normalized_mz_tolerance: baseline_sigma,
            loss_mz_power: self.loss.mz_power,
            loss_intensity_power: self.loss.intensity_power,
            count_weight: self.loss.count_weight,
            reconstruction_ordering: self.reconstruction_ordering.code(),
            regularization_l1: self.regularization.l1,
            regularization_l2: self.regularization.l2,
            reconstruction_weight: self.auxiliary.reconstruction_weight,
            masked_peak_weight: self.auxiliary.masked_peak_weight,
            intruder_peak_weight: self.auxiliary.intruder_peak_weight,
            precursor_reconstruction_weight: self.auxiliary.precursor_reconstruction_weight,
            masked_precursor_weight: self.auxiliary.masked_precursor_weight,
            precursor_mz_scale: self.precursor_mz_scale,
            similarity_ranking_weight: self.auxiliary.similarity_ranking_weight,
            similarity_ranking_latent_temperature: self
                .auxiliary
                .similarity_ranking_latent_temperature,
            similarity_ranking_teacher_temperature: self
                .auxiliary
                .similarity_ranking_teacher_temperature,
            similarity_ranking_min_gap: self.auxiliary.similarity_ranking_min_gap,
            latent_noise_std: self.auxiliary.latent_noise_std,
            similarity_ranking_pairs_per_batch: self.auxiliary.similarity_ranking_pairs_per_batch,
            chamfer_mz_weight: self.auxiliary.chamfer_mz_weight,
            mz_sigma_start,
            mz_sigma_end,
            mz_sigma_decay_steps: self.mz_sigma_decay_steps,
            mz_sigma_step: Arc::new(AtomicUsize::new(0)),
        }
    }
}

/// Encoder module.
#[derive(Module, Debug)]
pub struct Encoder<B: Backend> {
    layers: Vec<Linear<B>>,
    latent: Linear<B>,
    activation: Relu,
}

impl<B: Backend> Encoder<B> {
    /// Encodes a cleaned spectrum and optional metadata conditions.
    pub fn forward(&self, spectra: Tensor<B, 2>, conditions: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut features = Tensor::cat(vec![spectra, conditions], 1);
        for layer in &self.layers {
            features = self.activation.forward(layer.forward(features));
        }
        self.latent.forward(features)
    }
}

/// Decoder module.
#[derive(Module, Debug)]
pub struct Decoder<B: Backend> {
    layers: Vec<Linear<B>>,
    spectrum_output: Linear<B>,
    condition_output: Linear<B>,
    activation: Relu,
    condition_width: usize,
}

impl<B: Backend> Decoder<B> {
    /// Decodes a latent representation without direct metadata conditioning.
    pub fn forward(&self, latent: Tensor<B, 2>) -> DecoderOutput<B> {
        assert_eq!(
            self.condition_width, 0,
            "Decoder::forward requires a zero-width decoder condition configuration"
        );
        self.forward_features(latent)
    }

    /// Decodes a latent representation with explicit decoder-side conditions.
    pub fn forward_with_conditions(
        &self,
        latent: Tensor<B, 2>,
        conditions: Tensor<B, 2>,
    ) -> DecoderOutput<B> {
        let [_batch_size, condition_width] = conditions.dims();
        assert_eq!(
            condition_width, self.condition_width,
            "decoder condition tensor width does not match the decoder configuration"
        );
        self.forward_features(Tensor::cat(vec![latent, conditions], 1))
    }

    fn forward_features(&self, mut features: Tensor<B, 2>) -> DecoderOutput<B> {
        for layer in &self.layers {
            features = self.activation.forward(layer.forward(features));
        }
        DecoderOutput {
            spectrum: sigmoid(self.spectrum_output.forward(features.clone())),
            conditions: sigmoid(self.condition_output.forward(features)),
        }
    }
}

/// Output returned by the flat-vector decoder.
pub struct DecoderOutput<B: Backend> {
    /// Reconstructed cleaned spectrum vector.
    pub spectrum: Tensor<B, 2>,
    /// Reconstructed precursor condition vector.
    pub conditions: Tensor<B, 2>,
}

/// Output returned by the autoencoder.
pub struct AutoencoderOutput<B: Backend> {
    /// Latent spectral embedding.
    pub latent: Tensor<B, 2>,
    /// Reconstructed cleaned spectrum vector.
    pub reconstruction: Tensor<B, 2>,
    /// Reconstructed precursor condition vector.
    pub condition_reconstruction: Tensor<B, 2>,
}

/// Deterministic spectrum autoencoder.
#[derive(Module, Debug)]
pub struct SpectralAutoencoder<B: Backend> {
    /// Encoder deliverable.
    pub encoder: Encoder<B>,
    /// Decoder deliverable.
    pub decoder: Decoder<B>,
    auxiliary_heads: EmbeddingAuxiliaryHeads<B>,
    normalized_mz_tolerance: f64,
    loss_mz_power: f64,
    loss_intensity_power: f64,
    count_weight: f64,
    reconstruction_ordering: usize,
    regularization_l1: f64,
    regularization_l2: f64,
    reconstruction_weight: f64,
    masked_peak_weight: f64,
    intruder_peak_weight: f64,
    precursor_reconstruction_weight: f64,
    masked_precursor_weight: f64,
    precursor_mz_scale: f64,
    similarity_ranking_weight: f64,
    similarity_ranking_latent_temperature: f64,
    similarity_ranking_teacher_temperature: f64,
    similarity_ranking_min_gap: f64,
    latent_noise_std: f64,
    similarity_ranking_pairs_per_batch: usize,
    chamfer_mz_weight: f64,
    mz_sigma_start: f64,
    mz_sigma_end: f64,
    mz_sigma_decay_steps: usize,
    #[module(skip)]
    mz_sigma_step: Arc<AtomicUsize>,
}

impl<B: Backend> SpectralAutoencoder<B> {
    /// Current value of the m/z Gaussian gate bandwidth `σ`, taking the
    /// optional linear annealing schedule into account.
    pub fn current_sigma(&self) -> f64 {
        if self.mz_sigma_decay_steps == 0 {
            return self.mz_sigma_start;
        }
        let step = self.mz_sigma_step.load(Ordering::Relaxed);
        let frac = (step as f64 / self.mz_sigma_decay_steps as f64).min(1.0);
        self.mz_sigma_start * (1.0 - frac) + self.mz_sigma_end * frac
    }

    /// Bumps the σ schedule's step counter by one. Called from the training
    /// forward path only (validation forwards leave the counter alone).
    #[cfg(feature = "train")]
    fn advance_sigma_step(&self) {
        self.mz_sigma_step.fetch_add(1, Ordering::Relaxed);
    }

    /// Resets the schedule's step counter; called when resuming training from
    /// a checkpoint so the schedule lines up with the resumed epoch.
    pub fn set_mz_sigma_step(&self, step: usize) {
        self.mz_sigma_step.store(step, Ordering::Relaxed);
    }

    fn loss_config(&self) -> SetReconstructionLossConfig {
        SetReconstructionLossConfig {
            normalized_mz_tolerance: self.current_sigma(),
            mz_power: self.loss_mz_power,
            intensity_power: self.loss_intensity_power,
            count_weight: self.count_weight,
        }
    }

    fn regularization_config(&self) -> RegularizationConfig {
        RegularizationConfig {
            l1: self.regularization_l1,
            l2: self.regularization_l2,
        }
    }

    /// Runs the full autoencoder.
    pub fn forward(&self, spectra: Tensor<B, 2>, conditions: Tensor<B, 2>) -> AutoencoderOutput<B> {
        let latent = self.encoder.forward(spectra, conditions);
        let decoder_output = self.decoder.forward(latent.clone());
        AutoencoderOutput {
            latent,
            reconstruction: decoder_output.spectrum,
            condition_reconstruction: decoder_output.conditions,
        }
    }

    /// Runs the full autoencoder with explicit decoder-side conditions.
    pub fn forward_with_decoder_conditions(
        &self,
        spectra: Tensor<B, 2>,
        encoder_conditions: Tensor<B, 2>,
        decoder_conditions: Tensor<B, 2>,
    ) -> AutoencoderOutput<B> {
        let latent = self.encoder.forward(spectra, encoder_conditions);
        let decoder_output = self
            .decoder
            .forward_with_conditions(latent.clone(), decoder_conditions);
        AutoencoderOutput {
            latent,
            reconstruction: decoder_output.spectrum,
            condition_reconstruction: decoder_output.conditions,
        }
    }

    /// Computes ordered slot-wise spectral reconstruction loss on cleaned spectrum vectors.
    pub fn reconstruction_loss(
        &self,
        reconstruction: Tensor<B, 2>,
        target: Tensor<B, 2>,
    ) -> Tensor<B, 1> {
        let device = reconstruction.device();
        self.reconstruction_loss_raw(reconstruction, target)
            + self.regularization_config().penalty(self, &device)
    }

    fn reconstruction_loss_raw(
        &self,
        reconstruction: Tensor<B, 2>,
        target: Tensor<B, 2>,
    ) -> Tensor<B, 1> {
        flat_vector_reconstruction_loss_from_vectors(
            reconstruction,
            target,
            self.loss_config(),
            FlatVectorReconstructionOrdering::from_code(self.reconstruction_ordering),
        )
    }

    #[cfg(feature = "train")]
    fn forward_reconstruction(
        &self,
        batch: crate::batch::AutoencoderBatch<B>,
        use_latent_noise: bool,
    ) -> (
        Tensor<B, 2>,
        Tensor<B, 2>,
        AutoencoderLossBreakdown<B>,
        AutoencoderDiagnostics<B>,
    ) {
        if use_latent_noise {
            self.advance_sigma_step();
        }
        let target = batch.target_spectra.clone();
        let input_peak_mask = vector_target_mask(batch.spectra.clone());
        let latent = self.encoder.forward(batch.spectra, batch.conditions);
        let decoder_latent = if use_latent_noise {
            apply_latent_noise(latent.clone(), self.latent_noise_std)
        } else {
            latent.clone()
        };
        let decoder_output = self.decoder.forward(decoder_latent);
        let output = AutoencoderOutput {
            latent,
            reconstruction: decoder_output.spectrum,
            condition_reconstruction: decoder_output.conditions,
        };
        let device = output.reconstruction.device();

        let target_peak_mask = vector_target_mask(target.clone());
        let reconstruction_ordering =
            FlatVectorReconstructionOrdering::from_code(self.reconstruction_ordering);
        let reconstruction_config = self.loss_config();
        let masked = if self.masked_peak_weight > 0.0 {
            let masked_target_mask = target_peak_mask.clone()
                * vector_element_mask_to_peak_mask(batch.masked_spectra_mask);
            let has_masked_targets = masked_target_mask.clone().sum().greater_elem(0.0).float();
            let (reconstruction_loss, masked_loss) =
                flat_vector_reconstruction_losses_from_vectors_with_masks(
                    output.reconstruction.clone(),
                    target.clone(),
                    target_peak_mask.clone(),
                    masked_target_mask,
                    reconstruction_config,
                    reconstruction_ordering,
                );
            let reconstruction = reconstruction_loss * self.reconstruction_weight;
            let masked = masked_loss * has_masked_targets * self.masked_peak_weight;
            (reconstruction, masked)
        } else {
            let reconstruction = flat_vector_reconstruction_loss_from_vectors(
                output.reconstruction.clone(),
                target.clone(),
                reconstruction_config,
                reconstruction_ordering,
            ) * self.reconstruction_weight;
            (reconstruction, Tensor::zeros([1], &device))
        };
        let (reconstruction, masked) = masked;
        let chamfer_mz = if self.chamfer_mz_weight > 0.0 {
            slot_chamfer_magnet_mz(
                output.reconstruction.clone(),
                target.clone(),
                target_peak_mask,
            ) * self.chamfer_mz_weight
        } else {
            Tensor::zeros([1], &device)
        };
        let intruder = weighted_intruder_detection_loss(
            &self.auxiliary_heads,
            output.latent.clone(),
            batch.intruder_peak_mask,
            input_peak_mask.clone(),
            self.intruder_peak_weight,
        );
        let precursor_target = batch.target_conditions.clone();
        let precursor = weighted_precursor_reconstruction_output(
            output.condition_reconstruction.clone(),
            precursor_target.clone(),
            self.precursor_mz_scale,
            self.precursor_reconstruction_weight,
        );
        let masked_precursor = weighted_masked_precursor_reconstruction_output(
            output.condition_reconstruction.clone(),
            precursor_target.clone(),
            batch.masked_precursor_mask,
            self.precursor_mz_scale,
            self.masked_precursor_weight,
        );
        let similarity_ranking = weighted_similarity_ranking_output(
            output.latent.clone(),
            batch.similarity_ranking,
            self.similarity_ranking_pairs_per_batch,
            self.similarity_ranking_latent_temperature,
            self.similarity_ranking_teacher_temperature,
            self.similarity_ranking_min_gap,
            self.similarity_ranking_weight,
        );
        let regularization = self.regularization_config().penalty(self, &device);
        let diagnostics = AutoencoderDiagnostics {
            similarity_ranking_pairs: similarity_ranking.valid_pairs,
            similarity_ranking_accuracy: similarity_ranking.accuracy,
            similarity_ranking_mrr: similarity_ranking.mrr,
            precursor_mae_da: precursor.mae_da,
        };
        let losses = AutoencoderLossBreakdown {
            reconstruction,
            masked,
            intruder,
            precursor: precursor.loss,
            masked_precursor: masked_precursor.loss,
            similarity_ranking: similarity_ranking.loss,
            regularization,
            chamfer_mz,
        };

        (output.reconstruction, target, losses, diagnostics)
    }
}

#[cfg(feature = "train")]
mod train_impl {
    use burn::{
        tensor::backend::AutodiffBackend,
        train::{InferenceStep, TrainOutput, TrainStep},
    };

    use crate::{batch::AutoencoderBatch, training::AutoencoderTrainingOutput};

    use super::*;

    impl<B: AutodiffBackend> TrainStep for SpectralAutoencoder<B> {
        type Input = AutoencoderBatch<B>;
        type Output = AutoencoderTrainingOutput<B>;

        fn step(&self, batch: AutoencoderBatch<B>) -> TrainOutput<AutoencoderTrainingOutput<B>> {
            let (reconstruction, target, losses, diagnostics) =
                self.forward_reconstruction(batch, true);
            let item = AutoencoderTrainingOutput::new_with_diagnostics(
                reconstruction,
                target,
                losses,
                diagnostics,
            );

            TrainOutput::new(self, item.loss.clone().backward(), item)
        }
    }

    impl<B: Backend> InferenceStep for SpectralAutoencoder<B> {
        type Input = AutoencoderBatch<B>;
        type Output = AutoencoderTrainingOutput<B>;

        fn step(&self, batch: AutoencoderBatch<B>) -> AutoencoderTrainingOutput<B> {
            let (reconstruction, target, losses, diagnostics) =
                self.forward_reconstruction(batch, false);

            AutoencoderTrainingOutput::new_with_diagnostics(
                reconstruction,
                target,
                losses,
                diagnostics,
            )
        }
    }
}

/// Fluent builder for [`SpectralAutoencoderConfig`].
#[derive(Debug, Clone, Default)]
pub struct SpectralAutoencoderConfigBuilder {
    encoder: Option<EncoderConfig>,
    decoder: Option<DecoderConfig>,
    loss: Option<SetReconstructionLossConfig>,
    reconstruction_ordering: Option<FlatVectorReconstructionOrdering>,
    precursor_mz_scale: Option<f64>,
    regularization: Option<RegularizationConfig>,
    auxiliary: Option<AuxiliaryLossConfig>,
}

impl SpectralAutoencoderConfigBuilder {
    /// Creates an empty builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the encoder configuration (required).
    #[inline]
    #[must_use]
    pub fn with_encoder(mut self, value: EncoderConfig) -> Self {
        self.encoder = Some(value);
        self
    }

    /// Sets the decoder configuration (required).
    #[inline]
    #[must_use]
    pub fn with_decoder(mut self, value: DecoderConfig) -> Self {
        self.decoder = Some(value);
        self
    }

    /// Overrides the set-reconstruction loss config.
    #[inline]
    #[must_use]
    pub fn with_loss(mut self, value: SetReconstructionLossConfig) -> Self {
        self.loss = Some(value);
        self
    }

    /// Overrides the reconstruction-peak ordering policy.
    #[inline]
    #[must_use]
    pub fn with_reconstruction_ordering(mut self, value: FlatVectorReconstructionOrdering) -> Self {
        self.reconstruction_ordering = Some(value);
        self
    }

    /// Overrides the diagnostic precursor m/z scale.
    #[inline]
    #[must_use]
    pub fn with_precursor_mz_scale(mut self, value: f64) -> Self {
        self.precursor_mz_scale = Some(value);
        self
    }

    /// Overrides the regularization config.
    #[inline]
    #[must_use]
    pub fn with_regularization(mut self, value: RegularizationConfig) -> Self {
        self.regularization = Some(value);
        self
    }

    /// Overrides the auxiliary-loss config.
    #[inline]
    #[must_use]
    pub fn with_auxiliary(mut self, value: AuxiliaryLossConfig) -> Self {
        self.auxiliary = Some(value);
        self
    }

    /// Builds the config.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::IncompleteBuilder`] when `encoder` or `decoder`
    /// is unset.
    pub fn build(self) -> crate::Result<SpectralAutoencoderConfig> {
        let encoder = self
            .encoder
            .ok_or_else(|| crate::Error::IncompleteBuilder {
                config: "SpectralAutoencoderConfig",
                field: "encoder",
            })?;
        let decoder = self
            .decoder
            .ok_or_else(|| crate::Error::IncompleteBuilder {
                config: "SpectralAutoencoderConfig",
                field: "decoder",
            })?;
        Ok(SpectralAutoencoderConfig {
            encoder,
            decoder,
            loss: self.loss.unwrap_or_default(),
            reconstruction_ordering: self.reconstruction_ordering.unwrap_or_default(),
            precursor_mz_scale: self
                .precursor_mz_scale
                .unwrap_or_else(default_precursor_mz_scale),
            regularization: self.regularization.unwrap_or_default(),
            auxiliary: self.auxiliary.unwrap_or_default(),
            mz_sigma_start: 0.0,
            mz_sigma_end: 0.0,
            mz_sigma_decay_steps: 0,
        })
    }
}

#[cfg(all(test, feature = "ndarray"))]
mod tests {
    use burn::data::dataloader::batcher::Batcher;
    use burn::module::Module;

    use crate::batch::{AutoencoderBatch, AutoencoderBatcher, AutoencoderSample};

    use super::*;

    #[test]
    fn model_forward_shapes_are_stable() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let config = SpectralAutoencoderConfig::symmetric(120, 16, 32, vec![64]);
        let model = config.init::<B>(&device);
        let batch: AutoencoderBatch<B> = AutoencoderBatcher.batch(
            vec![AutoencoderSample {
                spectrum: vec![0.0; 120],
                conditions: vec![0.0; 16],
            }],
            &device,
        );

        let output = model.forward(batch.spectra, batch.conditions);
        assert_eq!(
            config.reconstruction_ordering,
            FlatVectorReconstructionOrdering::Slot
        );
        assert_eq!(output.latent.dims(), [1, 32]);
        assert_eq!(output.reconstruction.dims(), [1, 120]);
        assert_eq!(output.condition_reconstruction.dims(), [1, 16]);
    }

    #[test]
    fn twenty_million_run_config_is_stable() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let config = SpectralAutoencoderConfig::twenty_million_run();
        let model = config.init::<B>(&device);

        assert_eq!(config.encoder.spectrum_width, 120);
        assert_eq!(config.encoder.condition_width, 2);
        assert_eq!(config.decoder.condition_width, 0);
        assert_eq!(config.decoder.condition_output_width, 2);
        assert_eq!(config.encoder.latent_width, 256);
        assert_eq!(config.encoder.hidden_widths, vec![4096, 2048, 1024, 512]);
        assert_eq!(config.decoder.hidden_widths, vec![512, 1024, 2048, 4096]);
        assert_eq!(
            config.reconstruction_ordering,
            FlatVectorReconstructionOrdering::IntensityDescending
        );
        assert_eq!(config.regularization.l1, 0.0);
        assert_eq!(config.regularization.l2, 0.0);
        assert_eq!(config.auxiliary.masked_peak_weight, 0.1);
        assert_eq!(config.auxiliary.intruder_peak_weight, 0.05);
        assert_eq!(config.auxiliary.masked_precursor_weight, 0.05);
        assert_eq!(config.auxiliary.similarity_ranking_weight, 0.20);
        assert_eq!(config.auxiliary.similarity_ranking_latent_temperature, 0.10);
        assert_eq!(
            config.auxiliary.similarity_ranking_teacher_temperature,
            0.10
        );
        assert_eq!(config.auxiliary.latent_noise_std, 0.02);
        assert_eq!(model.num_params(), 23_338_107);
    }

    #[test]
    fn twenty_million_run_config_accepts_larger_peak_counts() {
        let config = SpectralAutoencoderConfig::twenty_million_run_with_peaks(128);

        assert_eq!(config.encoder.spectrum_width, 256);
        assert_eq!(config.decoder.spectrum_width, 256);
        assert_eq!(config.decoder.condition_width, 0);
        assert_eq!(config.encoder.latent_width, 256);
        assert_eq!(
            config.reconstruction_ordering,
            FlatVectorReconstructionOrdering::IntensityDescending
        );
    }

    #[test]
    fn current_sigma_returns_start_when_decay_disabled() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let config = SpectralAutoencoderConfig::symmetric(4, 2, 2, vec![3])
            .with_mz_sigma_schedule(0.05, 0.01, 0);
        let model = config.init::<B>(&device);
        assert!((model.current_sigma() - 0.05).abs() < 1.0e-12);
        // Advancing the counter is a no-op when decay_steps == 0.
        for _ in 0..1000 {
            model.set_mz_sigma_step(1_000_000);
        }
        assert!((model.current_sigma() - 0.05).abs() < 1.0e-12);
    }

    #[test]
    fn current_sigma_linearly_interpolates_between_start_and_end() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let config = SpectralAutoencoderConfig::symmetric(4, 2, 2, vec![3])
            .with_mz_sigma_schedule(0.05, 0.01, 100);
        let model = config.init::<B>(&device);

        model.set_mz_sigma_step(0);
        assert!((model.current_sigma() - 0.05).abs() < 1.0e-9);

        model.set_mz_sigma_step(50);
        assert!((model.current_sigma() - 0.03).abs() < 1.0e-9);

        model.set_mz_sigma_step(100);
        assert!((model.current_sigma() - 0.01).abs() < 1.0e-9);

        // Saturates at σ_end past decay_steps.
        model.set_mz_sigma_step(10_000);
        assert!((model.current_sigma() - 0.01).abs() < 1.0e-9);
    }

    #[test]
    fn config_with_zero_schedule_inherits_loss_normalized_mz_tolerance() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        // Default schedule fields are all zero; init should fall back to the
        // loss config's `normalized_mz_tolerance` (default 0.01).
        let config = SpectralAutoencoderConfig::symmetric(4, 2, 2, vec![3]);
        let model = config.init::<B>(&device);
        let expected = SetReconstructionLossConfig::default().normalized_mz_tolerance;
        assert!((model.current_sigma() - expected).abs() < 1.0e-12);
    }

    #[test]
    fn regularization_contributes_to_reconstruction_loss() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let config = SpectralAutoencoderConfig::symmetric(4, 2, 2, vec![3]).with_regularization(
            RegularizationConfig {
                l1: 1.0e-3,
                l2: 1.0e-3,
            },
        );
        let model = config.init::<B>(&device);
        let reconstruction = Tensor::<B, 2>::zeros([1, 4], &device);
        let target = Tensor::<B, 2>::zeros([1, 4], &device);

        let loss = model
            .reconstruction_loss(reconstruction, target)
            .into_scalar();

        assert!(loss > 0.0);
    }
}
