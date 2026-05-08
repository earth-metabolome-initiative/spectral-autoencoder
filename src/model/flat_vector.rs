//! Flat vector autoencoder baseline.

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

#[cfg(feature = "train")]
use crate::{
    model::auxiliary::{
        apply_latent_noise, weighted_cosine_distance_loss, weighted_intruder_detection_loss,
        weighted_masked_precursor_reconstruction_output, weighted_precursor_reconstruction_output,
        weighted_similarity_ranking_output,
    },
    model::reconstruction::{
        flat_vector_reconstruction_losses_from_vectors_with_masks,
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
    /// Auxiliary denoising, consistency, intruder, and similarity-ranking objectives.
    #[serde(default)]
    pub auxiliary: AuxiliaryLossConfig,
}

impl SpectralAutoencoderConfig {
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

    /// Creates an initialized autoencoder.
    pub fn init<B: Backend>(&self, device: &B::Device) -> SpectralAutoencoder<B> {
        SpectralAutoencoder {
            encoder: self.encoder.init(device),
            decoder: self.decoder.init(device),
            auxiliary_heads: EmbeddingAuxiliaryHeadsConfig {
                latent_width: self.encoder.latent_width,
                max_peaks: self.encoder.spectrum_width / 2,
                intruder_hidden_width: self.auxiliary.intruder_hidden_width,
            }
            .init(device),
            normalized_mz_tolerance: self.loss.normalized_mz_tolerance,
            loss_mz_power: self.loss.mz_power,
            loss_intensity_power: self.loss.intensity_power,
            count_weight: self.loss.count_weight,
            reconstruction_ordering: self.reconstruction_ordering.code(),
            regularization_l1: self.regularization.l1,
            regularization_l2: self.regularization.l2,
            reconstruction_weight: self.auxiliary.reconstruction_weight,
            masked_peak_weight: self.auxiliary.masked_peak_weight,
            consistency_weight: self.auxiliary.consistency_weight,
            intruder_peak_weight: self.auxiliary.intruder_peak_weight,
            precursor_reconstruction_weight: self.auxiliary.precursor_reconstruction_weight,
            masked_precursor_weight: self.auxiliary.masked_precursor_weight,
            precursor_mz_scale: self.precursor_mz_scale,
            similarity_ranking_weight: self.auxiliary.similarity_ranking_weight,
            similarity_ranking_margin: self.auxiliary.similarity_ranking_margin,
            similarity_ranking_min_gap: self.auxiliary.similarity_ranking_min_gap,
            latent_noise_std: self.auxiliary.latent_noise_std,
            similarity_ranking_pairs_per_batch: self.auxiliary.similarity_ranking_pairs_per_batch,
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
    consistency_weight: f64,
    intruder_peak_weight: f64,
    precursor_reconstruction_weight: f64,
    masked_precursor_weight: f64,
    precursor_mz_scale: f64,
    similarity_ranking_weight: f64,
    similarity_ranking_margin: f64,
    similarity_ranking_min_gap: f64,
    latent_noise_std: f64,
    similarity_ranking_pairs_per_batch: usize,
}

impl<B: Backend> SpectralAutoencoder<B> {
    fn loss_config(&self) -> SetReconstructionLossConfig {
        SetReconstructionLossConfig {
            normalized_mz_tolerance: self.normalized_mz_tolerance,
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
                    target_peak_mask,
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
        let consistency = weighted_cosine_distance_loss(
            output.latent.clone(),
            || {
                self.encoder
                    .forward(batch.consistency_spectra, batch.consistency_conditions)
            },
            self.consistency_weight,
            &device,
        );
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
            precursor_target,
            batch.masked_precursor_mask,
            self.precursor_mz_scale,
            self.masked_precursor_weight,
        );
        let similarity_ranking = weighted_similarity_ranking_output(
            output.latent.clone(),
            batch.similarity_ranking,
            self.similarity_ranking_pairs_per_batch,
            self.similarity_ranking_margin,
            self.similarity_ranking_min_gap,
            self.similarity_ranking_weight,
        );
        let regularization = self.regularization_config().penalty(self, &device);
        let diagnostics = AutoencoderDiagnostics {
            similarity_ranking_pairs: similarity_ranking.valid_pairs,
            similarity_ranking_accuracy: similarity_ranking.accuracy,
            precursor_mae_da: precursor.mae_da,
        };
        let losses = AutoencoderLossBreakdown {
            reconstruction,
            masked,
            consistency,
            intruder,
            precursor: precursor.loss,
            masked_precursor: masked_precursor.loss,
            similarity_ranking: similarity_ranking.loss,
            regularization,
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
        assert_eq!(config.auxiliary.masked_peak_weight, 0.25);
        assert_eq!(config.auxiliary.consistency_weight, 0.05);
        assert_eq!(config.auxiliary.intruder_peak_weight, 0.05);
        assert_eq!(config.auxiliary.masked_precursor_weight, 0.05);
        assert_eq!(config.auxiliary.similarity_ranking_weight, 0.05);
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
