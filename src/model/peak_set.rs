//! Peak-token transformer encoder and set decoder.

use burn::{
    module::{Initializer, Module, Param},
    nn::{
        Linear, LinearConfig, Relu,
        transformer::{
            TransformerDecoder, TransformerDecoderConfig, TransformerDecoderInput,
            TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput,
        },
    },
    prelude::*,
    tensor::{Tensor, activation::sigmoid},
};
use serde::{Deserialize, Serialize};

use crate::{
    conditioning::ConditioningConfig,
    model::RegularizationConfig,
    model::auxiliary::{
        AuxiliaryLossConfig, EmbeddingAuxiliaryHeads, EmbeddingAuxiliaryHeadsConfig,
        RetentionOrderBatch, apply_latent_noise, weighted_cosine_distance_loss,
        weighted_intruder_detection_loss, weighted_retention_order_output,
    },
    model::reconstruction::{SetReconstructionLossConfig, set_reconstruction_loss_from_triples},
    tokenize::SpectrumTokenizerConfig,
};

#[cfg(feature = "train")]
use crate::training::{AutoencoderDiagnostics, AutoencoderLossBreakdown};

/// Peak-token encoder configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeakSetEncoderConfig {
    /// Maximum number of peak tokens per spectrum.
    pub max_peaks: usize,
    /// Width of each peak token feature vector.
    pub token_feature_width: usize,
    /// Width of the metadata conditioning vector.
    pub condition_width: usize,
    /// Internal peak-token embedding width.
    pub token_embedding_width: usize,
    /// Number of self-attention heads.
    pub attention_heads: usize,
    /// Number of transformer encoder layers.
    pub transformer_layers: usize,
    /// Width of the transformer feed-forward block.
    pub transformer_feed_forward_width: usize,
    /// Transformer dropout probability.
    pub dropout: f64,
    /// Hidden layer widths after masked peak pooling.
    pub hidden_widths: Vec<usize>,
    /// Latent embedding width.
    pub latent_width: usize,
}

impl PeakSetEncoderConfig {
    /// Creates an initialized peak-token encoder.
    pub fn init<B: Backend>(&self, device: &B::Device) -> PeakSetEncoder<B> {
        assert!(self.max_peaks > 0, "max_peaks must be positive");
        assert!(self.attention_heads > 0, "attention_heads must be positive");
        assert_eq!(
            self.token_embedding_width % self.attention_heads,
            0,
            "token_embedding_width must be divisible by attention_heads"
        );

        let mut pooled_width = self.token_embedding_width + self.condition_width;
        let mut pooled_layers = Vec::with_capacity(self.hidden_widths.len());
        for &hidden_width in &self.hidden_widths {
            pooled_layers.push(LinearConfig::new(pooled_width, hidden_width).init(device));
            pooled_width = hidden_width;
        }

        PeakSetEncoder {
            token_projection: LinearConfig::new(
                self.token_feature_width,
                self.token_embedding_width,
            )
            .init(device),
            transformer: TransformerEncoderConfig::new(
                self.token_embedding_width,
                self.transformer_feed_forward_width,
                self.attention_heads,
                self.transformer_layers,
            )
            .with_dropout(self.dropout)
            .with_norm_first(true)
            .init(device),
            pooled_layers,
            latent: LinearConfig::new(pooled_width, self.latent_width).init(device),
            activation: Relu::new(),
            token_embedding_width: self.token_embedding_width,
        }
    }
}

/// Query-set decoder configuration for reconstructed peak candidates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeakSetDecoderConfig {
    /// Maximum number of reconstructed peak candidates.
    pub max_peaks: usize,
    /// Width of the latent embedding.
    pub latent_width: usize,
    /// Width of the metadata conditioning vector.
    pub condition_width: usize,
    /// Width of each learned output query.
    pub query_width: usize,
    /// Number of self/cross-attention heads in the decoder.
    pub attention_heads: usize,
    /// Number of transformer decoder layers.
    pub decoder_layers: usize,
    /// Width of the decoder feed-forward block.
    pub decoder_feed_forward_width: usize,
    /// Decoder dropout probability.
    pub dropout: f64,
}

impl PeakSetDecoderConfig {
    /// Returns the number of values predicted per peak candidate.
    #[must_use]
    pub const fn output_feature_width(&self) -> usize {
        3
    }

    /// Returns the flattened `(m/z, intensity, presence)` output width.
    #[must_use]
    pub const fn flattened_output_width(&self) -> usize {
        self.max_peaks * self.output_feature_width()
    }

    /// Creates an initialized peak-set decoder.
    pub fn init<B: Backend>(&self, device: &B::Device) -> PeakSetDecoder<B> {
        assert!(self.max_peaks > 0, "max_peaks must be positive");
        assert!(self.attention_heads > 0, "attention_heads must be positive");
        assert_eq!(
            self.query_width % self.attention_heads,
            0,
            "query_width must be divisible by attention_heads"
        );

        PeakSetDecoder {
            queries: Initializer::Normal {
                mean: 0.0,
                std: 0.02,
            }
            .init([self.max_peaks, self.query_width], device),
            memory_projection: LinearConfig::new(
                self.latent_width + self.condition_width,
                self.query_width,
            )
            .init(device),
            decoder: TransformerDecoderConfig::new(
                self.query_width,
                self.decoder_feed_forward_width,
                self.attention_heads,
                self.decoder_layers,
            )
            .with_dropout(self.dropout)
            .with_norm_first(true)
            .init(device),
            output: LinearConfig::new(self.query_width, self.output_feature_width()).init(device),
            max_peaks: self.max_peaks,
            query_width: self.query_width,
        }
    }
}

/// Peak-token autoencoder configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeakSetAutoencoderConfig {
    /// Encoder configuration.
    pub encoder: PeakSetEncoderConfig,
    /// Decoder configuration.
    pub decoder: PeakSetDecoderConfig,
    /// Set reconstruction loss configuration.
    pub loss: SetReconstructionLossConfig,
    /// L1/L2 model-parameter regularization.
    #[serde(default)]
    pub regularization: RegularizationConfig,
    /// Auxiliary denoising, consistency, and retention-order objectives.
    #[serde(default)]
    pub auxiliary: AuxiliaryLossConfig,
}

impl PeakSetAutoencoderConfig {
    /// Starting peak-set transformer configuration for the 20M-spectrum run.
    ///
    /// This is the recommended first large-run configuration: 60 peak tokens,
    /// a 512-wide encoder/decoder, six transformer encoder layers, four set
    /// decoder layers, and an 8-head attention layout. It is substantially
    /// larger than the smoke-test-sized defaults while staying below
    /// DreaMS-scale memory pressure on a single RTX 4090/5090-class GPU. The
    /// transformer internals stay 512-wide, while the deliverable embedding is
    /// a compact 64-dimensional latent.
    #[must_use]
    pub fn twenty_million_run() -> Self {
        let tokenizer = SpectrumTokenizerConfig::default();
        let condition_width = ConditioningConfig::default().vector_width();
        let token_embedding_width = 512;

        Self {
            encoder: PeakSetEncoderConfig {
                max_peaks: tokenizer.max_peaks,
                token_feature_width: tokenizer.feature_width(),
                condition_width,
                token_embedding_width,
                attention_heads: 8,
                transformer_layers: 6,
                transformer_feed_forward_width: token_embedding_width * 4,
                dropout: 0.1,
                hidden_widths: vec![2048, 1024],
                latent_width: 64,
            },
            decoder: PeakSetDecoderConfig {
                max_peaks: tokenizer.max_peaks,
                latent_width: 64,
                condition_width,
                query_width: token_embedding_width,
                attention_heads: 8,
                decoder_layers: 4,
                decoder_feed_forward_width: token_embedding_width * 4,
                dropout: 0.1,
            },
            loss: SetReconstructionLossConfig::default(),
            regularization: RegularizationConfig::default(),
            auxiliary: AuxiliaryLossConfig::default(),
        }
    }

    /// Creates a DreaMS-style top-N peak-token autoencoder configuration.
    #[must_use]
    pub fn symmetric(
        max_peaks: usize,
        token_feature_width: usize,
        condition_width: usize,
        latent_width: usize,
        token_embedding_width: usize,
        attention_heads: usize,
        hidden_widths: Vec<usize>,
    ) -> Self {
        Self {
            encoder: PeakSetEncoderConfig {
                max_peaks,
                token_feature_width,
                condition_width,
                token_embedding_width,
                attention_heads,
                transformer_layers: 2,
                transformer_feed_forward_width: token_embedding_width * 4,
                dropout: 0.1,
                hidden_widths,
                latent_width,
            },
            decoder: PeakSetDecoderConfig {
                max_peaks,
                latent_width,
                condition_width,
                query_width: token_embedding_width,
                attention_heads,
                decoder_layers: 2,
                decoder_feed_forward_width: token_embedding_width * 4,
                dropout: 0.1,
            },
            loss: SetReconstructionLossConfig::default(),
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

    /// Creates an initialized peak-token autoencoder.
    pub fn init<B: Backend>(&self, device: &B::Device) -> PeakSetAutoencoder<B> {
        PeakSetAutoencoder {
            encoder: self.encoder.init(device),
            decoder: self.decoder.init(device),
            auxiliary_heads: EmbeddingAuxiliaryHeadsConfig {
                latent_width: self.encoder.latent_width,
                max_peaks: self.encoder.max_peaks,
                retention_hidden_width: self.auxiliary.retention_hidden_width,
                intruder_hidden_width: self.auxiliary.intruder_hidden_width,
            }
            .init(device),
            normalized_mz_tolerance: self.loss.normalized_mz_tolerance,
            loss_mz_power: self.loss.mz_power,
            loss_intensity_power: self.loss.intensity_power,
            count_weight: self.loss.count_weight,
            regularization_l1: self.regularization.l1,
            regularization_l2: self.regularization.l2,
            reconstruction_weight: self.auxiliary.reconstruction_weight,
            masked_peak_weight: self.auxiliary.masked_peak_weight,
            consistency_weight: self.auxiliary.consistency_weight,
            retention_order_weight: self.auxiliary.retention_order_weight,
            intruder_peak_weight: self.auxiliary.intruder_peak_weight,
            latent_noise_std: self.auxiliary.latent_noise_std,
            retention_pairs_per_batch: self.auxiliary.retention_pairs_per_batch,
        }
    }
}

/// Encoder module for top-N peak-token spectra.
#[derive(Module, Debug)]
pub struct PeakSetEncoder<B: Backend> {
    token_projection: Linear<B>,
    transformer: TransformerEncoder<B>,
    pooled_layers: Vec<Linear<B>>,
    latent: Linear<B>,
    activation: Relu,
    token_embedding_width: usize,
}

impl<B: Backend> PeakSetEncoder<B> {
    /// Encodes peak tokens and optional metadata conditions.
    pub fn forward(
        &self,
        token_features: Tensor<B, 3>,
        peak_mask: Tensor<B, 2>,
        padding_mask: Tensor<B, 2, burn::tensor::Bool>,
        conditions: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let [batch_size, _max_peaks, _token_feature_width] = token_features.dims();

        let token_embeddings = self
            .activation
            .forward(self.token_projection.forward(token_features));
        let encoded = self
            .transformer
            .forward(TransformerEncoderInput::new(token_embeddings).mask_pad(padding_mask));

        let encoded = encoded * peak_mask.clone().unsqueeze_dim::<3>(2);
        let pooled = encoded.sum_dim(1);
        let counts = peak_mask
            .sum_dim(1)
            .clamp_min(1.0)
            .reshape([batch_size, 1, 1])
            .expand([batch_size, 1, self.token_embedding_width]);
        let pooled = (pooled / counts).reshape([batch_size, self.token_embedding_width]);

        let mut features = Tensor::cat(vec![pooled, conditions], 1);
        for layer in &self.pooled_layers {
            features = self.activation.forward(layer.forward(features));
        }
        self.latent.forward(features)
    }
}

/// Query-based set decoder for top-N peak-token spectra.
#[derive(Module, Debug)]
pub struct PeakSetDecoder<B: Backend> {
    queries: Param<Tensor<B, 2>>,
    memory_projection: Linear<B>,
    decoder: TransformerDecoder<B>,
    output: Linear<B>,
    max_peaks: usize,
    query_width: usize,
}

impl<B: Backend> PeakSetDecoder<B> {
    /// Decodes a latent representation into unordered normalized peak candidates.
    pub fn forward(&self, latent: Tensor<B, 2>, conditions: Tensor<B, 2>) -> Tensor<B, 3> {
        let [batch_size, _latent_width] = latent.dims();
        let memory = self
            .memory_projection
            .forward(Tensor::cat(vec![latent, conditions], 1))
            .unsqueeze_dim::<3>(1);
        let queries = self.queries.val().unsqueeze_dim::<3>(0).expand([
            batch_size,
            self.max_peaks,
            self.query_width,
        ]);
        let decoded = self
            .decoder
            .forward(TransformerDecoderInput::new(queries, memory));
        sigmoid(self.output.forward(decoded))
    }
}

/// Output returned by the peak-token autoencoder.
pub struct PeakSetAutoencoderOutput<B: Backend> {
    /// Latent spectral embedding.
    pub latent: Tensor<B, 2>,
    /// Reconstructed normalized `(m/z, intensity, presence)` peak candidates.
    pub reconstruction: Tensor<B, 3>,
}

/// Deterministic top-N peak-token spectrum autoencoder.
#[derive(Module, Debug)]
pub struct PeakSetAutoencoder<B: Backend> {
    /// Encoder deliverable.
    pub encoder: PeakSetEncoder<B>,
    /// Decoder deliverable.
    pub decoder: PeakSetDecoder<B>,
    auxiliary_heads: EmbeddingAuxiliaryHeads<B>,
    normalized_mz_tolerance: f64,
    loss_mz_power: f64,
    loss_intensity_power: f64,
    count_weight: f64,
    regularization_l1: f64,
    regularization_l2: f64,
    reconstruction_weight: f64,
    masked_peak_weight: f64,
    consistency_weight: f64,
    retention_order_weight: f64,
    intruder_peak_weight: f64,
    latent_noise_std: f64,
    retention_pairs_per_batch: usize,
}

#[cfg(feature = "train")]
struct PeakSetReconstructionOutput<B: Backend> {
    reconstruction: Tensor<B, 3>,
    target: Tensor<B, 2>,
    target_mask: Tensor<B, 2>,
    losses: AutoencoderLossBreakdown<B>,
    diagnostics: AutoencoderDiagnostics<B>,
}

impl<B: Backend> PeakSetAutoencoder<B> {
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

    /// Runs the full peak-token autoencoder.
    pub fn forward(
        &self,
        token_features: Tensor<B, 3>,
        peak_mask: Tensor<B, 2>,
        padding_mask: Tensor<B, 2, burn::tensor::Bool>,
        conditions: Tensor<B, 2>,
    ) -> PeakSetAutoencoderOutput<B> {
        let latent =
            self.encoder
                .forward(token_features, peak_mask, padding_mask, conditions.clone());
        let reconstruction = self.decoder.forward(latent.clone(), conditions);
        PeakSetAutoencoderOutput {
            latent,
            reconstruction,
        }
    }

    /// Computes a soft set-wise cosine reconstruction loss on normalized peak targets.
    pub fn reconstruction_loss(
        &self,
        reconstruction: Tensor<B, 3>,
        target: Tensor<B, 2>,
        target_mask: Tensor<B, 2>,
    ) -> Tensor<B, 1> {
        let device = reconstruction.device();
        self.set_reconstruction_loss_raw(reconstruction, target, target_mask)
            + self.regularization_config().penalty(self, &device)
    }

    fn set_reconstruction_loss_raw(
        &self,
        reconstruction: Tensor<B, 3>,
        target: Tensor<B, 2>,
        target_mask: Tensor<B, 2>,
    ) -> Tensor<B, 1> {
        set_reconstruction_loss_from_triples(
            reconstruction,
            target,
            target_mask,
            self.loss_config(),
        )
    }

    #[cfg(feature = "train")]
    fn forward_reconstruction(
        &self,
        batch: crate::batch::TokenizedAutoencoderBatch<B>,
        use_latent_noise: bool,
    ) -> PeakSetReconstructionOutput<B> {
        let target = batch.target_pairs.clone();
        let target_mask = batch.target_peak_mask.clone();
        let input_peak_mask = batch.peak_mask.clone();
        let latent = self.encoder.forward(
            batch.token_features,
            batch.peak_mask,
            batch.padding_mask,
            batch.conditions.clone(),
        );
        let decoder_latent = if use_latent_noise {
            apply_latent_noise(latent.clone(), self.latent_noise_std)
        } else {
            latent.clone()
        };
        let output = PeakSetAutoencoderOutput {
            latent,
            reconstruction: self.decoder.forward(decoder_latent, batch.conditions),
        };
        let device = output.reconstruction.device();
        let reconstruction = self.set_reconstruction_loss_raw(
            output.reconstruction.clone(),
            target.clone(),
            target_mask.clone(),
        ) * self.reconstruction_weight;

        let masked = if self.masked_peak_weight > 0.0 {
            let masked_target_mask = target_mask.clone() * batch.masked_peak_mask;
            let has_masked_targets = masked_target_mask.clone().sum().greater_elem(0.0).float();
            self.set_reconstruction_loss_raw(
                output.reconstruction.clone(),
                target.clone(),
                masked_target_mask,
            ) * has_masked_targets
                * self.masked_peak_weight
        } else {
            Tensor::zeros([1], &device)
        };
        let consistency = weighted_cosine_distance_loss(
            output.latent.clone(),
            || {
                self.encoder.forward(
                    batch.consistency_token_features,
                    batch.consistency_peak_mask,
                    batch.consistency_padding_mask,
                    batch.consistency_conditions,
                )
            },
            self.consistency_weight,
            &device,
        );
        let retention = weighted_retention_order_output(
            &self.auxiliary_heads,
            output.latent.clone(),
            RetentionOrderBatch {
                retention_time: batch.retention_time,
                retention_present: batch.retention_present,
                filename_id: batch.filename_id,
                partner_index: batch.retention_partner_index,
            },
            self.retention_pairs_per_batch,
            self.retention_order_weight,
        );
        let intruder = weighted_intruder_detection_loss(
            &self.auxiliary_heads,
            output.latent.clone(),
            batch.intruder_peak_mask,
            input_peak_mask,
            self.intruder_peak_weight,
        );
        let regularization = self.regularization_config().penalty(self, &device);
        let diagnostics = AutoencoderDiagnostics {
            retention_bce: retention.raw_loss,
            retention_pairs: retention.valid_pairs,
            retention_accuracy: retention.accuracy,
            retention_logit_std: retention.logit_std,
            retention_rt_gap: retention.mean_rt_delta,
        };
        let losses = AutoencoderLossBreakdown {
            reconstruction,
            masked,
            consistency,
            retention: retention.loss,
            intruder,
            regularization,
        };

        PeakSetReconstructionOutput {
            reconstruction: output.reconstruction,
            target,
            target_mask,
            losses,
            diagnostics,
        }
    }
}

#[cfg(feature = "train")]
fn target_triples<B: Backend>(
    target_pairs: Tensor<B, 2>,
    target_mask: Tensor<B, 2>,
    batch_size: usize,
    max_peaks: usize,
) -> Tensor<B, 3> {
    Tensor::cat(
        vec![
            target_pairs.reshape([batch_size, max_peaks, 2]),
            target_mask.unsqueeze_dim::<3>(2),
        ],
        2,
    )
}

#[cfg(feature = "train")]
mod train_impl {
    use burn::{
        tensor::backend::AutodiffBackend,
        train::{InferenceStep, TrainOutput, TrainStep},
    };

    use crate::{batch::TokenizedAutoencoderBatch, training::AutoencoderTrainingOutput};

    use super::*;

    impl<B: AutodiffBackend> TrainStep for PeakSetAutoencoder<B> {
        type Input = TokenizedAutoencoderBatch<B>;
        type Output = AutoencoderTrainingOutput<B>;

        fn step(
            &self,
            batch: TokenizedAutoencoderBatch<B>,
        ) -> TrainOutput<AutoencoderTrainingOutput<B>> {
            let output = self.forward_reconstruction(batch, true);
            let reconstruction = output.reconstruction;
            let [batch_size, max_peaks, output_width] = reconstruction.dims();
            let item = AutoencoderTrainingOutput::new_with_diagnostics(
                reconstruction.reshape([batch_size, max_peaks * output_width]),
                target_triples(output.target, output.target_mask, batch_size, max_peaks)
                    .reshape([batch_size, max_peaks * output_width]),
                output.losses,
                output.diagnostics,
            );

            TrainOutput::new(self, item.loss.clone().backward(), item)
        }
    }

    impl<B: Backend> InferenceStep for PeakSetAutoencoder<B> {
        type Input = TokenizedAutoencoderBatch<B>;
        type Output = AutoencoderTrainingOutput<B>;

        fn step(&self, batch: TokenizedAutoencoderBatch<B>) -> AutoencoderTrainingOutput<B> {
            let output = self.forward_reconstruction(batch, false);
            let reconstruction = output.reconstruction;
            let [batch_size, max_peaks, output_width] = reconstruction.dims();

            AutoencoderTrainingOutput::new_with_diagnostics(
                reconstruction.reshape([batch_size, max_peaks * output_width]),
                target_triples(output.target, output.target_mask, batch_size, max_peaks)
                    .reshape([batch_size, max_peaks * output_width]),
                output.losses,
                output.diagnostics,
            )
        }
    }
}

#[cfg(all(test, feature = "ndarray"))]
mod tests {
    use burn::data::dataloader::batcher::Batcher;
    use burn::module::Module;

    use crate::batch::{
        TokenizedAutoencoderBatch, TokenizedAutoencoderBatcher, TokenizedAutoencoderSample,
    };

    use super::*;

    #[test]
    fn peak_set_model_forward_shapes_are_stable() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let config = PeakSetAutoencoderConfig::symmetric(4, 5, 16, 32, 16, 4, vec![64]);
        let model = config.init::<B>(&device);
        let batch: TokenizedAutoencoderBatch<B> = TokenizedAutoencoderBatcher.batch(
            vec![TokenizedAutoencoderSample {
                token_features: vec![0.0; 20],
                target_pairs: vec![0.0; 8],
                peak_mask: vec![1.0, 1.0, 0.0, 0.0],
                padding_mask: vec![false, false, true, true],
                conditions: vec![0.0; 16],
                metadata: Default::default(),
            }],
            &device,
        );

        let output = model.forward(
            batch.token_features,
            batch.peak_mask,
            batch.padding_mask,
            batch.conditions,
        );
        assert_eq!(output.latent.dims(), [1, 32]);
        assert_eq!(output.reconstruction.dims(), [1, 4, 3]);
    }

    #[test]
    fn twenty_million_run_config_is_stable() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let config = PeakSetAutoencoderConfig::twenty_million_run();
        let model = config.init::<B>(&device);

        assert_eq!(config.encoder.max_peaks, 60);
        assert_eq!(config.encoder.token_feature_width, 19);
        assert_eq!(config.encoder.condition_width, 16);
        assert_eq!(config.encoder.token_embedding_width, 512);
        assert_eq!(config.encoder.attention_heads, 8);
        assert_eq!(config.encoder.transformer_layers, 6);
        assert_eq!(config.decoder.decoder_layers, 4);
        assert_eq!(config.encoder.hidden_widths, vec![2048, 1024]);
        assert_eq!(config.encoder.latent_width, 64);
        assert_eq!(config.decoder.latent_width, 64);
        assert_eq!(config.regularization.l1, 0.0);
        assert_eq!(config.regularization.l2, 0.0);
        assert_eq!(config.auxiliary.masked_peak_weight, 0.25);
        assert_eq!(config.auxiliary.consistency_weight, 0.05);
        assert_eq!(config.auxiliary.retention_order_weight, 0.2);
        assert_eq!(config.auxiliary.intruder_peak_weight, 0.05);
        assert_eq!(config.auxiliary.latent_noise_std, 0.02);
        assert_eq!(model.num_params(), 39_086_149);
    }

    #[test]
    fn regularization_contributes_to_set_reconstruction_loss() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let config = PeakSetAutoencoderConfig::symmetric(4, 5, 16, 32, 16, 4, vec![64])
            .with_regularization(RegularizationConfig {
                l1: 1.0e-3,
                l2: 1.0e-3,
            });
        let model = config.init::<B>(&device);
        let reconstruction = Tensor::<B, 3>::zeros([1, 4, 3], &device);
        let target = Tensor::<B, 2>::zeros([1, 8], &device);
        let target_mask = Tensor::<B, 2>::zeros([1, 4], &device);

        let loss = model
            .reconstruction_loss(reconstruction, target, target_mask)
            .into_scalar();

        assert!(loss > 1.0);
    }
}
