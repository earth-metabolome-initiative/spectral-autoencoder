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
    },
    model::reconstruction::{SetReconstructionLossConfig, set_reconstruction_loss_from_triples},
    tokenize::SpectrumTokenizerConfig,
};

fn default_precursor_mz_scale() -> f64 {
    ConditioningConfig::default().precursor_mz_scale()
}

#[cfg(feature = "train")]
use crate::{
    model::auxiliary::{
        apply_latent_noise, weighted_intruder_detection_loss,
        weighted_masked_precursor_reconstruction_output, weighted_precursor_reconstruction_output,
        weighted_similarity_ranking_output,
    },
    model::reconstruction::reconstruction_similarity_from_triples,
    training::{AutoencoderDiagnostics, AutoencoderLossBreakdown},
};

/// Peak-token encoder configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeakSetEncoderConfig {
    /// Maximum number of peak tokens per spectrum.
    pub max_peaks: usize,
    /// Width of each peak token feature vector.
    pub token_feature_width: usize,
    /// Width of the encoder-side metadata conditioning vector.
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
    /// Starts a fluent builder. All fields are required before
    /// [`PeakSetEncoderConfigBuilder::build`].
    #[must_use]
    pub fn builder() -> PeakSetEncoderConfigBuilder {
        PeakSetEncoderConfigBuilder::default()
    }

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

/// Fluent builder for [`PeakSetEncoderConfig`].
#[derive(Debug, Clone, Default)]
pub struct PeakSetEncoderConfigBuilder {
    max_peaks: Option<usize>,
    token_feature_width: Option<usize>,
    condition_width: Option<usize>,
    token_embedding_width: Option<usize>,
    attention_heads: Option<usize>,
    transformer_layers: Option<usize>,
    transformer_feed_forward_width: Option<usize>,
    dropout: Option<f64>,
    hidden_widths: Option<Vec<usize>>,
    latent_width: Option<usize>,
}

impl PeakSetEncoderConfigBuilder {
    /// Creates an empty builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the maximum number of peak tokens per spectrum (required).
    #[inline]
    #[must_use]
    pub fn with_max_peaks(mut self, value: usize) -> Self {
        self.max_peaks = Some(value);
        self
    }

    /// Sets the per-token feature width (required).
    #[inline]
    #[must_use]
    pub fn with_token_feature_width(mut self, value: usize) -> Self {
        self.token_feature_width = Some(value);
        self
    }

    /// Sets the encoder-side metadata-condition width (required; may be `0`).
    #[inline]
    #[must_use]
    pub fn with_condition_width(mut self, value: usize) -> Self {
        self.condition_width = Some(value);
        self
    }

    /// Sets the internal peak-token embedding width (required).
    #[inline]
    #[must_use]
    pub fn with_token_embedding_width(mut self, value: usize) -> Self {
        self.token_embedding_width = Some(value);
        self
    }

    /// Sets the number of self-attention heads (required).
    #[inline]
    #[must_use]
    pub fn with_attention_heads(mut self, value: usize) -> Self {
        self.attention_heads = Some(value);
        self
    }

    /// Sets the number of transformer encoder layers (required).
    #[inline]
    #[must_use]
    pub fn with_transformer_layers(mut self, value: usize) -> Self {
        self.transformer_layers = Some(value);
        self
    }

    /// Sets the transformer feed-forward block width (required).
    #[inline]
    #[must_use]
    pub fn with_transformer_feed_forward_width(mut self, value: usize) -> Self {
        self.transformer_feed_forward_width = Some(value);
        self
    }

    /// Sets the transformer dropout probability (required).
    #[inline]
    #[must_use]
    pub fn with_dropout(mut self, value: f64) -> Self {
        self.dropout = Some(value);
        self
    }

    /// Sets the post-pooling hidden-layer widths (required).
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
    pub fn build(self) -> crate::Result<PeakSetEncoderConfig> {
        let max_peaks = self
            .max_peaks
            .ok_or_else(|| crate::Error::IncompleteBuilder {
                config: "PeakSetEncoderConfig",
                field: "max_peaks",
            })?;
        let token_feature_width =
            self.token_feature_width
                .ok_or_else(|| crate::Error::IncompleteBuilder {
                    config: "PeakSetEncoderConfig",
                    field: "token_feature_width",
                })?;
        let condition_width =
            self.condition_width
                .ok_or_else(|| crate::Error::IncompleteBuilder {
                    config: "PeakSetEncoderConfig",
                    field: "condition_width",
                })?;
        let token_embedding_width =
            self.token_embedding_width
                .ok_or_else(|| crate::Error::IncompleteBuilder {
                    config: "PeakSetEncoderConfig",
                    field: "token_embedding_width",
                })?;
        let attention_heads =
            self.attention_heads
                .ok_or_else(|| crate::Error::IncompleteBuilder {
                    config: "PeakSetEncoderConfig",
                    field: "attention_heads",
                })?;
        let transformer_layers =
            self.transformer_layers
                .ok_or_else(|| crate::Error::IncompleteBuilder {
                    config: "PeakSetEncoderConfig",
                    field: "transformer_layers",
                })?;
        let transformer_feed_forward_width =
            self.transformer_feed_forward_width
                .ok_or_else(|| crate::Error::IncompleteBuilder {
                    config: "PeakSetEncoderConfig",
                    field: "transformer_feed_forward_width",
                })?;
        let dropout = self
            .dropout
            .ok_or_else(|| crate::Error::IncompleteBuilder {
                config: "PeakSetEncoderConfig",
                field: "dropout",
            })?;
        let hidden_widths = self
            .hidden_widths
            .ok_or_else(|| crate::Error::IncompleteBuilder {
                config: "PeakSetEncoderConfig",
                field: "hidden_widths",
            })?;
        let latent_width = self
            .latent_width
            .ok_or_else(|| crate::Error::IncompleteBuilder {
                config: "PeakSetEncoderConfig",
                field: "latent_width",
            })?;
        Ok(PeakSetEncoderConfig {
            max_peaks,
            token_feature_width,
            condition_width,
            token_embedding_width,
            attention_heads,
            transformer_layers,
            transformer_feed_forward_width,
            dropout,
            hidden_widths,
            latent_width,
        })
    }
}

/// Query-set decoder configuration for reconstructed peak candidates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeakSetDecoderConfig {
    /// Maximum number of reconstructed peak candidates.
    pub max_peaks: usize,
    /// Width of the latent embedding.
    pub latent_width: usize,
    /// Width of the decoder-side metadata conditioning vector.
    ///
    /// The diffusion-oriented default is zero so the decoder reconstructs from
    /// the latent embedding alone.
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
    /// Width of the reconstructed metadata condition vector.
    pub condition_output_width: usize,
}

impl PeakSetDecoderConfig {
    /// Starts a fluent builder. All fields are required before
    /// [`PeakSetDecoderConfigBuilder::build`].
    #[must_use]
    pub fn builder() -> PeakSetDecoderConfigBuilder {
        PeakSetDecoderConfigBuilder::default()
    }

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
            condition_output: LinearConfig::new(self.query_width, self.condition_output_width)
                .init(device),
            max_peaks: self.max_peaks,
            query_width: self.query_width,
            condition_width: self.condition_width,
        }
    }
}

/// Fluent builder for [`PeakSetDecoderConfig`].
#[derive(Debug, Clone, Default)]
pub struct PeakSetDecoderConfigBuilder {
    max_peaks: Option<usize>,
    latent_width: Option<usize>,
    condition_width: Option<usize>,
    query_width: Option<usize>,
    attention_heads: Option<usize>,
    decoder_layers: Option<usize>,
    decoder_feed_forward_width: Option<usize>,
    dropout: Option<f64>,
    condition_output_width: Option<usize>,
}

impl PeakSetDecoderConfigBuilder {
    /// Creates an empty builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the maximum number of reconstructed peak candidates (required).
    #[inline]
    #[must_use]
    pub fn with_max_peaks(mut self, value: usize) -> Self {
        self.max_peaks = Some(value);
        self
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

    /// Sets the learned-query width (required).
    #[inline]
    #[must_use]
    pub fn with_query_width(mut self, value: usize) -> Self {
        self.query_width = Some(value);
        self
    }

    /// Sets the number of decoder attention heads (required).
    #[inline]
    #[must_use]
    pub fn with_attention_heads(mut self, value: usize) -> Self {
        self.attention_heads = Some(value);
        self
    }

    /// Sets the number of transformer decoder layers (required).
    #[inline]
    #[must_use]
    pub fn with_decoder_layers(mut self, value: usize) -> Self {
        self.decoder_layers = Some(value);
        self
    }

    /// Sets the decoder feed-forward block width (required).
    #[inline]
    #[must_use]
    pub fn with_decoder_feed_forward_width(mut self, value: usize) -> Self {
        self.decoder_feed_forward_width = Some(value);
        self
    }

    /// Sets the decoder dropout probability (required).
    #[inline]
    #[must_use]
    pub fn with_dropout(mut self, value: f64) -> Self {
        self.dropout = Some(value);
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
    pub fn build(self) -> crate::Result<PeakSetDecoderConfig> {
        let max_peaks = self
            .max_peaks
            .ok_or_else(|| crate::Error::IncompleteBuilder {
                config: "PeakSetDecoderConfig",
                field: "max_peaks",
            })?;
        let latent_width = self
            .latent_width
            .ok_or_else(|| crate::Error::IncompleteBuilder {
                config: "PeakSetDecoderConfig",
                field: "latent_width",
            })?;
        let condition_width =
            self.condition_width
                .ok_or_else(|| crate::Error::IncompleteBuilder {
                    config: "PeakSetDecoderConfig",
                    field: "condition_width",
                })?;
        let query_width = self
            .query_width
            .ok_or_else(|| crate::Error::IncompleteBuilder {
                config: "PeakSetDecoderConfig",
                field: "query_width",
            })?;
        let attention_heads =
            self.attention_heads
                .ok_or_else(|| crate::Error::IncompleteBuilder {
                    config: "PeakSetDecoderConfig",
                    field: "attention_heads",
                })?;
        let decoder_layers =
            self.decoder_layers
                .ok_or_else(|| crate::Error::IncompleteBuilder {
                    config: "PeakSetDecoderConfig",
                    field: "decoder_layers",
                })?;
        let decoder_feed_forward_width =
            self.decoder_feed_forward_width
                .ok_or_else(|| crate::Error::IncompleteBuilder {
                    config: "PeakSetDecoderConfig",
                    field: "decoder_feed_forward_width",
                })?;
        let dropout = self
            .dropout
            .ok_or_else(|| crate::Error::IncompleteBuilder {
                config: "PeakSetDecoderConfig",
                field: "dropout",
            })?;
        let condition_output_width =
            self.condition_output_width
                .ok_or_else(|| crate::Error::IncompleteBuilder {
                    config: "PeakSetDecoderConfig",
                    field: "condition_output_width",
                })?;
        Ok(PeakSetDecoderConfig {
            max_peaks,
            latent_width,
            condition_width,
            query_width,
            attention_heads,
            decoder_layers,
            decoder_feed_forward_width,
            dropout,
            condition_output_width,
        })
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
    /// Scale used to denormalize reconstructed precursor m/z diagnostics.
    #[serde(default = "default_precursor_mz_scale")]
    pub precursor_mz_scale: f64,
    /// Auxiliary denoising, intruder, precursor, and similarity-ranking objectives.
    #[serde(default)]
    pub auxiliary: AuxiliaryLossConfig,
}

impl PeakSetAutoencoderConfig {
    /// Starts a fluent builder. `encoder`, `decoder`, and `loss` are required;
    /// remaining fields fall back to their type's default.
    #[must_use]
    pub fn builder() -> PeakSetAutoencoderConfigBuilder {
        PeakSetAutoencoderConfigBuilder::default()
    }

    /// Starting peak-set transformer configuration for the 20M-spectrum run.
    ///
    /// This is the recommended large-run configuration: top-N peak tokens,
    /// a 768-wide encoder/decoder, six transformer encoder layers, four set
    /// decoder layers, a 12-head attention layout, and a 256-dimensional
    /// latent embedding.
    #[must_use]
    pub fn twenty_million_run() -> Self {
        Self::twenty_million_run_with_peaks(SpectrumTokenizerConfig::default().max_peaks)
    }

    /// Starting peak-set transformer configuration for a top-N GeMS run.
    #[must_use]
    pub fn twenty_million_run_with_peaks(max_peaks: usize) -> Self {
        let tokenizer = SpectrumTokenizerConfig {
            max_peaks,
            ..SpectrumTokenizerConfig::default()
        };
        let condition_width = ConditioningConfig::default().vector_width();
        let token_embedding_width = 768;
        let latent_width = 256;

        Self {
            encoder: PeakSetEncoderConfig {
                max_peaks: tokenizer.max_peaks,
                token_feature_width: tokenizer.feature_width(),
                condition_width,
                token_embedding_width,
                attention_heads: 12,
                transformer_layers: 6,
                transformer_feed_forward_width: token_embedding_width * 4,
                dropout: 0.1,
                hidden_widths: vec![4096, 2048],
                latent_width,
            },
            decoder: PeakSetDecoderConfig {
                max_peaks: tokenizer.max_peaks,
                latent_width,
                condition_width: 0,
                query_width: token_embedding_width,
                attention_heads: 12,
                decoder_layers: 4,
                decoder_feed_forward_width: token_embedding_width * 4,
                dropout: 0.1,
                condition_output_width: condition_width,
            },
            loss: SetReconstructionLossConfig::default(),
            regularization: RegularizationConfig::default(),
            precursor_mz_scale: ConditioningConfig::default().precursor_mz_scale(),
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
                condition_width: 0,
                query_width: token_embedding_width,
                attention_heads,
                decoder_layers: 2,
                decoder_feed_forward_width: token_embedding_width * 4,
                dropout: 0.1,
                condition_output_width: condition_width,
            },
            loss: SetReconstructionLossConfig::default(),
            regularization: RegularizationConfig::default(),
            precursor_mz_scale: ConditioningConfig::default().precursor_mz_scale(),
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
    condition_output: Linear<B>,
    max_peaks: usize,
    query_width: usize,
    condition_width: usize,
}

impl<B: Backend> PeakSetDecoder<B> {
    /// Decodes a latent representation into unordered normalized peak candidates.
    pub fn forward(&self, latent: Tensor<B, 2>) -> PeakSetDecoderOutput<B> {
        assert_eq!(
            self.condition_width, 0,
            "PeakSetDecoder::forward requires a zero-width decoder condition configuration"
        );
        self.forward_memory(latent)
    }

    /// Decodes a latent representation with explicit decoder-side conditions.
    pub fn forward_with_conditions(
        &self,
        latent: Tensor<B, 2>,
        conditions: Tensor<B, 2>,
    ) -> PeakSetDecoderOutput<B> {
        let [_batch_size, condition_width] = conditions.dims();
        assert_eq!(
            condition_width, self.condition_width,
            "decoder condition tensor width does not match the decoder configuration"
        );
        self.forward_memory(Tensor::cat(vec![latent, conditions], 1))
    }

    fn forward_memory(&self, memory_input: Tensor<B, 2>) -> PeakSetDecoderOutput<B> {
        let [batch_size, _memory_width] = memory_input.dims();
        let memory = self
            .memory_projection
            .forward(memory_input)
            .unsqueeze_dim::<3>(1);
        let queries = self.queries.val().unsqueeze_dim::<3>(0).expand([
            batch_size,
            self.max_peaks,
            self.query_width,
        ]);
        let decoded = self
            .decoder
            .forward(TransformerDecoderInput::new(queries, memory.clone()));
        PeakSetDecoderOutput {
            peaks: sigmoid(self.output.forward(decoded)),
            conditions: sigmoid(
                self.condition_output
                    .forward(memory.reshape([batch_size, self.query_width])),
            ),
        }
    }
}

/// Output returned by the peak-set decoder.
pub struct PeakSetDecoderOutput<B: Backend> {
    /// Reconstructed normalized `(m/z, intensity, presence)` peak candidates.
    pub peaks: Tensor<B, 3>,
    /// Reconstructed precursor condition vector.
    pub conditions: Tensor<B, 2>,
}

/// Output returned by the peak-token autoencoder.
pub struct PeakSetAutoencoderOutput<B: Backend> {
    /// Latent spectral embedding.
    pub latent: Tensor<B, 2>,
    /// Reconstructed normalized `(m/z, intensity, presence)` peak candidates.
    pub reconstruction: Tensor<B, 3>,
    /// Reconstructed precursor condition vector.
    pub condition_reconstruction: Tensor<B, 2>,
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
        let latent = self
            .encoder
            .forward(token_features, peak_mask, padding_mask, conditions);
        let decoder_output = self.decoder.forward(latent.clone());
        PeakSetAutoencoderOutput {
            latent,
            reconstruction: decoder_output.peaks,
            condition_reconstruction: decoder_output.conditions,
        }
    }

    /// Runs the full peak-token autoencoder with explicit decoder-side conditions.
    pub fn forward_with_decoder_conditions(
        &self,
        token_features: Tensor<B, 3>,
        peak_mask: Tensor<B, 2>,
        padding_mask: Tensor<B, 2, burn::tensor::Bool>,
        encoder_conditions: Tensor<B, 2>,
        decoder_conditions: Tensor<B, 2>,
    ) -> PeakSetAutoencoderOutput<B> {
        let latent =
            self.encoder
                .forward(token_features, peak_mask, padding_mask, encoder_conditions);
        let decoder_output = self
            .decoder
            .forward_with_conditions(latent.clone(), decoder_conditions);
        PeakSetAutoencoderOutput {
            latent,
            reconstruction: decoder_output.peaks,
            condition_reconstruction: decoder_output.conditions,
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
            batch.conditions,
        );
        let decoder_latent = if use_latent_noise {
            apply_latent_noise(latent.clone(), self.latent_noise_std)
        } else {
            latent.clone()
        };
        let decoder_output = self.decoder.forward(decoder_latent);
        let output = PeakSetAutoencoderOutput {
            latent,
            reconstruction: decoder_output.peaks,
            condition_reconstruction: decoder_output.conditions,
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
        let reconstruction_similarity = reconstruction_similarity_from_triples(
            output.reconstruction.clone(),
            target.clone(),
            target_mask.clone(),
            precursor_target,
            output.condition_reconstruction.clone(),
            self.loss_config(),
        );
        let regularization = self.regularization_config().penalty(self, &device);
        let diagnostics = AutoencoderDiagnostics {
            similarity_ranking_pairs: similarity_ranking.valid_pairs,
            similarity_ranking_accuracy: similarity_ranking.accuracy,
            precursor_mae_da: precursor.mae_da,
            self_linear_cosine: reconstruction_similarity.linear_cosine,
            self_modified_linear_cosine: reconstruction_similarity.modified_linear_cosine,
            self_similarity_items: reconstruction_similarity.items,
        };
        let losses = AutoencoderLossBreakdown {
            reconstruction,
            masked,
            intruder,
            precursor: precursor.loss,
            masked_precursor: masked_precursor.loss,
            similarity_ranking: similarity_ranking.loss,
            regularization,
            chamfer_mz: Tensor::zeros([1], &device),
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

/// Fluent builder for [`PeakSetAutoencoderConfig`].
#[derive(Debug, Clone, Default)]
pub struct PeakSetAutoencoderConfigBuilder {
    encoder: Option<PeakSetEncoderConfig>,
    decoder: Option<PeakSetDecoderConfig>,
    loss: Option<SetReconstructionLossConfig>,
    regularization: Option<RegularizationConfig>,
    precursor_mz_scale: Option<f64>,
    auxiliary: Option<AuxiliaryLossConfig>,
}

impl PeakSetAutoencoderConfigBuilder {
    /// Creates an empty builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the encoder configuration (required).
    #[inline]
    #[must_use]
    pub fn with_encoder(mut self, value: PeakSetEncoderConfig) -> Self {
        self.encoder = Some(value);
        self
    }

    /// Sets the decoder configuration (required).
    #[inline]
    #[must_use]
    pub fn with_decoder(mut self, value: PeakSetDecoderConfig) -> Self {
        self.decoder = Some(value);
        self
    }

    /// Sets the set-reconstruction loss config (required).
    #[inline]
    #[must_use]
    pub fn with_loss(mut self, value: SetReconstructionLossConfig) -> Self {
        self.loss = Some(value);
        self
    }

    /// Overrides the regularization config.
    #[inline]
    #[must_use]
    pub fn with_regularization(mut self, value: RegularizationConfig) -> Self {
        self.regularization = Some(value);
        self
    }

    /// Overrides the diagnostic precursor m/z scale.
    #[inline]
    #[must_use]
    pub fn with_precursor_mz_scale(mut self, value: f64) -> Self {
        self.precursor_mz_scale = Some(value);
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
    /// Returns [`crate::Error::IncompleteBuilder`] when `encoder`, `decoder`,
    /// or `loss` is unset.
    pub fn build(self) -> crate::Result<PeakSetAutoencoderConfig> {
        let encoder = self
            .encoder
            .ok_or_else(|| crate::Error::IncompleteBuilder {
                config: "PeakSetAutoencoderConfig",
                field: "encoder",
            })?;
        let decoder = self
            .decoder
            .ok_or_else(|| crate::Error::IncompleteBuilder {
                config: "PeakSetAutoencoderConfig",
                field: "decoder",
            })?;
        let loss = self.loss.ok_or_else(|| crate::Error::IncompleteBuilder {
            config: "PeakSetAutoencoderConfig",
            field: "loss",
        })?;
        Ok(PeakSetAutoencoderConfig {
            encoder,
            decoder,
            loss,
            regularization: self.regularization.unwrap_or_default(),
            precursor_mz_scale: self
                .precursor_mz_scale
                .unwrap_or_else(default_precursor_mz_scale),
            auxiliary: self.auxiliary.unwrap_or_default(),
        })
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
        assert_eq!(output.condition_reconstruction.dims(), [1, 16]);
    }

    #[test]
    fn twenty_million_run_config_is_stable() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let config = PeakSetAutoencoderConfig::twenty_million_run();
        let model = config.init::<B>(&device);

        assert_eq!(config.encoder.max_peaks, 60);
        assert_eq!(config.encoder.token_feature_width, 19);
        assert_eq!(config.encoder.condition_width, 2);
        assert_eq!(config.decoder.condition_width, 0);
        assert_eq!(config.decoder.condition_output_width, 2);
        assert_eq!(config.encoder.token_embedding_width, 768);
        assert_eq!(config.encoder.attention_heads, 12);
        assert_eq!(config.encoder.transformer_layers, 6);
        assert_eq!(config.decoder.decoder_layers, 4);
        assert_eq!(config.encoder.hidden_widths, vec![4096, 2048]);
        assert_eq!(config.encoder.latent_width, 256);
        assert_eq!(config.decoder.latent_width, 256);
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
        assert_eq!(model.num_params(), 92_710_918);
    }

    #[test]
    fn twenty_million_run_config_accepts_larger_peak_counts() {
        let config = PeakSetAutoencoderConfig::twenty_million_run_with_peaks(128);

        assert_eq!(config.encoder.max_peaks, 128);
        assert_eq!(config.decoder.max_peaks, 128);
        assert_eq!(config.decoder.condition_width, 0);
        assert_eq!(config.encoder.token_feature_width, 19);
        assert_eq!(config.encoder.latent_width, 256);
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
