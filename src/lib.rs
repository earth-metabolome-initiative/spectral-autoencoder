//! Library components for MS/MS spectrum autoencoders.
//!
//! The crate is intentionally library-first: applications provide spectra and
//! training orchestration, while this crate owns deterministic preprocessing,
//! optional metadata conditioning, Burn model definitions, and validation
//! helpers.

mod peaks;

pub mod augmentation;
pub mod batch;
pub mod conditioning;
pub mod data;
#[cfg(feature = "embed")]
pub mod embed;
#[cfg(feature = "std")]
pub mod embedder;
pub mod error;
pub mod metrics;
pub mod model;
pub mod tokenize;
#[cfg(feature = "train")]
pub mod training;
pub mod vectorize;

#[cfg(feature = "std")]
pub use augmentation::{AugmentingAutoencoderBatcher, AugmentingTokenizedAutoencoderBatcher};
pub use augmentation::{
    SpectrumAugmentationConfig, SpectrumAugmentationConfigBuilder, SpectrumAugmenter,
};
pub use batch::{
    AutoencoderBatch, AutoencoderSample, TokenizedAutoencoderBatch, TokenizedAutoencoderSample,
};
#[cfg(feature = "std")]
pub use batch::{AutoencoderBatcher, TokenizedAutoencoderBatcher};
pub use conditioning::{ConditioningConfig, ConditioningConfigBuilder, ConditioningEncoder};
pub use data::{
    MgfSummary, TokenizedMgfIter, VectorizedMgfIter, summarize_mgf_path, tokenized_dataset_iter,
    tokenized_mgf_iter, vectorized_dataset_iter, vectorized_mgf_iter,
};
#[cfg(feature = "embed")]
pub use embed::{
    AnySource, EmbeddingRecord, EmbeddingSchema, EmbeddingSink, SinkOptions, SourceOptions,
    SpectrumSource, sink_for_path, source_for_path,
};
#[cfg(feature = "std")]
pub use embedder::{
    DEFAULT_EMBED_BATCH_SIZE, EmbedStream, EmbeddingRow, MODEL_CONFIG_FILE, MODEL_RECORD_FILE,
    SavedSpectrumModelConfig, SpectrumEmbedder, SpectrumEmbedderBuilder,
};
pub use error::{Error, Result};
pub use metrics::{
    DenseReconstructionMetrics, SpectralMetricConfig, SpectralMetricConfigBuilder, SpectralMetrics,
};
pub use model::{
    AutoencoderOutput, AuxiliaryLossConfig, AuxiliaryLossConfigBuilder, DEFAULT_CHAMFER_MZ_WEIGHT,
    DEFAULT_MZ_TOLERANCE_DECAY_EPOCHS, Decoder, DecoderConfig, DecoderConfigBuilder,
    EmbeddingAuxiliaryHeads, EmbeddingAuxiliaryHeadsConfig,
    EmbeddingAuxiliaryHeadsConfigBuilder, Encoder, EncoderConfig, EncoderConfigBuilder,
    FlatVectorReconstructionOrdering, PeakSetAutoencoder, PeakSetAutoencoderConfig,
    PeakSetAutoencoderConfigBuilder, PeakSetAutoencoderOutput, PeakSetDecoder,
    PeakSetDecoderConfig, PeakSetDecoderConfigBuilder, PeakSetEncoder, PeakSetEncoderConfig,
    PeakSetEncoderConfigBuilder, PeakSetLossConfig, RegularizationConfig,
    RegularizationConfigBuilder, SetReconstructionLossConfig, SetReconstructionLossConfigBuilder,
    SimilarityRankingBatch, SpectralAutoencoder, SpectralAutoencoderConfig,
    SpectralAutoencoderConfigBuilder,
};
pub use tokenize::{
    SpectrumTokenizer, SpectrumTokenizerConfig, SpectrumTokenizerConfigBuilder, SpectrumTokens,
};
#[cfg(feature = "train")]
pub use training::AutoencoderTrainingMetricsExt;
pub use vectorize::{
    SpectrumVector, SpectrumVectorizer, SpectrumVectorizerConfig, SpectrumVectorizerConfigBuilder,
};
