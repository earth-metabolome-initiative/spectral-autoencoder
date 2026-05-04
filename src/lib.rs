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
pub mod error;
pub mod metrics;
pub mod model;
pub mod tokenize;
#[cfg(feature = "train")]
pub mod training;
pub mod vectorize;

pub use augmentation::{
    AugmentingAutoencoderBatcher, AugmentingTokenizedAutoencoderBatcher,
    SpectrumAugmentationConfig, SpectrumAugmenter,
};
pub use batch::{
    AutoencoderBatch, AutoencoderBatcher, AutoencoderSample, SampleMetadata,
    TokenizedAutoencoderBatch, TokenizedAutoencoderBatcher, TokenizedAutoencoderSample,
    retention_partner_indices, retention_partner_indices_with_seed,
};
pub use conditioning::{ConditioningConfig, ConditioningEncoder};
pub use data::{
    MgfSummary, TokenizedMgfIter, VectorizedMgfIter, summarize_mgf_path, tokenized_mgf_iter,
    tokenized_mgf_paths_iter, vectorized_mgf_iter, vectorized_mgf_paths_iter,
};
pub use error::{Error, Result};
pub use metrics::{DenseReconstructionMetrics, SpectralMetricConfig, SpectralMetrics};
pub use model::{
    AutoencoderOutput, AuxiliaryLossConfig, Decoder, DecoderConfig, EmbeddingAuxiliaryHeads,
    EmbeddingAuxiliaryHeadsConfig, Encoder, EncoderConfig, PeakSetAutoencoder,
    PeakSetAutoencoderConfig, PeakSetAutoencoderOutput, PeakSetDecoder, PeakSetDecoderConfig,
    PeakSetEncoder, PeakSetEncoderConfig, PeakSetLossConfig, RegularizationConfig,
    SetReconstructionLossConfig, SpectralAutoencoder, SpectralAutoencoderConfig,
};
pub use tokenize::{SpectrumTokenizer, SpectrumTokenizerConfig, SpectrumTokens};
#[cfg(feature = "train")]
pub use training::AutoencoderTrainingMetricsExt;
pub use vectorize::{SpectrumVector, SpectrumVectorizer, SpectrumVectorizerConfig};
