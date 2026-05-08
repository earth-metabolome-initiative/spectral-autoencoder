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
#[cfg(feature = "cuda")]
pub mod linear_cosine_cuda;
pub mod metrics;
pub mod model;
pub mod tokenize;
#[cfg(feature = "train")]
pub mod training;
pub mod vectorize;

#[cfg(feature = "std")]
pub use augmentation::{AugmentingAutoencoderBatcher, AugmentingTokenizedAutoencoderBatcher};
pub use augmentation::{SpectrumAugmentationConfig, SpectrumAugmenter};
pub use batch::{
    AutoencoderBatch, AutoencoderSample, TokenizedAutoencoderBatch, TokenizedAutoencoderSample,
};
#[cfg(feature = "std")]
pub use batch::{AutoencoderBatcher, TokenizedAutoencoderBatcher};
pub use conditioning::{ConditioningConfig, ConditioningEncoder};
pub use data::{
    MgfSummary, TokenizedMgfIter, VectorizedMgfIter, summarize_mgf_path, tokenized_mgf_iter,
    tokenized_mgf_paths_iter, vectorized_mgf_iter, vectorized_mgf_paths_iter,
};
pub use error::{Error, Result};
pub use metrics::{DenseReconstructionMetrics, SpectralMetricConfig, SpectralMetrics};
pub use model::{
    AutoencoderOutput, AuxiliaryLossConfig, Decoder, DecoderConfig, EmbeddingAuxiliaryHeads,
    EmbeddingAuxiliaryHeadsConfig, Encoder, EncoderConfig, FlatVectorReconstructionOrdering,
    PeakSetAutoencoder, PeakSetAutoencoderConfig, PeakSetAutoencoderOutput, PeakSetDecoder,
    PeakSetDecoderConfig, PeakSetEncoder, PeakSetEncoderConfig, PeakSetLossConfig,
    RegularizationConfig, SetReconstructionLossConfig, SimilarityRankingBatch, SpectralAutoencoder,
    SpectralAutoencoderConfig,
};
pub use tokenize::{SpectrumTokenizer, SpectrumTokenizerConfig, SpectrumTokens};
#[cfg(feature = "train")]
pub use training::AutoencoderTrainingMetricsExt;
pub use vectorize::{SpectrumVector, SpectrumVectorizer, SpectrumVectorizerConfig};
