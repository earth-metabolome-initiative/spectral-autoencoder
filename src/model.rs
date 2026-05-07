//! Burn encoder and decoder modules.

pub mod auxiliary;
pub mod flat_vector;
pub mod peak_set;
pub mod reconstruction;
pub mod regularization;

pub use auxiliary::{
    AuxiliaryLossConfig, EmbeddingAuxiliaryHeads, EmbeddingAuxiliaryHeadsConfig,
    SimilarityRankingBatch,
};
pub use flat_vector::{
    AutoencoderOutput, Decoder, DecoderConfig, Encoder, EncoderConfig, SpectralAutoencoder,
    SpectralAutoencoderConfig,
};
pub use peak_set::{
    PeakSetAutoencoder, PeakSetAutoencoderConfig, PeakSetAutoencoderOutput, PeakSetDecoder,
    PeakSetDecoderConfig, PeakSetEncoder, PeakSetEncoderConfig,
};
pub use reconstruction::{
    FlatVectorReconstructionOrdering, PeakSetLossConfig, SetReconstructionLossConfig,
};
pub use regularization::RegularizationConfig;
