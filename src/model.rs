//! Burn encoder and decoder modules.

pub mod auxiliary;
pub mod flat_vector;
pub mod peak_set;
pub mod reconstruction;
pub mod regularization;

pub use auxiliary::{
    AuxiliaryLossConfig, AuxiliaryLossConfigBuilder, EmbeddingAuxiliaryHeads,
    EmbeddingAuxiliaryHeadsConfig, EmbeddingAuxiliaryHeadsConfigBuilder, SimilarityRankingBatch,
};
pub use flat_vector::{
    AutoencoderOutput, Decoder, DecoderConfig, DecoderConfigBuilder, Encoder, EncoderConfig,
    EncoderConfigBuilder, SpectralAutoencoder, SpectralAutoencoderConfig,
    SpectralAutoencoderConfigBuilder,
};
pub use peak_set::{
    PeakSetAutoencoder, PeakSetAutoencoderConfig, PeakSetAutoencoderConfigBuilder,
    PeakSetAutoencoderOutput, PeakSetDecoder, PeakSetDecoderConfig, PeakSetDecoderConfigBuilder,
    PeakSetEncoder, PeakSetEncoderConfig, PeakSetEncoderConfigBuilder,
};
pub use reconstruction::{
    FlatVectorReconstructionOrdering, PeakSetLossConfig, SetReconstructionLossConfig,
    SetReconstructionLossConfigBuilder,
};
pub use regularization::{RegularizationConfig, RegularizationConfigBuilder};
