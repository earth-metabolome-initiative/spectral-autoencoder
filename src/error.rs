//! Error types used by this crate.

use std::path::PathBuf;

/// Crate result alias.
pub type Result<T> = std::result::Result<T, Error>;

/// Error returned by data loading, preprocessing, model configuration, or
/// metric evaluation.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// MGF parsing or loading failed.
    #[error("failed to read MGF data: {0}")]
    Mascot(#[from] mascot_rs::prelude::MascotError),

    /// A spectral similarity configuration was invalid.
    #[error("invalid spectral similarity configuration: {0}")]
    SimilarityConfig(#[from] mass_spectrometry::structs::SimilarityConfigError),

    /// A spectral similarity computation failed.
    #[error("spectral similarity computation failed: {0}")]
    SimilarityComputation(#[from] mass_spectrometry::structs::SimilarityComputationError),

    /// A reconstructed vector could not be converted into a spectrum.
    #[error("invalid reconstructed spectrum: {0}")]
    SpectrumMutation(#[from] mass_spectrometry::structs::GenericSpectrumMutationError),

    /// Input data had an unexpected shape.
    #[error("invalid vector length {actual}; expected {expected}")]
    InvalidVectorLength {
        /// Observed vector length.
        actual: usize,
        /// Expected vector length.
        expected: usize,
    },

    /// No spectra were available in an input source.
    #[error("no spectra found in {path}")]
    EmptyInput {
        /// Source path.
        path: PathBuf,
    },
}
