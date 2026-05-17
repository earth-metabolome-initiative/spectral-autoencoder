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

    /// A config builder was missing a required field at build time.
    #[error("missing required field `{field}` while building {config}")]
    IncompleteBuilder {
        /// Name of the config type being built.
        config: &'static str,
        /// Name of the unset required field.
        field: &'static str,
    },

    /// I/O failure during checkpoint load or save.
    #[error("io error at {path}: {source}")]
    Io {
        /// Path being read or written.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// JSON (de)serialization failed.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    /// A checkpoint's model config did not match the expected variant.
    #[error("checkpoint at {path} has wrong variant: expected {expected}, found {found}")]
    CheckpointVariantMismatch {
        /// Checkpoint directory.
        path: PathBuf,
        /// Variant the caller asked for.
        expected: &'static str,
        /// Variant present on disk.
        found: &'static str,
    },

    /// Generic embed-pipeline error wrapping a string message. Used by I/O
    /// adapters (sources/sinks) and the embed bin's loop.
    #[error("{0}")]
    InvalidBatch(String),
}

impl Error {
    /// Convenience constructor for [`Error::Io`].
    #[must_use]
    pub fn io(path: &std::path::Path, source: std::io::Error) -> Self {
        Self::Io {
            path: path.to_path_buf(),
            source,
        }
    }
}
