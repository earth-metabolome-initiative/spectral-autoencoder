//! Input-side trait, dispatch, and per-format readers.

#[cfg(feature = "embed-mgf")]
mod mgf;
mod stdin;

#[cfg(feature = "embed-mgf")]
pub use mgf::MgfSource;
pub use stdin::StdinSource;

use std::path::Path;

#[cfg(feature = "embed-mgf")]
use mascot_rs::prelude::MascotGenericFormat;
use mass_spectrometry::prelude::SpectrumAlloc;

use crate::{Error, Result};

/// Streaming source of spectra.
///
/// Each source impl picks its own concrete `Item` type. The natural choice
/// is the format-native type the underlying parser produces, e.g.
/// `MascotGenericFormat<f32>` for MGF. Items must implement
/// [`SpectrumAlloc<Precision = f32>`] so they can be fed directly to
/// [`crate::SpectrumEmbedder::embed`] without an intermediate copy.
///
/// `SpectrumSource` is **not** dyn-compatible because it carries an
/// associated type. Heterogeneous dispatch over multiple source impls is
/// handled by the [`AnySource`] enum returned from [`source_for_path`].
pub trait SpectrumSource {
    /// Per-source concrete spectrum type.
    type Item: SpectrumAlloc<Precision = f32>;

    /// Yields the next record, or `None` when the source is exhausted.
    ///
    /// # Errors
    ///
    /// Returns an error when the underlying I/O fails or when a record is
    /// structurally invalid.
    fn next(&mut self) -> Result<Option<Self::Item>>;
}

/// Per-source tuning options threaded in by the bin's CLI. Reserved for
/// future format-specific knobs. Currently a placeholder, MGF parsing is
/// fully driven by `mascot-rs`'s defaults.
#[derive(Debug, Clone, Default)]
pub struct SourceOptions {}

/// Dispatch enum returned by [`source_for_path`]. Wraps the concrete
/// source variants in a single owned type the bin can drive without
/// trait-object machinery (`Spectrum` is not dyn-compatible).
///
/// All current variants yield `MascotGenericFormat<f32>`, so the
/// `SpectrumSource::Item` of `AnySource` is concrete.
#[allow(clippy::large_enum_variant)]
pub enum AnySource {
    /// Stdin reader (MGF semantics under the `embed-mgf` feature).
    Stdin(StdinSource),
    /// MGF file reader (`.mgf`, `.mgf.zst`, `.mgf.gz`).
    #[cfg(feature = "embed-mgf")]
    Mgf(MgfSource),
}

#[cfg(feature = "embed-mgf")]
impl SpectrumSource for AnySource {
    type Item = MascotGenericFormat<f32>;

    fn next(&mut self) -> Result<Option<Self::Item>> {
        match self {
            Self::Stdin(source) => source.next(),
            Self::Mgf(source) => source.next(),
        }
    }
}

#[cfg(not(feature = "embed-mgf"))]
impl SpectrumSource for AnySource {
    type Item = MascotGenericFormat<f32>;

    fn next(&mut self) -> Result<Option<Self::Item>> {
        match self {
            Self::Stdin(source) => source.next(),
        }
    }
}

/// Dispatches to the right concrete source based on the input path's
/// extension and wraps it in an [`AnySource`].
///
/// The bare path `-` (or `/dev/stdin`) is treated as stdin with MGF
/// semantics. Recognised extensions (requires `embed-mgf`):
///
/// - `.mgf`, `.mgf.zst`, `.mgf.gz` -> [`MgfSource`] (decompression handled
///   transparently by `mascot-rs`)
///
/// # Errors
///
/// Returns [`Error::InvalidBatch`] for unknown extensions or when the file
/// cannot be opened.
pub fn source_for_path(path: &Path, _options: &SourceOptions) -> Result<AnySource> {
    if is_stdin_path(path) {
        return Ok(AnySource::Stdin(StdinSource::new()));
    }

    let ext = compound_extension(path);
    match ext.as_deref() {
        #[cfg(feature = "embed-mgf")]
        Some("mgf" | "mgf.zst" | "mgf.zstd" | "mgf.gz" | "mgf.gzip") => {
            Ok(AnySource::Mgf(MgfSource::from_path(path)?))
        }
        Some(other) => Err(Error::InvalidBatch(format!(
            "unsupported input extension `.{other}` for {}",
            path.display()
        ))),
        None => Err(Error::InvalidBatch(format!(
            "input path {} has no extension; use `-` for stdin",
            path.display()
        ))),
    }
}

fn is_stdin_path(path: &Path) -> bool {
    path == Path::new("-") || path == Path::new("/dev/stdin")
}

/// Returns the file's compound extension (e.g. `"mgf.zst"` for
/// `library.mgf.zst`), lowercased. Falls back to the single-extension form
/// when no compound suffix is recognised.
fn compound_extension(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_str()?.to_ascii_lowercase();
    for compound in ["mgf.zst", "mgf.zstd", "mgf.gz", "mgf.gzip"] {
        if name.ends_with(&format!(".{compound}")) {
            return Some(compound.to_string());
        }
    }
    path.extension()
        .and_then(|s| s.to_str())
        .map(str::to_ascii_lowercase)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compound_extension_picks_mgf_compressed_suffix() {
        assert_eq!(
            compound_extension(Path::new("a.mgf")).as_deref(),
            Some("mgf")
        );
        assert_eq!(
            compound_extension(Path::new("a.MGF.zst")).as_deref(),
            Some("mgf.zst")
        );
        assert_eq!(
            compound_extension(Path::new("a.mgf.gz")).as_deref(),
            Some("mgf.gz")
        );
        assert_eq!(compound_extension(Path::new("/dev/stdin")), None);
    }

    #[test]
    fn stdin_dispatch_yields_stdin_source() {
        let source = source_for_path(Path::new("-"), &SourceOptions::default())
            .expect("stdin source dispatch");
        assert!(matches!(source, AnySource::Stdin(_)));
    }

    #[test]
    fn unknown_extension_is_rejected() {
        let result = source_for_path(Path::new("data.xyz"), &SourceOptions::default());
        assert!(matches!(result, Err(Error::InvalidBatch(_))));
    }
}
