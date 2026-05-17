//! `.mgf` reader using `mascot-rs`'s streaming path iterator.
//!
//! `MGFVec::<f32>::iter_from_path` opens the file, applies the right
//! decompressor for `.mgf` / `.mgf.zst` / `.mgf.gz`, and yields parsed
//! `MascotGenericFormat<f32>` records lazily. Those records already
//! implement [`mass_spectrometry::prelude::SpectrumAlloc`], so we hand
//! them straight to the encoder with no intermediate copy.
//!
//! Records that fail to parse are skipped silently. This matches the
//! default policy of `mascot_rs`'s streaming iterator.

use std::path::Path;

use mascot_rs::prelude::{MGFPathIter, MGFVec, MascotGenericFormat};
use mass_spectrometry::prelude::Spectrum as SpectrumTrait;

use crate::{Error, Result, embed::source::SpectrumSource};

/// Streaming `.mgf` reader. Compression is handled transparently for
/// `.mgf.zst`, `.mgf.zstd`, `.mgf.gz`, and `.mgf.gzip`.
pub struct MgfSource {
    iter: MGFPathIter<f32>,
}

impl MgfSource {
    /// Opens the MGF file at `path`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidBatch`] when the file cannot be opened or
    /// the decompressor cannot be initialised.
    pub fn from_path(path: &Path) -> Result<Self> {
        let iter = MGFVec::<f32>::iter_from_path(path).map_err(|source| {
            Error::InvalidBatch(format!("failed to open MGF {}: {source}", path.display()))
        })?;
        Ok(Self { iter })
    }
}

impl SpectrumSource for MgfSource {
    type Item = MascotGenericFormat<f32>;

    fn next(&mut self) -> Result<Option<Self::Item>> {
        loop {
            let Some(result) = self.iter.next() else {
                return Ok(None);
            };
            let Ok(record) = result else {
                // Mirror `mascot_rs`'s skip-invalid default policy.
                continue;
            };
            if record.len() == 0 {
                continue;
            }
            return Ok(Some(record));
        }
    }
}
