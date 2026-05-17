//! Stdin reader. Treats stdin as MGF input when the `embed-mgf` feature is
//! enabled. Otherwise errors on the first `next` call.

use crate::{Result, embed::source::SpectrumSource};

#[cfg(not(feature = "embed-mgf"))]
use crate::Error;
#[cfg(not(feature = "embed-mgf"))]
use mascot_rs::prelude::MascotGenericFormat;

#[cfg(feature = "embed-mgf")]
use std::io::BufReader;

#[cfg(feature = "embed-mgf")]
use mascot_rs::mascot_generic_format::MGFReader;
#[cfg(feature = "embed-mgf")]
use mascot_rs::prelude::{MGFIter, MascotGenericFormat};
#[cfg(feature = "embed-mgf")]
use mass_spectrometry::prelude::Spectrum as SpectrumTrait;

/// Stdin source. With the `embed-mgf` feature, parses stdin as MGF.
pub struct StdinSource {
    #[cfg(feature = "embed-mgf")]
    iter: MGFIter<f32, MGFReader<BufReader<std::io::Stdin>>>,
}

impl StdinSource {
    /// Builds a stdin source.
    #[must_use]
    pub fn new() -> Self {
        #[cfg(feature = "embed-mgf")]
        {
            let reader = MGFReader::new(BufReader::new(std::io::stdin()));
            Self {
                iter: MGFIter::<f32, _>::from_line_source(reader).skipping_invalid_records(),
            }
        }
        #[cfg(not(feature = "embed-mgf"))]
        {
            Self {}
        }
    }
}

impl Default for StdinSource {
    fn default() -> Self {
        Self::new()
    }
}

impl SpectrumSource for StdinSource {
    type Item = MascotGenericFormat<f32>;

    #[cfg(feature = "embed-mgf")]
    fn next(&mut self) -> Result<Option<Self::Item>> {
        loop {
            let Some(result) = self.iter.next() else {
                return Ok(None);
            };
            let Ok(record) = result else {
                continue;
            };
            if record.len() == 0 {
                continue;
            }
            return Ok(Some(record));
        }
    }

    #[cfg(not(feature = "embed-mgf"))]
    fn next(&mut self) -> Result<Option<Self::Item>> {
        Err(Error::InvalidBatch(
            "stdin input requires the `embed-mgf` feature".to_string(),
        ))
    }
}
