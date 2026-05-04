//! Fixed-length sparse peak-pair vectorization.

use mass_spectrometry::prelude::{GenericSpectrum, SpectrumAlloc};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::peaks::{decode_normalized_peak_pairs, filtered_top_peaks};

/// Dense vector representation of one cleaned spectrum.
#[derive(Debug, Clone, PartialEq)]
pub struct SpectrumVector {
    /// Concatenated `(m/z, intensity)` pairs.
    pub values: Vec<f32>,
    /// Number of retained non-padding peaks.
    pub retained_peaks: usize,
    /// Original number of peaks after finite/range filtering.
    pub candidate_peaks: usize,
}

/// Configuration for converting sparse spectra to fixed-length vectors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectrumVectorizerConfig {
    /// Maximum number of peaks retained per spectrum.
    pub max_peaks: usize,
    /// Minimum m/z retained.
    pub min_mz: f64,
    /// Maximum m/z represented as `1.0` in the vector.
    pub max_mz: f64,
    /// Power applied after max-intensity normalization.
    pub intensity_power: f64,
}

impl Default for SpectrumVectorizerConfig {
    fn default() -> Self {
        Self {
            max_peaks: 60,
            min_mz: 0.0,
            max_mz: 2_000.0,
            intensity_power: 0.5,
        }
    }
}

impl SpectrumVectorizerConfig {
    /// Returns the dense vector width.
    #[must_use]
    pub const fn vector_width(&self) -> usize {
        self.max_peaks * 2
    }
}

/// Converts spectra to and from the crate's fixed-length peak-pair vector.
#[derive(Debug, Clone)]
pub struct SpectrumVectorizer {
    config: SpectrumVectorizerConfig,
}

impl Default for SpectrumVectorizer {
    fn default() -> Self {
        Self::new(SpectrumVectorizerConfig::default())
    }
}

impl SpectrumVectorizer {
    /// Creates a new vectorizer.
    #[must_use]
    pub const fn new(config: SpectrumVectorizerConfig) -> Self {
        Self { config }
    }

    /// Returns the active configuration.
    #[must_use]
    pub const fn config(&self) -> &SpectrumVectorizerConfig {
        &self.config
    }

    /// Returns the dense vector width.
    #[must_use]
    pub const fn vector_width(&self) -> usize {
        self.config.vector_width()
    }

    /// Encodes a spectrum as top-intensity `(m/z, intensity)` pairs.
    pub fn encode<S>(&self, spectrum: &S) -> Result<SpectrumVector>
    where
        S: SpectrumAlloc,
        S::MutationError: Into<Error>,
    {
        let top_peaks = filtered_top_peaks(
            spectrum,
            self.config.max_peaks,
            self.config.min_mz,
            self.config.max_mz,
        )
        .map_err(Into::into)?;

        let max_intensity = top_peaks
            .peaks
            .iter()
            .map(|(_, intensity)| *intensity)
            .fold(0.0_f64, f64::max);

        let mut values = vec![0.0; self.vector_width()];
        for (index, (mz, intensity)) in top_peaks.peaks.iter().copied().enumerate() {
            let mz_value = (mz / self.config.max_mz).clamp(0.0, 1.0) as f32;
            let intensity_value = if max_intensity > 0.0 {
                (intensity / max_intensity)
                    .clamp(0.0, 1.0)
                    .powf(self.config.intensity_power) as f32
            } else {
                0.0
            };
            let offset = index * 2;
            values[offset] = mz_value;
            values[offset + 1] = intensity_value;
        }

        Ok(SpectrumVector {
            values,
            retained_peaks: top_peaks.peaks.len(),
            candidate_peaks: top_peaks.candidate_peaks,
        })
    }

    /// Decodes a vector into a normalized [`GenericSpectrum`].
    ///
    /// Intensities remain relative because the autoencoder reconstructs cleaned,
    /// normalized spectra.
    pub fn decode(&self, values: &[f32], precursor_mz: f64) -> Result<GenericSpectrum<f64>> {
        decode_normalized_peak_pairs(
            values,
            self.vector_width(),
            self.config.max_mz,
            precursor_mz,
        )
    }
}

#[cfg(test)]
mod tests {
    use mass_spectrometry::prelude::{GenericSpectrum, SpectrumMut};

    use super::*;

    #[test]
    fn vectorizer_keeps_top_sixty_by_default() {
        let vectorizer = SpectrumVectorizer::default();
        assert_eq!(vectorizer.vector_width(), 120);
    }

    #[test]
    fn vectorizer_orders_by_intensity_and_pads() -> Result<()> {
        let mut spectrum =
            GenericSpectrum::try_with_capacity(500.0, 3).expect("valid synthetic spectrum");
        spectrum.add_peak(100.0, 10.0).expect("valid peak");
        spectrum.add_peak(200.0, 30.0).expect("valid peak");
        spectrum.add_peak(300.0, 20.0).expect("valid peak");

        let vectorizer = SpectrumVectorizer::new(SpectrumVectorizerConfig {
            max_peaks: 2,
            min_mz: 0.0,
            max_mz: 1_000.0,
            intensity_power: 1.0,
        });
        let vector = vectorizer.encode(&spectrum)?;

        assert_eq!(vector.retained_peaks, 2);
        assert_eq!(vector.candidate_peaks, 3);
        assert_eq!(vector.values, vec![0.2, 1.0, 0.3, 20.0 / 30.0]);
        Ok(())
    }

    #[test]
    fn decode_rejects_wrong_width() {
        let vectorizer = SpectrumVectorizer::new(SpectrumVectorizerConfig {
            max_peaks: 1,
            ..SpectrumVectorizerConfig::default()
        });
        let error = vectorizer
            .decode(&[1.0], 100.0)
            .expect_err("wrong-width vector should fail");
        assert!(matches!(error, Error::InvalidVectorLength { .. }));
    }
}
