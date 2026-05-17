//! Top-N peak tokenization for set/transformer-style spectrum models.

use std::f64::consts::TAU;

use mass_spectrometry::prelude::{GenericSpectrum, SpectrumAlloc};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::peaks::{decode_normalized_peak_pairs, filtered_top_peaks};

/// Tokenized representation of one cleaned spectrum.
#[derive(Debug, Clone, PartialEq)]
pub struct SpectrumTokens {
    /// Flattened `[max_peaks, feature_width]` token feature matrix.
    pub features: Vec<f32>,
    /// Flattened `[max_peaks, 2]` normalized `(m/z, intensity)` target pairs.
    pub target_pairs: Vec<f32>,
    /// Float mask with `1.0` for real peaks and `0.0` for padding.
    pub peak_mask: Vec<f32>,
    /// Padding mask with `true` for padding positions.
    pub padding_mask: Vec<bool>,
    /// Number of retained non-padding peaks.
    pub retained_peaks: usize,
    /// Original number of peaks after finite/range filtering.
    pub candidate_peaks: usize,
}

/// Configuration for DreaMS-style top-N peak tokens.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectrumTokenizerConfig {
    /// Maximum number of peaks retained per spectrum.
    pub max_peaks: usize,
    /// Minimum m/z retained.
    pub min_mz: f64,
    /// Maximum m/z represented as `1.0`.
    pub max_mz: f64,
    /// Power applied after max-intensity normalization.
    pub intensity_power: f64,
    /// Number of sine/cosine m/z frequency pairs added to each token.
    pub mz_fourier_frequencies: usize,
}

impl Default for SpectrumTokenizerConfig {
    fn default() -> Self {
        Self {
            max_peaks: 60,
            min_mz: 0.0,
            max_mz: 2_000.0,
            intensity_power: 0.5,
            mz_fourier_frequencies: 8,
        }
    }
}

impl SpectrumTokenizerConfig {
    /// Starts a fluent builder seeded with [`Self::default`].
    #[must_use]
    pub fn builder() -> SpectrumTokenizerConfigBuilder {
        SpectrumTokenizerConfigBuilder::default()
    }

    /// Returns the number of features per peak token.
    #[must_use]
    pub const fn feature_width(&self) -> usize {
        3 + self.mz_fourier_frequencies * 2
    }

    /// Returns the flattened token-feature vector width.
    #[must_use]
    pub const fn flattened_feature_width(&self) -> usize {
        self.max_peaks * self.feature_width()
    }

    /// Returns the flattened reconstruction target width.
    #[must_use]
    pub const fn target_width(&self) -> usize {
        self.max_peaks * 2
    }
}

/// Fluent builder for [`SpectrumTokenizerConfig`].
#[derive(Debug, Clone, Default)]
pub struct SpectrumTokenizerConfigBuilder {
    config: SpectrumTokenizerConfig,
}

impl SpectrumTokenizerConfigBuilder {
    /// Creates a builder seeded with [`SpectrumTokenizerConfig::default`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the maximum number of peaks retained per spectrum.
    #[inline]
    #[must_use]
    pub fn with_max_peaks(mut self, value: usize) -> Self {
        self.config.max_peaks = value;
        self
    }

    /// Sets the minimum m/z retained.
    #[inline]
    #[must_use]
    pub fn with_min_mz(mut self, value: f64) -> Self {
        self.config.min_mz = value;
        self
    }

    /// Sets the maximum m/z represented as `1.0`.
    #[inline]
    #[must_use]
    pub fn with_max_mz(mut self, value: f64) -> Self {
        self.config.max_mz = value;
        self
    }

    /// Sets the post-normalization intensity power.
    #[inline]
    #[must_use]
    pub fn with_intensity_power(mut self, value: f64) -> Self {
        self.config.intensity_power = value;
        self
    }

    /// Sets the number of sine/cosine m/z frequency pairs per token.
    #[inline]
    #[must_use]
    pub fn with_mz_fourier_frequencies(mut self, value: usize) -> Self {
        self.config.mz_fourier_frequencies = value;
        self
    }

    /// Returns the configured [`SpectrumTokenizerConfig`].
    #[inline]
    #[must_use]
    pub fn build(self) -> SpectrumTokenizerConfig {
        self.config
    }
}

/// Converts spectra into padded peak-token matrices.
#[derive(Debug, Clone)]
pub struct SpectrumTokenizer {
    config: SpectrumTokenizerConfig,
}

impl Default for SpectrumTokenizer {
    fn default() -> Self {
        Self::new(SpectrumTokenizerConfig::default())
    }
}

impl SpectrumTokenizer {
    /// Creates a new tokenizer.
    #[must_use]
    pub const fn new(config: SpectrumTokenizerConfig) -> Self {
        Self { config }
    }

    /// Returns the active configuration.
    #[must_use]
    pub const fn config(&self) -> &SpectrumTokenizerConfig {
        &self.config
    }

    /// Returns the number of features per peak token.
    #[must_use]
    pub const fn feature_width(&self) -> usize {
        self.config.feature_width()
    }

    /// Returns the flattened token-feature vector width.
    #[must_use]
    pub const fn flattened_feature_width(&self) -> usize {
        self.config.flattened_feature_width()
    }

    /// Returns the flattened reconstruction target width.
    #[must_use]
    pub const fn target_width(&self) -> usize {
        self.config.target_width()
    }

    /// Encodes a spectrum as padded top-intensity peak tokens.
    pub fn encode<S>(&self, spectrum: &S) -> Result<SpectrumTokens>
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

        let feature_width = self.feature_width();
        let mut features = vec![0.0; self.flattened_feature_width()];
        let mut target_pairs = vec![0.0; self.target_width()];
        let mut peak_mask = vec![0.0; self.config.max_peaks];
        let mut padding_mask = vec![true; self.config.max_peaks];

        for (index, (mz, intensity)) in top_peaks.peaks.iter().copied().enumerate() {
            let mz_value = (mz / self.config.max_mz).clamp(0.0, 1.0);
            let intensity_value = if max_intensity > 0.0 {
                (intensity / max_intensity)
                    .clamp(0.0, 1.0)
                    .powf(self.config.intensity_power)
            } else {
                0.0
            };

            let feature_offset = index * feature_width;
            features[feature_offset] = mz_value as f32;
            features[feature_offset + 1] = intensity_value as f32;
            features[feature_offset + 2] = 1.0;
            self.write_fourier_features(&mut features, feature_offset + 3, mz_value);

            let target_offset = index * 2;
            target_pairs[target_offset] = mz_value as f32;
            target_pairs[target_offset + 1] = intensity_value as f32;
            peak_mask[index] = 1.0;
            padding_mask[index] = false;
        }

        Ok(SpectrumTokens {
            features,
            target_pairs,
            peak_mask,
            padding_mask,
            retained_peaks: top_peaks.peaks.len(),
            candidate_peaks: top_peaks.candidate_peaks,
        })
    }

    /// Decodes normalized `(m/z, intensity)` pairs into a spectrum.
    pub fn decode_pairs(
        &self,
        target_pairs: &[f32],
        precursor_mz: f64,
    ) -> Result<GenericSpectrum<f64>> {
        decode_normalized_peak_pairs(
            target_pairs,
            self.target_width(),
            self.config.max_mz,
            precursor_mz,
        )
    }

    fn write_fourier_features(&self, features: &mut [f32], offset: usize, mz_value: f64) {
        let mut frequency = 1.0;
        for index in 0..self.config.mz_fourier_frequencies {
            let angle = TAU * frequency * mz_value;
            features[offset + index * 2] = angle.sin() as f32;
            features[offset + index * 2 + 1] = angle.cos() as f32;
            frequency *= 2.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use mass_spectrometry::prelude::{GenericSpectrum, SpectrumMut};

    use super::*;

    #[test]
    fn tokenizer_default_keeps_sixty_peaks() {
        let tokenizer = SpectrumTokenizer::default();
        assert_eq!(tokenizer.config().max_peaks, 60);
        assert_eq!(tokenizer.feature_width(), 19);
        assert_eq!(tokenizer.target_width(), 120);
    }

    #[test]
    fn tokenizer_adds_masks_and_fourier_features() -> Result<()> {
        let mut spectrum =
            GenericSpectrum::try_with_capacity(500.0, 3).expect("valid synthetic spectrum");
        spectrum.add_peak(100.0, 10.0).expect("valid peak");
        spectrum.add_peak(200.0, 30.0).expect("valid peak");
        spectrum.add_peak(300.0, 20.0).expect("valid peak");

        let tokenizer = SpectrumTokenizer::new(SpectrumTokenizerConfig {
            max_peaks: 2,
            min_mz: 0.0,
            max_mz: 1_000.0,
            intensity_power: 1.0,
            mz_fourier_frequencies: 1,
        });
        let tokens = tokenizer.encode(&spectrum)?;

        assert_eq!(tokens.retained_peaks, 2);
        assert_eq!(tokens.candidate_peaks, 3);
        assert_eq!(tokens.features.len(), 10);
        assert_eq!(tokens.target_pairs, vec![0.2, 1.0, 0.3, 20.0 / 30.0]);
        assert_eq!(tokens.peak_mask, vec![1.0, 1.0]);
        assert_eq!(tokens.padding_mask, vec![false, false]);
        assert!((tokens.features[3] - (TAU * 0.2).sin() as f32).abs() < 1.0e-6);
        assert!((tokens.features[4] - (TAU * 0.2).cos() as f32).abs() < 1.0e-6);
        Ok(())
    }

    #[test]
    fn tokenizer_pads_short_spectra() -> Result<()> {
        let mut spectrum =
            GenericSpectrum::try_with_capacity(500.0, 1).expect("valid synthetic spectrum");
        spectrum.add_peak(100.0, 10.0).expect("valid peak");

        let tokenizer = SpectrumTokenizer::new(SpectrumTokenizerConfig {
            max_peaks: 2,
            mz_fourier_frequencies: 0,
            ..SpectrumTokenizerConfig::default()
        });
        let tokens = tokenizer.encode(&spectrum)?;

        assert_eq!(tokens.peak_mask, vec![1.0, 0.0]);
        assert_eq!(tokens.padding_mask, vec![false, true]);
        assert_eq!(&tokens.features[3..6], &[0.0, 0.0, 0.0]);
        Ok(())
    }
}
