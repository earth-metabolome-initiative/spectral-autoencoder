//! Reconstruction metrics for dense vectors and decoded spectra.

use mass_spectrometry::prelude::{
    LinearCosine, LinearEntropy, ScalarSimilarity, Spectrum, SpectrumFloat,
};
use serde::{Deserialize, Serialize};

use crate::error::Result;

/// Dense vector reconstruction metrics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DenseReconstructionMetrics {
    /// Mean squared error.
    pub mse: f64,
    /// Cosine similarity over the dense representation.
    pub cosine: f64,
}

impl DenseReconstructionMetrics {
    /// Computes dense metrics from two equally sized vectors.
    #[must_use]
    pub fn from_vectors(reference: &[f32], reconstruction: &[f32]) -> Option<Self> {
        if reference.len() != reconstruction.len() || reference.is_empty() {
            return None;
        }

        let mut squared_error = 0.0;
        let mut dot = 0.0;
        let mut reference_norm = 0.0;
        let mut reconstruction_norm = 0.0;
        for (&left, &right) in reference.iter().zip(reconstruction) {
            let left = f64::from(left);
            let right = f64::from(right);
            let delta = left - right;
            squared_error += delta * delta;
            dot += left * right;
            reference_norm += left * left;
            reconstruction_norm += right * right;
        }

        let cosine = if reference_norm > 0.0 && reconstruction_norm > 0.0 {
            dot / (reference_norm.sqrt() * reconstruction_norm.sqrt())
        } else {
            0.0
        };

        Some(Self {
            mse: squared_error / reference.len() as f64,
            cosine,
        })
    }
}

/// Spectral similarity metric configuration.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SpectralMetricConfig {
    /// Peak m/z tolerance.
    pub mz_tolerance: f64,
    /// m/z exponent for cosine scoring.
    pub cosine_mz_power: f64,
    /// intensity exponent for cosine scoring.
    pub cosine_intensity_power: f64,
    /// m/z exponent for entropy scoring.
    pub entropy_mz_power: f64,
    /// intensity exponent for entropy scoring.
    pub entropy_intensity_power: f64,
    /// Whether to use dynamically weighted entropy similarity.
    pub weighted_entropy: bool,
}

impl Default for SpectralMetricConfig {
    fn default() -> Self {
        Self {
            mz_tolerance: 0.02,
            cosine_mz_power: 0.0,
            cosine_intensity_power: 0.5,
            entropy_mz_power: 0.0,
            entropy_intensity_power: 1.0,
            weighted_entropy: true,
        }
    }
}

impl SpectralMetricConfig {
    /// Starts a fluent builder seeded with [`Self::default`].
    #[must_use]
    pub fn builder() -> SpectralMetricConfigBuilder {
        SpectralMetricConfigBuilder::default()
    }
}

/// Fluent builder for [`SpectralMetricConfig`].
#[derive(Debug, Clone, Copy, Default)]
pub struct SpectralMetricConfigBuilder {
    config: SpectralMetricConfig,
}

impl SpectralMetricConfigBuilder {
    /// Creates a builder seeded with [`SpectralMetricConfig::default`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the peak m/z tolerance.
    #[inline]
    #[must_use]
    pub fn with_mz_tolerance(mut self, value: f64) -> Self {
        self.config.mz_tolerance = value;
        self
    }

    /// Sets the cosine m/z exponent.
    #[inline]
    #[must_use]
    pub fn with_cosine_mz_power(mut self, value: f64) -> Self {
        self.config.cosine_mz_power = value;
        self
    }

    /// Sets the cosine intensity exponent.
    #[inline]
    #[must_use]
    pub fn with_cosine_intensity_power(mut self, value: f64) -> Self {
        self.config.cosine_intensity_power = value;
        self
    }

    /// Sets the entropy m/z exponent.
    #[inline]
    #[must_use]
    pub fn with_entropy_mz_power(mut self, value: f64) -> Self {
        self.config.entropy_mz_power = value;
        self
    }

    /// Sets the entropy intensity exponent.
    #[inline]
    #[must_use]
    pub fn with_entropy_intensity_power(mut self, value: f64) -> Self {
        self.config.entropy_intensity_power = value;
        self
    }

    /// Toggles the dynamically weighted entropy similarity.
    #[inline]
    #[must_use]
    pub fn with_weighted_entropy(mut self, value: bool) -> Self {
        self.config.weighted_entropy = value;
        self
    }

    /// Returns the configured [`SpectralMetricConfig`].
    #[inline]
    #[must_use]
    pub fn build(self) -> SpectralMetricConfig {
        self.config
    }
}

/// Domain-level spectral metrics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpectralMetrics {
    /// Cosine similarity score.
    pub cosine: f64,
    /// Number of cosine matched peaks.
    pub cosine_matches: usize,
    /// Entropy similarity score.
    pub entropy: f64,
    /// Number of entropy matched peaks.
    pub entropy_matches: usize,
}

impl SpectralMetrics {
    /// Computes spectral similarity metrics between two spectra.
    ///
    /// Cosine is evaluated with the linear-time scorer from
    /// `mass-spectrometry-traits`, so inputs must satisfy that scorer's peak
    /// spacing precondition for the configured tolerance.
    pub fn compare<Left, Right>(
        config: SpectralMetricConfig,
        reference: &Left,
        reconstruction: &Right,
    ) -> Result<Self>
    where
        Left: Spectrum,
        Right: Spectrum,
        Left::Precision: SpectrumFloat,
        Right::Precision: SpectrumFloat,
    {
        let cosine = LinearCosine::new(
            config.cosine_mz_power,
            config.cosine_intensity_power,
            config.mz_tolerance,
        )?;
        let entropy = LinearEntropy::new(
            config.entropy_mz_power,
            config.entropy_intensity_power,
            config.mz_tolerance,
            config.weighted_entropy,
        )?;

        let (cosine, cosine_matches) = cosine.similarity(reference, reconstruction)?;
        let (entropy, entropy_matches) = entropy.similarity(reference, reconstruction)?;

        Ok(Self {
            cosine,
            cosine_matches,
            entropy,
            entropy_matches,
        })
    }
}

#[cfg(test)]
mod tests {
    use mass_spectrometry::prelude::{GenericSpectrum, SpectrumMut};

    use super::*;

    #[test]
    fn dense_metrics_match_identity() {
        let metrics =
            DenseReconstructionMetrics::from_vectors(&[1.0, 2.0], &[1.0, 2.0]).expect("same width");
        assert_eq!(metrics.mse, 0.0);
        assert!((metrics.cosine - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn spectral_metrics_match_identity() {
        let mut spectrum =
            GenericSpectrum::try_with_capacity(250.0, 2).expect("valid synthetic spectrum");
        spectrum.add_peak(100.0, 1.0).expect("valid peak");
        spectrum.add_peak(150.0, 2.0).expect("valid peak");

        let metrics =
            SpectralMetrics::compare(SpectralMetricConfig::default(), &spectrum, &spectrum)
                .expect("identity spectra should compare");
        assert!((metrics.cosine - 1.0).abs() < 1.0e-12);
        assert!((metrics.entropy - 1.0).abs() < 1.0e-12);
    }
}
