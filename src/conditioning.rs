//! Precursor metadata conditioning for encoder inputs and decoder targets.

use mascot_rs::prelude::MascotGenericFormat;
use mass_spectrometry::prelude::{Spectrum, SpectrumFloat};
use serde::{Deserialize, Serialize};

/// Configuration for the precursor conditioning vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConditioningConfig {
    /// Scale divisor for precursor m/z.
    pub precursor_mz_scale: f64,
}

impl Default for ConditioningConfig {
    fn default() -> Self {
        Self {
            precursor_mz_scale: 2_000.0,
        }
    }
}

impl ConditioningConfig {
    /// Starts a fluent builder seeded with [`Self::default`].
    #[must_use]
    pub fn builder() -> ConditioningConfigBuilder {
        ConditioningConfigBuilder::default()
    }

    /// Returns the vector width for this configuration.
    #[must_use]
    pub const fn vector_width(&self) -> usize {
        2
    }

    /// Returns the precursor m/z scale used by the condition vector.
    #[must_use]
    pub const fn precursor_mz_scale(&self) -> f64 {
        self.precursor_mz_scale
    }
}

/// Fluent builder for [`ConditioningConfig`].
#[derive(Debug, Clone, Default)]
pub struct ConditioningConfigBuilder {
    config: ConditioningConfig,
}

impl ConditioningConfigBuilder {
    /// Creates a builder seeded with [`ConditioningConfig::default`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the precursor m/z scale divisor.
    #[inline]
    #[must_use]
    pub fn with_precursor_mz_scale(mut self, value: f64) -> Self {
        self.config.precursor_mz_scale = value;
        self
    }

    /// Returns the configured [`ConditioningConfig`].
    #[inline]
    #[must_use]
    pub fn build(self) -> ConditioningConfig {
        self.config
    }
}

/// Encodes precursor metadata into a stable numeric vector.
#[derive(Debug, Clone)]
pub struct ConditioningEncoder {
    config: ConditioningConfig,
}

impl Default for ConditioningEncoder {
    fn default() -> Self {
        Self::new(ConditioningConfig::default())
    }
}

impl ConditioningEncoder {
    /// Creates a metadata encoder.
    #[must_use]
    pub const fn new(config: ConditioningConfig) -> Self {
        Self { config }
    }

    /// Returns the active configuration.
    #[must_use]
    pub const fn config(&self) -> &ConditioningConfig {
        &self.config
    }

    /// Returns the vector width for this encoder.
    #[must_use]
    pub const fn vector_width(&self) -> usize {
        self.config.vector_width()
    }

    /// Returns an all-unknown vector for decoder calls without metadata.
    #[must_use]
    pub fn unknown(&self) -> Vec<f32> {
        vec![0.0, 0.0]
    }

    /// Encodes metadata available on an MGF record.
    pub fn encode<P>(&self, record: &MascotGenericFormat<P>) -> Vec<f32>
    where
        P: SpectrumFloat,
    {
        self.encode_precursor_mz(Some(record.precursor_mz().to_f64()))
    }

    /// Encodes a precursor m/z directly. `None` (or a non-finite / non-positive
    /// value) yields the unknown-precursor encoding.
    #[must_use]
    pub fn encode_precursor_mz(&self, precursor_mz: Option<f64>) -> Vec<f32> {
        match precursor_mz {
            Some(value) if value.is_finite() && value > 0.0 => vec![
                (value / self.config.precursor_mz_scale).clamp(0.0, 1.0) as f32,
                1.0,
            ],
            _ => self.unknown(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_width_is_stable() {
        let encoder = ConditioningEncoder::default();
        assert_eq!(encoder.vector_width(), 2);
        assert_eq!(encoder.unknown().len(), 2);
    }

    #[test]
    fn unknown_has_absent_precursor() {
        let values = ConditioningEncoder::default().unknown();
        assert_eq!(values, [0.0, 0.0]);
    }
}
