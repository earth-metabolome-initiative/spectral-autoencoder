//! Optional metadata conditioning for encoder and decoder inputs.

use mascot_rs::prelude::{Instrument, IonMode, MascotGenericFormat};
use mass_spectrometry::prelude::{Spectrum, SpectrumFloat};
use serde::{Deserialize, Serialize};

/// Configuration for the fixed metadata conditioning vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConditioningConfig {
    /// Include a constant value so the vector is never empty.
    pub include_constant: bool,
    /// Include precursor m/z and a presence flag.
    pub include_precursor_mz: bool,
    /// Scale divisor for precursor m/z.
    pub precursor_mz_scale: f64,
    /// Include charge and a presence flag.
    pub include_charge: bool,
    /// Scale divisor for absolute charge.
    pub charge_scale: f64,
    /// Include ion mode as unknown/positive/negative one-hot values.
    pub include_ion_mode: bool,
    /// Include normalized instrument class one-hot values.
    pub include_instrument: bool,
}

impl Default for ConditioningConfig {
    fn default() -> Self {
        Self {
            include_constant: true,
            include_precursor_mz: true,
            precursor_mz_scale: 2_000.0,
            include_charge: true,
            charge_scale: 5.0,
            include_ion_mode: true,
            include_instrument: true,
        }
    }
}

impl ConditioningConfig {
    /// Returns the vector width for this configuration.
    #[must_use]
    pub const fn vector_width(&self) -> usize {
        let mut width = 0;
        if self.include_constant {
            width += 1;
        }
        if self.include_precursor_mz {
            width += 2;
        }
        if self.include_charge {
            width += 2;
        }
        if self.include_ion_mode {
            width += 3;
        }
        if self.include_instrument {
            width += 8;
        }
        width
    }
}

/// Encodes optional MGF metadata into a stable numeric vector.
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
        let mut values = Vec::with_capacity(self.vector_width());
        if self.config.include_constant {
            values.push(1.0);
        }
        if self.config.include_precursor_mz {
            values.extend([0.0, 0.0]);
        }
        if self.config.include_charge {
            values.extend([0.0, 0.0]);
        }
        if self.config.include_ion_mode {
            values.extend([1.0, 0.0, 0.0]);
        }
        if self.config.include_instrument {
            values.extend([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        }
        values
    }

    /// Encodes metadata available on an MGF record.
    pub fn encode<P>(&self, record: &MascotGenericFormat<P>) -> Vec<f32>
    where
        P: SpectrumFloat,
    {
        let mut values = Vec::with_capacity(self.vector_width());
        if self.config.include_constant {
            values.push(1.0);
        }
        if self.config.include_precursor_mz {
            let precursor = record.precursor_mz().to_f64();
            if precursor.is_finite() && precursor > 0.0 {
                values.push((precursor / self.config.precursor_mz_scale).clamp(0.0, 1.0) as f32);
                values.push(1.0);
            } else {
                values.extend([0.0, 0.0]);
            }
        }
        if self.config.include_charge {
            if let Some(charge) = record.charge().filter(|charge| *charge != 0) {
                let scaled = f32::from(charge) / self.config.charge_scale as f32;
                values.push(scaled.clamp(-1.0, 1.0));
                values.push(1.0);
            } else {
                values.extend([0.0, 0.0]);
            }
        }
        if self.config.include_ion_mode {
            values.extend(ion_mode_vector(record.ion_mode()));
        }
        if self.config.include_instrument {
            values.extend(instrument_vector(record.source_instrument()));
        }
        values
    }
}

fn ion_mode_vector(ion_mode: Option<IonMode>) -> [f32; 3] {
    match ion_mode {
        None => [1.0, 0.0, 0.0],
        Some(IonMode::Positive) => [0.0, 1.0, 0.0],
        Some(IonMode::Negative) => [0.0, 0.0, 1.0],
    }
}

fn instrument_vector(instrument: Option<Instrument>) -> [f32; 8] {
    match instrument {
        None => [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        Some(Instrument::Orbitrap) => [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        Some(Instrument::TimeOfFlight) => [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        Some(Instrument::Quadrupole) => [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        Some(Instrument::IonTrap) => [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        Some(Instrument::FourierTransform) => [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        Some(Instrument::MagneticSector) => [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        Some(Instrument::Other) => [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_width_is_stable() {
        let encoder = ConditioningEncoder::default();
        assert_eq!(encoder.vector_width(), 16);
        assert_eq!(encoder.unknown().len(), 16);
    }

    #[test]
    fn unknown_has_explicit_unknown_buckets() {
        let values = ConditioningEncoder::default().unknown();
        assert_eq!(values[0], 1.0);
        assert_eq!(&values[5..8], &[1.0, 0.0, 0.0]);
        assert_eq!(&values[8..16], &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }
}
