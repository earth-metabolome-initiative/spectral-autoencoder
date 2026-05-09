//! Training-time spectral input augmentation.

use serde::{Deserialize, Serialize};
#[cfg(feature = "std")]
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "std")]
use burn::{data::dataloader::batcher::Batcher, prelude::*};

#[cfg(feature = "std")]
use crate::batch::{
    AutoencoderBatch, AutoencoderBatchLayout, AutoencoderBatchParts, AutoencoderSample,
    TokenizedAutoencoderBatch, TokenizedAutoencoderBatchLayout, TokenizedAutoencoderBatchParts,
    TokenizedAutoencoderSample, autoencoder_batch_from_parts,
    tokenized_autoencoder_batch_from_parts,
};

#[cfg(feature = "std")]
type AugmentedVector = (Vec<f32>, Vec<f32>, Vec<f32>);
#[cfg(feature = "std")]
type AugmentedTokenFeatures = (Vec<f32>, Vec<f32>, Vec<bool>, Vec<f32>, Vec<f32>);

/// Spectral input augmentation configuration.
///
/// Augmentations are applied only to model inputs. Reconstruction targets remain
/// the cleaned spectra, so these settings implement a denoising objective.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SpectrumAugmentationConfig {
    /// Probability of replacing a real peak m/z input with a mask value.
    #[serde(default)]
    pub mz_mask_probability: f32,
    /// Probability of removing a real peak from the input while keeping it in the target.
    #[serde(default)]
    pub peak_dropout_probability: f32,
    /// Per-spectrum signed m/z shift range in normalized m/z units.
    #[serde(default)]
    pub mz_shift_range: f32,
    /// Per-peak signed m/z jitter range in normalized m/z units.
    #[serde(default)]
    pub mz_jitter_range: f32,
    /// Multiplicative intensity jitter range around `1.0`.
    #[serde(default)]
    pub intensity_jitter_fraction: f32,
    /// Probability of atomically masking the precursor condition pair in the encoder input.
    #[serde(default)]
    pub precursor_mask_probability: f32,
    /// Probability of filling each originally empty peak slot with a synthetic intruder peak.
    #[serde(default)]
    pub intruder_peak_probability: f32,
}

impl Default for SpectrumAugmentationConfig {
    fn default() -> Self {
        Self {
            mz_mask_probability: 0.0,
            peak_dropout_probability: 0.0,
            mz_shift_range: 0.0,
            mz_jitter_range: 0.0,
            intensity_jitter_fraction: 0.0,
            precursor_mask_probability: 0.0,
            intruder_peak_probability: 0.0,
        }
    }
}

impl SpectrumAugmentationConfig {
    /// Conservative starting point inspired by DreaMS-style masked peak modeling.
    #[must_use]
    pub const fn masked_mz_pretraining() -> Self {
        Self {
            mz_mask_probability: 0.30,
            peak_dropout_probability: 0.0,
            mz_shift_range: 0.0,
            mz_jitter_range: 0.0,
            intensity_jitter_fraction: 0.05,
            precursor_mask_probability: 0.30,
            intruder_peak_probability: 0.02,
        }
    }

    /// Returns `true` when all augmentation knobs are disabled.
    #[must_use]
    pub fn is_disabled(&self) -> bool {
        self.mz_mask_probability <= 0.0
            && self.peak_dropout_probability <= 0.0
            && self.mz_shift_range <= 0.0
            && self.mz_jitter_range <= 0.0
            && self.intensity_jitter_fraction <= 0.0
            && self.precursor_mask_probability <= 0.0
            && self.intruder_peak_probability <= 0.0
    }
}

/// Deterministic spectral augmenter.
#[derive(Debug, Clone, Copy)]
pub struct SpectrumAugmenter {
    config: SpectrumAugmentationConfig,
}

impl Default for SpectrumAugmenter {
    fn default() -> Self {
        Self::new(SpectrumAugmentationConfig::default())
    }
}

impl SpectrumAugmenter {
    /// Creates a new augmenter.
    #[must_use]
    pub const fn new(config: SpectrumAugmentationConfig) -> Self {
        Self { config }
    }

    /// Returns the active configuration.
    #[must_use]
    pub const fn config(&self) -> &SpectrumAugmentationConfig {
        &self.config
    }

    #[cfg(feature = "std")]
    fn augment_vector(&self, values: &[f32], seed: u64) -> AugmentedVector {
        let peak_count = values.len() / 2;
        if self.config.is_disabled() {
            return (
                values.to_vec(),
                vec![0.0; values.len()],
                vec![0.0; peak_count],
            );
        }

        let mut rng = SmallRng::new(seed);
        let mz_shift = rng.signed(self.config.mz_shift_range.max(0.0));
        let mut output = values.to_vec();
        let mut masked = vec![0.0; values.len()];
        for (pair_index, pair) in output.chunks_exact_mut(2).enumerate() {
            if pair[0] <= 0.0 || pair[1] <= 0.0 {
                continue;
            }
            if rng.event(self.config.peak_dropout_probability) {
                pair[0] = 0.0;
                pair[1] = 0.0;
                masked[pair_index * 2] = 1.0;
                masked[pair_index * 2 + 1] = 1.0;
                continue;
            }
            if rng.event(self.config.mz_mask_probability) {
                pair[0] = 0.0;
                masked[pair_index * 2] = 1.0;
            } else {
                pair[0] = (pair[0] + mz_shift + rng.signed(self.config.mz_jitter_range.max(0.0)))
                    .clamp(0.0, 1.0);
            }
            pair[1] = jitter_intensity(pair[1], self.config.intensity_jitter_fraction, &mut rng);
        }
        let intruder_peak_mask = insert_vector_intruders(
            values,
            &mut output,
            self.config.intruder_peak_probability,
            &mut rng,
        );
        (output, masked, intruder_peak_mask)
    }

    #[cfg(feature = "std")]
    fn augment_token_features(
        &self,
        features: &[f32],
        peak_mask: &[f32],
        seed: u64,
    ) -> AugmentedTokenFeatures {
        if self.config.is_disabled() {
            return (
                features.to_vec(),
                peak_mask.to_vec(),
                peak_mask.iter().map(|mask| *mask <= 0.0).collect(),
                vec![0.0; peak_mask.len()],
                vec![0.0; peak_mask.len()],
            );
        }

        let max_peaks = peak_mask.len();
        let feature_width = features.len() / max_peaks;
        debug_assert!(feature_width >= 3);
        debug_assert_eq!(features.len(), max_peaks * feature_width);

        let mut rng = SmallRng::new(seed);
        let mz_shift = rng.signed(self.config.mz_shift_range.max(0.0));
        let mut output = features.to_vec();
        let mut input_peak_mask = peak_mask.to_vec();
        let mut padding_mask = peak_mask
            .iter()
            .map(|mask| *mask <= 0.0)
            .collect::<Vec<_>>();
        let mut masked_peak_mask = vec![0.0; max_peaks];

        for peak_index in 0..max_peaks {
            if peak_mask[peak_index] <= 0.0 {
                continue;
            }

            let offset = peak_index * feature_width;
            let row = &mut output[offset..offset + feature_width];
            if rng.event(self.config.peak_dropout_probability) {
                row.fill(0.0);
                input_peak_mask[peak_index] = 0.0;
                padding_mask[peak_index] = true;
                masked_peak_mask[peak_index] = 1.0;
                continue;
            }

            if rng.event(self.config.mz_mask_probability) {
                row[0] = 0.0;
                zero_fourier_features(row);
                masked_peak_mask[peak_index] = 1.0;
            } else {
                row[0] = (row[0] + mz_shift + rng.signed(self.config.mz_jitter_range.max(0.0)))
                    .clamp(0.0, 1.0);
                rewrite_fourier_features(row);
            }

            row[1] = jitter_intensity(row[1], self.config.intensity_jitter_fraction, &mut rng);
        }

        let intruder_peak_mask = insert_token_intruders(
            features,
            peak_mask,
            &mut output,
            &mut input_peak_mask,
            &mut padding_mask,
            self.config.intruder_peak_probability,
            &mut rng,
        );

        (
            output,
            input_peak_mask,
            padding_mask,
            masked_peak_mask,
            intruder_peak_mask,
        )
    }

    #[cfg(feature = "std")]
    fn augment_precursor_conditions(&self, conditions: &[f32], seed: u64) -> (Vec<f32>, f32) {
        let probability = self.config.precursor_mask_probability.clamp(0.0, 1.0);
        let precursor_present = conditions.get(1).copied().unwrap_or(0.0) > 0.0;
        if probability <= 0.0 || !precursor_present {
            return (conditions.to_vec(), 0.0);
        }

        let mut rng = SmallRng::new(seed);
        if rng.event(probability) {
            (vec![0.0; conditions.len()], 1.0)
        } else {
            (conditions.to_vec(), 0.0)
        }
    }
}

/// Batcher that corrupts vector inputs and keeps clean vector targets.
#[cfg(feature = "std")]
#[derive(Debug)]
pub struct AugmentingAutoencoderBatcher {
    augmenter: SpectrumAugmenter,
    next_seed: AtomicU64,
}

#[cfg(feature = "std")]
impl AugmentingAutoencoderBatcher {
    /// Creates an augmenting batcher.
    #[must_use]
    pub const fn new(config: SpectrumAugmentationConfig, seed: u64) -> Self {
        Self {
            augmenter: SpectrumAugmenter::new(config),
            next_seed: AtomicU64::new(seed),
        }
    }
}

#[cfg(feature = "std")]
impl<B: Backend> Batcher<B, AutoencoderSample, AutoencoderBatch<B>>
    for AugmentingAutoencoderBatcher
{
    fn batch(&self, items: Vec<AutoencoderSample>, device: &B::Device) -> AutoencoderBatch<B> {
        let batch_seed = self.next_seed.fetch_add(1, Ordering::Relaxed);
        let layout = AutoencoderBatchLayout::from_samples(&items);

        let mut spectra = Vec::with_capacity(layout.spectrum_capacity());
        let mut target_spectra = Vec::with_capacity(layout.spectrum_capacity());
        let mut conditions = Vec::with_capacity(layout.condition_capacity());
        let mut target_conditions = Vec::with_capacity(layout.condition_capacity());
        let mut masked_precursor_mask = Vec::with_capacity(layout.batch_capacity());
        let mut masked_spectra_mask = Vec::with_capacity(layout.spectrum_capacity());
        let mut intruder_peak_mask = Vec::with_capacity(layout.peak_capacity());
        for (index, item) in items.into_iter().enumerate() {
            let seed = mix_seed(batch_seed, index as u64);
            let (augmented_spectrum, masked, intruders) =
                self.augmenter.augment_vector(&item.spectrum, seed);
            spectra.extend(augmented_spectrum);
            target_spectra.extend(item.spectrum);
            masked_spectra_mask.extend(masked);
            intruder_peak_mask.extend(intruders);
            target_conditions.extend_from_slice(&item.conditions);
            let (input_conditions, masked_precursor) = self
                .augmenter
                .augment_precursor_conditions(&item.conditions, seed ^ 0x9e37_79b9_7f4a_7c15);
            conditions.extend(input_conditions);
            masked_precursor_mask.push(masked_precursor);
        }

        autoencoder_batch_from_parts(
            AutoencoderBatchParts {
                layout,
                spectra,
                target_spectra,
                conditions,
                target_conditions,
                masked_precursor_mask,
                masked_spectra_mask,
                intruder_peak_mask,
                similarity_ranking: None,
            },
            device,
        )
    }
}

/// Batcher that corrupts peak-token inputs and keeps clean set targets.
#[cfg(feature = "std")]
#[derive(Debug)]
pub struct AugmentingTokenizedAutoencoderBatcher {
    augmenter: SpectrumAugmenter,
    next_seed: AtomicU64,
}

#[cfg(feature = "std")]
impl AugmentingTokenizedAutoencoderBatcher {
    /// Creates an augmenting tokenized batcher.
    #[must_use]
    pub const fn new(config: SpectrumAugmentationConfig, seed: u64) -> Self {
        Self {
            augmenter: SpectrumAugmenter::new(config),
            next_seed: AtomicU64::new(seed),
        }
    }
}

#[cfg(feature = "std")]
impl<B: Backend> Batcher<B, TokenizedAutoencoderSample, TokenizedAutoencoderBatch<B>>
    for AugmentingTokenizedAutoencoderBatcher
{
    fn batch(
        &self,
        items: Vec<TokenizedAutoencoderSample>,
        device: &B::Device,
    ) -> TokenizedAutoencoderBatch<B> {
        let batch_seed = self.next_seed.fetch_add(1, Ordering::Relaxed);
        let layout = TokenizedAutoencoderBatchLayout::from_samples(&items);

        let mut token_features = Vec::with_capacity(layout.token_feature_capacity());
        let mut target_pairs = Vec::with_capacity(layout.target_capacity());
        let mut peak_mask = Vec::with_capacity(layout.peak_capacity());
        let mut target_peak_mask = Vec::with_capacity(layout.peak_capacity());
        let mut padding_mask = Vec::with_capacity(layout.peak_capacity());
        let mut conditions = Vec::with_capacity(layout.condition_capacity());
        let mut target_conditions = Vec::with_capacity(layout.condition_capacity());
        let mut masked_precursor_mask = Vec::with_capacity(layout.batch_capacity());
        let mut masked_peak_mask = Vec::with_capacity(layout.peak_capacity());
        let mut intruder_peak_mask = Vec::with_capacity(layout.peak_capacity());
        for (index, item) in items.into_iter().enumerate() {
            let seed = mix_seed(batch_seed, index as u64);
            let (input_features, input_peak_mask, input_padding_mask, masked_peaks, intruders) =
                self.augmenter
                    .augment_token_features(&item.token_features, &item.peak_mask, seed);
            token_features.extend(input_features);
            target_pairs.extend(item.target_pairs);
            peak_mask.extend(input_peak_mask);
            target_peak_mask.extend(item.peak_mask);
            padding_mask.extend(input_padding_mask);
            masked_peak_mask.extend(masked_peaks);
            intruder_peak_mask.extend(intruders);
            target_conditions.extend_from_slice(&item.conditions);
            let (input_conditions, masked_precursor) = self
                .augmenter
                .augment_precursor_conditions(&item.conditions, seed ^ 0x9e37_79b9_7f4a_7c15);
            conditions.extend(input_conditions);
            masked_precursor_mask.push(masked_precursor);
        }

        tokenized_autoencoder_batch_from_parts(
            TokenizedAutoencoderBatchParts {
                layout,
                token_features,
                target_pairs,
                peak_mask,
                target_peak_mask,
                padding_mask,
                conditions,
                target_conditions,
                masked_precursor_mask,
                masked_peak_mask,
                intruder_peak_mask,
                similarity_ranking: None,
            },
            device,
        )
    }
}

#[cfg(feature = "std")]
fn insert_vector_intruders(
    original: &[f32],
    output: &mut [f32],
    probability: f32,
    rng: &mut SmallRng,
) -> Vec<f32> {
    let peak_count = original.len() / 2;
    let mut intruder_peak_mask = vec![0.0; peak_count];
    let probability = probability.clamp(0.0, 1.0);
    if probability <= 0.0 {
        return intruder_peak_mask;
    }

    let Some(range) = peak_range_from_vector(original) else {
        return intruder_peak_mask;
    };

    for (peak_index, intruder_label) in intruder_peak_mask.iter_mut().enumerate() {
        let offset = peak_index * 2;
        if original[offset] > 0.0 || original[offset + 1] > 0.0 {
            continue;
        }
        if !rng.event(probability) {
            continue;
        }
        output[offset] = range.random_mz(rng);
        output[offset + 1] = range.random_intensity(rng);
        *intruder_label = 1.0;
    }

    intruder_peak_mask
}

#[cfg(feature = "std")]
fn insert_token_intruders(
    original_features: &[f32],
    original_peak_mask: &[f32],
    output: &mut [f32],
    input_peak_mask: &mut [f32],
    padding_mask: &mut [bool],
    probability: f32,
    rng: &mut SmallRng,
) -> Vec<f32> {
    let max_peaks = original_peak_mask.len();
    let feature_width = original_features.len() / max_peaks;
    let mut intruder_peak_mask = vec![0.0; max_peaks];
    let probability = probability.clamp(0.0, 1.0);
    if probability <= 0.0 {
        return intruder_peak_mask;
    }

    let Some(range) = peak_range_from_tokens(original_features, original_peak_mask) else {
        return intruder_peak_mask;
    };

    for (peak_index, intruder_label) in intruder_peak_mask.iter_mut().enumerate() {
        if original_peak_mask[peak_index] > 0.0 || !rng.event(probability) {
            continue;
        }
        let offset = peak_index * feature_width;
        let row = &mut output[offset..offset + feature_width];
        row.fill(0.0);
        row[0] = range.random_mz(rng);
        row[1] = range.random_intensity(rng);
        row[2] = 1.0;
        rewrite_fourier_features(row);
        input_peak_mask[peak_index] = 1.0;
        padding_mask[peak_index] = false;
        *intruder_label = 1.0;
    }

    intruder_peak_mask
}

#[cfg(feature = "std")]
fn peak_range_from_vector(values: &[f32]) -> Option<PeakRange> {
    values
        .chunks_exact(2)
        .filter_map(|pair| valid_peak(pair[0], pair[1]))
        .fold(None, update_peak_range)
}

#[cfg(feature = "std")]
fn peak_range_from_tokens(features: &[f32], peak_mask: &[f32]) -> Option<PeakRange> {
    let max_peaks = peak_mask.len();
    let feature_width = features.len() / max_peaks;
    peak_mask
        .iter()
        .enumerate()
        .filter(|(_, mask)| **mask > 0.0)
        .filter_map(|(peak_index, _)| {
            let offset = peak_index * feature_width;
            valid_peak(features[offset], features[offset + 1])
        })
        .fold(None, update_peak_range)
}

#[cfg(feature = "std")]
fn valid_peak(mz: f32, intensity: f32) -> Option<(f32, f32)> {
    (mz.is_finite() && intensity.is_finite() && mz > 0.0 && intensity > 0.0)
        .then_some((mz, intensity))
}

#[cfg(feature = "std")]
fn update_peak_range(range: Option<PeakRange>, peak: (f32, f32)) -> Option<PeakRange> {
    let (mz, intensity) = peak;
    Some(match range {
        Some(range) => PeakRange {
            min_mz: range.min_mz.min(mz),
            max_mz: range.max_mz.max(mz),
            min_intensity: range.min_intensity.min(intensity),
            max_intensity: range.max_intensity.max(intensity),
        },
        None => PeakRange {
            min_mz: mz,
            max_mz: mz,
            min_intensity: intensity,
            max_intensity: intensity,
        },
    })
}

#[derive(Debug, Clone, Copy)]
#[cfg(feature = "std")]
struct PeakRange {
    min_mz: f32,
    max_mz: f32,
    min_intensity: f32,
    max_intensity: f32,
}

#[cfg(feature = "std")]
impl PeakRange {
    fn random_mz(self, rng: &mut SmallRng) -> f32 {
        sample_range(self.min_mz, self.max_mz, rng)
    }

    fn random_intensity(self, rng: &mut SmallRng) -> f32 {
        sample_range(self.min_intensity, self.max_intensity, rng)
    }
}

#[cfg(feature = "std")]
fn sample_range(min: f32, max: f32, rng: &mut SmallRng) -> f32 {
    if max <= min {
        min
    } else {
        min + rng.unit() * (max - min)
    }
    .clamp(0.0, 1.0)
}

#[cfg(feature = "std")]
fn jitter_intensity(value: f32, fraction: f32, rng: &mut SmallRng) -> f32 {
    let fraction = fraction.max(0.0);
    if fraction <= 0.0 {
        return value;
    }
    (value * (1.0 + rng.signed(fraction))).clamp(0.0, 1.0)
}

#[cfg(feature = "std")]
fn zero_fourier_features(row: &mut [f32]) {
    for value in &mut row[3..] {
        *value = 0.0;
    }
}

#[cfg(feature = "std")]
fn rewrite_fourier_features(row: &mut [f32]) {
    let feature_count = row.len().saturating_sub(3);
    let fourier_pairs = feature_count / 2;
    let mut frequency = 1.0_f32;
    for index in 0..fourier_pairs {
        let angle = std::f32::consts::TAU * frequency * row[0];
        row[3 + index * 2] = angle.sin();
        row[3 + index * 2 + 1] = angle.cos();
        frequency *= 2.0;
    }
}

#[cfg(feature = "std")]
fn mix_seed(seed: u64, index: u64) -> u64 {
    seed ^ index.wrapping_mul(0x517c_c1b7_2722_0a95)
}

#[derive(Debug, Clone, Copy)]
#[cfg(feature = "std")]
struct SmallRng {
    state: u64,
}

#[cfg(feature = "std")]
impl SmallRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn unit(&mut self) -> f32 {
        const SCALE: f32 = 1.0 / ((1u64 << 24) as f32);
        ((self.next_u64() >> 40) as f32) * SCALE
    }

    fn signed(&mut self, range: f32) -> f32 {
        if range <= 0.0 {
            0.0
        } else {
            (self.unit() * 2.0 - 1.0) * range
        }
    }

    fn event(&mut self, probability: f32) -> bool {
        let probability = probability.clamp(0.0, 1.0);
        probability > 0.0 && self.unit() < probability
    }
}

#[cfg(all(test, feature = "std", feature = "ndarray"))]
mod tests {
    use super::*;

    #[test]
    fn vector_augmentation_keeps_clean_targets() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let batcher = AugmentingAutoencoderBatcher::new(
            SpectrumAugmentationConfig {
                peak_dropout_probability: 1.0,
                ..SpectrumAugmentationConfig::default()
            },
            7,
        );
        let batch: AutoencoderBatch<B> = batcher.batch(
            vec![AutoencoderSample {
                spectrum: vec![0.1, 0.8, 0.2, 0.4],
                conditions: vec![1.0],
            }],
            &device,
        );

        assert_eq!(
            batch.spectra.into_data().to_vec::<f32>().expect("input"),
            vec![0.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(
            batch
                .target_spectra
                .into_data()
                .to_vec::<f32>()
                .expect("target"),
            vec![0.1, 0.8, 0.2, 0.4]
        );
        assert_eq!(
            batch
                .masked_spectra_mask
                .into_data()
                .to_vec::<f32>()
                .expect("mask"),
            vec![1.0, 1.0, 1.0, 1.0]
        );
        assert_eq!(
            batch
                .intruder_peak_mask
                .into_data()
                .to_vec::<f32>()
                .expect("intruder mask"),
            vec![0.0, 0.0]
        );
    }

    #[test]
    fn token_augmentation_keeps_clean_targets_and_target_mask() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let batcher = AugmentingTokenizedAutoencoderBatcher::new(
            SpectrumAugmentationConfig {
                peak_dropout_probability: 1.0,
                ..SpectrumAugmentationConfig::default()
            },
            11,
        );
        let batch: TokenizedAutoencoderBatch<B> = batcher.batch(
            vec![TokenizedAutoencoderSample {
                token_features: vec![0.1, 0.8, 1.0, 0.2, 0.4, 1.0],
                target_pairs: vec![0.1, 0.8, 0.2, 0.4],
                peak_mask: vec![1.0, 1.0],
                padding_mask: vec![false, false],
                conditions: vec![1.0],
            }],
            &device,
        );

        assert_eq!(
            batch
                .token_features
                .into_data()
                .to_vec::<f32>()
                .expect("input"),
            vec![0.0; 6]
        );
        assert_eq!(
            batch
                .target_pairs
                .into_data()
                .to_vec::<f32>()
                .expect("target"),
            vec![0.1, 0.8, 0.2, 0.4]
        );
        assert_eq!(
            batch
                .peak_mask
                .into_data()
                .to_vec::<f32>()
                .expect("input mask"),
            vec![0.0, 0.0]
        );
        assert_eq!(
            batch
                .target_peak_mask
                .into_data()
                .to_vec::<f32>()
                .expect("target mask"),
            vec![1.0, 1.0]
        );
        assert_eq!(
            batch
                .masked_peak_mask
                .into_data()
                .to_vec::<f32>()
                .expect("masked peaks"),
            vec![1.0, 1.0]
        );
        assert_eq!(
            batch
                .intruder_peak_mask
                .into_data()
                .to_vec::<f32>()
                .expect("intruder peaks"),
            vec![0.0, 0.0]
        );
    }

    #[test]
    fn vector_augmentation_masks_precursor_atomically_and_keeps_target() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let batcher = AugmentingAutoencoderBatcher::new(
            SpectrumAugmentationConfig {
                precursor_mask_probability: 1.0,
                ..SpectrumAugmentationConfig::default()
            },
            23,
        );
        let batch: AutoencoderBatch<B> = batcher.batch(
            vec![AutoencoderSample {
                spectrum: vec![0.1, 0.8],
                conditions: vec![0.25, 1.0],
            }],
            &device,
        );

        assert_eq!(
            batch.conditions.into_data().to_vec::<f32>().expect("input"),
            vec![0.0, 0.0]
        );
        assert_eq!(
            batch
                .target_conditions
                .into_data()
                .to_vec::<f32>()
                .expect("target"),
            vec![0.25, 1.0]
        );
        assert_eq!(
            batch
                .masked_precursor_mask
                .into_data()
                .to_vec::<f32>()
                .expect("masked precursor"),
            vec![1.0]
        );
    }

    #[test]
    fn token_augmentation_masks_precursor_atomically_and_keeps_target() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let batcher = AugmentingTokenizedAutoencoderBatcher::new(
            SpectrumAugmentationConfig {
                precursor_mask_probability: 1.0,
                ..SpectrumAugmentationConfig::default()
            },
            29,
        );
        let batch: TokenizedAutoencoderBatch<B> = batcher.batch(
            vec![TokenizedAutoencoderSample {
                token_features: vec![0.1, 0.8, 1.0],
                target_pairs: vec![0.1, 0.8],
                peak_mask: vec![1.0],
                padding_mask: vec![false],
                conditions: vec![0.25, 1.0],
            }],
            &device,
        );

        assert_eq!(
            batch.conditions.into_data().to_vec::<f32>().expect("input"),
            vec![0.0, 0.0]
        );
        assert_eq!(
            batch
                .target_conditions
                .into_data()
                .to_vec::<f32>()
                .expect("target"),
            vec![0.25, 1.0]
        );
        assert_eq!(
            batch
                .masked_precursor_mask
                .into_data()
                .to_vec::<f32>()
                .expect("masked precursor"),
            vec![1.0]
        );
    }

    #[test]
    fn vector_intruders_fill_empty_slots_and_keep_targets_clean() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let batcher = AugmentingAutoencoderBatcher::new(
            SpectrumAugmentationConfig {
                intruder_peak_probability: 1.0,
                ..SpectrumAugmentationConfig::default()
            },
            13,
        );
        let batch: AutoencoderBatch<B> = batcher.batch(
            vec![AutoencoderSample {
                spectrum: vec![0.2, 0.7, 0.0, 0.0],
                conditions: vec![1.0],
            }],
            &device,
        );

        assert_eq!(
            batch
                .target_spectra
                .clone()
                .into_data()
                .to_vec::<f32>()
                .expect("target"),
            vec![0.2, 0.7, 0.0, 0.0]
        );
        assert_eq!(
            batch.spectra.into_data().to_vec::<f32>().expect("input"),
            vec![0.2, 0.7, 0.2, 0.7]
        );
        assert_eq!(
            batch
                .intruder_peak_mask
                .into_data()
                .to_vec::<f32>()
                .expect("intruder mask"),
            vec![0.0, 1.0]
        );
    }

    #[test]
    fn vector_intruders_skip_full_spectra() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let batcher = AugmentingAutoencoderBatcher::new(
            SpectrumAugmentationConfig {
                intruder_peak_probability: 1.0,
                ..SpectrumAugmentationConfig::default()
            },
            17,
        );
        let batch: AutoencoderBatch<B> = batcher.batch(
            vec![AutoencoderSample {
                spectrum: vec![0.2, 0.7, 0.3, 0.4],
                conditions: vec![1.0],
            }],
            &device,
        );

        assert_eq!(
            batch
                .intruder_peak_mask
                .into_data()
                .to_vec::<f32>()
                .expect("intruder mask"),
            vec![0.0, 0.0]
        );
    }

    #[test]
    fn token_intruders_fill_padding_and_keep_targets_clean() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let batcher = AugmentingTokenizedAutoencoderBatcher::new(
            SpectrumAugmentationConfig {
                intruder_peak_probability: 1.0,
                ..SpectrumAugmentationConfig::default()
            },
            19,
        );
        let batch: TokenizedAutoencoderBatch<B> = batcher.batch(
            vec![TokenizedAutoencoderSample {
                token_features: vec![0.2, 0.7, 1.0, 0.0, 0.0, 0.0],
                target_pairs: vec![0.2, 0.7, 0.0, 0.0],
                peak_mask: vec![1.0, 0.0],
                padding_mask: vec![false, true],
                conditions: vec![1.0],
            }],
            &device,
        );

        assert_eq!(
            batch
                .target_peak_mask
                .into_data()
                .to_vec::<f32>()
                .expect("target mask"),
            vec![1.0, 0.0]
        );
        assert_eq!(
            batch
                .peak_mask
                .into_data()
                .to_vec::<f32>()
                .expect("input mask"),
            vec![1.0, 1.0]
        );
        assert_eq!(
            batch
                .intruder_peak_mask
                .into_data()
                .to_vec::<f32>()
                .expect("intruder peaks"),
            vec![0.0, 1.0]
        );
    }
}
