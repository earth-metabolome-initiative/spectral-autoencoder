//! Burn batch types for autoencoder training.

use std::collections::HashMap;

use burn::{
    data::dataloader::batcher::Batcher,
    prelude::*,
    tensor::{Bool, Int, Tensor, TensorData},
};

/// Per-spectrum metadata used by auxiliary objectives.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SampleMetadata {
    /// Retention time in seconds when present.
    pub retention_time: Option<f32>,
    /// Stable source-file identifier when present.
    pub filename_id: Option<u32>,
}

impl SampleMetadata {
    /// Returns `(retention_time, present_flag)` for tensor batching.
    pub fn retention_parts(self) -> (f32, f32) {
        match self.retention_time {
            Some(value) if value.is_finite() && value > 0.0 => (value, 1.0),
            _ => (0.0, 0.0),
        }
    }

    /// Returns the filename identifier as a tensor value, or `-1.0` when absent.
    pub fn filename_part(self) -> f32 {
        self.filename_id.map_or(-1.0, |id| id as f32)
    }
}

/// One vectorized autoencoder sample.
#[derive(Debug, Clone, PartialEq)]
pub struct AutoencoderSample {
    /// Cleaned fixed-length spectrum vector.
    pub spectrum: Vec<f32>,
    /// Optional metadata conditioning vector with explicit unknown buckets.
    pub conditions: Vec<f32>,
    /// Metadata used only by auxiliary objectives.
    pub metadata: SampleMetadata,
}

/// Batched tensors for the autoencoder.
#[derive(Debug, Clone)]
pub struct AutoencoderBatch<B: Backend> {
    /// Input spectra fed to the encoder.
    pub spectra: Tensor<B, 2>,
    /// Cleaned reconstruction targets.
    pub target_spectra: Tensor<B, 2>,
    /// Metadata conditions fed to both encoder and decoder.
    pub conditions: Tensor<B, 2>,
    /// Second input view used for latent consistency.
    pub consistency_spectra: Tensor<B, 2>,
    /// Conditions for the second input view.
    pub consistency_conditions: Tensor<B, 2>,
    /// Float mask with `1.0` for masked or dropped spectral vector positions.
    pub masked_spectra_mask: Tensor<B, 2>,
    /// Float mask with `1.0` for synthetic intruder peak slots.
    pub intruder_peak_mask: Tensor<B, 2>,
    /// Retention time tensor with shape `[batch, 1]`; value is zero when absent.
    pub retention_time: Tensor<B, 2>,
    /// Retention-time presence flag with shape `[batch, 1]`.
    pub retention_present: Tensor<B, 2>,
    /// Source-file identifier with shape `[batch, 1]`; value is `-1` when absent.
    pub filename_id: Tensor<B, 2>,
    /// Partner row index for DreaMS-style same-file retention-order pairs.
    pub retention_partner_index: Tensor<B, 1, Int>,
}

/// One tokenized autoencoder sample.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenizedAutoencoderSample {
    /// Flattened `[max_peaks, token_feature_width]` token features.
    pub token_features: Vec<f32>,
    /// Flattened `[max_peaks, 2]` normalized reconstruction target.
    pub target_pairs: Vec<f32>,
    /// Float mask with `1.0` for real peaks and `0.0` for padding.
    pub peak_mask: Vec<f32>,
    /// Padding mask with `true` for padding positions.
    pub padding_mask: Vec<bool>,
    /// Optional metadata conditioning vector with explicit unknown buckets.
    pub conditions: Vec<f32>,
    /// Metadata used only by auxiliary objectives.
    pub metadata: SampleMetadata,
}

/// Batched tensors for token/set autoencoder training.
#[derive(Debug, Clone)]
pub struct TokenizedAutoencoderBatch<B: Backend> {
    /// Peak-token input tensor with shape `[batch, max_peaks, token_feature_width]`.
    pub token_features: Tensor<B, 3>,
    /// Flattened normalized target pairs with shape `[batch, max_peaks * 2]`.
    pub target_pairs: Tensor<B, 2>,
    /// Float input peak mask with shape `[batch, max_peaks]`.
    pub peak_mask: Tensor<B, 2>,
    /// Float target peak mask with shape `[batch, max_peaks]`.
    pub target_peak_mask: Tensor<B, 2>,
    /// Attention padding mask with shape `[batch, max_peaks]`.
    pub padding_mask: Tensor<B, 2, Bool>,
    /// Metadata conditions fed to both encoder and decoder.
    pub conditions: Tensor<B, 2>,
    /// Second peak-token input view used for latent consistency.
    pub consistency_token_features: Tensor<B, 3>,
    /// Peak mask for the second input view.
    pub consistency_peak_mask: Tensor<B, 2>,
    /// Padding mask for the second input view.
    pub consistency_padding_mask: Tensor<B, 2, Bool>,
    /// Conditions for the second input view.
    pub consistency_conditions: Tensor<B, 2>,
    /// Float mask with `1.0` for masked or dropped target peaks.
    pub masked_peak_mask: Tensor<B, 2>,
    /// Float mask with `1.0` for synthetic intruder peak tokens.
    pub intruder_peak_mask: Tensor<B, 2>,
    /// Retention time tensor with shape `[batch, 1]`; value is zero when absent.
    pub retention_time: Tensor<B, 2>,
    /// Retention-time presence flag with shape `[batch, 1]`.
    pub retention_present: Tensor<B, 2>,
    /// Source-file identifier with shape `[batch, 1]`; value is `-1` when absent.
    pub filename_id: Tensor<B, 2>,
    /// Partner row index for DreaMS-style same-file retention-order pairs.
    pub retention_partner_index: Tensor<B, 1, Int>,
}

/// Converts vectorized samples into Burn tensors.
#[derive(Debug, Clone, Default)]
pub struct AutoencoderBatcher;

impl<B: Backend> Batcher<B, AutoencoderSample, AutoencoderBatch<B>> for AutoencoderBatcher {
    fn batch(&self, items: Vec<AutoencoderSample>, device: &B::Device) -> AutoencoderBatch<B> {
        let layout = AutoencoderBatchLayout::from_samples(&items);

        let mut spectra = Vec::with_capacity(layout.spectrum_capacity());
        let mut target_spectra = Vec::with_capacity(layout.spectrum_capacity());
        let mut conditions = Vec::with_capacity(layout.condition_capacity());
        let mut consistency_spectra = Vec::with_capacity(layout.spectrum_capacity());
        let mut consistency_conditions = Vec::with_capacity(layout.condition_capacity());
        let mut masked_spectra_mask = Vec::with_capacity(layout.spectrum_capacity());
        let mut intruder_peak_mask = Vec::with_capacity(layout.peak_capacity());
        let mut retention_time = Vec::with_capacity(layout.metadata_capacity());
        let mut retention_present = Vec::with_capacity(layout.metadata_capacity());
        let mut filename_id = Vec::with_capacity(layout.metadata_capacity());
        for item in items {
            let (rt, rt_present) = item.metadata.retention_parts();
            spectra.extend_from_slice(&item.spectrum);
            target_spectra.extend_from_slice(&item.spectrum);
            consistency_spectra.extend(item.spectrum);
            conditions.extend_from_slice(&item.conditions);
            consistency_conditions.extend(item.conditions);
            masked_spectra_mask.resize(masked_spectra_mask.len() + layout.spectrum_width, 0.0);
            intruder_peak_mask.resize(intruder_peak_mask.len() + layout.peak_count(), 0.0);
            retention_time.push(rt);
            retention_present.push(rt_present);
            filename_id.push(item.metadata.filename_part());
        }
        let retention_partner_index =
            retention_partner_indices(&filename_id, &retention_time, &retention_present);

        autoencoder_batch_from_parts(
            AutoencoderBatchParts {
                layout,
                spectra,
                target_spectra,
                conditions,
                consistency_spectra,
                consistency_conditions,
                masked_spectra_mask,
                intruder_peak_mask,
                retention_time,
                retention_present,
                filename_id,
                retention_partner_index,
            },
            device,
        )
    }
}

/// Converts tokenized samples into Burn tensors.
#[derive(Debug, Clone, Default)]
pub struct TokenizedAutoencoderBatcher;

impl<B: Backend> Batcher<B, TokenizedAutoencoderSample, TokenizedAutoencoderBatch<B>>
    for TokenizedAutoencoderBatcher
{
    fn batch(
        &self,
        items: Vec<TokenizedAutoencoderSample>,
        device: &B::Device,
    ) -> TokenizedAutoencoderBatch<B> {
        let layout = TokenizedAutoencoderBatchLayout::from_samples(&items);

        let mut token_features = Vec::with_capacity(layout.token_feature_capacity());
        let mut target_pairs = Vec::with_capacity(layout.target_capacity());
        let mut peak_mask = Vec::with_capacity(layout.peak_capacity());
        let mut target_peak_mask = Vec::with_capacity(layout.peak_capacity());
        let mut padding_mask = Vec::with_capacity(layout.peak_capacity());
        let mut conditions = Vec::with_capacity(layout.condition_capacity());
        let mut consistency_token_features = Vec::with_capacity(layout.token_feature_capacity());
        let mut consistency_peak_mask = Vec::with_capacity(layout.peak_capacity());
        let mut consistency_padding_mask = Vec::with_capacity(layout.peak_capacity());
        let mut consistency_conditions = Vec::with_capacity(layout.condition_capacity());
        let mut masked_peak_mask = Vec::with_capacity(layout.peak_capacity());
        let mut intruder_peak_mask = Vec::with_capacity(layout.peak_capacity());
        let mut retention_time = Vec::with_capacity(layout.metadata_capacity());
        let mut retention_present = Vec::with_capacity(layout.metadata_capacity());
        let mut filename_id = Vec::with_capacity(layout.metadata_capacity());
        for item in items {
            let (rt, rt_present) = item.metadata.retention_parts();
            token_features.extend_from_slice(&item.token_features);
            consistency_token_features.extend(item.token_features);
            target_pairs.extend(item.target_pairs);
            peak_mask.extend_from_slice(&item.peak_mask);
            consistency_peak_mask.extend_from_slice(&item.peak_mask);
            target_peak_mask.extend_from_slice(&item.peak_mask);
            masked_peak_mask.resize(masked_peak_mask.len() + item.peak_mask.len(), 0.0);
            intruder_peak_mask.resize(intruder_peak_mask.len() + item.peak_mask.len(), 0.0);
            padding_mask.extend_from_slice(&item.padding_mask);
            consistency_padding_mask.extend(item.padding_mask);
            conditions.extend_from_slice(&item.conditions);
            consistency_conditions.extend(item.conditions);
            retention_time.push(rt);
            retention_present.push(rt_present);
            filename_id.push(item.metadata.filename_part());
        }
        let retention_partner_index =
            retention_partner_indices(&filename_id, &retention_time, &retention_present);

        tokenized_autoencoder_batch_from_parts(
            TokenizedAutoencoderBatchParts {
                layout,
                token_features,
                target_pairs,
                peak_mask,
                target_peak_mask,
                padding_mask,
                conditions,
                consistency_token_features,
                consistency_peak_mask,
                consistency_padding_mask,
                consistency_conditions,
                masked_peak_mask,
                intruder_peak_mask,
                retention_time,
                retention_present,
                filename_id,
                retention_partner_index,
            },
            device,
        )
    }
}

/// Chooses one deterministic same-file partner per row for retention-order training.
///
/// The returned indices are row-local. Rows without another valid same-file,
/// non-tied retention-time partner point to themselves and are later masked out.
#[must_use]
pub fn retention_partner_indices(
    filename_id: &[f32],
    retention_time: &[f32],
    retention_present: &[f32],
) -> Vec<i64> {
    retention_partner_indices_with_seed(filename_id, retention_time, retention_present, 0)
}

/// Chooses one seeded same-file partner per row for retention-order training.
///
/// The seed is intended to include epoch/batch information for training loaders
/// so the same cached spectra can expose different valid partners over time.
#[must_use]
pub fn retention_partner_indices_with_seed(
    filename_id: &[f32],
    retention_time: &[f32],
    retention_present: &[f32],
    seed: u64,
) -> Vec<i64> {
    assert_eq!(
        filename_id.len(),
        retention_time.len(),
        "filename ids and retention times must have the same length"
    );
    assert_eq!(
        filename_id.len(),
        retention_present.len(),
        "filename ids and retention masks must have the same length"
    );

    let mut partners: Vec<i64> = (0..filename_id.len() as i64).collect();
    let mut by_file: HashMap<i64, Vec<usize>> = HashMap::new();
    for (index, ((file_id, rt), present)) in filename_id
        .iter()
        .zip(retention_time)
        .zip(retention_present)
        .enumerate()
    {
        if *file_id >= 0.0 && *present > 0.0 && rt.is_finite() {
            by_file.entry(*file_id as i64).or_default().push(index);
        }
    }

    for (file_id, indices) in by_file {
        if indices.len() < 2 {
            continue;
        }
        for (position, &index) in indices.iter().enumerate() {
            let mut step =
                1 + (mixed_index_seed(file_id, index, seed) as usize % (indices.len() - 1));
            for _ in 0..indices.len() - 1 {
                let partner = indices[(position + step) % indices.len()];
                if (retention_time[partner] - retention_time[index]).abs() > 1.0e-6 {
                    partners[index] = partner as i64;
                    break;
                }
                step = if step == indices.len() - 1 {
                    1
                } else {
                    step + 1
                };
            }
        }
    }

    partners
}

fn mixed_index_seed(file_id: i64, index: usize, seed: u64) -> u64 {
    let mut value = (file_id as u64)
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .wrapping_add(index as u64)
        .wrapping_add(seed.rotate_left(17));
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct AutoencoderBatchLayout {
    batch_size: usize,
    spectrum_width: usize,
    condition_width: usize,
}

impl AutoencoderBatchLayout {
    pub(crate) fn from_samples(samples: &[AutoencoderSample]) -> Self {
        let first = samples
            .first()
            .expect("autoencoder batches are never empty");
        Self {
            batch_size: samples.len(),
            spectrum_width: first.spectrum.len(),
            condition_width: first.conditions.len(),
        }
    }

    pub(crate) const fn spectrum_capacity(&self) -> usize {
        self.batch_size * self.spectrum_width
    }

    pub(crate) const fn peak_count(&self) -> usize {
        self.spectrum_width / 2
    }

    pub(crate) const fn peak_capacity(&self) -> usize {
        self.batch_size * self.peak_count()
    }

    pub(crate) const fn condition_capacity(&self) -> usize {
        self.batch_size * self.condition_width
    }

    pub(crate) const fn metadata_capacity(&self) -> usize {
        self.batch_size
    }
}

pub(crate) struct AutoencoderBatchParts {
    pub(crate) layout: AutoencoderBatchLayout,
    pub(crate) spectra: Vec<f32>,
    pub(crate) target_spectra: Vec<f32>,
    pub(crate) conditions: Vec<f32>,
    pub(crate) consistency_spectra: Vec<f32>,
    pub(crate) consistency_conditions: Vec<f32>,
    pub(crate) masked_spectra_mask: Vec<f32>,
    pub(crate) intruder_peak_mask: Vec<f32>,
    pub(crate) retention_time: Vec<f32>,
    pub(crate) retention_present: Vec<f32>,
    pub(crate) filename_id: Vec<f32>,
    pub(crate) retention_partner_index: Vec<i64>,
}

pub(crate) fn autoencoder_batch_from_parts<B: Backend>(
    parts: AutoencoderBatchParts,
    device: &B::Device,
) -> AutoencoderBatch<B> {
    AutoencoderBatch {
        spectra: Tensor::<B, 2>::from_data(
            TensorData::new(
                parts.spectra,
                [parts.layout.batch_size, parts.layout.spectrum_width],
            ),
            device,
        ),
        target_spectra: Tensor::<B, 2>::from_data(
            TensorData::new(
                parts.target_spectra,
                [parts.layout.batch_size, parts.layout.spectrum_width],
            ),
            device,
        ),
        conditions: Tensor::<B, 2>::from_data(
            TensorData::new(
                parts.conditions,
                [parts.layout.batch_size, parts.layout.condition_width],
            ),
            device,
        ),
        consistency_spectra: Tensor::<B, 2>::from_data(
            TensorData::new(
                parts.consistency_spectra,
                [parts.layout.batch_size, parts.layout.spectrum_width],
            ),
            device,
        ),
        consistency_conditions: Tensor::<B, 2>::from_data(
            TensorData::new(
                parts.consistency_conditions,
                [parts.layout.batch_size, parts.layout.condition_width],
            ),
            device,
        ),
        masked_spectra_mask: Tensor::<B, 2>::from_data(
            TensorData::new(
                parts.masked_spectra_mask,
                [parts.layout.batch_size, parts.layout.spectrum_width],
            ),
            device,
        ),
        intruder_peak_mask: Tensor::<B, 2>::from_data(
            TensorData::new(
                parts.intruder_peak_mask,
                [parts.layout.batch_size, parts.layout.peak_count()],
            ),
            device,
        ),
        retention_time: Tensor::<B, 2>::from_data(
            TensorData::new(parts.retention_time, [parts.layout.batch_size, 1]),
            device,
        ),
        retention_present: Tensor::<B, 2>::from_data(
            TensorData::new(parts.retention_present, [parts.layout.batch_size, 1]),
            device,
        ),
        filename_id: Tensor::<B, 2>::from_data(
            TensorData::new(parts.filename_id, [parts.layout.batch_size, 1]),
            device,
        ),
        retention_partner_index: Tensor::<B, 1, Int>::from_data(
            TensorData::new(parts.retention_partner_index, [parts.layout.batch_size]),
            device,
        ),
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct TokenizedAutoencoderBatchLayout {
    batch_size: usize,
    max_peaks: usize,
    token_feature_width: usize,
    target_width: usize,
    condition_width: usize,
}

impl TokenizedAutoencoderBatchLayout {
    pub(crate) fn from_samples(samples: &[TokenizedAutoencoderSample]) -> Self {
        let first = samples
            .first()
            .expect("tokenized autoencoder batches are never empty");
        let max_peaks = first.peak_mask.len();
        Self {
            batch_size: samples.len(),
            max_peaks,
            token_feature_width: first.token_features.len() / max_peaks,
            target_width: first.target_pairs.len(),
            condition_width: first.conditions.len(),
        }
    }

    pub(crate) const fn token_feature_capacity(&self) -> usize {
        self.batch_size * self.max_peaks * self.token_feature_width
    }

    pub(crate) const fn target_capacity(&self) -> usize {
        self.batch_size * self.target_width
    }

    pub(crate) const fn peak_capacity(&self) -> usize {
        self.batch_size * self.max_peaks
    }

    pub(crate) const fn condition_capacity(&self) -> usize {
        self.batch_size * self.condition_width
    }

    pub(crate) const fn metadata_capacity(&self) -> usize {
        self.batch_size
    }
}

pub(crate) struct TokenizedAutoencoderBatchParts {
    pub(crate) layout: TokenizedAutoencoderBatchLayout,
    pub(crate) token_features: Vec<f32>,
    pub(crate) target_pairs: Vec<f32>,
    pub(crate) peak_mask: Vec<f32>,
    pub(crate) target_peak_mask: Vec<f32>,
    pub(crate) padding_mask: Vec<bool>,
    pub(crate) conditions: Vec<f32>,
    pub(crate) consistency_token_features: Vec<f32>,
    pub(crate) consistency_peak_mask: Vec<f32>,
    pub(crate) consistency_padding_mask: Vec<bool>,
    pub(crate) consistency_conditions: Vec<f32>,
    pub(crate) masked_peak_mask: Vec<f32>,
    pub(crate) intruder_peak_mask: Vec<f32>,
    pub(crate) retention_time: Vec<f32>,
    pub(crate) retention_present: Vec<f32>,
    pub(crate) filename_id: Vec<f32>,
    pub(crate) retention_partner_index: Vec<i64>,
}

pub(crate) fn tokenized_autoencoder_batch_from_parts<B: Backend>(
    parts: TokenizedAutoencoderBatchParts,
    device: &B::Device,
) -> TokenizedAutoencoderBatch<B> {
    TokenizedAutoencoderBatch {
        token_features: Tensor::<B, 3>::from_data(
            TensorData::new(
                parts.token_features,
                [
                    parts.layout.batch_size,
                    parts.layout.max_peaks,
                    parts.layout.token_feature_width,
                ],
            ),
            device,
        ),
        target_pairs: Tensor::<B, 2>::from_data(
            TensorData::new(
                parts.target_pairs,
                [parts.layout.batch_size, parts.layout.target_width],
            ),
            device,
        ),
        peak_mask: Tensor::<B, 2>::from_data(
            TensorData::new(
                parts.peak_mask,
                [parts.layout.batch_size, parts.layout.max_peaks],
            ),
            device,
        ),
        target_peak_mask: Tensor::<B, 2>::from_data(
            TensorData::new(
                parts.target_peak_mask,
                [parts.layout.batch_size, parts.layout.max_peaks],
            ),
            device,
        ),
        padding_mask: Tensor::<B, 2, Bool>::from_bool(
            TensorData::new(
                parts.padding_mask,
                [parts.layout.batch_size, parts.layout.max_peaks],
            ),
            device,
        ),
        conditions: Tensor::<B, 2>::from_data(
            TensorData::new(
                parts.conditions,
                [parts.layout.batch_size, parts.layout.condition_width],
            ),
            device,
        ),
        consistency_token_features: Tensor::<B, 3>::from_data(
            TensorData::new(
                parts.consistency_token_features,
                [
                    parts.layout.batch_size,
                    parts.layout.max_peaks,
                    parts.layout.token_feature_width,
                ],
            ),
            device,
        ),
        consistency_peak_mask: Tensor::<B, 2>::from_data(
            TensorData::new(
                parts.consistency_peak_mask,
                [parts.layout.batch_size, parts.layout.max_peaks],
            ),
            device,
        ),
        consistency_padding_mask: Tensor::<B, 2, Bool>::from_bool(
            TensorData::new(
                parts.consistency_padding_mask,
                [parts.layout.batch_size, parts.layout.max_peaks],
            ),
            device,
        ),
        consistency_conditions: Tensor::<B, 2>::from_data(
            TensorData::new(
                parts.consistency_conditions,
                [parts.layout.batch_size, parts.layout.condition_width],
            ),
            device,
        ),
        masked_peak_mask: Tensor::<B, 2>::from_data(
            TensorData::new(
                parts.masked_peak_mask,
                [parts.layout.batch_size, parts.layout.max_peaks],
            ),
            device,
        ),
        intruder_peak_mask: Tensor::<B, 2>::from_data(
            TensorData::new(
                parts.intruder_peak_mask,
                [parts.layout.batch_size, parts.layout.max_peaks],
            ),
            device,
        ),
        retention_time: Tensor::<B, 2>::from_data(
            TensorData::new(parts.retention_time, [parts.layout.batch_size, 1]),
            device,
        ),
        retention_present: Tensor::<B, 2>::from_data(
            TensorData::new(parts.retention_present, [parts.layout.batch_size, 1]),
            device,
        ),
        filename_id: Tensor::<B, 2>::from_data(
            TensorData::new(parts.filename_id, [parts.layout.batch_size, 1]),
            device,
        ),
        retention_partner_index: Tensor::<B, 1, Int>::from_data(
            TensorData::new(parts.retention_partner_index, [parts.layout.batch_size]),
            device,
        ),
    }
}

#[cfg(all(test, feature = "ndarray"))]
mod tests {
    use super::*;

    #[test]
    fn batcher_preserves_shapes() {
        type B = burn::backend::NdArray<f32, i64>;
        let batcher = AutoencoderBatcher;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let batch: AutoencoderBatch<B> = batcher.batch(
            vec![
                AutoencoderSample {
                    spectrum: vec![0.1, 0.2],
                    conditions: vec![1.0],
                    metadata: SampleMetadata::default(),
                },
                AutoencoderSample {
                    spectrum: vec![0.3, 0.4],
                    conditions: vec![0.0],
                    metadata: SampleMetadata {
                        retention_time: Some(12.0),
                        filename_id: Some(3),
                    },
                },
            ],
            &device,
        );

        assert_eq!(batch.spectra.dims(), [2, 2]);
        assert_eq!(batch.target_spectra.dims(), [2, 2]);
        assert_eq!(batch.conditions.dims(), [2, 1]);
        assert_eq!(batch.consistency_spectra.dims(), [2, 2]);
        assert_eq!(batch.masked_spectra_mask.dims(), [2, 2]);
        assert_eq!(batch.intruder_peak_mask.dims(), [2, 1]);
        assert_eq!(batch.retention_time.dims(), [2, 1]);
        assert_eq!(batch.filename_id.dims(), [2, 1]);
        assert_eq!(batch.retention_partner_index.dims(), [2]);
    }

    #[test]
    fn tokenized_batcher_preserves_shapes() {
        type B = burn::backend::NdArray<f32, i64>;
        let batcher = TokenizedAutoencoderBatcher;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let batch: TokenizedAutoencoderBatch<B> = batcher.batch(
            vec![
                TokenizedAutoencoderSample {
                    token_features: vec![0.1, 0.2, 1.0, 0.0, 0.0, 0.0],
                    target_pairs: vec![0.1, 0.2, 0.0, 0.0],
                    peak_mask: vec![1.0, 0.0],
                    padding_mask: vec![false, true],
                    conditions: vec![1.0],
                    metadata: SampleMetadata::default(),
                },
                TokenizedAutoencoderSample {
                    token_features: vec![0.3, 0.4, 1.0, 0.5, 0.6, 1.0],
                    target_pairs: vec![0.3, 0.4, 0.5, 0.6],
                    peak_mask: vec![1.0, 1.0],
                    padding_mask: vec![false, false],
                    conditions: vec![0.0],
                    metadata: SampleMetadata {
                        retention_time: Some(13.0),
                        filename_id: Some(3),
                    },
                },
            ],
            &device,
        );

        assert_eq!(batch.token_features.dims(), [2, 2, 3]);
        assert_eq!(batch.target_pairs.dims(), [2, 4]);
        assert_eq!(batch.peak_mask.dims(), [2, 2]);
        assert_eq!(batch.target_peak_mask.dims(), [2, 2]);
        assert_eq!(batch.padding_mask.dims(), [2, 2]);
        assert_eq!(batch.conditions.dims(), [2, 1]);
        assert_eq!(batch.consistency_token_features.dims(), [2, 2, 3]);
        assert_eq!(batch.consistency_peak_mask.dims(), [2, 2]);
        assert_eq!(batch.masked_peak_mask.dims(), [2, 2]);
        assert_eq!(batch.intruder_peak_mask.dims(), [2, 2]);
        assert_eq!(batch.retention_time.dims(), [2, 1]);
        assert_eq!(batch.filename_id.dims(), [2, 1]);
        assert_eq!(batch.retention_partner_index.dims(), [2]);
    }

    #[test]
    fn retention_partners_use_same_file_non_tied_rows() {
        let filename_id = vec![1.0, 2.0, 1.0, 1.0, 2.0, -1.0];
        let retention_time = vec![10.0, 20.0, 10.0, 40.0, 25.0, 50.0];
        let retention_present = vec![1.0; 6];

        let partners = retention_partner_indices(&filename_id, &retention_time, &retention_present);

        assert_eq!(partners[1], 4);
        assert_eq!(partners[4], 1);
        assert_eq!(partners[5], 5);
        for &index in &[0_usize, 2, 3] {
            let partner = partners[index] as usize;
            assert_eq!(filename_id[partner], filename_id[index]);
            assert_ne!(retention_time[partner], retention_time[index]);
        }
    }

    #[test]
    fn retention_partners_change_with_seed_when_choices_exist() {
        let filename_id = vec![1.0; 6];
        let retention_time = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let retention_present = vec![1.0; 6];

        let first = retention_partner_indices_with_seed(
            &filename_id,
            &retention_time,
            &retention_present,
            0,
        );
        let second = retention_partner_indices_with_seed(
            &filename_id,
            &retention_time,
            &retention_present,
            1,
        );

        assert_ne!(first, second);
        for (index, &partner) in second.iter().enumerate() {
            let partner = partner as usize;
            assert_eq!(filename_id[partner], filename_id[index]);
            assert_ne!(retention_time[partner], retention_time[index]);
        }
    }
}
