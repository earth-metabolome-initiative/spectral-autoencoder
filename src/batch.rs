//! Burn batch types for autoencoder training.

#[cfg(feature = "std")]
use burn::data::dataloader::batcher::Batcher;
#[cfg(feature = "std")]
use burn::tensor::TensorData;
use burn::{
    prelude::*,
    tensor::{Bool, Tensor},
};

use crate::model::auxiliary::SimilarityRankingBatch;

/// One vectorized autoencoder sample.
#[derive(Debug, Clone, PartialEq)]
pub struct AutoencoderSample {
    /// Cleaned fixed-length spectrum vector.
    pub spectrum: Vec<f32>,
    /// Optional metadata conditioning vector with explicit unknown buckets.
    pub conditions: Vec<f32>,
}

/// Batched tensors for the autoencoder.
#[derive(Debug, Clone)]
pub struct AutoencoderBatch<B: Backend> {
    /// Input spectra fed to the encoder.
    pub spectra: Tensor<B, 2>,
    /// Cleaned reconstruction targets.
    pub target_spectra: Tensor<B, 2>,
    /// Metadata conditions fed to the encoder.
    pub conditions: Tensor<B, 2>,
    /// Clean metadata condition targets reconstructed by the decoder.
    pub target_conditions: Tensor<B, 2>,
    /// Float mask with `1.0` for rows whose precursor condition was masked in the encoder input.
    pub masked_precursor_mask: Tensor<B, 2>,
    /// Second input view used for latent consistency.
    pub consistency_spectra: Tensor<B, 2>,
    /// Conditions for the second input view.
    pub consistency_conditions: Tensor<B, 2>,
    /// Float mask with `1.0` for masked or dropped spectral vector positions.
    pub masked_spectra_mask: Tensor<B, 2>,
    /// Float mask with `1.0` for synthetic intruder peak slots.
    pub intruder_peak_mask: Tensor<B, 2>,
    /// Teacher-supervised similarity-ranking partners and score gaps.
    pub similarity_ranking: SimilarityRankingBatch<B>,
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
    /// Metadata conditions fed to the encoder.
    pub conditions: Tensor<B, 2>,
    /// Clean metadata condition targets reconstructed by the decoder.
    pub target_conditions: Tensor<B, 2>,
    /// Float mask with `1.0` for rows whose precursor condition was masked in the encoder input.
    pub masked_precursor_mask: Tensor<B, 2>,
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
    /// Teacher-supervised similarity-ranking partners and score gaps.
    pub similarity_ranking: SimilarityRankingBatch<B>,
}

/// Converts vectorized samples into Burn tensors.
#[cfg(feature = "std")]
#[derive(Debug, Clone, Default)]
pub struct AutoencoderBatcher;

#[cfg(feature = "std")]
impl<B: Backend> Batcher<B, AutoencoderSample, AutoencoderBatch<B>> for AutoencoderBatcher {
    fn batch(&self, items: Vec<AutoencoderSample>, device: &B::Device) -> AutoencoderBatch<B> {
        let layout = AutoencoderBatchLayout::from_samples(&items);

        let mut spectra = Vec::with_capacity(layout.spectrum_capacity());
        let mut target_spectra = Vec::with_capacity(layout.spectrum_capacity());
        let mut conditions = Vec::with_capacity(layout.condition_capacity());
        let mut target_conditions = Vec::with_capacity(layout.condition_capacity());
        let mut masked_precursor_mask = Vec::with_capacity(layout.batch_size);
        let mut consistency_spectra = Vec::with_capacity(layout.spectrum_capacity());
        let mut consistency_conditions = Vec::with_capacity(layout.condition_capacity());
        let mut masked_spectra_mask = Vec::with_capacity(layout.spectrum_capacity());
        let mut intruder_peak_mask = Vec::with_capacity(layout.peak_capacity());
        for item in items {
            spectra.extend_from_slice(&item.spectrum);
            target_spectra.extend_from_slice(&item.spectrum);
            consistency_spectra.extend(item.spectrum);
            conditions.extend_from_slice(&item.conditions);
            target_conditions.extend_from_slice(&item.conditions);
            consistency_conditions.extend(item.conditions);
            masked_precursor_mask.push(0.0);
            masked_spectra_mask.resize(masked_spectra_mask.len() + layout.spectrum_width, 0.0);
            intruder_peak_mask.resize(intruder_peak_mask.len() + layout.peak_count(), 0.0);
        }

        autoencoder_batch_from_parts(
            AutoencoderBatchParts {
                layout,
                spectra,
                target_spectra,
                conditions,
                target_conditions,
                masked_precursor_mask,
                consistency_spectra,
                consistency_conditions,
                masked_spectra_mask,
                intruder_peak_mask,
                similarity_ranking: None,
            },
            device,
        )
    }
}

/// Converts tokenized samples into Burn tensors.
#[cfg(feature = "std")]
#[derive(Debug, Clone, Default)]
pub struct TokenizedAutoencoderBatcher;

#[cfg(feature = "std")]
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
        let mut target_conditions = Vec::with_capacity(layout.condition_capacity());
        let mut masked_precursor_mask = Vec::with_capacity(layout.batch_size);
        let mut consistency_token_features = Vec::with_capacity(layout.token_feature_capacity());
        let mut consistency_peak_mask = Vec::with_capacity(layout.peak_capacity());
        let mut consistency_padding_mask = Vec::with_capacity(layout.peak_capacity());
        let mut consistency_conditions = Vec::with_capacity(layout.condition_capacity());
        let mut masked_peak_mask = Vec::with_capacity(layout.peak_capacity());
        let mut intruder_peak_mask = Vec::with_capacity(layout.peak_capacity());
        for item in items {
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
            target_conditions.extend_from_slice(&item.conditions);
            consistency_conditions.extend(item.conditions);
            masked_precursor_mask.push(0.0);
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
                consistency_token_features,
                consistency_peak_mask,
                consistency_padding_mask,
                consistency_conditions,
                masked_peak_mask,
                intruder_peak_mask,
                similarity_ranking: None,
            },
            device,
        )
    }
}

#[derive(Debug, Clone, Copy)]
#[cfg(feature = "std")]
pub(crate) struct AutoencoderBatchLayout {
    batch_size: usize,
    spectrum_width: usize,
    condition_width: usize,
}

#[cfg(feature = "std")]
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

    pub(crate) const fn batch_capacity(&self) -> usize {
        self.batch_size
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
}

#[cfg(feature = "std")]
pub(crate) struct AutoencoderBatchParts<B: Backend> {
    pub(crate) layout: AutoencoderBatchLayout,
    pub(crate) spectra: Vec<f32>,
    pub(crate) target_spectra: Vec<f32>,
    pub(crate) conditions: Vec<f32>,
    pub(crate) target_conditions: Vec<f32>,
    pub(crate) masked_precursor_mask: Vec<f32>,
    pub(crate) consistency_spectra: Vec<f32>,
    pub(crate) consistency_conditions: Vec<f32>,
    pub(crate) masked_spectra_mask: Vec<f32>,
    pub(crate) intruder_peak_mask: Vec<f32>,
    pub(crate) similarity_ranking: Option<SimilarityRankingBatch<B>>,
}

#[cfg(feature = "std")]
pub(crate) fn autoencoder_batch_from_parts<B: Backend>(
    parts: AutoencoderBatchParts<B>,
    device: &B::Device,
) -> AutoencoderBatch<B> {
    let similarity_ranking = parts
        .similarity_ranking
        .unwrap_or_else(|| SimilarityRankingBatch::zeros(parts.layout.batch_size, device));
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
        target_conditions: Tensor::<B, 2>::from_data(
            TensorData::new(
                parts.target_conditions,
                [parts.layout.batch_size, parts.layout.condition_width],
            ),
            device,
        ),
        masked_precursor_mask: Tensor::<B, 2>::from_data(
            TensorData::new(parts.masked_precursor_mask, [parts.layout.batch_size, 1]),
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
        similarity_ranking,
    }
}

#[derive(Debug, Clone, Copy)]
#[cfg(feature = "std")]
pub(crate) struct TokenizedAutoencoderBatchLayout {
    batch_size: usize,
    max_peaks: usize,
    token_feature_width: usize,
    target_width: usize,
    condition_width: usize,
}

#[cfg(feature = "std")]
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

    pub(crate) const fn batch_capacity(&self) -> usize {
        self.batch_size
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
}

#[cfg(feature = "std")]
pub(crate) struct TokenizedAutoencoderBatchParts<B: Backend> {
    pub(crate) layout: TokenizedAutoencoderBatchLayout,
    pub(crate) token_features: Vec<f32>,
    pub(crate) target_pairs: Vec<f32>,
    pub(crate) peak_mask: Vec<f32>,
    pub(crate) target_peak_mask: Vec<f32>,
    pub(crate) padding_mask: Vec<bool>,
    pub(crate) conditions: Vec<f32>,
    pub(crate) target_conditions: Vec<f32>,
    pub(crate) masked_precursor_mask: Vec<f32>,
    pub(crate) consistency_token_features: Vec<f32>,
    pub(crate) consistency_peak_mask: Vec<f32>,
    pub(crate) consistency_padding_mask: Vec<bool>,
    pub(crate) consistency_conditions: Vec<f32>,
    pub(crate) masked_peak_mask: Vec<f32>,
    pub(crate) intruder_peak_mask: Vec<f32>,
    pub(crate) similarity_ranking: Option<SimilarityRankingBatch<B>>,
}

#[cfg(feature = "std")]
pub(crate) fn tokenized_autoencoder_batch_from_parts<B: Backend>(
    parts: TokenizedAutoencoderBatchParts<B>,
    device: &B::Device,
) -> TokenizedAutoencoderBatch<B> {
    let similarity_ranking = parts
        .similarity_ranking
        .unwrap_or_else(|| SimilarityRankingBatch::zeros(parts.layout.batch_size, device));
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
        target_conditions: Tensor::<B, 2>::from_data(
            TensorData::new(
                parts.target_conditions,
                [parts.layout.batch_size, parts.layout.condition_width],
            ),
            device,
        ),
        masked_precursor_mask: Tensor::<B, 2>::from_data(
            TensorData::new(parts.masked_precursor_mask, [parts.layout.batch_size, 1]),
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
        similarity_ranking,
    }
}

#[cfg(all(test, feature = "std", feature = "ndarray"))]
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
                },
                AutoencoderSample {
                    spectrum: vec![0.3, 0.4],
                    conditions: vec![0.0],
                },
            ],
            &device,
        );

        assert_eq!(batch.spectra.dims(), [2, 2]);
        assert_eq!(batch.target_spectra.dims(), [2, 2]);
        assert_eq!(batch.conditions.dims(), [2, 1]);
        assert_eq!(batch.target_conditions.dims(), [2, 1]);
        assert_eq!(batch.masked_precursor_mask.dims(), [2, 1]);
        assert_eq!(batch.consistency_spectra.dims(), [2, 2]);
        assert_eq!(batch.masked_spectra_mask.dims(), [2, 2]);
        assert_eq!(batch.intruder_peak_mask.dims(), [2, 1]);
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
                },
                TokenizedAutoencoderSample {
                    token_features: vec![0.3, 0.4, 1.0, 0.5, 0.6, 1.0],
                    target_pairs: vec![0.3, 0.4, 0.5, 0.6],
                    peak_mask: vec![1.0, 1.0],
                    padding_mask: vec![false, false],
                    conditions: vec![0.0],
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
        assert_eq!(batch.target_conditions.dims(), [2, 1]);
        assert_eq!(batch.masked_precursor_mask.dims(), [2, 1]);
        assert_eq!(batch.consistency_token_features.dims(), [2, 2, 3]);
        assert_eq!(batch.consistency_peak_mask.dims(), [2, 2]);
        assert_eq!(batch.masked_peak_mask.dims(), [2, 2]);
        assert_eq!(batch.intruder_peak_mask.dims(), [2, 2]);
    }
}
