//! Shared spectral reconstruction objectives.

use burn::{prelude::*, tensor::Tensor};
use serde::{Deserialize, Serialize};

/// Differentiable set reconstruction loss inspired by linear cosine scoring.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SetReconstructionLossConfig {
    /// Matching width on normalized m/z values.
    pub normalized_mz_tolerance: f64,
    /// m/z exponent for cosine-style peak products.
    pub mz_power: f64,
    /// Intensity exponent for cosine-style peak products.
    pub intensity_power: f64,
    /// Weight for matching the predicted peak count to the target count.
    pub count_weight: f64,
}

impl Default for SetReconstructionLossConfig {
    fn default() -> Self {
        Self {
            normalized_mz_tolerance: 0.01,
            mz_power: 0.0,
            intensity_power: 0.5,
            count_weight: 0.1,
        }
    }
}

/// Backward-compatible name for the peak-set reconstruction loss configuration.
pub type PeakSetLossConfig = SetReconstructionLossConfig;

/// Soft set-wise cosine reconstruction loss on normalized peak targets.
pub fn set_reconstruction_loss<B: Backend>(
    pred_mz: Tensor<B, 2>,
    pred_intensity: Tensor<B, 2>,
    pred_presence: Tensor<B, 2>,
    target_pairs: Tensor<B, 2>,
    target_mask: Tensor<B, 2>,
    config: SetReconstructionLossConfig,
) -> Tensor<B, 1> {
    let [batch_size, max_peaks] = target_mask.dims();
    let target_pairs = target_pairs.reshape([batch_size, max_peaks, 2]);
    let target_mz = target_pairs
        .clone()
        .narrow(2, 0, 1)
        .reshape([batch_size, max_peaks]);
    let target_intensity = target_pairs
        .narrow(2, 1, 1)
        .reshape([batch_size, max_peaks]);

    let pred_products = peak_products(
        pred_mz.clone(),
        pred_intensity,
        pred_presence.clone(),
        config,
    );
    let target_products = peak_products(
        target_mz.clone(),
        target_intensity,
        target_mask.clone(),
        config,
    );

    let mz_delta = (pred_mz.unsqueeze_dim::<3>(2) - target_mz.unsqueeze_dim::<3>(1)).abs();
    let sigma = config.normalized_mz_tolerance.max(1.0e-6);
    let scaled = mz_delta / sigma;
    let match_weights =
        (scaled.clone() * scaled * -0.5).exp() * target_mask.clone().unsqueeze_dim::<3>(1);
    let pair_scores = pred_products.clone().unsqueeze_dim::<3>(2)
        * target_products.clone().unsqueeze_dim::<3>(1)
        * match_weights;

    let pred_best = pair_scores
        .clone()
        .max_dim(2)
        .reshape([batch_size, max_peaks]);
    let target_best = pair_scores.max_dim(1).reshape([batch_size, max_peaks]);
    let score_sum = (pred_best.sum_dim(1) + target_best.sum_dim(1)) * 0.5;

    let pred_norm = (pred_products.clone().powf_scalar(2.0).sum_dim(1) + 1.0e-6).sqrt();
    let target_norm = (target_products.powf_scalar(2.0).sum_dim(1) + 1.0e-6).sqrt();
    let similarity = (score_sum / (pred_norm * target_norm))
        .clamp_min(0.0)
        .clamp_max(1.0);
    let cosine_loss = similarity * -1.0 + 1.0;

    let count_delta = (pred_presence.sum_dim(1) - target_mask.sum_dim(1)) / max_peaks as f64;
    let count_loss = count_delta.powf_scalar(2.0) * config.count_weight;
    (cosine_loss + count_loss).mean()
}

/// Set reconstruction loss for decoder outputs shaped as `[batch, peaks, 3]`.
pub fn set_reconstruction_loss_from_triples<B: Backend>(
    reconstruction: Tensor<B, 3>,
    target_pairs: Tensor<B, 2>,
    target_mask: Tensor<B, 2>,
    config: SetReconstructionLossConfig,
) -> Tensor<B, 1> {
    let [batch_size, max_peaks, _output_width] = reconstruction.dims();
    let (pred_mz, pred_intensity, pred_presence) =
        split_peak_predictions(reconstruction, batch_size, max_peaks);
    set_reconstruction_loss(
        pred_mz,
        pred_intensity,
        pred_presence,
        target_pairs,
        target_mask,
        config,
    )
}

/// Set reconstruction loss for flat vectors shaped as `[batch, peaks * 2]`.
pub fn set_reconstruction_loss_from_vectors<B: Backend>(
    reconstruction: Tensor<B, 2>,
    target: Tensor<B, 2>,
    config: SetReconstructionLossConfig,
) -> Tensor<B, 1> {
    let target_mask = vector_target_mask(target.clone());
    set_reconstruction_loss_from_vectors_with_mask(reconstruction, target, target_mask, config)
}

/// Set reconstruction loss for flat vectors with an explicit target peak mask.
pub fn set_reconstruction_loss_from_vectors_with_mask<B: Backend>(
    reconstruction: Tensor<B, 2>,
    target: Tensor<B, 2>,
    target_mask: Tensor<B, 2>,
    config: SetReconstructionLossConfig,
) -> Tensor<B, 1> {
    let [batch_size, vector_width] = reconstruction.dims();
    let max_peaks = vector_width / 2;
    debug_assert_eq!(vector_width % 2, 0);

    let predicted_pairs = reconstruction.reshape([batch_size, max_peaks, 2]);
    let pred_mz = predicted_pairs
        .clone()
        .narrow(2, 0, 1)
        .reshape([batch_size, max_peaks]);
    let pred_intensity = predicted_pairs
        .narrow(2, 1, 1)
        .reshape([batch_size, max_peaks]);
    let pred_presence = pred_intensity.clone();

    set_reconstruction_loss(
        pred_mz,
        pred_intensity,
        pred_presence,
        target,
        target_mask,
        config,
    )
}

/// Peak mask derived from a flattened target vector.
pub fn vector_target_mask<B: Backend>(target: Tensor<B, 2>) -> Tensor<B, 2> {
    let [batch_size, vector_width] = target.dims();
    let max_peaks = vector_width / 2;
    debug_assert_eq!(vector_width % 2, 0);

    target
        .reshape([batch_size, max_peaks, 2])
        .narrow(2, 1, 1)
        .reshape([batch_size, max_peaks])
        .greater_elem(0.0)
        .float()
}

/// Converts a flattened element mask into a peak-level mask.
pub fn vector_element_mask_to_peak_mask<B: Backend>(mask: Tensor<B, 2>) -> Tensor<B, 2> {
    let [batch_size, vector_width] = mask.dims();
    let max_peaks = vector_width / 2;
    debug_assert_eq!(vector_width % 2, 0);

    mask.reshape([batch_size, max_peaks, 2])
        .sum_dim(2)
        .reshape([batch_size, max_peaks])
        .greater_elem(0.0)
        .float()
}

fn split_peak_predictions<B: Backend>(
    predictions: Tensor<B, 3>,
    batch_size: usize,
    max_peaks: usize,
) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
    let mz = predictions
        .clone()
        .narrow(2, 0, 1)
        .reshape([batch_size, max_peaks]);
    let intensity = predictions
        .clone()
        .narrow(2, 1, 1)
        .reshape([batch_size, max_peaks]);
    let presence = predictions.narrow(2, 2, 1).reshape([batch_size, max_peaks]);
    (mz, intensity, presence)
}

fn peak_products<B: Backend>(
    mz: Tensor<B, 2>,
    intensity: Tensor<B, 2>,
    presence: Tensor<B, 2>,
    config: SetReconstructionLossConfig,
) -> Tensor<B, 2> {
    let mz_component = if config.mz_power == 0.0 {
        mz.ones_like()
    } else {
        mz.clamp_min(1.0e-6).powf_scalar(config.mz_power)
    };
    let intensity_component = intensity.clamp_min(0.0).powf_scalar(config.intensity_power);
    mz_component * intensity_component * presence
}

#[cfg(all(test, feature = "ndarray"))]
mod tests {
    use super::*;

    #[test]
    fn vector_and_triple_set_losses_match_for_equivalent_predictions() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let config = SetReconstructionLossConfig::default();
        let vector = Tensor::<B, 2>::from_floats([[0.1, 1.0, 0.2, 0.5]], &device);
        let triples = Tensor::<B, 3>::from_floats([[[0.1, 1.0, 1.0], [0.2, 0.5, 0.5]]], &device);
        let target = Tensor::<B, 2>::from_floats([[0.1, 1.0, 0.2, 0.5]], &device);
        let target_mask = Tensor::<B, 2>::from_floats([[1.0, 1.0]], &device);

        let vector_loss =
            set_reconstruction_loss_from_vectors(vector, target.clone(), config).into_scalar();
        let triple_loss =
            set_reconstruction_loss_from_triples(triples, target, target_mask, config)
                .into_scalar();

        assert!((vector_loss - triple_loss).abs() < 1.0e-6);
    }

    #[test]
    fn vector_masks_are_peak_level() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let mask = Tensor::<B, 2>::from_floats([[1.0, 0.0, 0.0, 1.0]], &device);

        let peak_mask = vector_element_mask_to_peak_mask(mask).into_data();

        assert_eq!(peak_mask.as_slice::<f32>().expect("f32 data"), &[1.0, 1.0]);
    }
}
