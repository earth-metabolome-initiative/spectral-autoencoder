//! Shared spectral reconstruction objectives.

use burn::{
    prelude::*,
    tensor::{Distribution, Int, Tensor},
};
use serde::{Deserialize, Serialize};

/// Additive penalty applied to padded targets inside the Chamfer m/z magnet
/// so they never win the per-pred-slot `min`. Well above the maximum possible
/// `(\Delta m/z)^2 \le 1` for normalised m/z values.
const CHAMFER_INACTIVE_PENALTY: f64 = 1.0e6;

/// Flat-vector reconstruction alignment strategy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum FlatVectorReconstructionOrdering {
    /// Compare decoder slot `k` with target slot `k`.
    #[default]
    Slot,
    /// Sort predicted and target peak pairs by descending intensity before comparison.
    IntensityDescending,
}

impl FlatVectorReconstructionOrdering {
    /// Stable label used in run headers and environment configuration.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Slot => "slot",
            Self::IntensityDescending => "intensity-desc",
        }
    }

    pub(crate) const fn code(self) -> usize {
        match self {
            Self::Slot => 0,
            Self::IntensityDescending => 1,
        }
    }

    pub(crate) const fn from_code(code: usize) -> Self {
        match code {
            0 => Self::Slot,
            1 => Self::IntensityDescending,
            _ => Self::Slot,
        }
    }
}

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

impl SetReconstructionLossConfig {
    /// Starts a fluent builder seeded with [`Self::default`].
    #[must_use]
    pub fn builder() -> SetReconstructionLossConfigBuilder {
        SetReconstructionLossConfigBuilder::default()
    }
}

/// Fluent builder for [`SetReconstructionLossConfig`].
#[derive(Debug, Clone, Copy, Default)]
pub struct SetReconstructionLossConfigBuilder {
    config: SetReconstructionLossConfig,
}

impl SetReconstructionLossConfigBuilder {
    /// Creates a builder seeded with [`SetReconstructionLossConfig::default`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the normalized m/z matching tolerance.
    #[inline]
    #[must_use]
    pub fn with_normalized_mz_tolerance(mut self, value: f64) -> Self {
        self.config.normalized_mz_tolerance = value;
        self
    }

    /// Sets the m/z exponent for cosine-style peak products.
    #[inline]
    #[must_use]
    pub fn with_mz_power(mut self, value: f64) -> Self {
        self.config.mz_power = value;
        self
    }

    /// Sets the intensity exponent for cosine-style peak products.
    #[inline]
    #[must_use]
    pub fn with_intensity_power(mut self, value: f64) -> Self {
        self.config.intensity_power = value;
        self
    }

    /// Sets the weight for matching predicted and target peak counts.
    #[inline]
    #[must_use]
    pub fn with_count_weight(mut self, value: f64) -> Self {
        self.config.count_weight = value;
        self
    }

    /// Returns the configured [`SetReconstructionLossConfig`].
    #[inline]
    #[must_use]
    pub fn build(self) -> SetReconstructionLossConfig {
        self.config
    }
}

#[cfg(feature = "std")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn reconstruction_similarity_for_match_mode<B: Backend>(
    pred_mz: Tensor<B, 2>,
    pred_products: Tensor<B, 2>,
    pred_precursor: Tensor<B, 2>,
    target_mz: Tensor<B, 2>,
    target_products: Tensor<B, 2>,
    target_mask: Tensor<B, 2>,
    target_precursor: Tensor<B, 2>,
    normalized_mz_tolerance: f64,
    modified: bool,
) -> Tensor<B, 1> {
    let [batch_size, max_peaks] = pred_mz.dims();
    let tolerance = normalized_mz_tolerance.max(1.0e-6);
    let ordinary_delta =
        (pred_mz.clone().unsqueeze_dim::<3>(2) - target_mz.clone().unsqueeze_dim::<3>(1)).abs();
    let ordinary_match = ordinary_delta
        .greater_elem(tolerance)
        .float()
        .mul_scalar(-1.0)
        + 1.0;
    let match_weights = if modified {
        let pred_shifted = pred_mz - pred_precursor.expand([batch_size, max_peaks]);
        let target_shifted = target_mz - target_precursor.expand([batch_size, max_peaks]);
        let shifted_delta =
            (pred_shifted.unsqueeze_dim::<3>(2) - target_shifted.unsqueeze_dim::<3>(1)).abs();
        let shifted_match = shifted_delta
            .greater_elem(tolerance)
            .float()
            .mul_scalar(-1.0)
            + 1.0;
        (ordinary_match + shifted_match).clamp_max(1.0)
    } else {
        ordinary_match
    } * target_mask.unsqueeze_dim::<3>(1);

    let pair_scores = pred_products.clone().unsqueeze_dim::<3>(2)
        * target_products.clone().unsqueeze_dim::<3>(1)
        * match_weights;
    let pred_best = pair_scores
        .clone()
        .max_dim(2)
        .reshape([batch_size, max_peaks]);
    let target_best = pair_scores.max_dim(1).reshape([batch_size, max_peaks]);
    let score_sum = (pred_best.sum_dim(1) + target_best.sum_dim(1)) * 0.5;
    let pred_norm = (pred_products.powf_scalar(2.0).sum_dim(1) + 1.0e-6).sqrt();
    let target_norm = (target_products.powf_scalar(2.0).sum_dim(1) + 1.0e-6).sqrt();

    (score_sum / (pred_norm * target_norm))
        .clamp_min(0.0)
        .clamp_max(1.0)
        .reshape([batch_size])
}

#[cfg(feature = "std")]
pub(crate) fn normalized_precursor<B: Backend>(conditions: Tensor<B, 2>) -> Tensor<B, 2> {
    conditions.narrow(1, 0, 1).clamp_min(0.0).clamp_max(1.0)
}

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

/// Ordered slot-wise reconstruction loss for flat vectors shaped as `[batch, peaks * 2]`.
pub fn slot_reconstruction_loss_from_vectors<B: Backend>(
    reconstruction: Tensor<B, 2>,
    target: Tensor<B, 2>,
    config: SetReconstructionLossConfig,
) -> Tensor<B, 1> {
    let target_mask = vector_target_mask(target.clone());
    slot_reconstruction_loss_from_vectors_with_mask(reconstruction, target, target_mask, config)
}

/// Ordered slot-wise reconstruction loss for flat vectors with an explicit target peak mask.
///
/// The pred-side L2 norm includes all `P` slots, so phantom intensity on
/// padded slots inflates the denominator and implicitly penalises spurious
/// predictions. This is the right semantics for the clean reconstruction
/// path; for the *masked* path use
/// [`slot_reconstruction_loss_restricted_from_vectors_with_mask`] instead,
/// which restricts both sides of the cosine to the same mask so the signal
/// is meaningful when the target mask is very sparse.
pub fn slot_reconstruction_loss_from_vectors_with_mask<B: Backend>(
    reconstruction: Tensor<B, 2>,
    target: Tensor<B, 2>,
    target_mask: Tensor<B, 2>,
    config: SetReconstructionLossConfig,
) -> Tensor<B, 1> {
    slot_reconstruction_loss_impl(reconstruction, target, target_mask, false, config)
}

/// Ordered slot-wise reconstruction loss restricted to the slots indicated
/// by `target_mask` on both sides of the cosine. The pred-side L2 norm is
/// gated by `target_mask` so it sums over the same support as the
/// target-side norm. Use this for the masked-peak path where `target_mask`
/// is the masked-and-real subset; the full reconstruction path should keep
/// the unrestricted variant so phantom-peak penalty still applies on
/// padded slots.
pub fn slot_reconstruction_loss_restricted_from_vectors_with_mask<B: Backend>(
    reconstruction: Tensor<B, 2>,
    target: Tensor<B, 2>,
    target_mask: Tensor<B, 2>,
    config: SetReconstructionLossConfig,
) -> Tensor<B, 1> {
    slot_reconstruction_loss_impl(reconstruction, target, target_mask, true, config)
}

fn slot_reconstruction_loss_impl<B: Backend>(
    reconstruction: Tensor<B, 2>,
    target: Tensor<B, 2>,
    target_mask: Tensor<B, 2>,
    restrict_pred_to_mask: bool,
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
    let pred_presence = pred_intensity.ones_like();

    let target_pairs = target.reshape([batch_size, max_peaks, 2]);
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

    let mz_delta = (pred_mz - target_mz).abs();
    let sigma = config.normalized_mz_tolerance.max(1.0e-6);
    let scaled = mz_delta / sigma;
    let match_weights = (scaled.clone() * scaled * -0.5).exp() * target_mask.clone();
    let score_sum = (pred_products.clone() * target_products.clone() * match_weights).sum_dim(1);

    let pred_for_norm = if restrict_pred_to_mask {
        pred_products.clone() * target_mask.clone()
    } else {
        pred_products.clone()
    };
    let pred_norm = (pred_for_norm.powf_scalar(2.0).sum_dim(1) + 1.0e-6).sqrt();
    let target_norm = (target_products.powf_scalar(2.0).sum_dim(1) + 1.0e-6).sqrt();
    let similarity = (score_sum / (pred_norm * target_norm))
        .clamp_min(0.0)
        .clamp_max(1.0);
    let cosine_loss = similarity * -1.0 + 1.0;

    cosine_loss.mean()
}

/// Permutation-invariant "magnet" term on m/z: for each predicted slot,
/// the minimum squared distance to any real-peak target m/z. Pulls dead
/// peaks (predictions outside the cosine's Gaussian gate) back toward the
/// nearest real target, providing gradient where the gate has saturated.
///
/// Shape conventions match
/// [`slot_reconstruction_loss_from_vectors_with_mask`]: `reconstruction`
/// and `target` are `[batch, 2*P]`, `target_mask` is `[batch, P]`. Padded
/// target slots are excluded from the min via the
/// [`CHAMFER_INACTIVE_PENALTY`] constant. Assumes every row has at least
/// one real peak (`Σ_p M_p > 0`).
///
/// `max_target_peaks` randomly subsamples `k` target peaks (with
/// replacement, **independently per batch row**, resampled per forward
/// pass) so the broadcast tensor shrinks from `[batch, P_pred, P_target]`
/// to `[batch, P_pred, k]`. Per-row sampling means each anchor gets its
/// own draw of targets, so per-batch loss variance averages over `batch`
/// independent draws instead of one shared draw across the whole batch.
/// `0` disables subsampling and is bit-identical to the original kernel.
/// Values `>= max_peaks` are treated the same as `0`.
pub fn slot_chamfer_magnet_mz<B: Backend>(
    reconstruction: Tensor<B, 2>,
    target: Tensor<B, 2>,
    target_mask: Tensor<B, 2>,
    max_target_peaks: usize,
) -> Tensor<B, 1> {
    let [batch_size, vector_width] = reconstruction.dims();
    let max_peaks = vector_width / 2;
    debug_assert_eq!(vector_width % 2, 0);

    let pred_mz = reconstruction
        .reshape([batch_size, max_peaks, 2])
        .narrow(2, 0, 1)
        .reshape([batch_size, max_peaks]);
    let target_mz = target
        .reshape([batch_size, max_peaks, 2])
        .narrow(2, 0, 1)
        .reshape([batch_size, max_peaks]);

    let (target_mz, target_mask) = if max_target_peaks == 0 || max_target_peaks >= max_peaks {
        (target_mz, target_mask)
    } else {
        let device = target_mz.device();
        // Per-row indices: each batch row gets its own random k-subset of
        // target peaks. Shape [batch, k] so gather on dim 1 produces a
        // [batch, k] view of target_mz and target_mask.
        let indices = Tensor::<B, 2>::random(
            [batch_size, max_target_peaks],
            Distribution::Uniform(0.0, max_peaks as f64),
            &device,
        )
        .int();
        (
            target_mz.gather(1, indices.clone()),
            target_mask.gather(1, indices),
        )
    };

    // [batch, P_pred, P_target_effective]
    let diff = pred_mz.unsqueeze_dim::<3>(2) - target_mz.unsqueeze_dim::<3>(1);
    let diff_sq = diff.powf_scalar(2.0);

    // Padded targets get an additive penalty large enough to lose every min.
    let target_mask_3d = target_mask.unsqueeze_dim::<3>(1);
    let inactive_penalty =
        (target_mask_3d.clone().ones_like() - target_mask_3d) * CHAMFER_INACTIVE_PENALTY;
    let masked = diff_sq + inactive_penalty;

    let min_dist_sq = masked.min_dim(2).reshape([batch_size, max_peaks]);
    min_dist_sq.mean()
}

/// Flat-vector reconstruction loss with configurable slot alignment.
pub fn flat_vector_reconstruction_loss_from_vectors<B: Backend>(
    reconstruction: Tensor<B, 2>,
    target: Tensor<B, 2>,
    config: SetReconstructionLossConfig,
    ordering: FlatVectorReconstructionOrdering,
) -> Tensor<B, 1> {
    let target_mask = vector_target_mask(target.clone());
    flat_vector_reconstruction_loss_from_vectors_with_mask(
        reconstruction,
        target,
        target_mask,
        config,
        ordering,
    )
}

/// Flat-vector reconstruction loss with configurable slot alignment and explicit target mask.
pub fn flat_vector_reconstruction_loss_from_vectors_with_mask<B: Backend>(
    reconstruction: Tensor<B, 2>,
    target: Tensor<B, 2>,
    target_mask: Tensor<B, 2>,
    config: SetReconstructionLossConfig,
    ordering: FlatVectorReconstructionOrdering,
) -> Tensor<B, 1> {
    match ordering {
        FlatVectorReconstructionOrdering::Slot => slot_reconstruction_loss_from_vectors_with_mask(
            reconstruction,
            target,
            target_mask,
            config,
        ),
        FlatVectorReconstructionOrdering::IntensityDescending => {
            let (reconstruction, target, target_mask) =
                sort_flat_vectors_by_intensity(reconstruction, target, target_mask);
            slot_reconstruction_loss_from_vectors_with_mask(
                reconstruction,
                target,
                target_mask,
                config,
            )
        }
    }
}

/// Restricted-cosine variant of [`flat_vector_reconstruction_loss_from_vectors_with_mask`].
/// Uses the same slot-ordering dispatch and per-row reshape, but delegates to
/// [`slot_reconstruction_loss_restricted_from_vectors_with_mask`] at the end.
pub fn flat_vector_reconstruction_loss_restricted_from_vectors_with_mask<B: Backend>(
    reconstruction: Tensor<B, 2>,
    target: Tensor<B, 2>,
    target_mask: Tensor<B, 2>,
    config: SetReconstructionLossConfig,
    ordering: FlatVectorReconstructionOrdering,
) -> Tensor<B, 1> {
    match ordering {
        FlatVectorReconstructionOrdering::Slot => {
            slot_reconstruction_loss_restricted_from_vectors_with_mask(
                reconstruction,
                target,
                target_mask,
                config,
            )
        }
        FlatVectorReconstructionOrdering::IntensityDescending => {
            let (reconstruction, target, target_mask) =
                sort_flat_vectors_by_intensity(reconstruction, target, target_mask);
            slot_reconstruction_loss_restricted_from_vectors_with_mask(
                reconstruction,
                target,
                target_mask,
                config,
            )
        }
    }
}

/// Clean and masked flat-vector reconstruction losses with shared alignment work.
///
/// The clean branch uses the unrestricted slot cosine
/// ([`slot_reconstruction_loss_from_vectors_with_mask`]), which keeps the
/// phantom-peak penalty from `||π||₂` summing over every slot. The masked
/// branch uses the restricted variant
/// ([`slot_reconstruction_loss_restricted_from_vectors_with_mask`]) so the
/// cosine numerator and both norms are gated by the same (sparse) masked
/// target subset; this makes the masked signal meaningful when only a
/// handful of slots are real-and-dropped per row.
pub fn flat_vector_reconstruction_losses_from_vectors_with_masks<B: Backend>(
    reconstruction: Tensor<B, 2>,
    target: Tensor<B, 2>,
    target_mask: Tensor<B, 2>,
    masked_target_mask: Tensor<B, 2>,
    config: SetReconstructionLossConfig,
    ordering: FlatVectorReconstructionOrdering,
) -> (Tensor<B, 1>, Tensor<B, 1>) {
    match ordering {
        FlatVectorReconstructionOrdering::Slot => (
            slot_reconstruction_loss_from_vectors_with_mask(
                reconstruction.clone(),
                target.clone(),
                target_mask,
                config,
            ),
            slot_reconstruction_loss_restricted_from_vectors_with_mask(
                reconstruction,
                target,
                masked_target_mask,
                config,
            ),
        ),
        FlatVectorReconstructionOrdering::IntensityDescending => {
            let (reconstruction, target, target_mask, masked_target_mask) =
                sort_flat_vectors_by_intensity_with_two_masks(
                    reconstruction,
                    target,
                    target_mask,
                    masked_target_mask,
                );
            (
                slot_reconstruction_loss_from_vectors_with_mask(
                    reconstruction.clone(),
                    target.clone(),
                    target_mask,
                    config,
                ),
                slot_reconstruction_loss_restricted_from_vectors_with_mask(
                    reconstruction,
                    target,
                    masked_target_mask,
                    config,
                ),
            )
        }
    }
}

/// Backward-compatible set reconstruction loss for flat vectors shaped as `[batch, peaks * 2]`.
pub fn set_reconstruction_loss_from_vectors<B: Backend>(
    reconstruction: Tensor<B, 2>,
    target: Tensor<B, 2>,
    config: SetReconstructionLossConfig,
) -> Tensor<B, 1> {
    let target_mask = vector_target_mask(target.clone());
    set_reconstruction_loss_from_vectors_with_mask(reconstruction, target, target_mask, config)
}

/// Backward-compatible set reconstruction loss for flat vectors with an explicit target peak mask.
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
    let pred_presence = pred_intensity.ones_like();

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

fn sort_flat_vectors_by_intensity<B: Backend>(
    reconstruction: Tensor<B, 2>,
    target: Tensor<B, 2>,
    target_mask: Tensor<B, 2>,
) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
    let [batch_size, vector_width] = reconstruction.dims();
    let max_peaks = vector_width / 2;
    debug_assert_eq!(vector_width % 2, 0);

    let predicted_pairs = reconstruction.reshape([batch_size, max_peaks, 2]);
    let target_pairs = target.reshape([batch_size, max_peaks, 2]);
    let predicted_intensity = predicted_pairs
        .clone()
        .narrow(2, 1, 1)
        .reshape([batch_size, max_peaks]);
    let target_intensity = target_pairs
        .clone()
        .narrow(2, 1, 1)
        .reshape([batch_size, max_peaks]);
    let predicted_order = predicted_intensity.argsort_descending(1);
    let target_order = target_intensity.argsort_descending(1);
    let sorted_reconstruction =
        gather_peak_pairs(predicted_pairs, predicted_order, batch_size, max_peaks)
            .reshape([batch_size, vector_width]);
    let sorted_target =
        gather_peak_pairs(target_pairs, target_order.clone(), batch_size, max_peaks)
            .reshape([batch_size, vector_width]);
    let sorted_target_mask = target_mask.gather(1, target_order);

    (sorted_reconstruction, sorted_target, sorted_target_mask)
}

fn sort_flat_vectors_by_intensity_with_two_masks<B: Backend>(
    reconstruction: Tensor<B, 2>,
    target: Tensor<B, 2>,
    target_mask: Tensor<B, 2>,
    masked_target_mask: Tensor<B, 2>,
) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
    let [batch_size, vector_width] = reconstruction.dims();
    let max_peaks = vector_width / 2;
    debug_assert_eq!(vector_width % 2, 0);

    let predicted_pairs = reconstruction.reshape([batch_size, max_peaks, 2]);
    let target_pairs = target.reshape([batch_size, max_peaks, 2]);
    let predicted_intensity = predicted_pairs
        .clone()
        .narrow(2, 1, 1)
        .reshape([batch_size, max_peaks]);
    let target_intensity = target_pairs
        .clone()
        .narrow(2, 1, 1)
        .reshape([batch_size, max_peaks]);
    let predicted_order = predicted_intensity.argsort_descending(1);
    let target_order = target_intensity.argsort_descending(1);
    let sorted_reconstruction =
        gather_peak_pairs(predicted_pairs, predicted_order, batch_size, max_peaks)
            .reshape([batch_size, vector_width]);
    let sorted_target =
        gather_peak_pairs(target_pairs, target_order.clone(), batch_size, max_peaks)
            .reshape([batch_size, vector_width]);
    let sorted_target_mask = target_mask.gather(1, target_order.clone());
    let sorted_masked_target_mask = masked_target_mask.gather(1, target_order);

    (
        sorted_reconstruction,
        sorted_target,
        sorted_target_mask,
        sorted_masked_target_mask,
    )
}

fn gather_peak_pairs<B: Backend>(
    pairs: Tensor<B, 3>,
    order: Tensor<B, 2, Int>,
    batch_size: usize,
    max_peaks: usize,
) -> Tensor<B, 3> {
    let indices = order
        .unsqueeze_dim::<3>(2)
        .expand([batch_size, max_peaks, 2]);
    pairs.gather(1, indices)
}

pub(crate) fn peak_products<B: Backend>(
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
        // The 2-wide vector path implicitly treats `pred_presence = 1` on every
        // slot, so the equivalent 3-wide triple form must set the presence
        // channel to 1.0 (not to the slot's intensity).
        let triples = Tensor::<B, 3>::from_floats([[[0.1, 1.0, 1.0], [0.2, 0.5, 1.0]]], &device);
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
    fn slot_vector_loss_penalizes_swapped_peak_order() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let config = SetReconstructionLossConfig {
            count_weight: 0.0,
            ..SetReconstructionLossConfig::default()
        };
        let target = Tensor::<B, 2>::from_floats([[0.1, 1.0, 0.9, 1.0]], &device);
        let exact = Tensor::<B, 2>::from_floats([[0.1, 1.0, 0.9, 1.0]], &device);
        let swapped = Tensor::<B, 2>::from_floats([[0.9, 1.0, 0.1, 1.0]], &device);

        let exact_loss =
            slot_reconstruction_loss_from_vectors(exact, target.clone(), config).into_scalar();
        let swapped_loss =
            slot_reconstruction_loss_from_vectors(swapped, target, config).into_scalar();

        assert!(exact_loss < swapped_loss);
        assert!(swapped_loss > 0.5);
    }

    #[test]
    fn intensity_ordered_vector_loss_matches_swapped_peak_slots() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let config = SetReconstructionLossConfig {
            count_weight: 0.0,
            ..SetReconstructionLossConfig::default()
        };
        let target = Tensor::<B, 2>::from_floats([[0.1, 1.0, 0.9, 0.5]], &device);
        let exact = Tensor::<B, 2>::from_floats([[0.1, 1.0, 0.9, 0.5]], &device);
        let swapped = Tensor::<B, 2>::from_floats([[0.9, 0.5, 0.1, 1.0]], &device);

        let exact_loss =
            slot_reconstruction_loss_from_vectors(exact, target.clone(), config).into_scalar();
        let sorted_loss = flat_vector_reconstruction_loss_from_vectors(
            swapped,
            target,
            config,
            FlatVectorReconstructionOrdering::IntensityDescending,
        )
        .into_scalar();

        assert!((exact_loss - sorted_loss).abs() < 1.0e-6);
    }

    #[test]
    fn slot_vector_loss_ignores_padded_mz_slots() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let config = SetReconstructionLossConfig::default();
        let target = Tensor::<B, 2>::from_floats([[0.1, 1.0, 0.7, 0.0]], &device);
        let target_mask = Tensor::<B, 2>::from_floats([[1.0, 0.0]], &device);
        let padded_a = Tensor::<B, 2>::from_floats([[0.1, 1.0, 0.0, 0.0]], &device);
        let padded_b = Tensor::<B, 2>::from_floats([[0.1, 1.0, 1.0, 0.0]], &device);

        let loss_a = slot_reconstruction_loss_from_vectors_with_mask(
            padded_a,
            target.clone(),
            target_mask.clone(),
            config,
        )
        .into_scalar();
        let loss_b =
            slot_reconstruction_loss_from_vectors_with_mask(padded_b, target, target_mask, config)
                .into_scalar();

        assert!((loss_a - loss_b).abs() < 1.0e-6);
    }

    #[test]
    fn intensity_ordered_vector_loss_ignores_padded_mz_slots() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let config = SetReconstructionLossConfig::default();
        let target = Tensor::<B, 2>::from_floats([[0.1, 1.0, 0.7, 0.0]], &device);
        let target_mask = Tensor::<B, 2>::from_floats([[1.0, 0.0]], &device);
        let padded_a = Tensor::<B, 2>::from_floats([[0.1, 1.0, 0.0, 0.0]], &device);
        let padded_b = Tensor::<B, 2>::from_floats([[0.1, 1.0, 1.0, 0.0]], &device);

        let loss_a = flat_vector_reconstruction_loss_from_vectors_with_mask(
            padded_a,
            target.clone(),
            target_mask.clone(),
            config,
            FlatVectorReconstructionOrdering::IntensityDescending,
        )
        .into_scalar();
        let loss_b = flat_vector_reconstruction_loss_from_vectors_with_mask(
            padded_b,
            target,
            target_mask,
            config,
            FlatVectorReconstructionOrdering::IntensityDescending,
        )
        .into_scalar();

        assert!((loss_a - loss_b).abs() < 1.0e-6);
    }

    #[test]
    fn paired_intensity_ordered_losses_match_separate_calls() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let config = SetReconstructionLossConfig::default();
        let reconstruction =
            Tensor::<B, 2>::from_floats([[0.8, 0.4, 0.2, 1.0, 0.5, 0.6, 0.0, 0.0]], &device);
        let target =
            Tensor::<B, 2>::from_floats([[0.2, 1.0, 0.5, 0.6, 0.8, 0.4, 0.0, 0.0]], &device);
        let target_mask = vector_target_mask(target.clone());
        let masked_target_mask = Tensor::<B, 2>::from_floats([[1.0, 0.0, 1.0, 0.0]], &device);

        let clean_separate = flat_vector_reconstruction_loss_from_vectors_with_mask(
            reconstruction.clone(),
            target.clone(),
            target_mask.clone(),
            config,
            FlatVectorReconstructionOrdering::IntensityDescending,
        )
        .into_scalar();
        let masked_separate = flat_vector_reconstruction_loss_restricted_from_vectors_with_mask(
            reconstruction.clone(),
            target.clone(),
            masked_target_mask.clone(),
            config,
            FlatVectorReconstructionOrdering::IntensityDescending,
        )
        .into_scalar();
        let (clean_paired, masked_paired) =
            flat_vector_reconstruction_losses_from_vectors_with_masks(
                reconstruction,
                target,
                target_mask,
                masked_target_mask,
                config,
                FlatVectorReconstructionOrdering::IntensityDescending,
            );

        assert!((clean_separate - clean_paired.into_scalar()).abs() < 1.0e-6);
        assert!((masked_separate - masked_paired.into_scalar()).abs() < 1.0e-6);
    }

    #[test]
    fn vector_masks_are_peak_level() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let mask = Tensor::<B, 2>::from_floats([[1.0, 0.0, 0.0, 1.0]], &device);

        let peak_mask = vector_element_mask_to_peak_mask(mask).into_data();

        assert_eq!(peak_mask.as_slice::<f32>().expect("f32 data"), &[1.0, 1.0]);
    }

    #[test]
    fn chamfer_magnet_is_zero_on_perfect_prediction() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        // Two real peaks at m/z 0.1 and 0.7, intensities arbitrary.
        let target = Tensor::<B, 2>::from_floats([[0.1, 1.0, 0.7, 0.5]], &device);
        let pred = target.clone();
        let target_mask = Tensor::<B, 2>::from_floats([[1.0, 1.0]], &device);

        let value: f32 = slot_chamfer_magnet_mz(pred, target, target_mask, 0).into_scalar();
        assert!(value.abs() < 1.0e-6, "magnet={value}, expected ~0");
    }

    #[test]
    fn chamfer_magnet_is_permutation_invariant_over_targets() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        // Same predictions, same target *set* but slots permuted.
        let pred = Tensor::<B, 2>::from_floats([[0.20, 1.0, 0.65, 0.5]], &device);
        let target_ab = Tensor::<B, 2>::from_floats([[0.10, 1.0, 0.70, 0.5]], &device);
        let target_ba = Tensor::<B, 2>::from_floats([[0.70, 0.5, 0.10, 1.0]], &device);
        let mask = Tensor::<B, 2>::from_floats([[1.0, 1.0]], &device);

        let v_ab: f32 =
            slot_chamfer_magnet_mz(pred.clone(), target_ab, mask.clone(), 0).into_scalar();
        let v_ba: f32 = slot_chamfer_magnet_mz(pred, target_ba, mask, 0).into_scalar();
        assert!((v_ab - v_ba).abs() < 1.0e-6, "{v_ab} vs {v_ba}");
    }

    #[test]
    fn chamfer_magnet_picks_nearest_real_target() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        // One pred at m/z 0.2. Real targets at 0.1 and 0.7; padded slot at 0.21
        // (would be closer if it counted but mask excludes it).
        let pred = Tensor::<B, 2>::from_floats([[0.20, 1.0, 0.20, 0.0, 0.20, 0.0]], &device);
        let target = Tensor::<B, 2>::from_floats([[0.10, 1.0, 0.70, 0.5, 0.21, 0.0]], &device);
        let target_mask = Tensor::<B, 2>::from_floats([[1.0, 1.0, 0.0]], &device);

        let value: f32 = slot_chamfer_magnet_mz(pred, target, target_mask, 0).into_scalar();
        // Each of three pred slots maps to nearest real target = 0.1, distance 0.01.
        let expected = ((0.20_f32 - 0.10_f32).powi(2) * 3.0) / 3.0;
        assert!(
            (value - expected).abs() < 1.0e-5,
            "value={value}, expected={expected}"
        );
    }

    #[test]
    fn chamfer_magnet_ignores_padded_targets_via_big_penalty() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        // Pred at 0.5. Only real target at 0.0; padded "near-target" at 0.5001.
        let pred = Tensor::<B, 2>::from_floats([[0.5, 1.0, 0.5, 0.0]], &device);
        let target = Tensor::<B, 2>::from_floats([[0.0, 1.0, 0.5001, 0.0]], &device);
        let target_mask = Tensor::<B, 2>::from_floats([[1.0, 0.0]], &device);

        let value: f32 = slot_chamfer_magnet_mz(pred, target, target_mask, 0).into_scalar();
        // Both pred slots map to the real target at 0.0; distance (0.5)^2 = 0.25 each.
        let expected = 0.25_f32;
        assert!(
            (value - expected).abs() < 1.0e-5,
            "value={value}, expected={expected}"
        );
    }

    #[test]
    fn chamfer_subsample_with_k_at_or_above_target_matches_unsubsampled() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        // Three real-peak targets at 0.1, 0.4, 0.7; pred slots at 0.15, 0.45, 0.65.
        let pred = Tensor::<B, 2>::from_floats([[0.15, 1.0, 0.45, 0.5, 0.65, 0.25]], &device);
        let target = Tensor::<B, 2>::from_floats([[0.10, 1.0, 0.40, 0.5, 0.70, 0.25]], &device);
        let mask = Tensor::<B, 2>::from_floats([[1.0, 1.0, 1.0]], &device);

        let unsub: f32 =
            slot_chamfer_magnet_mz(pred.clone(), target.clone(), mask.clone(), 0).into_scalar();
        let at_k: f32 =
            slot_chamfer_magnet_mz(pred.clone(), target.clone(), mask.clone(), 3).into_scalar();
        let above_k: f32 = slot_chamfer_magnet_mz(pred, target, mask, 100).into_scalar();
        assert!(
            unsub.is_finite() && unsub > 0.0,
            "baseline finite, got {unsub}"
        );
        assert_eq!(unsub, at_k, "k = P_target should short-circuit identically");
        assert_eq!(
            unsub, above_k,
            "k > P_target should short-circuit identically"
        );
    }

    #[test]
    fn chamfer_subsample_produces_finite_non_negative_output_with_k_less_than_target() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        // 8 target peaks, 8 pred slots. k = 2 → subsample to 2 random targets.
        let pred = Tensor::<B, 2>::from_floats(
            [[
                0.05, 1.0, 0.15, 1.0, 0.25, 1.0, 0.35, 1.0, 0.45, 1.0, 0.55, 1.0, 0.65, 1.0, 0.75,
                1.0,
            ]],
            &device,
        );
        let target = Tensor::<B, 2>::from_floats(
            [[
                0.10, 1.0, 0.20, 1.0, 0.30, 1.0, 0.40, 1.0, 0.50, 1.0, 0.60, 1.0, 0.70, 1.0, 0.80,
                1.0,
            ]],
            &device,
        );
        let mask = Tensor::<B, 2>::from_floats([[1.0; 8]], &device);

        let value: f32 = slot_chamfer_magnet_mz(pred, target, mask, 2).into_scalar();
        assert!(
            value.is_finite(),
            "subsampled value should be finite, got {value}"
        );
        assert!(
            value >= 0.0,
            "subsampled value should be non-negative, got {value}"
        );
    }

    #[test]
    fn restricted_cosine_is_unaffected_by_phantom_intensity_off_mask() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let config = SetReconstructionLossConfig::default();
        // One real-peak slot at (0.10, 1.0), one padded slot.
        let target = Tensor::<B, 2>::from_floats([[0.10, 1.0, 0.0, 0.0]], &device);
        let mask = Tensor::<B, 2>::from_floats([[1.0, 0.0]], &device);
        // Two predictions: identical on the masked-real slot, different in the
        // phantom intensity placed on the padded slot.
        let pred_clean = Tensor::<B, 2>::from_floats([[0.10, 1.0, 0.0, 0.0]], &device);
        let pred_noisy = Tensor::<B, 2>::from_floats([[0.10, 1.0, 0.5, 0.5]], &device);

        let restricted_clean = slot_reconstruction_loss_restricted_from_vectors_with_mask(
            pred_clean.clone(),
            target.clone(),
            mask.clone(),
            config,
        )
        .into_scalar();
        let restricted_noisy = slot_reconstruction_loss_restricted_from_vectors_with_mask(
            pred_noisy.clone(),
            target.clone(),
            mask.clone(),
            config,
        )
        .into_scalar();
        // Restricted variant gates pred_norm by the mask, so the phantom on the
        // padded slot must not change the loss.
        assert!(
            (restricted_clean - restricted_noisy).abs() < 1.0e-6,
            "restricted_clean={restricted_clean} restricted_noisy={restricted_noisy}"
        );

        // Sanity: the unrestricted variant does react to phantom intensity.
        let unrestricted_clean = slot_reconstruction_loss_from_vectors_with_mask(
            pred_clean,
            target.clone(),
            mask.clone(),
            config,
        )
        .into_scalar();
        let unrestricted_noisy =
            slot_reconstruction_loss_from_vectors_with_mask(pred_noisy, target, mask, config)
                .into_scalar();
        assert!(
            unrestricted_noisy > unrestricted_clean + 1.0e-3,
            "unrestricted should penalise phantom: clean={unrestricted_clean} noisy={unrestricted_noisy}"
        );
    }

    #[test]
    fn cosine_reaches_one_at_perfect_prediction_with_varied_intensities() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        // Two real peaks with non-uniform intensities. Under the previous
        // asymmetric-exponent form, sim plateaued around 0.96 here. With the
        // pred_presence-as-ones fix, sim must reach 1 at pred == target.
        let target = Tensor::<B, 2>::from_floats([[0.10, 1.0, 0.20, 0.5]], &device);
        let pred = target.clone();
        let mask = Tensor::<B, 2>::from_floats([[1.0, 1.0]], &device);

        let loss = slot_reconstruction_loss_from_vectors_with_mask(
            pred,
            target,
            mask,
            SetReconstructionLossConfig::default(),
        )
        .into_scalar();
        assert!(
            loss < 1.0e-3,
            "loss at perfect prediction should be ~0, got {loss}"
        );
    }

    #[test]
    fn count_weight_no_longer_affects_slot_reconstruction_loss() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let reconstruction = Tensor::<B, 2>::from_floats([[0.1, 1.0, 0.0, 0.0]], &device);
        let target = Tensor::<B, 2>::from_floats([[0.1, 1.0, 0.0, 0.0]], &device);
        let target_mask = Tensor::<B, 2>::from_floats([[1.0, 0.0]], &device);

        let no_count = SetReconstructionLossConfig {
            count_weight: 0.0,
            ..SetReconstructionLossConfig::default()
        };
        let high_count = SetReconstructionLossConfig {
            count_weight: 100.0,
            ..SetReconstructionLossConfig::default()
        };

        let loss_zero = slot_reconstruction_loss_from_vectors_with_mask(
            reconstruction.clone(),
            target.clone(),
            target_mask.clone(),
            no_count,
        )
        .into_scalar();
        let loss_high = slot_reconstruction_loss_from_vectors_with_mask(
            reconstruction,
            target,
            target_mask,
            high_count,
        )
        .into_scalar();

        // count_weight is no longer consulted by the slot loss.
        assert!(
            (loss_zero - loss_high).abs() < 1.0e-6,
            "loss_zero={loss_zero}, loss_high={loss_high}"
        );
    }
}
