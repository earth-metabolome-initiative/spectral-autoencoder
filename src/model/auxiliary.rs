//! Auxiliary pretraining losses over spectral embeddings.

use burn::{
    module::{Initializer, Param},
    nn::{Linear, LinearConfig, Relu},
    prelude::*,
    tensor::{Distribution, Int, Tensor, activation::sigmoid},
};
use serde::{Deserialize, Serialize};

/// Auxiliary objective weights for embedding-quality pretraining.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AuxiliaryLossConfig {
    /// Weight for the main clean reconstruction objective.
    pub reconstruction_weight: f64,
    /// Extra weight on reconstructing masked or dropped spectral inputs.
    pub masked_peak_weight: f64,
    /// Weight for keeping two augmented views close in latent space.
    pub consistency_weight: f64,
    /// Weight for retention-order prediction within the same source file.
    pub retention_order_weight: f64,
    /// Weight for detecting synthetic intruder peaks inserted into input spectra.
    pub intruder_peak_weight: f64,
    /// Gaussian decoder-input latent noise as a fraction of the batch latent standard deviation.
    ///
    /// This is applied only during training, after the encoder and before the decoder.
    /// A value of `0.0` disables latent denoising.
    pub latent_noise_std: f64,
    /// Maximum same-file retention pairs used per batch; `0` means one partner per row.
    pub retention_pairs_per_batch: usize,
    /// Hidden width of the retention-order auxiliary head.
    pub retention_hidden_width: usize,
    /// Hidden width of the per-slot intruder detection head.
    pub intruder_hidden_width: usize,
}

impl Default for AuxiliaryLossConfig {
    fn default() -> Self {
        Self {
            reconstruction_weight: 1.0,
            masked_peak_weight: 0.25,
            consistency_weight: 0.05,
            retention_order_weight: 0.2,
            intruder_peak_weight: 0.05,
            latent_noise_std: 0.02,
            retention_pairs_per_batch: 0,
            retention_hidden_width: 128,
            intruder_hidden_width: 128,
        }
    }
}

/// Configuration for small embedding-level auxiliary heads.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EmbeddingAuxiliaryHeadsConfig {
    /// Latent embedding width.
    pub latent_width: usize,
    /// Maximum number of peak slots in the model input.
    pub max_peaks: usize,
    /// Hidden width of the retention-order scalar rank head.
    pub retention_hidden_width: usize,
    /// Hidden width of the per-slot intruder detection MLP.
    pub intruder_hidden_width: usize,
}

impl EmbeddingAuxiliaryHeadsConfig {
    /// Creates initialized auxiliary heads.
    pub fn init<B: Backend>(&self, device: &B::Device) -> EmbeddingAuxiliaryHeads<B> {
        EmbeddingAuxiliaryHeads {
            retention_input: LinearConfig::new(self.latent_width, self.retention_hidden_width)
                .init(device),
            retention_output: LinearConfig::new(self.retention_hidden_width, 1).init(device),
            intruder_input: LinearConfig::new(self.latent_width, self.intruder_hidden_width)
                .init(device),
            intruder_slots: Initializer::Normal {
                mean: 0.0,
                std: 0.02,
            }
            .init([self.max_peaks, self.intruder_hidden_width], device),
            intruder_output: LinearConfig::new(self.intruder_hidden_width, 1).init(device),
            activation: Relu::new(),
            max_peaks: self.max_peaks,
            intruder_hidden_width: self.intruder_hidden_width,
        }
    }
}

/// Small heads that operate on encoder embeddings.
#[derive(Module, Debug)]
pub struct EmbeddingAuxiliaryHeads<B: Backend> {
    retention_input: Linear<B>,
    retention_output: Linear<B>,
    intruder_input: Linear<B>,
    intruder_slots: Param<Tensor<B, 2>>,
    intruder_output: Linear<B>,
    activation: Relu,
    max_peaks: usize,
    intruder_hidden_width: usize,
}

impl<B: Backend> EmbeddingAuxiliaryHeads<B> {
    /// Predicts a scalar retention rank for one embedding.
    pub fn retention_scores(&self, latent: Tensor<B, 2>) -> Tensor<B, 2> {
        self.retention_output.forward(
            self.activation
                .forward(self.retention_input.forward(latent)),
        )
    }

    /// Predicts whether the right embedding elutes after the left embedding.
    ///
    /// The pair logit is explicitly antisymmetric: `logit(left, right) =
    /// -logit(right, left)`. This prevents the retention objective from
    /// collapsing into a symmetric pair classifier that scores each reversed
    /// pair identically and sits at exactly 50% accuracy.
    pub fn retention_logits(&self, left: Tensor<B, 2>, right: Tensor<B, 2>) -> Tensor<B, 2> {
        self.retention_scores(right) - self.retention_scores(left)
    }

    /// Predicts one intruder logit per fixed input peak slot.
    pub fn intruder_logits(&self, latent: Tensor<B, 2>) -> Tensor<B, 2> {
        let [batch_size, _latent_width] = latent.dims();
        let latent_features = self
            .activation
            .forward(self.intruder_input.forward(latent))
            .unsqueeze_dim::<3>(1)
            .expand([batch_size, self.max_peaks, self.intruder_hidden_width]);
        let slot_features = self.intruder_slots.val().unsqueeze_dim::<3>(0).expand([
            batch_size,
            self.max_peaks,
            self.intruder_hidden_width,
        ]);
        self.intruder_output
            .forward(self.activation.forward(latent_features + slot_features))
            .reshape([batch_size, self.max_peaks])
    }
}

/// Retention-order objective result and diagnostics.
pub struct RetentionOrderOutput<B: Backend> {
    /// Weighted or unweighted retention-order loss, depending on caller.
    pub loss: Tensor<B, 1>,
    /// Unweighted binary cross-entropy retention-order loss.
    pub raw_loss: Tensor<B, 1>,
    /// Number of valid same-file retention pairs before reverse augmentation.
    pub valid_pairs: Tensor<B, 1>,
    /// Binary pair-order accuracy over valid retention comparisons.
    pub accuracy: Tensor<B, 1>,
    /// Standard deviation of retention-order logits over valid comparisons.
    pub logit_std: Tensor<B, 1>,
    /// Mean absolute retention-time gap over valid same-file pairs.
    pub mean_rt_delta: Tensor<B, 1>,
}

/// Batch fields used to build same-file retention-order pairs.
pub struct RetentionOrderBatch<B: Backend> {
    /// Retention time tensor with shape `[batch, 1]`; value is zero when absent.
    pub retention_time: Tensor<B, 2>,
    /// Retention-time presence flag with shape `[batch, 1]`.
    pub retention_present: Tensor<B, 2>,
    /// Source-file identifier with shape `[batch, 1]`; value is `-1` when absent.
    pub filename_id: Tensor<B, 2>,
    /// Partner row index for same-file retention-order pairs.
    pub partner_index: Tensor<B, 1, Int>,
}

/// Cosine distance between two augmented views of the same embeddings.
pub fn cosine_distance_loss<B: Backend>(left: Tensor<B, 2>, right: Tensor<B, 2>) -> Tensor<B, 1> {
    let numerator = (left.clone() * right.clone()).sum_dim(1);
    let left_norm = (left.powf_scalar(2.0).sum_dim(1) + 1.0e-6).sqrt();
    let right_norm = (right.powf_scalar(2.0).sum_dim(1) + 1.0e-6).sqrt();
    let similarity = (numerator / (left_norm * right_norm))
        .clamp_min(-1.0)
        .clamp_max(1.0);
    (similarity * -1.0 + 1.0).mean()
}

/// Applies mild Gaussian denoising noise to the decoder-side latent input.
///
/// The noise scale is relative to the current batch latent standard deviation,
/// and the scale is detached so the encoder is not rewarded for inflating or
/// shrinking latent variance just to control the injected noise.
pub fn apply_latent_noise<B: Backend>(latent: Tensor<B, 2>, std_fraction: f64) -> Tensor<B, 2> {
    if std_fraction <= 0.0 {
        return latent;
    }

    let [batch_size, latent_width] = latent.dims();
    let device = latent.device();
    let mean = latent
        .clone()
        .mean_dim(0)
        .expand([batch_size, latent_width]);
    let centered = latent.clone() - mean;
    let scale = (centered.powf_scalar(2.0).mean() + 1.0e-6)
        .sqrt()
        .detach()
        .reshape([1, 1])
        .expand([batch_size, latent_width]);
    let noise = Tensor::<B, 2>::random(
        [batch_size, latent_width],
        Distribution::Normal(0.0, std_fraction),
        &device,
    );

    latent + noise * scale
}

/// Weighted cosine distance, or zero when the weight is disabled.
pub fn weighted_cosine_distance_loss<B: Backend, F>(
    left: Tensor<B, 2>,
    right: F,
    weight: f64,
    device: &B::Device,
) -> Tensor<B, 1>
where
    F: FnOnce() -> Tensor<B, 2>,
{
    if weight > 0.0 {
        cosine_distance_loss(left, right()) * weight
    } else {
        Tensor::zeros([1], device)
    }
}

/// Binary retention-order objective over same-file partner pairs and their reverses.
pub fn retention_order_output<B: Backend>(
    heads: &EmbeddingAuxiliaryHeads<B>,
    latent: Tensor<B, 2>,
    batch: RetentionOrderBatch<B>,
    max_pairs: usize,
) -> RetentionOrderOutput<B> {
    let [batch_size, _latent_width] = latent.dims();
    let pair_count = if max_pairs == 0 {
        batch_size
    } else {
        batch_size.min(max_pairs)
    };
    if pair_count == 0 {
        return zero_retention_order_output(&latent.device());
    }

    let partner_index = batch.partner_index.narrow(0, 0, pair_count);
    let left_base = latent.clone().narrow(0, 0, pair_count);
    let right_base = latent.clone().select(0, partner_index.clone());
    let rt_left = batch.retention_time.clone().narrow(0, 0, pair_count);
    let rt_right = batch
        .retention_time
        .clone()
        .select(0, partner_index.clone());
    let present_left = batch.retention_present.clone().narrow(0, 0, pair_count);
    let present_right = batch
        .retention_present
        .clone()
        .select(0, partner_index.clone());
    let file_left = batch.filename_id.clone().narrow(0, 0, pair_count);
    let file_right = batch.filename_id.select(0, partner_index);

    let same_file = file_left
        .clone()
        .equal(file_right.clone())
        .bool_and(file_left.greater_equal_elem(0.0))
        .bool_and(file_right.greater_equal_elem(0.0));
    let valid_rt = present_left
        .greater_elem(0.0)
        .bool_and(present_right.greater_elem(0.0));
    let rt_delta = rt_right.clone() - rt_left.clone();
    let non_tie = rt_delta.clone().abs().greater_elem(1.0e-6);
    let pair_mask = same_file.bool_and(valid_rt).bool_and(non_tie).float();
    let valid_pairs = pair_mask.clone().sum();
    let forward_labels = rt_delta.clone().greater_elem(0.0).float();

    let left = Tensor::cat(vec![left_base.clone(), right_base.clone()], 0);
    let right = Tensor::cat(vec![right_base, left_base], 0);
    let reverse_labels = forward_labels.ones_like() - forward_labels.clone();
    let labels = Tensor::cat(vec![forward_labels, reverse_labels], 0);
    let mask = Tensor::cat(vec![pair_mask.clone(), pair_mask.clone()], 0);

    let logits = heads.retention_logits(left, right);
    let probabilities = sigmoid(logits.clone())
        .clamp_min(1.0e-6)
        .clamp_max(1.0 - 1.0e-6);
    let valid_comparisons = mask.clone().sum().clamp_min(1.0);
    let mean_logit = (logits.clone() * mask.clone()).sum() / valid_comparisons.clone();
    let [logit_count, logit_width] = logits.dims();
    let mean_logit = mean_logit
        .reshape([1, 1])
        .expand([logit_count, logit_width]);
    let logit_delta = logits - mean_logit;
    let logit_std =
        ((logit_delta.clone() * logit_delta * mask.clone()).sum() / valid_comparisons).sqrt();
    let predictions = probabilities.clone().greater_elem(0.5).float();
    let accuracy = ((predictions - labels.clone()).abs().lower_elem(0.5).float() * mask.clone())
        .sum()
        / mask.clone().sum().clamp_min(1.0);
    let inverse_labels = labels.ones_like() - labels.clone();
    let inverse_probabilities = probabilities.ones_like() - probabilities.clone();
    let bce = (labels * probabilities.log() + inverse_labels * inverse_probabilities.log()) * -1.0;
    let loss = (bce * mask.clone()).sum() / mask.sum().clamp_min(1.0);
    let mean_rt_delta =
        (rt_delta.clone().abs() * pair_mask.clone()).sum() / valid_pairs.clone().clamp_min(1.0);

    RetentionOrderOutput {
        loss: loss.clone(),
        raw_loss: loss,
        valid_pairs,
        accuracy,
        logit_std,
        mean_rt_delta,
    }
}

/// Binary retention-order loss over same-file partner pairs and their reverses.
pub fn retention_order_loss<B: Backend>(
    heads: &EmbeddingAuxiliaryHeads<B>,
    latent: Tensor<B, 2>,
    batch: RetentionOrderBatch<B>,
    max_pairs: usize,
) -> Tensor<B, 1> {
    retention_order_output(heads, latent, batch, max_pairs).loss
}

/// Weighted retention-order loss, or zero when the weight is disabled.
pub fn weighted_retention_order_loss<B: Backend>(
    heads: &EmbeddingAuxiliaryHeads<B>,
    latent: Tensor<B, 2>,
    batch: RetentionOrderBatch<B>,
    max_pairs: usize,
    weight: f64,
) -> Tensor<B, 1> {
    weighted_retention_order_output(heads, latent, batch, max_pairs, weight).loss
}

/// Weighted retention-order output, or zero diagnostics when the weight is disabled.
pub fn weighted_retention_order_output<B: Backend>(
    heads: &EmbeddingAuxiliaryHeads<B>,
    latent: Tensor<B, 2>,
    batch: RetentionOrderBatch<B>,
    max_pairs: usize,
    weight: f64,
) -> RetentionOrderOutput<B> {
    if weight > 0.0 {
        let mut output = retention_order_output(heads, latent, batch, max_pairs);
        output.loss = output.loss * weight;
        output
    } else {
        zero_retention_order_output(&latent.device())
    }
}

fn zero_retention_order_output<B: Backend>(device: &B::Device) -> RetentionOrderOutput<B> {
    RetentionOrderOutput {
        loss: Tensor::zeros([1], device),
        raw_loss: Tensor::zeros([1], device),
        valid_pairs: Tensor::zeros([1], device),
        accuracy: Tensor::zeros([1], device),
        logit_std: Tensor::zeros([1], device),
        mean_rt_delta: Tensor::zeros([1], device),
    }
}

/// Binary intruder detection loss over active input peak slots.
pub fn intruder_detection_loss<B: Backend>(
    heads: &EmbeddingAuxiliaryHeads<B>,
    latent: Tensor<B, 2>,
    intruder_peak_mask: Tensor<B, 2>,
    active_peak_mask: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let [batch_size, max_peaks] = intruder_peak_mask.dims();
    let probabilities = sigmoid(heads.intruder_logits(latent))
        .clamp_min(1.0e-6)
        .clamp_max(1.0 - 1.0e-6);
    let labels = intruder_peak_mask * active_peak_mask.clone();
    let mask = active_peak_mask.greater_elem(0.0).float();
    let positives = labels.clone().sum();
    let negatives = ((labels.ones_like() - labels.clone()) * mask.clone()).sum();
    let positive_weight = (negatives / (positives.clone() + 1.0e-6))
        .clamp_min(1.0)
        .clamp_max(60.0)
        .reshape([1, 1])
        .expand([batch_size, max_peaks]);
    let inverse_labels = labels.ones_like() - labels.clone();
    let inverse_probabilities = probabilities.ones_like() - probabilities.clone();
    let class_weights = inverse_labels.clone() + labels.clone() * positive_weight;
    let bce = (labels * probabilities.log() + inverse_labels * inverse_probabilities.log()) * -1.0;
    let weights = mask * class_weights;
    let has_intruders = positives.greater_elem(0.0).float();
    (bce * weights.clone()).sum() / weights.sum().clamp_min(1.0) * has_intruders
}

/// Weighted intruder detection loss, or zero when the weight is disabled.
pub fn weighted_intruder_detection_loss<B: Backend>(
    heads: &EmbeddingAuxiliaryHeads<B>,
    latent: Tensor<B, 2>,
    intruder_peak_mask: Tensor<B, 2>,
    active_peak_mask: Tensor<B, 2>,
    weight: f64,
) -> Tensor<B, 1> {
    if weight > 0.0 {
        intruder_detection_loss(heads, latent, intruder_peak_mask, active_peak_mask) * weight
    } else {
        let device = latent.device();
        Tensor::zeros([1], &device)
    }
}

#[cfg(all(test, feature = "ndarray"))]
mod tests {
    use super::*;

    #[test]
    fn latent_noise_is_noop_when_disabled() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let latent = Tensor::<B, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);

        let output = apply_latent_noise(latent, 0.0)
            .into_data()
            .to_vec::<f32>()
            .expect("latent values");

        assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn latent_noise_preserves_shape_and_finiteness() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let latent = Tensor::<B, 2>::ones([3, 5], &device);

        let output = apply_latent_noise(latent, 0.02);
        assert_eq!(output.dims(), [3, 5]);
        assert!(
            output
                .into_data()
                .to_vec::<f32>()
                .expect("latent values")
                .iter()
                .all(|value| value.is_finite())
        );
    }

    #[test]
    fn retention_logits_are_antisymmetric() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let heads = EmbeddingAuxiliaryHeadsConfig {
            latent_width: 2,
            max_peaks: 1,
            retention_hidden_width: 4,
            intruder_hidden_width: 4,
        }
        .init::<B>(&device);
        let left = Tensor::<B, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
        let right = Tensor::<B, 2>::from_floats([[2.0, 1.0], [4.0, 3.0]], &device);

        let residual = heads.retention_logits(left.clone(), right.clone())
            + heads.retention_logits(right, left);

        assert!(
            residual
                .into_data()
                .to_vec::<f32>()
                .expect("retention logits")
                .iter()
                .all(|value| value.abs() < 1.0e-6)
        );
    }

    #[test]
    fn retention_output_reports_unweighted_diagnostics() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let heads = EmbeddingAuxiliaryHeadsConfig {
            latent_width: 2,
            max_peaks: 1,
            retention_hidden_width: 4,
            intruder_hidden_width: 4,
        }
        .init::<B>(&device);
        let latent =
            Tensor::<B, 2>::from_floats([[0.0, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4]], &device);
        let retention_time = Tensor::<B, 2>::from_floats([[10.0], [20.0], [40.0], [80.0]], &device);
        let retention_present = Tensor::<B, 2>::ones([4, 1], &device);
        let filename_id = Tensor::<B, 2>::zeros([4, 1], &device);
        let partner_index = Tensor::<B, 1, Int>::from_data([3, 2, 1, 0], &device);

        let output = retention_order_output(
            &heads,
            latent.clone(),
            RetentionOrderBatch {
                retention_time: retention_time.clone(),
                retention_present: retention_present.clone(),
                filename_id: filename_id.clone(),
                partner_index: partner_index.clone(),
            },
            0,
        );
        let weighted = weighted_retention_order_output(
            &heads,
            latent,
            RetentionOrderBatch {
                retention_time,
                retention_present,
                filename_id,
                partner_index,
            },
            0,
            0.2,
        );

        assert!(output.valid_pairs.into_scalar() > 0.0);
        assert!(output.raw_loss.clone().into_scalar() > 0.0);
        assert!(output.mean_rt_delta.into_scalar() > 0.0);
        assert!(output.logit_std.into_scalar().is_finite());
        assert!((output.raw_loss.into_scalar() * 0.2 - weighted.loss.into_scalar()).abs() < 1.0e-5);
        assert!(weighted.raw_loss.into_scalar() > 0.0);
    }

    #[test]
    fn intruder_loss_is_zero_without_positive_labels() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let heads = EmbeddingAuxiliaryHeadsConfig {
            latent_width: 4,
            max_peaks: 3,
            retention_hidden_width: 4,
            intruder_hidden_width: 4,
        }
        .init::<B>(&device);
        let latent = Tensor::<B, 2>::zeros([2, 4], &device);
        let intruders = Tensor::<B, 2>::zeros([2, 3], &device);
        let active = Tensor::<B, 2>::ones([2, 3], &device);

        let loss = intruder_detection_loss(&heads, latent, intruders, active).into_scalar();

        assert_eq!(loss, 0.0);
    }

    #[test]
    fn intruder_loss_is_finite_with_positive_labels() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let heads = EmbeddingAuxiliaryHeadsConfig {
            latent_width: 4,
            max_peaks: 3,
            retention_hidden_width: 4,
            intruder_hidden_width: 4,
        }
        .init::<B>(&device);
        let latent = Tensor::<B, 2>::zeros([2, 4], &device);
        let intruders = Tensor::<B, 2>::from_floats([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], &device);
        let active = Tensor::<B, 2>::ones([2, 3], &device);

        let loss = intruder_detection_loss(&heads, latent, intruders, active).into_scalar();

        assert!(loss.is_finite());
        assert!(loss > 0.0);
    }
}
