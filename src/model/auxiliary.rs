//! Auxiliary pretraining losses over spectral embeddings.

use burn::{
    module::{Initializer, Param},
    nn::{Linear, LinearConfig, Relu},
    prelude::*,
    tensor::TensorData,
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
    /// Weight for detecting synthetic intruder peaks inserted into input spectra.
    pub intruder_peak_weight: f64,
    /// Weight for reconstructing precursor m/z and its presence flag.
    pub precursor_reconstruction_weight: f64,
    /// Weight for reconstructing precursor m/z when it was masked from the encoder input.
    pub masked_precursor_weight: f64,
    /// Weight for preserving clean-spectrum similarity order in latent space.
    pub similarity_ranking_weight: f64,
    /// Margin applied to latent cosine ordering for the similarity-ranking loss.
    pub similarity_ranking_margin: f64,
    /// Minimum clean-spectrum cosine gap required for a sampled ranking pair.
    pub similarity_ranking_min_gap: f64,
    /// Gaussian decoder-input latent noise as a fraction of the batch latent standard deviation.
    ///
    /// This is applied only during training, after the encoder and before the decoder.
    /// A value of `0.0` disables latent denoising.
    pub latent_noise_std: f64,
    /// Maximum in-batch anchors used by the similarity-ranking loss; `0` means all rows.
    pub similarity_ranking_pairs_per_batch: usize,
    /// Hidden width of the per-slot intruder detection head.
    pub intruder_hidden_width: usize,
}

impl Default for AuxiliaryLossConfig {
    fn default() -> Self {
        Self {
            reconstruction_weight: 1.0,
            masked_peak_weight: 0.25,
            consistency_weight: 0.05,
            intruder_peak_weight: 0.05,
            precursor_reconstruction_weight: 0.05,
            masked_precursor_weight: 0.05,
            similarity_ranking_weight: 0.05,
            similarity_ranking_margin: 0.05,
            similarity_ranking_min_gap: 0.05,
            latent_noise_std: 0.02,
            similarity_ranking_pairs_per_batch: 0,
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
    /// Hidden width of the per-slot intruder detection MLP.
    pub intruder_hidden_width: usize,
}

impl EmbeddingAuxiliaryHeadsConfig {
    /// Creates initialized auxiliary heads.
    pub fn init<B: Backend>(&self, device: &B::Device) -> EmbeddingAuxiliaryHeads<B> {
        EmbeddingAuxiliaryHeads {
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
    intruder_input: Linear<B>,
    intruder_slots: Param<Tensor<B, 2>>,
    intruder_output: Linear<B>,
    activation: Relu,
    max_peaks: usize,
    intruder_hidden_width: usize,
}

impl<B: Backend> EmbeddingAuxiliaryHeads<B> {
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

/// Batch fields used to supervise similarity-ranking from non-differentiable
/// teacher scores.
#[derive(Debug, Clone)]
pub struct SimilarityRankingBatch<B: Backend> {
    /// First partner row index for each anchor.
    pub partner_a_index: Tensor<B, 1, Int>,
    /// Second partner row index for each anchor.
    pub partner_b_index: Tensor<B, 1, Int>,
    /// Teacher score delta `score(anchor, a) - score(anchor, b)`.
    pub target_delta: Tensor<B, 2>,
}

impl<B: Backend> SimilarityRankingBatch<B> {
    /// Creates an empty ranking batch with no valid teacher gaps.
    pub fn zeros(batch_size: usize, device: &B::Device) -> Self {
        let indices = vec![0_i64; batch_size];
        Self {
            partner_a_index: Tensor::<B, 1, Int>::from_data(
                TensorData::new(indices.clone(), [batch_size]),
                device,
            ),
            partner_b_index: Tensor::<B, 1, Int>::from_data(
                TensorData::new(indices, [batch_size]),
                device,
            ),
            target_delta: Tensor::<B, 2>::zeros([batch_size, 1], device),
        }
    }
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

/// Precursor reconstruction objective result and diagnostics.
pub struct PrecursorReconstructionOutput<B: Backend> {
    /// Weighted or unweighted precursor reconstruction loss, depending on caller.
    pub loss: Tensor<B, 1>,
    /// Mean absolute precursor m/z error in Da, gated by the target presence flag.
    pub mae_da: Tensor<B, 1>,
}

/// Reconstructs normalized precursor m/z plus its presence flag from decoder output.
pub fn precursor_reconstruction_output<B: Backend>(
    prediction: Tensor<B, 2>,
    target: Tensor<B, 2>,
    precursor_mz_scale: f64,
) -> PrecursorReconstructionOutput<B> {
    let predicted_mz = prediction
        .clone()
        .narrow(1, 0, 1)
        .clamp_min(0.0)
        .clamp_max(1.0);
    let predicted_present = prediction
        .narrow(1, 1, 1)
        .clamp_min(1.0e-6)
        .clamp_max(1.0 - 1.0e-6);
    let target_mz = target.clone().narrow(1, 0, 1).clamp_min(0.0).clamp_max(1.0);
    let target_present = target.narrow(1, 1, 1).clamp_min(0.0).clamp_max(1.0);
    let present_count = target_present.clone().sum().clamp_min(1.0);

    let mz_delta = predicted_mz - target_mz;
    let mz_loss =
        (mz_delta.clone().powf_scalar(2.0) * target_present.clone()).sum() / present_count.clone();
    let mae_da =
        (mz_delta.abs() * target_present.clone()).sum() / present_count * precursor_mz_scale;

    let inverse_target = target_present.ones_like() - target_present.clone();
    let inverse_prediction = predicted_present.ones_like() - predicted_present.clone();
    let presence_bce = (target_present * predicted_present.log()
        + inverse_target * inverse_prediction.log())
        * -1.0;

    PrecursorReconstructionOutput {
        loss: mz_loss + presence_bce.mean(),
        mae_da,
    }
}

/// Weighted precursor reconstruction loss, or zero when the weight is disabled.
pub fn weighted_precursor_reconstruction_output<B: Backend>(
    prediction: Tensor<B, 2>,
    target: Tensor<B, 2>,
    precursor_mz_scale: f64,
    weight: f64,
) -> PrecursorReconstructionOutput<B> {
    if weight > 0.0 {
        let mut output = precursor_reconstruction_output(prediction, target, precursor_mz_scale);
        output.loss = output.loss * weight;
        output
    } else {
        zero_precursor_reconstruction_output(&prediction.device())
    }
}

/// Reconstructs precursor conditions only for rows whose precursor was masked in the input.
pub fn masked_precursor_reconstruction_output<B: Backend>(
    prediction: Tensor<B, 2>,
    target: Tensor<B, 2>,
    masked_precursor_mask: Tensor<B, 2>,
    precursor_mz_scale: f64,
) -> PrecursorReconstructionOutput<B> {
    let predicted_mz = prediction
        .clone()
        .narrow(1, 0, 1)
        .clamp_min(0.0)
        .clamp_max(1.0);
    let predicted_present = prediction
        .narrow(1, 1, 1)
        .clamp_min(1.0e-6)
        .clamp_max(1.0 - 1.0e-6);
    let target_mz = target.clone().narrow(1, 0, 1).clamp_min(0.0).clamp_max(1.0);
    let target_present = target.narrow(1, 1, 1).clamp_min(0.0).clamp_max(1.0);
    let effective_mask = masked_precursor_mask.clamp_min(0.0).clamp_max(1.0) * target_present;
    let masked_count = effective_mask.clone().sum();
    let has_masked = masked_count.clone().greater_elem(0.0).float();
    let masked_count = masked_count.clamp_min(1.0);

    let mz_delta = predicted_mz - target_mz;
    let mz_loss =
        (mz_delta.clone().powf_scalar(2.0) * effective_mask.clone()).sum() / masked_count.clone();
    let mae_da =
        (mz_delta.abs() * effective_mask.clone()).sum() / masked_count.clone() * precursor_mz_scale;

    let inverse_target = effective_mask.ones_like() - effective_mask.clone();
    let inverse_prediction = predicted_present.ones_like() - predicted_present.clone();
    let presence_bce = (effective_mask.clone() * predicted_present.log()
        + inverse_target * inverse_prediction.log())
        * -1.0;
    let presence_loss = (presence_bce * effective_mask).sum() / masked_count;

    PrecursorReconstructionOutput {
        loss: (mz_loss + presence_loss) * has_masked.clone(),
        mae_da: mae_da * has_masked,
    }
}

/// Weighted masked-precursor reconstruction loss, or zero when the weight is disabled.
pub fn weighted_masked_precursor_reconstruction_output<B: Backend>(
    prediction: Tensor<B, 2>,
    target: Tensor<B, 2>,
    masked_precursor_mask: Tensor<B, 2>,
    precursor_mz_scale: f64,
    weight: f64,
) -> PrecursorReconstructionOutput<B> {
    if weight > 0.0 {
        let mut output = masked_precursor_reconstruction_output(
            prediction,
            target,
            masked_precursor_mask,
            precursor_mz_scale,
        );
        output.loss = output.loss * weight;
        output
    } else {
        zero_precursor_reconstruction_output(&prediction.device())
    }
}

fn zero_precursor_reconstruction_output<B: Backend>(
    device: &B::Device,
) -> PrecursorReconstructionOutput<B> {
    PrecursorReconstructionOutput {
        loss: Tensor::zeros([1], device),
        mae_da: Tensor::zeros([1], device),
    }
}

/// Similarity-ranking objective result and diagnostics.
pub struct SimilarityRankingOutput<B: Backend> {
    /// Weighted or unweighted similarity-ranking loss, depending on caller.
    pub loss: Tensor<B, 1>,
    /// Number of valid anchors whose teacher similarity gap exceeded the threshold.
    pub valid_pairs: Tensor<B, 1>,
    /// Fraction of valid anchors where latent similarity preserved target ordering.
    pub accuracy: Tensor<B, 1>,
}

/// In-batch ranking loss that preserves teacher similarity order.
///
/// The teacher scores are treated as fixed labels. Gradients flow only through
/// the latent cosine similarities.
pub fn similarity_ranking_loss<B: Backend>(
    latent: Tensor<B, 2>,
    batch: SimilarityRankingBatch<B>,
    max_pairs: usize,
    margin: f64,
    min_gap: f64,
) -> Tensor<B, 1> {
    similarity_ranking_output(latent, batch, max_pairs, margin, min_gap).loss
}

/// In-batch ranking objective and diagnostics for teacher similarity order.
pub fn similarity_ranking_output<B: Backend>(
    latent: Tensor<B, 2>,
    batch: SimilarityRankingBatch<B>,
    max_pairs: usize,
    margin: f64,
    min_gap: f64,
) -> SimilarityRankingOutput<B> {
    let [batch_size, _latent_width] = latent.dims();
    if batch_size < 3 {
        return zero_similarity_ranking_output(&latent.device());
    }

    let pair_count = if max_pairs == 0 {
        batch_size
    } else {
        batch_size.min(max_pairs)
    };
    if pair_count == 0 {
        return zero_similarity_ranking_output(&latent.device());
    }

    let anchor_latent = latent.clone().narrow(0, 0, pair_count);
    let latent_a = latent
        .clone()
        .select(0, batch.partner_a_index.narrow(0, 0, pair_count));
    let latent_b = latent
        .clone()
        .select(0, batch.partner_b_index.narrow(0, 0, pair_count));
    let latent_delta = row_cosine_similarity(anchor_latent.clone(), latent_a)
        - row_cosine_similarity(anchor_latent, latent_b);
    let target_delta = batch.target_delta.narrow(0, 0, pair_count).detach();
    let target_gap = target_delta.clone().abs();
    let target_direction = target_delta / (target_gap.clone() + 1.0e-6);
    let valid = target_gap.clone().greater_elem(min_gap).float();
    let valid_pairs = valid.clone().sum();
    let gap_weights = target_gap * valid.clone();
    let gap_weight_sum = gap_weights.clone().sum().clamp_min(1.0e-6);
    let ordered_delta = target_direction * latent_delta;
    let hinge = (margin - ordered_delta.clone()).clamp_min(0.0);
    let accuracy = (ordered_delta.greater_elem(0.0).float() * valid.clone()).sum()
        / valid_pairs.clone().clamp_min(1.0);

    SimilarityRankingOutput {
        loss: (hinge * gap_weights).sum() / gap_weight_sum,
        valid_pairs,
        accuracy,
    }
}

/// Weighted similarity-ranking loss, or zero when the weight is disabled.
pub fn weighted_similarity_ranking_loss<B: Backend>(
    latent: Tensor<B, 2>,
    batch: SimilarityRankingBatch<B>,
    max_pairs: usize,
    margin: f64,
    min_gap: f64,
    weight: f64,
) -> Tensor<B, 1> {
    weighted_similarity_ranking_output(latent, batch, max_pairs, margin, min_gap, weight).loss
}

/// Weighted similarity-ranking output, or zero diagnostics when the weight is disabled.
pub fn weighted_similarity_ranking_output<B: Backend>(
    latent: Tensor<B, 2>,
    batch: SimilarityRankingBatch<B>,
    max_pairs: usize,
    margin: f64,
    min_gap: f64,
    weight: f64,
) -> SimilarityRankingOutput<B> {
    if weight > 0.0 {
        let mut output = similarity_ranking_output(latent, batch, max_pairs, margin, min_gap);
        output.loss = output.loss * weight;
        output
    } else {
        zero_similarity_ranking_output(&latent.device())
    }
}

fn zero_similarity_ranking_output<B: Backend>(device: &B::Device) -> SimilarityRankingOutput<B> {
    SimilarityRankingOutput {
        loss: Tensor::zeros([1], device),
        valid_pairs: Tensor::zeros([1], device),
        accuracy: Tensor::zeros([1], device),
    }
}

fn row_cosine_similarity<B: Backend>(left: Tensor<B, 2>, right: Tensor<B, 2>) -> Tensor<B, 2> {
    let numerator = (left.clone() * right.clone()).sum_dim(1);
    let left_norm = (left.powf_scalar(2.0).sum_dim(1) + 1.0e-6).sqrt();
    let right_norm = (right.powf_scalar(2.0).sum_dim(1) + 1.0e-6).sqrt();
    (numerator / (left_norm * right_norm))
        .clamp_min(-1.0)
        .clamp_max(1.0)
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
    fn intruder_loss_is_zero_without_positive_labels() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let heads = EmbeddingAuxiliaryHeadsConfig {
            latent_width: 4,
            max_peaks: 3,
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

    #[test]
    fn precursor_reconstruction_prefers_accurate_predictions() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let target = Tensor::<B, 2>::from_floats([[0.25, 1.0], [0.50, 1.0]], &device);
        let accurate = Tensor::<B, 2>::from_floats([[0.25, 1.0], [0.50, 1.0]], &device);
        let inaccurate = Tensor::<B, 2>::from_floats([[0.75, 0.5], [0.10, 0.5]], &device);

        let accurate = precursor_reconstruction_output(accurate, target.clone(), 2_000.0);
        let inaccurate = precursor_reconstruction_output(inaccurate, target, 2_000.0);

        assert!(accurate.loss.into_scalar() < inaccurate.loss.into_scalar());
        assert!(accurate.mae_da.into_scalar() < inaccurate.mae_da.into_scalar());
    }

    #[test]
    fn precursor_reconstruction_ignores_missing_mz_value_for_mae() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let target = Tensor::<B, 2>::from_floats([[0.0, 0.0]], &device);
        let prediction = Tensor::<B, 2>::from_floats([[0.75, 0.0]], &device);

        let output = precursor_reconstruction_output(prediction, target, 2_000.0);

        assert_eq!(output.mae_da.into_scalar(), 0.0);
        assert!(output.loss.into_scalar().is_finite());
    }

    #[test]
    fn masked_precursor_reconstruction_is_zero_without_masked_rows() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let target = Tensor::<B, 2>::from_floats([[0.25, 1.0]], &device);
        let prediction = Tensor::<B, 2>::from_floats([[0.75, 0.5]], &device);
        let mask = Tensor::<B, 2>::zeros([1, 1], &device);

        let output = masked_precursor_reconstruction_output(prediction, target, mask, 2_000.0);

        assert_eq!(output.loss.into_scalar(), 0.0);
        assert_eq!(output.mae_da.into_scalar(), 0.0);
    }

    #[test]
    fn masked_precursor_reconstruction_is_finite_for_masked_rows() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let target = Tensor::<B, 2>::from_floats([[0.25, 1.0], [0.50, 1.0]], &device);
        let prediction = Tensor::<B, 2>::from_floats([[0.75, 0.5], [0.50, 1.0]], &device);
        let mask = Tensor::<B, 2>::from_floats([[1.0], [0.0]], &device);

        let output = masked_precursor_reconstruction_output(prediction, target, mask, 2_000.0);
        let loss = output.loss.into_scalar();
        let mae = output.mae_da.into_scalar();

        assert!(loss.is_finite());
        assert!(loss > 0.0);
        assert_eq!(mae, 1_000.0);
    }

    #[test]
    fn similarity_ranking_loss_is_finite_for_ordered_targets() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let latent =
            Tensor::<B, 2>::from_floats([[0.0, 1.0], [1.0, 0.0], [0.0, 0.9], [0.9, 0.0]], &device);
        let batch = SimilarityRankingBatch {
            partner_a_index: Tensor::<B, 1, Int>::from_data(
                TensorData::new(vec![2_i64, 3, 0, 1], [4]),
                &device,
            ),
            partner_b_index: Tensor::<B, 1, Int>::from_data(
                TensorData::new(vec![1_i64, 0, 3, 2], [4]),
                &device,
            ),
            target_delta: Tensor::<B, 2>::from_floats([[0.7], [0.7], [0.7], [0.7]], &device),
        };

        let loss = similarity_ranking_loss(latent, batch, 0, 0.05, 0.01).into_scalar();

        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }

    #[test]
    fn similarity_ranking_output_reports_valid_pairs_and_accuracy() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let latent =
            Tensor::<B, 2>::from_floats([[0.0, 1.0], [1.0, 0.0], [0.0, 0.9], [0.9, 0.0]], &device);
        let batch = SimilarityRankingBatch {
            partner_a_index: Tensor::<B, 1, Int>::from_data(
                TensorData::new(vec![2_i64, 3, 0, 1], [4]),
                &device,
            ),
            partner_b_index: Tensor::<B, 1, Int>::from_data(
                TensorData::new(vec![1_i64, 0, 3, 2], [4]),
                &device,
            ),
            target_delta: Tensor::<B, 2>::from_floats([[0.7], [0.7], [0.7], [0.7]], &device),
        };

        let output = similarity_ranking_output(latent, batch, 0, 0.05, 0.01);
        let accuracy = output.accuracy.into_scalar();

        assert!(output.valid_pairs.into_scalar() > 0.0);
        assert!(accuracy.is_finite());
        assert!((0.0..=1.0).contains(&accuracy));
    }

    #[test]
    fn similarity_ranking_loss_weights_hinge_by_metric_gap() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let latent = Tensor::<B, 2>::from_floats([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], &device);
        let batch = SimilarityRankingBatch {
            partner_a_index: Tensor::<B, 1, Int>::from_data(
                TensorData::new(vec![1_i64, 2, 0], [3]),
                &device,
            ),
            partner_b_index: Tensor::<B, 1, Int>::from_data(
                TensorData::new(vec![2_i64, 0, 1], [3]),
                &device,
            ),
            target_delta: Tensor::<B, 2>::from_floats([[0.9], [0.1], [0.0]], &device),
        };

        let loss = similarity_ranking_loss(latent, batch, 0, 0.5, 0.01).into_scalar();

        assert!(
            (loss - 0.15).abs() < 1.0e-3,
            "expected gap-weighted hinge loss near 0.15, got {loss}"
        );
    }

    #[test]
    fn similarity_ranking_loss_is_zero_for_too_small_batches() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let latent = Tensor::<B, 2>::zeros([2, 2], &device);
        let batch = SimilarityRankingBatch::zeros(2, &device);

        let loss = similarity_ranking_loss(latent, batch, 0, 0.05, 0.01).into_scalar();

        assert_eq!(loss, 0.0);
    }
}
