use burn::tensor::backend::Backend;
use burn::tensor::ops::{FloatTensor, IntTensor};
use burn::tensor::{Int as TensorInt, Tensor as BurnTensor, TensorPrimitive};

/// Backend extension for paired preprocessed linear-cosine scoring.
pub trait LinearCosineKernelBackend: Backend {
    /// Compute paired `[batch]` similarities, comparing left row `i` with right
    /// row `i`.
    fn linear_cosine_preprocessed_paired_kernel(
        left_mz: FloatTensor<Self>,
        left_intensity: FloatTensor<Self>,
        right_mz: FloatTensor<Self>,
        right_intensity: FloatTensor<Self>,
        mz_power: FloatTensor<Self>,
        intensity_power: FloatTensor<Self>,
        mz_tolerance: FloatTensor<Self>,
        config: LinearCosineKernelConfig,
    ) -> FloatTensor<Self>;

    /// Build a full similarity-ranking batch on the device.
    fn linear_cosine_similarity_ranking_kernel(
        teacher_mz: FloatTensor<Self>,
        teacher_intensity: FloatTensor<Self>,
        teacher_precursor: FloatTensor<Self>,
        config: SimilarityRankingKernelConfig,
    ) -> (IntTensor<Self>, IntTensor<Self>, FloatTensor<Self>);
}

/// Similarity teacher metric code for ordinary linear cosine.
pub const SIMILARITY_METRIC_LINEAR_COSINE: u32 = 0;

/// Similarity teacher metric code for precursor-shifted modified linear cosine.
pub const SIMILARITY_METRIC_MODIFIED_LINEAR_COSINE: u32 = 1;

/// Scalar options for the paired preprocessed linear-cosine kernel.
#[derive(Clone, Copy, Debug)]
pub struct LinearCosineKernelConfig {
    /// Numerical stabilizer used by the scorer normalizations.
    pub epsilon: f64,
}

/// Scalar options for the GPU-only similarity-ranking teacher.
#[derive(Clone, Copy, Debug)]
pub struct SimilarityRankingKernelConfig {
    /// Start row of the current batch inside the cached teacher window.
    pub batch_start: usize,
    /// Number of batch items to score.
    pub batch_items: usize,
    /// Random candidate partners sampled for each anchor.
    pub candidates_per_anchor: usize,
    /// Linear-cosine m/z exponent.
    pub mz_power: f64,
    /// Linear-cosine intensity exponent.
    pub intensity_power: f64,
    /// Peak matching tolerance in Da.
    pub mz_tolerance: f64,
    /// Similarity metric used by the teacher.
    pub metric: u32,
    /// Compile-time peak capacity used by modified linear cosine scratch space.
    pub max_peaks: usize,
    /// Per-batch deterministic seed.
    pub seed: u64,
    /// Numerical stabilizer used by the scorer normalizations.
    pub epsilon: f64,
}

/// Compute paired preprocessed linear-cosine similarities on the backend.
pub fn linear_cosine_preprocessed_paired_kernel<B: LinearCosineKernelBackend>(
    left_mz: BurnTensor<B, 2>,
    left_intensity: BurnTensor<B, 2>,
    right_mz: BurnTensor<B, 2>,
    right_intensity: BurnTensor<B, 2>,
    mz_power: BurnTensor<B, 1>,
    intensity_power: BurnTensor<B, 1>,
    mz_tolerance: BurnTensor<B, 1>,
    config: LinearCosineKernelConfig,
) -> BurnTensor<B, 1> {
    let output = B::linear_cosine_preprocessed_paired_kernel(
        left_mz.into_primitive().tensor(),
        left_intensity.into_primitive().tensor(),
        right_mz.into_primitive().tensor(),
        right_intensity.into_primitive().tensor(),
        mz_power.into_primitive().tensor(),
        intensity_power.into_primitive().tensor(),
        mz_tolerance.into_primitive().tensor(),
        config,
    );

    BurnTensor::from_primitive(TensorPrimitive::Float(output))
}

/// Build similarity-ranking labels on the backend.
pub fn linear_cosine_similarity_ranking_kernel<B: LinearCosineKernelBackend>(
    teacher_mz: BurnTensor<B, 2>,
    teacher_intensity: BurnTensor<B, 2>,
    teacher_precursor: BurnTensor<B, 1>,
    config: SimilarityRankingKernelConfig,
) -> (
    BurnTensor<B, 1, TensorInt>,
    BurnTensor<B, 1, TensorInt>,
    BurnTensor<B, 2>,
) {
    let (partner_a, partner_b, target_delta) = B::linear_cosine_similarity_ranking_kernel(
        teacher_mz.into_primitive().tensor(),
        teacher_intensity.into_primitive().tensor(),
        teacher_precursor.into_primitive().tensor(),
        config,
    );

    (
        BurnTensor::new(partner_a),
        BurnTensor::new(partner_b),
        BurnTensor::<B, 1>::from_primitive(TensorPrimitive::Float(target_delta))
            .reshape([config.batch_items, 1]),
    )
}
