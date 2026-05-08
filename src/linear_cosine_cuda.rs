//! CUDA kernels for preprocessed spectral similarities.
//!
//! The kernel expects fixed-width rows whose peaks are already sorted by m/z and
//! already merged with the same tolerance used by the CPU `LinearCosine`
//! teacher. Zero intensity marks padding.

#![allow(
    clippy::cast_possible_truncation,
    clippy::too_many_arguments,
    clippy::trivially_copy_pass_by_ref
)]

#[cfg(feature = "train")]
use burn::backend::Autodiff;
#[cfg(feature = "train")]
use burn::backend::autodiff::checkpoint::{base::Checkpointer, strategy::CheckpointStrategy};
#[cfg(feature = "train")]
use burn::backend::autodiff::grads::Gradients;
#[cfg(feature = "train")]
use burn::backend::autodiff::ops::{Backward, Ops, OpsKind};
use burn::tensor::backend::Backend;
use burn::tensor::ops::{FloatTensor, IntTensor};
use burn::tensor::{Element, Int as TensorInt, Shape, Tensor as BurnTensor, TensorPrimitive};
use burn_cubecl::cubecl::prelude::*;
use burn_cubecl::cubecl::{CubeDim, calculate_cube_count_elemwise};
use burn_cubecl::ops::numeric::empty_device_dtype;
use burn_cubecl::{BoolElement, CubeBackend, CubeRuntime, FloatElement, IntElement};
#[cfg(feature = "cuda-fusion")]
use burn_fusion::{
    Fusion, FusionBackend,
    stream::{Operation, OperationStreams},
};
#[cfg(feature = "cuda-fusion")]
use burn_ir::{CustomOpIr, OperationIr, OperationOutput, TensorIr};
#[cfg(feature = "cuda-fusion")]
use core::marker::PhantomData;

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
        config: SimilarityRankingKernelConfig,
    ) -> (IntTensor<Self>, IntTensor<Self>, FloatTensor<Self>);
}

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
    config: SimilarityRankingKernelConfig,
) -> (
    BurnTensor<B, 1, TensorInt>,
    BurnTensor<B, 1, TensorInt>,
    BurnTensor<B, 2>,
) {
    let (partner_a, partner_b, target_delta) = B::linear_cosine_similarity_ranking_kernel(
        teacher_mz.into_primitive().tensor(),
        teacher_intensity.into_primitive().tensor(),
        config,
    );

    (
        BurnTensor::new(partner_a),
        BurnTensor::new(partner_b),
        BurnTensor::<B, 1>::from_primitive(TensorPrimitive::Float(target_delta))
            .reshape([config.batch_items, 1]),
    )
}

impl<R, F, I, BT> LinearCosineKernelBackend for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn linear_cosine_preprocessed_paired_kernel(
        left_mz: FloatTensor<Self>,
        left_intensity: FloatTensor<Self>,
        right_mz: FloatTensor<Self>,
        right_intensity: FloatTensor<Self>,
        mz_power: FloatTensor<Self>,
        intensity_power: FloatTensor<Self>,
        mz_tolerance: FloatTensor<Self>,
        config: LinearCosineKernelConfig,
    ) -> FloatTensor<Self> {
        left_mz.assert_is_on_same_device(&left_intensity);
        left_mz.assert_is_on_same_device(&right_mz);
        left_mz.assert_is_on_same_device(&right_intensity);
        left_mz.assert_is_on_same_device(&mz_power);
        left_mz.assert_is_on_same_device(&intensity_power);
        left_mz.assert_is_on_same_device(&mz_tolerance);

        let [batch_size, _left_peaks] = left_mz.shape.dims();
        let [right_rows, _right_peaks] = right_mz.shape.dims();
        assert_eq!(
            batch_size, right_rows,
            "paired linear cosine requires the same number of left and right rows"
        );

        let output_shape = Shape::new([batch_size]);
        let total_elem = output_shape.num_elements();
        let output = empty_device_dtype(
            left_mz.client.clone(),
            left_mz.device.clone(),
            output_shape,
            left_mz.dtype,
        );
        let cube_dim = CubeDim::new(&left_mz.client, total_elem);
        let cube_count = calculate_cube_count_elemwise(&left_mz.client, total_elem, cube_dim);

        linear_cosine_preprocessed_paired_sorted_forward::launch::<F, R>(
            &left_mz.client,
            cube_count,
            cube_dim,
            left_mz.as_tensor_arg(1),
            left_intensity.as_tensor_arg(1),
            right_mz.as_tensor_arg(1),
            right_intensity.as_tensor_arg(1),
            mz_power.as_tensor_arg(1),
            intensity_power.as_tensor_arg(1),
            mz_tolerance.as_tensor_arg(1),
            output.as_tensor_arg(1),
            ScalarArg::new(config.epsilon as f32),
        )
        .expect("paired linear cosine forward kernel launch failed");

        output
    }

    fn linear_cosine_similarity_ranking_kernel(
        teacher_mz: FloatTensor<Self>,
        teacher_intensity: FloatTensor<Self>,
        config: SimilarityRankingKernelConfig,
    ) -> (IntTensor<Self>, IntTensor<Self>, FloatTensor<Self>) {
        teacher_mz.assert_is_on_same_device(&teacher_intensity);

        let [teacher_rows, _teacher_peaks] = teacher_mz.shape.dims();
        assert!(
            config.batch_start + config.batch_items <= teacher_rows,
            "similarity-ranking batch is outside the teacher cache window"
        );

        let index_shape = Shape::new([config.batch_items]);
        let delta_shape = Shape::new([config.batch_items]);
        let partner_a = empty_device_dtype(
            teacher_mz.client.clone(),
            teacher_mz.device.clone(),
            index_shape.clone(),
            I::dtype(),
        );
        let partner_b = empty_device_dtype(
            teacher_mz.client.clone(),
            teacher_mz.device.clone(),
            index_shape,
            I::dtype(),
        );
        let target_delta = empty_device_dtype(
            teacher_mz.client.clone(),
            teacher_mz.device.clone(),
            delta_shape,
            teacher_mz.dtype,
        );

        let total_elem = config.batch_items;
        let cube_dim = CubeDim::new(&teacher_mz.client, total_elem);
        let cube_count = calculate_cube_count_elemwise(&teacher_mz.client, total_elem, cube_dim);

        linear_cosine_similarity_ranking_forward::launch::<F, I, R>(
            &teacher_mz.client,
            cube_count,
            cube_dim,
            teacher_mz.as_tensor_arg(1),
            teacher_intensity.as_tensor_arg(1),
            partner_a.as_tensor_arg(1),
            partner_b.as_tensor_arg(1),
            target_delta.as_tensor_arg(1),
            ScalarArg::new(config.batch_start as u32),
            ScalarArg::new(config.batch_items as u32),
            ScalarArg::new(config.candidates_per_anchor as u32),
            ScalarArg::new(config.mz_power as f32),
            ScalarArg::new(config.intensity_power as f32),
            ScalarArg::new(config.mz_tolerance as f32),
            ScalarArg::new(config.seed as u32),
            ScalarArg::new(config.epsilon as f32),
        )
        .expect("similarity-ranking forward kernel launch failed");

        (partner_a, partner_b, target_delta)
    }
}

#[cfg(feature = "cuda-fusion")]
impl<B> LinearCosineKernelBackend for Fusion<B>
where
    B: FusionBackend + LinearCosineKernelBackend + Send + Sync,
{
    fn linear_cosine_preprocessed_paired_kernel(
        left_mz: FloatTensor<Self>,
        left_intensity: FloatTensor<Self>,
        right_mz: FloatTensor<Self>,
        right_intensity: FloatTensor<Self>,
        mz_power: FloatTensor<Self>,
        intensity_power: FloatTensor<Self>,
        mz_tolerance: FloatTensor<Self>,
        config: LinearCosineKernelConfig,
    ) -> FloatTensor<Self> {
        let [batch_size, _left_peaks] = left_mz.shape.dims();
        let [right_rows, _right_peaks] = right_mz.shape.dims();
        assert_eq!(
            batch_size, right_rows,
            "paired linear cosine requires the same number of left and right rows"
        );

        let output_shape = Shape::new([batch_size]);
        let streams = OperationStreams::with_inputs([
            &left_mz,
            &left_intensity,
            &right_mz,
            &right_intensity,
            &mz_power,
            &intensity_power,
            &mz_tolerance,
        ]);
        let client = left_mz.client.clone();
        let output = TensorIr::uninit(client.create_empty_handle(), output_shape, left_mz.dtype);
        let desc = CustomOpIr::new(
            "linear_cosine_preprocessed_paired_forward",
            &[
                left_mz.into_ir(),
                left_intensity.into_ir(),
                right_mz.into_ir(),
                right_intensity.into_ir(),
                mz_power.into_ir(),
                intensity_power.into_ir(),
                mz_tolerance.into_ir(),
            ],
            &[output],
        );

        client
            .register(
                streams,
                OperationIr::Custom(desc.clone()),
                LinearCosinePairedFusionForward::<B> {
                    desc,
                    config,
                    backend: PhantomData,
                },
            )
            .output()
    }

    fn linear_cosine_similarity_ranking_kernel(
        teacher_mz: FloatTensor<Self>,
        teacher_intensity: FloatTensor<Self>,
        config: SimilarityRankingKernelConfig,
    ) -> (IntTensor<Self>, IntTensor<Self>, FloatTensor<Self>) {
        let [teacher_rows, _teacher_peaks] = teacher_mz.shape.dims();
        assert!(
            config.batch_start + config.batch_items <= teacher_rows,
            "similarity-ranking batch is outside the teacher cache window"
        );

        let streams = OperationStreams::with_inputs([&teacher_mz, &teacher_intensity]);
        let client = teacher_mz.client.clone();
        let index_shape = Shape::new([config.batch_items]);
        let delta_shape = Shape::new([config.batch_items]);
        let partner_a = TensorIr::uninit(
            client.create_empty_handle(),
            index_shape.clone(),
            <B as Backend>::IntElem::dtype(),
        );
        let partner_b = TensorIr::uninit(
            client.create_empty_handle(),
            index_shape,
            <B as Backend>::IntElem::dtype(),
        );
        let target_delta =
            TensorIr::uninit(client.create_empty_handle(), delta_shape, teacher_mz.dtype);
        let desc = CustomOpIr::new(
            "linear_cosine_similarity_ranking_forward",
            &[teacher_mz.into_ir(), teacher_intensity.into_ir()],
            &[partner_a, partner_b, target_delta],
        );

        let mut outputs = client.register(
            streams,
            OperationIr::Custom(desc.clone()),
            LinearCosineRankingFusionForward::<B> {
                desc,
                config,
                backend: PhantomData,
            },
        );
        let target_delta = outputs.pop().expect("ranking custom op has delta output");
        let partner_b = outputs
            .pop()
            .expect("ranking custom op has second partner output");
        let partner_a = outputs
            .pop()
            .expect("ranking custom op has first partner output");

        (partner_a, partner_b, target_delta)
    }
}

#[cfg(feature = "cuda-fusion")]
#[derive(Debug)]
struct LinearCosinePairedFusionForward<B: FusionBackend> {
    desc: CustomOpIr,
    config: LinearCosineKernelConfig,
    backend: PhantomData<B>,
}

#[cfg(feature = "cuda-fusion")]
impl<B> Operation<B::FusionRuntime> for LinearCosinePairedFusionForward<B>
where
    B: FusionBackend + LinearCosineKernelBackend + Send + Sync,
{
    fn execute(&self, handles: &mut burn_ir::HandleContainer<B::Handle>) {
        let (inputs, outputs) = self.desc.as_fixed::<7, 1>();
        let output = B::linear_cosine_preprocessed_paired_kernel(
            handles.get_float_tensor::<B>(&inputs[0]),
            handles.get_float_tensor::<B>(&inputs[1]),
            handles.get_float_tensor::<B>(&inputs[2]),
            handles.get_float_tensor::<B>(&inputs[3]),
            handles.get_float_tensor::<B>(&inputs[4]),
            handles.get_float_tensor::<B>(&inputs[5]),
            handles.get_float_tensor::<B>(&inputs[6]),
            self.config,
        );

        handles.register_float_tensor::<B>(&outputs[0].id, output);
    }
}

#[cfg(feature = "cuda-fusion")]
#[derive(Debug)]
struct LinearCosineRankingFusionForward<B: FusionBackend> {
    desc: CustomOpIr,
    config: SimilarityRankingKernelConfig,
    backend: PhantomData<B>,
}

#[cfg(feature = "cuda-fusion")]
impl<B> Operation<B::FusionRuntime> for LinearCosineRankingFusionForward<B>
where
    B: FusionBackend + LinearCosineKernelBackend + Send + Sync,
{
    fn execute(&self, handles: &mut burn_ir::HandleContainer<B::Handle>) {
        let (inputs, outputs) = self.desc.as_fixed::<2, 3>();
        let (partner_a, partner_b, target_delta) = B::linear_cosine_similarity_ranking_kernel(
            handles.get_float_tensor::<B>(&inputs[0]),
            handles.get_float_tensor::<B>(&inputs[1]),
            self.config,
        );

        handles.register_int_tensor::<B>(&outputs[0].id, partner_a);
        handles.register_int_tensor::<B>(&outputs[1].id, partner_b);
        handles.register_float_tensor::<B>(&outputs[2].id, target_delta);
    }
}

#[cfg(feature = "train")]
impl<B, C> LinearCosineKernelBackend for Autodiff<B, C>
where
    B: LinearCosineKernelBackend,
    C: CheckpointStrategy,
{
    fn linear_cosine_preprocessed_paired_kernel(
        left_mz: FloatTensor<Self>,
        left_intensity: FloatTensor<Self>,
        right_mz: FloatTensor<Self>,
        right_intensity: FloatTensor<Self>,
        mz_power: FloatTensor<Self>,
        intensity_power: FloatTensor<Self>,
        mz_tolerance: FloatTensor<Self>,
        config: LinearCosineKernelConfig,
    ) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct NoGradientBackward;

        impl<B> Backward<B, 7> for NoGradientBackward
        where
            B: LinearCosineKernelBackend,
        {
            type State = ();

            fn backward(
                self,
                _ops: Ops<Self::State, 7>,
                _grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
            }
        }

        match NoGradientBackward
            .prepare::<C>([
                left_mz.node.clone(),
                left_intensity.node.clone(),
                right_mz.node.clone(),
                right_intensity.node.clone(),
                mz_power.node.clone(),
                intensity_power.node.clone(),
                mz_tolerance.node.clone(),
            ])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => {
                let output = B::linear_cosine_preprocessed_paired_kernel(
                    left_mz.primitive.clone(),
                    left_intensity.primitive.clone(),
                    right_mz.primitive.clone(),
                    right_intensity.primitive.clone(),
                    mz_power.primitive.clone(),
                    intensity_power.primitive.clone(),
                    mz_tolerance.primitive.clone(),
                    config,
                );

                prep.finish((), output)
            }
            OpsKind::UnTracked(prep) => {
                let output = B::linear_cosine_preprocessed_paired_kernel(
                    left_mz.primitive,
                    left_intensity.primitive,
                    right_mz.primitive,
                    right_intensity.primitive,
                    mz_power.primitive,
                    intensity_power.primitive,
                    mz_tolerance.primitive,
                    config,
                );

                prep.finish(output)
            }
        }
    }

    fn linear_cosine_similarity_ranking_kernel(
        teacher_mz: FloatTensor<Self>,
        teacher_intensity: FloatTensor<Self>,
        config: SimilarityRankingKernelConfig,
    ) -> (IntTensor<Self>, IntTensor<Self>, FloatTensor<Self>) {
        #[derive(Debug)]
        struct NoGradientBackward;

        impl<B> Backward<B, 2> for NoGradientBackward
        where
            B: LinearCosineKernelBackend,
        {
            type State = ();

            fn backward(
                self,
                _ops: Ops<Self::State, 2>,
                _grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
            }
        }

        match NoGradientBackward
            .prepare::<C>([teacher_mz.node.clone(), teacher_intensity.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => {
                let (partner_a, partner_b, target_delta) =
                    B::linear_cosine_similarity_ranking_kernel(
                        teacher_mz.primitive.clone(),
                        teacher_intensity.primitive.clone(),
                        config,
                    );

                (partner_a, partner_b, prep.finish((), target_delta))
            }
            OpsKind::UnTracked(prep) => {
                let (partner_a, partner_b, target_delta) =
                    B::linear_cosine_similarity_ranking_kernel(
                        teacher_mz.primitive,
                        teacher_intensity.primitive,
                        config,
                    );

                (partner_a, partner_b, prep.finish(target_delta))
            }
        }
    }
}

#[cube(launch)]
fn linear_cosine_preprocessed_paired_sorted_forward<F: Float>(
    left_mz: &Tensor<F>,
    left_intensity: &Tensor<F>,
    right_mz: &Tensor<F>,
    right_intensity: &Tensor<F>,
    mz_power: &Tensor<F>,
    intensity_power: &Tensor<F>,
    mz_tolerance: &Tensor<F>,
    output: &mut Tensor<F>,
    epsilon: f32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let right_rows = right_mz.shape(0);
    let row = ABSOLUTE_POS;

    if row >= right_rows {
        terminate!();
    }

    let tolerance = mz_tolerance[row * mz_tolerance.stride(0)];
    let eps = F::cast_from(epsilon);
    let mz_p = mz_power[row * mz_power.stride(0)];
    let intensity_p = intensity_power[row * intensity_power.stride(0)];

    output[ABSOLUTE_POS] = linear_cosine_score_rows(
        left_mz,
        left_intensity,
        row,
        right_mz,
        right_intensity,
        row,
        mz_p,
        intensity_p,
        tolerance,
        eps,
    );
}

#[cube(launch)]
fn linear_cosine_similarity_ranking_forward<F: Float, I: Int>(
    teacher_mz: &Tensor<F>,
    teacher_intensity: &Tensor<F>,
    partner_a: &mut Tensor<I>,
    partner_b: &mut Tensor<I>,
    target_delta: &mut Tensor<F>,
    batch_start: u32,
    batch_items: u32,
    candidates_per_anchor: u32,
    mz_power: f32,
    intensity_power: f32,
    mz_tolerance: f32,
    seed: u32,
    epsilon: f32,
) {
    if ABSOLUTE_POS >= partner_a.len() {
        terminate!();
    }

    let anchor = ABSOLUTE_POS;
    if anchor >= batch_items as usize || batch_items < 3 {
        terminate!();
    }

    let candidate_count = candidates_per_anchor.max(2).min(batch_items - 1);
    let teacher_anchor = batch_start as usize + anchor;
    let mz_p = F::cast_from(mz_power);
    let intensity_p = F::cast_from(intensity_power);
    let tolerance = F::cast_from(mz_tolerance);
    let eps = F::cast_from(epsilon);
    let zero = F::new(0.0_f32);
    let mut state = seed ^ (((anchor as u32) + 1u32) * 40503u32) ^ (batch_start >> 16);
    if state == 0u32 {
        state = 0x6d2b_79f5u32;
    }

    let mut best_local = anchor;
    let mut worst_local = anchor;
    let mut best_score = F::new(-1.0_f32);
    let mut worst_score = F::new(2.0_f32);

    for _candidate in 0..candidate_count {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let mut local_partner = (state % (batch_items - 1)) as usize;
        if local_partner >= anchor {
            local_partner += 1;
        }

        let score = linear_cosine_score_rows(
            teacher_mz,
            teacher_intensity,
            teacher_anchor,
            teacher_mz,
            teacher_intensity,
            batch_start as usize + local_partner,
            mz_p,
            intensity_p,
            tolerance,
            eps,
        );

        if score > best_score {
            best_score = score;
            best_local = local_partner;
        }
        if score < worst_score {
            worst_score = score;
            worst_local = local_partner;
        }
    }

    partner_a[anchor] = I::cast_from(best_local as u32);
    partner_b[anchor] = I::cast_from(worst_local as u32);
    target_delta[anchor] = (best_score - worst_score).max(zero);
}

#[cube]
fn linear_cosine_score_rows<F: Float>(
    left_mz: &Tensor<F>,
    left_intensity: &Tensor<F>,
    left_row: usize,
    right_mz: &Tensor<F>,
    right_intensity: &Tensor<F>,
    right_row: usize,
    mz_p: F,
    intensity_p: F,
    tolerance: F,
    eps: F,
) -> F {
    let left_peaks = left_mz.shape(1);
    let right_peaks = right_mz.shape(1);
    let zero = F::new(0.0_f32);
    let one = F::new(1.0_f32);

    let mut left_intensity_max = zero;
    let mut left_mz_max = zero;
    for peak in 0..left_peaks {
        let intensity =
            left_intensity[left_row * left_intensity.stride(0) + peak * left_intensity.stride(1)];
        if intensity > zero {
            let mz = left_mz[left_row * left_mz.stride(0) + peak * left_mz.stride(1)];
            left_intensity_max = left_intensity_max.max(intensity.max(eps).powf(intensity_p));
            left_mz_max = left_mz_max.max(mz.max(eps).powf(mz_p));
        }
    }
    left_intensity_max += eps;
    left_mz_max += eps;

    let mut left_product_max = zero;
    let mut left_norm_square = zero;
    for peak in 0..left_peaks {
        let intensity =
            left_intensity[left_row * left_intensity.stride(0) + peak * left_intensity.stride(1)];
        if intensity > zero {
            let mz = left_mz[left_row * left_mz.stride(0) + peak * left_mz.stride(1)];
            let product = (intensity.max(eps).powf(intensity_p) / left_intensity_max)
                * (mz.max(eps).powf(mz_p) / left_mz_max);
            left_product_max = left_product_max.max(product);
        }
    }
    left_product_max += eps;
    for peak in 0..left_peaks {
        let product = peak_product(
            left_mz,
            left_intensity,
            left_row,
            peak,
            mz_p,
            intensity_p,
            left_mz_max,
            left_intensity_max,
            left_product_max,
            eps,
        );
        left_norm_square += product * product;
    }

    let mut right_intensity_max = zero;
    let mut right_mz_max = zero;
    for peak in 0..right_peaks {
        let intensity = right_intensity
            [right_row * right_intensity.stride(0) + peak * right_intensity.stride(1)];
        if intensity > zero {
            let mz = right_mz[right_row * right_mz.stride(0) + peak * right_mz.stride(1)];
            right_intensity_max = right_intensity_max.max(intensity.max(eps).powf(intensity_p));
            right_mz_max = right_mz_max.max(mz.max(eps).powf(mz_p));
        }
    }
    right_intensity_max += eps;
    right_mz_max += eps;

    let mut right_product_max = zero;
    let mut right_norm_square = zero;
    for peak in 0..right_peaks {
        let intensity = right_intensity
            [right_row * right_intensity.stride(0) + peak * right_intensity.stride(1)];
        if intensity > zero {
            let mz = right_mz[right_row * right_mz.stride(0) + peak * right_mz.stride(1)];
            let product = (intensity.max(eps).powf(intensity_p) / right_intensity_max)
                * (mz.max(eps).powf(mz_p) / right_mz_max);
            right_product_max = right_product_max.max(product);
        }
    }
    right_product_max += eps;
    for peak in 0..right_peaks {
        let product = peak_product(
            right_mz,
            right_intensity,
            right_row,
            peak,
            mz_p,
            intensity_p,
            right_mz_max,
            right_intensity_max,
            right_product_max,
            eps,
        );
        right_norm_square += product * product;
    }

    let mut left_cursor = 0usize;
    let mut right_cursor = 0usize;
    let mut score = zero;
    while left_cursor < left_peaks && right_cursor < right_peaks {
        let left_intensity_value = left_intensity
            [left_intensity.stride(0) * left_row + left_intensity.stride(1) * left_cursor];
        if left_intensity_value <= zero {
            left_cursor += 1;
        } else {
            let right_product = peak_product(
                right_mz,
                right_intensity,
                right_row,
                right_cursor,
                mz_p,
                intensity_p,
                right_mz_max,
                right_intensity_max,
                right_product_max,
                eps,
            );
            if right_product <= zero {
                right_cursor += 1;
            } else {
                let mz_left =
                    left_mz[left_mz.stride(0) * left_row + left_mz.stride(1) * left_cursor];
                let mz_right =
                    right_mz[right_mz.stride(0) * right_row + right_mz.stride(1) * right_cursor];
                let delta = mz_left - mz_right;

                if delta.abs() <= tolerance {
                    let left_product = peak_product(
                        left_mz,
                        left_intensity,
                        left_row,
                        left_cursor,
                        mz_p,
                        intensity_p,
                        left_mz_max,
                        left_intensity_max,
                        left_product_max,
                        eps,
                    );
                    score += left_product * right_product;
                    left_cursor += 1;
                    right_cursor += 1;
                } else if mz_left + tolerance < mz_right {
                    left_cursor += 1;
                } else {
                    right_cursor += 1;
                }
            }
        }
    }

    let left_norm = (left_norm_square + eps).sqrt();
    let right_norm = (right_norm_square + eps).sqrt();
    let similarity = score / (left_norm * right_norm + eps);
    similarity.max(zero).min(one)
}

#[cube]
fn peak_product<F: Float>(
    mz_tensor: &Tensor<F>,
    intensity_tensor: &Tensor<F>,
    row: usize,
    peak: usize,
    mz_power: F,
    intensity_power: F,
    mz_max: F,
    intensity_max: F,
    product_max: F,
    epsilon: F,
) -> F {
    let zero = F::new(0.0_f32);
    let intensity =
        intensity_tensor[row * intensity_tensor.stride(0) + peak * intensity_tensor.stride(1)];
    let mut product = zero;
    if intensity > zero {
        let mz = mz_tensor[row * mz_tensor.stride(0) + peak * mz_tensor.stride(1)];
        let intensity_component = intensity.max(epsilon).powf(intensity_power) / intensity_max;
        let mz_component = mz.max(epsilon).powf(mz_power) / mz_max;

        product = intensity_component * mz_component / product_max;
    }

    product
}

#[cfg(test)]
mod tests {
    use super::*;

    use burn::{
        backend::{Autodiff, Cuda, cuda::CudaDevice},
        tensor::{Tensor, TensorData},
    };
    use mass_spectrometry::prelude::{LinearCosine, ScalarSimilarity, Spectrum};

    type TestBackend = Autodiff<Cuda<f32, i32>>;

    #[derive(Clone, Copy)]
    struct TestSpectrum<'a> {
        precursor_mz: f32,
        mz: &'a [f32],
        intensity: &'a [f32],
    }

    impl Spectrum for TestSpectrum<'_> {
        type Precision = f32;

        type SortedIntensitiesIter<'a>
            = std::iter::Copied<std::slice::Iter<'a, f32>>
        where
            Self: 'a;
        type SortedMzIter<'a>
            = std::iter::Copied<std::slice::Iter<'a, f32>>
        where
            Self: 'a;
        type SortedPeaksIter<'a>
            = std::iter::Zip<
            std::iter::Copied<std::slice::Iter<'a, f32>>,
            std::iter::Copied<std::slice::Iter<'a, f32>>,
        >
        where
            Self: 'a;

        fn len(&self) -> usize {
            self.mz.len()
        }

        fn intensities(&self) -> Self::SortedIntensitiesIter<'_> {
            self.intensity.iter().copied()
        }

        fn intensity_nth(&self, n: usize) -> Self::Precision {
            self.intensity[n]
        }

        fn mz(&self) -> Self::SortedMzIter<'_> {
            self.mz.iter().copied()
        }

        fn mz_from(&self, index: usize) -> Self::SortedMzIter<'_> {
            self.mz[index..].iter().copied()
        }

        fn mz_nth(&self, n: usize) -> Self::Precision {
            self.mz[n]
        }

        fn peaks(&self) -> Self::SortedPeaksIter<'_> {
            self.mz.iter().copied().zip(self.intensity.iter().copied())
        }

        fn peak_nth(&self, n: usize) -> (Self::Precision, Self::Precision) {
            (self.mz[n], self.intensity[n])
        }

        fn precursor_mz(&self) -> Self::Precision {
            self.precursor_mz
        }
    }

    #[test]
    fn paired_kernel_matches_cpu_linear_cosine_for_preprocessed_rows() {
        let device = CudaDevice::default();
        let rows = 4;
        let peaks = 5;
        let left_mz = vec![
            100.000_f32,
            150.000,
            220.000,
            400.000,
            0.000,
            90.000,
            180.000,
            300.000,
            450.000,
            0.000,
            10.000,
            20.000,
            30.000,
            0.000,
            0.000,
            50.000,
            75.000,
            100.000,
            0.000,
            0.000,
        ];
        let left_intensity = vec![
            1.0_f32, 0.3, 0.7, 0.2, 0.0, 0.4, 1.0, 0.6, 0.1, 0.0, 1.0, 0.5, 0.2, 0.0, 0.0, 1.0,
            0.5, 0.25, 0.0, 0.0,
        ];
        let right_mz = vec![
            100.010_f32,
            149.990,
            220.030,
            400.040,
            0.000,
            90.030,
            180.060,
            300.000,
            460.000,
            0.000,
            100.000,
            200.000,
            300.000,
            0.000,
            0.000,
            50.010,
            76.000,
            0.000,
            0.000,
            0.000,
        ];
        let right_intensity = vec![
            1.0_f32, 0.3, 0.7, 0.2, 0.0, 0.4, 1.0, 0.6, 0.1, 0.0, 0.7, 0.4, 0.3, 0.0, 0.0, 0.8,
            0.1, 0.0, 0.0, 0.0,
        ];
        let mz_power = 0.15_f32;
        let intensity_power = 0.7_f32;
        let mz_tolerance = 0.05_f32;

        let scores = linear_cosine_preprocessed_paired_kernel(
            Tensor::<TestBackend, 2>::from_data(
                TensorData::new(left_mz.clone(), [rows, peaks]),
                &device,
            ),
            Tensor::<TestBackend, 2>::from_data(
                TensorData::new(left_intensity.clone(), [rows, peaks]),
                &device,
            ),
            Tensor::<TestBackend, 2>::from_data(
                TensorData::new(right_mz.clone(), [rows, peaks]),
                &device,
            ),
            Tensor::<TestBackend, 2>::from_data(
                TensorData::new(right_intensity.clone(), [rows, peaks]),
                &device,
            ),
            Tensor::<TestBackend, 1>::from_data(
                TensorData::new(vec![mz_power; rows], [rows]),
                &device,
            ),
            Tensor::<TestBackend, 1>::from_data(
                TensorData::new(vec![intensity_power; rows], [rows]),
                &device,
            ),
            Tensor::<TestBackend, 1>::from_data(
                TensorData::new(vec![mz_tolerance; rows], [rows]),
                &device,
            ),
            LinearCosineKernelConfig { epsilon: 1.0e-8 },
        )
        .into_data()
        .to_vec::<f32>()
        .expect("kernel output should be f32");

        let scorer = LinearCosine::new(
            f64::from(mz_power),
            f64::from(intensity_power),
            f64::from(mz_tolerance),
        )
        .expect("CPU linear cosine config should be valid");
        for row in 0..rows {
            let (left_mz_row, left_intensity_row) =
                nonzero_row(&left_mz, &left_intensity, row, peaks);
            let (right_mz_row, right_intensity_row) =
                nonzero_row(&right_mz, &right_intensity, row, peaks);
            let left = TestSpectrum {
                precursor_mz: 1.0,
                mz: &left_mz_row,
                intensity: &left_intensity_row,
            };
            let right = TestSpectrum {
                precursor_mz: 1.0,
                mz: &right_mz_row,
                intensity: &right_intensity_row,
            };
            let expected = scorer
                .similarity(&left, &right)
                .map(|(score, _)| score as f32)
                .expect("CPU linear cosine should score test spectra");
            assert!(
                (scores[row] - expected).abs() < 1.0e-5,
                "row {row}: cuda={} cpu={expected}",
                scores[row],
            );
        }
    }

    #[test]
    fn ranking_kernel_matches_cpu_reference_for_sampled_partners() {
        let device = CudaDevice::default();
        let rows = 6;
        let peaks = 5;
        let mz = vec![
            100.000_f32,
            150.000,
            220.000,
            400.000,
            0.000,
            100.010,
            150.010,
            220.010,
            405.000,
            0.000,
            90.000,
            180.000,
            300.000,
            450.000,
            0.000,
            91.000,
            181.000,
            300.020,
            600.000,
            0.000,
            50.000,
            75.000,
            100.000,
            0.000,
            0.000,
            50.010,
            75.010,
            120.000,
            0.000,
            0.000,
        ];
        let intensity = vec![
            1.0_f32, 0.3, 0.7, 0.2, 0.0, 0.9, 0.4, 0.6, 0.1, 0.0, 0.4, 1.0, 0.6, 0.1, 0.0, 0.3,
            0.8, 0.7, 0.2, 0.0, 1.0, 0.5, 0.25, 0.0, 0.0, 0.8, 0.7, 0.1, 0.0, 0.0,
        ];
        let config = SimilarityRankingKernelConfig {
            batch_start: 1,
            batch_items: 4,
            candidates_per_anchor: 3,
            mz_power: 0.15,
            intensity_power: 0.7,
            mz_tolerance: 0.05,
            seed: 12_345,
            epsilon: 1.0e-8,
        };

        let (partner_a, partner_b, target_delta) = linear_cosine_similarity_ranking_kernel(
            Tensor::<TestBackend, 2>::from_data(
                TensorData::new(mz.clone(), [rows, peaks]),
                &device,
            ),
            Tensor::<TestBackend, 2>::from_data(
                TensorData::new(intensity.clone(), [rows, peaks]),
                &device,
            ),
            config,
        );

        let partner_a = partner_a
            .into_data()
            .to_vec::<i32>()
            .expect("partner A indices should be i32");
        let partner_b = partner_b
            .into_data()
            .to_vec::<i32>()
            .expect("partner B indices should be i32");
        let target_delta = target_delta
            .into_data()
            .to_vec::<f32>()
            .expect("target deltas should be f32");

        let scorer =
            LinearCosine::new(config.mz_power, config.intensity_power, config.mz_tolerance)
                .expect("CPU linear cosine config should be valid");
        for anchor in 0..config.batch_items {
            let (expected_a, expected_b, expected_delta) =
                ranking_reference(&mz, &intensity, rows, peaks, anchor, config, &scorer);
            assert_eq!(partner_a[anchor] as usize, expected_a, "anchor {anchor}");
            assert_eq!(partner_b[anchor] as usize, expected_b, "anchor {anchor}");
            assert!(
                (target_delta[anchor] - expected_delta).abs() < 1.0e-5,
                "anchor {anchor}: cuda={} cpu={expected_delta}",
                target_delta[anchor],
            );
        }
    }

    fn ranking_reference(
        mz_values: &[f32],
        intensity_values: &[f32],
        rows: usize,
        peaks: usize,
        anchor: usize,
        config: SimilarityRankingKernelConfig,
        scorer: &LinearCosine,
    ) -> (usize, usize, f32) {
        assert!(config.batch_start + config.batch_items <= rows);
        let mut state = config.seed as u32
            ^ (((anchor as u32) + 1) * 40503)
            ^ ((config.batch_start as u32) >> 16);
        if state == 0 {
            state = 0x6d2b_79f5;
        }
        let mut best_index = anchor;
        let mut worst_index = anchor;
        let mut best_score = f32::NEG_INFINITY;
        let mut worst_score = f32::INFINITY;
        let candidates = config
            .candidates_per_anchor
            .max(2)
            .min(config.batch_items.saturating_sub(1));
        let (anchor_mz, anchor_intensity) = nonzero_row(
            mz_values,
            intensity_values,
            config.batch_start + anchor,
            peaks,
        );
        let anchor_spectrum = TestSpectrum {
            precursor_mz: 1.0,
            mz: &anchor_mz,
            intensity: &anchor_intensity,
        };

        for _ in 0..candidates {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let mut local_partner = (state % ((config.batch_items - 1) as u32)) as usize;
            if local_partner >= anchor {
                local_partner += 1;
            }
            let (partner_mz, partner_intensity) = nonzero_row(
                mz_values,
                intensity_values,
                config.batch_start + local_partner,
                peaks,
            );
            let partner_spectrum = TestSpectrum {
                precursor_mz: 1.0,
                mz: &partner_mz,
                intensity: &partner_intensity,
            };
            let score = scorer
                .similarity(&anchor_spectrum, &partner_spectrum)
                .map(|(score, _)| score as f32)
                .unwrap_or(0.0);
            if score > best_score {
                best_score = score;
                best_index = local_partner;
            }
            if score < worst_score {
                worst_score = score;
                worst_index = local_partner;
            }
        }

        (best_index, worst_index, (best_score - worst_score).max(0.0))
    }

    fn nonzero_row(
        mz_values: &[f32],
        intensity_values: &[f32],
        row: usize,
        peaks: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let start = row * peaks;
        let end = start + peaks;
        mz_values[start..end]
            .iter()
            .copied()
            .zip(intensity_values[start..end].iter().copied())
            .filter(|(_mz, intensity)| *intensity > 0.0)
            .unzip()
    }
}
