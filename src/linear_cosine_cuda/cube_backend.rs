use super::api::{
    LinearCosineKernelBackend, LinearCosineKernelConfig, SimilarityRankingKernelConfig,
};
use super::kernels::{
    linear_cosine_preprocessed_paired_sorted_forward, linear_cosine_similarity_ranking_forward,
};
use burn::tensor::Shape;
use burn::tensor::ops::{FloatTensor, IntTensor};
use burn_cubecl::cubecl::prelude::*;
use burn_cubecl::cubecl::{CubeDim, calculate_cube_count_elemwise};
use burn_cubecl::ops::numeric::empty_device_dtype;
use burn_cubecl::{BoolElement, CubeBackend, CubeRuntime, FloatElement, IntElement};

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
        teacher_precursor: FloatTensor<Self>,
        config: SimilarityRankingKernelConfig,
    ) -> (IntTensor<Self>, IntTensor<Self>, FloatTensor<Self>) {
        teacher_mz.assert_is_on_same_device(&teacher_intensity);
        teacher_mz.assert_is_on_same_device(&teacher_precursor);

        let [teacher_rows, teacher_peaks] = teacher_mz.shape.dims();
        let [precursor_rows] = teacher_precursor.shape.dims();
        assert_eq!(
            teacher_rows, precursor_rows,
            "similarity-ranking precursor cache must have one value per teacher row"
        );
        assert!(
            teacher_peaks <= config.max_peaks,
            "similarity-ranking max_peaks must cover the teacher peak width"
        );
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
            teacher_precursor.as_tensor_arg(1),
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
            config.metric,
            config.max_peaks,
        )
        .expect("similarity-ranking forward kernel launch failed");

        (partner_a, partner_b, target_delta)
    }
}
