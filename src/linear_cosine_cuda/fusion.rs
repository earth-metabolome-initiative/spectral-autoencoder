use super::api::{
    LinearCosineKernelBackend, LinearCosineKernelConfig, SimilarityRankingKernelConfig,
};
use burn::tensor::backend::Backend;
use burn::tensor::ops::{FloatTensor, IntTensor};
use burn::tensor::{Element, Shape};
use burn_fusion::{
    Fusion, FusionBackend,
    stream::{Operation, OperationStreams},
};
use burn_ir::{CustomOpIr, OperationIr, OperationOutput, TensorIr};
use core::marker::PhantomData;

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
        teacher_precursor: FloatTensor<Self>,
        config: SimilarityRankingKernelConfig,
    ) -> (IntTensor<Self>, IntTensor<Self>, FloatTensor<Self>) {
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

        let streams =
            OperationStreams::with_inputs([&teacher_mz, &teacher_intensity, &teacher_precursor]);
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
            &[
                teacher_mz.into_ir(),
                teacher_intensity.into_ir(),
                teacher_precursor.into_ir(),
            ],
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
        let (inputs, outputs) = self.desc.as_fixed::<3, 3>();
        let (partner_a, partner_b, target_delta) = B::linear_cosine_similarity_ranking_kernel(
            handles.get_float_tensor::<B>(&inputs[0]),
            handles.get_float_tensor::<B>(&inputs[1]),
            handles.get_float_tensor::<B>(&inputs[2]),
            self.config,
        );

        handles.register_int_tensor::<B>(&outputs[0].id, partner_a);
        handles.register_int_tensor::<B>(&outputs[1].id, partner_b);
        handles.register_float_tensor::<B>(&outputs[2].id, target_delta);
    }
}
