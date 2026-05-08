use super::api::{
    LinearCosineKernelBackend, LinearCosineKernelConfig, SimilarityRankingKernelConfig,
};
use burn::backend::Autodiff;
use burn::backend::autodiff::checkpoint::{base::Checkpointer, strategy::CheckpointStrategy};
use burn::backend::autodiff::grads::Gradients;
use burn::backend::autodiff::ops::{Backward, Ops, OpsKind};
use burn::tensor::ops::{FloatTensor, IntTensor};

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
        teacher_precursor: FloatTensor<Self>,
        config: SimilarityRankingKernelConfig,
    ) -> (IntTensor<Self>, IntTensor<Self>, FloatTensor<Self>) {
        #[derive(Debug)]
        struct NoGradientBackward;

        impl<B> Backward<B, 3> for NoGradientBackward
        where
            B: LinearCosineKernelBackend,
        {
            type State = ();

            fn backward(
                self,
                _ops: Ops<Self::State, 3>,
                _grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
            }
        }

        match NoGradientBackward
            .prepare::<C>([
                teacher_mz.node.clone(),
                teacher_intensity.node.clone(),
                teacher_precursor.node.clone(),
            ])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => {
                let (partner_a, partner_b, target_delta) =
                    B::linear_cosine_similarity_ranking_kernel(
                        teacher_mz.primitive.clone(),
                        teacher_intensity.primitive.clone(),
                        teacher_precursor.primitive.clone(),
                        config,
                    );

                (partner_a, partner_b, prep.finish((), target_delta))
            }
            OpsKind::UnTracked(prep) => {
                let (partner_a, partner_b, target_delta) =
                    B::linear_cosine_similarity_ranking_kernel(
                        teacher_mz.primitive,
                        teacher_intensity.primitive,
                        teacher_precursor.primitive,
                        config,
                    );

                (partner_a, partner_b, prep.finish(target_delta))
            }
        }
    }
}
