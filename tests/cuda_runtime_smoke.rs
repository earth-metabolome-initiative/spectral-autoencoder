#![cfg(all(feature = "cuda-fusion", feature = "train"))]

use burn::{data::dataloader::batcher::Batcher, train::TrainStep};
use spectral_autoencoder::{
    PeakSetAutoencoderConfig, TokenizedAutoencoderBatch, TokenizedAutoencoderBatcher,
    TokenizedAutoencoderSample,
};

#[test]
#[ignore = "requires a CUDA GPU; first-run Burn autotune is slow"]
fn peak_set_autoencoder_runs_cuda_fusion_train_step() {
    type Backend = burn::backend::Autodiff<burn::backend::Cuda<f32, i32>>;

    let device = burn::backend::cuda::CudaDevice::new(0);
    let config = PeakSetAutoencoderConfig::symmetric(4, 5, 16, 32, 16, 4, vec![64]);
    let model = config.init::<Backend>(&device);

    let batch: TokenizedAutoencoderBatch<Backend> = TokenizedAutoencoderBatcher.batch(
        vec![TokenizedAutoencoderSample {
            token_features: vec![
                0.05, 1.0, 1.0, 0.0, 1.0, //
                0.12, 0.8, 1.0, 0.0, 1.0, //
                0.31, 0.6, 1.0, 0.0, 1.0, //
                0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            target_pairs: vec![0.05, 1.0, 0.12, 0.8, 0.31, 0.6, 0.0, 0.0],
            peak_mask: vec![1.0, 1.0, 1.0, 0.0],
            padding_mask: vec![false, false, false, true],
            conditions: vec![0.0; 16],
        }],
        &device,
    );

    let output = model.step(batch);
    let loss = output.item.loss.to_data();
    assert_eq!(loss.shape.dims(), [1]);
    assert!(loss.as_slice::<f32>().expect("f32 loss")[0].is_finite());
}
