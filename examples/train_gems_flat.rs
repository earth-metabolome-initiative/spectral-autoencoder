#[cfg(all(feature = "cuda-fusion", feature = "train"))]
#[path = "support/gems_common.rs"]
mod gems_common;

#[cfg(all(feature = "cuda-fusion", feature = "train"))]
#[path = "support/gems_flat.rs"]
mod gems_flat;

#[cfg(all(feature = "cuda-fusion", feature = "train"))]
mod app {
    use std::sync::Arc;

    use burn::{
        module::Module,
        optim::{AdamConfig, decay::WeightDecayConfig, lr_scheduler::constant::ConstantLr},
        record::CompactRecorder,
        train::{Learner, SupervisedTraining},
    };
    use spectral_autoencoder::{
        AutoencoderTrainingMetricsExt, ConditioningEncoder, SpectrumAugmentationConfig,
        SpectrumVectorizer, SpectrumVectorizerConfig, VectorizedMgfIter,
    };

    use crate::gems_common::{
        GeMSProgress, InnerBackend, RunArgs, StreamingTrainingLoaderConfig, TrainingBackend,
        augmentation_config_from_env, auxiliary_loss_config_from_env, open_gems_a10_iter,
        print_streaming_run_header, save_model_record, similarity_teacher_config_from_env,
        warm_start_model,
    };
    use crate::gems_flat::{flat_vector_config_from_env, streaming_vectorized_loader};

    pub fn main() -> Result<(), Box<dyn std::error::Error>> {
        let args = RunArgs::from_env("runs/gems-a10-flat-smoke", 256)?;
        std::fs::create_dir_all(&args.output_dir)?;

        let progress = Arc::new(GeMSProgress::new(
            args.train_batches,
            args.valid_batches,
            args.batch_size,
            "vectorizing",
        ));
        let device = burn::backend::cuda::CudaDevice::new(args.device);
        let augmentation =
            augmentation_config_from_env(SpectrumAugmentationConfig::masked_mz_pretraining());
        let vectorizer_config = SpectrumVectorizerConfig {
            max_peaks: args.max_peaks,
            ..SpectrumVectorizerConfig::default()
        };
        let config = flat_vector_config_from_env(args.max_peaks)?;
        let auxiliary = auxiliary_loss_config_from_env(config.auxiliary);
        let similarity_teacher = similarity_teacher_config_from_env(auxiliary)?;
        let loader_config = StreamingTrainingLoaderConfig::from_env(similarity_teacher)?;
        let train_builder = args.gems_builder.clone();
        let train_vectorizer_config = vectorizer_config.clone();
        let train_loader = streaming_vectorized_loader::<TrainingBackend, _>(
            &args,
            device.clone(),
            progress.train.clone(),
            args.train_start_item(),
            Some(augmentation),
            loader_config,
            move || open_records(train_builder.clone(), train_vectorizer_config.clone()),
        );
        let valid_builder = args.gems_builder.clone();
        let valid_vectorizer_config = vectorizer_config.clone();
        let valid_loader = streaming_vectorized_loader::<InnerBackend, _>(
            &args,
            device.clone(),
            progress.valid.clone(),
            args.valid_start_item(),
            None,
            loader_config,
            move || open_records(valid_builder.clone(), valid_vectorizer_config.clone()),
        );

        let config = config.with_auxiliary(auxiliary);
        let model = config.init::<TrainingBackend>(&device);
        let model = warm_start_model::<TrainingBackend, _>(
            &progress,
            model,
            &device,
            args.warm_start_model.as_deref(),
            "flat-vector",
        )?;
        let parameter_count = model.num_params();
        let optim = AdamConfig::new()
            .with_weight_decay(Some(WeightDecayConfig::new(args.weight_decay as f32)))
            .init();
        let learner = Learner::new(model, optim, ConstantLr::new(args.learning_rate));

        print_streaming_run_header(
            "flat-vector",
            &args,
            parameter_count,
            loader_config,
            auxiliary,
            similarity_teacher,
            Some(config.reconstruction_ordering),
        );
        progress.start_training("starting GeMS flat-vector streaming training");
        let training = SupervisedTraining::new(&args.output_dir, train_loader, valid_loader)
            .num_epochs(args.epochs)
            .with_autoencoder_metrics()
            .summary();
        let training = if args.checkpoints {
            training.with_file_checkpointer(CompactRecorder::new())
        } else {
            training
        };
        let training = if let Some(epoch) = args.resume_epoch {
            training.checkpoint(epoch)
        } else {
            training
        };
        let trained = training.launch(learner);
        progress.finish_training("finished GeMS flat-vector streaming training");

        save_model_record(
            &progress,
            trained.model.into_record(),
            args.output_dir.join("flat_vector_model"),
            "save flat-vector model",
            "saved flat-vector model record",
        )
    }

    fn open_records(
        builder: mascot_rs::prelude::GemsA10Builder<f32>,
        vectorizer_config: SpectrumVectorizerConfig,
    ) -> spectral_autoencoder::Result<VectorizedMgfIter> {
        let records = open_gems_a10_iter(builder)?;
        Ok(VectorizedMgfIter::from_records(
            records,
            SpectrumVectorizer::new(vectorizer_config),
            ConditioningEncoder::default(),
        ))
    }
}

#[cfg(all(feature = "cuda-fusion", feature = "train"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    app::main()
}

#[cfg(not(all(feature = "cuda-fusion", feature = "train")))]
fn main() {
    eprintln!(
        "train_gems_flat requires --no-default-features --features std,cuda-fusion,train,tui"
    );
}
