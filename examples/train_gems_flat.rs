#[cfg(all(feature = "cuda-fusion", feature = "train"))]
#[path = "support/gems_common.rs"]
mod gems_common;

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
        SpectrumVectorizer, SpectrumVectorizerConfig, VectorizedMgfIter, vectorized_mgf_paths_iter,
    };

    use crate::gems_common::{
        CachedTrainingLoaderConfig, GeMSProgress, InnerBackend, RunArgs, TrainingBackend,
        augmentation_config_from_env, auxiliary_loss_config_from_env, cached_vectorized_loader,
        flat_vector_config_from_env, print_run_header, save_model_record,
        similarity_teacher_config_from_env, warm_start_model,
    };

    const FLAT_CACHE_DEFAULT_PERCENT: f64 = 100.0;

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
        let train_loader_config = CachedTrainingLoaderConfig::new(
            args.train_gpu_cache_percent(FLAT_CACHE_DEFAULT_PERCENT),
            similarity_teacher,
        );
        let valid_loader_config = CachedTrainingLoaderConfig::new(
            args.valid_gpu_cache_percent(FLAT_CACHE_DEFAULT_PERCENT),
            similarity_teacher,
        );
        let train_records_paths = args.mgf_paths.clone();
        let train_vectorizer_config = vectorizer_config.clone();
        let train_loader = cached_vectorized_loader::<TrainingBackend, _>(
            &args,
            device.clone(),
            progress.train.clone(),
            args.train_start_item(),
            Some(augmentation),
            train_loader_config,
            move || open_records(&train_records_paths, train_vectorizer_config.clone()),
        );
        let valid_records_paths = args.mgf_paths.clone();
        let valid_vectorizer_config = vectorizer_config.clone();
        let valid_loader = cached_vectorized_loader::<InnerBackend, _>(
            &args,
            device.clone(),
            progress.valid.clone(),
            args.valid_start_item(),
            None,
            valid_loader_config,
            move || open_records(&valid_records_paths, valid_vectorizer_config.clone()),
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

        print_run_header(
            "flat-vector",
            &args,
            parameter_count,
            FLAT_CACHE_DEFAULT_PERCENT,
            auxiliary,
            similarity_teacher,
            Some(config.reconstruction_ordering),
        );
        progress.start_training("starting GeMS flat-vector cached training");
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
        progress.finish_training("finished GeMS flat-vector cached training");

        save_model_record(
            &progress,
            trained.model.into_record(),
            args.output_dir.join("flat_vector_model"),
            "save flat-vector model",
            "saved flat-vector model record",
        )
    }

    fn open_records(
        paths: &[std::path::PathBuf],
        vectorizer_config: SpectrumVectorizerConfig,
    ) -> spectral_autoencoder::Result<VectorizedMgfIter> {
        vectorized_mgf_paths_iter(
            paths,
            SpectrumVectorizer::new(vectorizer_config),
            ConditioningEncoder::default(),
        )
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
