#[cfg(all(feature = "cuda-fusion", feature = "train"))]
#[path = "support/gems_common.rs"]
mod gems_common;

#[cfg(all(feature = "cuda-fusion", feature = "train"))]
#[path = "support/gems_peak_set.rs"]
mod gems_peak_set;

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
        AutoencoderTrainingMetricsExt, ConditioningEncoder, PeakSetAutoencoderConfig,
        SpectrumAugmentationConfig, SpectrumTokenizer, SpectrumTokenizerConfig, TokenizedMgfIter,
    };

    use crate::gems_common::{
        CachedTrainingLoaderConfig, GeMSProgress, InnerBackend, RunArgs, TrainingBackend,
        augmentation_config_from_env, auxiliary_loss_config_from_env, open_gems_a10_iter,
        print_run_header, save_model_record, similarity_teacher_config_from_env, warm_start_model,
    };
    use crate::gems_peak_set::cached_tokenized_loader;

    const PEAK_CACHE_DEFAULT_PERCENT: f64 = 100.0;

    pub fn main() -> Result<(), Box<dyn std::error::Error>> {
        let args = RunArgs::from_env("runs/gems-a10-smoke", 32)?;
        std::fs::create_dir_all(&args.output_dir)?;

        let progress = Arc::new(GeMSProgress::new(
            args.train_batches,
            args.valid_batches,
            args.batch_size,
            "tokenizing",
        ));
        let device = burn::backend::cuda::CudaDevice::new(args.device);
        let augmentation =
            augmentation_config_from_env(SpectrumAugmentationConfig::masked_mz_pretraining());
        let tokenizer_config = SpectrumTokenizerConfig {
            max_peaks: args.max_peaks,
            ..SpectrumTokenizerConfig::default()
        };
        let config = PeakSetAutoencoderConfig::twenty_million_run_with_peaks(args.max_peaks);
        let auxiliary = auxiliary_loss_config_from_env(config.auxiliary);
        let similarity_teacher = similarity_teacher_config_from_env(auxiliary)?;
        let train_loader_config = CachedTrainingLoaderConfig::new(
            args.train_gpu_cache_percent(PEAK_CACHE_DEFAULT_PERCENT),
            similarity_teacher,
        );
        let valid_loader_config = CachedTrainingLoaderConfig::new(
            args.valid_gpu_cache_percent(PEAK_CACHE_DEFAULT_PERCENT),
            similarity_teacher,
        );
        let train_builder = args.gems_builder.clone();
        let train_tokenizer_config = tokenizer_config.clone();
        let train_loader = cached_tokenized_loader::<TrainingBackend, _>(
            &args,
            device.clone(),
            progress.train.clone(),
            args.train_start_item(),
            Some(augmentation),
            train_loader_config,
            move || open_records(train_builder.clone(), train_tokenizer_config.clone()),
        );
        let valid_builder = args.gems_builder.clone();
        let valid_tokenizer_config = tokenizer_config.clone();
        let valid_loader = cached_tokenized_loader::<InnerBackend, _>(
            &args,
            device.clone(),
            progress.valid.clone(),
            args.valid_start_item(),
            None,
            valid_loader_config,
            move || open_records(valid_builder.clone(), valid_tokenizer_config.clone()),
        );

        let config = config.with_auxiliary(auxiliary);
        let model = config.init::<TrainingBackend>(&device);
        let model = warm_start_model::<TrainingBackend, _>(
            &progress,
            model,
            &device,
            args.warm_start_model.as_deref(),
            "peak-set",
        )?;
        let parameter_count = model.num_params();
        let optim = AdamConfig::new()
            .with_weight_decay(Some(WeightDecayConfig::new(args.weight_decay as f32)))
            .init();
        let learner = Learner::new(model, optim, ConstantLr::new(args.learning_rate));

        print_run_header(
            "peak-set",
            &args,
            parameter_count,
            PEAK_CACHE_DEFAULT_PERCENT,
            auxiliary,
            similarity_teacher,
            None,
        );
        progress.start_training("starting GeMS peak-set cached training");
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
        progress.finish_training("finished GeMS peak-set cached training");

        save_model_record(
            &progress,
            trained.model.into_record(),
            args.output_dir.join("peak_set_model"),
            "save peak-set model",
            "saved peak-set model record",
        )
    }

    fn open_records(
        builder: mascot_rs::prelude::GemsA10Builder<f32>,
        tokenizer_config: SpectrumTokenizerConfig,
    ) -> spectral_autoencoder::Result<TokenizedMgfIter> {
        let records = open_gems_a10_iter(builder)?;
        Ok(TokenizedMgfIter::from_records(
            records,
            SpectrumTokenizer::new(tokenizer_config),
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
        "train_gems_peak_set requires --no-default-features --features std,cuda-fusion,train,tui"
    );
}
