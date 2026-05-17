use std::{
    collections::hash_map::DefaultHasher,
    fs,
    hash::{Hash, Hasher},
    io,
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Instant, UNIX_EPOCH},
};

use burn::{
    data::dataloader::{DataLoader, DataLoaderIterator, Progress},
    tensor::{Bool, Distribution, Tensor, TensorData, backend::Backend},
};
use spectral_autoencoder::{
    SpectrumAugmentationConfig, TokenizedAutoencoderBatch, TokenizedMgfIter,
};

use mass_spectrometry::burn::AllMetricsBackend;

use crate::common::{
    LoaderProgress, PairSamplingEpoch, PreprocessedCacheOptions, RunArgs, SimilarityTeacherConfig,
    StreamingLoaderOptions, StreamingTrainingLoaderConfig, TeacherGpuCache, TeacherSpectraBuilder,
    TeacherSpectraCache, finish_loader_once, loader_epoch_items, mask_precursor_conditions,
    open_records_or_panic, probability_mask, signed_random, similarity_pair_seed,
    skip_split_records, teacher_similarity_ranking_batch,
};
use crate::streaming::{
    HostWindowPlan, LoaderProfileAccumulator, LoaderWindowProfile, LoaderWorkerError,
    OrderedHostWindowStream, host_window_plan, spawn_ordered_host_workers,
};

const TOKEN_CACHE_MAGIC: &[u8; 8] = b"SATKC01\0";
const TOKEN_CACHE_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy)]
pub struct TokenCacheShape {
    pub max_peaks: usize,
    pub token_feature_width: usize,
    pub target_width: usize,
    pub condition_width: usize,
}

pub struct TokenizedLoaderOptions<B: Backend> {
    pub device: B::Device,
    pub progress: LoaderProgress,
    pub start_item: usize,
    pub augment: Option<SpectrumAugmentationConfig>,
    pub loader_config: StreamingTrainingLoaderConfig,
    pub token_cache_shape: TokenCacheShape,
    pub cache: PreprocessedCacheOptions,
}

pub fn streaming_tokenized_loader<B, Open>(
    args: &RunArgs,
    options: TokenizedLoaderOptions<B>,
    open_records: Open,
) -> Arc<dyn DataLoader<B, TokenizedAutoencoderBatch<B>>>
where
    B: AllMetricsBackend + 'static,
    Open: Fn() -> spectral_autoencoder::Result<TokenizedMgfIter> + Send + Sync + Clone + 'static,
{
    let TokenizedLoaderOptions {
        device,
        progress,
        start_item,
        augment,
        loader_config,
        token_cache_shape,
        cache,
    } = options;
    let preprocessed_cache_path = token_preprocessed_cache_path(
        args,
        &cache,
        start_item,
        progress.max_batches,
        token_cache_shape,
    );
    let loader = Arc::new(StreamingTokenizedMgfLoader::new(
        StreamingLoaderOptions {
            mgf_source: args.mgf_source.clone(),
            batch_size: args.batch_size,
            max_batches: progress.max_batches,
            start_item,
            window_items: loader_config.window_items(args.batch_size, progress.max_batches),
            device,
            progress,
            augment,
            similarity_teacher: loader_config.similarity_teacher,
            randomize_pair_sampling: augment.is_some(),
        },
        preprocessed_cache_path,
        cache.refresh,
        loader_config,
        token_cache_shape,
        open_records,
    ));
    loader.ensure_preprocessed_cache_file();
    loader
}

fn token_preprocessed_cache_path(
    args: &RunArgs,
    cache: &PreprocessedCacheOptions,
    start_item: usize,
    max_batches: usize,
    shape: TokenCacheShape,
) -> Option<PathBuf> {
    assert!(
        cache.enabled,
        "--preprocessed-cache=false is not supported; the host-worker loader requires the token cache"
    );

    let default_cache_dir = format!(
        "datasets/gems-a10-top-{}-peaks/preprocessed-token",
        args.max_peaks
    );
    let cache_dir = cache
        .dir
        .clone()
        .unwrap_or_else(|| PathBuf::from(default_cache_dir));
    let total_items = args.batch_size.saturating_mul(max_batches);
    let fingerprint = token_preprocessed_cache_fingerprint(args, start_item, total_items, shape);
    Some(cache_dir.join(format!(
        "token-v{TOKEN_CACHE_VERSION}-{fingerprint:016x}.saetc"
    )))
}

fn token_preprocessed_cache_fingerprint(
    args: &RunArgs,
    start_item: usize,
    total_items: usize,
    shape: TokenCacheShape,
) -> u64 {
    let mut hasher = DefaultHasher::new();
    TOKEN_CACHE_MAGIC.hash(&mut hasher);
    TOKEN_CACHE_VERSION.hash(&mut hasher);
    args.mgf_source.hash(&mut hasher);
    args.max_peaks.hash(&mut hasher);
    start_item.hash(&mut hasher);
    total_items.hash(&mut hasher);
    shape.max_peaks.hash(&mut hasher);
    shape.token_feature_width.hash(&mut hasher);
    shape.target_width.hash(&mut hasher);
    shape.condition_width.hash(&mut hasher);
    for path in &args.mgf_paths {
        path.as_os_str().to_string_lossy().hash(&mut hasher);
        if let Ok(metadata) = fs::metadata(path) {
            metadata.len().hash(&mut hasher);
            if let Ok(modified) = metadata.modified()
                && let Ok(duration) = modified.duration_since(UNIX_EPOCH)
            {
                duration.as_secs().hash(&mut hasher);
                duration.subsec_nanos().hash(&mut hasher);
            }
        }
    }
    hasher.finish()
}

type AugmentedTokenBatch<B> = (
    Tensor<B, 3>,
    Tensor<B, 2>,
    Tensor<B, 2, Bool>,
    Tensor<B, 2>,
    Tensor<B, 2>,
    Tensor<B, 2>,
    Tensor<B, 2>,
);

fn augment_token_batch<B: Backend>(
    token_features: Tensor<B, 3>,
    peak_mask: Tensor<B, 2>,
    padding_mask: Tensor<B, 2, Bool>,
    conditions: Tensor<B, 2>,
    config: SpectrumAugmentationConfig,
) -> AugmentedTokenBatch<B> {
    let [batch_size, max_peaks, feature_width] = token_features.dims();
    if config.is_disabled() {
        let device = token_features.device();
        return (
            token_features,
            peak_mask,
            padding_mask,
            conditions,
            Tensor::<B, 2>::zeros([batch_size, 1], &device),
            Tensor::<B, 2>::zeros([batch_size, max_peaks], &device),
            Tensor::<B, 2>::zeros([batch_size, max_peaks], &device),
        );
    }

    let device = token_features.device();
    let mut mz = token_features
        .clone()
        .narrow(2, 0, 1)
        .reshape([batch_size, max_peaks]);
    let mut intensity = token_features
        .clone()
        .narrow(2, 1, 1)
        .reshape([batch_size, max_peaks]);
    let mut presence = token_features
        .clone()
        .narrow(2, 2, 1)
        .reshape([batch_size, max_peaks]);
    let mut input_peak_mask = peak_mask;
    let mut input_padding_mask = padding_mask;
    let original_active = input_peak_mask.clone().greater_elem(0.0);
    let empty_original_slots = original_active.clone().bool_not();
    let real_peak_present = original_active
        .clone()
        .float()
        .sum_dim(1)
        .greater_elem(0.0)
        .expand([batch_size, max_peaks]);
    let source_min_mz = mz
        .clone()
        .mask_fill(original_active.clone().bool_not(), 1.0)
        .min_dim(1)
        .expand([batch_size, max_peaks]);
    let source_max_mz = mz
        .clone()
        .mask_fill(original_active.clone().bool_not(), 0.0)
        .max_dim(1)
        .expand([batch_size, max_peaks]);
    let source_min_intensity = intensity
        .clone()
        .mask_fill(original_active.clone().bool_not(), 1.0)
        .min_dim(1)
        .expand([batch_size, max_peaks]);
    let source_max_intensity = intensity
        .clone()
        .mask_fill(original_active.clone().bool_not(), 0.0)
        .max_dim(1)
        .expand([batch_size, max_peaks]);
    let mut active = original_active.clone();
    let mut masked_peak_mask =
        Tensor::<B, 2>::zeros([batch_size, max_peaks], &device).greater_elem(0.0);

    if let Some(dropout_mask) = probability_mask(
        [batch_size, max_peaks],
        &device,
        config.peak_dropout_probability,
    ) {
        let dropout_mask = dropout_mask.bool_and(active.clone());
        active = active.bool_and(dropout_mask.clone().bool_not());
        masked_peak_mask = masked_peak_mask.bool_or(dropout_mask.clone());
        mz = mz.mask_fill(dropout_mask.clone(), 0.0);
        intensity = intensity.mask_fill(dropout_mask.clone(), 0.0);
        presence = presence.mask_fill(dropout_mask.clone(), 0.0);
        input_peak_mask = input_peak_mask.mask_fill(dropout_mask.clone(), 0.0);
        input_padding_mask = input_padding_mask.bool_or(dropout_mask);
    }

    let mut mz_mask = Tensor::<B, 2>::zeros([batch_size, max_peaks], &device).greater_elem(0.0);
    if let Some(mask) =
        probability_mask([batch_size, max_peaks], &device, config.mz_mask_probability)
    {
        mz_mask = mask.bool_and(active.clone());
        masked_peak_mask = masked_peak_mask.bool_or(mz_mask.clone());
        mz = mz.mask_fill(mz_mask.clone(), 0.0);
    }

    let mz_editable = active.clone().bool_and(mz_mask.clone().bool_not());
    let mz_shift_range = config.mz_shift_range.max(0.0);
    let mz_jitter_range = config.mz_jitter_range.max(0.0);
    if mz_shift_range > 0.0 || mz_jitter_range > 0.0 {
        let shift =
            signed_random([batch_size, 1], &device, mz_shift_range).expand([batch_size, max_peaks]);
        let jitter = signed_random([batch_size, max_peaks], &device, mz_jitter_range);
        let shifted = (mz.clone() + shift + jitter).clamp_min(0.0).clamp_max(1.0);
        mz = mz.mask_where(mz_editable, shifted);
    }

    let intensity_jitter = config.intensity_jitter_fraction.max(0.0);
    if intensity_jitter > 0.0 {
        let scale = (signed_random([batch_size, max_peaks], &device, intensity_jitter) + 1.0)
            .clamp_min(0.0);
        let jittered = (intensity.clone() * scale).clamp_max(1.0);
        intensity = intensity.mask_where(active.clone(), jittered);
    }

    let mut intruder_peak_mask =
        Tensor::<B, 2>::zeros([batch_size, max_peaks], &device).greater_elem(0.0);
    if let Some(mask) = probability_mask(
        [batch_size, max_peaks],
        &device,
        config.intruder_peak_probability,
    ) {
        intruder_peak_mask = mask
            .bool_and(empty_original_slots)
            .bool_and(real_peak_present);
        let random_mz = Tensor::<B, 2>::random(
            [batch_size, max_peaks],
            Distribution::Uniform(0.0, 1.0),
            &device,
        );
        let random_intensity = Tensor::<B, 2>::random(
            [batch_size, max_peaks],
            Distribution::Uniform(0.0, 1.0),
            &device,
        );
        let intruder_mz = (source_min_mz.clone()
            + random_mz * (source_max_mz - source_min_mz.clone()))
        .clamp_min(0.0)
        .clamp_max(1.0);
        let intruder_intensity = (source_min_intensity.clone()
            + random_intensity * (source_max_intensity - source_min_intensity.clone()))
        .clamp_min(0.0)
        .clamp_max(1.0);
        mz = mz.mask_where(intruder_peak_mask.clone(), intruder_mz);
        intensity = intensity.mask_where(intruder_peak_mask.clone(), intruder_intensity);
        presence = presence.mask_fill(intruder_peak_mask.clone(), 1.0);
        input_peak_mask = input_peak_mask.mask_fill(intruder_peak_mask.clone(), 1.0);
        input_padding_mask = input_padding_mask.bool_and(intruder_peak_mask.clone().bool_not());
        active = active.bool_or(intruder_peak_mask.clone());
    }

    let mut feature_parts = vec![
        mz.clone().unsqueeze_dim::<3>(2),
        intensity.unsqueeze_dim::<3>(2),
        presence.unsqueeze_dim::<3>(2),
    ];
    let token_features = if feature_width > 3 {
        let fourier = fourier_features(mz, feature_width - 3, active.bool_and(mz_mask.bool_not()));
        feature_parts.push(fourier);
        Tensor::cat(feature_parts, 2)
    } else {
        Tensor::cat(feature_parts, 2)
    };

    let (conditions, masked_precursor_mask) =
        mask_precursor_conditions(conditions, config.precursor_mask_probability);

    (
        token_features,
        input_peak_mask,
        input_padding_mask,
        conditions,
        masked_precursor_mask,
        masked_peak_mask.float(),
        intruder_peak_mask.float(),
    )
}

fn fourier_features<B: Backend>(
    mz: Tensor<B, 2>,
    feature_count: usize,
    active: Tensor<B, 2, Bool>,
) -> Tensor<B, 3> {
    let [batch_size, max_peaks] = mz.dims();
    let mut features = Vec::with_capacity(feature_count);
    let fourier_pairs = feature_count / 2;
    let mut frequency = 1.0_f32;
    for _ in 0..fourier_pairs {
        let angle = mz.clone() * (std::f32::consts::TAU * frequency);
        features.push(angle.clone().sin().unsqueeze_dim::<3>(2));
        features.push(angle.cos().unsqueeze_dim::<3>(2));
        frequency *= 2.0;
    }
    if feature_count % 2 == 1 {
        features.push(
            Tensor::<B, 2>::zeros([batch_size, max_peaks], &mz.device()).unsqueeze_dim::<3>(2),
        );
    }

    let active = active
        .unsqueeze_dim::<3>(2)
        .expand([batch_size, max_peaks, feature_count]);
    Tensor::cat(features, 2).mask_fill(active.bool_not(), 0.0)
}

struct StreamingTokenizedMgfLoader<B, Open>
where
    B: Backend,
{
    mgf_source: String,
    progress: LoaderProgress,
    batch_size: usize,
    max_batches: usize,
    epoch_items: usize,
    start_item: usize,
    window_items: usize,
    loader_workers: usize,
    host_prefetch_windows: usize,
    loader_profile_every: usize,
    loader_profile_sink: crate::streaming::LoaderProfileSink,
    preprocessed_cache_path: Option<PathBuf>,
    preprocessed_cache_refresh: bool,
    cache_total_items: usize,
    cache_item_offset: usize,
    token_cache_shape: TokenCacheShape,
    device: B::Device,
    augment: Option<SpectrumAugmentationConfig>,
    similarity_teacher: SimilarityTeacherConfig,
    open_records: Open,
    disk_cache: Arc<Mutex<Option<TokenCacheFileMeta>>>,
    pair_sampling_epoch: PairSamplingEpoch,
}

impl<B, Open> Clone for StreamingTokenizedMgfLoader<B, Open>
where
    B: Backend,
    B::Device: Clone,
    Open: Clone,
{
    fn clone(&self) -> Self {
        Self {
            mgf_source: self.mgf_source.clone(),
            progress: self.progress.clone(),
            batch_size: self.batch_size,
            max_batches: self.max_batches,
            epoch_items: self.epoch_items,
            start_item: self.start_item,
            window_items: self.window_items,
            loader_workers: self.loader_workers,
            host_prefetch_windows: self.host_prefetch_windows,
            loader_profile_every: self.loader_profile_every,
            loader_profile_sink: self.loader_profile_sink.clone(),
            preprocessed_cache_path: self.preprocessed_cache_path.clone(),
            preprocessed_cache_refresh: self.preprocessed_cache_refresh,
            cache_total_items: self.cache_total_items,
            cache_item_offset: self.cache_item_offset,
            token_cache_shape: self.token_cache_shape,
            device: self.device.clone(),
            augment: self.augment,
            similarity_teacher: self.similarity_teacher,
            open_records: self.open_records.clone(),
            disk_cache: self.disk_cache.clone(),
            pair_sampling_epoch: self.pair_sampling_epoch.clone(),
        }
    }
}

impl<B, Open> StreamingTokenizedMgfLoader<B, Open>
where
    B: Backend,
{
    fn new(
        options: StreamingLoaderOptions<B>,
        preprocessed_cache_path: Option<PathBuf>,
        preprocessed_cache_refresh: bool,
        loader_config: StreamingTrainingLoaderConfig,
        token_cache_shape: TokenCacheShape,
        open_records: Open,
    ) -> Self {
        let epoch_items = loader_epoch_items(options.batch_size, options.max_batches);
        let cache_total_items = epoch_items;
        Self {
            mgf_source: options.mgf_source,
            progress: options.progress,
            batch_size: options.batch_size,
            max_batches: options.max_batches,
            epoch_items,
            start_item: options.start_item,
            window_items: options.window_items,
            loader_workers: loader_config.loader_workers,
            host_prefetch_windows: loader_config.host_prefetch_windows,
            loader_profile_every: loader_config.loader_profile_every,
            loader_profile_sink: loader_config.loader_profile_sink.clone(),
            preprocessed_cache_path,
            preprocessed_cache_refresh,
            cache_total_items,
            cache_item_offset: 0,
            token_cache_shape,
            device: options.device,
            augment: options.augment,
            similarity_teacher: options.similarity_teacher,
            open_records,
            disk_cache: Arc::new(Mutex::new(None)),
            pair_sampling_epoch: PairSamplingEpoch::new(options.randomize_pair_sampling),
        }
    }
}

impl<B, Open> StreamingTokenizedMgfLoader<B, Open>
where
    B: Backend,
    Open: Fn() -> spectral_autoencoder::Result<TokenizedMgfIter> + Send + Sync + Clone + 'static,
{
    fn ensure_preprocessed_cache_file(&self) {
        let Some(path) = &self.preprocessed_cache_path else {
            return;
        };
        let total_items = self.cache_total_items;
        let refresh = self.preprocessed_cache_refresh;

        if !refresh && path.try_exists().unwrap_or(false) {
            if let Err(error) = self.token_cache_file_meta(path, total_items) {
                panic!(
                    "invalid preprocessed token cache {}: {error}; pass --preprocessed-cache-refresh to rebuild it",
                    path.display()
                );
            }
            return;
        }

        if let Err(error) = self.write_token_cache_file_streaming(path, total_items) {
            panic!(
                "failed to prepare preprocessed token cache {}: {error}",
                path.display()
            );
        }
        let metadata = self
            .token_cache_file_meta(path, total_items)
            .unwrap_or_else(|error| {
                panic!(
                    "failed to reopen preprocessed token cache {}: {error}",
                    path.display()
                )
            });
        *self
            .disk_cache
            .lock()
            .expect("token disk cache metadata lock should not be poisoned") = Some(metadata);
    }

    fn token_cache_file_meta(
        &self,
        path: &Path,
        expected_items: usize,
    ) -> io::Result<TokenCacheFileMeta> {
        if let Some(metadata) = self
            .disk_cache
            .lock()
            .expect("token disk cache metadata lock should not be poisoned")
            .clone()
        {
            return Ok(metadata);
        }
        let metadata = read_token_cache_file_meta(path, expected_items, self.token_cache_shape)?;
        *self
            .disk_cache
            .lock()
            .expect("token disk cache metadata lock should not be poisoned") =
            Some(metadata.clone());
        Ok(metadata)
    }

    fn token_host_window_stream(&self) -> OrderedHostWindowStream<TokenHostWindow> {
        let path = self.preprocessed_cache_path.as_ref().unwrap_or_else(|| {
            panic!(
                "token host-worker loading requires the preprocessed cache for {}",
                self.mgf_source
            )
        });
        // Refresh is consumed during cache preparation; iteration only needs the rebuilt file.
        if !path.try_exists().unwrap_or(false) {
            panic!(
                "preprocessed token cache {} is unavailable; it should have been prepared before iteration",
                path.display()
            );
        }
        let total_items = self.epoch_items;
        let metadata = self
            .token_cache_file_meta(path, self.cache_total_items)
            .unwrap_or_else(|error| {
                panic!(
                    "invalid preprocessed token cache {}: {error}",
                    path.display()
                )
            });
        let plans = host_window_plan(total_items, self.window_items);
        let build_context = TokenHostWindowBuildContext {
            metadata,
            cache_item_offset: self.cache_item_offset,
            similarity_teacher: self.similarity_teacher,
            progress: self.progress.clone(),
            batch_size: self.batch_size,
            max_batches: self.max_batches,
            epoch_items: self.epoch_items,
            prefix: format!("{} disk", self.progress.label),
        };

        spawn_ordered_host_workers(
            plans,
            self.loader_workers,
            self.host_prefetch_windows,
            move |_worker_id, plan| build_token_host_window(&build_context, plan),
        )
    }

    fn write_token_cache_file_streaming(&self, path: &Path, total_items: usize) -> io::Result<()> {
        let mut records = open_records_or_panic(&self.open_records, "tokenized", &self.mgf_source);
        self.skip_preload_offset(&mut records);

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("token-cache.saetc");
        let tmp_path = path.with_file_name(format!("{file_name}.tmp-{}", std::process::id()));
        let tmp_targets_path =
            path.with_file_name(format!("{file_name}.targets.tmp-{}", std::process::id()));
        let tmp_peak_mask_path =
            path.with_file_name(format!("{file_name}.peak-mask.tmp-{}", std::process::id()));
        let tmp_padding_path =
            path.with_file_name(format!("{file_name}.padding.tmp-{}", std::process::id()));
        let tmp_conditions_path =
            path.with_file_name(format!("{file_name}.conditions.tmp-{}", std::process::id()));

        let result = (|| -> io::Result<()> {
            let mut writer =
                BufWriter::with_capacity(8 * 1024 * 1024, fs::File::create(&tmp_path)?);
            let mut target_writer =
                BufWriter::with_capacity(8 * 1024 * 1024, fs::File::create(&tmp_targets_path)?);
            let mut peak_mask_writer =
                BufWriter::with_capacity(8 * 1024 * 1024, fs::File::create(&tmp_peak_mask_path)?);
            let mut padding_writer =
                BufWriter::with_capacity(8 * 1024 * 1024, fs::File::create(&tmp_padding_path)?);
            let mut conditions_writer =
                BufWriter::with_capacity(8 * 1024 * 1024, fs::File::create(&tmp_conditions_path)?);
            let mut written_items = 0usize;
            let mut wrote_header = false;

            while written_items < total_items {
                let target_items = self.window_items.min(total_items - written_items);
                let cache = self
                    .read_token_cpu_cache(
                        &mut records,
                        target_items,
                        written_items / self.batch_size,
                    )
                    .ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::UnexpectedEof,
                            format!(
                                "MGF ended after {written_items} spectra, expected {total_items}"
                            ),
                        )
                    })?;

                if !wrote_header {
                    validate_token_cache_shape(&cache, self.token_cache_shape)?;
                    writer.write_all(TOKEN_CACHE_MAGIC)?;
                    write_u32(&mut writer, TOKEN_CACHE_VERSION)?;
                    write_u64(&mut writer, total_items as u64)?;
                    write_u64(&mut writer, cache.max_peaks as u64)?;
                    write_u64(&mut writer, cache.token_feature_width as u64)?;
                    write_u64(&mut writer, cache.target_width as u64)?;
                    write_u64(&mut writer, cache.condition_width as u64)?;
                    wrote_header = true;
                } else {
                    validate_token_cache_shape(&cache, self.token_cache_shape)?;
                }

                write_f32_slice_untracked(&mut writer, &cache.token_features)?;
                write_f32_slice_untracked(&mut target_writer, &cache.target_pairs)?;
                write_f32_slice_untracked(&mut peak_mask_writer, &cache.peak_mask)?;
                write_bool_slice_untracked(&mut padding_writer, &cache.padding_mask)?;
                write_f32_slice_untracked(&mut conditions_writer, &cache.conditions)?;
                written_items += cache.items;
            }

            target_writer.flush()?;
            peak_mask_writer.flush()?;
            padding_writer.flush()?;
            conditions_writer.flush()?;
            drop(target_writer);
            drop(peak_mask_writer);
            drop(padding_writer);
            drop(conditions_writer);
            append_file(&tmp_targets_path, &mut writer)?;
            append_file(&tmp_peak_mask_path, &mut writer)?;
            append_file(&tmp_padding_path, &mut writer)?;
            append_file(&tmp_conditions_path, &mut writer)?;
            writer.flush()
        })();

        match result {
            Ok(()) => {
                fs::rename(&tmp_path, path)?;
                let _ = fs::remove_file(&tmp_targets_path);
                let _ = fs::remove_file(&tmp_peak_mask_path);
                let _ = fs::remove_file(&tmp_padding_path);
                let _ = fs::remove_file(&tmp_conditions_path);
                Ok(())
            }
            Err(error) => {
                let _ = fs::remove_file(&tmp_path);
                let _ = fs::remove_file(&tmp_targets_path);
                let _ = fs::remove_file(&tmp_peak_mask_path);
                let _ = fs::remove_file(&tmp_padding_path);
                let _ = fs::remove_file(&tmp_conditions_path);
                Err(error)
            }
        }
    }

    fn skip_preload_offset(&self, records: &mut TokenizedMgfIter) {
        skip_split_records(records, self.start_item, &self.progress, None);
    }

    fn read_token_cpu_cache(
        &self,
        records: &mut TokenizedMgfIter,
        target_items: usize,
        batches_processed: usize,
    ) -> Option<TokenCpuCache> {
        if target_items == 0 {
            return None;
        }

        let mut token_features = Vec::new();
        let mut target_pairs = Vec::new();
        let mut peak_mask = Vec::new();
        let mut padding_mask = Vec::new();
        let mut conditions = Vec::new();
        let mut max_peaks = 0usize;
        let mut token_feature_width = 0usize;
        let mut target_width = 0usize;
        let mut condition_width = 0usize;
        let mut items = 0usize;

        while items < target_items {
            let sample = match records.next() {
                Some(Ok(sample)) => sample,
                Some(Err(error)) => {
                    panic!("failed to tokenize MGF record: {error}");
                }
                None => break,
            };

            if items == 0 {
                max_peaks = sample.peak_mask.len();
                token_feature_width = sample.token_features.len() / max_peaks;
                target_width = sample.target_pairs.len();
                condition_width = sample.conditions.len();
                token_features.reserve(target_items * max_peaks * token_feature_width);
                target_pairs.reserve(target_items * target_width);
                peak_mask.reserve(target_items * max_peaks);
                padding_mask.reserve(target_items * max_peaks);
                conditions.reserve(target_items * condition_width);
            }
            token_features.extend(sample.token_features);
            target_pairs.extend(sample.target_pairs);
            peak_mask.extend(sample.peak_mask);
            padding_mask.extend(sample.padding_mask);
            conditions.extend(sample.conditions);
            items += 1;
            if items.is_multiple_of(self.batch_size) || items == target_items {
                self.progress.cache_filling(
                    items,
                    target_items,
                    batches_processed,
                    self.max_batches,
                );
            }
        }

        (items > 0).then_some(TokenCpuCache {
            token_features,
            target_pairs,
            peak_mask,
            padding_mask,
            conditions,
            items,
            max_peaks,
            token_feature_width,
            target_width,
            condition_width,
        })
    }

    fn move_token_host_window_to_gpu(&self, host_window: &TokenHostWindow) -> TokenGpuCache<B> {
        move_token_host_window_to_gpu(host_window, &self.device)
    }
}

impl<B, Open> DataLoader<B, TokenizedAutoencoderBatch<B>> for StreamingTokenizedMgfLoader<B, Open>
where
    B: AllMetricsBackend + 'static,
    Open: Fn() -> spectral_autoencoder::Result<TokenizedMgfIter> + Send + Sync + Clone + 'static,
{
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<TokenizedAutoencoderBatch<B>> + 'a> {
        self.progress.start_epoch(self.epoch_items);
        let epoch_index = self.pair_sampling_epoch.next();

        Box::new(StreamingTokenizedMgfIter {
            loader: self,
            host_windows: self.token_host_window_stream(),
            cache: None,
            cache_offset: 0,
            batches_processed: 0,
            items_processed: 0,
            epoch_index,
            profile: LoaderProfileAccumulator::new(
                format!("{} token", self.progress.label),
                self.loader_profile_every,
                self.loader_profile_sink.clone(),
            ),
            finished: false,
        })
    }

    fn num_items(&self) -> usize {
        self.epoch_items
    }

    fn to_device(
        &self,
        device: &B::Device,
    ) -> Arc<dyn DataLoader<B, TokenizedAutoencoderBatch<B>>> {
        Arc::new(Self {
            device: device.clone(),
            ..self.clone()
        })
    }

    fn slice(
        &self,
        start: usize,
        end: usize,
    ) -> Arc<dyn DataLoader<B, TokenizedAutoencoderBatch<B>>> {
        let start_item = start.min(self.epoch_items);
        let end_item = end.min(self.epoch_items).max(start_item);
        let epoch_items = end_item - start_item;
        let max_batches = epoch_items.div_ceil(self.batch_size);
        let mut progress = self.progress.clone_for_slice();
        progress.max_batches = max_batches;
        Arc::new(Self {
            max_batches,
            epoch_items,
            start_item: self.start_item + start_item,
            cache_item_offset: self.cache_item_offset + start_item,
            progress,
            ..self.clone()
        })
    }
}

struct TokenCpuCache {
    token_features: Vec<f32>,
    target_pairs: Vec<f32>,
    peak_mask: Vec<f32>,
    padding_mask: Vec<bool>,
    conditions: Vec<f32>,
    items: usize,
    max_peaks: usize,
    token_feature_width: usize,
    target_width: usize,
    condition_width: usize,
}

struct TokenHostWindow {
    token_features: Vec<f32>,
    target_pairs: Vec<f32>,
    peak_mask: Vec<f32>,
    padding_mask: Vec<bool>,
    conditions: Vec<f32>,
    teacher: Option<Arc<TeacherSpectraCache>>,
    profile: LoaderWindowProfile,
    items: usize,
    max_peaks: usize,
    token_feature_width: usize,
    target_width: usize,
    condition_width: usize,
}

#[derive(Clone)]
struct TokenCacheFileMeta {
    path: PathBuf,
    items: usize,
    max_peaks: usize,
    token_feature_width: usize,
    target_width: usize,
    condition_width: usize,
    token_features_offset: u64,
    target_pairs_offset: u64,
    peak_mask_offset: u64,
    padding_mask_offset: u64,
    conditions_offset: u64,
}

struct TokenHostWindowBuildContext {
    metadata: TokenCacheFileMeta,
    cache_item_offset: usize,
    similarity_teacher: SimilarityTeacherConfig,
    batch_size: usize,
    max_batches: usize,
    epoch_items: usize,
    progress: LoaderProgress,
    prefix: String,
}

fn build_token_host_window(
    context: &TokenHostWindowBuildContext,
    plan: HostWindowPlan,
) -> Result<TokenHostWindow, LoaderWorkerError> {
    let producer_start = Instant::now();
    let disk_start = Instant::now();
    let cache = read_token_cpu_cache_window_file(
        &context.metadata,
        context.cache_item_offset + plan.start_item,
        plan.items,
        context.prefix.clone(),
    )?;
    let disk_read = disk_start.elapsed();
    if cache.items != plan.items {
        return Err(LoaderWorkerError::new(format!(
            "token cache ended after {} items for planned {}-item window at {}",
            cache.items, plan.items, plan.start_item
        )));
    }

    let host_pack_start = Instant::now();
    context.progress.cache_filling(
        plan.start_item + cache.items,
        context.epoch_items,
        (plan.start_item + cache.items).div_ceil(context.batch_size),
        context.max_batches,
    );
    let host_pack = host_pack_start.elapsed();

    let teacher_start = Instant::now();
    let teacher = context.similarity_teacher.enabled().then(|| {
        Arc::new(teacher_spectra_cache_from_target_pairs(
            context.similarity_teacher,
            &cache.target_pairs,
            &cache.conditions,
            cache.items,
            cache.target_width,
            cache.condition_width,
        ))
    });
    let teacher_build = teacher_start.elapsed();

    Ok(TokenHostWindow {
        token_features: cache.token_features,
        target_pairs: cache.target_pairs,
        peak_mask: cache.peak_mask,
        padding_mask: cache.padding_mask,
        conditions: cache.conditions,
        teacher,
        profile: LoaderWindowProfile {
            disk_read,
            host_pack,
            teacher_build,
            producer_total: producer_start.elapsed(),
            ..LoaderWindowProfile::default()
        },
        items: cache.items,
        max_peaks: cache.max_peaks,
        token_feature_width: cache.token_feature_width,
        target_width: cache.target_width,
        condition_width: cache.condition_width,
    })
}

fn teacher_spectra_cache_from_target_pairs(
    config: SimilarityTeacherConfig,
    target_pairs: &[f32],
    conditions: &[f32],
    items: usize,
    target_width: usize,
    condition_width: usize,
) -> TeacherSpectraCache {
    let mut builder = TeacherSpectraBuilder::new(config, items);
    for item in 0..items {
        let start = item * target_width;
        let end = start + target_width;
        let condition_start = item * condition_width;
        let condition_end = condition_start + condition_width;
        builder.push_pairs(
            &target_pairs[start..end],
            &conditions[condition_start..condition_end],
        );
    }
    builder.finish()
}

fn move_token_host_window_to_gpu<B: Backend>(
    host_window: &TokenHostWindow,
    device: &B::Device,
) -> TokenGpuCache<B> {
    let teacher_gpu = host_window.teacher.as_ref().and_then(|teacher| {
        TeacherGpuCache::from_cpu_window(teacher, 0, host_window.items, device)
    });

    TokenGpuCache {
        token_features: Tensor::<B, 3>::from_data(
            TensorData::new(
                host_window.token_features.clone(),
                [
                    host_window.items,
                    host_window.max_peaks,
                    host_window.token_feature_width,
                ],
            ),
            device,
        ),
        target_pairs: Tensor::<B, 2>::from_data(
            TensorData::new(
                host_window.target_pairs.clone(),
                [host_window.items, host_window.target_width],
            ),
            device,
        ),
        peak_mask: Tensor::<B, 2>::from_data(
            TensorData::new(
                host_window.peak_mask.clone(),
                [host_window.items, host_window.max_peaks],
            ),
            device,
        ),
        target_peak_mask: Tensor::<B, 2>::from_data(
            TensorData::new(
                host_window.peak_mask.clone(),
                [host_window.items, host_window.max_peaks],
            ),
            device,
        ),
        padding_mask: Tensor::<B, 2, Bool>::from_bool(
            TensorData::new(
                host_window.padding_mask.clone(),
                [host_window.items, host_window.max_peaks],
            ),
            device,
        ),
        conditions: Tensor::<B, 2>::from_data(
            TensorData::new(
                host_window.conditions.clone(),
                [host_window.items, host_window.condition_width],
            ),
            device,
        ),
        teacher_gpu,
        items: host_window.items,
    }
}

fn validate_token_cache_shape(cache: &TokenCpuCache, shape: TokenCacheShape) -> io::Result<()> {
    if cache.max_peaks != shape.max_peaks
        || cache.token_feature_width != shape.token_feature_width
        || cache.target_width != shape.target_width
        || cache.condition_width != shape.condition_width
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "token cache shape changed: got max_peaks={} feature_width={} target_width={} condition_width={}, expected max_peaks={} feature_width={} target_width={} condition_width={}",
                cache.max_peaks,
                cache.token_feature_width,
                cache.target_width,
                cache.condition_width,
                shape.max_peaks,
                shape.token_feature_width,
                shape.target_width,
                shape.condition_width,
            ),
        ));
    }
    Ok(())
}

fn read_token_cache_file_meta(
    path: &Path,
    expected_items: usize,
    expected_shape: TokenCacheShape,
) -> io::Result<TokenCacheFileMeta> {
    let mut reader = BufReader::with_capacity(8 * 1024 * 1024, fs::File::open(path)?);
    let mut magic = [0_u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != TOKEN_CACHE_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "unexpected token-cache magic",
        ));
    }

    let version = read_u32(&mut reader)?;
    if version != TOKEN_CACHE_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported token-cache version {version}"),
        ));
    }

    let items = read_usize(&mut reader, "items")?;
    let max_peaks = read_usize(&mut reader, "max peaks")?;
    let token_feature_width = read_usize(&mut reader, "token feature width")?;
    let target_width = read_usize(&mut reader, "target width")?;
    let condition_width = read_usize(&mut reader, "condition width")?;
    if items != expected_items {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("cache has {items} items, expected {expected_items}"),
        ));
    }
    let actual = TokenCacheShape {
        max_peaks,
        token_feature_width,
        target_width,
        condition_width,
    };
    if actual.max_peaks != expected_shape.max_peaks
        || actual.token_feature_width != expected_shape.token_feature_width
        || actual.target_width != expected_shape.target_width
        || actual.condition_width != expected_shape.condition_width
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "token-cache shape does not match the active tokenizer",
        ));
    }

    let token_features_offset = reader.stream_position()?;
    let target_pairs_offset = token_features_offset
        .checked_add(bytes_for_floats(
            items
                .saturating_mul(max_peaks)
                .saturating_mul(token_feature_width),
        )?)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "token-cache offset overflow"))?;
    let peak_mask_offset = target_pairs_offset
        .checked_add(bytes_for_floats(items.saturating_mul(target_width))?)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "token-cache offset overflow"))?;
    let padding_mask_offset = peak_mask_offset
        .checked_add(bytes_for_floats(items.saturating_mul(max_peaks))?)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "token-cache offset overflow"))?;
    let conditions_offset = padding_mask_offset
        .checked_add(bytes_for_bools(items.saturating_mul(max_peaks))?)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "token-cache offset overflow"))?;

    Ok(TokenCacheFileMeta {
        path: path.to_path_buf(),
        items,
        max_peaks,
        token_feature_width,
        target_width,
        condition_width,
        token_features_offset,
        target_pairs_offset,
        peak_mask_offset,
        padding_mask_offset,
        conditions_offset,
    })
}

fn read_token_cpu_cache_window_file(
    metadata: &TokenCacheFileMeta,
    start_item: usize,
    target_items: usize,
    _prefix: String,
) -> io::Result<TokenCpuCache> {
    let items = target_items.min(metadata.items.saturating_sub(start_item));
    let mut reader = BufReader::with_capacity(8 * 1024 * 1024, fs::File::open(&metadata.path)?);
    let token_features_start = metadata.token_features_offset
        + bytes_for_floats(
            start_item
                .saturating_mul(metadata.max_peaks)
                .saturating_mul(metadata.token_feature_width),
        )?;
    let target_pairs_start =
        metadata.target_pairs_offset + bytes_for_floats(start_item * metadata.target_width)?;
    let peak_mask_start =
        metadata.peak_mask_offset + bytes_for_floats(start_item * metadata.max_peaks)?;
    let padding_mask_start =
        metadata.padding_mask_offset + bytes_for_bools(start_item * metadata.max_peaks)?;
    let conditions_start =
        metadata.conditions_offset + bytes_for_floats(start_item * metadata.condition_width)?;

    let token_features = read_f32_vec_at(
        &mut reader,
        token_features_start,
        items * metadata.max_peaks * metadata.token_feature_width,
    )?;
    let target_pairs = read_f32_vec_at(
        &mut reader,
        target_pairs_start,
        items * metadata.target_width,
    )?;
    let peak_mask = read_f32_vec_at(&mut reader, peak_mask_start, items * metadata.max_peaks)?;
    let padding_mask =
        read_bool_vec_at(&mut reader, padding_mask_start, items * metadata.max_peaks)?;
    let conditions = read_f32_vec_at(
        &mut reader,
        conditions_start,
        items * metadata.condition_width,
    )?;

    Ok(TokenCpuCache {
        token_features,
        target_pairs,
        peak_mask,
        padding_mask,
        conditions,
        items,
        max_peaks: metadata.max_peaks,
        token_feature_width: metadata.token_feature_width,
        target_width: metadata.target_width,
        condition_width: metadata.condition_width,
    })
}

fn append_file(path: &Path, writer: &mut impl Write) -> io::Result<()> {
    let mut reader = BufReader::with_capacity(8 * 1024 * 1024, fs::File::open(path)?);
    io::copy(&mut reader, writer)?;
    Ok(())
}

fn read_usize(reader: &mut impl Read, name: &'static str) -> io::Result<usize> {
    usize::try_from(read_u64(reader)?).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("token-cache {name} does not fit usize"),
        )
    })
}

fn read_u32(reader: &mut impl Read) -> io::Result<u32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(reader: &mut impl Read) -> io::Result<u64> {
    let mut bytes = [0_u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn write_u32(writer: &mut impl Write, value: u32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn write_u64(writer: &mut impl Write, value: u64) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn bytes_for_floats(floats: usize) -> io::Result<u64> {
    let bytes = floats
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "token-cache size overflow"))?;
    u64::try_from(bytes).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "token-cache byte count does not fit u64",
        )
    })
}

fn bytes_for_bools(values: usize) -> io::Result<u64> {
    u64::try_from(values).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "token-cache byte count does not fit u64",
        )
    })
}

fn read_f32_vec_at<R: Read + Seek>(
    reader: &mut R,
    offset: u64,
    len: usize,
) -> io::Result<Vec<f32>> {
    const CHUNK_FLOATS: usize = 1 << 20;
    let mut values = Vec::with_capacity(len);
    let mut remaining = len;
    let mut bytes = vec![0_u8; CHUNK_FLOATS * std::mem::size_of::<f32>()];

    reader.seek(SeekFrom::Start(offset))?;
    while remaining > 0 {
        let chunk_len = remaining.min(CHUNK_FLOATS);
        let byte_len = chunk_len * std::mem::size_of::<f32>();
        reader.read_exact(&mut bytes[..byte_len])?;
        values.extend(
            bytes[..byte_len]
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])),
        );
        remaining -= chunk_len;
    }

    Ok(values)
}

fn read_bool_vec_at<R: Read + Seek>(
    reader: &mut R,
    offset: u64,
    len: usize,
) -> io::Result<Vec<bool>> {
    const CHUNK_VALUES: usize = 1 << 20;
    let mut values = Vec::with_capacity(len);
    let mut remaining = len;
    let mut bytes = vec![0_u8; CHUNK_VALUES];

    reader.seek(SeekFrom::Start(offset))?;
    while remaining > 0 {
        let chunk_len = remaining.min(CHUNK_VALUES);
        reader.read_exact(&mut bytes[..chunk_len])?;
        values.extend(bytes[..chunk_len].iter().map(|value| *value != 0));
        remaining -= chunk_len;
    }

    Ok(values)
}

fn write_f32_slice_untracked(writer: &mut impl Write, values: &[f32]) -> io::Result<()> {
    const CHUNK_FLOATS: usize = 1 << 20;
    let mut bytes = Vec::with_capacity(CHUNK_FLOATS * std::mem::size_of::<f32>());
    for chunk in values.chunks(CHUNK_FLOATS) {
        bytes.clear();
        bytes.extend(chunk.iter().flat_map(|value| value.to_le_bytes()));
        writer.write_all(&bytes)?;
    }
    Ok(())
}

fn write_bool_slice_untracked(writer: &mut impl Write, values: &[bool]) -> io::Result<()> {
    const CHUNK_VALUES: usize = 1 << 20;
    let mut bytes = Vec::with_capacity(CHUNK_VALUES);
    for chunk in values.chunks(CHUNK_VALUES) {
        bytes.clear();
        bytes.extend(chunk.iter().map(|value| u8::from(*value)));
        writer.write_all(&bytes)?;
    }
    Ok(())
}

#[derive(Clone)]
struct TokenGpuCache<B: Backend> {
    token_features: Tensor<B, 3>,
    target_pairs: Tensor<B, 2>,
    peak_mask: Tensor<B, 2>,
    target_peak_mask: Tensor<B, 2>,
    padding_mask: Tensor<B, 2, Bool>,
    conditions: Tensor<B, 2>,
    teacher_gpu: Option<TeacherGpuCache<B>>,
    items: usize,
}

struct StreamingTokenizedMgfIter<'a, B, Open>
where
    B: Backend,
{
    loader: &'a StreamingTokenizedMgfLoader<B, Open>,
    host_windows: OrderedHostWindowStream<TokenHostWindow>,
    cache: Option<TokenGpuCache<B>>,
    cache_offset: usize,
    batches_processed: usize,
    items_processed: usize,
    epoch_index: u64,
    profile: LoaderProfileAccumulator,
    finished: bool,
}

impl<B, Open> Iterator for StreamingTokenizedMgfIter<'_, B, Open>
where
    B: AllMetricsBackend,
    Open: Fn() -> spectral_autoencoder::Result<TokenizedMgfIter> + Send + Sync + Clone + 'static,
{
    type Item = TokenizedAutoencoderBatch<B>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.batches_processed >= self.loader.max_batches {
            self.finish();
            return None;
        }

        if self
            .cache
            .as_ref()
            .is_none_or(|cache| self.cache_offset >= cache.items)
        {
            self.cache = self.fill_cache();
            self.cache_offset = 0;
        }

        let cache = match &self.cache {
            Some(cache) => cache,
            None => {
                self.finish();
                return None;
            }
        };
        let batch_items = self
            .loader
            .batch_size
            .min(cache.items.saturating_sub(self.cache_offset));
        if batch_items == 0 {
            self.finish();
            return None;
        }

        let token_features = cache
            .token_features
            .clone()
            .narrow(0, self.cache_offset, batch_items);
        let target_pairs = cache
            .target_pairs
            .clone()
            .narrow(0, self.cache_offset, batch_items);
        let peak_mask = cache
            .peak_mask
            .clone()
            .narrow(0, self.cache_offset, batch_items);
        let target_peak_mask =
            cache
                .target_peak_mask
                .clone()
                .narrow(0, self.cache_offset, batch_items);
        let padding_mask = cache
            .padding_mask
            .clone()
            .narrow(0, self.cache_offset, batch_items);
        let conditions = cache
            .conditions
            .clone()
            .narrow(0, self.cache_offset, batch_items);
        let target_conditions = conditions.clone();

        let (
            token_features,
            peak_mask,
            padding_mask,
            conditions,
            masked_precursor_mask,
            masked_peak_mask,
            intruder_peak_mask,
        ) = match self.loader.augment {
            Some(config) => {
                let clean_token_features = token_features.clone();
                let clean_peak_mask = peak_mask.clone();
                let clean_padding_mask = padding_mask.clone();
                let (
                    token_features,
                    peak_mask,
                    padding_mask,
                    conditions,
                    masked_precursor_mask,
                    masked_peak_mask,
                    intruder_peak_mask,
                ) = augment_token_batch(
                    clean_token_features.clone(),
                    clean_peak_mask.clone(),
                    clean_padding_mask.clone(),
                    conditions.clone(),
                    config,
                );
                (
                    token_features,
                    peak_mask,
                    padding_mask,
                    conditions,
                    masked_precursor_mask,
                    masked_peak_mask,
                    intruder_peak_mask,
                )
            }
            None => {
                let device = token_features.device();
                let [batch_size, max_peaks, _feature_width] = token_features.dims();
                (
                    token_features.clone(),
                    peak_mask.clone(),
                    padding_mask.clone(),
                    conditions.clone(),
                    Tensor::<B, 2>::zeros([batch_size, 1], &device),
                    Tensor::<B, 2>::zeros([batch_size, max_peaks], &device),
                    Tensor::<B, 2>::zeros([batch_size, max_peaks], &device),
                )
            }
        };
        let similarity_ranking = teacher_similarity_ranking_batch(
            cache.teacher_gpu.as_ref(),
            self.cache_offset,
            batch_items,
            similarity_pair_seed(
                self.epoch_index,
                self.batches_processed,
                self.items_processed,
            ),
            self.loader.similarity_teacher,
            &self.loader.device,
        );

        self.cache_offset += batch_items;
        self.batches_processed += 1;
        self.items_processed += batch_items;
        self.loader.progress.batch_loaded(
            batch_items,
            self.batches_processed,
            self.loader.max_batches,
        );

        Some(TokenizedAutoencoderBatch {
            token_features,
            target_pairs,
            peak_mask,
            target_peak_mask,
            padding_mask,
            conditions,
            target_conditions,
            masked_precursor_mask,
            masked_peak_mask,
            intruder_peak_mask,
            similarity_ranking,
        })
    }
}

impl<B, Open> StreamingTokenizedMgfIter<'_, B, Open>
where
    B: AllMetricsBackend,
    Open: Fn() -> spectral_autoencoder::Result<TokenizedMgfIter> + Send + Sync + Clone + 'static,
{
    fn fill_cache(&mut self) -> Option<TokenGpuCache<B>> {
        let remaining_items = self.loader.epoch_items.saturating_sub(self.items_processed);
        let target_items = self.loader.window_items.min(remaining_items);
        if target_items == 0 {
            return None;
        }

        let wait_start = Instant::now();
        let host_window = self.host_windows.next_window("tokenized");
        let wait = wait_start.elapsed();
        let host_window = host_window?;
        if host_window.items > target_items {
            panic!(
                "token host worker produced {} items for a {}-item target window",
                host_window.items, target_items
            );
        }

        let mut profile = host_window.profile;
        profile.wait = wait;
        let upload_start = Instant::now();
        let gpu_cache = self.loader.move_token_host_window_to_gpu(&host_window);
        profile.tensor_upload = upload_start.elapsed();
        self.profile.record(profile);
        Some(gpu_cache)
    }

    fn finish(&mut self) {
        finish_loader_once(
            &self.loader.progress,
            self.items_processed,
            self.batches_processed,
            &mut self.finished,
        );
    }
}

impl<B, Open> DataLoaderIterator<TokenizedAutoencoderBatch<B>>
    for StreamingTokenizedMgfIter<'_, B, Open>
where
    B: AllMetricsBackend,
    Open: Fn() -> spectral_autoencoder::Result<TokenizedMgfIter> + Send + Sync + Clone + 'static,
{
    fn progress(&self) -> Progress {
        Progress {
            items_processed: self.items_processed,
            items_total: self.loader.epoch_items,
        }
    }
}

/// Trains the peak-set autoencoder end-to-end. Called from the clap-driven
/// `train peak-set` subcommand entry point.
pub fn run(cli: &crate::cli::PeakSetArgs) -> Result<(), Box<dyn std::error::Error>> {
    use std::sync::Arc;

    use burn::{
        module::Module,
        optim::{AdamConfig, decay::WeightDecayConfig, lr_scheduler::constant::ConstantLr},
        record::CompactRecorder,
        train::{Learner, SupervisedTraining},
    };
    use spectral_autoencoder::{
        AutoencoderTrainingMetricsExt, ConditioningEncoder, PeakSetAutoencoderConfig,
        SpectrumAugmentationConfig, SpectrumTokenizerConfig,
    };

    use crate::cli::{
        augmentation_config, auxiliary_loss_config, run_args_from_shared,
        similarity_teacher_config, streaming_loader_config,
    };
    use crate::common::{
        GeMSProgress, InnerBackend, PreprocessedCacheOptions, TrainingBackend,
        print_streaming_run_header, save_model_record, warm_start_model,
    };
    use crate::peak_set::{TokenCacheShape, TokenizedLoaderOptions, streaming_tokenized_loader};

    let shared = &cli.shared;
    let args = run_args_from_shared(
        shared,
        cli.batch_size,
        cli.train_batches,
        cli.valid_batches,
        cli.epochs,
    )?;
    std::fs::create_dir_all(&args.output_dir)?;

    let progress = Arc::new(GeMSProgress::new(
        args.train_batches,
        args.valid_batches,
        args.batch_size,
        "tokenizing",
        shared.progress,
    ));
    let device = burn::backend::cuda::CudaDevice::new(args.device);
    let augmentation = augmentation_config(shared, SpectrumAugmentationConfig::masked_mz_pretraining());
    let tokenizer_config = SpectrumTokenizerConfig {
        max_peaks: args.max_peaks,
        ..SpectrumTokenizerConfig::default()
    };
    let token_cache_shape = TokenCacheShape {
        max_peaks: tokenizer_config.max_peaks,
        token_feature_width: tokenizer_config.feature_width(),
        target_width: tokenizer_config.target_width(),
        condition_width: ConditioningEncoder::default().vector_width(),
    };
    let config = PeakSetAutoencoderConfig::twenty_million_run_with_peaks(args.max_peaks);
    let auxiliary = auxiliary_loss_config(shared, config.auxiliary);
    let similarity_teacher = similarity_teacher_config(shared, auxiliary)?;
    let loader_config = streaming_loader_config(shared, similarity_teacher)?;
    let cache = PreprocessedCacheOptions {
        enabled: cli.preprocessed_cache,
        dir: cli.preprocessed_cache_dir.clone(),
        refresh: cli.preprocessed_cache_refresh,
    };
    let train_builder = args.gems_builder.clone();
    let train_tokenizer_config = tokenizer_config.clone();
    let train_loader = streaming_tokenized_loader::<TrainingBackend, _>(
        &args,
        TokenizedLoaderOptions {
            device: device.clone(),
            progress: progress.train.clone(),
            start_item: args.train_start_item(),
            augment: Some(augmentation),
            loader_config: loader_config.clone(),
            token_cache_shape,
            cache: cache.clone(),
        },
        move || open_records(train_builder.clone(), train_tokenizer_config.clone()),
    );
    let valid_builder = args.gems_builder.clone();
    let valid_tokenizer_config = tokenizer_config.clone();
    let valid_loader = streaming_tokenized_loader::<InnerBackend, _>(
        &args,
        TokenizedLoaderOptions {
            device: device.clone(),
            progress: progress.valid.clone(),
            start_item: args.valid_start_item(),
            augment: None,
            loader_config: loader_config.clone(),
            token_cache_shape,
            cache: cache.clone(),
        },
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

    print_streaming_run_header(
        "peak-set",
        &args,
        parameter_count,
        &loader_config,
        auxiliary,
        similarity_teacher,
        None,
    );
    progress.start_training("starting GeMS peak-set streaming training");
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
    progress.finish_training("finished GeMS peak-set streaming training");

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
    tokenizer_config: spectral_autoencoder::SpectrumTokenizerConfig,
) -> spectral_autoencoder::Result<spectral_autoencoder::TokenizedMgfIter> {
    use spectral_autoencoder::{ConditioningEncoder, SpectrumTokenizer, TokenizedMgfIter};
    let records = crate::common::open_gems_a10_iter(builder)?;
    Ok(TokenizedMgfIter::from_records(
        records,
        SpectrumTokenizer::new(tokenizer_config),
        ConditioningEncoder::default(),
    ))
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    #[test]
    fn token_cache_roundtrip_preserves_host_arrays() {
        let path = temp_cache_path("token-roundtrip", "saetc");
        let shape = TokenCacheShape {
            max_peaks: 2,
            token_feature_width: 3,
            target_width: 4,
            condition_width: 2,
        };
        let token_features = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let target_pairs = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let peak_mask = vec![1.0, 0.0, 1.0, 1.0];
        let padding_mask = vec![false, true, false, false];
        let conditions = vec![0.25, 1.0, 0.5, 1.0];
        {
            let mut writer = BufWriter::new(fs::File::create(&path).expect("create token cache"));
            writer.write_all(TOKEN_CACHE_MAGIC).expect("write magic");
            write_u32(&mut writer, TOKEN_CACHE_VERSION).expect("write version");
            write_u64(&mut writer, 2).expect("write items");
            write_u64(&mut writer, shape.max_peaks as u64).expect("write max peaks");
            write_u64(&mut writer, shape.token_feature_width as u64).expect("write feature width");
            write_u64(&mut writer, shape.target_width as u64).expect("write target width");
            write_u64(&mut writer, shape.condition_width as u64).expect("write condition width");
            write_f32_slice_untracked(&mut writer, &token_features).expect("write features");
            write_f32_slice_untracked(&mut writer, &target_pairs).expect("write targets");
            write_f32_slice_untracked(&mut writer, &peak_mask).expect("write peak mask");
            write_bool_slice_untracked(&mut writer, &padding_mask).expect("write padding mask");
            write_f32_slice_untracked(&mut writer, &conditions).expect("write conditions");
            writer.flush().expect("flush token cache");
        }

        let metadata = read_token_cache_file_meta(&path, 2, shape).expect("read metadata");
        let window = read_token_cpu_cache_window_file(&metadata, 0, 2, "test".to_string())
            .expect("read cache");
        assert_eq!(window.items, 2);
        assert_eq!(window.token_features, token_features);
        assert_eq!(window.target_pairs, target_pairs);
        assert_eq!(window.peak_mask, peak_mask);
        assert_eq!(window.padding_mask, padding_mask);
        assert_eq!(window.conditions, conditions);
        fs::remove_file(path).ok();
    }

    fn temp_cache_path(label: &str, extension: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time after epoch")
            .as_nanos();
        env::temp_dir().join(format!(
            "spectral-autoencoder-{label}-{}-{nanos}.{extension}",
            std::process::id()
        ))
    }
}
