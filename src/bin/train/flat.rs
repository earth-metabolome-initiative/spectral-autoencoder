use std::{
    collections::hash_map::DefaultHasher,
    error::Error as StdError,
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
    tensor::{Distribution, Tensor, TensorData, backend::Backend},
};
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use spectral_autoencoder::{
    AutoencoderBatch, FlatVectorReconstructionOrdering, SpectralAutoencoderConfig,
    SpectrumAugmentationConfig, VectorizedMgfIter,
};

use mass_spectrometry::burn::AllMetricsBackend;

use crate::common::{
    LoaderProgress, PairSamplingEpoch, PreprocessedCacheOptions, RunArgs, SimilarityTeacherConfig,
    StreamingLoaderOptions, StreamingTrainingLoaderConfig, TeacherGpuCache, TeacherSpectraBuilder,
    TeacherSpectraCache, finish_loader_once, loader_epoch_items, mask_precursor_conditions,
    open_records_or_panic, probability_mask, signed_random, similarity_pair_seed,
    skip_split_records, teacher_similarity_ranking_batch, tokenization_style,
};
use crate::streaming::{
    HostWindowPlan, LoaderProfileAccumulator, LoaderWindowProfile, LoaderWorkerError,
    OrderedHostWindowStream, host_window_plan, spawn_ordered_host_workers,
};

/// Builds the flat-vector model config, applying CLI overrides on top of the
/// 20M-spectrum baseline.
pub fn flat_vector_config(
    max_peaks: usize,
    latent_width: Option<usize>,
    hidden_widths: Option<Vec<usize>>,
    ordering: Option<FlatVectorReconstructionOrdering>,
) -> Result<SpectralAutoencoderConfig, Box<dyn StdError>> {
    let mut config = SpectralAutoencoderConfig::twenty_million_run_with_peaks(max_peaks);
    let latent_width = latent_width.unwrap_or(config.encoder.latent_width);
    let hidden_widths = hidden_widths.unwrap_or_else(|| config.encoder.hidden_widths.clone());
    if hidden_widths.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "--hidden-widths must not be empty",
        )
        .into());
    }

    config.encoder.latent_width = latent_width;
    config.decoder.latent_width = latent_width;
    config.encoder.hidden_widths = hidden_widths.clone();
    config.decoder.hidden_widths = hidden_widths.into_iter().rev().collect();
    if let Some(value) = ordering {
        config.reconstruction_ordering = value;
    }
    Ok(config)
}

pub fn streaming_vectorized_loader<B, Open>(
    args: &RunArgs,
    cache: &PreprocessedCacheOptions,
    device: B::Device,
    progress: LoaderProgress,
    start_item: usize,
    augment: Option<SpectrumAugmentationConfig>,
    loader_config: StreamingTrainingLoaderConfig,
    open_records: Open,
) -> Arc<dyn DataLoader<B, AutoencoderBatch<B>>>
where
    B: AllMetricsBackend + 'static,
    Open: Fn() -> spectral_autoencoder::Result<VectorizedMgfIter> + Send + Sync + Clone + 'static,
{
    let preprocessed_cache_path =
        flat_preprocessed_cache_path(args, cache, start_item, progress.max_batches);
    let loader = Arc::new(StreamingVectorizedMgfLoader::new(
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
        args.gpu_transfer_chunk_batches,
        loader_config,
        open_records,
    ));
    loader.ensure_preprocessed_cache_file();
    loader
}

const FLAT_CACHE_MAGIC: &[u8; 8] = b"SAFLC01\0";
const FLAT_CACHE_VERSION: u32 = 5;

fn flat_preprocessed_cache_path(
    args: &RunArgs,
    cache: &PreprocessedCacheOptions,
    start_item: usize,
    max_batches: usize,
) -> Option<PathBuf> {
    assert!(
        cache.enabled,
        "--preprocessed-cache=false is not supported; the host-worker loader requires the flat cache"
    );

    let default_cache_dir = format!(
        "datasets/gems-a10-top-{}-peaks/preprocessed-flat",
        args.max_peaks
    );
    let cache_dir = cache
        .dir
        .clone()
        .unwrap_or_else(|| PathBuf::from(default_cache_dir));
    let total_items = args.batch_size.saturating_mul(max_batches);
    let fingerprint = flat_preprocessed_cache_fingerprint(args, start_item, total_items);
    Some(cache_dir.join(format!(
        "flat-v{FLAT_CACHE_VERSION}-{fingerprint:016x}.saefc"
    )))
}

fn flat_preprocessed_cache_fingerprint(
    args: &RunArgs,
    start_item: usize,
    total_items: usize,
) -> u64 {
    let mut hasher = DefaultHasher::new();
    FLAT_CACHE_MAGIC.hash(&mut hasher);
    FLAT_CACHE_VERSION.hash(&mut hasher);
    args.mgf_source.hash(&mut hasher);
    start_item.hash(&mut hasher);
    total_items.hash(&mut hasher);
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

fn gpu_transfer_chunk_items(batch_size: usize, chunk_batches: usize, total_items: usize) -> usize {
    if total_items == 0 {
        return 0;
    }

    let batch_size = batch_size.max(1);
    let chunk_batches = chunk_batches.max(1);
    batch_size
        .saturating_mul(chunk_batches)
        .max(1)
        .min(total_items)
}

type AugmentedFlatBatch<B> = (
    Tensor<B, 2>,
    Tensor<B, 2>,
    Tensor<B, 2>,
    Tensor<B, 2>,
    Tensor<B, 2>,
);

fn augment_flat_batch<B: Backend>(
    spectra: Tensor<B, 2>,
    conditions: Tensor<B, 2>,
    config: SpectrumAugmentationConfig,
) -> AugmentedFlatBatch<B> {
    let [batch_size, spectrum_width] = spectra.dims();
    if config.is_disabled() {
        let device = spectra.device();
        let peak_count = spectrum_width / 2;
        return (
            spectra,
            conditions,
            Tensor::<B, 2>::zeros([batch_size, 1], &device),
            Tensor::<B, 2>::zeros([batch_size, spectrum_width], &device),
            Tensor::<B, 2>::zeros([batch_size, peak_count], &device),
        );
    }

    let peak_count = spectrum_width / 2;
    let device = spectra.device();
    let pairs = spectra.reshape([batch_size, peak_count, 2]);
    let mut mz = pairs
        .clone()
        .narrow(2, 0, 1)
        .reshape([batch_size, peak_count]);
    let mut intensity = pairs
        .clone()
        .narrow(2, 1, 1)
        .reshape([batch_size, peak_count]);
    let original_mz = mz.clone();
    let original_intensity = intensity.clone();
    let original_active = original_mz
        .clone()
        .greater_elem(0.0)
        .bool_and(original_intensity.clone().greater_elem(0.0));
    let mut active = original_active.clone();
    let empty_original_slots = original_active.clone().bool_not();
    let real_peak_present = original_active
        .clone()
        .float()
        .sum_dim(1)
        .greater_elem(0.0)
        .expand([batch_size, peak_count]);
    let source_min_mz = original_mz
        .clone()
        .mask_fill(original_active.clone().bool_not(), 1.0)
        .min_dim(1)
        .expand([batch_size, peak_count]);
    let source_max_mz = original_mz
        .clone()
        .mask_fill(original_active.clone().bool_not(), 0.0)
        .max_dim(1)
        .expand([batch_size, peak_count]);
    let source_min_intensity = original_intensity
        .clone()
        .mask_fill(original_active.clone().bool_not(), 1.0)
        .min_dim(1)
        .expand([batch_size, peak_count]);
    let source_max_intensity = original_intensity
        .clone()
        .mask_fill(original_active.bool_not(), 0.0)
        .max_dim(1)
        .expand([batch_size, peak_count]);

    let mut masked_mz_mask =
        Tensor::<B, 2>::zeros([batch_size, peak_count], &device).greater_elem(0.0);
    let mut masked_intensity_mask =
        Tensor::<B, 2>::zeros([batch_size, peak_count], &device).greater_elem(0.0);

    if let Some(dropout_mask) = probability_mask(
        [batch_size, peak_count],
        &device,
        config.peak_dropout_probability,
    ) {
        let dropout_mask = dropout_mask.bool_and(active.clone());
        active = active.bool_and(dropout_mask.clone().bool_not());
        masked_mz_mask = masked_mz_mask.bool_or(dropout_mask.clone());
        masked_intensity_mask = masked_intensity_mask.bool_or(dropout_mask.clone());
        mz = mz.mask_fill(dropout_mask.clone(), 0.0);
        intensity = intensity.mask_fill(dropout_mask, 0.0);
    }

    let mut mz_mask = Tensor::<B, 2>::zeros([batch_size, peak_count], &device).greater_elem(0.0);
    if let Some(mask) = probability_mask(
        [batch_size, peak_count],
        &device,
        config.mz_mask_probability,
    ) {
        mz_mask = mask.bool_and(active.clone());
        masked_mz_mask = masked_mz_mask.bool_or(mz_mask.clone());
        mz = mz.mask_fill(mz_mask.clone(), 0.0);
    }

    let mz_editable = active.clone().bool_and(mz_mask.bool_not());
    let mz_shift_range = config.mz_shift_range.max(0.0);
    let mz_jitter_range = config.mz_jitter_range.max(0.0);
    if mz_shift_range > 0.0 || mz_jitter_range > 0.0 {
        let shift = signed_random([batch_size, 1], &device, mz_shift_range)
            .expand([batch_size, peak_count]);
        let jitter = signed_random([batch_size, peak_count], &device, mz_jitter_range);
        let shifted = (mz.clone() + shift + jitter).clamp_min(0.0).clamp_max(1.0);
        mz = mz.mask_where(mz_editable, shifted);
    }

    let intensity_jitter = config.intensity_jitter_fraction.max(0.0);
    if intensity_jitter > 0.0 {
        let scale = (signed_random([batch_size, peak_count], &device, intensity_jitter) + 1.0)
            .clamp_min(0.0);
        let jittered = (intensity.clone() * scale).clamp_max(1.0);
        intensity = intensity.mask_where(active, jittered);
    }

    let mut intruder_peak_mask =
        Tensor::<B, 2>::zeros([batch_size, peak_count], &device).greater_elem(0.0);
    if let Some(mask) = probability_mask(
        [batch_size, peak_count],
        &device,
        config.intruder_peak_probability,
    ) {
        intruder_peak_mask = mask
            .bool_and(empty_original_slots)
            .bool_and(real_peak_present);
        let random_mz = Tensor::<B, 2>::random(
            [batch_size, peak_count],
            Distribution::Uniform(0.0, 1.0),
            &device,
        );
        let random_intensity = Tensor::<B, 2>::random(
            [batch_size, peak_count],
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
    }

    let spectra = Tensor::cat(
        vec![mz.unsqueeze_dim::<3>(2), intensity.unsqueeze_dim::<3>(2)],
        2,
    )
    .reshape([batch_size, spectrum_width]);

    let masked_spectra_mask = Tensor::cat(
        vec![
            masked_mz_mask.float().unsqueeze_dim::<3>(2),
            masked_intensity_mask.float().unsqueeze_dim::<3>(2),
        ],
        2,
    )
    .reshape([batch_size, spectrum_width]);

    let (conditions, masked_precursor_mask) =
        mask_precursor_conditions(conditions, config.precursor_mask_probability);

    (
        spectra,
        conditions,
        masked_precursor_mask,
        masked_spectra_mask,
        intruder_peak_mask.float(),
    )
}

struct StreamingVectorizedMgfLoader<B, Open>
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
    gpu_transfer_chunk_batches: usize,
    device: B::Device,
    augment: Option<SpectrumAugmentationConfig>,
    similarity_teacher: SimilarityTeacherConfig,
    open_records: Open,
    disk_cache: Arc<Mutex<Option<FlatCacheFileMeta>>>,
    pair_sampling_epoch: PairSamplingEpoch,
}

impl<B, Open> Clone for StreamingVectorizedMgfLoader<B, Open>
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
            gpu_transfer_chunk_batches: self.gpu_transfer_chunk_batches,
            device: self.device.clone(),
            augment: self.augment,
            similarity_teacher: self.similarity_teacher,
            open_records: self.open_records.clone(),
            disk_cache: self.disk_cache.clone(),
            pair_sampling_epoch: self.pair_sampling_epoch.clone(),
        }
    }
}

impl<B, Open> StreamingVectorizedMgfLoader<B, Open>
where
    B: Backend,
{
    fn new(
        options: StreamingLoaderOptions<B>,
        preprocessed_cache_path: Option<PathBuf>,
        preprocessed_cache_refresh: bool,
        gpu_transfer_chunk_batches: usize,
        loader_config: StreamingTrainingLoaderConfig,
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
            gpu_transfer_chunk_batches,
            device: options.device,
            augment: options.augment,
            similarity_teacher: options.similarity_teacher,
            open_records,
            disk_cache: Arc::new(Mutex::new(None)),
            pair_sampling_epoch: PairSamplingEpoch::new(options.randomize_pair_sampling),
        }
    }
}

impl<B, Open> StreamingVectorizedMgfLoader<B, Open>
where
    B: Backend,
    Open: Fn() -> spectral_autoencoder::Result<VectorizedMgfIter> + Send + Sync + Clone + 'static,
{
    fn ensure_preprocessed_cache_file(&self) {
        let Some(path) = &self.preprocessed_cache_path else {
            return;
        };
        let total_items = self.cache_total_items;
        let refresh = self.preprocessed_cache_refresh;

        if !refresh && path.try_exists().unwrap_or(false) {
            if let Err(error) = self.flat_cache_file_meta(path, total_items) {
                panic!(
                    "invalid preprocessed flat cache {}: {error}; pass --preprocessed-cache-refresh to rebuild it",
                    path.display()
                );
            }
            return;
        }

        if let Err(error) = self.write_flat_cpu_cache_file_streaming(path, total_items) {
            panic!(
                "failed to prepare preprocessed flat cache {}: {error}",
                path.display()
            );
        }
        let metadata = self
            .flat_cache_file_meta(path, total_items)
            .unwrap_or_else(|error| {
                panic!(
                    "failed to reopen preprocessed flat cache {}: {error}",
                    path.display()
                )
            });
        *self
            .disk_cache
            .lock()
            .expect("flat disk cache metadata lock should not be poisoned") = Some(metadata);
    }

    fn flat_cache_file_meta(
        &self,
        path: &Path,
        expected_items: usize,
    ) -> io::Result<FlatCacheFileMeta> {
        if let Some(metadata) = self
            .disk_cache
            .lock()
            .expect("flat disk cache metadata lock should not be poisoned")
            .clone()
        {
            return Ok(metadata);
        }
        let metadata = read_flat_cache_file_meta(path, expected_items)?;
        *self
            .disk_cache
            .lock()
            .expect("flat disk cache metadata lock should not be poisoned") =
            Some(metadata.clone());
        Ok(metadata)
    }

    fn flat_host_window_stream(&self) -> OrderedHostWindowStream<FlatHostWindow> {
        let path = self.preprocessed_cache_path.as_ref().unwrap_or_else(|| {
            panic!(
                "flat host-worker loading requires the preprocessed cache for {}",
                self.mgf_source
            )
        });
        // Refresh is consumed during cache preparation; iteration only needs the rebuilt file.
        if !path.try_exists().unwrap_or(false) {
            panic!(
                "preprocessed flat cache {} is unavailable; it should have been prepared before iteration",
                path.display()
            );
        }
        let total_items = self.epoch_items;
        let metadata = self
            .flat_cache_file_meta(path, self.cache_total_items)
            .unwrap_or_else(|error| {
                panic!(
                    "invalid preprocessed flat cache {}: {error}",
                    path.display()
                )
            });
        let plans = host_window_plan(total_items, self.window_items);
        let build_context = FlatHostWindowBuildContext {
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
            move |_worker_id, plan| build_flat_host_window(&build_context, plan),
        )
    }

    fn write_flat_cpu_cache_file_streaming(
        &self,
        path: &Path,
        total_items: usize,
    ) -> io::Result<()> {
        let mut records = open_records_or_panic(&self.open_records, "vectorized", &self.mgf_source);
        self.skip_preload_offset(&mut records);

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("flat-cache.saefc");
        let tmp_path = path.with_file_name(format!("{file_name}.tmp-{}", std::process::id()));
        let tmp_conditions_path =
            path.with_file_name(format!("{file_name}.conditions.tmp-{}", std::process::id()));

        let bar = preload_bar(
            format!("{} cpu", self.progress.label),
            total_items,
            "decompressing MGF and vectorizing spectra to disk cache",
        );

        let result = (|| -> io::Result<()> {
            let mut writer =
                BufWriter::with_capacity(8 * 1024 * 1024, fs::File::create(&tmp_path)?);
            let mut conditions_writer =
                BufWriter::with_capacity(8 * 1024 * 1024, fs::File::create(&tmp_conditions_path)?);
            let mut written_items = 0usize;
            let mut spectrum_width = 0usize;
            let mut condition_width = 0usize;

            while written_items < total_items {
                let target_items = self.window_items.min(total_items - written_items);
                let cache = self
                    .read_flat_cpu_cache(
                        &mut records,
                        target_items,
                        written_items / self.batch_size,
                        Some(&bar),
                    )
                    .ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::UnexpectedEof,
                            format!(
                                "MGF ended after {written_items} spectra, expected {total_items}"
                            ),
                        )
                    })?;

                if written_items == 0 {
                    spectrum_width = cache.spectrum_width;
                    condition_width = cache.condition_width;
                    writer.write_all(FLAT_CACHE_MAGIC)?;
                    write_u32(&mut writer, FLAT_CACHE_VERSION)?;
                    write_u64(&mut writer, total_items as u64)?;
                    write_u64(&mut writer, spectrum_width as u64)?;
                    write_u64(&mut writer, condition_width as u64)?;
                } else if cache.spectrum_width != spectrum_width
                    || cache.condition_width != condition_width
                {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "vectorized flat cache width changed within the MGF stream",
                    ));
                }

                bar.set_message("writing spectra vector cache");
                write_f32_slice_untracked(&mut writer, &cache.spectra)?;
                bar.set_message("writing conditions vector cache");
                write_f32_slice_untracked(&mut conditions_writer, &cache.conditions)?;
                written_items += cache.items;
            }

            conditions_writer.flush()?;
            drop(conditions_writer);
            let mut conditions_reader =
                BufReader::with_capacity(8 * 1024 * 1024, fs::File::open(&tmp_conditions_path)?);
            io::copy(&mut conditions_reader, &mut writer)?;
            writer.flush()
        })();

        match result {
            Ok(()) => {
                fs::rename(&tmp_path, path)?;
                let _ = fs::remove_file(&tmp_conditions_path);
                bar.finish_with_message(format!(
                    "wrote preprocessed vectors to {}",
                    path.display()
                ));
                Ok(())
            }
            Err(error) => {
                bar.finish_with_message(format!("failed writing preprocessed vectors: {error}"));
                let _ = fs::remove_file(&tmp_path);
                let _ = fs::remove_file(&tmp_conditions_path);
                Err(error)
            }
        }
    }

    fn skip_preload_offset(&self, records: &mut VectorizedMgfIter) {
        if self.start_item == 0 {
            return;
        }

        let bar = preload_bar(
            format!("{} seek", self.progress.label),
            self.start_item,
            "seeking split offset",
        );
        skip_split_records(records, self.start_item, &self.progress, Some(&bar));
        bar.finish_with_message(format!("split offset reached at {}", self.start_item));
    }

    fn read_flat_cpu_cache(
        &self,
        records: &mut VectorizedMgfIter,
        target_items: usize,
        batches_processed: usize,
        preload_bar: Option<&ProgressBar>,
    ) -> Option<FlatCpuCache> {
        if target_items == 0 {
            return None;
        }

        let mut spectra = Vec::new();
        let mut conditions = Vec::new();
        let mut spectrum_width = 0usize;
        let mut condition_width = 0usize;
        let mut items = 0usize;
        let mut reported_items = 0usize;

        while items < target_items {
            let sample = match records.next() {
                Some(Ok(sample)) => sample,
                Some(Err(error)) => {
                    panic!("failed to vectorize MGF record: {error}");
                }
                None => break,
            };

            if items == 0 {
                spectrum_width = sample.spectrum.len();
                condition_width = sample.conditions.len();
                spectra.reserve(target_items * spectrum_width);
                conditions.reserve(target_items * condition_width);
            }
            spectra.extend(sample.spectrum);
            conditions.extend(sample.conditions);
            items += 1;
            if items.is_multiple_of(self.batch_size) || items == target_items {
                if let Some(bar) = preload_bar {
                    bar.inc((items - reported_items) as u64);
                    reported_items = items;
                    bar.set_message(format!(
                        "CPU vectorizing batches {}/{}",
                        items / self.batch_size,
                        self.max_batches,
                    ));
                }
                self.progress.cache_filling(
                    items,
                    target_items,
                    batches_processed,
                    self.max_batches,
                );
            }
        }

        if items == 0 {
            return None;
        }
        if let Some(bar) = preload_bar
            && items > reported_items
        {
            bar.inc((items - reported_items) as u64);
        }

        Some(FlatCpuCache {
            spectra,
            conditions,
            items,
            spectrum_width,
            condition_width,
        })
    }

    fn move_flat_host_window_to_gpu(
        &self,
        host_window: &FlatHostWindow,
        transfer_bar: Option<&ProgressBar>,
    ) -> FlatGpuCache<B> {
        move_flat_cache_window_to_gpu(
            host_window,
            self.batch_size,
            self.gpu_transfer_chunk_batches,
            &self.device,
            transfer_bar,
        )
    }
}

fn move_flat_cache_window_to_gpu<B: Backend>(
    host_window: &FlatHostWindow,
    batch_size: usize,
    chunk_batches: usize,
    device: &B::Device,
    transfer_bar: Option<&ProgressBar>,
) -> FlatGpuCache<B> {
    let items = host_window.items;
    if items == 0 {
        return FlatGpuCache {
            chunks: Vec::new(),
            items,
        };
    }
    let chunk_items = gpu_transfer_chunk_items(batch_size, chunk_batches, items);
    let chunk_count = items.div_ceil(chunk_items);
    let mut chunks = Vec::with_capacity(chunk_count);

    for (chunk_index, start) in (0..items).step_by(chunk_items).enumerate() {
        let end = (start + chunk_items).min(items);
        let chunk_len = end - start;
        let chunk_label = format!("{}/{}", chunk_index + 1, chunk_count);

        if let Some(bar) = transfer_bar {
            bar.set_message(format!("moving spectra chunk {chunk_label} to GPU"));
        }
        let spectra_start = start * host_window.spectrum_width;
        let spectra_end = end * host_window.spectrum_width;
        let spectra = Tensor::<B, 2>::from_data(
            TensorData::new(
                host_window.spectra[spectra_start..spectra_end].to_vec(),
                [chunk_len, host_window.spectrum_width],
            ),
            device,
        );
        if let Some(bar) = transfer_bar {
            bar.inc(chunk_len as u64);
            bar.set_message(format!("moving condition chunk {chunk_label} to GPU"));
        }
        let conditions_start = start * host_window.condition_width;
        let conditions_end = end * host_window.condition_width;
        let conditions = Tensor::<B, 2>::from_data(
            TensorData::new(
                host_window.conditions[conditions_start..conditions_end].to_vec(),
                [chunk_len, host_window.condition_width],
            ),
            device,
        );
        if let Some(bar) = transfer_bar {
            bar.inc(chunk_len as u64);
        }
        let teacher_gpu = host_window.teacher.as_ref().and_then(|teacher| {
            if let Some(bar) = transfer_bar {
                bar.set_message(format!("moving teacher chunk {chunk_label} to GPU"));
            }
            let cache = TeacherGpuCache::from_cpu_window(teacher, start, end, device);
            if cache.is_some()
                && let Some(bar) = transfer_bar
            {
                bar.inc(chunk_len as u64);
            }
            cache
        });

        chunks.push(FlatGpuCacheChunk {
            spectra,
            conditions,
            teacher_gpu,
            items: chunk_len,
        });
    }

    FlatGpuCache { chunks, items }
}

fn teacher_spectra_cache_from_target_pairs(
    config: SimilarityTeacherConfig,
    target_pairs: &[f32],
    conditions: &[f32],
    items: usize,
    target_width: usize,
    condition_width: usize,
    progress: Option<&ProgressBar>,
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
        if let Some(bar) = progress
            && ((item + 1).is_multiple_of(8192) || item + 1 == items)
        {
            bar.set_message("preprocessing teacher spectra");
            bar.set_position((item + 1) as u64);
        }
    }
    builder.finish()
}

impl<B, Open> DataLoader<B, AutoencoderBatch<B>> for StreamingVectorizedMgfLoader<B, Open>
where
    B: AllMetricsBackend + 'static,
    Open: Fn() -> spectral_autoencoder::Result<VectorizedMgfIter> + Send + Sync + Clone + 'static,
{
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<AutoencoderBatch<B>> + 'a> {
        self.progress.start_epoch(self.epoch_items);
        let epoch_index = self.pair_sampling_epoch.next();

        Box::new(StreamingVectorizedMgfIter {
            loader: self,
            host_windows: self.flat_host_window_stream(),
            cache: None,
            cache_offset: 0,
            batches_processed: 0,
            items_processed: 0,
            epoch_index,
            profile: LoaderProfileAccumulator::new(
                format!("{} flat", self.progress.label),
                self.loader_profile_every,
                self.loader_profile_sink.clone(),
            ),
            finished: false,
        })
    }

    fn num_items(&self) -> usize {
        self.epoch_items
    }

    fn to_device(&self, device: &B::Device) -> Arc<dyn DataLoader<B, AutoencoderBatch<B>>> {
        Arc::new(Self {
            device: device.clone(),
            ..self.clone()
        })
    }

    fn slice(&self, start: usize, end: usize) -> Arc<dyn DataLoader<B, AutoencoderBatch<B>>> {
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

struct FlatCpuCache {
    spectra: Vec<f32>,
    conditions: Vec<f32>,
    items: usize,
    spectrum_width: usize,
    condition_width: usize,
}

struct FlatHostWindow {
    spectra: Vec<f32>,
    conditions: Vec<f32>,
    teacher: Option<Arc<TeacherSpectraCache>>,
    profile: LoaderWindowProfile,
    items: usize,
    spectrum_width: usize,
    condition_width: usize,
}

struct FlatHostWindowBuildContext {
    metadata: FlatCacheFileMeta,
    cache_item_offset: usize,
    similarity_teacher: SimilarityTeacherConfig,
    batch_size: usize,
    max_batches: usize,
    epoch_items: usize,
    progress: LoaderProgress,
    prefix: String,
}

fn build_flat_host_window(
    context: &FlatHostWindowBuildContext,
    plan: HostWindowPlan,
) -> Result<FlatHostWindow, LoaderWorkerError> {
    let producer_start = Instant::now();
    let disk_start = Instant::now();
    let cache = read_flat_cpu_cache_window_file(
        &context.metadata,
        context.cache_item_offset + plan.start_item,
        plan.items,
        context.prefix.clone(),
        None,
        false,
    )?;
    let disk_read = disk_start.elapsed();
    if cache.items != plan.items {
        return Err(LoaderWorkerError::new(format!(
            "flat cache ended after {} items for planned {}-item window at {}",
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
            &cache.spectra,
            &cache.conditions,
            cache.items,
            cache.spectrum_width,
            cache.condition_width,
            None,
        ))
    });
    let teacher_build = teacher_start.elapsed();

    Ok(FlatHostWindow {
        spectra: cache.spectra,
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
        spectrum_width: cache.spectrum_width,
        condition_width: cache.condition_width,
    })
}

#[derive(Clone)]
struct FlatCacheFileMeta {
    path: PathBuf,
    items: usize,
    spectrum_width: usize,
    condition_width: usize,
    spectra_offset: u64,
    conditions_offset: u64,
}

fn read_flat_cache_file_meta(path: &Path, expected_items: usize) -> io::Result<FlatCacheFileMeta> {
    let mut reader = BufReader::with_capacity(8 * 1024 * 1024, fs::File::open(path)?);
    let mut magic = [0_u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != FLAT_CACHE_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "unexpected flat-cache magic",
        ));
    }

    let version = read_u32(&mut reader)?;
    if version != FLAT_CACHE_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported flat-cache version {version}"),
        ));
    }

    let items = read_usize(&mut reader, "items")?;
    let spectrum_width = read_usize(&mut reader, "spectrum width")?;
    let condition_width = read_usize(&mut reader, "condition width")?;
    if items != expected_items {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("cache has {items} items, expected {expected_items}"),
        ));
    }

    let spectra_offset = reader.stream_position()?;
    let spectra_bytes = bytes_for_floats(items.saturating_mul(spectrum_width))?;
    let conditions_offset = spectra_offset.checked_add(spectra_bytes).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "flat-cache condition offset overflow",
        )
    })?;
    Ok(FlatCacheFileMeta {
        path: path.to_path_buf(),
        items,
        spectrum_width,
        condition_width,
        spectra_offset,
        conditions_offset,
    })
}

fn read_flat_cpu_cache_window_file(
    metadata: &FlatCacheFileMeta,
    start_item: usize,
    target_items: usize,
    prefix: String,
    transfer_bar: Option<&ProgressBar>,
    show_progress: bool,
) -> io::Result<FlatCpuCache> {
    let items = target_items.min(metadata.items.saturating_sub(start_item));
    let mut reader = BufReader::with_capacity(8 * 1024 * 1024, fs::File::open(&metadata.path)?);
    let payload_bytes = bytes_for_floats(
        items
            .saturating_mul(metadata.spectrum_width)
            .saturating_add(items.saturating_mul(metadata.condition_width)),
    )?;
    let bytes_bar = (show_progress && transfer_bar.is_none())
        .then(|| preload_bytes_bar(prefix, payload_bytes, "loading vector cache window"));

    if let Some(bar) = transfer_bar {
        bar.set_message("reading spectra window from vector cache");
    }
    let spectra_start = metadata.spectra_offset
        + bytes_for_floats(start_item.saturating_mul(metadata.spectrum_width))?;
    let spectra = read_f32_vec_at(
        &mut reader,
        spectra_start,
        items * metadata.spectrum_width,
        bytes_bar.as_ref(),
        "reading spectra window from vector cache",
    )?;
    if let Some(bar) = transfer_bar {
        bar.set_message("reading condition window from vector cache");
    }
    let conditions_start = metadata.conditions_offset
        + bytes_for_floats(start_item.saturating_mul(metadata.condition_width))?;
    let conditions = read_f32_vec_at(
        &mut reader,
        conditions_start,
        items * metadata.condition_width,
        bytes_bar.as_ref(),
        "reading condition window from vector cache",
    )?;

    if let Some(bar) = bytes_bar {
        bar.finish_with_message(format!(
            "loaded {} vector-cache spectra from {}",
            items,
            metadata.path.display()
        ));
    }
    Ok(FlatCpuCache {
        spectra,
        conditions,
        items,
        spectrum_width: metadata.spectrum_width,
        condition_width: metadata.condition_width,
    })
}

fn read_usize(reader: &mut impl Read, name: &'static str) -> io::Result<usize> {
    usize::try_from(read_u64(reader)?).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("flat-cache {name} does not fit usize"),
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
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "flat-cache size overflow"))?;
    u64::try_from(bytes).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "flat-cache byte count does not fit u64",
        )
    })
}

fn read_f32_vec_at<R: Read + Seek>(
    reader: &mut R,
    offset: u64,
    len: usize,
    bar: Option<&ProgressBar>,
    message: &'static str,
) -> io::Result<Vec<f32>> {
    const CHUNK_FLOATS: usize = 1 << 20;
    let mut values = Vec::with_capacity(len);
    let mut remaining = len;
    let mut bytes = vec![0_u8; CHUNK_FLOATS * std::mem::size_of::<f32>()];

    reader.seek(SeekFrom::Start(offset))?;
    while remaining > 0 {
        let chunk_len = remaining.min(CHUNK_FLOATS);
        let byte_len = chunk_len * std::mem::size_of::<f32>();
        if let Some(bar) = bar {
            bar.set_message(message);
        }
        reader.read_exact(&mut bytes[..byte_len])?;
        values.extend(
            bytes[..byte_len]
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])),
        );
        if let Some(bar) = bar {
            bar.inc(byte_len as u64);
        }
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

#[derive(Clone)]
struct FlatGpuCacheChunk<B: Backend> {
    spectra: Tensor<B, 2>,
    conditions: Tensor<B, 2>,
    teacher_gpu: Option<TeacherGpuCache<B>>,
    items: usize,
}

#[derive(Clone)]
struct FlatGpuCache<B: Backend> {
    chunks: Vec<FlatGpuCacheChunk<B>>,
    items: usize,
}

impl<B: Backend> FlatGpuCache<B> {
    fn chunk_at(&self, mut offset: usize) -> Option<(&FlatGpuCacheChunk<B>, usize)> {
        for chunk in &self.chunks {
            if offset < chunk.items {
                return Some((chunk, offset));
            }
            offset -= chunk.items;
        }
        None
    }
}

struct StreamingVectorizedMgfIter<'a, B, Open>
where
    B: Backend,
{
    loader: &'a StreamingVectorizedMgfLoader<B, Open>,
    host_windows: OrderedHostWindowStream<FlatHostWindow>,
    cache: Option<FlatGpuCache<B>>,
    cache_offset: usize,
    batches_processed: usize,
    items_processed: usize,
    epoch_index: u64,
    profile: LoaderProfileAccumulator,
    finished: bool,
}

impl<B, Open> Iterator for StreamingVectorizedMgfIter<'_, B, Open>
where
    B: AllMetricsBackend,
    Open: Fn() -> spectral_autoencoder::Result<VectorizedMgfIter> + Send + Sync + Clone + 'static,
{
    type Item = AutoencoderBatch<B>;

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
            self.cache = None;
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
        let (chunk, chunk_offset) = match cache.chunk_at(self.cache_offset) {
            Some(chunk) => chunk,
            None => {
                self.finish();
                return None;
            }
        };
        let batch_items = self
            .loader
            .batch_size
            .min(chunk.items.saturating_sub(chunk_offset));
        if batch_items == 0 {
            self.finish();
            return None;
        }

        let clean_spectra = chunk.spectra.clone().narrow(0, chunk_offset, batch_items);
        let target_spectra = clean_spectra.clone();
        let clean_conditions = chunk
            .conditions
            .clone()
            .narrow(0, chunk_offset, batch_items);
        let target_conditions = clean_conditions.clone();
        let (spectra, conditions, masked_precursor_mask, masked_spectra_mask, intruder_peak_mask) =
            match self.loader.augment {
                Some(config) => {
                    let (
                        spectra,
                        conditions,
                        masked_precursor_mask,
                        masked_spectra_mask,
                        intruder_peak_mask,
                    ) = augment_flat_batch(clean_spectra.clone(), clean_conditions.clone(), config);
                    (
                        spectra,
                        conditions,
                        masked_precursor_mask,
                        masked_spectra_mask,
                        intruder_peak_mask,
                    )
                }
                None => {
                    let device = clean_spectra.device();
                    let [batch_size, spectrum_width] = clean_spectra.dims();
                    let peak_count = spectrum_width / 2;
                    (
                        clean_spectra.clone(),
                        clean_conditions.clone(),
                        Tensor::<B, 2>::zeros([batch_size, 1], &device),
                        Tensor::<B, 2>::zeros([batch_size, spectrum_width], &device),
                        Tensor::<B, 2>::zeros([batch_size, peak_count], &device),
                    )
                }
            };
        let similarity_ranking = teacher_similarity_ranking_batch(
            chunk.teacher_gpu.as_ref(),
            chunk_offset,
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

        Some(AutoencoderBatch {
            spectra,
            target_spectra,
            conditions,
            target_conditions,
            masked_precursor_mask,
            masked_spectra_mask,
            intruder_peak_mask,
            similarity_ranking,
        })
    }
}

impl<B, Open> StreamingVectorizedMgfIter<'_, B, Open>
where
    B: Backend,
    Open: Fn() -> spectral_autoencoder::Result<VectorizedMgfIter> + Send + Sync + Clone + 'static,
{
    fn fill_cache(&mut self) -> Option<FlatGpuCache<B>> {
        let remaining_items = self.loader.epoch_items.saturating_sub(self.items_processed);
        let target_items = self.loader.window_items.min(remaining_items);
        if target_items == 0 {
            return None;
        }

        let wait_start = Instant::now();
        let host_window = self.host_windows.next_window("flat vector");
        let wait = wait_start.elapsed();
        let host_window = host_window?;
        if host_window.items > target_items {
            panic!(
                "flat host worker produced {} items for a {}-item target window",
                host_window.items, target_items
            );
        }

        let mut profile = host_window.profile;
        profile.wait = wait;
        let upload_start = Instant::now();
        let gpu_cache = self.loader.move_flat_host_window_to_gpu(&host_window, None);
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

impl<B, Open> DataLoaderIterator<AutoencoderBatch<B>> for StreamingVectorizedMgfIter<'_, B, Open>
where
    B: AllMetricsBackend,
    Open: Fn() -> spectral_autoencoder::Result<VectorizedMgfIter> + Send + Sync + Clone + 'static,
{
    fn progress(&self) -> Progress {
        Progress {
            items_processed: self.items_processed,
            items_total: self.loader.epoch_items,
        }
    }
}

fn preload_bar(prefix: String, total_items: usize, message: &'static str) -> ProgressBar {
    let bar = ProgressBar::with_draw_target(
        Some(total_items as u64),
        ProgressDrawTarget::stderr_with_hz(10),
    );
    bar.set_style(tokenization_style());
    bar.set_prefix(prefix);
    bar.set_message(message);
    bar
}

fn preload_bytes_bar(prefix: String, total_bytes: u64, message: &'static str) -> ProgressBar {
    let bar =
        ProgressBar::with_draw_target(Some(total_bytes), ProgressDrawTarget::stderr_with_hz(10));
    bar.set_style(download_style());
    bar.set_prefix(prefix);
    bar.set_message(message);
    bar
}

fn download_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{prefix:>17} [{elapsed_precise}] {wide_bar:.cyan/blue} {bytes}/{total_bytes} {bytes_per_sec} ETA {eta_precise} {msg}",
    )
    .expect("valid indicatif template")
    .progress_chars("=> ")
}

/// Trains the flat-vector autoencoder end-to-end. Called from the clap-driven
/// `train flat` subcommand entry point.
pub fn run(cli: &crate::cli::FlatArgs) -> Result<(), Box<dyn std::error::Error>> {
    use std::sync::Arc;

    use burn::{
        module::Module,
        optim::{AdamConfig, decay::WeightDecayConfig, lr_scheduler::constant::ConstantLr},
        record::CompactRecorder,
        train::{Learner, SupervisedTraining},
    };
    use spectral_autoencoder::{
        AutoencoderTrainingMetricsExt, SpectrumAugmentationConfig, SpectrumVectorizerConfig,
    };

    use crate::cli::{
        augmentation_config, auxiliary_loss_config, parse_usize_list, run_args_from_shared,
        similarity_teacher_config, streaming_loader_config,
    };
    use crate::common::{
        GeMSProgress, InnerBackend, PreprocessedCacheOptions, TrainingBackend,
        print_streaming_run_header, save_model_record, warm_start_model,
    };
    use crate::flat::{flat_vector_config, streaming_vectorized_loader};

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
        "vectorizing",
        shared.progress,
    ));
    let device = burn::backend::cuda::CudaDevice::new(args.device);
    let augmentation = augmentation_config(shared, SpectrumAugmentationConfig::masked_mz_pretraining());
    let vectorizer_config = SpectrumVectorizerConfig {
        max_peaks: args.max_peaks,
        ..SpectrumVectorizerConfig::default()
    };
    let hidden_widths = cli
        .hidden_widths
        .as_deref()
        .map(parse_usize_list)
        .transpose()?;
    let config = flat_vector_config(
        args.max_peaks,
        cli.latent_width,
        hidden_widths,
        cli.reconstruction_ordering.map(Into::into),
    )?;
    let auxiliary = auxiliary_loss_config(shared, config.auxiliary);
    let similarity_teacher = similarity_teacher_config(shared, auxiliary)?;
    let loader_config = streaming_loader_config(shared, similarity_teacher)?;
    let cache = PreprocessedCacheOptions {
        enabled: cli.preprocessed_cache,
        dir: cli.preprocessed_cache_dir.clone(),
        refresh: cli.preprocessed_cache_refresh,
    };
    let train_builder = args.gems_builder.clone();
    let train_vectorizer_config = vectorizer_config.clone();
    let train_cache = cache.clone();
    let train_loader = streaming_vectorized_loader::<TrainingBackend, _>(
        &args,
        &train_cache,
        device.clone(),
        progress.train.clone(),
        args.train_start_item(),
        Some(augmentation),
        loader_config.clone(),
        move || open_records(train_builder.clone(), train_vectorizer_config.clone()),
    );
    let valid_builder = args.gems_builder.clone();
    let valid_vectorizer_config = vectorizer_config.clone();
    let valid_cache = cache.clone();
    let valid_loader = streaming_vectorized_loader::<InnerBackend, _>(
        &args,
        &valid_cache,
        device.clone(),
        progress.valid.clone(),
        args.valid_start_item(),
        None,
        loader_config.clone(),
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
        &loader_config,
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
    vectorizer_config: spectral_autoencoder::SpectrumVectorizerConfig,
) -> spectral_autoencoder::Result<spectral_autoencoder::VectorizedMgfIter> {
    use spectral_autoencoder::{ConditioningEncoder, SpectrumVectorizer, VectorizedMgfIter};
    let records = crate::common::open_gems_a10_iter(builder)?;
    Ok(VectorizedMgfIter::from_records(
        records,
        SpectrumVectorizer::new(vectorizer_config),
        ConditioningEncoder::default(),
    ))
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    #[test]
    fn flat_cache_range_read_preserves_spectra_and_conditions() {
        let path = temp_cache_path("flat-range", "saefc");
        {
            let mut writer = BufWriter::new(fs::File::create(&path).expect("create flat cache"));
            writer.write_all(FLAT_CACHE_MAGIC).expect("write magic");
            write_u32(&mut writer, FLAT_CACHE_VERSION).expect("write version");
            write_u64(&mut writer, 3).expect("write items");
            write_u64(&mut writer, 2).expect("write spectrum width");
            write_u64(&mut writer, 2).expect("write condition width");
            write_f32_slice_untracked(&mut writer, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .expect("write spectra");
            write_f32_slice_untracked(&mut writer, &[10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
                .expect("write conditions");
            writer.flush().expect("flush flat cache");
        }

        let metadata = read_flat_cache_file_meta(&path, 3).expect("read metadata");
        let window =
            read_flat_cpu_cache_window_file(&metadata, 1, 2, "test".to_string(), None, false)
                .expect("read range");
        assert_eq!(window.items, 2);
        assert_eq!(window.spectra, vec![3.0, 4.0, 5.0, 6.0]);
        assert_eq!(window.conditions, vec![12.0, 13.0, 14.0, 15.0]);
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
