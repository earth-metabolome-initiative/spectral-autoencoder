use std::{
    collections::hash_map::DefaultHasher,
    env,
    error::Error as StdError,
    fs,
    hash::{Hash, Hasher},
    io,
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::UNIX_EPOCH,
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

use crate::gems_common::{
    CachedLoaderOptions, CachedTrainingLoaderConfig, LoaderProgress, PairSamplingEpoch, RunArgs,
    SimilarityTeacherBackend, SimilarityTeacherConfig, TeacherGpuCache, TeacherSpectraBuilder,
    TeacherSpectraCache, bool_var, cache_items, finish_loader_once, is_full_gpu_cache,
    loader_epoch_items, mask_precursor_conditions, open_records_or_panic, probability_mask,
    signed_random, similarity_pair_seed, skip_split_records, teacher_similarity_ranking_batch,
    tokenization_style, usize_var,
};

pub fn flat_vector_config_from_env(
    max_peaks: usize,
) -> Result<SpectralAutoencoderConfig, Box<dyn StdError>> {
    let mut config = SpectralAutoencoderConfig::twenty_million_run_with_peaks(max_peaks);
    let latent_width = usize_var("GEMS_FLAT_LATENT_WIDTH", config.encoder.latent_width);
    let hidden_widths = usize_list_var("GEMS_FLAT_HIDDEN_WIDTHS", &config.encoder.hidden_widths)?;

    config.encoder.latent_width = latent_width;
    config.decoder.latent_width = latent_width;
    config.encoder.hidden_widths = hidden_widths.clone();
    config.decoder.hidden_widths = hidden_widths.into_iter().rev().collect();
    config.reconstruction_ordering =
        flat_reconstruction_ordering_from_env(config.reconstruction_ordering)?;
    Ok(config)
}

fn flat_reconstruction_ordering_from_env(
    default: FlatVectorReconstructionOrdering,
) -> Result<FlatVectorReconstructionOrdering, Box<dyn StdError>> {
    let value = env::var("GEMS_FLAT_RECONSTRUCTION_ORDERING")
        .unwrap_or_else(|_| default.label().to_string())
        .to_ascii_lowercase();
    match value.as_str() {
        "slot" | "slots" | "strict" | "slot-wise" | "slot_wise" => {
            Ok(FlatVectorReconstructionOrdering::Slot)
        }
        "intensity"
        | "intensity-desc"
        | "intensity_desc"
        | "intensity-descending"
        | "intensity_descending" => Ok(FlatVectorReconstructionOrdering::IntensityDescending),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "GEMS_FLAT_RECONSTRUCTION_ORDERING must be slot or intensity-desc",
        )
        .into()),
    }
}

pub fn cached_vectorized_loader<B, Open>(
    args: &RunArgs,
    device: B::Device,
    progress: LoaderProgress,
    start_item: usize,
    augment: Option<SpectrumAugmentationConfig>,
    loader_config: CachedTrainingLoaderConfig,
    open_records: Open,
) -> Arc<dyn DataLoader<B, AutoencoderBatch<B>>>
where
    B: SimilarityTeacherBackend + 'static,
    Open: Fn() -> spectral_autoencoder::Result<VectorizedMgfIter> + Send + Sync + Clone + 'static,
{
    let preprocessed_cache_path =
        flat_preprocessed_cache_path(args, start_item, progress.max_batches);
    let loader = Arc::new(CachedVectorizedMgfLoader::new(
        CachedLoaderOptions {
            mgf_source: args.mgf_source.clone(),
            batch_size: args.batch_size,
            max_batches: progress.max_batches,
            start_item,
            cache_items: cache_items(
                args.batch_size,
                progress.max_batches,
                loader_config.cache_percent,
            ),
            device,
            progress,
            augment,
            similarity_teacher: loader_config.similarity_teacher,
            randomize_pair_sampling: augment.is_some(),
        },
        preprocessed_cache_path,
        open_records,
    ));
    loader.preload_cache();
    loader
}

const FLAT_CACHE_MAGIC: &[u8; 8] = b"SAFLC01\0";
const FLAT_CACHE_VERSION: u32 = 5;

fn flat_preprocessed_cache_path(
    args: &RunArgs,
    start_item: usize,
    max_batches: usize,
) -> Option<PathBuf> {
    if !bool_var("GEMS_FLAT_PREPROCESSED_CACHE", true) {
        return None;
    }

    let default_cache_dir = format!(
        "datasets/gems-a10-top-{}-peaks/preprocessed-flat",
        args.max_peaks
    );
    let cache_dir = env::var_os("GEMS_FLAT_PREPROCESSED_CACHE_DIR")
        .map(PathBuf::from)
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

fn gpu_transfer_chunk_items(batch_size: usize, total_items: usize) -> usize {
    if total_items == 0 {
        return 0;
    }

    let batch_size = batch_size.max(1);
    let chunk_batches = usize_var("GEMS_GPU_TRANSFER_CHUNK_BATCHES", 64).max(1);
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

struct CachedVectorizedMgfLoader<B, Open>
where
    B: Backend,
{
    mgf_source: String,
    progress: LoaderProgress,
    batch_size: usize,
    max_batches: usize,
    start_item: usize,
    cache_items: usize,
    preprocessed_cache_path: Option<PathBuf>,
    device: B::Device,
    augment: Option<SpectrumAugmentationConfig>,
    similarity_teacher: SimilarityTeacherConfig,
    open_records: Open,
    cpu_cache: Arc<Mutex<Option<Arc<FlatCpuCache>>>>,
    initial_cache: Arc<Mutex<Option<FlatGpuCache<B>>>>,
    full_cache: Arc<Mutex<Option<FlatGpuCache<B>>>>,
    pair_sampling_epoch: PairSamplingEpoch,
}

impl<B, Open> Clone for CachedVectorizedMgfLoader<B, Open>
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
            start_item: self.start_item,
            cache_items: self.cache_items,
            preprocessed_cache_path: self.preprocessed_cache_path.clone(),
            device: self.device.clone(),
            augment: self.augment,
            similarity_teacher: self.similarity_teacher,
            open_records: self.open_records.clone(),
            cpu_cache: self.cpu_cache.clone(),
            initial_cache: self.initial_cache.clone(),
            full_cache: self.full_cache.clone(),
            pair_sampling_epoch: self.pair_sampling_epoch.clone(),
        }
    }
}

impl<B, Open> CachedVectorizedMgfLoader<B, Open>
where
    B: Backend,
{
    fn new(
        options: CachedLoaderOptions<B>,
        preprocessed_cache_path: Option<PathBuf>,
        open_records: Open,
    ) -> Self {
        Self {
            mgf_source: options.mgf_source,
            progress: options.progress,
            batch_size: options.batch_size,
            max_batches: options.max_batches,
            start_item: options.start_item,
            cache_items: options.cache_items,
            preprocessed_cache_path,
            device: options.device,
            augment: options.augment,
            similarity_teacher: options.similarity_teacher,
            open_records,
            cpu_cache: Arc::new(Mutex::new(None)),
            initial_cache: Arc::new(Mutex::new(None)),
            full_cache: Arc::new(Mutex::new(None)),
            pair_sampling_epoch: PairSamplingEpoch::new(options.randomize_pair_sampling),
        }
    }

    fn is_full_cache(&self) -> bool {
        is_full_gpu_cache(self.cache_items, self.batch_size, self.max_batches)
    }

    fn gpu_transfer_stages(&self) -> usize {
        if self.similarity_teacher.enabled() {
            3
        } else {
            2
        }
    }
}

impl<B, Open> CachedVectorizedMgfLoader<B, Open>
where
    B: Backend,
    Open: Fn() -> spectral_autoencoder::Result<VectorizedMgfIter> + Send + Sync + Clone + 'static,
{
    fn preload_cache(&self) {
        let Some(cpu_cache) = self.preload_cpu_cache() else {
            return;
        };

        if self.is_full_cache() {
            if self
                .full_cache
                .lock()
                .expect("full vectorized cache lock should not be poisoned")
                .is_some()
            {
                return;
            }
            let transfer_bar = preload_transfer_bar(
                format!("{} gpu", self.progress.label),
                cpu_cache.items,
                "moving full CPU cache to GPU",
                self.gpu_transfer_stages(),
            );
            let cache = self.move_flat_cache_window_to_gpu(
                &cpu_cache,
                0,
                cpu_cache.items,
                Some(&transfer_bar),
            );
            transfer_bar.finish_with_message(format!("cached {} spectra on GPU", cache.items));
            *self
                .full_cache
                .lock()
                .expect("full vectorized cache lock should not be poisoned") = Some(cache);
        } else {
            if self
                .initial_cache
                .lock()
                .expect("initial vectorized cache lock should not be poisoned")
                .is_some()
            {
                return;
            }
            let target_items = self.cache_items.min(cpu_cache.items);
            let transfer_bar = preload_transfer_bar(
                format!("{} gpu", self.progress.label),
                target_items,
                "moving initial CPU cache window to GPU",
                self.gpu_transfer_stages(),
            );
            let cache = self.move_flat_cache_window_to_gpu(
                &cpu_cache,
                0,
                target_items,
                Some(&transfer_bar),
            );
            transfer_bar.finish_with_message(format!(
                "cached initial {} / {} spectra on GPU",
                cache.items, cpu_cache.items
            ));
            *self
                .initial_cache
                .lock()
                .expect("initial vectorized cache lock should not be poisoned") = Some(cache);
        }
    }

    fn preload_cpu_cache(&self) -> Option<Arc<FlatCpuCache>> {
        if let Some(cache) = self
            .cpu_cache
            .lock()
            .expect("CPU vectorized cache lock should not be poisoned")
            .clone()
        {
            return Some(cache);
        }

        let total_items = loader_epoch_items(self.batch_size, self.max_batches);
        if let Some(path) = &self.preprocessed_cache_path
            && !bool_var("GEMS_FLAT_PREPROCESSED_CACHE_REFRESH", false)
            && path.try_exists().unwrap_or(false)
        {
            match read_flat_cpu_cache_file(
                path,
                total_items,
                format!("{} disk", self.progress.label),
            ) {
                Ok(mut cpu_cache) => {
                    self.attach_flat_teacher_cache(&mut cpu_cache);
                    let cpu_cache = Arc::new(cpu_cache);
                    *self
                        .cpu_cache
                        .lock()
                        .expect("CPU vectorized cache lock should not be poisoned") =
                        Some(cpu_cache.clone());
                    return Some(cpu_cache);
                }
                Err(error) => {
                    eprintln!(
                        "ignoring preprocessed flat cache {}: {error}",
                        path.display()
                    );
                }
            }
        }

        let mut records = open_records_or_panic(&self.open_records, "vectorized", &self.mgf_source);
        self.skip_preload_offset(&mut records);

        let bar = preload_bar(
            format!("{} cpu", self.progress.label),
            total_items,
            "decompressing MGF and vectorizing spectra into CPU memory",
        );
        let cpu_cache = self.read_flat_cpu_cache(&mut records, total_items, 0, Some(&bar));
        match cpu_cache {
            Some(mut cpu_cache) => {
                bar.finish_with_message(format!(
                    "preprocessed {} spectra into CPU memory",
                    cpu_cache.items
                ));
                self.attach_flat_teacher_cache(&mut cpu_cache);
                if let Some(path) = &self.preprocessed_cache_path
                    && let Err(error) = write_flat_cpu_cache_file(
                        path,
                        &cpu_cache,
                        format!("{} disk", self.progress.label),
                    )
                {
                    eprintln!(
                        "could not write preprocessed flat cache {}: {error}",
                        path.display()
                    );
                }
                let cpu_cache = Arc::new(cpu_cache);
                *self
                    .cpu_cache
                    .lock()
                    .expect("CPU vectorized cache lock should not be poisoned") =
                    Some(cpu_cache.clone());
                Some(cpu_cache)
            }
            None => {
                bar.finish_with_message("no spectra cached");
                None
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

    fn attach_flat_teacher_cache(&self, cpu_cache: &mut FlatCpuCache) {
        if !self.similarity_teacher.enabled() || cpu_cache.teacher.is_some() {
            return;
        }
        let bar = preload_bar(
            format!("{} teacher", self.progress.label),
            cpu_cache.items,
            "preprocessing teacher spectra",
        );
        let teacher = teacher_spectra_cache_from_target_pairs(
            self.similarity_teacher,
            &cpu_cache.spectra,
            cpu_cache.items,
            cpu_cache.spectrum_width,
            Some(&bar),
        );
        bar.finish_with_message(format!("preprocessed {} teacher spectra", cpu_cache.items));
        cpu_cache.teacher = Some(Arc::new(teacher));
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
        let mut teacher_builder = self
            .similarity_teacher
            .enabled()
            .then(|| TeacherSpectraBuilder::new(self.similarity_teacher, target_items));

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
            if let Some(builder) = &mut teacher_builder {
                builder.push_pairs(&sample.spectrum);
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
            teacher: teacher_builder.map(|builder| Arc::new(builder.finish())),
        })
    }

    fn move_flat_cache_window_to_gpu(
        &self,
        cpu_cache: &FlatCpuCache,
        start_item: usize,
        target_items: usize,
        transfer_bar: Option<&ProgressBar>,
    ) -> FlatGpuCache<B> {
        let items = target_items.min(cpu_cache.items.saturating_sub(start_item));
        let chunk_items = gpu_transfer_chunk_items(self.batch_size, items);
        let chunk_count = items.div_ceil(chunk_items);
        let mut chunks = Vec::with_capacity(chunk_count);
        let end_item = start_item + items;

        for (chunk_index, start) in (start_item..end_item).step_by(chunk_items).enumerate() {
            let end = (start + chunk_items).min(end_item);
            let chunk_len = end - start;
            let chunk_label = format!("{}/{}", chunk_index + 1, chunk_count);

            if let Some(bar) = transfer_bar {
                bar.set_message(format!("moving spectra chunk {chunk_label} to GPU"));
            }
            let spectra_start = start * cpu_cache.spectrum_width;
            let spectra_end = end * cpu_cache.spectrum_width;
            let spectra = Tensor::<B, 2>::from_data(
                TensorData::new(
                    cpu_cache.spectra[spectra_start..spectra_end].to_vec(),
                    [chunk_len, cpu_cache.spectrum_width],
                ),
                &self.device,
            );
            if let Some(bar) = transfer_bar {
                bar.inc(chunk_len as u64);
                bar.set_message(format!("moving condition chunk {chunk_label} to GPU"));
            }
            let conditions_start = start * cpu_cache.condition_width;
            let conditions_end = end * cpu_cache.condition_width;
            let conditions = Tensor::<B, 2>::from_data(
                TensorData::new(
                    cpu_cache.conditions[conditions_start..conditions_end].to_vec(),
                    [chunk_len, cpu_cache.condition_width],
                ),
                &self.device,
            );
            if let Some(bar) = transfer_bar {
                bar.inc(chunk_len as u64);
            }
            let teacher_gpu = self
                .similarity_teacher
                .enabled()
                .then(|| {
                    cpu_cache.teacher.as_ref().and_then(|teacher| {
                        if let Some(bar) = transfer_bar {
                            bar.set_message(format!("moving teacher chunk {chunk_label} to GPU"));
                        }
                        let cache =
                            TeacherGpuCache::from_cpu_window(teacher, start, end, &self.device);
                        if cache.is_some()
                            && let Some(bar) = transfer_bar
                        {
                            bar.inc(chunk_len as u64);
                        }
                        cache
                    })
                })
                .flatten();

            chunks.push(FlatGpuCacheChunk {
                spectra,
                conditions,
                teacher_gpu,
                items: chunk_len,
            });
        }

        FlatGpuCache { chunks, items }
    }
}

fn teacher_spectra_cache_from_target_pairs(
    config: SimilarityTeacherConfig,
    target_pairs: &[f32],
    items: usize,
    target_width: usize,
    progress: Option<&ProgressBar>,
) -> TeacherSpectraCache {
    let mut builder = TeacherSpectraBuilder::new(config, items);
    for item in 0..items {
        let start = item * target_width;
        let end = start + target_width;
        builder.push_pairs(&target_pairs[start..end]);
        if let Some(bar) = progress
            && ((item + 1).is_multiple_of(8192) || item + 1 == items)
        {
            bar.set_message("preprocessing teacher spectra");
            bar.set_position((item + 1) as u64);
        }
    }
    builder.finish()
}

impl<B, Open> DataLoader<B, AutoencoderBatch<B>> for CachedVectorizedMgfLoader<B, Open>
where
    B: SimilarityTeacherBackend + 'static,
    Open: Fn() -> spectral_autoencoder::Result<VectorizedMgfIter> + Send + Sync + Clone + 'static,
{
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<AutoencoderBatch<B>> + 'a> {
        self.progress
            .start_epoch(loader_epoch_items(self.batch_size, self.max_batches));
        let epoch_index = self.pair_sampling_epoch.next();

        if self.is_full_cache()
            && let Some(cache) = self
                .full_cache
                .lock()
                .expect("full vectorized cache lock should not be poisoned")
                .clone()
        {
            return Box::new(CachedVectorizedMgfIter {
                loader: self,
                records: None,
                cache: Some(cache),
                cache_offset: 0,
                batches_processed: 0,
                items_processed: 0,
                epoch_index,
                finished: false,
            });
        }

        if self
            .cpu_cache
            .lock()
            .expect("CPU vectorized cache lock should not be poisoned")
            .is_some()
        {
            let initial_cache = self
                .initial_cache
                .lock()
                .expect("initial vectorized cache lock should not be poisoned")
                .take();
            return Box::new(CachedVectorizedMgfIter {
                loader: self,
                records: None,
                cache: initial_cache,
                cache_offset: 0,
                batches_processed: 0,
                items_processed: 0,
                epoch_index,
                finished: false,
            });
        }

        let mut records = open_records_or_panic(&self.open_records, "vectorized", &self.mgf_source);
        skip_split_records(&mut records, self.start_item, &self.progress, None);

        Box::new(CachedVectorizedMgfIter {
            loader: self,
            records: Some(records),
            cache: None,
            cache_offset: 0,
            batches_processed: 0,
            items_processed: 0,
            epoch_index,
            finished: false,
        })
    }

    fn num_items(&self) -> usize {
        loader_epoch_items(self.batch_size, self.max_batches)
    }

    fn to_device(&self, device: &B::Device) -> Arc<dyn DataLoader<B, AutoencoderBatch<B>>> {
        Arc::new(Self {
            device: device.clone(),
            initial_cache: Arc::new(Mutex::new(None)),
            full_cache: Arc::new(Mutex::new(None)),
            ..self.clone()
        })
    }

    fn slice(&self, start: usize, end: usize) -> Arc<dyn DataLoader<B, AutoencoderBatch<B>>> {
        let start_batch = start / self.batch_size;
        let end_batch = end.div_ceil(self.batch_size);
        Arc::new(Self {
            max_batches: end_batch.saturating_sub(start_batch),
            start_item: self.start_item + start_batch * self.batch_size,
            cpu_cache: Arc::new(Mutex::new(None)),
            initial_cache: Arc::new(Mutex::new(None)),
            full_cache: Arc::new(Mutex::new(None)),
            progress: self.progress.clone_for_slice(),
            ..self.clone()
        })
    }
}

struct FlatCpuCache {
    spectra: Vec<f32>,
    conditions: Vec<f32>,
    teacher: Option<Arc<TeacherSpectraCache>>,
    items: usize,
    spectrum_width: usize,
    condition_width: usize,
}

impl FlatCpuCache {
    fn payload_bytes(&self) -> io::Result<u64> {
        let floats = self.spectra.len().saturating_add(self.conditions.len());
        bytes_for_floats(floats)
    }
}

fn read_flat_cpu_cache_file(
    path: &Path,
    expected_items: usize,
    prefix: String,
) -> io::Result<FlatCpuCache> {
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

    let payload_bytes = bytes_for_floats(
        items
            .saturating_mul(spectrum_width)
            .saturating_add(items.saturating_mul(condition_width)),
    )?;
    let bar = preload_bytes_bar(prefix, payload_bytes, "loading preprocessed vector cache");
    let spectra = read_f32_vec(
        &mut reader,
        items * spectrum_width,
        &bar,
        "loading spectra from vector cache",
    )?;
    let conditions = read_f32_vec(
        &mut reader,
        items * condition_width,
        &bar,
        "loading conditions from vector cache",
    )?;
    bar.finish_with_message(format!(
        "loaded preprocessed vectors from {}",
        path.display()
    ));

    Ok(FlatCpuCache {
        spectra,
        conditions,
        teacher: None,
        items,
        spectrum_width,
        condition_width,
    })
}

fn write_flat_cpu_cache_file(path: &Path, cache: &FlatCpuCache, prefix: String) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("flat-cache.saefc");
    let tmp_path = path.with_file_name(format!("{file_name}.tmp-{}", std::process::id()));

    let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, fs::File::create(&tmp_path)?);
    writer.write_all(FLAT_CACHE_MAGIC)?;
    write_u32(&mut writer, FLAT_CACHE_VERSION)?;
    write_u64(&mut writer, cache.items as u64)?;
    write_u64(&mut writer, cache.spectrum_width as u64)?;
    write_u64(&mut writer, cache.condition_width as u64)?;

    let bar = preload_bytes_bar(
        prefix,
        cache.payload_bytes()?,
        "writing preprocessed vector cache",
    );
    let result = (|| -> io::Result<()> {
        write_f32_slice(
            &mut writer,
            &cache.spectra,
            &bar,
            "writing spectra vector cache",
        )?;
        write_f32_slice(
            &mut writer,
            &cache.conditions,
            &bar,
            "writing conditions vector cache",
        )?;
        writer.flush()
    })();

    match result {
        Ok(()) => {
            drop(writer);
            fs::rename(&tmp_path, path)?;
            bar.finish_with_message(format!("wrote preprocessed vectors to {}", path.display()));
            Ok(())
        }
        Err(error) => {
            bar.finish_with_message(format!("failed writing preprocessed vectors: {error}"));
            let _ = fs::remove_file(&tmp_path);
            Err(error)
        }
    }
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

fn read_f32_vec(
    reader: &mut impl Read,
    len: usize,
    bar: &ProgressBar,
    message: &'static str,
) -> io::Result<Vec<f32>> {
    const CHUNK_FLOATS: usize = 1 << 20;
    let mut values = Vec::with_capacity(len);
    let mut remaining = len;
    let mut bytes = vec![0_u8; CHUNK_FLOATS * std::mem::size_of::<f32>()];

    while remaining > 0 {
        let chunk_len = remaining.min(CHUNK_FLOATS);
        let byte_len = chunk_len * std::mem::size_of::<f32>();
        bar.set_message(message);
        reader.read_exact(&mut bytes[..byte_len])?;
        values.extend(
            bytes[..byte_len]
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])),
        );
        bar.inc(byte_len as u64);
        remaining -= chunk_len;
    }

    Ok(values)
}

fn write_f32_slice(
    writer: &mut impl Write,
    values: &[f32],
    bar: &ProgressBar,
    message: &'static str,
) -> io::Result<()> {
    const CHUNK_FLOATS: usize = 1 << 20;
    let mut bytes = Vec::with_capacity(CHUNK_FLOATS * std::mem::size_of::<f32>());
    for chunk in values.chunks(CHUNK_FLOATS) {
        bytes.clear();
        bytes.extend(chunk.iter().flat_map(|value| value.to_le_bytes()));
        bar.set_message(message);
        writer.write_all(&bytes)?;
        bar.inc(bytes.len() as u64);
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

struct CachedVectorizedMgfIter<'a, B, Open>
where
    B: Backend,
{
    loader: &'a CachedVectorizedMgfLoader<B, Open>,
    records: Option<VectorizedMgfIter>,
    cache: Option<FlatGpuCache<B>>,
    cache_offset: usize,
    batches_processed: usize,
    items_processed: usize,
    epoch_index: u64,
    finished: bool,
}

impl<B, Open> Iterator for CachedVectorizedMgfIter<'_, B, Open>
where
    B: SimilarityTeacherBackend,
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
        let (
            spectra,
            conditions,
            masked_precursor_mask,
            masked_spectra_mask,
            intruder_peak_mask,
            consistency_spectra,
            consistency_conditions,
        ) = match self.loader.augment {
            Some(config) => {
                let (
                    spectra,
                    conditions,
                    masked_precursor_mask,
                    masked_spectra_mask,
                    intruder_peak_mask,
                ) = augment_flat_batch(clean_spectra.clone(), clean_conditions.clone(), config);
                let (consistency_spectra, consistency_conditions, _, _, _) = augment_flat_batch(
                    clean_spectra.clone(),
                    clean_conditions.clone(),
                    config.without_intruder_peaks(),
                );
                (
                    spectra,
                    conditions,
                    masked_precursor_mask,
                    masked_spectra_mask,
                    intruder_peak_mask,
                    consistency_spectra,
                    consistency_conditions,
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
                    clean_spectra,
                    clean_conditions,
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
            consistency_spectra,
            consistency_conditions,
            masked_spectra_mask,
            intruder_peak_mask,
            similarity_ranking,
        })
    }
}

impl<B, Open> CachedVectorizedMgfIter<'_, B, Open>
where
    B: Backend,
    Open: Fn() -> spectral_autoencoder::Result<VectorizedMgfIter> + Send + Sync + Clone + 'static,
{
    fn fill_cache(&mut self) -> Option<FlatGpuCache<B>> {
        let remaining_items = loader_epoch_items(self.loader.batch_size, self.loader.max_batches)
            .saturating_sub(self.items_processed);
        let target_items = self.loader.cache_items.min(remaining_items);
        if target_items == 0 {
            return None;
        }

        if let Some(cpu_cache) = self
            .loader
            .cpu_cache
            .lock()
            .expect("CPU vectorized cache lock should not be poisoned")
            .clone()
        {
            return Some(self.loader.move_flat_cache_window_to_gpu(
                &cpu_cache,
                self.items_processed,
                target_items,
                None,
            ));
        }

        let records = self.records.as_mut()?;
        let cpu_cache =
            self.loader
                .read_flat_cpu_cache(records, target_items, self.batches_processed, None)?;
        let cache = self
            .loader
            .move_flat_cache_window_to_gpu(&cpu_cache, 0, cpu_cache.items, None);
        if self.loader.is_full_cache() {
            *self
                .loader
                .full_cache
                .lock()
                .expect("full vectorized cache lock should not be poisoned") = Some(cache.clone());
        }
        Some(cache)
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

impl<B, Open> DataLoaderIterator<AutoencoderBatch<B>> for CachedVectorizedMgfIter<'_, B, Open>
where
    B: SimilarityTeacherBackend,
    Open: Fn() -> spectral_autoencoder::Result<VectorizedMgfIter> + Send + Sync + Clone + 'static,
{
    fn progress(&self) -> Progress {
        Progress {
            items_processed: self.items_processed,
            items_total: loader_epoch_items(self.loader.batch_size, self.loader.max_batches),
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

fn preload_transfer_bar(
    prefix: String,
    items: usize,
    message: &'static str,
    stages: usize,
) -> ProgressBar {
    let bar = ProgressBar::with_draw_target(
        Some((items * stages) as u64),
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

fn usize_list_var(name: &str, default: &[usize]) -> Result<Vec<usize>, Box<dyn StdError>> {
    let Some(value) = env::var(name).ok() else {
        return Ok(default.to_vec());
    };
    let value = value.trim();
    if value.is_empty() {
        return Ok(default.to_vec());
    }
    let mut values = Vec::new();
    for item in value.split(',') {
        let item = item.trim();
        if item.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("{name} contains an empty width"),
            )
            .into());
        }
        let width = item.parse::<usize>().map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("failed to parse {name} item {item:?}: {error}"),
            )
        })?;
        if width == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("{name} widths must be greater than zero"),
            )
            .into());
        }
        values.push(width);
    }
    Ok(values)
}
