use std::{
    env,
    error::Error as StdError,
    fmt::Display,
    io,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use burn::{
    backend::{Autodiff, Cuda},
    module::Module,
    record::{CompactRecorder, Record, Recorder},
    tensor::{Bool, Distribution, Tensor, TensorData, backend::Backend},
};
use clap::ValueEnum;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use mascot_rs::prelude::{
    Dataset, GEMS_A10_TOP_60_ZENODO_DOI, GEMS_A10_TOP_128_ZENODO_DOI, GemsA10Builder, GemsA10Iter,
    MGFVec, MascotError,
};
use mass_spectrometry::burn::{
    AllMetricsBackend, KernelMetric, LinearCosineMetric, LinearEntropyMetric,
    ModifiedLinearCosineMetric, ModifiedLinearEntropyMetric, RankingConfig, RankingWindow,
    SpectrumBatch, ranking_kernel,
};
use spectral_autoencoder::{
    AuxiliaryLossConfig, ConditioningConfig, FlatVectorReconstructionOrdering,
    SimilarityRankingBatch, SpectrumAugmentationConfig,
};

pub type InnerBackend = Cuda<f32, i32>;
pub type TrainingBackend = Autodiff<InnerBackend>;

#[derive(Debug, Clone)]
pub struct RunArgs {
    pub mgf_source: String,
    pub mgf_paths: Vec<PathBuf>,
    pub gems_builder: GemsA10Builder<f32>,
    pub max_peaks: usize,
    pub output_dir: PathBuf,
    pub device: usize,
    pub batch_size: usize,
    pub train_batches: usize,
    pub valid_batches: usize,
    pub epochs: usize,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub checkpoints: bool,
    pub resume_epoch: Option<usize>,
    pub warm_start_model: Option<PathBuf>,
    pub train_offset: Option<usize>,
    pub valid_offset: Option<usize>,
    pub gpu_transfer_chunk_batches: usize,
}

/// Per-variant on-disk preprocessed-cache options.
#[derive(Debug, Clone, Default)]
pub struct PreprocessedCacheOptions {
    /// `true` to keep the cache enabled; `false` disables and panics on use
    /// (matches the previous behaviour, which required the cache for the
    /// host-worker streaming loader).
    pub enabled: bool,
    /// Explicit cache directory override; `None` falls back to the
    /// `datasets/.../preprocessed-{flat|tokens}` default.
    pub dir: Option<PathBuf>,
    /// Rebuild the cache before training.
    pub refresh: bool,
}

impl RunArgs {
    pub const fn valid_items(&self) -> usize {
        self.batch_size * self.valid_batches
    }

    pub fn train_start_item(&self) -> usize {
        self.train_offset.unwrap_or_else(|| self.valid_items())
    }

    pub fn valid_start_item(&self) -> usize {
        self.valid_offset.unwrap_or(0)
    }
}

pub struct ResolvedGemsA10 {
    pub source: String,
    pub paths: Vec<PathBuf>,
    pub builder: GemsA10Builder<f32>,
}

pub fn resolve_mascot_gems_a10(
    max_peaks: usize,
    dataset_dir: Option<&Path>,
    dataset_parts: Option<&str>,
    download: bool,
    force_download: bool,
    progress: ProgressMode,
) -> Result<ResolvedGemsA10, Box<dyn StdError>> {
    let default_directory = format!("datasets/gems-a10-top-{max_peaks}-peaks");
    let target_directory: PathBuf = dataset_dir
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from(&default_directory));
    let builder = match max_peaks {
        60 => MGFVec::<f32>::gems_a10_top_60_peaks(),
        128 => MGFVec::<f32>::gems_a10_top_128_peaks(),
        other => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("--max-peaks {other} is unsupported; use 60 or 128"),
            )
            .into());
        }
    };
    let mut builder = builder
        .target_directory(&target_directory)
        .force_download(false);
    if let Some(value) = dataset_parts
        && let Some(parts) = parse_dataset_parts(value)?
    {
        builder = builder.parts(parts)?;
    }
    if let Some(token) = gems_a10_token() {
        builder = builder.token(token);
    }
    if progress.visible() {
        builder = builder.verbose();
    }

    ensure_mascot_gems_a10_files(
        &builder.clone().force_download(force_download),
        download,
        force_download,
    )?;
    let paths = builder.paths();
    let doi = match max_peaks {
        60 => GEMS_A10_TOP_60_ZENODO_DOI,
        128 => GEMS_A10_TOP_128_ZENODO_DOI,
        _ => unreachable!("unsupported GeMS peak count should already be rejected"),
    };
    Ok(ResolvedGemsA10 {
        source: format!(
            "mascot-rs GeMS-A10 top-{max_peaks} {doi} ({} files in {})",
            paths.len(),
            target_directory.display()
        ),
        paths,
        builder,
    })
}

fn parse_dataset_parts(value: &str) -> Result<Option<Vec<u8>>, Box<dyn StdError>> {
    let value = value.trim();
    if value.is_empty() || value.eq_ignore_ascii_case("all") {
        return Ok(None);
    }

    let mut parts = Vec::new();
    for token in value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
    {
        if let Some((start, end)) = token.split_once("..") {
            let start = start.trim().parse::<u8>()?;
            let end = end.trim().parse::<u8>()?;
            parts.extend(start..end);
        } else if let Some((start, end)) = token.split_once('-') {
            let start = start.trim().parse::<u8>()?;
            let end = end.trim().parse::<u8>()?;
            parts.extend(start..=end);
        } else {
            parts.push(token.parse::<u8>()?);
        }
    }

    if parts.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "--dataset-parts did not contain any part numbers",
        )
        .into());
    }
    Ok(Some(parts))
}

fn ensure_mascot_gems_a10_files(
    builder: &GemsA10Builder<f32>,
    download: bool,
    force_download: bool,
) -> Result<(), Box<dyn StdError>> {
    let paths = builder.paths();
    let missing_or_forced = paths
        .iter()
        .filter(|path| force_download || !path.try_exists().unwrap_or(false))
        .count();
    if missing_or_forced == 0 {
        return Ok(());
    }
    if !download {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "{missing_or_forced} GeMS-A10 file(s) are missing under {}; pass --download or select fewer parts with --dataset-parts",
                paths.first()
                    .and_then(|path| path.parent())
                    .map_or_else(|| Path::new(".").display().to_string(), |path| path.display().to_string())
            ),
        )
        .into());
    }

    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?
        .block_on(builder.clone().download())?;
    Ok(())
}

pub fn open_gems_a10_iter(
    builder: GemsA10Builder<f32>,
) -> spectral_autoencoder::Result<GemsA10Iter<f32>> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|source| MascotError::InputIo { source })
        .and_then(|runtime| runtime.block_on(<GemsA10Builder<f32> as Dataset>::mgf_iter(builder)))
        .map_err(Into::into)
}

fn gems_a10_token() -> Option<String> {
    env::var("GEMS_ZENODO_TOKEN")
        .ok()
        .or_else(|| env::var("ZENODO_TOKEN").ok())
        .filter(|token| !token.trim().is_empty())
}

pub fn print_streaming_run_header(
    model_name: &str,
    args: &RunArgs,
    parameter_count: usize,
    streaming: &StreamingTrainingLoaderConfig,
    auxiliary: AuxiliaryLossConfig,
    similarity_teacher: SimilarityTeacherConfig,
    flat_reconstruction_ordering: Option<FlatVectorReconstructionOrdering>,
) {
    println!("GeMS {model_name} streaming-window training");
    println!("mgf source: {}", args.mgf_source);
    println!("mgf files: {}", args.mgf_paths.len());
    println!("max peaks: {}", args.max_peaks);
    if let Some(first_path) = args.mgf_paths.first() {
        println!("mgf first: {}", first_path.display());
    }
    if args.mgf_paths.len() > 1
        && let Some(last_path) = args.mgf_paths.last()
    {
        println!("mgf last: {}", last_path.display());
    }
    println!("output: {}", args.output_dir.display());
    println!("device: cuda:{}", args.device);
    println!("batch size: {}", args.batch_size);
    println!("train batches per epoch: {}", args.train_batches);
    println!("valid batches per epoch: {}", args.valid_batches);
    println!(
        "train split starts at spectrum: {}",
        args.train_start_item()
    );
    println!(
        "valid split starts at spectrum: {}",
        args.valid_start_item()
    );
    println!("epochs: {}", args.epochs);
    println!("learning rate: {}", args.learning_rate);
    println!("weight decay: {}", args.weight_decay);
    println!(
        "checkpoints: {}",
        if args.checkpoints {
            format!("enabled ({}/checkpoint)", args.output_dir.display())
        } else {
            "disabled".to_string()
        }
    );
    if let Some(epoch) = args.resume_epoch {
        println!("resume checkpoint epoch: {epoch}");
    }
    if let Some(path) = &args.warm_start_model {
        println!("warm-start model: {}", path.display());
    }
    println!(
        "gpu feed: streaming windows ({} batches/window, {} host worker(s), {} host prefetch window(s), profile every {} window(s))",
        streaming.gpu_window_batches,
        streaming.loader_workers,
        streaming.host_prefetch_windows,
        streaming.loader_profile_every
    );
    println!("model parameters: {parameter_count}");
    if let Some(ordering) = flat_reconstruction_ordering {
        println!("flat reconstruction ordering: {}", ordering.label());
    }
    println!("similarity teacher: {}", similarity_teacher.summary());
    println!(
        "auxiliary losses: reconstruction {} masked {} intruder {} precursor {} masked-precursor {} similarity-ranking {} similarity-ranking-latent-temperature {} latent-noise-std {} similarity-ranking-pairs/batch {}",
        auxiliary.reconstruction_weight,
        auxiliary.masked_peak_weight,
        auxiliary.intruder_peak_weight,
        auxiliary.precursor_reconstruction_weight,
        auxiliary.masked_precursor_weight,
        auxiliary.similarity_ranking_weight,
        auxiliary.similarity_ranking_latent_temperature,
        auxiliary.latent_noise_std,
        if auxiliary.similarity_ranking_pairs_per_batch == 0 {
            "all".to_string()
        } else {
            auxiliary.similarity_ranking_pairs_per_batch.to_string()
        }
    );
}

#[derive(Debug, Clone, Copy)]
pub enum SimilarityTeacherMetric {
    LinearCosine,
    ModifiedLinearCosine,
    LinearEntropy,
    ModifiedLinearEntropy,
}

impl SimilarityTeacherMetric {
    fn label(self) -> &'static str {
        match self {
            Self::LinearCosine => "linear cosine",
            Self::ModifiedLinearCosine => "modified linear cosine",
            Self::LinearEntropy => "linear entropy",
            Self::ModifiedLinearEntropy => "modified linear entropy",
        }
    }

    pub fn is_entropy(self) -> bool {
        matches!(self, Self::LinearEntropy | Self::ModifiedLinearEntropy)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SimilarityTeacherConfig {
    enabled: bool,
    metric: SimilarityTeacherMetric,
    mz_tolerance: f64,
    max_mz: f64,
    mz_power: f64,
    intensity_power: f64,
    candidates_per_anchor: usize,
    weighted_entropy: bool,
}

impl SimilarityTeacherConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        auxiliary: AuxiliaryLossConfig,
        metric: SimilarityTeacherMetric,
        mz_tolerance: f64,
        max_mz: f64,
        mz_power: f64,
        intensity_power: f64,
        candidates_per_anchor: usize,
        weighted_entropy: bool,
    ) -> Result<Self, Box<dyn StdError>> {
        let enabled = auxiliary.similarity_ranking_weight > 0.0;
        let candidates_per_anchor = if enabled {
            candidates_per_anchor.max(2)
        } else {
            candidates_per_anchor
        };
        let config = Self {
            enabled,
            metric,
            mz_tolerance,
            max_mz,
            mz_power,
            intensity_power,
            candidates_per_anchor,
            weighted_entropy,
        };
        config.validate()?;
        Ok(config)
    }

    pub(crate) const fn enabled(self) -> bool {
        self.enabled
    }

    fn summary(self) -> String {
        if !self.enabled() {
            return "disabled".to_string();
        }
        let weighted_suffix = if self.metric.is_entropy() && self.weighted_entropy {
            " (weighted)"
        } else {
            ""
        };
        format!(
            "online {}{} teacher on CUDA, tolerance {} Da, candidates/anchor {}",
            self.metric.label(),
            weighted_suffix,
            self.mz_tolerance,
            self.candidates_per_anchor
        )
    }

    fn validate(self) -> Result<(), Box<dyn StdError>> {
        if !self.enabled() {
            return Ok(());
        }
        if !(self.max_mz.is_finite() && self.max_mz > 0.0) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "--similarity-ranking-max-mz must be finite and positive",
            )
            .into());
        }
        if !cfg!(feature = "cuda") {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "similarity ranking requires a CUDA feature",
            )
            .into());
        }
        if !(self.mz_tolerance.is_finite() && self.mz_tolerance > 0.0) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "--similarity-ranking-mz-tolerance must be finite and positive",
            )
            .into());
        }
        if !(self.mz_power.is_finite() && self.mz_power >= 0.0) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "--similarity-ranking-mz-power must be finite and non-negative",
            )
            .into());
        }
        if !(self.intensity_power.is_finite() && self.intensity_power >= 0.0) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "--similarity-ranking-intensity-power must be finite and non-negative",
            )
            .into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct StreamingTrainingLoaderConfig {
    pub(crate) gpu_window_batches: usize,
    pub(crate) loader_workers: usize,
    pub(crate) host_prefetch_windows: usize,
    pub(crate) loader_profile_every: usize,
    pub(crate) similarity_teacher: SimilarityTeacherConfig,
    pub(crate) loader_profile_sink: crate::streaming::LoaderProfileSink,
}

impl StreamingTrainingLoaderConfig {
    #[must_use]
    pub fn new(
        gpu_window_batches: usize,
        loader_workers: usize,
        host_prefetch_windows: usize,
        loader_profile_every: usize,
        similarity_teacher: SimilarityTeacherConfig,
        loader_profile_sink: crate::streaming::LoaderProfileSink,
    ) -> Self {
        Self {
            gpu_window_batches,
            loader_workers,
            host_prefetch_windows,
            loader_profile_every,
            similarity_teacher,
            loader_profile_sink,
        }
    }

    pub(crate) fn window_items(&self, batch_size: usize, max_batches: usize) -> usize {
        let epoch_items = loader_epoch_items(batch_size, max_batches);
        let window_batches = if max_batches > 1 {
            self.gpu_window_batches.min(max_batches - 1)
        } else {
            1
        };
        batch_size
            .saturating_mul(window_batches)
            .max(batch_size)
            .min(epoch_items)
    }
}

pub fn save_model_record<R>(
    progress: &GeMSProgress,
    record: R,
    model_path: PathBuf,
    spinner_message: &'static str,
    finish_message: &'static str,
) -> Result<(), Box<dyn std::error::Error>>
where
    R: Record<InnerBackend>,
{
    let save_bar = progress.spinner(spinner_message);
    CompactRecorder::new().record(record, model_path.clone())?;
    save_bar.finish_with_message(finish_message);
    println!("saved model: {}.mpk", model_path.display());
    Ok(())
}

pub fn warm_start_model<B, M>(
    progress: &GeMSProgress,
    model: M,
    device: &B::Device,
    model_path: Option<&Path>,
    model_name: &str,
) -> Result<M, Box<dyn std::error::Error>>
where
    B: Backend,
    M: Module<B>,
{
    let Some(model_path) = model_path else {
        return Ok(model);
    };

    let load_bar = progress.spinner("load warm-start model");
    let record = CompactRecorder::new().load(model_path.to_path_buf(), device)?;
    load_bar.finish_with_message("loaded warm-start model record");
    println!(
        "warm-started {model_name} model from {}",
        model_path.display()
    );
    Ok(model.load_record(record))
}

pub(crate) struct StreamingLoaderOptions<B: Backend> {
    pub(crate) mgf_source: String,
    pub(crate) batch_size: usize,
    pub(crate) max_batches: usize,
    pub(crate) start_item: usize,
    pub(crate) window_items: usize,
    pub(crate) device: B::Device,
    pub(crate) progress: LoaderProgress,
    pub(crate) augment: Option<SpectrumAugmentationConfig>,
    pub(crate) similarity_teacher: SimilarityTeacherConfig,
    pub(crate) randomize_pair_sampling: bool,
}

pub(crate) fn loader_epoch_items(batch_size: usize, max_batches: usize) -> usize {
    batch_size.saturating_mul(max_batches)
}

#[derive(Clone)]
pub(crate) struct PairSamplingEpoch {
    randomize: bool,
    epoch: Arc<AtomicU64>,
}

impl PairSamplingEpoch {
    pub(crate) fn new(randomize: bool) -> Self {
        Self {
            randomize,
            epoch: Arc::new(AtomicU64::new(0)),
        }
    }

    pub(crate) fn next(&self) -> u64 {
        if self.randomize {
            self.epoch.fetch_add(1, Ordering::Relaxed) + 1
        } else {
            0
        }
    }
}

pub(crate) fn open_records_or_panic<I, E, Open>(
    open_records: &Open,
    kind: &str,
    mgf_source: &str,
) -> I
where
    Open: Fn() -> Result<I, E>,
    E: Display,
{
    open_records().unwrap_or_else(|error| {
        panic!("failed to open cached {kind} MGF iterator for {mgf_source}: {error}")
    })
}

pub(crate) fn finish_loader_once(
    progress: &LoaderProgress,
    items_processed: usize,
    batches_processed: usize,
    finished: &mut bool,
) {
    if *finished {
        return;
    }
    *finished = true;
    progress.finish(items_processed, batches_processed);
}

fn pair_sampling_seed(epoch_index: u64, batch_index: usize, item_offset: usize) -> u64 {
    let mut value = epoch_index
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .wrapping_add((batch_index as u64).rotate_left(23))
        .wrapping_add((item_offset as u64).rotate_left(41));
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

pub(crate) fn similarity_pair_seed(
    epoch_index: u64,
    batch_index: usize,
    item_offset: usize,
) -> u64 {
    pair_sampling_seed(
        epoch_index ^ 0xa5a5_5a5a_d3c1_b2e9,
        batch_index,
        item_offset,
    )
}

#[derive(Clone)]
pub(crate) struct TeacherSpectraCache {
    fixed_mz: Vec<f32>,
    fixed_intensity: Vec<f32>,
    precursor_mz: Vec<f32>,
    fixed_peak_width: usize,
    items: usize,
}

impl TeacherSpectraCache {
    fn items(&self) -> usize {
        self.items
    }
}

pub(crate) struct TeacherSpectraBuilder {
    config: SimilarityTeacherConfig,
    fixed_mz: Vec<f32>,
    fixed_intensity: Vec<f32>,
    precursor_mz: Vec<f32>,
    fixed_peak_width: usize,
    expected_items: usize,
    items: usize,
}

impl TeacherSpectraBuilder {
    pub(crate) fn new(config: SimilarityTeacherConfig, items: usize) -> Self {
        Self {
            config,
            fixed_mz: Vec::new(),
            fixed_intensity: Vec::new(),
            precursor_mz: Vec::new(),
            fixed_peak_width: 0,
            expected_items: items,
            items: 0,
        }
    }

    pub(crate) fn push_pairs(&mut self, target_pairs: &[f32], conditions: &[f32]) {
        if self.fixed_peak_width == 0 {
            self.fixed_peak_width = target_pairs.len() / 2;
            self.fixed_mz
                .reserve(self.expected_items.saturating_mul(self.fixed_peak_width));
            self.fixed_intensity
                .reserve(self.expected_items.saturating_mul(self.fixed_peak_width));
            self.precursor_mz.reserve(self.expected_items);
        }
        let peaks = preprocess_teacher_pairs(target_pairs, self.config);
        for peak_index in 0..self.fixed_peak_width {
            let (mz, intensity) = peaks.get(peak_index).copied().unwrap_or((0.0, 0.0));
            self.fixed_mz.push(mz);
            self.fixed_intensity.push(intensity);
        }
        self.precursor_mz
            .push(teacher_precursor_from_conditions(conditions));
        self.items += 1;
    }

    pub(crate) fn finish(self) -> TeacherSpectraCache {
        TeacherSpectraCache {
            fixed_mz: self.fixed_mz,
            fixed_intensity: self.fixed_intensity,
            precursor_mz: self.precursor_mz,
            fixed_peak_width: self.fixed_peak_width,
            items: self.items,
        }
    }
}

#[derive(Clone)]
pub(crate) struct TeacherGpuCache<B: Backend> {
    mz: Tensor<B, 2>,
    intensity: Tensor<B, 2>,
    precursor: Tensor<B, 1>,
    peak_width: usize,
}

impl<B: Backend> TeacherGpuCache<B> {
    pub(crate) fn from_cpu_window(
        cache: &TeacherSpectraCache,
        start: usize,
        end: usize,
        device: &B::Device,
    ) -> Option<Self> {
        let peak_width = cache.fixed_peak_width;
        if peak_width == 0 || start >= end || end > cache.items() {
            return None;
        }
        let start_offset = start * peak_width;
        let end_offset = end * peak_width;
        let items = end - start;
        Some(Self {
            mz: Tensor::<B, 2>::from_data(
                TensorData::new(
                    cache.fixed_mz[start_offset..end_offset].to_vec(),
                    [items, peak_width],
                ),
                device,
            ),
            intensity: Tensor::<B, 2>::from_data(
                TensorData::new(
                    cache.fixed_intensity[start_offset..end_offset].to_vec(),
                    [items, peak_width],
                ),
                device,
            ),
            precursor: Tensor::<B, 1>::from_data(
                TensorData::new(cache.precursor_mz[start..end].to_vec(), [items]),
                device,
            ),
            peak_width,
        })
    }
}

fn teacher_precursor_from_conditions(conditions: &[f32]) -> f32 {
    let precursor = conditions.first().copied().unwrap_or(0.0);
    let present = conditions.get(1).copied().unwrap_or(0.0);
    if present > 0.0 && precursor.is_finite() && precursor > 0.0 {
        precursor * ConditioningConfig::default().precursor_mz_scale() as f32
    } else {
        panic!("similarity ranking requires finite precursor m/z conditions");
    }
}

fn preprocess_teacher_pairs(
    target_pairs: &[f32],
    config: SimilarityTeacherConfig,
) -> Vec<(f32, f32)> {
    let mut peaks = target_pairs
        .chunks_exact(2)
        .filter_map(|pair| {
            let mz = f64::from(pair[0]) * config.max_mz;
            let intensity = f64::from(pair[1]);
            if mz.is_finite() && intensity.is_finite() && mz > 0.0 && intensity > 0.0 {
                Some((mz, intensity))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    if peaks.len() < 2 {
        return peaks
            .into_iter()
            .map(|(mz, intensity)| (mz as f32, intensity as f32))
            .collect();
    }

    peaks.sort_by(|left, right| left.0.total_cmp(&right.0));
    let merge_window = config.mz_tolerance + config.mz_tolerance;
    let mut order = (0..peaks.len()).collect::<Vec<_>>();
    order.sort_by(|&left, &right| {
        peaks[right]
            .1
            .total_cmp(&peaks[left].1)
            .then_with(|| peaks[left].0.total_cmp(&peaks[right].0))
    });

    let mut consumed = vec![false; peaks.len()];
    let mut survivors = Vec::with_capacity(peaks.len());
    for index in order {
        if consumed[index] {
            continue;
        }
        consumed[index] = true;
        let dominant_mz = peaks[index].0;
        let mut summed_intensity = peaks[index].1;

        let mut left = index;
        while left > 0 {
            left -= 1;
            if consumed[left] {
                continue;
            }
            if dominant_mz - peaks[left].0 <= merge_window {
                summed_intensity = (summed_intensity + peaks[left].1).min(f64::MAX);
                consumed[left] = true;
            } else {
                break;
            }
        }

        for right in (index + 1)..peaks.len() {
            if consumed[right] {
                continue;
            }
            if peaks[right].0 - dominant_mz <= merge_window {
                summed_intensity = (summed_intensity + peaks[right].1).min(f64::MAX);
                consumed[right] = true;
            } else {
                break;
            }
        }
        survivors.push((dominant_mz as f32, summed_intensity as f32));
    }

    survivors.sort_by(|left, right| left.0.total_cmp(&right.0));
    survivors
}

pub(crate) fn teacher_similarity_ranking_batch<B: AllMetricsBackend>(
    teacher_gpu: Option<&TeacherGpuCache<B>>,
    batch_start: usize,
    batch_items: usize,
    seed: u64,
    config: SimilarityTeacherConfig,
    device: &B::Device,
) -> SimilarityRankingBatch<B> {
    if !config.enabled() || batch_items < 3 {
        return SimilarityRankingBatch::zeros(batch_items, device);
    };

    let Some(teacher_gpu) = teacher_gpu else {
        panic!("similarity ranking is enabled but the GPU teacher cache is missing");
    };
    if teacher_gpu.peak_width == 0 {
        panic!("similarity ranking is enabled but the GPU teacher cache has no peaks");
    }

    #[cfg(feature = "cuda")]
    {
        let teacher_batch = SpectrumBatch::new(
            teacher_gpu.mz.clone(),
            teacher_gpu.intensity.clone(),
            teacher_gpu.precursor.clone(),
        );
        let window = RankingWindow::new()
            .with_batch_start(batch_start)
            .with_batch_items(batch_items)
            .with_candidates_per_anchor(config.candidates_per_anchor)
            .with_seed(seed);
        let mz_power = config.mz_power as f32;
        let intensity_power = config.intensity_power as f32;
        let mz_tolerance = config.mz_tolerance as f32;
        let max_peaks = teacher_gpu.peak_width;
        let weighted = config.weighted_entropy;

        match config.metric {
            SimilarityTeacherMetric::LinearCosine => {
                let scoring = LinearCosineMetric::scoring_params()
                    .with_mz_power(mz_power)
                    .with_intensity_power(intensity_power)
                    .with_mz_tolerance(mz_tolerance)
                    .with_max_peaks(max_peaks);
                ranking_kernel(teacher_batch, RankingConfig::from_parts(scoring, window)).into()
            }
            SimilarityTeacherMetric::ModifiedLinearCosine => {
                let scoring = ModifiedLinearCosineMetric::scoring_params()
                    .with_mz_power(mz_power)
                    .with_intensity_power(intensity_power)
                    .with_mz_tolerance(mz_tolerance)
                    .with_max_peaks(max_peaks);
                ranking_kernel(teacher_batch, RankingConfig::from_parts(scoring, window)).into()
            }
            SimilarityTeacherMetric::LinearEntropy => {
                let scoring = LinearEntropyMetric::scoring_params()
                    .with_mz_power(mz_power)
                    .with_intensity_power(intensity_power)
                    .with_mz_tolerance(mz_tolerance)
                    .with_max_peaks(max_peaks)
                    .with_weighted(weighted);
                ranking_kernel(teacher_batch, RankingConfig::from_parts(scoring, window)).into()
            }
            SimilarityTeacherMetric::ModifiedLinearEntropy => {
                let scoring = ModifiedLinearEntropyMetric::scoring_params()
                    .with_mz_power(mz_power)
                    .with_intensity_power(intensity_power)
                    .with_mz_tolerance(mz_tolerance)
                    .with_max_peaks(max_peaks)
                    .with_weighted(weighted);
                ranking_kernel(teacher_batch, RankingConfig::from_parts(scoring, window)).into()
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        panic!("similarity ranking requires a CUDA feature");
    }
}

pub(crate) fn mask_precursor_conditions<B: Backend>(
    conditions: Tensor<B, 2>,
    probability: f32,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let [batch_size, condition_width] = conditions.dims();
    let device = conditions.device();
    let zero_mask = Tensor::<B, 2>::zeros([batch_size, 1], &device);
    if condition_width < 2 {
        return (conditions, zero_mask);
    }

    let Some(mask) = probability_mask([batch_size, 1], &device, probability) else {
        return (conditions, zero_mask);
    };
    let present = conditions.clone().narrow(1, 1, 1).greater_elem(0.0);
    let mask = mask.bool_and(present);
    let expanded_mask = mask.clone().expand([batch_size, condition_width]);
    (conditions.mask_fill(expanded_mask, 0.0), mask.float())
}

pub(crate) fn probability_mask<B: Backend, const D: usize>(
    shape: [usize; D],
    device: &B::Device,
    probability: f32,
) -> Option<Tensor<B, D, Bool>> {
    let probability = probability.clamp(0.0, 1.0);
    (probability > 0.0).then(|| {
        Tensor::<B, D>::random(shape, Distribution::Uniform(0.0, 1.0), device)
            .lower_elem(probability)
    })
}

pub(crate) fn signed_random<B: Backend, const D: usize>(
    shape: [usize; D],
    device: &B::Device,
    range: f32,
) -> Tensor<B, D> {
    if range <= 0.0 {
        return Tensor::<B, D>::zeros(shape, device);
    }

    (Tensor::<B, D>::random(shape, Distribution::Uniform(0.0, 1.0), device) * 2.0 - 1.0) * range
}

pub(crate) fn skip_split_records<I, R>(
    records: &mut R,
    start_item: usize,
    progress: &LoaderProgress,
    preload_bar: Option<&ProgressBar>,
) where
    R: Iterator<Item = spectral_autoencoder::Result<I>>,
{
    if start_item == 0 {
        return;
    }

    let mut skipped_items = 0usize;
    let mut reported_items = 0usize;
    while skipped_items < start_item {
        match records.next() {
            Some(Ok(_)) => {
                skipped_items += 1;
                if skipped_items.is_multiple_of(progress.batch_size) || skipped_items == start_item
                {
                    if let Some(bar) = preload_bar {
                        bar.inc((skipped_items - reported_items) as u64);
                        reported_items = skipped_items;
                        bar.set_message(format!(
                            "seeking split offset {skipped_items}/{start_item}"
                        ));
                    }
                    progress.split_skipping(skipped_items, start_item);
                }
            }
            Some(Err(error)) => {
                panic!("failed to read MGF record while seeking split offset: {error}");
            }
            None => break,
        }
    }
}

pub struct GeMSProgress {
    multi: Arc<MultiProgress>,
    pub train: LoaderProgress,
    pub valid: LoaderProgress,
}

impl GeMSProgress {
    pub fn new(
        train_batches: usize,
        valid_batches: usize,
        batch_size: usize,
        action: &str,
        progress_mode: ProgressMode,
    ) -> Self {
        let multi = Arc::new(MultiProgress::with_draw_target(progress_mode.draw_target()));
        Self {
            train: LoaderProgress::new(
                multi.clone(),
                "train loader",
                train_batches,
                batch_size,
                action,
                progress_mode,
            ),
            valid: LoaderProgress::new(
                multi.clone(),
                "valid loader",
                valid_batches,
                batch_size,
                action,
                progress_mode,
            ),
            multi,
        }
    }

    pub fn start_training(&self, message: &'static str) {
        self.multi.println(message).ok();
    }

    pub fn finish_training(&self, message: &'static str) {
        self.multi.println(message).ok();
    }

    pub fn spinner(&self, message: &'static str) -> ProgressBar {
        let spinner = self.multi.add(ProgressBar::new_spinner());
        spinner.set_style(spinner_style());
        spinner.enable_steady_tick(Duration::from_millis(100));
        spinner.set_message(message);
        spinner
    }
}

#[derive(Clone)]
pub struct LoaderProgress {
    bar: ProgressBar,
    pub(crate) label: String,
    pub(crate) max_batches: usize,
    pub(crate) batch_size: usize,
    action: String,
    progress_mode: ProgressMode,
}

impl LoaderProgress {
    fn new(
        multi: Arc<MultiProgress>,
        label: &'static str,
        max_batches: usize,
        batch_size: usize,
        action: &str,
        progress_mode: ProgressMode,
    ) -> Self {
        let bar = multi.add(ProgressBar::new((max_batches * batch_size) as u64));
        bar.set_style(tokenization_style());
        bar.set_prefix(label);
        bar.set_message("waiting for epoch");
        Self {
            bar,
            label: label.to_string(),
            max_batches,
            batch_size,
            action: action.to_string(),
            progress_mode,
        }
    }

    pub(crate) fn clone_for_slice(&self) -> Self {
        Self {
            bar: self.bar.clone(),
            label: self.label.clone(),
            max_batches: self.max_batches,
            batch_size: self.batch_size,
            action: self.action.clone(),
            progress_mode: self.progress_mode,
        }
    }

    pub(crate) fn start_epoch(&self, total_items: usize) {
        self.bar.reset();
        self.bar.set_length(total_items as u64);
        self.bar.set_prefix(self.label.clone());
        self.bar.set_message(format!(
            "opening MGF stream; target batches 0/{}",
            self.max_batches
        ));
    }

    pub(crate) fn batch_loaded(
        &self,
        batch_items: usize,
        batches_processed: usize,
        max_batches: usize,
    ) {
        self.bar.inc(batch_items as u64);
        self.bar.set_message(format!(
            "{} batches {batches_processed}/{max_batches}",
            self.action,
        ));
    }

    pub(crate) fn cache_filling(
        &self,
        cached_items: usize,
        target_items: usize,
        batches_processed: usize,
        max_batches: usize,
    ) {
        self.bar.set_message(format!(
            "{} cache window {cached_items}/{target_items}; target batches {batches_processed}/{max_batches}",
            self.action,
        ));
    }

    pub(crate) fn split_skipping(&self, skipped_items: usize, target_items: usize) {
        self.bar.set_message(format!(
            "seeking split offset {skipped_items}/{target_items}",
        ));
    }

    pub(crate) fn finish(&self, items_processed: usize, batches_processed: usize) {
        self.bar.finish_with_message(format!(
            "{} {items_processed} spectra in {batches_processed} batches",
            self.action
        ));
    }
}

pub(crate) fn tokenization_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{prefix:>17} [{elapsed_precise}] {wide_bar:.cyan/blue} {pos}/{len} spectra {per_sec} ETA {eta_precise} {msg}",
    )
    .expect("valid indicatif template")
    .progress_chars("=> ")
}

fn spinner_style() -> ProgressStyle {
    ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] {msg}")
        .expect("valid indicatif template")
}

/// Progress UI mode, set via `--progress`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub enum ProgressMode {
    /// Pick automatically: hidden under the `tui` feature, bars otherwise.
    Auto,
    /// Render indicatif bars to stderr.
    Bars,
    /// Suppress all bars.
    Hidden,
}

impl ProgressMode {
    fn resolved(self) -> ResolvedProgress {
        match self {
            Self::Bars => ResolvedProgress::Bars,
            Self::Hidden => ResolvedProgress::Hidden,
            Self::Auto => {
                if cfg!(feature = "tui") {
                    ResolvedProgress::Hidden
                } else {
                    ResolvedProgress::Bars
                }
            }
        }
    }

    pub(crate) fn draw_target(self) -> ProgressDrawTarget {
        match self.resolved() {
            ResolvedProgress::Bars => ProgressDrawTarget::stderr_with_hz(10),
            ResolvedProgress::Hidden => ProgressDrawTarget::hidden(),
        }
    }

    pub fn visible(self) -> bool {
        matches!(self.resolved(), ResolvedProgress::Bars)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResolvedProgress {
    Bars,
    Hidden,
}
