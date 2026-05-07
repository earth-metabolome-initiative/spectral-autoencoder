use std::{
    collections::hash_map::DefaultHasher,
    env,
    error::Error as StdError,
    fs,
    hash::{Hash, Hasher},
    io,
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, UNIX_EPOCH},
};

use burn::{
    backend::{Autodiff, Cuda},
    data::dataloader::{DataLoader, DataLoaderIterator, Progress},
    module::Module,
    record::{CompactRecorder, Record, Recorder},
    tensor::{Bool, Distribution, Int, Tensor, TensorData, backend::Backend},
};
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use mascot_rs::prelude::{
    GEMS_A10_TOP_60_ZENODO_DOI, GEMS_A10_TOP_128_ZENODO_DOI, GemsA10Builder, MGFVec,
};
use mass_spectrometry::prelude::{LinearCosine, LinearEntropy, ScalarSimilarity, Spectrum};
#[cfg(feature = "cuda")]
use spectral_autoencoder::linear_cosine_cuda::{
    LinearCosineKernelBackend, LinearCosineKernelConfig, linear_cosine_preprocessed_paired_kernel,
};
use spectral_autoencoder::{
    AutoencoderBatch, AutoencoderSample, AuxiliaryLossConfig, FlatVectorReconstructionOrdering,
    SimilarityRankingBatch, SpectralAutoencoderConfig, SpectralMetricConfig,
    SpectrumAugmentationConfig, TokenizedAutoencoderBatch, TokenizedAutoencoderSample,
    TokenizedMgfIter, VectorizedMgfIter,
};

pub type InnerBackend = Cuda<f32, i32>;
pub type TrainingBackend = Autodiff<InnerBackend>;

#[cfg(feature = "cuda")]
pub trait SimilarityTeacherBackend: Backend + LinearCosineKernelBackend {}
#[cfg(feature = "cuda")]
impl<B> SimilarityTeacherBackend for B where B: Backend + LinearCosineKernelBackend {}

#[cfg(not(feature = "cuda"))]
pub trait SimilarityTeacherBackend: Backend {}
#[cfg(not(feature = "cuda"))]
impl<B> SimilarityTeacherBackend for B where B: Backend {}

#[derive(Debug, Clone)]
pub struct RunArgs {
    pub mgf_source: String,
    pub mgf_paths: Vec<PathBuf>,
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
}

impl RunArgs {
    pub fn from_env(
        default_output_dir: &str,
        default_batch_size: usize,
    ) -> Result<Self, Box<dyn StdError>> {
        let max_peaks = usize_var("GEMS_MAX_PEAKS", 128);
        let (mgf_source, mgf_paths) = resolve_mascot_gems_a10_paths(max_peaks)?;
        let resume_epoch = optional_usize_var("GEMS_RESUME_EPOCH");
        if resume_epoch == Some(0) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "GEMS_RESUME_EPOCH must be greater than zero",
            )
            .into());
        }
        let warm_start_model = optional_path_var("GEMS_WARM_START_MODEL");
        if resume_epoch.is_some() && warm_start_model.is_some() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "set either GEMS_RESUME_EPOCH or GEMS_WARM_START_MODEL, not both",
            )
            .into());
        }
        let checkpoints = bool_var("GEMS_CHECKPOINTS", true);
        if resume_epoch.is_some() && !checkpoints {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "GEMS_RESUME_EPOCH requires GEMS_CHECKPOINTS=1",
            )
            .into());
        }

        Ok(Self {
            mgf_source,
            mgf_paths,
            max_peaks,
            output_dir: path_var("GEMS_RUN_DIR", default_output_dir),
            device: usize_var("GEMS_CUDA_DEVICE", 0),
            batch_size: usize_var("GEMS_BATCH_SIZE", default_batch_size),
            train_batches: usize_var("GEMS_TRAIN_BATCHES", 200),
            valid_batches: usize_var("GEMS_VALID_BATCHES", 20),
            epochs: usize_var("GEMS_EPOCHS", 1),
            learning_rate: f64_var("GEMS_LR", 1.0e-4),
            weight_decay: f64_var("GEMS_WEIGHT_DECAY", 1.0e-4),
            checkpoints,
            resume_epoch,
            warm_start_model,
        })
    }

    pub fn gpu_cache_percent(&self, default_percent: f64) -> f64 {
        f64_var("GEMS_GPU_CACHE_PERCENT", default_percent).clamp(0.0, 100.0)
    }

    pub fn train_gpu_cache_percent(&self, default_percent: f64) -> f64 {
        f64_var(
            "GEMS_TRAIN_GPU_CACHE_PERCENT",
            self.gpu_cache_percent(default_percent),
        )
        .clamp(0.0, 100.0)
    }

    pub fn valid_gpu_cache_percent(&self, default_percent: f64) -> f64 {
        f64_var(
            "GEMS_VALID_GPU_CACHE_PERCENT",
            self.gpu_cache_percent(default_percent),
        )
        .clamp(0.0, 100.0)
    }

    pub const fn valid_items(&self) -> usize {
        self.batch_size * self.valid_batches
    }

    pub fn train_start_item(&self) -> usize {
        usize_var("GEMS_TRAIN_OFFSET", self.valid_items())
    }

    pub fn valid_start_item(&self) -> usize {
        usize_var("GEMS_VALID_OFFSET", 0)
    }
}

fn resolve_mascot_gems_a10_paths(
    max_peaks: usize,
) -> Result<(String, Vec<PathBuf>), Box<dyn StdError>> {
    let default_directory = format!("datasets/gems-a10-top-{max_peaks}-peaks");
    let target_directory = env::var_os("GEMS_A10_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(&default_directory));
    let force_download = bool_var("GEMS_A10_FORCE_DOWNLOAD", false);
    let builder = match max_peaks {
        60 => MGFVec::<f64>::gems_a10_top_60_peaks(),
        128 => MGFVec::<f64>::gems_a10_top_128_peaks(),
        other => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("GEMS_MAX_PEAKS={other} is unsupported; use 60 or 128"),
            )
            .into());
        }
    };
    let mut builder = builder
        .target_directory(&target_directory)
        .force_download(force_download);
    if let Some(parts) = gems_a10_parts_from_env()? {
        builder = builder.parts(parts)?;
    }
    if let Some(token) = gems_a10_token() {
        builder = builder.token(token);
    }
    if ProgressMode::from_env().visible() {
        builder = builder.verbose();
    }

    ensure_mascot_gems_a10_files(&builder, force_download)?;
    let paths = builder.paths();
    let doi = match max_peaks {
        60 => GEMS_A10_TOP_60_ZENODO_DOI,
        128 => GEMS_A10_TOP_128_ZENODO_DOI,
        _ => unreachable!("unsupported GeMS peak count should already be rejected"),
    };
    Ok((
        format!(
            "mascot-rs GeMS-A10 top-{max_peaks} {doi} ({} files in {})",
            paths.len(),
            target_directory.display()
        ),
        paths,
    ))
}

fn gems_a10_parts_from_env() -> Result<Option<Vec<u8>>, Box<dyn StdError>> {
    let Some(value) = env::var("GEMS_A10_PARTS").ok() else {
        return Ok(None);
    };
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
            "GEMS_A10_PARTS did not contain any part numbers",
        )
        .into());
    }
    Ok(Some(parts))
}

fn ensure_mascot_gems_a10_files(
    builder: &GemsA10Builder<f64>,
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
    if !bool_var("GEMS_A10_DOWNLOAD", true) {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "{missing_or_forced} GeMS-A10 file(s) are missing under {}; set GEMS_A10_DOWNLOAD=1 or select fewer parts with GEMS_A10_PARTS",
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

fn gems_a10_token() -> Option<String> {
    env::var("GEMS_ZENODO_TOKEN")
        .ok()
        .or_else(|| env::var("ZENODO_TOKEN").ok())
        .filter(|token| !token.trim().is_empty())
}

pub fn print_run_header(
    model_name: &str,
    args: &RunArgs,
    parameter_count: usize,
    cache_default_percent: f64,
    auxiliary: AuxiliaryLossConfig,
    similarity_teacher: SimilarityTeacherConfig,
    flat_reconstruction_ordering: Option<FlatVectorReconstructionOrdering>,
) {
    println!("GeMS {model_name} cached-window training");
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
        "gpu cache: enabled (train {}%, valid {}%)",
        args.train_gpu_cache_percent(cache_default_percent),
        args.valid_gpu_cache_percent(cache_default_percent)
    );
    println!("model parameters: {parameter_count}");
    if let Some(ordering) = flat_reconstruction_ordering {
        println!("flat reconstruction ordering: {}", ordering.label());
    }
    println!("similarity teacher: {}", similarity_teacher.summary());
    println!(
        "auxiliary losses: reconstruction {} masked {} consistency {} intruder {} similarity-ranking {} latent-noise-std {} similarity-ranking-pairs/batch {}",
        auxiliary.reconstruction_weight,
        auxiliary.masked_peak_weight,
        auxiliary.consistency_weight,
        auxiliary.intruder_peak_weight,
        auxiliary.similarity_ranking_weight,
        auxiliary.latent_noise_std,
        if auxiliary.similarity_ranking_pairs_per_batch == 0 {
            "all".to_string()
        } else {
            auxiliary.similarity_ranking_pairs_per_batch.to_string()
        }
    );
}

#[allow(dead_code)]
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

#[allow(dead_code)]
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

pub fn auxiliary_loss_config_from_env(default: AuxiliaryLossConfig) -> AuxiliaryLossConfig {
    AuxiliaryLossConfig {
        reconstruction_weight: f64_var(
            "GEMS_AUX_RECONSTRUCTION_WEIGHT",
            default.reconstruction_weight,
        ),
        masked_peak_weight: f64_var("GEMS_AUX_MASKED_WEIGHT", default.masked_peak_weight),
        consistency_weight: f64_var("GEMS_AUX_CONSISTENCY_WEIGHT", default.consistency_weight),
        intruder_peak_weight: f64_var("GEMS_AUX_INTRUDER_WEIGHT", default.intruder_peak_weight),
        similarity_ranking_weight: f64_var(
            "GEMS_AUX_SIMILARITY_RANKING_WEIGHT",
            default.similarity_ranking_weight,
        ),
        similarity_ranking_margin: f64_var(
            "GEMS_SIMILARITY_RANKING_MARGIN",
            default.similarity_ranking_margin,
        ),
        similarity_ranking_min_gap: f64_var(
            "GEMS_SIMILARITY_RANKING_MIN_GAP",
            default.similarity_ranking_min_gap,
        ),
        latent_noise_std: f64_var("GEMS_LATENT_NOISE_STD", default.latent_noise_std),
        similarity_ranking_pairs_per_batch: usize_var(
            "GEMS_SIMILARITY_RANKING_PAIRS_PER_BATCH",
            default.similarity_ranking_pairs_per_batch,
        ),
        intruder_hidden_width: usize_var(
            "GEMS_INTRUDER_HIDDEN_WIDTH",
            default.intruder_hidden_width,
        ),
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SimilarityTeacherConfig {
    enabled: bool,
    metric: SimilarityTeacherMetric,
    execution: SimilarityTeacherExecution,
    mz_tolerance: f64,
    max_mz: f64,
    cosine_mz_power: f64,
    cosine_intensity_power: f64,
    entropy_mz_power: f64,
    entropy_intensity_power: f64,
    weighted_entropy: bool,
    candidates_per_anchor: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SimilarityTeacherExecution {
    Cpu,
    Cuda,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SimilarityTeacherMetric {
    LinearCosine,
    LinearEntropy,
}

impl SimilarityTeacherMetric {
    fn from_env() -> Result<Self, Box<dyn StdError>> {
        let value = env::var("GEMS_SIMILARITY_RANKING_METRIC")
            .unwrap_or_else(|_| "linear-cosine".to_string())
            .to_ascii_lowercase();
        match value.as_str() {
            "cosine" | "linear-cosine" | "linear_cosine" => Ok(Self::LinearCosine),
            "entropy" | "linear-entropy" | "linear_entropy" => Ok(Self::LinearEntropy),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "GEMS_SIMILARITY_RANKING_METRIC must be linear-cosine or linear-entropy",
            )
            .into()),
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::LinearCosine => "linear cosine",
            Self::LinearEntropy => "linear entropy",
        }
    }
}

impl SimilarityTeacherExecution {
    fn from_env(metric: SimilarityTeacherMetric) -> Result<Self, Box<dyn StdError>> {
        let default = if cfg!(feature = "cuda") && metric == SimilarityTeacherMetric::LinearCosine {
            "cuda"
        } else {
            "cpu"
        };
        let value = env::var("GEMS_SIMILARITY_RANKING_TEACHER")
            .unwrap_or_else(|_| default.to_string())
            .to_ascii_lowercase();
        match value.as_str() {
            "cpu" | "linear" | "linear-cpu" => Ok(Self::Cpu),
            "cuda" | "gpu" | "linear-cuda" | "cuda-linear" => Ok(Self::Cuda),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "GEMS_SIMILARITY_RANKING_TEACHER must be cpu or cuda",
            )
            .into()),
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Cuda => "CUDA",
        }
    }
}

impl SimilarityTeacherConfig {
    fn from_env(auxiliary: AuxiliaryLossConfig) -> Result<Self, Box<dyn StdError>> {
        let metrics = SpectralMetricConfig::default();
        let metric = SimilarityTeacherMetric::from_env()?;
        reject_similarity_teacher_blend_weight("GEMS_SIMILARITY_RANKING_COSINE_WEIGHT")?;
        reject_similarity_teacher_blend_weight("GEMS_SIMILARITY_RANKING_ENTROPY_WEIGHT")?;
        let mut config = Self {
            enabled: auxiliary.similarity_ranking_weight > 0.0,
            metric,
            execution: SimilarityTeacherExecution::from_env(metric)?,
            mz_tolerance: f64_var("GEMS_SIMILARITY_RANKING_MZ_TOLERANCE", metrics.mz_tolerance),
            max_mz: f64_var("GEMS_SIMILARITY_RANKING_MAX_MZ", 2_000.0),
            cosine_mz_power: f64_var(
                "GEMS_SIMILARITY_RANKING_COSINE_MZ_POWER",
                metrics.cosine_mz_power,
            ),
            cosine_intensity_power: f64_var(
                "GEMS_SIMILARITY_RANKING_COSINE_INTENSITY_POWER",
                metrics.cosine_intensity_power,
            ),
            entropy_mz_power: f64_var(
                "GEMS_SIMILARITY_RANKING_ENTROPY_MZ_POWER",
                metrics.entropy_mz_power,
            ),
            entropy_intensity_power: f64_var(
                "GEMS_SIMILARITY_RANKING_ENTROPY_INTENSITY_POWER",
                metrics.entropy_intensity_power,
            ),
            weighted_entropy: bool_var(
                "GEMS_SIMILARITY_RANKING_WEIGHTED_ENTROPY",
                metrics.weighted_entropy,
            ),
            candidates_per_anchor: usize_var("GEMS_SIMILARITY_RANKING_CANDIDATES", 4),
        };
        if config.enabled && config.candidates_per_anchor < 2 {
            config.candidates_per_anchor = 2;
        }
        config.validate()?;
        Ok(config)
    }

    const fn enabled(self) -> bool {
        self.enabled
    }

    fn summary(self) -> String {
        if !self.enabled() {
            return "disabled".to_string();
        }
        format!(
            "online {} teacher on {}, tolerance {} Da, candidates/anchor {}",
            self.metric.label(),
            self.execution.label(),
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
                "GEMS_SIMILARITY_RANKING_MAX_MZ must be finite and positive",
            )
            .into());
        }
        if self.execution == SimilarityTeacherExecution::Cuda {
            if !cfg!(feature = "cuda") {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "GEMS_SIMILARITY_RANKING_TEACHER=cuda requires a CUDA feature",
                )
                .into());
            }
            if self.metric != SimilarityTeacherMetric::LinearCosine {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "GEMS_SIMILARITY_RANKING_TEACHER=cuda currently supports only GEMS_SIMILARITY_RANKING_METRIC=linear-cosine",
                )
                .into());
            }
        }
        match self.metric {
            SimilarityTeacherMetric::LinearCosine => {
                LinearCosine::new(
                    self.cosine_mz_power,
                    self.cosine_intensity_power,
                    self.mz_tolerance,
                )
                .map_err(|error| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("invalid linear-cosine teacher config: {error}"),
                    )
                })?;
            }
            SimilarityTeacherMetric::LinearEntropy => {
                LinearEntropy::new(
                    self.entropy_mz_power,
                    self.entropy_intensity_power,
                    self.mz_tolerance,
                    self.weighted_entropy,
                )
                .map_err(|error| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("invalid linear-entropy teacher config: {error}"),
                    )
                })?;
            }
        }
        Ok(())
    }

    const fn use_cuda_teacher(self) -> bool {
        matches!(self.execution, SimilarityTeacherExecution::Cuda)
    }
}

fn reject_similarity_teacher_blend_weight(name: &str) -> Result<(), Box<dyn StdError>> {
    if env::var_os(name).is_none() {
        return Ok(());
    }
    Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        format!("{name} is no longer supported; use GEMS_SIMILARITY_RANKING_METRIC to choose one teacher metric"),
    )
    .into())
}

pub fn similarity_teacher_config_from_env(
    auxiliary: AuxiliaryLossConfig,
) -> Result<SimilarityTeacherConfig, Box<dyn StdError>> {
    SimilarityTeacherConfig::from_env(auxiliary)
}

#[derive(Debug, Clone, Copy)]
pub struct CachedTrainingLoaderConfig {
    cache_percent: f64,
    similarity_teacher: SimilarityTeacherConfig,
}

impl CachedTrainingLoaderConfig {
    pub const fn new(cache_percent: f64, similarity_teacher: SimilarityTeacherConfig) -> Self {
        Self {
            cache_percent,
            similarity_teacher,
        }
    }
}

pub fn augmentation_config_from_env(
    default: SpectrumAugmentationConfig,
) -> SpectrumAugmentationConfig {
    SpectrumAugmentationConfig {
        intruder_peak_probability: f32_var(
            "GEMS_INTRUDER_PROBABILITY",
            default.intruder_peak_probability,
        ),
        ..default
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

#[allow(dead_code)]
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
            preprocessed_cache_path: flat_preprocessed_cache_path(
                args,
                start_item,
                progress.max_batches,
            ),
            device,
            progress,
            augment,
            similarity_teacher: loader_config.similarity_teacher,
            randomize_pair_sampling: augment.is_some(),
        },
        open_records,
    ));
    loader.preload_cache();
    loader
}

#[allow(dead_code)]
pub fn cached_tokenized_loader<B, Open>(
    args: &RunArgs,
    device: B::Device,
    progress: LoaderProgress,
    start_item: usize,
    augment: Option<SpectrumAugmentationConfig>,
    loader_config: CachedTrainingLoaderConfig,
    open_records: Open,
) -> Arc<dyn DataLoader<B, TokenizedAutoencoderBatch<B>>>
where
    B: SimilarityTeacherBackend + 'static,
    Open: Fn() -> spectral_autoencoder::Result<TokenizedMgfIter> + Send + Sync + Clone + 'static,
{
    Arc::new(CachedTokenizedMgfLoader::new(
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
            preprocessed_cache_path: None,
            device,
            progress,
            augment,
            similarity_teacher: loader_config.similarity_teacher,
            randomize_pair_sampling: augment.is_some(),
        },
        open_records,
    ))
}

struct CachedLoaderOptions<B: Backend> {
    mgf_source: String,
    batch_size: usize,
    max_batches: usize,
    start_item: usize,
    cache_items: usize,
    preprocessed_cache_path: Option<PathBuf>,
    device: B::Device,
    progress: LoaderProgress,
    augment: Option<SpectrumAugmentationConfig>,
    similarity_teacher: SimilarityTeacherConfig,
    randomize_pair_sampling: bool,
}

fn cache_items(batch_size: usize, max_batches: usize, cache_percent: f64) -> usize {
    let epoch_items = batch_size.saturating_mul(max_batches);
    if epoch_items == 0 {
        return 0;
    }

    let requested = ((epoch_items as f64) * cache_percent / 100.0).ceil() as usize;
    let requested = requested.max(batch_size).min(epoch_items);
    requested.div_ceil(batch_size) * batch_size
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

fn similarity_pair_seed(epoch_index: u64, batch_index: usize, item_offset: usize) -> u64 {
    pair_sampling_seed(
        epoch_index ^ 0xa5a5_5a5a_d3c1_b2e9,
        batch_index,
        item_offset,
    )
}

#[derive(Clone)]
struct TeacherSpectraCache {
    mz: Vec<f32>,
    intensity: Vec<f32>,
    offsets: Vec<usize>,
    fixed_mz: Vec<f32>,
    fixed_intensity: Vec<f32>,
    fixed_peak_width: usize,
}

impl TeacherSpectraCache {
    fn from_target_pairs(
        config: SimilarityTeacherConfig,
        target_pairs: &[f32],
        items: usize,
        target_width: usize,
        progress: Option<&ProgressBar>,
    ) -> Self {
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

    fn view(&self, index: usize) -> TeacherSpectrumView<'_> {
        let start = self.offsets[index];
        let end = self.offsets[index + 1];
        TeacherSpectrumView {
            precursor_mz: 1.0,
            mz: &self.mz[start..end],
            intensity: &self.intensity[start..end],
        }
    }

    fn items(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }
}

struct TeacherSpectraBuilder {
    config: SimilarityTeacherConfig,
    mz: Vec<f32>,
    intensity: Vec<f32>,
    offsets: Vec<usize>,
    fixed_mz: Vec<f32>,
    fixed_intensity: Vec<f32>,
    fixed_peak_width: usize,
    expected_items: usize,
}

impl TeacherSpectraBuilder {
    fn new(config: SimilarityTeacherConfig, items: usize) -> Self {
        let mut offsets = Vec::with_capacity(items + 1);
        offsets.push(0);
        Self {
            config,
            mz: Vec::with_capacity(items.saturating_mul(48)),
            intensity: Vec::with_capacity(items.saturating_mul(48)),
            offsets,
            fixed_mz: Vec::new(),
            fixed_intensity: Vec::new(),
            fixed_peak_width: 0,
            expected_items: items,
        }
    }

    fn push_pairs(&mut self, target_pairs: &[f32]) {
        if self.fixed_peak_width == 0 {
            self.fixed_peak_width = target_pairs.len() / 2;
            self.fixed_mz
                .reserve(self.expected_items.saturating_mul(self.fixed_peak_width));
            self.fixed_intensity
                .reserve(self.expected_items.saturating_mul(self.fixed_peak_width));
        }
        let peaks = preprocess_teacher_pairs(target_pairs, self.config);
        self.mz.extend(peaks.iter().map(|peak| peak.0));
        self.intensity.extend(peaks.iter().map(|peak| peak.1));
        self.offsets.push(self.mz.len());
        for peak_index in 0..self.fixed_peak_width {
            let (mz, intensity) = peaks.get(peak_index).copied().unwrap_or((0.0, 0.0));
            self.fixed_mz.push(mz);
            self.fixed_intensity.push(intensity);
        }
    }

    fn finish(self) -> TeacherSpectraCache {
        TeacherSpectraCache {
            mz: self.mz,
            intensity: self.intensity,
            offsets: self.offsets,
            fixed_mz: self.fixed_mz,
            fixed_intensity: self.fixed_intensity,
            fixed_peak_width: self.fixed_peak_width,
        }
    }
}

#[derive(Clone)]
struct TeacherGpuCache<B: Backend> {
    mz: Tensor<B, 2>,
    intensity: Tensor<B, 2>,
    peak_width: usize,
}

impl<B: Backend> TeacherGpuCache<B> {
    fn from_cpu_window(
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
            peak_width,
        })
    }
}

#[derive(Clone, Copy)]
struct TeacherSpectrumView<'a> {
    precursor_mz: f32,
    mz: &'a [f32],
    intensity: &'a [f32],
}

impl Spectrum for TeacherSpectrumView<'_> {
    type Precision = f32;

    type SortedIntensitiesIter<'a>
        = std::iter::Copied<std::slice::Iter<'a, f32>>
    where
        Self: 'a;
    type SortedMzIter<'a>
        = std::iter::Copied<std::slice::Iter<'a, f32>>
    where
        Self: 'a;
    type SortedPeaksIter<'a>
        = std::iter::Zip<
        std::iter::Copied<std::slice::Iter<'a, f32>>,
        std::iter::Copied<std::slice::Iter<'a, f32>>,
    >
    where
        Self: 'a;

    fn len(&self) -> usize {
        self.mz.len()
    }

    fn intensities(&self) -> Self::SortedIntensitiesIter<'_> {
        self.intensity.iter().copied()
    }

    fn intensity_nth(&self, n: usize) -> Self::Precision {
        self.intensity[n]
    }

    fn mz(&self) -> Self::SortedMzIter<'_> {
        self.mz.iter().copied()
    }

    fn mz_from(&self, index: usize) -> Self::SortedMzIter<'_> {
        self.mz[index..].iter().copied()
    }

    fn mz_nth(&self, n: usize) -> Self::Precision {
        self.mz[n]
    }

    fn peaks(&self) -> Self::SortedPeaksIter<'_> {
        self.mz.iter().copied().zip(self.intensity.iter().copied())
    }

    fn peak_nth(&self, n: usize) -> (Self::Precision, Self::Precision) {
        (self.mz[n], self.intensity[n])
    }

    fn precursor_mz(&self) -> Self::Precision {
        self.precursor_mz
    }
}

enum SimilarityTeacherScorer {
    LinearCosine(LinearCosine),
    LinearEntropy(LinearEntropy),
}

impl SimilarityTeacherScorer {
    fn new(config: SimilarityTeacherConfig) -> Option<Self> {
        if !config.enabled() {
            return None;
        }
        match config.metric {
            SimilarityTeacherMetric::LinearCosine => LinearCosine::new(
                config.cosine_mz_power,
                config.cosine_intensity_power,
                config.mz_tolerance,
            )
            .ok()
            .map(Self::LinearCosine),
            SimilarityTeacherMetric::LinearEntropy => LinearEntropy::new(
                config.entropy_mz_power,
                config.entropy_intensity_power,
                config.mz_tolerance,
                config.weighted_entropy,
            )
            .ok()
            .map(Self::LinearEntropy),
        }
    }

    fn score(&self, left: TeacherSpectrumView<'_>, right: TeacherSpectrumView<'_>) -> f32 {
        match self {
            Self::LinearCosine(cosine) => cosine
                .similarity(&left, &right)
                .map(|(score, _)| score as f32)
                .unwrap_or(0.0),
            Self::LinearEntropy(entropy) => entropy
                .similarity(&left, &right)
                .map(|(score, _)| score as f32)
                .unwrap_or(0.0),
        }
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

#[derive(Clone, Copy)]
struct TeacherBatchStart {
    cpu: usize,
    gpu: usize,
}

fn teacher_similarity_ranking_batch<B: SimilarityTeacherBackend>(
    teacher: Option<&TeacherSpectraCache>,
    teacher_gpu: Option<&TeacherGpuCache<B>>,
    start: TeacherBatchStart,
    batch_items: usize,
    seed: u64,
    config: SimilarityTeacherConfig,
    device: &B::Device,
) -> SimilarityRankingBatch<B> {
    if config.use_cuda_teacher()
        && let Some(batch) = teacher_similarity_ranking_batch_cuda(
            teacher_gpu,
            start.gpu,
            batch_items,
            seed,
            config,
            device,
        )
    {
        return batch;
    }

    let Some(teacher) = teacher else {
        return SimilarityRankingBatch::zeros(batch_items, device);
    };
    let Some(scorer) = SimilarityTeacherScorer::new(config) else {
        return SimilarityRankingBatch::zeros(batch_items, device);
    };
    if batch_items < 3 {
        return SimilarityRankingBatch::zeros(batch_items, device);
    }

    let candidates = config
        .candidates_per_anchor
        .max(2)
        .min(batch_items.saturating_sub(1));
    let mut partner_a = Vec::with_capacity(batch_items);
    let mut partner_b = Vec::with_capacity(batch_items);
    let mut target_delta = Vec::with_capacity(batch_items);

    for anchor in 0..batch_items {
        let anchor_view = teacher.view(start.cpu + anchor);
        let mut state = seed
            .wrapping_add((anchor as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))
            .wrapping_add((start.cpu as u64).rotate_left(29));
        let mut best_index = anchor;
        let mut worst_index = anchor;
        let mut best_score = f32::NEG_INFINITY;
        let mut worst_score = f32::INFINITY;

        for _ in 0..candidates {
            let local_partner = sample_nonself_index(&mut state, batch_items, anchor);
            let score = scorer.score(anchor_view, teacher.view(start.cpu + local_partner));
            if score > best_score {
                best_score = score;
                best_index = local_partner;
            }
            if score < worst_score {
                worst_score = score;
                worst_index = local_partner;
            }
        }

        partner_a.push(best_index as i64);
        partner_b.push(worst_index as i64);
        target_delta.push((best_score - worst_score).max(0.0));
    }

    SimilarityRankingBatch {
        partner_a_index: Tensor::<B, 1, Int>::from_data(
            TensorData::new(partner_a, [batch_items]),
            device,
        ),
        partner_b_index: Tensor::<B, 1, Int>::from_data(
            TensorData::new(partner_b, [batch_items]),
            device,
        ),
        target_delta: Tensor::<B, 2>::from_data(
            TensorData::new(target_delta, [batch_items, 1]),
            device,
        ),
    }
}

#[cfg(feature = "cuda")]
fn teacher_similarity_ranking_batch_cuda<B: SimilarityTeacherBackend>(
    teacher: Option<&TeacherGpuCache<B>>,
    start: usize,
    batch_items: usize,
    seed: u64,
    config: SimilarityTeacherConfig,
    device: &B::Device,
) -> Option<SimilarityRankingBatch<B>> {
    let teacher = teacher?;
    if batch_items < 3 || teacher.peak_width == 0 {
        return None;
    }

    let candidates = config
        .candidates_per_anchor
        .max(2)
        .min(batch_items.saturating_sub(1));
    let pair_count = batch_items.checked_mul(candidates)?;
    let mut anchor_indices = Vec::with_capacity(pair_count);
    let mut candidate_indices = Vec::with_capacity(pair_count);
    let mut candidate_local_indices = Vec::with_capacity(pair_count);

    for anchor in 0..batch_items {
        let mut state = seed
            .wrapping_add((anchor as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))
            .wrapping_add((start as u64).rotate_left(29));
        for _ in 0..candidates {
            let local_partner = sample_nonself_index(&mut state, batch_items, anchor);
            anchor_indices.push((start + anchor) as i64);
            candidate_indices.push((start + local_partner) as i64);
            candidate_local_indices.push(local_partner);
        }
    }

    let anchor_index =
        Tensor::<B, 1, Int>::from_data(TensorData::new(anchor_indices, [pair_count]), device);
    let candidate_index =
        Tensor::<B, 1, Int>::from_data(TensorData::new(candidate_indices, [pair_count]), device);
    let scores = linear_cosine_preprocessed_paired_kernel(
        teacher.mz.clone().select(0, anchor_index.clone()),
        teacher.intensity.clone().select(0, anchor_index),
        teacher.mz.clone().select(0, candidate_index.clone()),
        teacher.intensity.clone().select(0, candidate_index),
        Tensor::<B, 1>::from_data(
            TensorData::new(
                vec![config.cosine_mz_power as f32; pair_count],
                [pair_count],
            ),
            device,
        ),
        Tensor::<B, 1>::from_data(
            TensorData::new(
                vec![config.cosine_intensity_power as f32; pair_count],
                [pair_count],
            ),
            device,
        ),
        Tensor::<B, 1>::from_data(
            TensorData::new(vec![config.mz_tolerance as f32; pair_count], [pair_count]),
            device,
        ),
        LinearCosineKernelConfig { epsilon: 1.0e-8 },
    )
    .into_data()
    .to_vec::<f32>()
    .expect("CUDA teacher scores should be f32");

    let mut partner_a = Vec::with_capacity(batch_items);
    let mut partner_b = Vec::with_capacity(batch_items);
    let mut target_delta = Vec::with_capacity(batch_items);
    for anchor in 0..batch_items {
        let score_start = anchor * candidates;
        let mut best_index = anchor;
        let mut worst_index = anchor;
        let mut best_score = f32::NEG_INFINITY;
        let mut worst_score = f32::INFINITY;
        for candidate_offset in 0..candidates {
            let score_index = score_start + candidate_offset;
            let score = scores[score_index];
            let score = if score.is_finite() { score } else { 0.0 };
            let local_partner = candidate_local_indices[score_index];
            if score > best_score {
                best_score = score;
                best_index = local_partner;
            }
            if score < worst_score {
                worst_score = score;
                worst_index = local_partner;
            }
        }
        partner_a.push(best_index as i64);
        partner_b.push(worst_index as i64);
        target_delta.push((best_score - worst_score).max(0.0));
    }

    Some(SimilarityRankingBatch {
        partner_a_index: Tensor::<B, 1, Int>::from_data(
            TensorData::new(partner_a, [batch_items]),
            device,
        ),
        partner_b_index: Tensor::<B, 1, Int>::from_data(
            TensorData::new(partner_b, [batch_items]),
            device,
        ),
        target_delta: Tensor::<B, 2>::from_data(
            TensorData::new(target_delta, [batch_items, 1]),
            device,
        ),
    })
}

#[cfg(not(feature = "cuda"))]
fn teacher_similarity_ranking_batch_cuda<B: Backend>(
    _teacher: Option<&TeacherGpuCache<B>>,
    _start: usize,
    _batch_items: usize,
    _seed: u64,
    _config: SimilarityTeacherConfig,
    _device: &B::Device,
) -> Option<SimilarityRankingBatch<B>> {
    None
}

fn sample_nonself_index(state: &mut u64, len: usize, excluded: usize) -> usize {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    let value = (*state).wrapping_mul(0x2545_f491_4f6c_dd1d);
    let mut index = (value % (len as u64 - 1)) as usize;
    if index >= excluded {
        index += 1;
    }
    index
}

const FLAT_CACHE_MAGIC: &[u8; 8] = b"SAFLC01\0";
const FLAT_CACHE_VERSION: u32 = 4;

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

#[allow(dead_code)]
type AugmentedFlatBatch<B> = (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>);

#[allow(dead_code)]
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

    (
        spectra,
        augment_conditions(conditions, config.condition_dropout_probability),
        masked_spectra_mask,
        intruder_peak_mask.float(),
    )
}

#[allow(dead_code)]
type AugmentedTokenBatch<B> = (
    Tensor<B, 3>,
    Tensor<B, 2>,
    Tensor<B, 2, Bool>,
    Tensor<B, 2>,
    Tensor<B, 2>,
    Tensor<B, 2>,
);

#[allow(dead_code)]
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

    (
        token_features,
        input_peak_mask,
        input_padding_mask,
        augment_conditions(conditions, config.condition_dropout_probability),
        masked_peak_mask.float(),
        intruder_peak_mask.float(),
    )
}

fn augment_conditions<B: Backend>(conditions: Tensor<B, 2>, probability: f32) -> Tensor<B, 2> {
    let probability = probability.clamp(0.0, 1.0);
    if probability <= 0.0 {
        return conditions;
    }

    let shape = conditions.dims();
    let device = conditions.device();
    let mask = Tensor::<B, 2>::random(shape, Distribution::Uniform(0.0, 1.0), &device)
        .lower_elem(probability);
    conditions.mask_fill(mask, 0.0)
}

#[allow(dead_code)]
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

fn probability_mask<B: Backend, const D: usize>(
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

fn signed_random<B: Backend, const D: usize>(
    shape: [usize; D],
    device: &B::Device,
    range: f32,
) -> Tensor<B, D> {
    if range <= 0.0 {
        return Tensor::<B, D>::zeros(shape, device);
    }

    (Tensor::<B, D>::random(shape, Distribution::Uniform(0.0, 1.0), device) * 2.0 - 1.0) * range
}

trait MgfRecordIter<I>: Iterator<Item = spectral_autoencoder::Result<I>> {
    fn skipped_records(&self) -> usize;
}

impl MgfRecordIter<TokenizedAutoencoderSample> for TokenizedMgfIter {
    fn skipped_records(&self) -> usize {
        self.skipped_records()
    }
}

impl MgfRecordIter<AutoencoderSample> for VectorizedMgfIter {
    fn skipped_records(&self) -> usize {
        self.skipped_records()
    }
}

fn skip_split_records<I, R>(
    records: &mut R,
    start_item: usize,
    progress: &LoaderProgress,
    preload_bar: Option<&ProgressBar>,
) where
    R: MgfRecordIter<I>,
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
                            "seeking split offset {skipped_items}/{start_item}; skipped malformed {}",
                            records.skipped_records()
                        ));
                    }
                    progress.split_skipping(skipped_items, start_item, records.skipped_records());
                }
            }
            Some(Err(error)) => {
                progress.set_skipped(records.skipped_records());
                if progress.visible() {
                    eprintln!("skipping MGF record while seeking split offset: {error}");
                }
            }
            None => break,
        }
    }
}

#[allow(dead_code)]
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
    randomize_pair_sampling: bool,
    pair_sampling_epoch: Arc<AtomicU64>,
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
            randomize_pair_sampling: self.randomize_pair_sampling,
            pair_sampling_epoch: self.pair_sampling_epoch.clone(),
        }
    }
}

#[allow(dead_code)]
impl<B, Open> CachedVectorizedMgfLoader<B, Open>
where
    B: Backend,
{
    fn new(options: CachedLoaderOptions<B>, open_records: Open) -> Self {
        Self {
            mgf_source: options.mgf_source,
            progress: options.progress,
            batch_size: options.batch_size,
            max_batches: options.max_batches,
            start_item: options.start_item,
            cache_items: options.cache_items,
            preprocessed_cache_path: options.preprocessed_cache_path,
            device: options.device,
            augment: options.augment,
            similarity_teacher: options.similarity_teacher,
            open_records,
            cpu_cache: Arc::new(Mutex::new(None)),
            initial_cache: Arc::new(Mutex::new(None)),
            full_cache: Arc::new(Mutex::new(None)),
            randomize_pair_sampling: options.randomize_pair_sampling,
            pair_sampling_epoch: Arc::new(AtomicU64::new(0)),
        }
    }

    fn is_full_cache(&self) -> bool {
        self.cache_items >= self.batch_size.saturating_mul(self.max_batches)
    }

    fn next_pair_sampling_epoch(&self) -> u64 {
        if self.randomize_pair_sampling {
            self.pair_sampling_epoch.fetch_add(1, Ordering::Relaxed) + 1
        } else {
            0
        }
    }

    fn gpu_transfer_stages(&self) -> usize {
        if self.similarity_teacher.use_cuda_teacher() {
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

        let total_items = self.batch_size.saturating_mul(self.max_batches);
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
                    if self.progress.visible() {
                        eprintln!(
                            "ignoring preprocessed flat cache {}: {error}",
                            path.display()
                        );
                    }
                }
            }
        }

        let mut records = (self.open_records)().unwrap_or_else(|error| {
            panic!(
                "failed to open cached vectorized MGF iterator for {}: {error}",
                self.mgf_source
            )
        });
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
                    && self.progress.visible()
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
        let teacher = TeacherSpectraCache::from_target_pairs(
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
                    self.progress.set_skipped(records.skipped_records());
                    if self.progress.visible() {
                        eprintln!("skipping MGF record: {error}");
                    }
                    continue;
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
                        "CPU vectorizing batches {}/{} skipped {}",
                        items / self.batch_size,
                        self.max_batches,
                        records.skipped_records()
                    ));
                }
                self.progress.cache_filling(
                    items,
                    target_items,
                    batches_processed,
                    self.max_batches,
                    records.skipped_records(),
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
                .use_cuda_teacher()
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
                teacher: cpu_cache.teacher.clone(),
                teacher_gpu,
                teacher_start: start,
                items: chunk_len,
            });
        }

        FlatGpuCache { chunks, items }
    }
}

impl<B, Open> DataLoader<B, AutoencoderBatch<B>> for CachedVectorizedMgfLoader<B, Open>
where
    B: SimilarityTeacherBackend + 'static,
    Open: Fn() -> spectral_autoencoder::Result<VectorizedMgfIter> + Send + Sync + Clone + 'static,
{
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<AutoencoderBatch<B>> + 'a> {
        self.progress
            .start_epoch(self.batch_size * self.max_batches);
        let epoch_index = self.next_pair_sampling_epoch();

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

        let mut records = (self.open_records)().unwrap_or_else(|error| {
            panic!(
                "failed to open cached vectorized MGF iterator for {}: {error}",
                self.mgf_source
            )
        });
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
        self.batch_size * self.max_batches
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

#[allow(dead_code)]
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

#[allow(dead_code)]
#[derive(Clone)]
struct FlatGpuCacheChunk<B: Backend> {
    spectra: Tensor<B, 2>,
    conditions: Tensor<B, 2>,
    teacher: Option<Arc<TeacherSpectraCache>>,
    teacher_gpu: Option<TeacherGpuCache<B>>,
    teacher_start: usize,
    items: usize,
}

#[allow(dead_code)]
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

#[allow(dead_code)]
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
        let (
            spectra,
            conditions,
            masked_spectra_mask,
            intruder_peak_mask,
            consistency_spectra,
            consistency_conditions,
        ) = match self.loader.augment {
            Some(config) => {
                let (spectra, conditions, masked_spectra_mask, intruder_peak_mask) =
                    augment_flat_batch(clean_spectra.clone(), clean_conditions.clone(), config);
                let (consistency_spectra, consistency_conditions, _, _) = augment_flat_batch(
                    clean_spectra.clone(),
                    clean_conditions.clone(),
                    config.without_intruder_peaks(),
                );
                (
                    spectra,
                    conditions,
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
                    Tensor::<B, 2>::zeros([batch_size, spectrum_width], &device),
                    Tensor::<B, 2>::zeros([batch_size, peak_count], &device),
                    clean_spectra,
                    clean_conditions,
                )
            }
        };
        let similarity_ranking = teacher_similarity_ranking_batch(
            chunk.teacher.as_deref(),
            chunk.teacher_gpu.as_ref(),
            TeacherBatchStart {
                cpu: chunk.teacher_start + chunk_offset,
                gpu: chunk_offset,
            },
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
            self.skipped_records(),
        );

        Some(AutoencoderBatch {
            spectra,
            target_spectra,
            conditions,
            consistency_spectra,
            consistency_conditions,
            masked_spectra_mask,
            intruder_peak_mask,
            similarity_ranking,
        })
    }
}

#[allow(dead_code)]
impl<B, Open> CachedVectorizedMgfIter<'_, B, Open>
where
    B: Backend,
    Open: Fn() -> spectral_autoencoder::Result<VectorizedMgfIter> + Send + Sync + Clone + 'static,
{
    fn fill_cache(&mut self) -> Option<FlatGpuCache<B>> {
        let remaining_items = self
            .loader
            .batch_size
            .saturating_mul(self.loader.max_batches)
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
        if !self.finished {
            self.finished = true;
            self.loader.progress.finish(
                self.items_processed,
                self.batches_processed,
                self.skipped_records(),
            );
        }
    }

    fn skipped_records(&self) -> usize {
        self.records
            .as_ref()
            .map_or(0, VectorizedMgfIter::skipped_records)
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
            items_total: self.loader.batch_size * self.loader.max_batches,
        }
    }
}

#[allow(dead_code)]
struct CachedTokenizedMgfLoader<B, Open>
where
    B: Backend,
{
    mgf_source: String,
    progress: LoaderProgress,
    batch_size: usize,
    max_batches: usize,
    start_item: usize,
    cache_items: usize,
    device: B::Device,
    augment: Option<SpectrumAugmentationConfig>,
    similarity_teacher: SimilarityTeacherConfig,
    open_records: Open,
    full_cache: Arc<Mutex<Option<TokenGpuCache<B>>>>,
    randomize_pair_sampling: bool,
    pair_sampling_epoch: Arc<AtomicU64>,
}

impl<B, Open> Clone for CachedTokenizedMgfLoader<B, Open>
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
            device: self.device.clone(),
            augment: self.augment,
            similarity_teacher: self.similarity_teacher,
            open_records: self.open_records.clone(),
            full_cache: self.full_cache.clone(),
            randomize_pair_sampling: self.randomize_pair_sampling,
            pair_sampling_epoch: self.pair_sampling_epoch.clone(),
        }
    }
}

#[allow(dead_code)]
impl<B, Open> CachedTokenizedMgfLoader<B, Open>
where
    B: Backend,
{
    fn new(options: CachedLoaderOptions<B>, open_records: Open) -> Self {
        Self {
            mgf_source: options.mgf_source,
            progress: options.progress,
            batch_size: options.batch_size,
            max_batches: options.max_batches,
            start_item: options.start_item,
            cache_items: options.cache_items,
            device: options.device,
            augment: options.augment,
            similarity_teacher: options.similarity_teacher,
            open_records,
            full_cache: Arc::new(Mutex::new(None)),
            randomize_pair_sampling: options.randomize_pair_sampling,
            pair_sampling_epoch: Arc::new(AtomicU64::new(0)),
        }
    }

    fn is_full_cache(&self) -> bool {
        self.cache_items >= self.batch_size.saturating_mul(self.max_batches)
    }

    fn next_pair_sampling_epoch(&self) -> u64 {
        if self.randomize_pair_sampling {
            self.pair_sampling_epoch.fetch_add(1, Ordering::Relaxed) + 1
        } else {
            0
        }
    }
}

impl<B, Open> DataLoader<B, TokenizedAutoencoderBatch<B>> for CachedTokenizedMgfLoader<B, Open>
where
    B: SimilarityTeacherBackend + 'static,
    Open: Fn() -> spectral_autoencoder::Result<TokenizedMgfIter> + Send + Sync + Clone + 'static,
{
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<TokenizedAutoencoderBatch<B>> + 'a> {
        self.progress
            .start_epoch(self.batch_size * self.max_batches);
        let epoch_index = self.next_pair_sampling_epoch();

        if self.is_full_cache()
            && let Some(cache) = self
                .full_cache
                .lock()
                .expect("full tokenized cache lock should not be poisoned")
                .clone()
        {
            return Box::new(CachedTokenizedMgfIter {
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

        let mut records = (self.open_records)().unwrap_or_else(|error| {
            panic!(
                "failed to open cached tokenized MGF iterator for {}: {error}",
                self.mgf_source
            )
        });
        skip_split_records(&mut records, self.start_item, &self.progress, None);

        Box::new(CachedTokenizedMgfIter {
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
        self.batch_size * self.max_batches
    }

    fn to_device(
        &self,
        device: &B::Device,
    ) -> Arc<dyn DataLoader<B, TokenizedAutoencoderBatch<B>>> {
        Arc::new(Self {
            device: device.clone(),
            full_cache: Arc::new(Mutex::new(None)),
            ..self.clone()
        })
    }

    fn slice(
        &self,
        start: usize,
        end: usize,
    ) -> Arc<dyn DataLoader<B, TokenizedAutoencoderBatch<B>>> {
        let start_batch = start / self.batch_size;
        let end_batch = end.div_ceil(self.batch_size);
        Arc::new(Self {
            max_batches: end_batch.saturating_sub(start_batch),
            start_item: self.start_item + start_batch * self.batch_size,
            full_cache: Arc::new(Mutex::new(None)),
            progress: self.progress.clone_for_slice(),
            ..self.clone()
        })
    }
}

#[allow(dead_code)]
#[derive(Clone)]
struct TokenGpuCache<B: Backend> {
    token_features: Tensor<B, 3>,
    target_pairs: Tensor<B, 2>,
    peak_mask: Tensor<B, 2>,
    target_peak_mask: Tensor<B, 2>,
    padding_mask: Tensor<B, 2, Bool>,
    conditions: Tensor<B, 2>,
    teacher: Option<Arc<TeacherSpectraCache>>,
    teacher_gpu: Option<TeacherGpuCache<B>>,
    items: usize,
}

#[allow(dead_code)]
struct CachedTokenizedMgfIter<'a, B, Open>
where
    B: Backend,
{
    loader: &'a CachedTokenizedMgfLoader<B, Open>,
    records: Option<TokenizedMgfIter>,
    cache: Option<TokenGpuCache<B>>,
    cache_offset: usize,
    batches_processed: usize,
    items_processed: usize,
    epoch_index: u64,
    finished: bool,
}

impl<B, Open> Iterator for CachedTokenizedMgfIter<'_, B, Open>
where
    B: SimilarityTeacherBackend,
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

        let (
            token_features,
            peak_mask,
            padding_mask,
            conditions,
            masked_peak_mask,
            intruder_peak_mask,
            consistency_token_features,
            consistency_peak_mask,
            consistency_padding_mask,
            consistency_conditions,
        ) = match self.loader.augment {
            Some(config) => {
                let clean_token_features = token_features.clone();
                let clean_peak_mask = peak_mask.clone();
                let clean_padding_mask = padding_mask.clone();
                let clean_conditions = conditions.clone();
                let (
                    token_features,
                    peak_mask,
                    padding_mask,
                    conditions,
                    masked_peak_mask,
                    intruder_peak_mask,
                ) = augment_token_batch(
                    clean_token_features.clone(),
                    clean_peak_mask.clone(),
                    clean_padding_mask.clone(),
                    clean_conditions.clone(),
                    config,
                );
                let (
                    consistency_token_features,
                    consistency_peak_mask,
                    consistency_padding_mask,
                    consistency_conditions,
                    _,
                    _,
                ) = augment_token_batch(
                    clean_token_features,
                    clean_peak_mask,
                    clean_padding_mask,
                    clean_conditions,
                    config.without_intruder_peaks(),
                );
                (
                    token_features,
                    peak_mask,
                    padding_mask,
                    conditions,
                    masked_peak_mask,
                    intruder_peak_mask,
                    consistency_token_features,
                    consistency_peak_mask,
                    consistency_padding_mask,
                    consistency_conditions,
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
                    Tensor::<B, 2>::zeros([batch_size, max_peaks], &device),
                    Tensor::<B, 2>::zeros([batch_size, max_peaks], &device),
                    token_features,
                    peak_mask,
                    padding_mask,
                    conditions,
                )
            }
        };
        let similarity_ranking = teacher_similarity_ranking_batch(
            cache.teacher.as_deref(),
            cache.teacher_gpu.as_ref(),
            TeacherBatchStart {
                cpu: self.cache_offset,
                gpu: self.cache_offset,
            },
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
            self.skipped_records(),
        );

        Some(TokenizedAutoencoderBatch {
            token_features,
            target_pairs,
            peak_mask,
            target_peak_mask,
            padding_mask,
            conditions,
            consistency_token_features,
            consistency_peak_mask,
            consistency_padding_mask,
            consistency_conditions,
            masked_peak_mask,
            intruder_peak_mask,
            similarity_ranking,
        })
    }
}

#[allow(dead_code)]
impl<B, Open> CachedTokenizedMgfIter<'_, B, Open>
where
    B: SimilarityTeacherBackend,
{
    fn fill_cache(&mut self) -> Option<TokenGpuCache<B>> {
        let records = self.records.as_mut()?;
        let remaining_items = self
            .loader
            .batch_size
            .saturating_mul(self.loader.max_batches)
            .saturating_sub(self.items_processed);
        let target_items = self.loader.cache_items.min(remaining_items);
        if target_items == 0 {
            return None;
        }

        let mut token_features = Vec::new();
        let mut target_pairs = Vec::new();
        let mut peak_mask = Vec::new();
        let mut target_peak_mask = Vec::new();
        let mut padding_mask = Vec::new();
        let mut conditions = Vec::new();
        let mut teacher_builder = self
            .loader
            .similarity_teacher
            .enabled()
            .then(|| TeacherSpectraBuilder::new(self.loader.similarity_teacher, target_items));
        let mut max_peaks = 0usize;
        let mut token_feature_width = 0usize;
        let mut target_width = 0usize;
        let mut condition_width = 0usize;
        let mut items = 0usize;

        while items < target_items {
            let sample = match records.next() {
                Some(Ok(sample)) => sample,
                Some(Err(error)) => {
                    self.loader.progress.set_skipped(records.skipped_records());
                    if self.loader.progress.visible() {
                        eprintln!("skipping MGF record: {error}");
                    }
                    continue;
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
                target_peak_mask.reserve(target_items * max_peaks);
                padding_mask.reserve(target_items * max_peaks);
                conditions.reserve(target_items * condition_width);
            }
            if let Some(builder) = &mut teacher_builder {
                builder.push_pairs(&sample.target_pairs);
            }
            token_features.extend(sample.token_features);
            target_pairs.extend(sample.target_pairs);
            peak_mask.extend_from_slice(&sample.peak_mask);
            target_peak_mask.extend(sample.peak_mask);
            padding_mask.extend(sample.padding_mask);
            conditions.extend(sample.conditions);
            items += 1;
            if items.is_multiple_of(self.loader.batch_size) || items == target_items {
                self.loader.progress.cache_filling(
                    items,
                    target_items,
                    self.batches_processed,
                    self.loader.max_batches,
                    records.skipped_records(),
                );
            }
        }

        if items == 0 {
            return None;
        }

        let teacher = teacher_builder.map(|builder| Arc::new(builder.finish()));
        let teacher_gpu = self
            .loader
            .similarity_teacher
            .use_cuda_teacher()
            .then(|| {
                teacher.as_ref().and_then(|teacher| {
                    TeacherGpuCache::from_cpu_window(teacher, 0, items, &self.loader.device)
                })
            })
            .flatten();

        let cache = TokenGpuCache {
            token_features: Tensor::<B, 3>::from_data(
                TensorData::new(token_features, [items, max_peaks, token_feature_width]),
                &self.loader.device,
            ),
            target_pairs: Tensor::<B, 2>::from_data(
                TensorData::new(target_pairs, [items, target_width]),
                &self.loader.device,
            ),
            peak_mask: Tensor::<B, 2>::from_data(
                TensorData::new(peak_mask, [items, max_peaks]),
                &self.loader.device,
            ),
            target_peak_mask: Tensor::<B, 2>::from_data(
                TensorData::new(target_peak_mask, [items, max_peaks]),
                &self.loader.device,
            ),
            padding_mask: Tensor::<B, 2, Bool>::from_bool(
                TensorData::new(padding_mask, [items, max_peaks]),
                &self.loader.device,
            ),
            conditions: Tensor::<B, 2>::from_data(
                TensorData::new(conditions, [items, condition_width]),
                &self.loader.device,
            ),
            teacher,
            teacher_gpu,
            items,
        };
        if self.loader.is_full_cache() {
            *self
                .loader
                .full_cache
                .lock()
                .expect("full tokenized cache lock should not be poisoned") = Some(cache.clone());
        }
        Some(cache)
    }

    fn finish(&mut self) {
        if !self.finished {
            self.finished = true;
            self.loader.progress.finish(
                self.items_processed,
                self.batches_processed,
                self.skipped_records(),
            );
        }
    }

    fn skipped_records(&self) -> usize {
        self.records
            .as_ref()
            .map_or(0, TokenizedMgfIter::skipped_records)
    }
}

impl<B, Open> DataLoaderIterator<TokenizedAutoencoderBatch<B>>
    for CachedTokenizedMgfIter<'_, B, Open>
where
    B: SimilarityTeacherBackend,
{
    fn progress(&self) -> Progress {
        Progress {
            items_processed: self.items_processed,
            items_total: self.loader.batch_size * self.loader.max_batches,
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
    ) -> Self {
        let progress_mode = ProgressMode::from_env();
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
    label: String,
    max_batches: usize,
    batch_size: usize,
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

    fn visible(&self) -> bool {
        self.progress_mode.visible()
    }

    fn clone_for_slice(&self) -> Self {
        Self {
            bar: self.bar.clone(),
            label: self.label.clone(),
            max_batches: self.max_batches,
            batch_size: self.batch_size,
            action: self.action.clone(),
            progress_mode: self.progress_mode,
        }
    }

    fn start_epoch(&self, total_items: usize) {
        self.bar.reset();
        self.bar.set_length(total_items as u64);
        self.bar.set_prefix(self.label.clone());
        self.bar.set_message(format!(
            "opening MGF stream; target batches 0/{} skipped 0",
            self.max_batches
        ));
    }

    fn set_skipped(&self, skipped_records: usize) {
        if skipped_records > 0 {
            self.bar.set_message(format!(
                "{}; target batches {}/{} skipped {}",
                self.action,
                self.bar.position() as usize / self.batch_size,
                self.max_batches,
                skipped_records
            ));
        }
    }

    fn batch_loaded(
        &self,
        batch_items: usize,
        batches_processed: usize,
        max_batches: usize,
        skipped_records: usize,
    ) {
        self.bar.inc(batch_items as u64);
        self.bar.set_message(format!(
            "{} batches {batches_processed}/{max_batches} skipped {skipped_records}",
            self.action
        ));
    }

    fn cache_filling(
        &self,
        cached_items: usize,
        target_items: usize,
        batches_processed: usize,
        max_batches: usize,
        skipped_records: usize,
    ) {
        self.bar.set_message(format!(
            "{} cache window {cached_items}/{target_items}; target batches {batches_processed}/{max_batches} skipped {skipped_records}",
            self.action
        ));
    }

    fn split_skipping(&self, skipped_items: usize, target_items: usize, skipped_records: usize) {
        self.bar.set_message(format!(
            "seeking split offset {skipped_items}/{target_items}; skipped malformed {skipped_records}",
        ));
    }

    fn finish(&self, items_processed: usize, batches_processed: usize, skipped_records: usize) {
        self.bar.finish_with_message(format!(
            "{} {items_processed} spectra in {batches_processed} batches; skipped {skipped_records}",
            self.action
        ));
    }
}

fn tokenization_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{prefix:>17} [{elapsed_precise}] {wide_bar:.cyan/blue} {pos}/{len} spectra {per_sec} ETA {eta_precise} {msg}",
    )
    .expect("valid indicatif template")
    .progress_chars("=> ")
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

fn spinner_style() -> ProgressStyle {
    ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] {msg}")
        .expect("valid indicatif template")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProgressMode {
    Bars,
    Hidden,
}

impl ProgressMode {
    fn from_env() -> Self {
        match env::var("GEMS_PROGRESS")
            .ok()
            .map(|value| value.to_ascii_lowercase())
            .as_deref()
        {
            Some("bars" | "bar" | "indicatif" | "on" | "1" | "true") => Self::Bars,
            Some("hidden" | "hide" | "off" | "0" | "false" | "none") => Self::Hidden,
            _ if cfg!(feature = "tui") => Self::Hidden,
            _ => Self::Bars,
        }
    }

    fn draw_target(self) -> ProgressDrawTarget {
        match self {
            Self::Bars => ProgressDrawTarget::stderr_with_hz(10),
            Self::Hidden => ProgressDrawTarget::hidden(),
        }
    }

    const fn visible(self) -> bool {
        matches!(self, Self::Bars)
    }
}

fn path_var(name: &str, default: &str) -> PathBuf {
    env::var_os(name)
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(default).to_path_buf())
}

fn optional_path_var(name: &str) -> Option<PathBuf> {
    env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn usize_var(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

#[allow(dead_code)]
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

fn optional_usize_var(name: &str) -> Option<usize> {
    env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .and_then(|value| value.parse().ok())
}

fn f64_var(name: &str, default: f64) -> f64 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn f32_var(name: &str, default: f32) -> f32 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn bool_var(name: &str, default: bool) -> bool {
    match env::var(name)
        .ok()
        .map(|value| value.to_ascii_lowercase())
        .as_deref()
    {
        Some("1" | "true" | "yes" | "y" | "on") => true,
        Some("0" | "false" | "no" | "n" | "off") => false,
        _ => default,
    }
}
