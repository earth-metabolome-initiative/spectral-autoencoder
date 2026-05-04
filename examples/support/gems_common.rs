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
use mascot_rs::prelude::{GEMS_A10_TOP_60_ZENODO_DOI, GemsA10Builder, MGFVec};
use spectral_autoencoder::{
    AutoencoderBatch, AutoencoderSample, AuxiliaryLossConfig, SpectrumAugmentationConfig,
    TokenizedAutoencoderBatch, TokenizedAutoencoderSample, TokenizedMgfIter, VectorizedMgfIter,
    retention_partner_indices_with_seed,
};

pub type InnerBackend = Cuda<f32, i32>;
pub type TrainingBackend = Autodiff<InnerBackend>;

#[derive(Debug, Clone)]
pub struct RunArgs {
    pub mgf_source: String,
    pub mgf_paths: Vec<PathBuf>,
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
        let (mgf_source, mgf_paths) = resolve_mascot_gems_a10_paths()?;
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

fn resolve_mascot_gems_a10_paths() -> Result<(String, Vec<PathBuf>), Box<dyn StdError>> {
    let target_directory = path_var("GEMS_A10_DIR", "datasets/gems-a10-top-60-peaks");
    let force_download = bool_var("GEMS_A10_FORCE_DOWNLOAD", false);
    let mut builder = MGFVec::<usize, f64>::gems_a10_top_60_peaks()
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
    Ok((
        format!(
            "mascot-rs GeMS-A10 top-60 {GEMS_A10_TOP_60_ZENODO_DOI} ({} files in {})",
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
) {
    println!("GeMS {model_name} cached-window training");
    println!("mgf source: {}", args.mgf_source);
    println!("mgf files: {}", args.mgf_paths.len());
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
    println!(
        "auxiliary losses: reconstruction {} masked {} consistency {} retention {} intruder {} latent-noise-std {} retention-pairs/batch {}",
        auxiliary.reconstruction_weight,
        auxiliary.masked_peak_weight,
        auxiliary.consistency_weight,
        auxiliary.retention_order_weight,
        auxiliary.intruder_peak_weight,
        auxiliary.latent_noise_std,
        if auxiliary.retention_pairs_per_batch == 0 {
            "all".to_string()
        } else {
            auxiliary.retention_pairs_per_batch.to_string()
        }
    );
}

pub fn auxiliary_loss_config_from_env(default: AuxiliaryLossConfig) -> AuxiliaryLossConfig {
    AuxiliaryLossConfig {
        reconstruction_weight: f64_var(
            "GEMS_AUX_RECONSTRUCTION_WEIGHT",
            default.reconstruction_weight,
        ),
        masked_peak_weight: f64_var("GEMS_AUX_MASKED_WEIGHT", default.masked_peak_weight),
        consistency_weight: f64_var("GEMS_AUX_CONSISTENCY_WEIGHT", default.consistency_weight),
        retention_order_weight: f64_var(
            "GEMS_AUX_RETENTION_WEIGHT",
            default.retention_order_weight,
        ),
        intruder_peak_weight: f64_var("GEMS_AUX_INTRUDER_WEIGHT", default.intruder_peak_weight),
        latent_noise_std: f64_var("GEMS_LATENT_NOISE_STD", default.latent_noise_std),
        retention_pairs_per_batch: usize_var(
            "GEMS_RETENTION_PAIRS_PER_BATCH",
            default.retention_pairs_per_batch,
        ),
        retention_hidden_width: usize_var(
            "GEMS_RETENTION_HIDDEN_WIDTH",
            default.retention_hidden_width,
        ),
        intruder_hidden_width: usize_var(
            "GEMS_INTRUDER_HIDDEN_WIDTH",
            default.intruder_hidden_width,
        ),
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
    cache_percent: f64,
    open_records: Open,
) -> Arc<dyn DataLoader<B, AutoencoderBatch<B>>>
where
    B: Backend + 'static,
    Open: Fn() -> spectral_autoencoder::Result<VectorizedMgfIter> + Send + Sync + Clone + 'static,
{
    let loader = Arc::new(CachedVectorizedMgfLoader::new(
        CachedLoaderOptions {
            mgf_source: args.mgf_source.clone(),
            batch_size: args.batch_size,
            max_batches: progress.max_batches,
            start_item,
            cache_items: cache_items(args.batch_size, progress.max_batches, cache_percent),
            preprocessed_cache_path: flat_preprocessed_cache_path(
                args,
                start_item,
                progress.max_batches,
            ),
            device,
            progress,
            augment,
            randomize_retention_pairs: augment.is_some(),
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
    cache_percent: f64,
    open_records: Open,
) -> Arc<dyn DataLoader<B, TokenizedAutoencoderBatch<B>>>
where
    B: Backend + 'static,
    Open: Fn() -> spectral_autoencoder::Result<TokenizedMgfIter> + Send + Sync + Clone + 'static,
{
    Arc::new(CachedTokenizedMgfLoader::new(
        CachedLoaderOptions {
            mgf_source: args.mgf_source.clone(),
            batch_size: args.batch_size,
            max_batches: progress.max_batches,
            start_item,
            cache_items: cache_items(args.batch_size, progress.max_batches, cache_percent),
            preprocessed_cache_path: None,
            device,
            progress,
            augment,
            randomize_retention_pairs: augment.is_some(),
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
    randomize_retention_pairs: bool,
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

fn retention_pair_seed(epoch_index: u64, batch_index: usize, item_offset: usize) -> u64 {
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

const FLAT_CACHE_MAGIC: &[u8; 8] = b"SAFLC01\0";
const FLAT_CACHE_VERSION: u32 = 2;

fn flat_preprocessed_cache_path(
    args: &RunArgs,
    start_item: usize,
    max_batches: usize,
) -> Option<PathBuf> {
    if !bool_var("GEMS_FLAT_PREPROCESSED_CACHE", true) {
        return None;
    }

    let cache_dir = path_var(
        "GEMS_FLAT_PREPROCESSED_CACHE_DIR",
        "datasets/gems-a10-top-60-peaks/preprocessed-flat",
    );
    let fingerprint = flat_preprocessed_cache_fingerprint(args, start_item, max_batches);
    Some(cache_dir.join(format!(
        "flat-v{FLAT_CACHE_VERSION}-{fingerprint:016x}.saefc"
    )))
}

fn flat_preprocessed_cache_fingerprint(
    args: &RunArgs,
    start_item: usize,
    max_batches: usize,
) -> u64 {
    let mut hasher = DefaultHasher::new();
    FLAT_CACHE_MAGIC.hash(&mut hasher);
    FLAT_CACHE_VERSION.hash(&mut hasher);
    args.mgf_source.hash(&mut hasher);
    args.batch_size.hash(&mut hasher);
    max_batches.hash(&mut hasher);
    start_item.hash(&mut hasher);
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
    open_records: Open,
    cpu_cache: Arc<Mutex<Option<Arc<FlatCpuCache>>>>,
    initial_cache: Arc<Mutex<Option<FlatGpuCache<B>>>>,
    full_cache: Arc<Mutex<Option<FlatGpuCache<B>>>>,
    randomize_retention_pairs: bool,
    retention_epoch: Arc<AtomicU64>,
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
            open_records: self.open_records.clone(),
            cpu_cache: self.cpu_cache.clone(),
            initial_cache: self.initial_cache.clone(),
            full_cache: self.full_cache.clone(),
            randomize_retention_pairs: self.randomize_retention_pairs,
            retention_epoch: self.retention_epoch.clone(),
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
            open_records,
            cpu_cache: Arc::new(Mutex::new(None)),
            initial_cache: Arc::new(Mutex::new(None)),
            full_cache: Arc::new(Mutex::new(None)),
            randomize_retention_pairs: options.randomize_retention_pairs,
            retention_epoch: Arc::new(AtomicU64::new(0)),
        }
    }

    fn is_full_cache(&self) -> bool {
        self.cache_items >= self.batch_size.saturating_mul(self.max_batches)
    }

    fn next_retention_epoch(&self) -> u64 {
        if self.randomize_retention_pairs {
            self.retention_epoch.fetch_add(1, Ordering::Relaxed) + 1
        } else {
            0
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
                Ok(cpu_cache) => {
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
            Some(cpu_cache) => {
                bar.finish_with_message(format!(
                    "preprocessed {} spectra into CPU memory",
                    cpu_cache.items
                ));
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
        let mut retention_time = Vec::new();
        let mut retention_present = Vec::new();
        let mut filename_id = Vec::new();
        let mut spectrum_width = 0usize;
        let mut condition_width = 0usize;
        let mut items = 0usize;
        let mut reported_items = 0usize;

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
                retention_time.reserve(target_items);
                retention_present.reserve(target_items);
                filename_id.reserve(target_items);
            }
            let (rt, rt_present) = sample.metadata.retention_parts();
            spectra.extend(sample.spectrum);
            conditions.extend(sample.conditions);
            retention_time.push(rt);
            retention_present.push(rt_present);
            filename_id.push(sample.metadata.filename_part());
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
            retention_time,
            retention_present,
            filename_id,
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
            let retention_time = Tensor::<B, 2>::from_data(
                TensorData::new(
                    cpu_cache.retention_time[start..end].to_vec(),
                    [chunk_len, 1],
                ),
                &self.device,
            );
            let retention_present = Tensor::<B, 2>::from_data(
                TensorData::new(
                    cpu_cache.retention_present[start..end].to_vec(),
                    [chunk_len, 1],
                ),
                &self.device,
            );
            let filename_id = Tensor::<B, 2>::from_data(
                TensorData::new(cpu_cache.filename_id[start..end].to_vec(), [chunk_len, 1]),
                &self.device,
            );
            if let Some(bar) = transfer_bar {
                bar.inc(chunk_len as u64);
            }

            chunks.push(FlatGpuCacheChunk {
                spectra,
                conditions,
                retention_time,
                retention_present,
                filename_id,
                retention_time_values: cpu_cache.retention_time[start..end].to_vec(),
                retention_present_values: cpu_cache.retention_present[start..end].to_vec(),
                filename_id_values: cpu_cache.filename_id[start..end].to_vec(),
                items: chunk_len,
            });
        }

        FlatGpuCache { chunks, items }
    }
}

impl<B, Open> DataLoader<B, AutoencoderBatch<B>> for CachedVectorizedMgfLoader<B, Open>
where
    B: Backend + 'static,
    Open: Fn() -> spectral_autoencoder::Result<VectorizedMgfIter> + Send + Sync + Clone + 'static,
{
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<AutoencoderBatch<B>> + 'a> {
        self.progress
            .start_epoch(self.batch_size * self.max_batches);
        let epoch_index = self.next_retention_epoch();

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
    retention_time: Vec<f32>,
    retention_present: Vec<f32>,
    filename_id: Vec<f32>,
    items: usize,
    spectrum_width: usize,
    condition_width: usize,
}

impl FlatCpuCache {
    fn payload_bytes(&self) -> io::Result<u64> {
        let floats = self
            .spectra
            .len()
            .saturating_add(self.conditions.len())
            .saturating_add(self.retention_time.len())
            .saturating_add(self.retention_present.len())
            .saturating_add(self.filename_id.len());
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
            .saturating_add(items.saturating_mul(condition_width))
            .saturating_add(items * 3),
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
    let retention_time = read_f32_vec(&mut reader, items, &bar, "loading RT from vector cache")?;
    let retention_present = read_f32_vec(
        &mut reader,
        items,
        &bar,
        "loading RT masks from vector cache",
    )?;
    let filename_id = read_f32_vec(
        &mut reader,
        items,
        &bar,
        "loading filename ids from vector cache",
    )?;
    bar.finish_with_message(format!(
        "loaded preprocessed vectors from {}",
        path.display()
    ));

    Ok(FlatCpuCache {
        spectra,
        conditions,
        retention_time,
        retention_present,
        filename_id,
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
        write_f32_slice(
            &mut writer,
            &cache.retention_time,
            &bar,
            "writing RT vector cache",
        )?;
        write_f32_slice(
            &mut writer,
            &cache.retention_present,
            &bar,
            "writing RT mask vector cache",
        )?;
        write_f32_slice(
            &mut writer,
            &cache.filename_id,
            &bar,
            "writing filename id vector cache",
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
    retention_time: Tensor<B, 2>,
    retention_present: Tensor<B, 2>,
    filename_id: Tensor<B, 2>,
    retention_time_values: Vec<f32>,
    retention_present_values: Vec<f32>,
    filename_id_values: Vec<f32>,
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
    B: Backend,
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
        let retention_time = chunk
            .retention_time
            .clone()
            .narrow(0, chunk_offset, batch_items);
        let retention_present =
            chunk
                .retention_present
                .clone()
                .narrow(0, chunk_offset, batch_items);
        let filename_id = chunk
            .filename_id
            .clone()
            .narrow(0, chunk_offset, batch_items);
        let retention_partner_index = Tensor::<B, 1, Int>::from_data(
            TensorData::new(
                retention_partner_indices_with_seed(
                    &chunk.filename_id_values[chunk_offset..chunk_offset + batch_items],
                    &chunk.retention_time_values[chunk_offset..chunk_offset + batch_items],
                    &chunk.retention_present_values[chunk_offset..chunk_offset + batch_items],
                    retention_pair_seed(
                        self.epoch_index,
                        self.batches_processed,
                        self.items_processed,
                    ),
                ),
                [batch_items],
            ),
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
            retention_time,
            retention_present,
            filename_id,
            retention_partner_index,
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
    B: Backend,
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
    open_records: Open,
    full_cache: Arc<Mutex<Option<TokenGpuCache<B>>>>,
    randomize_retention_pairs: bool,
    retention_epoch: Arc<AtomicU64>,
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
            open_records: self.open_records.clone(),
            full_cache: self.full_cache.clone(),
            randomize_retention_pairs: self.randomize_retention_pairs,
            retention_epoch: self.retention_epoch.clone(),
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
            open_records,
            full_cache: Arc::new(Mutex::new(None)),
            randomize_retention_pairs: options.randomize_retention_pairs,
            retention_epoch: Arc::new(AtomicU64::new(0)),
        }
    }

    fn is_full_cache(&self) -> bool {
        self.cache_items >= self.batch_size.saturating_mul(self.max_batches)
    }

    fn next_retention_epoch(&self) -> u64 {
        if self.randomize_retention_pairs {
            self.retention_epoch.fetch_add(1, Ordering::Relaxed) + 1
        } else {
            0
        }
    }
}

impl<B, Open> DataLoader<B, TokenizedAutoencoderBatch<B>> for CachedTokenizedMgfLoader<B, Open>
where
    B: Backend + 'static,
    Open: Fn() -> spectral_autoencoder::Result<TokenizedMgfIter> + Send + Sync + Clone + 'static,
{
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<TokenizedAutoencoderBatch<B>> + 'a> {
        self.progress
            .start_epoch(self.batch_size * self.max_batches);
        let epoch_index = self.next_retention_epoch();

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
    retention_time: Tensor<B, 2>,
    retention_present: Tensor<B, 2>,
    filename_id: Tensor<B, 2>,
    retention_time_values: Vec<f32>,
    retention_present_values: Vec<f32>,
    filename_id_values: Vec<f32>,
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
    B: Backend,
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
        let retention_time = cache
            .retention_time
            .clone()
            .narrow(0, self.cache_offset, batch_items);
        let retention_present =
            cache
                .retention_present
                .clone()
                .narrow(0, self.cache_offset, batch_items);
        let filename_id = cache
            .filename_id
            .clone()
            .narrow(0, self.cache_offset, batch_items);
        let retention_partner_index = Tensor::<B, 1, Int>::from_data(
            TensorData::new(
                retention_partner_indices_with_seed(
                    &cache.filename_id_values[self.cache_offset..self.cache_offset + batch_items],
                    &cache.retention_time_values
                        [self.cache_offset..self.cache_offset + batch_items],
                    &cache.retention_present_values
                        [self.cache_offset..self.cache_offset + batch_items],
                    retention_pair_seed(
                        self.epoch_index,
                        self.batches_processed,
                        self.items_processed,
                    ),
                ),
                [batch_items],
            ),
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
            retention_time,
            retention_present,
            filename_id,
            retention_partner_index,
        })
    }
}

#[allow(dead_code)]
impl<B, Open> CachedTokenizedMgfIter<'_, B, Open>
where
    B: Backend,
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
        let mut retention_time = Vec::new();
        let mut retention_present = Vec::new();
        let mut filename_id = Vec::new();
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
                retention_time.reserve(target_items);
                retention_present.reserve(target_items);
                filename_id.reserve(target_items);
            }
            let (rt, rt_present) = sample.metadata.retention_parts();
            token_features.extend(sample.token_features);
            target_pairs.extend(sample.target_pairs);
            peak_mask.extend_from_slice(&sample.peak_mask);
            target_peak_mask.extend(sample.peak_mask);
            padding_mask.extend(sample.padding_mask);
            conditions.extend(sample.conditions);
            retention_time.push(rt);
            retention_present.push(rt_present);
            filename_id.push(sample.metadata.filename_part());
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
            retention_time: Tensor::<B, 2>::from_data(
                TensorData::new(retention_time.clone(), [items, 1]),
                &self.loader.device,
            ),
            retention_present: Tensor::<B, 2>::from_data(
                TensorData::new(retention_present.clone(), [items, 1]),
                &self.loader.device,
            ),
            filename_id: Tensor::<B, 2>::from_data(
                TensorData::new(filename_id.clone(), [items, 1]),
                &self.loader.device,
            ),
            retention_time_values: retention_time,
            retention_present_values: retention_present,
            filename_id_values: filename_id,
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
    B: Backend,
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
        "{prefix:>17} [{elapsed_precise}] {wide_bar:.cyan/blue} {pos}/{len} spectra {per_sec} {msg}",
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

fn preload_transfer_bar(prefix: String, items: usize, message: &'static str) -> ProgressBar {
    let bar = ProgressBar::with_draw_target(
        Some((items * 2) as u64),
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
        "{prefix:>17} [{elapsed_precise}] {wide_bar:.cyan/blue} {bytes}/{total_bytes} {bytes_per_sec} {msg}",
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
