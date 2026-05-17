//! Command-line interface for the `train` bin.
//!
//! All knobs that used to be read from `GEMS_*` environment variables in the
//! previous example invocation are now first-class clap flags, split into a
//! shared [`SharedArgs`] struct (training-loop, dataset, loader,
//! augmentation, auxiliary, similarity-ranking) and per-variant
//! [`FlatArgs`] / [`PeakSetArgs`] structs that carry only the knobs unique
//! to one model family (or whose defaults differ between the two).
//!
//! `ZENODO_TOKEN` remains an env var on purpose (tokens shouldn't show up in
//! shell history).

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};
use spectral_autoencoder::{
    AuxiliaryLossConfig, FlatVectorReconstructionOrdering, SpectralMetricConfig,
    SpectrumAugmentationConfig,
};

use crate::common::{
    ProgressMode, RunArgs, SimilarityTeacherConfig, SimilarityTeacherMetric,
    StreamingTrainingLoaderConfig, resolve_mascot_gems_a10,
};

/// Top-level CLI: a subcommand per model variant.
#[derive(Debug, Parser)]
#[command(
    name = "train",
    version,
    about = "Train a spectral autoencoder on GeMS-A10",
    disable_help_subcommand = true
)]
pub struct Cli {
    /// Which model variant to train.
    #[command(subcommand)]
    pub variant: ModelVariant,
}

/// Subcommand: which model family + per-variant defaults.
#[derive(Debug, Subcommand)]
pub enum ModelVariant {
    /// Train the flat-vector autoencoder.
    Flat(FlatArgs),
    /// Train the peak-set transformer autoencoder.
    #[command(name = "peak-set")]
    PeakSet(PeakSetArgs),
}

/// Shared flags applicable to both variants.
#[derive(Debug, Clone, Args)]
pub struct SharedArgs {
    // -- Run + dataset -------------------------------------------------------
    /// Output directory for checkpoints, model record, and TUI logs.
    #[arg(long)]
    pub run_dir: PathBuf,

    /// Maximum number of peaks retained per spectrum (60 or 128).
    #[arg(long, default_value_t = 128)]
    pub max_peaks: usize,

    /// CUDA device ordinal.
    #[arg(long, default_value_t = 0)]
    pub cuda_device: usize,

    /// Learning rate.
    #[arg(long, default_value_t = 1.0e-4)]
    pub learning_rate: f64,

    /// AdamW weight decay.
    #[arg(long, default_value_t = 1.0e-4)]
    pub weight_decay: f64,

    /// Save Burn-native checkpoints during training.
    #[arg(long, default_value_t = true)]
    pub checkpoints: bool,

    /// Resume from this epoch's checkpoint instead of starting from scratch.
    #[arg(long)]
    pub resume_epoch: Option<usize>,

    /// Warm-start from this model record path (mutually exclusive with `--resume-epoch`).
    #[arg(long)]
    pub warm_start_model: Option<PathBuf>,

    /// Overrides the default offset of the training split.
    #[arg(long)]
    pub train_offset: Option<usize>,

    /// Overrides the default offset of the validation split.
    #[arg(long)]
    pub valid_offset: Option<usize>,

    /// Progress UI mode.
    #[arg(long, value_enum, default_value_t = ProgressMode::Auto)]
    pub progress: ProgressMode,

    /// Optional path to write averaged loader timings to.
    #[arg(long)]
    pub loader_profile_log: Option<PathBuf>,

    /// Alias for `--loader-profile-log`.
    #[arg(long)]
    pub loader_profile_output: Option<PathBuf>,

    // -- Dataset / Zenodo ----------------------------------------------------
    /// GeMS-A10 dataset cache directory.
    #[arg(long)]
    pub dataset_dir: Option<PathBuf>,

    /// Comma-separated list of dataset parts (e.g. `0,1,2-5`).
    #[arg(long)]
    pub dataset_parts: Option<String>,

    /// Permit downloading missing dataset files.
    #[arg(long, default_value_t = true)]
    pub download: bool,

    /// Force-redownload existing dataset files.
    #[arg(long, default_value_t = false)]
    pub force_download: bool,

    // -- Streaming loader ----------------------------------------------------
    /// Number of batches per GPU window.
    #[arg(long, default_value_t = 8)]
    pub gpu_window_batches: usize,

    /// Optional GPU transfer chunk size in batches.
    #[arg(long, default_value_t = 64)]
    pub gpu_transfer_chunk_batches: usize,

    /// Host-side decode/preprocess workers.
    #[arg(long, default_value_t = 16)]
    pub loader_workers: usize,

    /// Number of host-side prefetched windows.
    #[arg(long, default_value_t = 8)]
    pub host_prefetch_windows: usize,

    /// Emit averaged loader timings every N windows (`0` disables).
    #[arg(long, default_value_t = 0)]
    pub loader_profile_every: usize,

    // -- Auxiliary losses ----------------------------------------------------
    /// Weight for the clean-reconstruction loss.
    #[arg(long)]
    pub aux_reconstruction_weight: Option<f64>,

    /// Weight for the masked-peak loss.
    #[arg(long)]
    pub aux_masked_weight: Option<f64>,

    /// Weight for the intruder-peak loss.
    #[arg(long)]
    pub aux_intruder_weight: Option<f64>,

    /// Weight for the precursor-reconstruction loss.
    #[arg(long)]
    pub aux_precursor_weight: Option<f64>,

    /// Weight for the masked-precursor loss.
    #[arg(long)]
    pub aux_masked_precursor_weight: Option<f64>,

    /// Weight for the similarity-ranking loss.
    #[arg(long)]
    pub aux_similarity_ranking_weight: Option<f64>,

    /// Decoder-input latent noise std fraction.
    #[arg(long)]
    pub latent_noise_std: Option<f64>,

    /// Per-slot intruder-head hidden width.
    #[arg(long)]
    pub intruder_hidden_width: Option<usize>,

    /// Precursor-condition input mask probability.
    #[arg(long)]
    pub precursor_mask_probability: Option<f32>,

    /// Intruder-peak insertion probability.
    #[arg(long)]
    pub intruder_probability: Option<f32>,

    // -- Similarity ranking --------------------------------------------------
    /// Similarity teacher metric.
    #[arg(long, value_enum)]
    pub similarity_ranking_metric: Option<SimilarityMetricArg>,

    /// Tolerance for similarity-ranking peak matching (Da).
    #[arg(long)]
    pub similarity_ranking_mz_tolerance: Option<f64>,

    /// Maximum m/z used by the similarity teacher.
    #[arg(long)]
    pub similarity_ranking_max_mz: Option<f64>,

    /// m/z exponent used by the similarity teacher.
    #[arg(long)]
    pub similarity_ranking_mz_power: Option<f64>,

    /// Intensity exponent used by the similarity teacher.
    #[arg(long)]
    pub similarity_ranking_intensity_power: Option<f64>,

    /// Candidates sampled per anchor.
    #[arg(long)]
    pub similarity_ranking_candidates: Option<usize>,

    /// Latent temperature in the ranking softmax CE.
    #[arg(long)]
    pub similarity_ranking_latent_temperature: Option<f64>,

    /// Teacher temperature (deprecated; kept for compatibility).
    #[arg(long)]
    pub similarity_ranking_teacher_temperature: Option<f64>,

    /// Minimum clean-spectrum gap required for a ranking pair.
    #[arg(long)]
    pub similarity_ranking_min_gap: Option<f64>,

    /// Maximum in-batch anchors used by similarity ranking (`0` = all).
    #[arg(long)]
    pub similarity_ranking_pairs_per_batch: Option<usize>,

    /// Toggle Shannon-entropy reweighting for entropy teachers.
    #[arg(long)]
    pub similarity_ranking_weighted_entropy: Option<bool>,
}

/// Flat-variant CLI arguments.
#[derive(Debug, Clone, Args)]
pub struct FlatArgs {
    #[command(flatten)]
    pub shared: SharedArgs,

    /// Training batch size.
    #[arg(long, default_value_t = 32_768)]
    pub batch_size: usize,

    /// Training batches per epoch.
    #[arg(long, default_value_t = 605)]
    pub train_batches: usize,

    /// Validation batches per epoch.
    #[arg(long, default_value_t = 6)]
    pub valid_batches: usize,

    /// Number of epochs to train.
    #[arg(long, default_value_t = 10)]
    pub epochs: usize,

    /// Flat-vector latent embedding width.
    #[arg(long)]
    pub latent_width: Option<usize>,

    /// Flat-vector encoder/decoder hidden widths (comma-separated).
    #[arg(long)]
    pub hidden_widths: Option<String>,

    /// Use the on-disk preprocessed vectorised cache.
    #[arg(long, default_value_t = true)]
    pub preprocessed_cache: bool,

    /// Custom cache directory for the preprocessed vectorised cache.
    #[arg(long)]
    pub preprocessed_cache_dir: Option<PathBuf>,

    /// Rebuild the preprocessed vectorised cache before training.
    #[arg(long, default_value_t = false)]
    pub preprocessed_cache_refresh: bool,

    /// Reconstruction-peak ordering policy.
    #[arg(long, value_enum)]
    pub reconstruction_ordering: Option<ReconstructionOrderingArg>,
}

/// Peak-set CLI arguments.
#[derive(Debug, Clone, Args)]
pub struct PeakSetArgs {
    #[command(flatten)]
    pub shared: SharedArgs,

    /// Training batch size.
    #[arg(long, default_value_t = 64)]
    pub batch_size: usize,

    /// Training batches per epoch.
    #[arg(long, default_value_t = 20_000)]
    pub train_batches: usize,

    /// Validation batches per epoch.
    #[arg(long, default_value_t = 1024)]
    pub valid_batches: usize,

    /// Number of epochs to train.
    #[arg(long, default_value_t = 5)]
    pub epochs: usize,

    /// Use the on-disk preprocessed tokenised cache.
    #[arg(long, default_value_t = true)]
    pub preprocessed_cache: bool,

    /// Custom cache directory for the preprocessed tokenised cache.
    #[arg(long)]
    pub preprocessed_cache_dir: Option<PathBuf>,

    /// Rebuild the preprocessed tokenised cache before training.
    #[arg(long, default_value_t = false)]
    pub preprocessed_cache_refresh: bool,
}

/// Clap-friendly version of [`SimilarityTeacherMetric`].
#[derive(Debug, Clone, Copy, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub enum SimilarityMetricArg {
    LinearCosine,
    ModifiedLinearCosine,
    LinearEntropy,
    ModifiedLinearEntropy,
}

impl From<SimilarityMetricArg> for SimilarityTeacherMetric {
    fn from(value: SimilarityMetricArg) -> Self {
        match value {
            SimilarityMetricArg::LinearCosine => Self::LinearCosine,
            SimilarityMetricArg::ModifiedLinearCosine => Self::ModifiedLinearCosine,
            SimilarityMetricArg::LinearEntropy => Self::LinearEntropy,
            SimilarityMetricArg::ModifiedLinearEntropy => Self::ModifiedLinearEntropy,
        }
    }
}

/// Clap-friendly version of [`FlatVectorReconstructionOrdering`].
#[derive(Debug, Clone, Copy, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub enum ReconstructionOrderingArg {
    Slot,
    IntensityDesc,
}

impl From<ReconstructionOrderingArg> for FlatVectorReconstructionOrdering {
    fn from(value: ReconstructionOrderingArg) -> Self {
        match value {
            ReconstructionOrderingArg::Slot => Self::Slot,
            ReconstructionOrderingArg::IntensityDesc => Self::IntensityDescending,
        }
    }
}

/// Materialises the per-run knobs that aren't part of a downstream config.
pub fn run_args_from_shared(
    shared: &SharedArgs,
    batch_size: usize,
    train_batches: usize,
    valid_batches: usize,
    epochs: usize,
) -> Result<RunArgs, Box<dyn std::error::Error>> {
    if shared.resume_epoch == Some(0) {
        return Err(invalid("--resume-epoch must be greater than zero"));
    }
    if shared.resume_epoch.is_some() && shared.warm_start_model.is_some() {
        return Err(invalid(
            "set either --resume-epoch or --warm-start-model, not both",
        ));
    }
    if shared.resume_epoch.is_some() && !shared.checkpoints {
        return Err(invalid("--resume-epoch requires checkpoints to be enabled"));
    }

    let gems = resolve_mascot_gems_a10(
        shared.max_peaks,
        shared.dataset_dir.as_deref(),
        shared.dataset_parts.as_deref(),
        shared.download,
        shared.force_download,
        shared.progress,
    )?;

    Ok(RunArgs {
        mgf_source: gems.source,
        mgf_paths: gems.paths,
        gems_builder: gems.builder,
        max_peaks: shared.max_peaks,
        output_dir: shared.run_dir.clone(),
        device: shared.cuda_device,
        batch_size,
        train_batches,
        valid_batches,
        epochs,
        learning_rate: shared.learning_rate,
        weight_decay: shared.weight_decay,
        checkpoints: shared.checkpoints,
        resume_epoch: shared.resume_epoch,
        warm_start_model: shared.warm_start_model.clone(),
        train_offset: shared.train_offset,
        valid_offset: shared.valid_offset,
        gpu_transfer_chunk_batches: shared.gpu_transfer_chunk_batches.max(1),
    })
}

/// Parses a comma-separated list of `usize` values (e.g. `"4096,2048,1024"`).
pub fn parse_usize_list(value: &str) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let mut out = Vec::new();
    for token in value.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        out.push(token.parse::<usize>().map_err(|error| {
            Box::<dyn std::error::Error>::from(format!(
                "could not parse `{token}` as usize: {error}"
            ))
        })?);
    }
    if out.is_empty() {
        return Err(invalid("--hidden-widths list cannot be empty"));
    }
    Ok(out)
}

/// Builds [`AuxiliaryLossConfig`] from CLI overrides on top of a variant default.
#[must_use]
pub fn auxiliary_loss_config(shared: &SharedArgs, default: AuxiliaryLossConfig) -> AuxiliaryLossConfig {
    AuxiliaryLossConfig {
        reconstruction_weight: shared
            .aux_reconstruction_weight
            .unwrap_or(default.reconstruction_weight),
        masked_peak_weight: shared
            .aux_masked_weight
            .unwrap_or(default.masked_peak_weight),
        intruder_peak_weight: shared
            .aux_intruder_weight
            .unwrap_or(default.intruder_peak_weight),
        precursor_reconstruction_weight: shared
            .aux_precursor_weight
            .unwrap_or(default.precursor_reconstruction_weight),
        masked_precursor_weight: shared
            .aux_masked_precursor_weight
            .unwrap_or(default.masked_precursor_weight),
        similarity_ranking_weight: shared
            .aux_similarity_ranking_weight
            .unwrap_or(default.similarity_ranking_weight),
        similarity_ranking_latent_temperature: shared
            .similarity_ranking_latent_temperature
            .unwrap_or(default.similarity_ranking_latent_temperature),
        similarity_ranking_teacher_temperature: shared
            .similarity_ranking_teacher_temperature
            .unwrap_or(default.similarity_ranking_teacher_temperature),
        similarity_ranking_min_gap: shared
            .similarity_ranking_min_gap
            .unwrap_or(default.similarity_ranking_min_gap),
        latent_noise_std: shared.latent_noise_std.unwrap_or(default.latent_noise_std),
        similarity_ranking_pairs_per_batch: shared
            .similarity_ranking_pairs_per_batch
            .unwrap_or(default.similarity_ranking_pairs_per_batch),
        intruder_hidden_width: shared
            .intruder_hidden_width
            .unwrap_or(default.intruder_hidden_width),
    }
}

/// Builds [`SimilarityTeacherConfig`] from CLI overrides + metric defaults.
pub fn similarity_teacher_config(
    shared: &SharedArgs,
    auxiliary: AuxiliaryLossConfig,
) -> Result<SimilarityTeacherConfig, Box<dyn std::error::Error>> {
    let metrics = SpectralMetricConfig::default();
    let metric: SimilarityTeacherMetric = shared
        .similarity_ranking_metric
        .map(Into::into)
        .unwrap_or(SimilarityTeacherMetric::LinearCosine);
    let (default_mz_power, default_intensity_power) = if metric.is_entropy() {
        (metrics.entropy_mz_power, metrics.entropy_intensity_power)
    } else {
        (metrics.cosine_mz_power, metrics.cosine_intensity_power)
    };
    SimilarityTeacherConfig::build(
        auxiliary,
        metric,
        shared
            .similarity_ranking_mz_tolerance
            .unwrap_or(metrics.mz_tolerance),
        shared.similarity_ranking_max_mz.unwrap_or(2_000.0),
        shared
            .similarity_ranking_mz_power
            .unwrap_or(default_mz_power),
        shared
            .similarity_ranking_intensity_power
            .unwrap_or(default_intensity_power),
        shared.similarity_ranking_candidates.unwrap_or(4),
        shared
            .similarity_ranking_weighted_entropy
            .unwrap_or(metrics.weighted_entropy),
    )
}

/// Builds [`StreamingTrainingLoaderConfig`] from CLI flags.
pub fn streaming_loader_config(
    shared: &SharedArgs,
    similarity_teacher: SimilarityTeacherConfig,
) -> Result<StreamingTrainingLoaderConfig, Box<dyn std::error::Error>> {
    if shared.gpu_window_batches == 0 {
        return Err(invalid("--gpu-window-batches must be positive"));
    }
    if shared.loader_workers == 0 {
        return Err(invalid("--loader-workers must be positive"));
    }
    if shared.host_prefetch_windows == 0 {
        return Err(invalid("--host-prefetch-windows must be positive"));
    }
    let sink = crate::streaming::LoaderProfileSink::resolve(
        shared.loader_profile_every,
        loader_profile_path(shared).as_deref(),
        &shared.run_dir,
    );
    Ok(StreamingTrainingLoaderConfig::new(
        shared.gpu_window_batches,
        shared.loader_workers,
        shared.host_prefetch_windows,
        shared.loader_profile_every,
        similarity_teacher,
        sink,
    ))
}

/// Applies CLI augmentation overrides on top of a base config.
#[must_use]
pub fn augmentation_config(
    shared: &SharedArgs,
    default: SpectrumAugmentationConfig,
) -> SpectrumAugmentationConfig {
    SpectrumAugmentationConfig {
        precursor_mask_probability: shared
            .precursor_mask_probability
            .unwrap_or(default.precursor_mask_probability),
        intruder_peak_probability: shared
            .intruder_probability
            .unwrap_or(default.intruder_peak_probability),
        ..default
    }
}

/// Returns the configured loader-profile log path, prefer
/// `--loader-profile-log`, fall back to `--loader-profile-output`.
#[must_use]
pub fn loader_profile_path(shared: &SharedArgs) -> Option<PathBuf> {
    shared
        .loader_profile_log
        .clone()
        .or_else(|| shared.loader_profile_output.clone())
}

fn invalid(message: &str) -> Box<dyn std::error::Error> {
    Box::new(std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        message.to_string(),
    ))
}
