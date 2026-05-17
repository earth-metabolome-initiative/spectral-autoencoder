//! Command-line argument parsing for the `embed` bin.

use std::path::PathBuf;

use clap::Parser;
use spectral_autoencoder::DEFAULT_EMBED_BATCH_SIZE;

/// Inference CLI.
#[derive(Debug, Clone, Parser)]
#[command(
    name = "embed",
    version,
    about = "Embed MS/MS spectra with a trained spectral autoencoder",
    disable_help_subcommand = true
)]
pub struct Args {
    /// Input path. Use `-` (or `/dev/stdin`) for stdin (MGF semantics).
    /// Recognised extensions: `.mgf`, `.mgf.zst`, `.mgf.gz`.
    pub input: PathBuf,

    /// Output path. Use `-` (or `/dev/stdout`) for stdout (TSV).
    /// Recognised extensions: `.tsv`, `.csv`, `.jsonl`, `.parquet`.
    pub output: PathBuf,

    /// Training run directory containing `model-config.json` and `model.mpk`.
    #[arg(long)]
    pub checkpoint: PathBuf,

    /// Spectra processed per embedder forward pass. Overrides the embedder
    /// builder's default.
    #[arg(long, default_value_t = DEFAULT_EMBED_BATCH_SIZE)]
    pub batch_size: usize,

    /// Silently drop inputs that fail to vectorise / tokenise instead of
    /// aborting on the first bad row.
    #[arg(long, default_value_t = false)]
    pub skip_errors: bool,

    /// Parquet compression: `snappy` | `none` | `zstd` | `gzip`. Ignored when
    /// the output is not Parquet.
    #[arg(long)]
    pub parquet_compression: Option<String>,

    /// CUDA device ordinal (ignored on the CPU/`ndarray` backend).
    #[arg(long, default_value_t = 0)]
    pub cuda_device: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn command_definition_is_valid() {
        Args::command().debug_assert();
    }

    #[test]
    fn parses_minimal_positional_arguments() {
        let args = Args::try_parse_from([
            "embed",
            "library.mgf",
            "-",
            "--checkpoint",
            "runs/foo",
        ])
        .expect("minimal parse");
        assert_eq!(args.input, PathBuf::from("library.mgf"));
        assert_eq!(args.output, PathBuf::from("-"));
        assert_eq!(args.checkpoint, PathBuf::from("runs/foo"));
        assert_eq!(args.batch_size, DEFAULT_EMBED_BATCH_SIZE);
        assert!(!args.skip_errors);
    }
}
