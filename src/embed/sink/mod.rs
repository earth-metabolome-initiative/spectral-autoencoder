//! Output-side trait, dispatch, and per-format writers.

mod jsonl;
#[cfg(feature = "embed-parquet")]
mod parquet;
mod stdout;
mod tsv;

pub use jsonl::JsonlSink;
#[cfg(feature = "embed-parquet")]
pub use parquet::ParquetSink;
pub use stdout::StdoutSink;
pub use tsv::TsvSink;

use std::path::Path;

use crate::{Error, Result, embed::record::EmbeddingRecord, embed::record::EmbeddingSchema};

/// Streaming sink for [`EmbeddingRecord`] rows.
pub trait EmbeddingSink {
    /// Called once before any [`write`](Self::write) with the row schema.
    /// Parquet pins its column types from this; text sinks may ignore it.
    ///
    /// # Errors
    ///
    /// Returns an error when the underlying writer cannot emit the header /
    /// schema (typically an I/O failure).
    fn open(&mut self, schema: &EmbeddingSchema) -> Result<()>;

    /// Writes one record.
    ///
    /// # Errors
    ///
    /// Returns an error when the underlying writer fails or the record's
    /// latent length does not match the schema set in [`open`](Self::open).
    fn write(&mut self, record: &EmbeddingRecord) -> Result<()>;

    /// Flushes any buffered data and finalises the underlying writer.
    ///
    /// # Errors
    ///
    /// Returns an error when the underlying flush / close fails.
    fn finish(self: Box<Self>) -> Result<()>;
}

/// Sink-side tuning options threaded in by the bin's CLI.
#[derive(Debug, Clone, Default)]
pub struct SinkOptions {
    /// Optional Parquet compression override. Accepts the standard
    /// arrow-parquet names (`"snappy"`, `"zstd"`, `"gzip"`, `"none"`).
    /// Defaults to `"snappy"` when unset.
    pub parquet_compression: Option<String>,
}

/// Dispatches to the right concrete [`EmbeddingSink`] based on the output
/// path's extension.
///
/// The bare path `-` (or `/dev/stdout`) is treated as stdout with TSV
/// formatting. Recognised extensions:
///
/// - `.tsv` -> [`TsvSink`] tab-separated
/// - `.csv` -> [`TsvSink`] comma-separated
/// - `.jsonl` -> [`JsonlSink`]
/// - `.parquet` -> [`ParquetSink`] (requires the `embed-parquet` feature)
///
/// # Errors
///
/// Returns [`Error::InvalidBatch`] for unknown extensions or when the output
/// file cannot be created.
pub fn sink_for_path(
    path: &Path,
    #[cfg_attr(not(feature = "embed-parquet"), allow(unused_variables))] options: &SinkOptions,
) -> Result<Box<dyn EmbeddingSink>> {
    if is_stdout_path(path) {
        return Ok(Box::new(StdoutSink::new()));
    }

    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .map(str::to_ascii_lowercase);
    match ext.as_deref() {
        Some("tsv") => Ok(Box::new(TsvSink::from_path(path, b'\t')?)),
        Some("csv") => Ok(Box::new(TsvSink::from_path(path, b',')?)),
        Some("jsonl") => Ok(Box::new(JsonlSink::from_path(path)?)),
        #[cfg(feature = "embed-parquet")]
        Some("parquet") => Ok(Box::new(ParquetSink::from_path(path, options)?)),
        Some(other) => Err(Error::InvalidBatch(format!(
            "unsupported output extension `.{other}` for {}",
            path.display()
        ))),
        None => Err(Error::InvalidBatch(format!(
            "output path {} has no extension; use `-` for stdout",
            path.display()
        ))),
    }
}

fn is_stdout_path(path: &Path) -> bool {
    path == Path::new("-") || path == Path::new("/dev/stdout")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stdout_dispatch_yields_stdout_sink() {
        let sink =
            sink_for_path(Path::new("-"), &SinkOptions::default()).expect("stdout sink dispatch");
        drop(sink);
    }

    #[test]
    fn unknown_extension_is_rejected() {
        let result = sink_for_path(Path::new("out.xyz"), &SinkOptions::default());
        assert!(matches!(result, Err(Error::InvalidBatch(_))));
    }
}
