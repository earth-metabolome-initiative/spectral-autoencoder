//! Record types passed from the encoder to the sinks.
//!
//! [`EmbeddingRecord`] wraps the encoder-layer [`crate::EmbeddingRow`] in its
//! own type so the sink trait can grow source-side metadata fields later
//! without changing the encoder API.

use crate::EmbeddingRow;

/// One row of output written by an [`super::sink::EmbeddingSink`].
///
/// Currently identical to [`EmbeddingRow`] from the encoder layer. Carried
/// as a distinct type so the sink trait can grow source-side metadata
/// fields later without affecting the encoder API.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingRecord {
    /// The encoder's row for this spectrum.
    pub row: EmbeddingRow,
}

impl From<EmbeddingRow> for EmbeddingRecord {
    fn from(row: EmbeddingRow) -> Self {
        Self { row }
    }
}

/// Shape information passed to [`super::sink::EmbeddingSink::open`] once
/// before any record is written.
///
/// Parquet uses [`latent_width`](Self::latent_width) to pin its
/// `FixedSizeList<Float32, N>` schema up front. Streaming text sinks (TSV,
/// JSONL, stdout) ignore the schema and emit columns lazily based on the
/// first record they see.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbeddingSchema {
    /// Length of the latent embedding column across every row.
    pub latent_width: usize,
}
