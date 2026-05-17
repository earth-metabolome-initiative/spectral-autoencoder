//! `.parquet` writer using `parquet::arrow::ArrowWriter`.
//!
//! Schema (pinned at [`EmbeddingSink::open`]):
//! - `linear_cosine`: `Float32`
//! - `modified_linear_cosine`: `Float32`
//! - `log_mse`: `Float32`
//! - `latent`: `FixedSizeList<Float32, latent_width>`
//!
//! Row order is preserved from the input. The caller joins by position.

use std::{fs::File, path::Path, sync::Arc};

use arrow::array::{ArrayRef, FixedSizeListArray, Float32Array, Float32Builder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

use crate::{
    Error, Result,
    embed::record::{EmbeddingRecord, EmbeddingSchema},
    embed::sink::{EmbeddingSink, SinkOptions},
};

/// Parquet writer that flushes one `RecordBatch` per `flush_threshold` rows.
pub struct ParquetSink {
    path: std::path::PathBuf,
    file: Option<File>,
    compression: Compression,
    writer: Option<ArrowWriter<File>>,
    schema: Option<Arc<Schema>>,
    latent_width: Option<usize>,
    pending_linear_cosine: Vec<f32>,
    pending_modified_linear_cosine: Vec<f32>,
    pending_log_mse: Vec<f32>,
    pending_latent: Vec<f32>,
    flush_threshold: usize,
}

impl ParquetSink {
    /// Creates a new writer at `path` with the requested compression.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] when the file cannot be created, or
    /// [`Error::InvalidBatch`] for an unsupported compression name.
    pub fn from_path(path: &Path, options: &SinkOptions) -> Result<Self> {
        let file = File::create(path).map_err(|source| Error::io(path, source))?;
        let compression = parse_compression(options.parquet_compression.as_deref())?;
        Ok(Self {
            path: path.to_path_buf(),
            file: Some(file),
            compression,
            writer: None,
            schema: None,
            latent_width: None,
            pending_linear_cosine: Vec::new(),
            pending_modified_linear_cosine: Vec::new(),
            pending_log_mse: Vec::new(),
            pending_latent: Vec::new(),
            flush_threshold: 4096,
        })
    }

    fn flush_pending(&mut self) -> Result<()> {
        let row_count = self.pending_linear_cosine.len();
        if row_count == 0 {
            return Ok(());
        }
        let latent_width = self
            .latent_width
            .ok_or_else(|| Error::InvalidBatch("parquet sink flushed before open()".into()))?;
        let schema = self
            .schema
            .clone()
            .ok_or_else(|| Error::InvalidBatch("parquet sink schema not initialized".into()))?;
        let writer = self
            .writer
            .as_mut()
            .ok_or_else(|| Error::InvalidBatch("parquet writer not initialized".into()))?;

        let linear_array: ArrayRef = Arc::new(Float32Array::from(std::mem::take(
            &mut self.pending_linear_cosine,
        )));
        let modified_array: ArrayRef = Arc::new(Float32Array::from(std::mem::take(
            &mut self.pending_modified_linear_cosine,
        )));
        let log_mse_array: ArrayRef = Arc::new(Float32Array::from(std::mem::take(
            &mut self.pending_log_mse,
        )));

        let mut builder = Float32Builder::with_capacity(self.pending_latent.len());
        for value in self.pending_latent.drain(..) {
            builder.append_value(value);
        }
        let values = builder.finish();
        let latent_field = Arc::new(Field::new("item", DataType::Float32, false));
        let latent_width_i32 = latent_width_to_i32(latent_width)?;
        let latent_array: ArrayRef = Arc::new(
            FixedSizeListArray::try_new(latent_field, latent_width_i32, Arc::new(values), None)
                .map_err(|source| {
                    Error::InvalidBatch(format!("parquet latent column build failed: {source}"))
                })?,
        );

        let batch = RecordBatch::try_new(
            schema,
            vec![linear_array, modified_array, log_mse_array, latent_array],
        )
        .map_err(|source| {
            Error::InvalidBatch(format!("parquet record batch build failed: {source}"))
        })?;
        writer
            .write(&batch)
            .map_err(|source| Error::InvalidBatch(format!("parquet write failed: {source}")))?;
        Ok(())
    }
}

impl EmbeddingSink for ParquetSink {
    fn open(&mut self, schema: &EmbeddingSchema) -> Result<()> {
        let latent_width = schema.latent_width;
        let latent_width_i32 = latent_width_to_i32(latent_width)?;
        let latent_field = Arc::new(Field::new("item", DataType::Float32, false));
        let arrow_schema = Arc::new(Schema::new(vec![
            Field::new("linear_cosine", DataType::Float32, false),
            Field::new("modified_linear_cosine", DataType::Float32, false),
            Field::new("log_mse", DataType::Float32, false),
            Field::new(
                "latent",
                DataType::FixedSizeList(latent_field, latent_width_i32),
                false,
            ),
        ]));
        let props = WriterProperties::builder()
            .set_compression(self.compression)
            .build();
        let file = self.file.take().ok_or_else(|| {
            Error::InvalidBatch(format!(
                "parquet sink already opened or moved for {}",
                self.path.display()
            ))
        })?;
        let writer =
            ArrowWriter::try_new(file, arrow_schema.clone(), Some(props)).map_err(|source| {
                Error::InvalidBatch(format!("parquet writer init failed: {source}"))
            })?;
        self.writer = Some(writer);
        self.schema = Some(arrow_schema);
        self.latent_width = Some(latent_width);
        Ok(())
    }

    fn write(&mut self, record: &EmbeddingRecord) -> Result<()> {
        let row = &record.row;
        let expected = self.latent_width.ok_or_else(|| {
            Error::InvalidBatch("parquet sink wrote before open() was called".into())
        })?;
        if row.latent.len() != expected {
            return Err(Error::InvalidBatch(format!(
                "parquet sink expected latent width {expected}, got {}",
                row.latent.len()
            )));
        }
        self.pending_linear_cosine
            .push(row.reconstruction_linear_cosine);
        self.pending_modified_linear_cosine
            .push(row.reconstruction_modified_linear_cosine);
        self.pending_log_mse.push(row.reconstruction_log_mse);
        self.pending_latent.extend_from_slice(&row.latent);
        if self.pending_linear_cosine.len() >= self.flush_threshold {
            self.flush_pending()?;
        }
        Ok(())
    }

    fn finish(mut self: Box<Self>) -> Result<()> {
        self.flush_pending()?;
        if let Some(writer) = self.writer.take() {
            writer.close().map_err(|source| {
                Error::InvalidBatch(format!("parquet writer close failed: {source}"))
            })?;
        }
        Ok(())
    }
}

fn parse_compression(name: Option<&str>) -> Result<Compression> {
    match name {
        None | Some("snappy") => Ok(Compression::SNAPPY),
        Some("none" | "uncompressed") => Ok(Compression::UNCOMPRESSED),
        Some("zstd") => Ok(Compression::ZSTD(Default::default())),
        Some("gzip") => Ok(Compression::GZIP(Default::default())),
        Some(other) => Err(Error::InvalidBatch(format!(
            "unsupported parquet compression `{other}`; valid: snappy, none, zstd, gzip"
        ))),
    }
}

fn latent_width_to_i32(width: usize) -> Result<i32> {
    i32::try_from(width).map_err(|_| {
        Error::InvalidBatch(format!(
            "parquet sink: latent_width {width} exceeds i32::MAX"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EmbeddingRow;

    #[test]
    fn writes_two_rows_with_correct_schema() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("out.parquet");
        let mut sink =
            Box::new(ParquetSink::from_path(&path, &SinkOptions::default()).expect("open"));
        sink.open(&EmbeddingSchema { latent_width: 3 })
            .expect("open");
        for _ in 0..2 {
            sink.write(
                &EmbeddingRow {
                    latent: vec![0.1, 0.2, 0.3],
                    reconstruction_linear_cosine: 0.8,
                    reconstruction_modified_linear_cosine: 0.9,
                    reconstruction_log_mse: 0.05,
                }
                .into(),
            )
            .expect("write");
        }
        sink.finish().expect("finish");

        let file = File::open(&path).expect("reopen");
        let reader = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)
            .expect("reader")
            .build()
            .expect("build reader");
        let batches: std::result::Result<Vec<_>, _> = reader.collect();
        let batches = batches.expect("collect");
        let total_rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
        assert_eq!(total_rows, 2);
        let first = &batches[0];
        assert_eq!(first.schema().field(0).name(), "linear_cosine");
        assert_eq!(first.schema().field(3).name(), "latent");
        match first.schema().field(3).data_type() {
            DataType::FixedSizeList(_, size) => assert_eq!(size, &3),
            other => panic!("expected fixed-size list, got {other:?}"),
        }
    }
}
