//! `.jsonl` writer: one JSON object per line.

use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

use serde::Serialize;

use crate::{
    Error, Result,
    embed::record::{EmbeddingRecord, EmbeddingSchema},
    embed::sink::EmbeddingSink,
};

/// Line-delimited JSON writer.
pub struct JsonlSink {
    writer: BufWriter<File>,
}

#[derive(Serialize)]
struct JsonlRow<'a> {
    linear_cosine: f32,
    modified_linear_cosine: f32,
    log_mse: f32,
    latent: &'a [f32],
}

impl JsonlSink {
    /// Creates a new writer at `path`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] when the file cannot be created.
    pub fn from_path(path: &Path) -> Result<Self> {
        let file = File::create(path).map_err(|source| Error::io(path, source))?;
        Ok(Self {
            writer: BufWriter::new(file),
        })
    }
}

impl EmbeddingSink for JsonlSink {
    fn open(&mut self, _schema: &EmbeddingSchema) -> Result<()> {
        // JSON Lines is self-describing per record; no header needed.
        Ok(())
    }

    fn write(&mut self, record: &EmbeddingRecord) -> Result<()> {
        let row = &record.row;
        let payload = JsonlRow {
            linear_cosine: row.reconstruction_linear_cosine,
            modified_linear_cosine: row.reconstruction_modified_linear_cosine,
            log_mse: row.reconstruction_log_mse,
            latent: &row.latent,
        };
        let line = serde_json::to_string(&payload)?;
        self.writer
            .write_all(line.as_bytes())
            .map_err(|source| Error::InvalidBatch(format!("jsonl write failed: {source}")))?;
        self.writer
            .write_all(b"\n")
            .map_err(|source| Error::InvalidBatch(format!("jsonl write failed: {source}")))?;
        Ok(())
    }

    fn finish(mut self: Box<Self>) -> Result<()> {
        self.writer
            .flush()
            .map_err(|source| Error::InvalidBatch(format!("jsonl flush failed: {source}")))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EmbeddingRow;
    use std::io::Read;

    #[test]
    fn round_trips_one_row() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("out.jsonl");
        let mut sink = Box::new(JsonlSink::from_path(&path).expect("open"));
        sink.open(&EmbeddingSchema { latent_width: 2 })
            .expect("open");
        sink.write(
            &EmbeddingRow {
                latent: vec![0.1_f32, -0.5],
                reconstruction_linear_cosine: 0.9,
                reconstruction_modified_linear_cosine: 0.95,
                reconstruction_log_mse: 0.01,
            }
            .into(),
        )
        .expect("write");
        sink.finish().expect("finish");

        let mut text = String::new();
        File::open(&path)
            .expect("reopen")
            .read_to_string(&mut text)
            .expect("read");
        let value: serde_json::Value = serde_json::from_str(text.trim()).expect("parse");
        assert_eq!(value["latent"][0], 0.1_f32);
        assert_eq!(value["latent"][1], -0.5_f32);
        assert_eq!(value["linear_cosine"], 0.9_f32);
        assert!(value.get("identifier").is_none());
    }
}
