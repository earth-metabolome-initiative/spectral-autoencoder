//! Stdout writer with TSV formatting.

use std::io::{Stdout, Write};

use crate::{
    Error, Result,
    embed::record::{EmbeddingRecord, EmbeddingSchema},
    embed::sink::EmbeddingSink,
};

/// TSV-to-stdout writer.
pub struct StdoutSink {
    out: Stdout,
    latent_width: Option<usize>,
}

impl StdoutSink {
    /// Creates a new stdout sink.
    #[must_use]
    pub fn new() -> Self {
        Self {
            out: std::io::stdout(),
            latent_width: None,
        }
    }
}

impl Default for StdoutSink {
    fn default() -> Self {
        Self::new()
    }
}

impl EmbeddingSink for StdoutSink {
    fn open(&mut self, schema: &EmbeddingSchema) -> Result<()> {
        let mut headers = String::new();
        headers.push_str("linear_cosine\tmodified_linear_cosine\tlog_mse");
        for index in 0..schema.latent_width {
            headers.push('\t');
            headers.push_str(&format!("latent_{index}"));
        }
        headers.push('\n');
        let mut handle = self.out.lock();
        handle
            .write_all(headers.as_bytes())
            .map_err(|source| Error::InvalidBatch(format!("stdout write failed: {source}")))?;
        self.latent_width = Some(schema.latent_width);
        Ok(())
    }

    fn write(&mut self, record: &EmbeddingRecord) -> Result<()> {
        let row = &record.row;
        let expected = self.latent_width.ok_or_else(|| {
            Error::InvalidBatch("stdout sink wrote before open() was called".to_string())
        })?;
        if row.latent.len() != expected {
            return Err(Error::InvalidBatch(format!(
                "stdout sink expected latent width {expected}, got {}",
                row.latent.len()
            )));
        }
        let mut line = String::new();
        line.push_str(&format!("{:.7}", row.reconstruction_linear_cosine));
        line.push('\t');
        line.push_str(&format!("{:.7}", row.reconstruction_modified_linear_cosine));
        line.push('\t');
        line.push_str(&format!("{:.7}", row.reconstruction_log_mse));
        for value in &row.latent {
            line.push('\t');
            line.push_str(&format!("{value:.7}"));
        }
        line.push('\n');
        let mut handle = self.out.lock();
        handle
            .write_all(line.as_bytes())
            .map_err(|source| Error::InvalidBatch(format!("stdout write failed: {source}")))?;
        Ok(())
    }

    fn finish(self: Box<Self>) -> Result<()> {
        let mut handle = self.out.lock();
        handle
            .flush()
            .map_err(|source| Error::InvalidBatch(format!("stdout flush failed: {source}")))?;
        Ok(())
    }
}
