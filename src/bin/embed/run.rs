//! End-to-end orchestrator: open source -> load embedder -> stream embed -> write.
//!
//! The CLI dispatcher returns an [`AnySource`] enum (file-format selection
//! is a runtime concern). After construction, everything is generic: the
//! source is wrapped into a fallible `Iterator<Item = Result<S>>`, errors
//! are short-circuited via `?` into a buffered iterator over the source's
//! `Item` type, and that buffer is handed straight to
//! [`SpectrumEmbedder::embed_stream`]. The batch size is a builder concern
//! on the embedder, not a parameter of this loop.

use std::time::Instant;

use burn::prelude::Backend;
use spectral_autoencoder::{
    AnySource, EmbeddingRecord, EmbeddingSchema, SinkOptions, SourceOptions, SpectrumEmbedder,
    SpectrumSource, sink_for_path, source_for_path,
};

use crate::cli::Args;

pub type AppResult<T> = Result<T, Box<dyn std::error::Error>>;

pub fn run<B: Backend<FloatElem = f32>>(args: Args, device: B::Device) -> AppResult<()> {
    if args.batch_size == 0 {
        return Err("--batch-size must be greater than zero".into());
    }

    let source_options = SourceOptions::default();
    let sink_options = SinkOptions {
        parquet_compression: args.parquet_compression.clone(),
    };

    let mut source = source_for_path(&args.input, &source_options)?;
    let mut embedder = SpectrumEmbedder::<B>::builder(args.checkpoint.clone(), device)
        .with_skip_errors(args.skip_errors)
        .with_batch_size(args.batch_size)
        .build()?;
    let mut sink = sink_for_path(&args.output, &sink_options)?;
    sink.open(&EmbeddingSchema {
        latent_width: embedder.latent_width(),
    })?;

    // Drain the fallible source into the embedder eagerly per batch:
    // pull `batch_size` items (propagating any source I/O error), then
    // hand the buffer to `embed_stream`, drain rows, write to sink.
    let batch_size = embedder.batch_size();
    let mut buffer: Vec<<AnySource as SpectrumSource>::Item> = Vec::with_capacity(batch_size);
    let mut written = 0_usize;
    let mut skipped = 0_usize;
    let start = Instant::now();

    loop {
        buffer.clear();
        let exhausted = fill_buffer(&mut source, &mut buffer, batch_size)?;
        if buffer.is_empty() {
            break;
        }
        let inputs = buffer.len();
        let mut rows_yielded = 0_usize;
        for result in embedder.embed_stream(buffer.drain(..)) {
            let row = result?;
            let record: EmbeddingRecord = row.into();
            sink.write(&record)?;
            rows_yielded += 1;
        }
        written += rows_yielded;
        skipped += inputs - rows_yielded;
        if exhausted {
            break;
        }
    }

    sink.finish()?;
    eprintln!(
        "embed_done processed={written} skipped={skipped} elapsed_sec={:.2}",
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

/// Pulls up to `cap` items from `source` into `buffer`. Returns `Ok(true)`
/// when the source is exhausted (fewer than `cap` items were available).
fn fill_buffer<S>(source: &mut S, buffer: &mut Vec<S::Item>, cap: usize) -> AppResult<bool>
where
    S: SpectrumSource,
{
    for _ in 0..cap {
        match source.next()? {
            Some(item) => buffer.push(item),
            None => return Ok(true),
        }
    }
    Ok(false)
}
