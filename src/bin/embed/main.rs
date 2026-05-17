//! Inference binary: load a trained checkpoint, embed MS/MS spectra from
//! one of several supported input formats, write embeddings to one of
//! several supported output formats.
//!
//! All file-format knowledge lives in the library
//! ([`spectral_autoencoder::source_for_path`] and
//! [`spectral_autoencoder::sink_for_path`]). This bin is just clap +
//! backend selection + the source -> encoder -> sink loop.

use clap::Parser;

mod cli;
mod run;

use crate::run::AppResult;

#[cfg(feature = "cuda")]
fn main() -> AppResult<()> {
    type Backend = burn::backend::Cuda<f32, i32>;
    let args = cli::Args::parse();
    let device = burn::backend::cuda::CudaDevice::new(args.cuda_device);
    run::run::<Backend>(args, device)
}

#[cfg(not(feature = "cuda"))]
fn main() -> AppResult<()> {
    type Backend = burn::backend::NdArray<f32, i64>;
    let args = cli::Args::parse();
    let device = burn::backend::ndarray::NdArrayDevice::default();
    run::run::<Backend>(args, device)
}
