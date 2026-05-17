//! Training binary for the spectral autoencoder.
//!
//! Selects the model variant via a clap subcommand and delegates to
//! [`flat::run`] / [`peak_set::run`] for the full training pipeline.
//!
//! ```text
//! cargo run --release --bin train --no-default-features \
//!     --features std,cuda-fusion,train,tui -- flat \
//!     --run-dir runs/gems-a10-flat --epochs 10
//!
//! cargo run --release --bin train --no-default-features \
//!     --features std,cuda-fusion,train,tui -- peak-set \
//!     --run-dir runs/gems-a10-peak-set --epochs 5
//! ```

#![allow(dead_code)]

use clap::Parser;

mod cli;
mod common;
mod flat;
mod peak_set;
mod streaming;

use cli::{Cli, ModelVariant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli_args = Cli::parse();
    match cli_args.variant {
        ModelVariant::Flat(args) => flat::run(&args),
        ModelVariant::PeakSet(args) => peak_set::run(&args),
    }
}
