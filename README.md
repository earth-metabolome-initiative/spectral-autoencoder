# spectral-autoencoder

[![CI](https://github.com/earth-metabolome-initiative/spectral-autoencoder/actions/workflows/ci.yml/badge.svg)](https://github.com/earth-metabolome-initiative/spectral-autoencoder/actions/workflows/ci.yml)
[![Codecov](https://codecov.io/gh/earth-metabolome-initiative/spectral-autoencoder/branch/main/graph/badge.svg)](https://codecov.io/gh/earth-metabolome-initiative/spectral-autoencoder)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust 2024](https://img.shields.io/badge/rust-2024-orange.svg)](https://doc.rust-lang.org/edition-guide/rust-2024/)

MS/MS spectrum autoencoders built with [Burn](https://burn.dev/).

The crate includes MGF ingestion, spectrum preprocessing, metadata conditioning,
Burn models, training metrics, and GeMS-A10 training examples.

![MS/MS spectral autoencoder overview](https://raw.githubusercontent.com/earth-metabolome-initiative/spectral-autoencoder/main/assets/autoencoder-overview.svg)

## Data

The models operate on cleaned top-N MS/MS peak lists, not binned spectra. The
library preprocessing default keeps 60 peaks per spectrum; the GeMS examples
select the peak count with `GEMS_MAX_PEAKS` and currently default to 128.
GeMS conditioning is limited to normalized precursor m/z plus a presence flag;
the decoders reconstruct that precursor condition alongside the MS2 spectrum.

Spectra are read through
[`mascot-rs`](https://github.com/LucaCappelletti94/mascot-rs). Evaluation
metrics use
[`mass-spectrometry-traits`](https://github.com/earth-metabolome-initiative/mass-spectrometry-traits).
Training uses differentiable Burn-native reconstruction and auxiliary losses.

## Models

Both model families expose a `twenty_million_run()` preset for GeMS-A10:

- `SpectralAutoencoderConfig::twenty_million_run()` is the flat-vector model:
  top-N `(m/z, intensity)` pairs, deep MLP encoder/decoder, and a
  256-dimensional latent.
- `PeakSetAutoencoderConfig::twenty_million_run()` is the peak-set model:
  top-N masked peak tokens, transformer encoder, learned-query set decoder,
  and a 256-dimensional latent.

Explicit global L1/L2 parameter penalties are disabled in these presets. The
training examples use AdamW weight decay for L2 regularization.

## Training Tasks

The GeMS training examples use these losses:

- clean-spectrum reconstruction
- precursor m/z reconstruction
- masked precursor m/z reconstruction
- masked-peak reconstruction
- synthetic intruder-peak detection
- similarity-ranking against an online spectral-similarity teacher
- decoder-side latent noise for reconstruction robustness

Input augmentations corrupt only the model input. Reconstruction targets remain
the cleaned spectra.

## GeMS-A10 Training

The examples use mascot-rs' GeMS-A10 Zenodo loaders. Data is cached under
`datasets/gems-a10-top-128-peaks` by default. The flat-vector example also keeps
a persistent preprocessed vector cache so restarts can skip repeated
vectorization and stream fixed-size GPU windows from disk.
Set `GEMS_LOADER_PROFILE_EVERY` to emit averaged loader timings; with the Burn
TUI feature enabled they are written to `$GEMS_RUN_DIR/loader-profile.log` by
default so terminal rendering stays clean.
The flat-vector example defaults to the tuned 32,768-spectrum batch,
8-batch GPU window, 16-worker, 8-window host-prefetch setup shown below.

Run the flat-vector model:

```bash
RUSTFLAGS="-C target-cpu=native" \
GEMS_RUN_DIR=runs/gems-a10-top128-flat-bs32768-window8-10epoch \
GEMS_GPU_WINDOW_BATCHES=8 \
GEMS_LOADER_WORKERS=16 \
GEMS_HOST_PREFETCH_WINDOWS=8 \
GEMS_BATCH_SIZE=32768 \
GEMS_VALID_BATCHES=6 \
GEMS_TRAIN_BATCHES=605 \
GEMS_EPOCHS=10 \
cargo run --release --example train_gems_flat --no-default-features --features std,cuda-fusion,train,tui
```

Run the peak-set model:

```bash
RUSTFLAGS="-C target-cpu=native" \
GEMS_RUN_DIR=runs/gems-a10-top128-peak-window16-bs128-5epoch \
GEMS_GPU_WINDOW_BATCHES=8 \
GEMS_LOADER_WORKERS=16 \
GEMS_HOST_PREFETCH_WINDOWS=8 \
GEMS_BATCH_SIZE=128 \
GEMS_VALID_BATCHES=512 \
GEMS_TRAIN_BATCHES=10000 \
GEMS_EPOCHS=5 \
cargo run --release --example train_gems_peak_set --no-default-features --features std,cuda-fusion,train,tui
```

## Checkpoints

GeMS examples write Burn checkpoints by default under `GEMS_RUN_DIR/checkpoint`.

Resume a checkpointed run:

```bash
RUSTFLAGS="-C target-cpu=native" \
GEMS_RUN_DIR=runs/gems-a10-top128-flat-bs32768-window8-10epoch \
GEMS_RESUME_EPOCH=10 \
GEMS_EPOCHS=20 \
cargo run --release --example train_gems_flat --no-default-features --features std,cuda-fusion,train,tui
```

`GEMS_RESUME_EPOCH` restores model, optimizer, and scheduler state. For older
runs that only have a final model record, use `GEMS_WARM_START_MODEL`; that
loads weights but starts a fresh optimizer.

## Features

Default features are `std`, `ndarray`, `train`, and `tui`. CUDA training uses
`cuda-fusion`, which enables Burn CUDA with fusion and autotune.
