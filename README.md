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
`datasets/gems-a10-top-128-peaks` by default. The examples also keep persistent
preprocessed caches so restarts can skip repeated vectorization or tokenization
and stream fixed-size GPU windows from disk.
Set `GEMS_LOADER_PROFILE_EVERY` to emit averaged loader timings; with the Burn
TUI feature enabled they are written to `$GEMS_RUN_DIR/loader-profile.log` by
default so terminal rendering stays clean.
The similarity-ranking objective is a gap-weighted hard-label softmax
cross-entropy over sampled spectral-similarity candidates. It preserves the
teacher ordering without regressing exact similarity magnitudes; tune its
softmax scale with `GEMS_SIMILARITY_RANKING_LATENT_TEMPERATURE` and its sampled
candidate count with `GEMS_SIMILARITY_RANKING_CANDIDATES`.
The flat-vector example defaults to the tuned 32,768-spectrum batch,
8-batch GPU window, 16-worker, 8-window host-prefetch setup shown below.

Run the flat-vector model:

```bash
RUSTFLAGS="-C target-cpu=native" cargo run --release --bin train \
    --no-default-features --features std,cuda-fusion,train,tui -- flat \
    --run-dir runs/gems-a10-top128-flat-bs32768-window8-10epoch \
    --gpu-window-batches 8 \
    --loader-workers 16 \
    --host-prefetch-windows 8 \
    --batch-size 32768 \
    --valid-batches 6 \
    --train-batches 605 \
    --epochs 10
```

Run the peak-set model:

```bash
RUSTFLAGS="-C target-cpu=native" cargo run --release --bin train \
    --no-default-features --features std,cuda-fusion,train,tui -- peak-set \
    --run-dir runs/gems-a10-top128-peak-window8-bs64-fusion-5epoch \
    --gpu-window-batches 8 \
    --loader-workers 16 \
    --host-prefetch-windows 8 \
    --loader-profile-every 100 \
    --batch-size 64 \
    --valid-batches 1024 \
    --train-batches 20000 \
    --epochs 5
```

`cargo run --bin train -- --help` lists every flag. The two variants share a
common pool of CLI flags (training-loop, dataset/Zenodo, loader, augmentation,
auxiliary, similarity-ranking) plus a handful of variant-specific flags
(reconstruction ordering and hidden widths for `flat`, preprocessed-cache
selection for both).

## Checkpoints

The training bin writes Burn checkpoints by default under
`<run-dir>/checkpoint`.

Resume a checkpointed run:

```bash
RUSTFLAGS="-C target-cpu=native" cargo run --release --bin train \
    --no-default-features --features std,cuda-fusion,train,tui -- flat \
    --run-dir runs/gems-a10-top128-flat-bs32768-window8-10epoch \
    --resume-epoch 10 \
    --epochs 20
```

`--resume-epoch` restores model, optimizer, and scheduler state. For older
runs that only have a final model record, use `--warm-start-model <path>`; that
loads weights but starts a fresh optimizer.

## Inference

The `embed` bin loads a trained checkpoint and writes latent embeddings plus
three reconstruction-quality signals (linear cosine, modified linear cosine,
log MSE) per input spectrum. Input formats: `.mgf`, `.mgf.zst`, `.mgf.gz`,
plus stdin (MGF semantics). Output formats: `.tsv`, `.csv`, `.jsonl`,
`.parquet`, plus stdout (TSV).

CUDA backend:

```bash
cargo run --release --bin embed --no-default-features \
    --features cuda,embed,embed-mgf,embed-parquet -- \
    library.mgf embeddings.parquet \
    --checkpoint runs/gems-a10-top128-flat-bs32768-window8-10epoch \
    --batch-size 4096
```

CPU (`ndarray`) backend. Same flags, just drop the `cuda` feature:

```bash
cargo run --release --bin embed --no-default-features \
    --features embed,embed-mgf,embed-parquet -- \
    library.mgf embeddings.parquet \
    --checkpoint runs/gems-a10-top128-flat-bs32768-window8-10epoch \
    --batch-size 4096
```

The `--cuda-device` flag is parsed in both builds but only honoured under
the `cuda` feature; on CPU it's silently ignored.

Output rows preserve input order. Failed inputs (empty peaks, parse errors)
abort the batch with a clear error message unless `--skip-errors` is set, in
which case they are dropped from the output (the result is then a strict
subset of the input).

The Zenodo download token, when needed, is read from the `ZENODO_TOKEN`
environment variable (kept as env-var rather than a flag so tokens don't end
up in shell history).

## Features

Default features are `std`, `ndarray`, `train`, and `tui`. The CUDA training
runs with `cuda` or `cuda-fusion`; the documented commands use `cuda-fusion`,
and plain `cuda` is the fallback when Burn fusion or autotune allocates too
much temporary GPU memory.
