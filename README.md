# spectral-autoencoder

MS/MS spectrum autoencoders built with [Burn](https://burn.dev/).

The crate includes MGF ingestion, spectrum preprocessing, metadata conditioning,
Burn models, training metrics, and GeMS-A10 training examples.

## Data

The models operate on cleaned top-N MS/MS peak lists, not binned spectra. The
library preprocessing default keeps 60 peaks per spectrum; the GeMS examples
select the peak count with `GEMS_MAX_PEAKS` and currently default to 128.
Optional metadata is encoded with explicit unknown buckets so the same models
can run when precursor or instrument fields are missing.

Spectra are read through
[`mascot-rs`](https://github.com/LucaCappelletti94/mascot-rs). Evaluation
metrics use
[`mass-spectrometry-traits`](https://github.com/earth-metabolome-initiative/mass-spectrometry-traits).
Training uses differentiable Burn-native reconstruction and auxiliary losses.

## Models

Both model families expose a `twenty_million_run()` preset for GeMS-A10:

- `SpectralAutoencoderConfig::twenty_million_run()` is the flat-vector model:
  top-N `(m/z, intensity)` pairs, deep MLP encoder/decoder, and a
  96-dimensional latent.
- `PeakSetAutoencoderConfig::twenty_million_run()` is the peak-set model:
  top-N masked peak tokens, transformer encoder, learned-query set decoder,
  and a 96-dimensional latent.

Explicit global L1/L2 parameter penalties are disabled in these presets. The
training examples use AdamW weight decay for L2 regularization.

## Training Tasks

The GeMS training examples use these losses:

- clean-spectrum reconstruction
- masked-peak reconstruction
- latent consistency between two augmented input views
- synthetic intruder-peak detection
- similarity-ranking against an online spectral-similarity teacher
- decoder-side latent noise for reconstruction robustness

Input augmentations corrupt only the model input. Reconstruction targets remain
the cleaned spectra.

## GeMS-A10 Training

The examples use mascot-rs' GeMS-A10 Zenodo loaders. Data is cached under
`datasets/gems-a10-top-128-peaks` by default. The flat-vector example also keeps
a persistent preprocessed CPU cache so restarts can skip repeated vectorization.

Run the flat-vector model:

```bash
GEMS_RUN_DIR=runs/gems-a10-top128-flat-cache100-bs1024-10epoch \
GEMS_GPU_CACHE_PERCENT=100 \
GEMS_BATCH_SIZE=1024 \
GEMS_VALID_BATCHES=192 \
GEMS_TRAIN_BATCHES=19336 \
GEMS_EPOCHS=10 \
cargo run --release --example train_gems_flat --no-default-features --features std,cuda-fusion,train,tui
```

Run the peak-set model:

```bash
GEMS_RUN_DIR=runs/gems-a10-top128-peak-cache80-bs128-5epoch \
GEMS_GPU_CACHE_PERCENT=80 \
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
GEMS_RUN_DIR=runs/gems-a10-top128-flat-cache100-bs1024-10epoch \
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
