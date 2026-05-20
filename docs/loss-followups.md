# Loss follow-ups

Open ideas for the spectral-autoencoder training losses, queued for later
discussion. Each entry notes what the issue is, the file:line it lives at,
and a rough sketch of what an implementation might look like.

## Masked path

### 1. Sparse masked-cosine signal

The masked-peak cosine fires only on slots that were both real-peak in the
target and zeroed out at encoder input. With a typical 15% input dropout
and 30-100 real peaks per spectrum, ~5-15 slots per row contribute. The
gradient is meaningful but noisy compared to the clean-path cosine.

`src/model/flat_vector.rs:798-810` (masked-peak loss computation),
`src/augmentation.rs` (drop-rate config).

Options worth trying:

- Anneal the mask rate over training: start high (≥ 50%) so the encoder
  must do real reconstruction work, decay toward a lower rate (~10-15%)
  late. Mirrors how BERT-style masked-LM schedules sometimes look.
- Increase the per-batch effective signal by sampling multiple input
  corruptions per spectrum within the same forward pass. Cheap compute,
  more gradient per row.
- Bias the drop toward high-intensity peaks (harder reconstruction
  target) versus uniformly random; biases learning toward semantically
  important peaks.

### 2. Mask token vs zeroing collides with "real empty"

A masked input slot is set to zero, indistinguishable from a genuinely
padded empty slot. The encoder can't tell "this slot was intentionally
hidden" from "this slot was never a peak". This is a real information
loss in the input representation.

`src/augmentation.rs` (mask application), `src/model/flat_vector.rs`
(encoder input).

Implementation paths:

- Add a parallel per-slot "is-masked" channel to the encoder input,
  doubling the input width (or appending one channel per pair). The
  encoder learns to treat zeros-with-mask-flag-on differently from
  zeros-with-mask-flag-off.
- Introduce a single learnable "mask token" pair (m_mask, I_mask)
  inserted into masked slots instead of zeros. The model learns the
  token's parameters during training.

Both require config + encoder input width changes; not invasive but
architecture-level.

### 3. No m/z or intensity MAE diagnostic for masked peaks

The clean reconstruction reports `precursor_mae_da`; analogous
diagnostics for the masked-peak path would help observability without
changing the loss.

`src/training.rs:64-77` (AutoencoderDiagnostics struct),
`src/model/flat_vector.rs::forward_reconstruction` (computation site).

Sketch: compute `mean_{p ∈ M*} |m̂_p − m_p| · precursor_mz_scale` and
`mean_{p ∈ M*} |Î_p − I_p|`, register two new diagnostic fields and TUI
metrics. No new loss; pure observability.

### 4. Huber on the masked-precursor m/z residual

`masked_precursor_reconstruction_output` at `src/model/auxiliary.rs:542`
uses plain squared error on the (normalized) m/z residual. On occasional
far-off predictions, the quadratic dominates and pulls the gradient
toward extreme adjustments. Huber would cap the linear-region gradient
and keep training more stable when the precursor head is still learning
its scale.

Sketch: introduce a `huber_beta: f64` field on `AuxiliaryLossConfig`,
replace `mz_delta.powf_scalar(2.0)` with the standard Huber form
`if |x| ≤ β { x²/2 } else { β·(|x| − β/2) }`. Single-call helper or
inline.

## General

### 5. σ schedule for the precursor head

The precursor head's m/z prediction has no Gaussian gate equivalent —
it's a direct MSE. If we want progressive precision there too, we'd
add a separate annealing schedule (or share the spectrum-side σ
schedule's progress fraction without σ itself), e.g. a Huber β that
shrinks over training.

Out of scope but logical extension if (4) lands.
