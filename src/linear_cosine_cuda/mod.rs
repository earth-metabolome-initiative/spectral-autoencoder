//! CUDA kernels for preprocessed spectral similarities.
//!
//! The kernel expects fixed-width rows whose peaks are already sorted by m/z and
//! already merged with the same tolerance used by the CPU `LinearCosine`
//! teacher. Zero intensity marks padding.

#![allow(
    clippy::cast_possible_truncation,
    clippy::too_many_arguments,
    clippy::trivially_copy_pass_by_ref
)]

mod api;
#[cfg(feature = "train")]
mod autodiff;
mod cube_backend;
#[cfg(feature = "cuda-fusion")]
mod fusion;
mod kernels;
#[cfg(test)]
mod tests;

pub use api::{
    LinearCosineKernelBackend, LinearCosineKernelConfig, SIMILARITY_METRIC_LINEAR_COSINE,
    SIMILARITY_METRIC_MODIFIED_LINEAR_COSINE, SimilarityRankingKernelConfig,
    linear_cosine_preprocessed_paired_kernel, linear_cosine_similarity_ranking_kernel,
};
