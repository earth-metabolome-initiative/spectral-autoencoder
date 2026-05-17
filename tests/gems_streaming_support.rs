//! Standalone shim so the unit tests inside `src/bin/train/streaming.rs`
//! can run via `cargo test` without requiring the train bin's full
//! feature set (cuda + train + tui). The bin's other modules are not
//! pulled in, so some streaming items show up as dead code here. That is
//! expected. Only the in-module `#[cfg(test)]` paths actually run.

#![allow(dead_code)]

#[path = "../src/bin/train/streaming.rs"]
mod streaming;
