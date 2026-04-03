//! VectorIndex — the in-memory KNN engine, mutation interface, MoE router, and HNSW index.

pub mod core;
pub mod hnsw;
pub mod mutate;
pub mod router;

pub use core::*;
pub use hnsw::HnswLayer;
pub use router::RouterIndex;
