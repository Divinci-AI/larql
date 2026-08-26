//! `nvfp4-gptq-v1` — fixed-grid GPTQ, the calibration-aware
//! [`super::nvfp4_pack::EncoderRecipe`] frozen in `ENCODER-R4.md`.
//!
//! Only the E2M1 code nibbles may differ from `nvfp4-nearest-v1`. Every
//! scale byte is byte-identical **by construction**: the tensor scale
//! and every per-(row, group) E4M3 scale are computed once from the
//! original weights, before any error compensation, by calling
//! `nvfp4-nearest-v1`'s own scale-derivation code rather than a second
//! implementation of the same formula. Compensated weights read against
//! those frozen scales; nothing about GPTQ's elimination can trigger a
//! rescale.
//!
//! ```text
//! W0
//!   │
//!   ├─ tensor_scale, per-group E4M3 scales  (nvfp4-nearest-v1's own code, frozen)
//!   │
//!   ├─ raw calibration Hessian H = XᵀX
//!   │     │
//!   │     ├─ dead-coordinate partition          (hessian.rs)
//!   │     ├─ Cholesky(H_λ) → H⁻¹ → Cholesky(H⁻¹) (sequential.rs)
//!   │     └─ per-row sequential column update    (sequential.rs)
//!   │
//!   └─ merge: dead columns keep nearest's code, alive columns take
//!      GPTQ's                                    (pack.rs)
//!        │
//!        ▼
//!      Q_N(W)
//! ```
//!
//! This is milestone one of `ENCODER-R4.md`'s Sequence step 8: one
//! tensor, given its calibration Hessian directly, quantised
//! deterministically. It does not yet capture calibration activations
//! (the sequential candidate-path forward-pass harness), does not wire
//! into the VINDEX3 REPRESENT dispatch, and does not use the
//! LAPACK-accelerated Cholesky R4.2 benchmarked — all three are later,
//! separately-scoped expansions ("one tensor → one layer → sequential
//! full model"), not implied by this module existing.

pub mod hessian;
pub mod pack;
pub mod sequential;

pub use hessian::SiteHessian;
pub use pack::{quantize_nvfp4_gptq, GptqPackOutcome};
pub use sequential::{EliminationPlan, RowResult};
