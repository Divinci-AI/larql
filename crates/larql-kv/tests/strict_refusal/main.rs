//! The strict guarantee, end to end: a refused expert cannot become a token.
//!
//! The `FfnBackend` ring gave the refusal a typed channel. The dispatch ring
//! widened the six `kv_*_via_dispatch` helpers so it survives the per-layer
//! loop, and `StandardEngine` terminates it as `EngineError::Execution`. These
//! tests drive the whole chain from the engine's public surface, because that
//! is the only altitude at which the guarantee is observable — every
//! intermediate ring can be correct while the composition still erases.
//!
//! ## What is pinned
//!
//! ```text
//! strict       + refusing route  → Err, original RefusalKind, no hidden state
//! best-effort  + refusing route  → Ok, complete dense half, refusal recorded
//! not applicable                 → Ok, local dispatch, nothing recorded
//! ```
//!
//! The first is the merge gate ([`gate`]). [`outcomes`] holds the other two,
//! and the question strict semantics raise that the old silent path did not:
//! what a refusal leaves behind. "Fix the residency and retry" is only a
//! supported workflow if a refused step mutated nothing — so a decode either
//! rewinds what it appended, or says it could not and refuses to continue.
//!
//! ## Scope
//!
//! `prefill_quant` / `decode_step_quant` are deliberately absent: by contract
//! they discard the caller's FFN and substitute a `WalkFfn` built from the
//! vindex, so a strict route cannot reach the dispatch ring through them. The
//! resident pair is the production strict path — it threads the caller's FFN
//! through the same `do_prefill` / `do_decode_step` bodies — and is covered.
//!
//! The engines that route their FFN through `larql_kv::engines::layer_ffn_or_moe`
//! rather than through the dispatch helpers (markov-rs, turbo-quant,
//! unlimited-context, boundary-per-layer) are **not** covered here, because
//! that copy still swallows. Widening it is the next ring.

mod entry;
mod gate;
mod outcomes;
mod routes;
