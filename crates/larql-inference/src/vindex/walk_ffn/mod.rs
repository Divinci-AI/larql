//! `WalkFfn` — FFN backend that replaces dense matmul with vindex lookups.
//!
//! Routing table (priority order, see `forward_routed`):
//!
//! | # | Condition                                            | Path                         |
//! | - | ---------------------------------------------------- | ---------------------------- |
//! | 0 | `seq_len == 1`, L1 cache hit, `Observe::Skip` only   | `l1_cache_hit`               |
//! | 1a| `index.has_overrides_at(layer)` + delta preconditions| `override:base_delta`        |
//! | 1b| `index.has_overrides_at(layer)`                      | `override:sparse`            |
//! | 2 | `config.is_sparse(layer)`                            | `sparse:*`                   |
//! | 3 | `index.has_fp4_storage()`                            | `fp4_storage:sparse`         |
//! | 4 | `has_interleaved_kquant()` + gated FFN               | `interleaved_kquant:native`  |
//! | 5 | `has_interleaved_q4()` + backend has Q4              | `interleaved_q4:*`           |
//! | 6 | `has_interleaved()`                                  | `interleaved`                |
//! | 7 | `has_full_mmap_ffn()`                                | `full_mmap`                  |
//! | 8 | `has_interleaved_kquant()`                           | `interleaved_kquant:dequant` |
//! | 9 | `has_down_features()` + safetensors weights loaded   | `exact`                      |
//! | 10| Fallback: sparse matmul against safetensors weights  | `weights_fallback:*`         |
//!
//! Priority rationale: overrides must bypass everything (whole-layer
//! paths silently lose overridden features). Overridden layers first
//! try the exact base+delta path (`base_delta.rs` — dense base plus
//! O(|patch|) per-slot corrections, 2026-07-30 review item 16), which
//! declines to the walk when its exactness preconditions don't hold.
//! FP4/FP8 is handled by the
//! sparse path because the format is per-feature by construction —
//! there is no batched FP4 dense path on CPU. Q4K/Q4/f32 interleaved
//! are perf-preference ordered. `exact` and `weights_fallback` are
//! correctness baselines that require safetensors weights.
//!
//! Each walk path lives in its own module under this directory:
//!
//! - `base_delta.rs`              — exact base+delta execution for patched layers
//! - `sparse.rs`                  — per-feature walk, unified ffn_row_* dispatch
//!   (`sparse_gemv.rs` full-K gemv, `sparse_route.rs` hit selection,
//!   `sparse_parallel.rs` parallel Q4K down, `sparse_gather.rs` gather kernel)
//! - `interleaved.rs`             — f32 interleaved mmap, three BLAS gemms
//! - `interleaved_q4.rs`            — Q4_0 interleaved, CPU kernel / Metal Q4
//! - `interleaved_kquant_native.rs` — K-quant direct matvec, no dequant cache
//! - `interleaved_kquant_dequant.rs`— K-quant dequant, full f32 dense after decode
//! - `full_mmap.rs`               — gate/up/down in three separate mmap files
//! - `exact.rs`                   — gate/up from safetensors, down from mmap
//! - `observe.rs`                 — Observe mode (forward / forward_observed split)
//! - `helpers.rs`                 — cross-path utilities + trace metadata
//! - `builders.rs`                — constructors and builder methods
//! - `timings.rs`                 — phase-timing sink + env-var trace plumbing
//!
//! Adding a new storage format should almost never touch `mod.rs` — add
//! a new module with a single walk function, one branch in the routing
//! ladder, and a unit test in `routing_tests.rs`.
//!
//! Observation (2026-07-30 review, item 15): `forward` runs every path
//! in `Observe::Skip` mode, which by construction never allocates or
//! fills an activation buffer; `forward_observed` runs `Observe::Record`
//! and returns an honest [`FfnActivations`] — sparse paths emit exactly
//! the `(feature, activation)` pairs they computed, dense paths hand
//! over their intrinsic activation matrix, and the L1 cache read is
//! bypassed (recompute) rather than fabricating zeros.

use ndarray::Array2;

use crate::ffn::sparse_compute::{sparse_ffn_forward, sparse_ffn_forward_observed};
use crate::ffn::{FfnActivations, FfnBackend};
use crate::model::ModelWeights;
use crate::vindex::l1_cache::FfnL1Cache;
use crate::vindex::walk_config::WalkFfnConfig;
use larql_compute::prelude::*;
use observe::Observe;

use larql_vindex::{GateIndex, WalkHit, WalkTrace};

mod base_delta;
mod builders;
mod exact;
mod full_mmap;
mod helpers;
mod interleaved;
mod interleaved_kquant_dequant;
mod interleaved_kquant_native;
mod interleaved_q4;
mod observe;
mod selector;
mod sparse;
mod sparse_gather;
mod sparse_gemv;
mod sparse_parallel;
mod sparse_route;
mod thresholds;
mod timings;

#[cfg(test)]
mod base_delta_tests;
#[cfg(test)]
mod dispatch_tests;
#[cfg(test)]
mod routing_tests;

pub use helpers::DispatchEntry;
pub use timings::PhaseTimingsHandle;

pub struct WalkFfn<'a> {
    pub weights: &'a ModelWeights,
    pub index: &'a dyn GateIndex,
    pub config: WalkFfnConfig,
    pub backend: Option<&'a dyn ComputeBackend>,
    trace_residuals: std::cell::RefCell<Vec<(usize, Vec<f32>)>>,
    record_trace: bool,
    l1_cache: Option<FfnL1Cache>,
    /// Dispatch-trace sink. `None` = disabled. When `Some`, every walk
    /// path appends a (layer, name) entry on exit. Used by the routing
    /// unit tests and by the env-var dispatch trace for Q2 debugging.
    dispatch_trace: std::cell::RefCell<Option<Vec<DispatchEntry>>>,
    /// Phase-timing sink for `sparse:parallel_q4k_down`. `None` =
    /// disabled. When `Some`, the branch records cache_fetch / scan /
    /// reduce timings via atomic adds.
    pub(super) phase_timings: Option<std::sync::Arc<PhaseTimingsHandle>>,
    /// Lazy cache of per-feature `‖down_row‖` per layer. Built on first
    /// use when the selector is `GateXDownNorm` or `GateXUpDownNorm`.
    pub(super) down_norms_cache: std::cell::RefCell<Vec<Option<std::sync::Arc<Vec<f32>>>>>,
    /// Lazy cache of per-feature `‖up_row‖` per layer. Built on first
    /// use when the selector is `GateXUpDownNorm`.
    pub(super) up_norms_cache: std::cell::RefCell<Vec<Option<std::sync::Arc<Vec<f32>>>>>,
    /// Count of joint-selector calls that silently degraded to the
    /// production GateOnly chain (missing batched scores or norms).
    /// A non-zero count on an A/B sweep means the labelled selector
    /// was not the selector that ran (2026-07-30 review, M10). Also
    /// surfaced per-call as a `selector:fallback` dispatch-trace entry.
    pub(super) selector_fallbacks: std::cell::Cell<u64>,
}

impl<'a> WalkFfn<'a> {
    fn top_k_for(&self, layer: usize) -> usize {
        self.config.k_for(layer).unwrap_or(usize::MAX)
    }

    /// Number of joint-selector calls that degraded to the production
    /// GateOnly chain so far. Check this after an A/B sweep: non-zero
    /// means the sweep's selector label lies for at least some calls.
    pub fn selector_fallback_count(&self) -> u64 {
        self.selector_fallbacks.get()
    }

    pub fn l1_cache_stats(&self) -> Option<(u64, u64)> {
        self.l1_cache.as_ref().map(|c| (c.hits(), c.misses()))
    }

    /// Drain the dispatch trace and return its accumulated entries.
    /// Returns empty if the trace wasn't enabled.
    pub fn take_dispatch_trace(&self) -> Vec<DispatchEntry> {
        self.dispatch_trace
            .borrow_mut()
            .as_mut()
            .map(std::mem::take)
            .unwrap_or_default()
    }

    /// Record a dispatch entry; no-op when the trace is disabled.
    /// Called by each walk path on successful exit.
    ///
    /// Also emits to stderr when `LARQL_WALK_TRACE=1` — makes silent
    /// fallbacks immediately visible without requiring the caller to
    /// opt into the in-memory trace. The env var check is cheap on
    /// the unset path (one thread-local lookup per layer).
    pub(super) fn trace_path(&self, layer: usize, path: &'static str) {
        if let Some(vec) = self.dispatch_trace.borrow_mut().as_mut() {
            vec.push(DispatchEntry { layer, path });
        }
        if timings::walk_trace_env_enabled() {
            eprintln!("[walk_ffn] L{layer} → {path}");
        }
    }

    pub fn take_residuals(&self) -> Vec<(usize, Vec<f32>)> {
        self.trace_residuals.borrow_mut().drain(..).collect()
    }

    /// Non-draining snapshot of the residuals captured so far. Used by the
    /// early-exit path to inspect the stored-layer residuals at the resolved
    /// layer *mid-forward* (after layer L*, before the tail runs) without
    /// consuming the trace — the full forward, if it continues on a miss,
    /// keeps appending.
    pub fn peek_residuals(&self) -> Vec<(usize, Vec<f32>)> {
        self.trace_residuals.borrow().clone()
    }

    pub fn take_trace(&self) -> WalkTrace {
        let residuals = self
            .trace_residuals
            .borrow_mut()
            .drain(..)
            .collect::<Vec<_>>();
        let mut layers = Vec::with_capacity(residuals.len());
        for (layer, residual) in residuals {
            let r = ndarray::Array1::from_vec(residual);
            let hits = self.index.gate_knn(layer, &r, self.top_k_for(layer));
            let walk_hits: Vec<WalkHit> = hits
                .into_iter()
                .filter_map(|(feature, gate_score)| {
                    let meta = self.index.feature_meta(layer, feature)?.clone();
                    Some(WalkHit {
                        layer,
                        feature,
                        gate_score,
                        meta,
                    })
                })
                .collect();
            layers.push((layer, walk_hits));
        }
        WalkTrace { layers }
    }

    /// Ladder steps 4–9 — the whole-layer, override-blind paths, in
    /// priority order. Returns `None` when every step declines (caller
    /// falls through to `weights_fallback`).
    ///
    /// Factored out of `forward_with_activation` so the base+delta
    /// path (`base_delta.rs`) can produce its dense base through the
    /// EXACT same code the unpatched ladder runs — base numerics
    /// identical to the unpatched model by construction (2026-07-30
    /// review, item 16 exactness condition 4). On a patched index
    /// these paths read only base bytes, so their output is the
    /// pre-patch dense result — which is precisely what base+delta
    /// needs to correct, and precisely why the ordinary override arm
    /// must never land here directly.
    fn forward_unpatched_whole_layer(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> Option<(Array2<f32>, Array2<f32>)> {
        // 4. Q4K native — direct matvec via `kquant_matmul_transb`. Same
        //    kernel `ffn_decode_step_native` uses. Goes ahead of Q4_0 /
        //    f32 interleaved / full_mmap / dequant because for a vindex
        //    that has both Q4K and one of those, this is the fast path.
        if self.index.has_interleaved_kquant() {
            if let Some(r) = self.walk_ffn_kquant_native(layer, x) {
                return Some(r);
            }
        }

        // 5. Q4_0 interleaved — batched GPU submission when the
        //    backend implements `q4_matvec_pair_batch` (probed by
        //    calling — see `walk_ffn_q4_interleaved`), C kernel
        //    otherwise. Gate on the format the data actually is:
        //    the slab is Q4_0, and gating on Q4_K here used to
        //    admit `CpuBackend` (which advertises Q4_K matvec but
        //    not the batch API) into an unwrap-on-None panic.
        if self.index.has_interleaved_q4()
            && self
                .backend
                .is_some_and(|be| be.supports_quant(::larql_compute::QuantFormat::Q4_0))
        {
            if let Some(r) = self.walk_ffn_q4_interleaved(layer, x) {
                return Some(r);
            }
        }

        // 6. f32 interleaved.
        if self.index.has_interleaved() {
            if let Some(r) = self.walk_ffn_interleaved(layer, x) {
                return Some(r);
            }
        }

        // 7. Full mmap — gate/up/down in separate files.
        if self.index.has_full_mmap_ffn() {
            if let Some(r) = self.walk_ffn_full_mmap(layer, x) {
                return Some(r);
            }
        }

        // 8. Q4K interleaved dequant — fallback for non-gated archs and
        //    any case where `walk_ffn_kquant_native` returns `None`.
        if self.index.has_interleaved_kquant() {
            if let Some(r) = self.walk_ffn_kquant_dequant(layer, x) {
                return Some(r);
            }
        }

        // 9. Exact — down from mmap, gate/up from safetensors.
        if self.index.has_down_features() {
            return Some(self.walk_ffn_exact(layer, x));
        }

        None
    }

    /// Ladder step 10 — sparse matmul against safetensors weights, and
    /// the only path besides the sparse walk that honours feature
    /// overrides. Doubles as the hard fallback for overridden layers
    /// whose sparse walk fails: every whole-layer ladder path is
    /// override-blind, so an overridden layer must land here, never
    /// fall through (2026-07-30 review, finding M7).
    fn weights_fallback(
        &self,
        layer: usize,
        x: &Array2<f32>,
        observe: Observe,
    ) -> (Array2<f32>, Option<FfnActivations>) {
        let top_k = self.top_k_for(layer);
        let features = self.index.gate_knn_batch(layer, x, top_k);
        let has_any_override = features.iter().any(|&f| {
            self.index.down_override(layer, f).is_some()
                || self.index.up_override(layer, f).is_some()
        }) || self.index.has_overrides_at(layer);

        if has_any_override {
            let slot_overrides: Vec<crate::ffn::FeatureSlotOverride<'_>> = features
                .iter()
                .map(|&f| crate::ffn::FeatureSlotOverride {
                    feature: f,
                    gate: self.index.gate_override(layer, f),
                    up: self.index.up_override(layer, f),
                    down: self.index.down_override(layer, f),
                })
                .filter(|o| o.gate.is_some() || o.up.is_some() || o.down.is_some())
                .collect();
            self.trace_path(layer, "weights_fallback:override");
            return if observe.recording() {
                let (out, obs) = crate::ffn::sparse_ffn_forward_with_full_overrides_observed(
                    self.weights,
                    layer,
                    x,
                    &features,
                    &slot_overrides,
                );
                (out, Some(obs))
            } else {
                (
                    crate::ffn::sparse_ffn_forward_with_full_overrides(
                        self.weights,
                        layer,
                        x,
                        &features,
                        &slot_overrides,
                    ),
                    None,
                )
            };
        }
        self.trace_path(layer, "weights_fallback:sparse");
        if observe.recording() {
            let (out, obs) = sparse_ffn_forward_observed(self.weights, layer, x, &features);
            (out, Some(obs))
        } else {
            (sparse_ffn_forward(self.weights, layer, x, &features), None)
        }
    }
}

impl<'a> WalkFfn<'a> {
    /// The routing ladder, parameterised on observation mode. Both trait
    /// entry points run THIS body, so `forward` and `forward_observed`
    /// can never route differently — the only mode-dependent behaviour
    /// is whether activations are recorded and whether the L1 cache
    /// read is consulted.
    ///
    /// Invariant: returns `Some` observation iff `observe` is `Record`
    /// — except the `Skip`-only L1 hit, which returns `None` trivially.
    fn forward_routed(
        &self,
        layer: usize,
        x: &Array2<f32>,
        observe: Observe,
    ) -> (Array2<f32>, Option<FfnActivations>) {
        let num_features = self.index.num_features(layer);
        if num_features == 0 {
            self.trace_path(layer, "zero_features_dense");
            let dense_ffn = crate::ffn::WeightFfn {
                weights: self.weights,
            };
            return if observe.recording() {
                let (out, obs) = dense_ffn.forward_observed(layer, x);
                (out, Some(obs))
            } else {
                (dense_ffn.forward(layer, x), None)
            };
        }

        if self.record_trace {
            let seq_len = x.shape()[0];
            let last_row = x.row(seq_len - 1).to_vec();
            self.trace_residuals.borrow_mut().push((layer, last_row));
        }

        // Override-aware routing: patched layers bypass the L1 cache and
        // every whole-layer path below, because those would silently
        // produce wrong activations for overridden features.
        if self.index.has_overrides_at(layer) {
            // 1a. Exact base+delta (2026-07-30 review, item 16): dense
            //     base through the unpatched whole-layer ladder, plus
            //     O(|patch|) per-slot corrections. Declines (None) when
            //     its exactness preconditions don't hold — the patched
            //     layer then takes the sparse walk as before, so the
            //     override→forced-sparse cliff only remains where
            //     base+delta cannot be exact.
            if let Some(result) = self.walk_ffn_base_delta(layer, x, observe) {
                return result;
            }
            // 1b. Override-aware sparse walk.
            if let Some(result) = self.walk_ffn_sparse(layer, x, observe) {
                // The sparse path has already called trace_path — no
                // need to rewrite; its name carries the specialisation.
                return result;
            }
            // Sparse failed (missing/corrupt FFN payload). Do NOT fall
            // through: the L1 cache and every whole-layer ladder path
            // below are override-blind and would silently serve
            // pre-patch weights. The safetensors fallback is the only
            // other override-aware path — take it directly.
            return self.weights_fallback(layer, x, observe);
        }

        // L1 cache: single-position only. Key is a path-independent
        // hash of the residual, so any walk path that produces the
        // same output fills the same slot.
        let seq_len = x.shape()[0];
        let l1_key: Option<u64> = if seq_len == 1 && self.l1_cache.is_some() {
            let x_row = x.row(0);
            let owned;
            let slice: &[f32] = if let Some(s) = x_row.as_slice() {
                s
            } else {
                owned = x_row.to_vec();
                &owned
            };
            Some(FfnL1Cache::residual_key(slice))
        } else {
            None
        };

        // The cache stores outputs only, so a hit can serve `forward`
        // but has nothing honest to say about activations. An observed
        // call therefore BYPASSES the read and recomputes (2026-07-30
        // review, item 15 — the hit used to fabricate an all-zero
        // activation matrix); the recomputed result refreshes the slot.
        if !observe.recording() {
            if let Some(key) = l1_key {
                if let Some(cache) = &self.l1_cache {
                    if let Some(cached) = cache.get(layer, key) {
                        let hidden = x.shape()[1];
                        let mut out = Array2::<f32>::zeros((1, hidden));
                        out.row_mut(0)
                            .assign(&ndarray::ArrayView1::from(cached.as_slice()));
                        self.trace_path(layer, "l1_cache_hit");
                        return (out, None);
                    }
                }
            }
        }

        // Routing ladder. Each branch either `break`s with a result or
        // falls through to the next. See the routing table in the
        // module doc for priority order.
        let result: (Array2<f32>, Option<FfnActivations>) = 'routing: {
            // 2. Explicit sparse K from the user.
            if self.config.is_sparse(layer) {
                if let Some(r) = self.walk_ffn_sparse(layer, x, observe) {
                    break 'routing r;
                }
            }

            // 3. FP4/FP8 storage (exp 26) — no dedicated dense path.
            //    The sparse walk's unified ffn_row_* dispatch handles
            //    FP4/FP8 transparently via GateIndex. Routing FP4
            //    vindexes through sparse here is the whole point of
            //    the trait refactor: zero format-specific code in the
            //    walk kernel.
            if self.index.has_fp4_storage() {
                if let Some(r) = self.walk_ffn_sparse(layer, x, observe) {
                    break 'routing r;
                }
            }

            // 4–9. Whole-layer paths (shared with base+delta's base).
            //      Their activation matrix is an intrinsic intermediate
            //      (the down projection consumes it), so `Skip` merely
            //      drops it — no extra allocation either way.
            if let Some((out, act)) = self.forward_unpatched_whole_layer(layer, x) {
                break 'routing (out, observe.dense(act));
            }

            // 10. Last resort: sparse matmul against safetensors weights.
            //     Fires when the vindex has no FFN payload of its own
            //     (extract_level = Browse without pinned weights).
            break 'routing self.weights_fallback(layer, x, observe);
        };

        if let Some(key) = l1_key {
            if let Some(cache) = &self.l1_cache {
                cache.insert(layer, key, result.0.row(0).to_vec());
            }
        }

        result
    }
}

impl<'a> FfnBackend for WalkFfn<'a> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        // `Observe::Skip` — by construction no walk path touches an
        // activation buffer on this route.
        self.forward_routed(layer, x, Observe::Skip).0
    }

    fn forward_observed(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, FfnActivations) {
        let (out, obs) = self.forward_routed(layer, x, Observe::Record);
        (
            out,
            obs.expect("forward_routed(Record) always yields an observation"),
        )
    }

    fn name(&self) -> &str {
        "walk"
    }
}
