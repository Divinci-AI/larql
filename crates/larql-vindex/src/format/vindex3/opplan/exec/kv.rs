//! Caller-owned continuation state for the executor's traversals
//! (VI3-INF-2/3).
//!
//! [`DecodeSession`](super::decode::DecodeSession) used to own its K/V
//! rows as private per-layer `Vec<Vec<f32>>`. That was right for
//! proving decode semantics and wrong as the production
//! continuation-state architecture: the state a conversation carries
//! between steps is *policy* (residency, quantisation, windowing,
//! checkpointing), and policy composes outside the executor. This
//! module is the seam: the session drives a [`KvState`] provider and
//! owns none of the rows.
//!
//! The provider learns its geometry **from the plan** —
//! [`plan_kv_geometry`] reads each layer's KV row width and attention
//! window out of the executable program itself. No head-count
//! inference from a family registry, no `ModelArchitecture` questions:
//! sliding/full and head dims are explicit properties of the program.
//!
//! Contract notes:
//!
//! - Rows are stored exactly as the backend returned them (post-norm,
//!   post-rope) and returned position-ordered from position 0. A
//!   provider must hold **every** appended row — the span logic, not
//!   the store, excludes positions a window masks out (the cache may
//!   hold a position the span must exclude; dropping it is a policy
//!   the executor has not been taught to coordinate with).
//! - The `&[Vec<f32>]` row-slice shape mirrors
//!   [`AttentionStepCall`](super::backend::AttentionStepCall) and
//!   changes only with it; a flat or device-resident representation is
//!   a later rung tied to that backend contract.

use super::super::ComponentOpPlan;

/// One layer's continuation-state geometry, read from the plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerKvGeometry {
    /// Row width of one position's K (and V): `num_kv_heads * head_dim`.
    pub kv_dim: usize,
    /// The layer's attention window in positions; `None` = full span.
    /// Informational for sizing/policy — see the module contract: the
    /// store still holds every row.
    pub window: Option<usize>,
}

/// Every layer's [`LayerKvGeometry`], in layer order, from the plan.
pub fn plan_kv_geometry(plan: &ComponentOpPlan) -> Vec<LayerKvGeometry> {
    plan.layers
        .iter()
        .map(|layer| {
            // A layer whose attention is not softmax keeps no KV row, so it
            // has no geometry to state. Reaching here with one means an
            // executor ran a plan the entry points should have refused
            // (see `refuse_unexecutable_attention`) — a fabricated row
            // width here would silently mis-size every later layer's store.
            let attention = layer.attention.softmax().unwrap_or_else(|| {
                panic!(
                    "layer {} carries {}, which keeps no KV row; \
                     execution should have refused this plan",
                    layer.layer,
                    layer.attention.declared_name(),
                )
            });
            LayerKvGeometry {
                kv_dim: attention.num_kv_heads * attention.head_dim,
                window: attention.window,
            }
        })
        .collect()
}

/// Per-layer K/V continuation state, owned by the caller for its
/// entire lifetime — execution modes merely consume and update it.
/// The batch prefill ([`prefill_plan`](super::prefill_plan)) and the
/// incremental [`DecodeSession`](super::decode::DecodeSession) drive
/// the **same** provider; there is no batch-state → decode-state
/// translation anywhere.
///
/// Every traversal calls [`prepare`](Self::prepare) before driving the
/// provider, reads earlier rows, and appends new positions' pairs.
/// The decode step appends layer 0..n for position p, then again for
/// p+1; the batch prefill appends all positions for layer 0, then all
/// for layer 1 — a provider must not assume one interleaving.
pub trait KvState {
    /// Announce the traversal's geometry before any append. An
    /// announcement, **not** a reset: a provider already holding rows
    /// (a prefilled state being resumed) keeps them.
    fn prepare(&mut self, layers: &[LayerKvGeometry]);

    /// Append one position's K and V rows for `layer`.
    fn append(&mut self, layer: usize, key: Vec<f32>, value: Vec<f32>);

    /// All K rows appended for `layer`, position-ordered from 0.
    fn keys(&self, layer: usize) -> &[Vec<f32>];

    /// All V rows appended for `layer`, position-ordered from 0.
    fn values(&self, layer: usize) -> &[Vec<f32>];

    /// The logical continuation position: the next position this state
    /// continues from. Owned explicitly by the provider — **never**
    /// derived from a physical row count, because a windowed or
    /// compressed provider may one day retain fewer rows than the
    /// positions it logically represents. A session resuming over this
    /// state starts here; there is no separate start-position argument
    /// anywhere that could disagree with it.
    fn position(&self) -> usize;

    /// Record that the state now continues from `position`. The
    /// driving traversal is the only writer, and calls are monotonic —
    /// once per consumed position on the decode path, once at the end
    /// of a batch prefill.
    fn set_position(&mut self, position: usize);
}

/// The default provider: plain per-layer row vectors — exactly the
/// state [`DecodeSession`](super::decode::DecodeSession) used to own
/// privately, now behind the seam. The decode-vs-batch parity gates
/// pin that this indirection changed nothing.
#[derive(Default)]
pub struct RowKvState {
    layers: Vec<LayerRows>,
    position: usize,
}

#[derive(Default)]
struct LayerRows {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
}

impl KvState for RowKvState {
    fn prepare(&mut self, layers: &[LayerKvGeometry]) {
        if self.layers.is_empty() {
            self.layers = layers.iter().map(|_| LayerRows::default()).collect();
        } else {
            // A held state is being resumed; it must be state for a
            // program of this shape. Fail loud — silently reshaping
            // continuation state would be a wrong-conversation bug.
            assert_eq!(
                self.layers.len(),
                layers.len(),
                "resumed KV state holds {} layers but the plan declares {}",
                self.layers.len(),
                layers.len()
            );
        }
    }

    fn append(&mut self, layer: usize, key: Vec<f32>, value: Vec<f32>) {
        let rows = &mut self.layers[layer];
        rows.keys.push(key);
        rows.values.push(value);
    }

    fn keys(&self, layer: usize) -> &[Vec<f32>] {
        &self.layers[layer].keys
    }

    fn values(&self, layer: usize) -> &[Vec<f32>] {
        &self.layers[layer].values
    }

    fn position(&self) -> usize {
        self.position
    }

    fn set_position(&mut self, position: usize) {
        self.position = position;
    }
}
