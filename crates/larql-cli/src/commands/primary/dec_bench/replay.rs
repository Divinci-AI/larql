//! Pure replay logic for the DEC loadgen: sweep-plan expansion, wire-frame
//! construction, and per-point statistics. Everything transport-shaped lives
//! in `replay_runtime.rs`; every byte on the wire goes through the SAME codec
//! functions the production client uses (parity discipline).

use larql_compute::cpu::ops::q4k_q8k_dot::quantize_x_to_q8k;
use larql_inference::ffn::moe_remote::multi_layer_wire::{
    MULTI_LAYER_BATCH_PATH, MULTI_LAYER_BATCH_Q8K_PATH,
};
use larql_inference::ffn::moe_remote::{
    decode_multi_layer_response, encode_multi_layer_request, encode_multi_layer_request_q8k,
    MultiLayerTask, MultiLayerTaskQ8K, MULTI_LAYER_BATCH_CONTENT_TYPE,
    MULTI_LAYER_BATCH_Q8K_CONTENT_TYPE,
};
use larql_inference::ffn::remote::{WALK_FFN_PATH, WALK_FFN_Q8K_PATH};
use larql_inference::{encode_binary_request, encode_q8k_batch_request};

use super::super::bench::row::compute_percentiles;
use super::capture_format::CapturePool;

// ── Sweep plan ────────────────────────────────────────────────────────────────

/// One wire-format arm of the sweep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WireArm {
    F32,
    F16,
    I8,
    Q8k,
}

impl WireArm {
    pub fn parse_list(s: &str) -> Result<Vec<WireArm>, String> {
        s.split(',')
            .map(|w| match w.trim() {
                "f32" => Ok(WireArm::F32),
                "f16" => Ok(WireArm::F16),
                "i8" => Ok(WireArm::I8),
                "q8k" => Ok(WireArm::Q8k),
                other => Err(format!(
                    "unknown wire format {other:?} (expected f32, f16, i8, q8k)"
                )),
            })
            .collect()
    }

    pub fn label(self) -> &'static str {
        match self {
            WireArm::F32 => "f32",
            WireArm::F16 => "f16",
            WireArm::I8 => "i8",
            WireArm::Q8k => "q8k",
        }
    }

    /// Numeric twin of [`Self::label`] for numeric-only metric ingesters.
    pub fn code(self) -> u32 {
        match self {
            WireArm::F32 => 0,
            WireArm::F16 => 1,
            WireArm::I8 => 2,
            WireArm::Q8k => 3,
        }
    }

    /// `Accept` header for the strict arm (no fallback formats offered).
    /// `None` for Q8K, which is its own endpoint rather than a negotiated CT.
    pub fn accept(self) -> Option<&'static str> {
        match self {
            WireArm::F32 => Some(larql_inference::BINARY_CT),
            WireArm::F16 => Some(larql_inference::F16_CT),
            WireArm::I8 => Some(larql_inference::I8_CT),
            WireArm::Q8k => None,
        }
    }
}

// ── Endpoint seam ─────────────────────────────────────────────────────────────

/// CLI-level endpoint family (`--endpoint`). Combined with the wire axis it
/// resolves to a concrete [`Endpoint`] via [`Endpoint::resolve`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndpointKind {
    /// Dense/shared-expert FFN path (`/v1/walk-ffn[-q8k]`).
    WalkFfn,
    /// Routed-experts multi-layer path (`/v1/experts/multi-layer-batch[-q8k]`).
    Experts,
}

impl EndpointKind {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.trim() {
            "walk-ffn" => Ok(Self::WalkFfn),
            "experts" => Ok(Self::Experts),
            other => Err(format!(
                "unknown endpoint {other:?} (expected walk-ffn, experts)"
            )),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::WalkFfn => "walk-ffn",
            Self::Experts => "experts",
        }
    }
}

/// Which movement-ratio denominator an endpoint's byte accounting uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenominatorSource {
    /// `ffn_weights.per_layer_dense_bytes` summed over replayed layers.
    Dense,
    /// `ffn_weights.moe.per_expert_bytes` × captured routing (naive/union).
    RoutedExperts,
}

/// One concrete server endpoint of the sweep. Owns the request path,
/// content-type/Accept behaviour, frame build + response decode (via the
/// production codecs — parity discipline), `server_ms` availability, and the
/// movement-ratio denominator source. The former two-way `send_one` branch
/// made explicit (audit §4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endpoint {
    WalkFfn,
    WalkFfnQ8k,
    ExpertsMultiLayer,
    ExpertsMultiLayerQ8k,
}

impl Endpoint {
    /// Map (endpoint family, wire arm) to a concrete endpoint. The
    /// multi-layer expert wire has no f16/i8 variant — those combinations
    /// are a loud arg-validation error, not a silent fallback.
    pub fn resolve(kind: EndpointKind, wire: WireArm) -> Result<Self, String> {
        match (kind, wire) {
            (EndpointKind::WalkFfn, WireArm::Q8k) => Ok(Self::WalkFfnQ8k),
            (EndpointKind::WalkFfn, _) => Ok(Self::WalkFfn),
            (EndpointKind::Experts, WireArm::F32) => Ok(Self::ExpertsMultiLayer),
            (EndpointKind::Experts, WireArm::Q8k) => Ok(Self::ExpertsMultiLayerQ8k),
            (EndpointKind::Experts, w) => Err(format!(
                "--endpoint experts has no {} arm — the multi-layer expert wire carries \
                 f32 or q8k frames only (use --wire f32,q8k)",
                w.label()
            )),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::WalkFfn => "walk-ffn",
            Self::WalkFfnQ8k => "walk-ffn-q8k",
            Self::ExpertsMultiLayer => "experts-ml",
            Self::ExpertsMultiLayerQ8k => "experts-ml-q8k",
        }
    }

    /// Numeric twin of [`Self::label`] for numeric-only metric ingesters.
    pub fn code(self) -> u32 {
        match self {
            Self::WalkFfn => 0,
            Self::WalkFfnQ8k => 1,
            Self::ExpertsMultiLayer => 2,
            Self::ExpertsMultiLayerQ8k => 3,
        }
    }

    /// Request path — the production constants, not re-declared strings.
    pub fn path(self) -> &'static str {
        match self {
            Self::WalkFfn => WALK_FFN_PATH,
            Self::WalkFfnQ8k => WALK_FFN_Q8K_PATH,
            Self::ExpertsMultiLayer => MULTI_LAYER_BATCH_PATH,
            Self::ExpertsMultiLayerQ8k => MULTI_LAYER_BATCH_Q8K_PATH,
        }
    }

    /// Request `Content-Type`.
    pub fn request_content_type(self) -> &'static str {
        match self {
            Self::WalkFfn => larql_inference::BINARY_CT,
            Self::WalkFfnQ8k => larql_inference::Q8K_BATCH_CT,
            Self::ExpertsMultiLayer => MULTI_LAYER_BATCH_CONTENT_TYPE,
            Self::ExpertsMultiLayerQ8k => MULTI_LAYER_BATCH_Q8K_CONTENT_TYPE,
        }
    }

    /// `Accept` header. Only the f32 walk-ffn endpoint negotiates response
    /// formats (per wire arm); q8k walk-ffn is its own endpoint with a fixed
    /// response, and both multi-layer endpoints always answer with the f32
    /// multi-layer CT (mirrors the production shard client's Accept).
    pub fn accept(self, wire: WireArm) -> Option<&'static str> {
        match self {
            Self::WalkFfn => wire.accept(),
            Self::WalkFfnQ8k => None,
            Self::ExpertsMultiLayer | Self::ExpertsMultiLayerQ8k => {
                Some(MULTI_LAYER_BATCH_CONTENT_TYPE)
            }
        }
    }

    /// Whether the response embeds a server compute-latency field. Only the
    /// f32 walk-ffn response carries one; the q8k walk-ffn and both
    /// multi-layer responses have no latency field (audit §2).
    pub fn has_server_ms(self) -> bool {
        matches!(self, Self::WalkFfn)
    }

    /// Movement-ratio denominator family.
    pub fn denominator(self) -> DenominatorSource {
        match self {
            Self::WalkFfn | Self::WalkFfnQ8k => DenominatorSource::Dense,
            Self::ExpertsMultiLayer | Self::ExpertsMultiLayerQ8k => {
                DenominatorSource::RoutedExperts
            }
        }
    }

    /// Routed endpoints replay captured routing — the pool must carry the
    /// `--routing` sidecars.
    pub fn requires_routing(self) -> bool {
        self.denominator() == DenominatorSource::RoutedExperts
    }
}

/// How the per-layer requests of one step are dispatched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchMode {
    /// Sequential per-layer round trips; step time = Σ layer RTTs.
    Streaming,
    /// All layers fired in parallel (mirrors `forward_predispatch_all`);
    /// step time = fan-out wall time.
    Batch,
}

impl DispatchMode {
    pub fn parse_list(s: &str) -> Result<Vec<DispatchMode>, String> {
        s.split(',')
            .map(|d| match d.trim() {
                "streaming" => Ok(DispatchMode::Streaming),
                "batch" => Ok(DispatchMode::Batch),
                other => Err(format!(
                    "unknown dispatch mode {other:?} (expected streaming, batch)"
                )),
            })
            .collect()
    }

    pub fn label(self) -> &'static str {
        match self {
            DispatchMode::Streaming => "streaming",
            DispatchMode::Batch => "batch",
        }
    }

    pub fn code(self) -> u32 {
        match self {
            DispatchMode::Streaming => 0,
            DispatchMode::Batch => 1,
        }
    }
}

/// One point of the sweep: (batch, wire, dispatch) on a resolved endpoint.
#[derive(Debug, Clone, Copy)]
pub struct SweepPoint {
    pub batch: usize,
    pub wire: WireArm,
    pub dispatch: DispatchMode,
    pub endpoint: Endpoint,
}

pub fn parse_batch_list(s: &str) -> Result<Vec<usize>, String> {
    s.split(',')
        .map(|b| {
            b.trim()
                .parse::<usize>()
                .map_err(|_| format!("invalid batch size {b:?}"))
                .and_then(|n| {
                    if n == 0 {
                        Err("batch size 0 is invalid".into())
                    } else {
                        Ok(n)
                    }
                })
        })
        .collect()
}

/// Full cross product, ordered batch-major so all wire/dispatch arms of one
/// batch size run adjacently (comparable thermal window). Errs on wire arms
/// the endpoint family cannot serve (f16/i8 on `experts` — arg-validation
/// time, before any request is sent).
pub fn expand_sweep(
    batches: &[usize],
    wires: &[WireArm],
    dispatches: &[DispatchMode],
    kind: EndpointKind,
) -> Result<Vec<SweepPoint>, String> {
    let endpoints: Vec<Endpoint> = wires
        .iter()
        .map(|&w| Endpoint::resolve(kind, w))
        .collect::<Result<_, _>>()?;
    let mut out = Vec::with_capacity(batches.len() * wires.len() * dispatches.len());
    for &batch in batches {
        for (&wire, &endpoint) in wires.iter().zip(endpoints.iter()) {
            for &dispatch in dispatches {
                out.push(SweepPoint {
                    batch,
                    wire,
                    dispatch,
                    endpoint,
                });
            }
        }
    }
    Ok(out)
}

// ── Frame construction ────────────────────────────────────────────────────────

/// Build a B-row `/v1/walk-ffn` request frame for one layer.
///
/// `rows` is `batch × hidden` contiguous f32s. `top_k` is hard-wired to 0:
/// the server's L2 FFN cache engages when `seq_len == 1 && top_k > 0`, which
/// would falsify repeated B=1 measurements with cache hits.
pub fn build_walk_ffn_frame(layer: usize, rows: &[f32], batch: usize) -> Vec<u8> {
    encode_binary_request(Some(layer), None, rows, batch, true, 0)
}

/// Build a B-row `/v1/walk-ffn-q8k` request frame for one layer: B entries
/// sharing `layer`, each one row quantised through the production
/// `quantize_x_to_q8k` path.
pub fn build_q8k_frame(layer: usize, rows: &[f32], batch: usize, hidden: usize) -> Vec<u8> {
    let q8ks: Vec<_> = (0..batch)
        .map(|i| quantize_x_to_q8k(&rows[i * hidden..(i + 1) * hidden]))
        .collect();
    let entries: Vec<(usize, &_)> = q8ks.iter().map(|q| (layer, q)).collect();
    encode_q8k_batch_request(&entries)
}

/// Build a B-task `/v1/experts/multi-layer-batch` frame for one (step,
/// layer): task `r` is raw post-attention row `r` (the server applies
/// pre_experts_norm itself) plus that row's captured `(expert_id, weight)`
/// routing. Encoded with the production codec (parity discipline).
pub fn build_experts_ml_frame(
    layer: usize,
    rows: &[f32],
    routing: &[Vec<(u32, f32)>],
    batch: usize,
    hidden: usize,
) -> Vec<u8> {
    let tasks: Vec<MultiLayerTask> = (0..batch)
        .map(|r| MultiLayerTask {
            layer,
            residual: rows[r * hidden..(r + 1) * hidden].to_vec(),
            expert_ids: routing[r].iter().map(|&(id, _)| id).collect(),
            weights: routing[r].iter().map(|&(_, w)| w).collect(),
        })
        .collect();
    encode_multi_layer_request(&tasks)
}

/// Build a B-task `/v1/experts/multi-layer-batch-q8k` frame for one (step,
/// layer): task `r` is pre-experts-normed row `r` quantised through the
/// production `quantize_x_to_q8k` path, plus that row's captured routing.
pub fn build_experts_ml_q8k_frame(
    layer: usize,
    normed_rows: &[f32],
    routing: &[Vec<(u32, f32)>],
    batch: usize,
    hidden: usize,
) -> Vec<u8> {
    let tasks: Vec<MultiLayerTaskQ8K> = (0..batch)
        .map(|r| {
            let q8k = quantize_x_to_q8k(&normed_rows[r * hidden..(r + 1) * hidden]);
            MultiLayerTaskQ8K {
                layer,
                hidden,
                qs: q8k.qs,
                d: q8k.d,
                sums: q8k.sums,
                expert_ids: routing[r].iter().map(|&(id, _)| id).collect(),
                weights: routing[r].iter().map(|&(_, w)| w).collect(),
            }
        })
        .collect();
    encode_multi_layer_request_q8k(&tasks)
}

/// Decode + validate a multi-layer-batch response through the production
/// decoder. Returns whether any `h2` value is non-zero — the post-warmup
/// corruption guard (a routed replay whose responses are all-zero means
/// wrong expert ids / shard ownership, the audit §1a failure class).
pub fn check_experts_response(
    body: &[u8],
    layer: usize,
    batch: usize,
    hidden: usize,
) -> Result<bool, String> {
    let results = decode_multi_layer_response(body)
        .ok_or_else(|| format!("experts replay layer {layer}: response failed to decode"))?;
    if results.len() != batch
        || results
            .iter()
            .any(|r| r.layer != layer || r.h2.len() != hidden)
    {
        return Err(format!(
            "experts replay layer {layer}: expected {batch} results × hidden {hidden} for \
             layer {layer}, got {} results",
            results.len()
        ));
    }
    Ok(results.iter().any(|r| r.h2.iter().any(|&v| v != 0.0)))
}

/// Pool-sourced inputs for one (step, layer) request frame. `rows` comes
/// from the endpoint's source plane; `routing` is per-row captured pairs
/// (experts endpoints only).
pub struct FrameInputs {
    pub layer: usize,
    pub rows: Vec<f32>,
    pub routing: Option<Vec<Vec<(u32, f32)>>>,
}

/// Gather one (step, layer) frame's inputs from the pool:
///   * walk-ffn endpoints — dense-prenormed `residuals.bin` rows;
///   * experts f32 — raw post-attention rows (`raw.bin`) + routing;
///   * experts q8k — pre-experts-normed rows (`normed.bin`) + routing.
///
/// A row whose routing record is all-sentinel on a replayed layer becomes a
/// zero-expert task (valid wire, costs nothing server-side — the layer set
/// is pre-filtered to MoE layers via [`routed_layer_subset`]).
pub fn gather_frame_inputs(
    pool: &CapturePool,
    endpoint: Endpoint,
    layer: usize,
    step: usize,
    batch: usize,
) -> Result<FrameInputs, String> {
    let (rows, routing) = match endpoint {
        Endpoint::WalkFfn | Endpoint::WalkFfnQ8k => (pool.rows(batch, step, layer)?, None),
        Endpoint::ExpertsMultiLayer | Endpoint::ExpertsMultiLayerQ8k => {
            let rows = if endpoint == Endpoint::ExpertsMultiLayer {
                pool.raw_rows(batch, step, layer)?
            } else {
                pool.normed_rows(batch, step, layer)?
            };
            let mut routing = Vec::with_capacity(batch);
            for prompt in 0..batch {
                routing.push(
                    pool.routing(prompt, step, layer)?
                        .map(|pairs| pairs.to_vec())
                        .unwrap_or_default(),
                );
            }
            (rows, Some(routing))
        }
    };
    Ok(FrameInputs {
        layer,
        rows,
        routing,
    })
}

/// Build the endpoint's wire frame from gathered inputs, through the same
/// codec functions the production client uses.
pub fn build_frame(
    endpoint: Endpoint,
    inputs: &FrameInputs,
    batch: usize,
    hidden: usize,
) -> Result<Vec<u8>, String> {
    let routing = || {
        inputs
            .routing
            .as_deref()
            .ok_or_else(|| "experts frame build: inputs carry no routing".to_string())
    };
    Ok(match endpoint {
        Endpoint::WalkFfn => build_walk_ffn_frame(inputs.layer, &inputs.rows, batch),
        Endpoint::WalkFfnQ8k => build_q8k_frame(inputs.layer, &inputs.rows, batch, hidden),
        Endpoint::ExpertsMultiLayer => {
            build_experts_ml_frame(inputs.layer, &inputs.rows, routing()?, batch, hidden)
        }
        Endpoint::ExpertsMultiLayerQ8k => {
            build_experts_ml_q8k_frame(inputs.layer, &inputs.rows, routing()?, batch, hidden)
        }
    })
}

/// Short wire label for a response content-type (unknown CTs pass through
/// verbatim so the run record never hides what the server sent).
pub fn wire_label_for_content_type(ct: &str) -> String {
    match ct {
        larql_inference::BINARY_CT => "f32".into(),
        larql_inference::F16_CT => "f16".into(),
        larql_inference::I8_CT => "i8".into(),
        larql_inference::Q8K_BATCH_CT => "q8k".into(),
        MULTI_LAYER_BATCH_CONTENT_TYPE => "experts-ml".into(),
        MULTI_LAYER_BATCH_Q8K_CONTENT_TYPE => "experts-ml-q8k".into(),
        other => other.to_string(),
    }
}

// ── Per-point statistics ──────────────────────────────────────────────────────

/// One request's measurements (one layer of one step of one repeat).
#[derive(Debug, Clone)]
pub struct RequestSample {
    pub layer: usize,
    pub client_ms: f64,
    /// Server compute ms from the embedded response header; `None` on the
    /// Q8K endpoint (its response carries no latency field).
    pub server_ms: Option<f64>,
    pub bytes_sent: u64,
    pub bytes_recv: u64,
    /// Wire format the server actually served (from the response
    /// content-type). Accept negotiation may fall back — e.g. an i8 arm
    /// served as f32 — and the run record must say so.
    pub served_wire: String,
    /// Whether any decoded response value was non-zero. Consumed by the
    /// post-warmup corruption guard on routed points (audit §1a class);
    /// computed at decode time, outside the timed window.
    pub any_nonzero: bool,
}

/// Accumulated raw measurements for one sweep point.
#[derive(Debug, Default)]
pub struct SweepPointStats {
    /// One entry per (repeat × step): full-pass step time in ms.
    pub step_ms: Vec<f64>,
    pub samples: Vec<RequestSample>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct LayerSummary {
    pub layer: usize,
    pub client_ms_p50: f64,
    pub client_ms_p99: f64,
}

/// Summarised sweep point, ready for the run record and pulse file.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DecPointSummary {
    pub batch: usize,
    /// Endpoint the point measured (`walk-ffn`, `walk-ffn-q8k`,
    /// `experts-ml`, `experts-ml-q8k`).
    pub endpoint: String,
    /// Numeric twin of `endpoint` (0/1/2/3) for numeric-only ingesters.
    pub endpoint_code: u32,
    pub wire_format: String,
    pub wire_format_code: u32,
    /// Wire format(s) the server actually served this point (sorted,
    /// deduped). Differs from `wire_format` when Accept negotiation fell
    /// back — the arm's bandwidth number then belongs to the served format.
    pub served_wire: Vec<String>,
    pub dispatch_mode: String,
    pub dispatch_mode_code: u32,
    pub steps: usize,
    pub step_ms_mean: f64,
    pub step_ms_p50: f64,
    pub step_ms_p99: f64,
    /// Rows served per second: `batch × 1000 / step_ms_mean`.
    pub tok_s: f64,
    /// (bytes sent + received) ÷ (steps × batch) — wire cost per token row.
    pub payload_bytes_tok: f64,
    /// Dense-endpoint denominator: FFN weight bytes the endpoint touches
    /// per token over the replayed layers. Constant across a run's points
    /// (kept per-point so every summary is self-contained); `None` on
    /// routed endpoints and when `/v1/stats` has no `ffn_weights`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weight_bytes_tok: Option<f64>,
    /// Routed-endpoint denominator, per-row naive: Σ over (step, layer,
    /// row) |E_row| × per_expert_bytes ÷ (steps × batch). PRIMARY for the
    /// movement ratio — the server streams each row's experts independently
    /// (audit §2: no cross-row weight sharing).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weight_bytes_tok_naive: Option<f64>,
    /// Routed-endpoint batch-union bound: Σ over (step, layer) |⋃_rows E|
    /// × per_expert_bytes ÷ (steps × batch) — the DEC-3 metrology bound.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weight_bytes_tok_union: Option<f64>,
    /// payload_bytes_tok ÷ denominator (dense bytes, or NAIVE for routed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub movement_ratio: Option<f64>,
    /// union ÷ naive — 1.0 = fully disjoint expert sets across rows.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub experts_union_frac: Option<f64>,
    pub server_ms_p50: Option<f64>,
    pub server_ms_p99: Option<f64>,
    pub per_layer: Vec<LayerSummary>,
}

/// Summarise one sweep point. `dense_weight_bytes_tok` feeds dense
/// endpoints' denominator; `routed` feeds routed endpoints' — the caller
/// passes whichever matches `point.endpoint.denominator()` (the other is
/// ignored).
pub fn summarize(
    point: &SweepPoint,
    stats: &SweepPointStats,
    dense_weight_bytes_tok: Option<f64>,
    routed: Option<&RoutedDenominators>,
) -> DecPointSummary {
    let (mean, p50, p99) = compute_percentiles(&stats.step_ms);
    let tok_s = if mean > 0.0 {
        point.batch as f64 * 1000.0 / mean
    } else {
        0.0
    };
    let total_bytes: u64 = stats
        .samples
        .iter()
        .map(|s| s.bytes_sent + s.bytes_recv)
        .sum();
    let token_rows = (stats.step_ms.len() * point.batch) as f64;
    let payload_bytes_tok = if token_rows > 0.0 {
        total_bytes as f64 / token_rows
    } else {
        0.0
    };

    let server_samples: Vec<f64> = stats.samples.iter().filter_map(|s| s.server_ms).collect();
    let (server_ms_p50, server_ms_p99) = if server_samples.is_empty() {
        (None, None)
    } else {
        let (_, p50, p99) = compute_percentiles(&server_samples);
        (Some(p50), Some(p99))
    };

    let mut served_wire: Vec<String> = stats
        .samples
        .iter()
        .map(|s| s.served_wire.clone())
        .collect();
    served_wire.sort();
    served_wire.dedup();

    let mut layers: Vec<usize> = stats.samples.iter().map(|s| s.layer).collect();
    layers.sort_unstable();
    layers.dedup();
    let per_layer = layers
        .into_iter()
        .map(|layer| {
            let vals: Vec<f64> = stats
                .samples
                .iter()
                .filter(|s| s.layer == layer)
                .map(|s| s.client_ms)
                .collect();
            let (_, p50, p99) = compute_percentiles(&vals);
            LayerSummary {
                layer,
                client_ms_p50: p50,
                client_ms_p99: p99,
            }
        })
        .collect();

    let (weight_bytes_tok, weight_bytes_tok_naive, weight_bytes_tok_union, experts_union_frac) =
        match point.endpoint.denominator() {
            DenominatorSource::Dense => (dense_weight_bytes_tok, None, None, None),
            DenominatorSource::RoutedExperts => {
                let naive = routed.map(|d| d.weight_bytes_tok_naive);
                let union = routed.map(|d| d.weight_bytes_tok_union);
                let frac = routed.and_then(|d| {
                    (d.weight_bytes_tok_naive > 0.0)
                        .then(|| d.weight_bytes_tok_union / d.weight_bytes_tok_naive)
                });
                (None, naive, union, frac)
            }
        };
    // Primary denominator: dense bytes for dense endpoints, NAIVE per-row
    // bytes for routed (the server streams per-row — audit §2).
    let ratio = weight_bytes_tok
        .or(weight_bytes_tok_naive)
        .and_then(|w| movement_ratio(payload_bytes_tok, w));

    DecPointSummary {
        batch: point.batch,
        endpoint: point.endpoint.label().into(),
        endpoint_code: point.endpoint.code(),
        wire_format: point.wire.label().into(),
        wire_format_code: point.wire.code(),
        served_wire,
        dispatch_mode: point.dispatch.label().into(),
        dispatch_mode_code: point.dispatch.code(),
        steps: stats.step_ms.len(),
        step_ms_mean: mean,
        step_ms_p50: p50,
        step_ms_p99: p99,
        tok_s,
        payload_bytes_tok,
        weight_bytes_tok,
        weight_bytes_tok_naive,
        weight_bytes_tok_union,
        movement_ratio: ratio,
        experts_union_frac,
        server_ms_p50,
        server_ms_p99,
        per_layer,
    }
}

// ── Movement ratio ────────────────────────────────────────────────────────────

/// `dec/movement_ratio` = wire bytes crossing the attention↔weights boundary
/// per token ÷ weight bytes the measured endpoint touches per token
/// (docs/dec-funnel.md §1). Offload ≈ 1.0 by construction; this architecture
/// targets 1e-3–1e-4.
pub fn movement_ratio(payload_bytes_tok: f64, weight_bytes_tok: f64) -> Option<f64> {
    if weight_bytes_tok > 0.0 && payload_bytes_tok >= 0.0 {
        Some(payload_bytes_tok / weight_bytes_tok)
    } else {
        None
    }
}

/// Sum the dense FFN weight bytes per token over `layers` from a `/v1/stats`
/// response (`ffn_weights.per_layer_dense_bytes`). This is the exact byte
/// count `/v1/walk-ffn` touches per token on those layers — the movement-
/// ratio denominator for the measured endpoint. Layers with `null` entries
/// (no interleaved k-quant data) contribute nothing and are reported back so
/// the caller can flag partial coverage.
///
/// Returns `(dense_bytes, missing_layer_count)`, or `None` when the stats
/// response has no `ffn_weights` block at all.
pub fn weight_bytes_per_token(stats: &serde_json::Value, layers: &[usize]) -> Option<(f64, usize)> {
    let per_layer = stats
        .get("ffn_weights")?
        .get("per_layer_dense_bytes")?
        .as_array()?;
    let mut total = 0.0f64;
    let mut missing = 0usize;
    for &l in layers {
        match per_layer.get(l).and_then(|v| v.as_u64()) {
            Some(b) => total += b as f64,
            None => missing += 1,
        }
    }
    Some((total, missing))
}

// ── Routed denominators ──────────────────────────────────────────────────────

/// Routed-endpoint movement-ratio denominators for one sweep point, from
/// the captured routing across its (steps × batch rows).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoutedDenominators {
    /// Σ over (step, layer, row) |E_row| × per_expert_bytes ÷ (steps × batch).
    pub weight_bytes_tok_naive: f64,
    /// Σ over (step, layer) |⋃_rows E| × per_expert_bytes ÷ (steps × batch).
    pub weight_bytes_tok_union: f64,
}

/// Compute both routed denominators. `expert_sets[cell][row]` is one
/// (step, layer) cell's per-row expert-id lists (sentinel-stripped —
/// records may be shorter than top_k). NAIVE is the primary movement-ratio
/// denominator: the server streams each row's experts independently (audit
/// §2 verified no cross-row weight sharing); UNION is reported alongside as
/// the DEC-3 metrology bound.
pub fn routed_weight_bytes_per_token(
    expert_sets: &[Vec<Vec<u32>>],
    per_expert_bytes: f64,
    steps: usize,
    batch: usize,
) -> RoutedDenominators {
    let tokens = (steps * batch) as f64;
    if tokens == 0.0 {
        return RoutedDenominators {
            weight_bytes_tok_naive: 0.0,
            weight_bytes_tok_union: 0.0,
        };
    }
    let mut naive_experts = 0usize;
    let mut union_experts = 0usize;
    for cell in expert_sets {
        let mut union: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
        for row in cell {
            naive_experts += row.len();
            union.extend(row.iter().copied());
        }
        union_experts += union.len();
    }
    RoutedDenominators {
        weight_bytes_tok_naive: naive_experts as f64 * per_expert_bytes / tokens,
        weight_bytes_tok_union: union_experts as f64 * per_expert_bytes / tokens,
    }
}

/// Gather one sweep point's captured expert sets from the pool and compute
/// its [`RoutedDenominators`]. `layers` must already be the MoE subset
/// (non-MoE cells contribute empty rows, which would silently deflate the
/// per-token average — [`routed_layer_subset`] excludes them up front).
pub fn routed_denominators_for_point(
    pool: &CapturePool,
    layers: &[usize],
    steps: usize,
    batch: usize,
    per_expert_bytes: f64,
) -> Result<RoutedDenominators, String> {
    let mut cells = Vec::with_capacity(steps * layers.len());
    for step in 0..steps {
        for &layer in layers {
            let mut rows = Vec::with_capacity(batch);
            for prompt in 0..batch {
                rows.push(
                    pool.routing(prompt, step, layer)?
                        .map(|pairs| pairs.iter().map(|&(id, _)| id).collect())
                        .unwrap_or_default(),
                );
            }
            cells.push(rows);
        }
    }
    Ok(routed_weight_bytes_per_token(
        &cells,
        per_expert_bytes,
        steps,
        batch,
    ))
}

/// Filter `layers` down to those that carried MoE routing in the pool (any
/// (prompt, step) cell non-empty). Non-MoE layers are excluded from the
/// routed sweep's layer set and denominator. Errs on walk-ffn-only pools.
pub fn routed_layer_subset(pool: &CapturePool, layers: &[usize]) -> Result<Vec<usize>, String> {
    let m = &pool.manifest;
    let mut out = Vec::with_capacity(layers.len());
    'layers: for &layer in layers {
        for prompt in 0..m.prompts.len() {
            for step in 0..m.steps {
                if pool.routing(prompt, step, layer)?.is_some() {
                    out.push(layer);
                    continue 'layers;
                }
            }
        }
    }
    Ok(out)
}

/// Routed-denominator facts from `/v1/stats` (`ffn_weights.moe`). A null
/// `per_expert_bytes` is a loud error — without it the routed movement
/// ratio has no denominator (audit §1c: pre-fix servers reported null on
/// sharded topologies).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoeWeightStats {
    pub per_expert_bytes: f64,
    pub top_k: Option<u64>,
    pub num_experts: Option<u64>,
}

pub fn moe_weight_stats(stats: &serde_json::Value) -> Result<MoeWeightStats, String> {
    let moe = stats
        .get("ffn_weights")
        .and_then(|f| f.get("moe"))
        .filter(|m| !m.is_null())
        .ok_or(
            "server /v1/stats has no ffn_weights.moe block — routed endpoints need MoE \
             weight accounting (is the server serving a dense model?)",
        )?;
    let per_expert_bytes = moe.get("per_expert_bytes").and_then(|v| v.as_u64()).ok_or(
        "server /v1/stats ffn_weights.moe.per_expert_bytes is null — the routed \
             movement-ratio denominator is unavailable (weights not loaded, or a pre-fix \
             server probing expert entry 0 on a shard that doesn't own it)",
    )? as f64;
    Ok(MoeWeightStats {
        per_expert_bytes,
        top_k: moe.get("top_k").and_then(|v| v.as_u64()),
        num_experts: moe.get("num_experts").and_then(|v| v.as_u64()),
    })
}

/// Parse an inclusive `"A-B"` layer range into a layer list, defaulting to
/// `0..num_layers` when absent.
pub fn parse_layer_range(spec: Option<&str>, num_layers: usize) -> Result<Vec<usize>, String> {
    match spec {
        None => Ok((0..num_layers).collect()),
        Some(s) => {
            let (a, b) = s
                .split_once('-')
                .ok_or_else(|| format!("invalid layer range {s:?} (expected A-B)"))?;
            let a: usize = a
                .trim()
                .parse()
                .map_err(|_| format!("invalid layer range start {a:?}"))?;
            let b: usize = b
                .trim()
                .parse()
                .map_err(|_| format!("invalid layer range end {b:?}"))?;
            if a > b || b >= num_layers {
                return Err(format!(
                    "layer range {a}-{b} out of bounds (model has {num_layers} layers)"
                ));
            }
            Ok((a..=b).collect())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── parsing ────────────────────────────────────────────────────────

    #[test]
    fn parse_wire_list_all_arms() {
        let arms = WireArm::parse_list("f32, f16,i8,q8k").unwrap();
        assert_eq!(
            arms,
            vec![WireArm::F32, WireArm::F16, WireArm::I8, WireArm::Q8k]
        );
        assert!(WireArm::parse_list("f64").is_err());
    }

    #[test]
    fn wire_arm_labels_codes_accepts() {
        assert_eq!(WireArm::F32.label(), "f32");
        assert_eq!(WireArm::Q8k.code(), 3);
        assert_eq!(WireArm::F32.accept(), Some("application/x-larql-ffn"));
        assert_eq!(WireArm::F16.accept(), Some("application/x-larql-ffn-f16"));
        assert_eq!(WireArm::I8.accept(), Some("application/x-larql-ffn-i8"));
        assert_eq!(WireArm::Q8k.accept(), None, "q8k is its own endpoint");
    }

    #[test]
    fn parse_dispatch_and_batch_lists() {
        let d = DispatchMode::parse_list("streaming,batch").unwrap();
        assert_eq!(d, vec![DispatchMode::Streaming, DispatchMode::Batch]);
        assert!(DispatchMode::parse_list("parallel").is_err());

        assert_eq!(parse_batch_list("1,8,16").unwrap(), vec![1, 8, 16]);
        assert!(parse_batch_list("0").is_err());
        assert!(parse_batch_list("x").is_err());
    }

    #[test]
    fn expand_sweep_is_batch_major_cross_product() {
        let points = expand_sweep(
            &[1, 8],
            &[WireArm::F32, WireArm::F16],
            &[DispatchMode::Streaming],
            EndpointKind::WalkFfn,
        )
        .unwrap();
        assert_eq!(points.len(), 4);
        assert_eq!(points[0].batch, 1);
        assert_eq!(points[1].batch, 1);
        assert_eq!(points[2].batch, 8);
        assert_eq!(points[1].wire, WireArm::F16);
        assert!(points.iter().all(|p| p.endpoint == Endpoint::WalkFfn));
    }

    #[test]
    fn expand_sweep_resolves_endpoints_and_rejects_unservable_wires() {
        let points = expand_sweep(
            &[1],
            &[WireArm::F32, WireArm::Q8k],
            &[DispatchMode::Streaming],
            EndpointKind::Experts,
        )
        .unwrap();
        assert_eq!(points[0].endpoint, Endpoint::ExpertsMultiLayer);
        assert_eq!(points[1].endpoint, Endpoint::ExpertsMultiLayerQ8k);

        let q8k_points = expand_sweep(
            &[1],
            &[WireArm::Q8k],
            &[DispatchMode::Streaming],
            EndpointKind::WalkFfn,
        )
        .unwrap();
        assert_eq!(q8k_points[0].endpoint, Endpoint::WalkFfnQ8k);

        // f16/i8 on experts must fail at expansion (arg-validation) time.
        for wire in [WireArm::F16, WireArm::I8] {
            let err = expand_sweep(
                &[1],
                &[WireArm::F32, wire],
                &[DispatchMode::Streaming],
                EndpointKind::Experts,
            )
            .unwrap_err();
            assert!(err.contains(wire.label()), "error names the bad arm: {err}");
        }
    }

    // ── frames ─────────────────────────────────────────────────────────

    #[test]
    fn walk_ffn_frame_encodes_batch_rows_and_zero_top_k() {
        let hidden = 4;
        let batch = 3;
        let rows: Vec<f32> = (0..batch * hidden).map(|i| i as f32 * 0.5).collect();
        let frame = build_walk_ffn_frame(7, &rows, batch);
        // Header: layer, seq_len, flags, top_k — then batch×hidden f32s.
        assert_eq!(frame.len(), 16 + batch * hidden * 4);
        assert_eq!(u32::from_le_bytes(frame[0..4].try_into().unwrap()), 7);
        assert_eq!(
            u32::from_le_bytes(frame[4..8].try_into().unwrap()) as usize,
            batch
        );
        assert_eq!(
            u32::from_le_bytes(frame[12..16].try_into().unwrap()),
            0,
            "top_k must be 0 — non-zero engages the server L2 cache at seq_len==1"
        );
        // Row 1's first float lands after row 0.
        let v = f32::from_le_bytes(frame[16 + hidden * 4..20 + hidden * 4].try_into().unwrap());
        assert_eq!(v, hidden as f32 * 0.5);
    }

    #[test]
    fn q8k_frame_carries_one_entry_per_row_same_layer() {
        let hidden = 256; // one Q8K superblock
        let batch = 3;
        let rows: Vec<f32> = (0..batch * hidden)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        let frame = build_q8k_frame(9, &rows, batch, hidden);
        let entries = larql_inference::ffn::remote::decode_q8k_batch_request(&frame).unwrap();
        assert_eq!(entries.len(), batch);
        assert!(entries.iter().all(|e| e.layer_idx == 9));
        // Each entry must be that row's quantisation, not row 0 repeated.
        let q1 = quantize_x_to_q8k(&rows[hidden..2 * hidden]);
        assert_eq!(entries[1].q8k.qs, q1.qs);
        assert_eq!(entries[1].q8k.d, q1.d);
    }

    // ── summaries ──────────────────────────────────────────────────────

    fn sample(layer: usize, client_ms: f64, server_ms: Option<f64>) -> RequestSample {
        RequestSample {
            layer,
            client_ms,
            server_ms,
            bytes_sent: 100,
            bytes_recv: 60,
            served_wire: "f16".into(),
            any_nonzero: true,
        }
    }

    #[test]
    fn summarize_computes_throughput_payload_and_per_layer() {
        let point = SweepPoint {
            batch: 8,
            wire: WireArm::F16,
            dispatch: DispatchMode::Batch,
            endpoint: Endpoint::WalkFfn,
        };
        let stats = SweepPointStats {
            step_ms: vec![10.0, 20.0],
            samples: vec![
                sample(0, 4.0, Some(3.0)),
                sample(1, 6.0, Some(5.0)),
                sample(0, 8.0, Some(6.0)),
                sample(1, 12.0, Some(9.0)),
            ],
        };
        let s = summarize(&point, &stats, None, None);
        assert_eq!(s.batch, 8);
        assert_eq!(s.endpoint, "walk-ffn");
        assert_eq!(s.endpoint_code, 0);
        assert_eq!(s.wire_format, "f16");
        assert_eq!(s.dispatch_mode, "batch");
        assert_eq!(s.steps, 2);
        assert!((s.step_ms_mean - 15.0).abs() < 1e-9);
        // tok_s = 8 rows × 1000 / 15 ms
        assert!((s.tok_s - 8.0 * 1000.0 / 15.0).abs() < 1e-6);
        // payload = 4 samples × 160 bytes / (2 steps × 8 rows)
        assert!((s.payload_bytes_tok - (4.0 * 160.0) / 16.0).abs() < 1e-9);
        assert!(s.server_ms_p50.is_some());
        assert_eq!(s.per_layer.len(), 2);
        assert_eq!(s.per_layer[0].layer, 0);
        assert!(s.per_layer[1].client_ms_p99 >= s.per_layer[1].client_ms_p50);
    }

    #[test]
    fn summarize_handles_empty_and_no_server_ms() {
        let point = SweepPoint {
            batch: 1,
            wire: WireArm::Q8k,
            dispatch: DispatchMode::Streaming,
            endpoint: Endpoint::WalkFfnQ8k,
        };
        let s = summarize(&point, &SweepPointStats::default(), None, None);
        assert_eq!(s.tok_s, 0.0);
        assert_eq!(s.payload_bytes_tok, 0.0);
        assert!(s.server_ms_p50.is_none());

        let stats = SweepPointStats {
            step_ms: vec![5.0],
            samples: vec![sample(0, 5.0, None)],
        };
        let s = summarize(&point, &stats, None, None);
        assert!(s.server_ms_p50.is_none(), "q8k has no embedded latency");
    }

    // ── movement ratio + weight bytes ──────────────────────────────────

    #[test]
    fn summarize_reports_served_wire_including_fallback() {
        // An i8-requested arm that the server served as f32 must say so —
        // the bandwidth number belongs to what was actually on the wire.
        let point = SweepPoint {
            batch: 1,
            wire: WireArm::I8,
            dispatch: DispatchMode::Streaming,
            endpoint: Endpoint::WalkFfn,
        };
        let mut s0 = sample(0, 1.0, Some(0.5));
        s0.served_wire = "f32".into();
        let mut s1 = sample(1, 1.0, Some(0.5));
        s1.served_wire = "f32".into();
        let stats = SweepPointStats {
            step_ms: vec![2.0],
            samples: vec![s0, s1],
        };
        let s = summarize(&point, &stats, None, None);
        assert_eq!(s.wire_format, "i8");
        assert_eq!(s.served_wire, vec!["f32".to_string()]);
    }

    #[test]
    fn wire_label_maps_known_cts_and_passes_through_unknown() {
        assert_eq!(
            wire_label_for_content_type("application/x-larql-ffn"),
            "f32"
        );
        assert_eq!(
            wire_label_for_content_type("application/x-larql-ffn-f16"),
            "f16"
        );
        assert_eq!(
            wire_label_for_content_type("application/x-larql-ffn-i8"),
            "i8"
        );
        assert_eq!(
            wire_label_for_content_type("application/x-larql-ffn-q8k-batch"),
            "q8k"
        );
        assert_eq!(wire_label_for_content_type("text/plain"), "text/plain");
    }

    #[test]
    fn movement_ratio_basic_and_guards() {
        assert_eq!(movement_ratio(1.0, 1000.0), Some(0.001));
        assert_eq!(movement_ratio(1.0, 0.0), None);
        assert_eq!(movement_ratio(-1.0, 10.0), None);
    }

    #[test]
    fn weight_bytes_per_token_sums_selected_layers() {
        let stats = serde_json::json!({
            "ffn_weights": {
                "per_layer_dense_bytes": [100, 200, null, 400],
            }
        });
        let (bytes, missing) = weight_bytes_per_token(&stats, &[0, 1, 2, 3]).unwrap();
        assert_eq!(bytes, 700.0);
        assert_eq!(missing, 1);

        let (bytes, missing) = weight_bytes_per_token(&stats, &[1]).unwrap();
        assert_eq!(bytes, 200.0);
        assert_eq!(missing, 0);

        assert!(weight_bytes_per_token(&serde_json::json!({}), &[0]).is_none());
    }

    #[test]
    fn parse_layer_range_default_and_bounds() {
        assert_eq!(parse_layer_range(None, 3).unwrap(), vec![0, 1, 2]);
        assert_eq!(parse_layer_range(Some("1-2"), 4).unwrap(), vec![1, 2]);
        assert!(parse_layer_range(Some("2-1"), 4).is_err());
        assert!(parse_layer_range(Some("0-4"), 4).is_err());
        assert!(parse_layer_range(Some("x"), 4).is_err());
    }

    // ── endpoint seam ──────────────────────────────────────────────────

    #[test]
    fn endpoint_kind_parses_and_labels() {
        assert_eq!(
            EndpointKind::parse("walk-ffn").unwrap(),
            EndpointKind::WalkFfn
        );
        assert_eq!(
            EndpointKind::parse(" experts ").unwrap(),
            EndpointKind::Experts
        );
        assert!(EndpointKind::parse("expert").is_err());
        assert_eq!(EndpointKind::WalkFfn.label(), "walk-ffn");
        assert_eq!(EndpointKind::Experts.label(), "experts");
    }

    #[test]
    fn endpoint_resolution_maps_wire_axis() {
        use Endpoint::*;
        assert_eq!(
            Endpoint::resolve(EndpointKind::WalkFfn, WireArm::F32).unwrap(),
            WalkFfn
        );
        assert_eq!(
            Endpoint::resolve(EndpointKind::WalkFfn, WireArm::F16).unwrap(),
            WalkFfn,
            "f16/i8 are Accept negotiation on the same endpoint"
        );
        assert_eq!(
            Endpoint::resolve(EndpointKind::WalkFfn, WireArm::I8).unwrap(),
            WalkFfn
        );
        assert_eq!(
            Endpoint::resolve(EndpointKind::WalkFfn, WireArm::Q8k).unwrap(),
            WalkFfnQ8k
        );
        assert_eq!(
            Endpoint::resolve(EndpointKind::Experts, WireArm::F32).unwrap(),
            ExpertsMultiLayer
        );
        assert_eq!(
            Endpoint::resolve(EndpointKind::Experts, WireArm::Q8k).unwrap(),
            ExpertsMultiLayerQ8k
        );
        assert!(Endpoint::resolve(EndpointKind::Experts, WireArm::F16).is_err());
        assert!(Endpoint::resolve(EndpointKind::Experts, WireArm::I8).is_err());
    }

    #[test]
    fn endpoint_labels_codes_paths_and_behaviour() {
        use Endpoint::*;
        for (e, label, code) in [
            (WalkFfn, "walk-ffn", 0),
            (WalkFfnQ8k, "walk-ffn-q8k", 1),
            (ExpertsMultiLayer, "experts-ml", 2),
            (ExpertsMultiLayerQ8k, "experts-ml-q8k", 3),
        ] {
            assert_eq!(e.label(), label);
            assert_eq!(e.code(), code);
        }
        // Paths are the production constants.
        assert_eq!(WalkFfn.path(), WALK_FFN_PATH);
        assert_eq!(WalkFfnQ8k.path(), WALK_FFN_Q8K_PATH);
        assert_eq!(ExpertsMultiLayer.path(), MULTI_LAYER_BATCH_PATH);
        assert_eq!(ExpertsMultiLayerQ8k.path(), MULTI_LAYER_BATCH_Q8K_PATH);
        // Content types.
        assert_eq!(WalkFfn.request_content_type(), larql_inference::BINARY_CT);
        assert_eq!(
            WalkFfnQ8k.request_content_type(),
            larql_inference::Q8K_BATCH_CT
        );
        assert_eq!(
            ExpertsMultiLayer.request_content_type(),
            MULTI_LAYER_BATCH_CONTENT_TYPE
        );
        assert_eq!(
            ExpertsMultiLayerQ8k.request_content_type(),
            MULTI_LAYER_BATCH_Q8K_CONTENT_TYPE
        );
        // Accept: walk-ffn negotiates per wire arm; q8k walk-ffn none; both
        // multi-layer endpoints ask for the fixed f32 multi-layer CT.
        assert_eq!(WalkFfn.accept(WireArm::F16), WireArm::F16.accept());
        assert_eq!(WalkFfnQ8k.accept(WireArm::Q8k), None);
        assert_eq!(
            ExpertsMultiLayer.accept(WireArm::F32),
            Some(MULTI_LAYER_BATCH_CONTENT_TYPE)
        );
        assert_eq!(
            ExpertsMultiLayerQ8k.accept(WireArm::Q8k),
            Some(MULTI_LAYER_BATCH_CONTENT_TYPE)
        );
        // server_ms: only the f32 walk-ffn response embeds latency.
        assert!(WalkFfn.has_server_ms());
        assert!(!WalkFfnQ8k.has_server_ms());
        assert!(!ExpertsMultiLayer.has_server_ms());
        assert!(!ExpertsMultiLayerQ8k.has_server_ms());
        // Denominator source + routing requirement.
        assert_eq!(WalkFfn.denominator(), DenominatorSource::Dense);
        assert_eq!(WalkFfnQ8k.denominator(), DenominatorSource::Dense);
        assert_eq!(
            ExpertsMultiLayer.denominator(),
            DenominatorSource::RoutedExperts
        );
        assert_eq!(
            ExpertsMultiLayerQ8k.denominator(),
            DenominatorSource::RoutedExperts
        );
        assert!(!WalkFfn.requires_routing());
        assert!(ExpertsMultiLayer.requires_routing());
        assert!(ExpertsMultiLayerQ8k.requires_routing());
    }

    #[test]
    fn wire_label_maps_multi_layer_cts() {
        assert_eq!(
            wire_label_for_content_type(MULTI_LAYER_BATCH_CONTENT_TYPE),
            "experts-ml"
        );
        assert_eq!(
            wire_label_for_content_type(MULTI_LAYER_BATCH_Q8K_CONTENT_TYPE),
            "experts-ml-q8k"
        );
    }

    // ── routed frames (round-trip vs the production decoders) ──────────

    #[test]
    fn experts_ml_frame_round_trips_through_production_decoder() {
        let hidden = 4;
        let batch = 3;
        let rows: Vec<f32> = (0..batch * hidden).map(|i| i as f32 * 0.25).collect();
        let routing = vec![
            vec![(5u32, 0.75f32), (9u32, 0.25f32)],
            vec![(1u32, 1.0f32)],
            vec![], // sentinel-only record → zero-expert task
        ];
        let frame = build_experts_ml_frame(7, &rows, &routing, batch, hidden);
        let tasks = larql_inference::ffn::moe_remote::decode_multi_layer_request(&frame).unwrap();
        assert_eq!(tasks.len(), batch);
        for (r, t) in tasks.iter().enumerate() {
            assert_eq!(t.layer, 7);
            assert_eq!(t.residual, rows[r * hidden..(r + 1) * hidden].to_vec());
            let ids: Vec<u32> = routing[r].iter().map(|&(id, _)| id).collect();
            let wts: Vec<f32> = routing[r].iter().map(|&(_, w)| w).collect();
            assert_eq!(t.expert_ids, ids);
            assert_eq!(t.weights, wts);
        }
    }

    #[test]
    fn experts_ml_q8k_frame_round_trips_through_production_decoder() {
        let hidden = 256; // one Q8K superblock
        let batch = 2;
        let rows: Vec<f32> = (0..batch * hidden)
            .map(|i| (i as f32 * 0.013).sin())
            .collect();
        let routing = vec![vec![(3u32, 0.6f32), (7u32, 0.4f32)], vec![(42u32, 1.0f32)]];
        let frame = build_experts_ml_q8k_frame(11, &rows, &routing, batch, hidden);
        let tasks =
            larql_inference::ffn::moe_remote::decode_multi_layer_request_q8k(&frame).unwrap();
        assert_eq!(tasks.len(), batch);
        for (r, t) in tasks.iter().enumerate() {
            assert_eq!(t.layer, 11);
            assert_eq!(t.hidden, hidden);
            // Each task must be that row's quantisation, not row 0 repeated.
            let q = quantize_x_to_q8k(&rows[r * hidden..(r + 1) * hidden]);
            assert_eq!(t.qs, q.qs);
            assert_eq!(t.d, q.d);
            assert_eq!(t.sums, q.sums);
            let ids: Vec<u32> = routing[r].iter().map(|&(id, _)| id).collect();
            assert_eq!(t.expert_ids, ids);
            let wts: Vec<f32> = routing[r].iter().map(|&(_, w)| w).collect();
            assert_eq!(t.weights, wts);
        }
    }

    #[test]
    fn check_experts_response_validates_shape_and_flags_nonzero() {
        use larql_inference::ffn::moe_remote::{encode_multi_layer_response, MultiLayerResult};
        let hidden = 4;
        let mk = |vals: Vec<Vec<f32>>, layer: usize| {
            encode_multi_layer_response(
                &vals
                    .into_iter()
                    .map(|h2| MultiLayerResult { layer, h2 })
                    .collect::<Vec<_>>(),
            )
        };
        // Healthy: right batch/layer/hidden, non-zero values.
        let body = mk(vec![vec![0.0, 1.5, 0.0, 0.0], vec![0.0; 4]], 3);
        assert!(check_experts_response(&body, 3, 2, hidden).unwrap());
        // All-zero: decodes fine but flags the audit §1a corruption class.
        let body = mk(vec![vec![0.0; 4], vec![0.0; 4]], 3);
        assert!(!check_experts_response(&body, 3, 2, hidden).unwrap());
        // Wrong batch.
        let body = mk(vec![vec![1.0; 4]], 3);
        assert!(check_experts_response(&body, 3, 2, hidden).is_err());
        // Wrong layer.
        let body = mk(vec![vec![1.0; 4], vec![1.0; 4]], 5);
        assert!(check_experts_response(&body, 3, 2, hidden).is_err());
        // Wrong hidden.
        let body = mk(vec![vec![1.0; 3], vec![1.0; 3]], 3);
        assert!(check_experts_response(&body, 3, 2, hidden).is_err());
        // Garbage bytes.
        assert!(check_experts_response(&[1, 2, 3], 3, 2, hidden).is_err());
    }

    // ── pool-facing gather + layer subset ──────────────────────────────

    use super::super::capture_format::{CapturePool as Pool, RoutingCapture};

    /// Routed fixture pool: layer 0 routes (prompt p, step s) → experts
    /// {(p+s) % 3 @ 0.75, 3 @ 0.25}; layer 1 is non-MoE. Raw rows are the
    /// residual plane × 2; normed rows are × 0.5.
    fn routed_fixture(dir: &std::path::Path, prompts: usize, steps: usize, hidden: usize) {
        let layers = 2;
        let residuals: Vec<Vec<Vec<Vec<f32>>>> = (0..prompts)
            .map(|p| {
                (0..steps)
                    .map(|s| {
                        (0..layers)
                            .map(|l| {
                                (0..hidden)
                                    .map(|h| (p * 1000 + s * 100 + l * 10 + h) as f32)
                                    .collect()
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();
        let scale = |k: f32| -> Vec<Vec<Vec<Vec<f32>>>> {
            residuals
                .iter()
                .map(|p| {
                    p.iter()
                        .map(|s| {
                            s.iter()
                                .map(|l| l.iter().map(|v| v * k).collect())
                                .collect()
                        })
                        .collect()
                })
                .collect()
        };
        let routing: Vec<Vec<Vec<Vec<(u32, f32)>>>> = (0..prompts)
            .map(|p| {
                (0..steps)
                    .map(|s| {
                        (0..layers)
                            .map(|l| {
                                if l == 0 {
                                    vec![(((p + s) % 3) as u32, 0.75f32), (3u32, 0.25f32)]
                                } else {
                                    Vec::new()
                                }
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();
        let rc = RoutingCapture {
            top_k: 2,
            raw: scale(2.0),
            normed: scale(0.5),
            routing,
        };
        let texts: Vec<String> = (0..prompts).map(|i| format!("p{i}")).collect();
        Pool::write_with_routing(dir, "m", hidden, layers, &texts, &residuals, Some(&rc), 0)
            .unwrap();
    }

    #[test]
    fn routed_layer_subset_excludes_non_moe_layers() {
        let dir = tempfile::tempdir().unwrap();
        routed_fixture(dir.path(), 2, 2, 4);
        let pool = Pool::open(dir.path()).unwrap();
        assert_eq!(routed_layer_subset(&pool, &[0, 1]).unwrap(), vec![0]);
        assert_eq!(
            routed_layer_subset(&pool, &[1]).unwrap(),
            Vec::<usize>::new()
        );
        assert!(routed_layer_subset(&pool, &[0, 5]).is_err(), "oob layer");
    }

    #[test]
    fn routed_layer_subset_errs_on_walk_ffn_only_pool() {
        let dir = tempfile::tempdir().unwrap();
        let cap: Vec<Vec<Vec<Vec<f32>>>> = vec![vec![vec![vec![0.0; 4]; 2]; 1]];
        Pool::write(dir.path(), "m", 4, 2, &["p".into()], &cap, 0).unwrap();
        let pool = Pool::open(dir.path()).unwrap();
        assert!(routed_layer_subset(&pool, &[0]).is_err());
    }

    #[test]
    fn gather_frame_inputs_selects_the_endpoint_plane() {
        let dir = tempfile::tempdir().unwrap();
        routed_fixture(dir.path(), 2, 2, 4);
        let pool = Pool::open(dir.path()).unwrap();

        // Walk-ffn endpoints read the dense-prenormed residual plane.
        let dense = gather_frame_inputs(&pool, Endpoint::WalkFfn, 0, 1, 2).unwrap();
        assert_eq!(dense.rows, pool.rows(2, 1, 0).unwrap());
        assert!(dense.routing.is_none());
        let dense_q8k = gather_frame_inputs(&pool, Endpoint::WalkFfnQ8k, 0, 1, 2).unwrap();
        assert_eq!(dense_q8k.rows, pool.rows(2, 1, 0).unwrap());
        assert!(dense_q8k.routing.is_none());

        // Experts f32 reads raw rows (residual × 2 in this fixture).
        let f32_in = gather_frame_inputs(&pool, Endpoint::ExpertsMultiLayer, 0, 1, 2).unwrap();
        assert_eq!(f32_in.rows, pool.raw_rows(2, 1, 0).unwrap());
        for (raw, res) in f32_in.rows.iter().zip(dense.rows.iter()) {
            assert_eq!(*raw, res * 2.0);
        }
        // Experts q8k reads pre-experts-normed rows (× 0.5).
        let q8k_in = gather_frame_inputs(&pool, Endpoint::ExpertsMultiLayerQ8k, 0, 1, 2).unwrap();
        assert_eq!(q8k_in.rows, pool.normed_rows(2, 1, 0).unwrap());

        // Per-row routing: prompt 0 step 1 → (1, 0.75), prompt 1 → (2, 0.75).
        let routing = f32_in.routing.as_ref().unwrap();
        assert_eq!(routing.len(), 2);
        assert_eq!(routing[0], vec![(1u32, 0.75f32), (3u32, 0.25f32)]);
        assert_eq!(routing[1], vec![(2u32, 0.75f32), (3u32, 0.25f32)]);
        assert_eq!(q8k_in.routing.as_ref().unwrap(), routing);

        // Non-MoE layer: rows gather fine, routing rows are empty.
        let non_moe = gather_frame_inputs(&pool, Endpoint::ExpertsMultiLayer, 1, 0, 2).unwrap();
        assert!(non_moe.routing.as_ref().unwrap().iter().all(Vec::is_empty));

        // Walk-ffn-only pool: experts gather errors loudly.
        let dense_dir = tempfile::tempdir().unwrap();
        let cap: Vec<Vec<Vec<Vec<f32>>>> = vec![vec![vec![vec![0.0; 4]; 2]; 1]];
        Pool::write(dense_dir.path(), "m", 4, 2, &["p".into()], &cap, 0).unwrap();
        let dense_pool = Pool::open(dense_dir.path()).unwrap();
        assert!(gather_frame_inputs(&dense_pool, Endpoint::ExpertsMultiLayer, 0, 0, 1).is_err());
        assert!(gather_frame_inputs(&dense_pool, Endpoint::ExpertsMultiLayerQ8k, 0, 0, 1).is_err());
    }

    #[test]
    fn build_frame_dispatches_per_endpoint_and_pins_walk_ffn_bytes() {
        // hidden must be a Q8K super-block multiple (256): the q8k frame
        // builders quantise rows via `quantize_x_to_q8k`, which
        // debug-asserts 256-alignment — a toy width panics in debug builds.
        const HIDDEN: usize = 256;
        let dir = tempfile::tempdir().unwrap();
        routed_fixture(dir.path(), 2, 2, HIDDEN);
        let pool = Pool::open(dir.path()).unwrap();

        // Walk-ffn: byte-identical to the direct builder (pinned behaviour).
        let dense = gather_frame_inputs(&pool, Endpoint::WalkFfn, 0, 0, 2).unwrap();
        assert_eq!(
            build_frame(Endpoint::WalkFfn, &dense, 2, HIDDEN).unwrap(),
            build_walk_ffn_frame(0, &dense.rows, 2)
        );
        assert_eq!(
            build_frame(Endpoint::WalkFfnQ8k, &dense, 2, HIDDEN).unwrap(),
            build_q8k_frame(0, &dense.rows, 2, HIDDEN)
        );

        // Experts: frame carries per-row routing; missing routing is an error.
        let inputs = gather_frame_inputs(&pool, Endpoint::ExpertsMultiLayer, 0, 0, 2).unwrap();
        let frame = build_frame(Endpoint::ExpertsMultiLayer, &inputs, 2, HIDDEN).unwrap();
        let tasks = larql_inference::ffn::moe_remote::decode_multi_layer_request(&frame).unwrap();
        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[1].expert_ids, vec![1, 3]); // prompt 1 step 0 → (0+1)%3
        assert!(build_frame(Endpoint::ExpertsMultiLayer, &dense, 2, HIDDEN).is_err());
    }

    // ── routed denominators ────────────────────────────────────────────

    #[test]
    fn routed_weight_bytes_overlapping_and_disjoint_sets() {
        // One cell, two rows sharing one expert: naive counts 4, union 3.
        let cells = vec![vec![vec![1u32, 2], vec![2u32, 3]]];
        let d = routed_weight_bytes_per_token(&cells, 100.0, 1, 2);
        assert_eq!(d.weight_bytes_tok_naive, 4.0 * 100.0 / 2.0);
        assert_eq!(d.weight_bytes_tok_union, 3.0 * 100.0 / 2.0);

        // Disjoint sets: union == naive.
        let cells = vec![vec![vec![0u32, 1], vec![2u32, 3]]];
        let d = routed_weight_bytes_per_token(&cells, 10.0, 1, 2);
        assert_eq!(d.weight_bytes_tok_naive, d.weight_bytes_tok_union);
        assert_eq!(d.weight_bytes_tok_naive, 4.0 * 10.0 / 2.0);

        // Identical sets across rows: union collapses to one row's cost.
        let cells = vec![vec![vec![7u32, 9], vec![7u32, 9], vec![7u32, 9]]];
        let d = routed_weight_bytes_per_token(&cells, 10.0, 1, 3);
        assert_eq!(d.weight_bytes_tok_naive, 6.0 * 10.0 / 3.0);
        assert_eq!(d.weight_bytes_tok_union, 2.0 * 10.0 / 3.0);
    }

    #[test]
    fn routed_weight_bytes_sentinel_stripped_short_records_and_multi_cell() {
        // Sentinel-stripped short records: row lengths differ (a stripped
        // zero-weight pair), and one row is empty.
        let cells = vec![
            vec![vec![1u32, 2], vec![5u32]], // step 0, layer A
            vec![vec![2u32], vec![]],        // step 1, layer A
        ];
        let d = routed_weight_bytes_per_token(&cells, 8.0, 2, 2);
        // naive = (2 + 1) + (1 + 0) = 4 experts over 2 steps × 2 rows.
        assert_eq!(d.weight_bytes_tok_naive, 4.0 * 8.0 / 4.0);
        // union = |{1,2,5}| + |{2}| = 4.
        assert_eq!(d.weight_bytes_tok_union, 4.0 * 8.0 / 4.0);

        // Degenerate: zero tokens → zeros, no NaN.
        let d = routed_weight_bytes_per_token(&[], 8.0, 0, 0);
        assert_eq!(d.weight_bytes_tok_naive, 0.0);
        assert_eq!(d.weight_bytes_tok_union, 0.0);
    }

    #[test]
    fn routed_denominators_for_point_matches_hand_count() {
        let dir = tempfile::tempdir().unwrap();
        routed_fixture(dir.path(), 3, 2, 4);
        let pool = Pool::open(dir.path()).unwrap();
        let peb = 50.0;
        // Layer 0 only (the MoE subset); batch 3, steps 2.
        // Every row has 2 experts → naive = 2 × 3 rows × 2 steps = 12.
        // Union per (step, layer-0) cell: experts {(p+s)%3 for p} ∪ {3} —
        // step 0: {0,1,2,3} = 4; step 1: {1,2,0,3} = 4 → union total 8.
        let d = routed_denominators_for_point(&pool, &[0], 2, 3, peb).unwrap();
        assert_eq!(d.weight_bytes_tok_naive, 12.0 * peb / 6.0);
        assert_eq!(d.weight_bytes_tok_union, 8.0 * peb / 6.0);

        // Smaller batch narrows the union.
        let d1 = routed_denominators_for_point(&pool, &[0], 2, 1, peb).unwrap();
        assert_eq!(d1.weight_bytes_tok_naive, 4.0 * peb / 2.0);
        assert_eq!(d1.weight_bytes_tok_union, 4.0 * peb / 2.0);
    }

    #[test]
    fn moe_weight_stats_parses_and_errs_loudly() {
        let stats = serde_json::json!({
            "ffn_weights": {
                "per_layer_dense_bytes": [null, null],
                "moe": {
                    "num_experts": 128,
                    "top_k": 4,
                    "per_expert_bytes": 123456,
                }
            }
        });
        let m = moe_weight_stats(&stats).unwrap();
        assert_eq!(m.per_expert_bytes, 123456.0);
        assert_eq!(m.top_k, Some(4));
        assert_eq!(m.num_experts, Some(128));

        // Null per_expert_bytes → loud error (audit §1c).
        let stats = serde_json::json!({
            "ffn_weights": { "moe": { "per_expert_bytes": null, "top_k": 4 } }
        });
        assert!(moe_weight_stats(&stats)
            .unwrap_err()
            .contains("per_expert_bytes"));

        // No moe block (dense server) and no ffn_weights at all.
        let stats = serde_json::json!({ "ffn_weights": { "moe": null } });
        assert!(moe_weight_stats(&stats).is_err());
        assert!(moe_weight_stats(&serde_json::json!({})).is_err());
    }

    #[test]
    fn summarize_fills_routed_denominators_and_dense_weight_bytes() {
        let mk_stats = || SweepPointStats {
            step_ms: vec![10.0],
            samples: vec![sample(0, 10.0, None)],
        };
        // Routed point: naive is the primary movement denominator.
        let point = SweepPoint {
            batch: 1,
            wire: WireArm::F32,
            dispatch: DispatchMode::Streaming,
            endpoint: Endpoint::ExpertsMultiLayer,
        };
        let routed = RoutedDenominators {
            weight_bytes_tok_naive: 1600.0,
            weight_bytes_tok_union: 400.0,
        };
        let s = summarize(&point, &mk_stats(), Some(9999.0), Some(&routed));
        assert_eq!(s.endpoint, "experts-ml");
        assert_eq!(s.endpoint_code, 2);
        assert_eq!(s.weight_bytes_tok, None, "dense denom ignored on routed");
        assert_eq!(s.weight_bytes_tok_naive, Some(1600.0));
        assert_eq!(s.weight_bytes_tok_union, Some(400.0));
        // payload = 160 B/tok; ratio uses NAIVE.
        assert_eq!(s.movement_ratio, Some(160.0 / 1600.0));
        assert_eq!(s.experts_union_frac, Some(0.25));

        // Dense point: weight_bytes_tok carried per-point, routed fields absent.
        let point = SweepPoint {
            batch: 1,
            wire: WireArm::F32,
            dispatch: DispatchMode::Streaming,
            endpoint: Endpoint::WalkFfn,
        };
        let s = summarize(&point, &mk_stats(), Some(3200.0), Some(&routed));
        assert_eq!(s.weight_bytes_tok, Some(3200.0));
        assert_eq!(s.weight_bytes_tok_naive, None);
        assert_eq!(s.weight_bytes_tok_union, None);
        assert_eq!(s.experts_union_frac, None);
        assert_eq!(s.movement_ratio, Some(160.0 / 3200.0));

        // Dense point without stats: no denominator keys at all.
        let s = summarize(&point, &mk_stats(), None, None);
        assert_eq!(s.weight_bytes_tok, None);
        assert_eq!(s.movement_ratio, None);
        let json = serde_json::to_value(&s).unwrap();
        assert!(json.get("weight_bytes_tok").is_none(), "omitted when None");
        assert!(json.get("movement_ratio").is_none());
    }
}
