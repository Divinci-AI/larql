//! Pure replay logic for the DEC loadgen: sweep-plan expansion, wire-frame
//! construction, and per-point statistics. Everything transport-shaped lives
//! in `replay_runtime.rs`; every byte on the wire goes through the SAME codec
//! functions the production client uses (parity discipline).

use larql_compute::cpu::ops::q4k_q8k_dot::quantize_x_to_q8k;
use larql_inference::{encode_binary_request, encode_q8k_batch_request};

use super::super::bench::row::compute_percentiles;

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

/// One point of the sweep: (batch, wire, dispatch).
#[derive(Debug, Clone, Copy)]
pub struct SweepPoint {
    pub batch: usize,
    pub wire: WireArm,
    pub dispatch: DispatchMode,
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
/// batch size run adjacently (comparable thermal window).
pub fn expand_sweep(
    batches: &[usize],
    wires: &[WireArm],
    dispatches: &[DispatchMode],
) -> Vec<SweepPoint> {
    let mut out = Vec::with_capacity(batches.len() * wires.len() * dispatches.len());
    for &batch in batches {
        for &wire in wires {
            for &dispatch in dispatches {
                out.push(SweepPoint {
                    batch,
                    wire,
                    dispatch,
                });
            }
        }
    }
    out
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

/// Short wire label for a response content-type (unknown CTs pass through
/// verbatim so the run record never hides what the server sent).
pub fn wire_label_for_content_type(ct: &str) -> String {
    match ct {
        larql_inference::BINARY_CT => "f32".into(),
        larql_inference::F16_CT => "f16".into(),
        larql_inference::I8_CT => "i8".into(),
        larql_inference::Q8K_BATCH_CT => "q8k".into(),
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
    pub server_ms_p50: Option<f64>,
    pub server_ms_p99: Option<f64>,
    pub per_layer: Vec<LayerSummary>,
}

pub fn summarize(point: &SweepPoint, stats: &SweepPointStats) -> DecPointSummary {
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

    DecPointSummary {
        batch: point.batch,
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
pub fn weight_bytes_per_token(
    stats: &serde_json::Value,
    layers: &[usize],
) -> Option<(f64, usize)> {
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
        );
        assert_eq!(points.len(), 4);
        assert_eq!(points[0].batch, 1);
        assert_eq!(points[1].batch, 1);
        assert_eq!(points[2].batch, 8);
        assert_eq!(points[1].wire, WireArm::F16);
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
        let rows: Vec<f32> = (0..batch * hidden).map(|i| (i as f32 * 0.01).sin()).collect();
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
        }
    }

    #[test]
    fn summarize_computes_throughput_payload_and_per_layer() {
        let point = SweepPoint {
            batch: 8,
            wire: WireArm::F16,
            dispatch: DispatchMode::Batch,
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
        let s = summarize(&point, &stats);
        assert_eq!(s.batch, 8);
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
        };
        let s = summarize(&point, &SweepPointStats::default());
        assert_eq!(s.tok_s, 0.0);
        assert_eq!(s.payload_bytes_tok, 0.0);
        assert!(s.server_ms_p50.is_none());

        let stats = SweepPointStats {
            step_ms: vec![5.0],
            samples: vec![sample(0, 5.0, None)],
        };
        let s = summarize(&point, &stats);
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
        };
        let mut s0 = sample(0, 1.0, Some(0.5));
        s0.served_wire = "f32".into();
        let mut s1 = sample(1, 1.0, Some(0.5));
        s1.served_wire = "f32".into();
        let stats = SweepPointStats {
            step_ms: vec![2.0],
            samples: vec![s0, s1],
        };
        let s = summarize(&point, &stats);
        assert_eq!(s.wire_format, "i8");
        assert_eq!(s.served_wire, vec!["f32".to_string()]);
    }

    #[test]
    fn wire_label_maps_known_cts_and_passes_through_unknown() {
        assert_eq!(wire_label_for_content_type("application/x-larql-ffn"), "f32");
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
}
