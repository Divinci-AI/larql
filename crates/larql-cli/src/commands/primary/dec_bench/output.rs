//! DEC replay run record — the full-fidelity JSON sidecar written by
//! `--output-file` (the pulse JSONL is the compact rig-facing view).

use serde::Serialize;

use super::capture_format::CaptureManifest;
use super::replay::DecPointSummary;

#[derive(Serialize)]
pub struct DecBenchJsonResult {
    /// Unix seconds, as a string (matches ADR-0012 bench JSON).
    pub timestamp: String,
    /// Which server endpoint the sweep measured. `walk-ffn` exercises the
    /// dense/shared-expert FFN path; routed experts are a separate arm.
    pub endpoint: String,
    pub ffn_url: String,
    pub capture: CaptureSummary,
    /// `/v1/stats` echo taken before the sweep.
    pub stats_before: serde_json::Value,
    /// `/v1/stats` echo taken after the sweep (includes the server's own
    /// `layer_latency` EMA/p99 accumulated over the run).
    pub stats_after: serde_json::Value,
    /// Layers replayed.
    pub layers: Vec<usize>,
    pub steps: usize,
    pub repeats: usize,
    pub warmup_passes: usize,
    /// Dense FFN weight bytes per token over `layers` (movement-ratio
    /// denominator for the measured endpoint); `None` when the server
    /// doesn't expose `ffn_weights`.
    pub weight_bytes_tok: Option<f64>,
    /// Layers whose dense byte count was unavailable in `/v1/stats` — a
    /// non-zero value flags a partial denominator.
    pub weight_bytes_missing_layers: usize,
    pub net_rtt_ms: Option<f64>,
    pub net_gbps: Option<f64>,
    pub points: Vec<DecPointSummary>,
}

/// Capture-pool identity embedded in the run record (not the full manifest —
/// prompt texts stay in the pool directory).
#[derive(Serialize)]
pub struct CaptureSummary {
    pub model: String,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub steps: usize,
    pub num_prompts: usize,
    pub created_unix: u64,
}

impl From<&CaptureManifest> for CaptureSummary {
    fn from(m: &CaptureManifest) -> Self {
        Self {
            model: m.model.clone(),
            hidden_size: m.hidden_size,
            num_layers: m.num_layers,
            steps: m.steps,
            num_prompts: m.prompts.len(),
            created_unix: m.created_unix,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::capture_format::{CaptureManifest, PromptMeta, CAPTURE_VERSION};
    use super::*;

    #[test]
    fn capture_summary_from_manifest_drops_prompt_texts() {
        let m = CaptureManifest {
            version: CAPTURE_VERSION,
            model: "gemma4-26b-a4b-q4k".into(),
            hidden_size: 5376,
            num_layers: 48,
            steps: 16,
            dtype: "f32-le".into(),
            prompts: vec![
                PromptMeta {
                    id: 0,
                    text: "secret-ish prompt text".into(),
                    steps_captured: 16,
                },
                PromptMeta {
                    id: 1,
                    text: "another".into(),
                    steps_captured: 20,
                },
            ],
            created_unix: 42,
        };
        let s = CaptureSummary::from(&m);
        assert_eq!(s.num_prompts, 2);
        assert_eq!(s.steps, 16);
        let json = serde_json::to_value(&s).unwrap();
        assert!(json.get("prompts").is_none());
        assert_eq!(json["hidden_size"], 5376);
    }

    #[test]
    fn run_record_serializes_with_schema_fields() {
        let m = CaptureManifest {
            version: CAPTURE_VERSION,
            model: "m".into(),
            hidden_size: 4,
            num_layers: 2,
            steps: 1,
            dtype: "f32-le".into(),
            prompts: vec![PromptMeta {
                id: 0,
                text: "p".into(),
                steps_captured: 1,
            }],
            created_unix: 0,
        };
        let r = DecBenchJsonResult {
            timestamp: "123".into(),
            endpoint: "walk-ffn".into(),
            ffn_url: "http://127.0.0.1:8080".into(),
            capture: CaptureSummary::from(&m),
            stats_before: serde_json::json!({"layers": 2}),
            stats_after: serde_json::json!({"layers": 2}),
            layers: vec![0, 1],
            steps: 1,
            repeats: 3,
            warmup_passes: 1,
            weight_bytes_tok: Some(1000.0),
            weight_bytes_missing_layers: 0,
            net_rtt_ms: None,
            net_gbps: None,
            points: vec![],
        };
        let v = serde_json::to_value(&r).unwrap();
        assert_eq!(v["endpoint"], "walk-ffn");
        assert_eq!(v["capture"]["num_prompts"], 1);
        assert_eq!(v["weight_bytes_tok"], 1000.0);
        assert!(v["points"].as_array().unwrap().is_empty());
    }
}
