//! `dec/*` pulse emission — JSONL lines consumable by the chuk-train rig.
//!
//! Contract (rig side): each line is one JSON object with a numeric `step`;
//! every other numeric field becomes a metric sample, non-numeric fields are
//! dropped. String axes (`dec/wire_format`, `dec/dispatch_mode`) are part of
//! the spec §7 schema and included for humans, with numeric `_code` twins so
//! the axes survive a numeric-only ingester.

use super::replay::DecPointSummary;

/// Build one pulse line for a sweep point. `step` is the sweep-point index.
#[allow(clippy::too_many_arguments)]
pub fn pulse_line(
    step: usize,
    s: &DecPointSummary,
    weight_bytes_tok: Option<f64>,
    movement_ratio: Option<f64>,
    net_rtt_ms: Option<f64>,
    net_gbps: Option<f64>,
    per_layer: bool,
) -> serde_json::Value {
    let mut obj = serde_json::Map::new();
    obj.insert("step".into(), step.into());
    obj.insert("dec/batch".into(), s.batch.into());
    obj.insert("dec/wire_format".into(), s.wire_format.clone().into());
    obj.insert("dec/wire_format_code".into(), s.wire_format_code.into());
    obj.insert("dec/dispatch_mode".into(), s.dispatch_mode.clone().into());
    obj.insert(
        "dec/dispatch_mode_code".into(),
        s.dispatch_mode_code.into(),
    );
    obj.insert("dec/step_ms_mean".into(), num(s.step_ms_mean));
    obj.insert("dec/step_ms_p50".into(), num(s.step_ms_p50));
    obj.insert("dec/step_ms_p99".into(), num(s.step_ms_p99));
    obj.insert("dec/tok_s".into(), num(s.tok_s));
    obj.insert("dec/payload_bytes_tok".into(), num(s.payload_bytes_tok));
    if let Some(w) = weight_bytes_tok {
        obj.insert("dec/weight_bytes_tok".into(), num(w));
    }
    if let Some(r) = movement_ratio {
        obj.insert("dec/movement_ratio".into(), num(r));
    }
    if let Some(p50) = s.server_ms_p50 {
        obj.insert("dec/server_ms_p50".into(), num(p50));
    }
    if let Some(p99) = s.server_ms_p99 {
        obj.insert("dec/server_ms_p99".into(), num(p99));
    }
    if let Some(rtt) = net_rtt_ms {
        obj.insert("net/rtt_ms".into(), num(rtt));
    }
    if let Some(gbps) = net_gbps {
        obj.insert("net/gbps".into(), num(gbps));
    }
    if per_layer {
        for l in &s.per_layer {
            obj.insert(
                format!("dec/layer{}_ms_p50", l.layer),
                num(l.client_ms_p50),
            );
            obj.insert(
                format!("dec/layer{}_ms_p99", l.layer),
                num(l.client_ms_p99),
            );
        }
    }
    serde_json::Value::Object(obj)
}

fn num(v: f64) -> serde_json::Value {
    serde_json::Number::from_f64(v)
        .map(serde_json::Value::Number)
        .unwrap_or(serde_json::Value::Null)
}

/// Serialise pulse lines as JSONL (one compact object per line, trailing
/// newline).
pub fn to_jsonl(lines: &[serde_json::Value]) -> String {
    let mut out = String::new();
    for l in lines {
        out.push_str(&l.to_string());
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::replay::{DecPointSummary, LayerSummary};
    use super::*;

    fn summary() -> DecPointSummary {
        DecPointSummary {
            batch: 8,
            wire_format: "f16".into(),
            wire_format_code: 1,
            served_wire: vec!["f16".into()],
            dispatch_mode: "batch".into(),
            dispatch_mode_code: 1,
            steps: 16,
            step_ms_mean: 12.5,
            step_ms_p50: 12.0,
            step_ms_p99: 20.0,
            tok_s: 640.0,
            payload_bytes_tok: 21504.0,
            server_ms_p50: Some(9.0),
            server_ms_p99: Some(15.0),
            per_layer: vec![LayerSummary {
                layer: 3,
                client_ms_p50: 0.4,
                client_ms_p99: 0.9,
            }],
        }
    }

    #[test]
    fn pulse_line_has_numeric_step_and_schema_keys() {
        let line = pulse_line(
            7,
            &summary(),
            Some(2.0e9),
            Some(1.1e-5),
            Some(0.05),
            Some(10.0),
            false,
        );
        assert_eq!(line["step"], 7);
        assert_eq!(line["dec/batch"], 8);
        assert_eq!(line["dec/wire_format"], "f16");
        assert_eq!(line["dec/wire_format_code"], 1);
        assert_eq!(line["dec/dispatch_mode_code"], 1);
        assert!(line["dec/step_ms_p50"].is_number());
        assert!(line["dec/step_ms_p99"].is_number());
        assert!(line["dec/tok_s"].is_number());
        assert!(line["dec/payload_bytes_tok"].is_number());
        assert!(line["dec/weight_bytes_tok"].is_number());
        assert!(line["dec/movement_ratio"].is_number());
        assert!(line["dec/server_ms_p50"].is_number());
        assert!(line["net/rtt_ms"].is_number());
        assert!(line["net/gbps"].is_number());
        // Per-layer keys only under the flag.
        assert!(line.get("dec/layer3_ms_p50").is_none());
    }

    #[test]
    fn pulse_line_optional_fields_omitted_and_per_layer_gated() {
        let mut s = summary();
        s.server_ms_p50 = None;
        s.server_ms_p99 = None;
        let line = pulse_line(0, &s, None, None, None, None, true);
        assert!(line.get("dec/weight_bytes_tok").is_none());
        assert!(line.get("dec/movement_ratio").is_none());
        assert!(line.get("dec/server_ms_p50").is_none());
        assert!(line.get("net/rtt_ms").is_none());
        assert!(line["dec/layer3_ms_p50"].is_number());
        assert!(line["dec/layer3_ms_p99"].is_number());
    }

    #[test]
    fn to_jsonl_one_compact_line_each() {
        let lines = vec![
            pulse_line(0, &summary(), None, None, None, None, false),
            pulse_line(1, &summary(), None, None, None, None, false),
        ];
        let jsonl = to_jsonl(&lines);
        let rows: Vec<&str> = jsonl.trim_end().split('\n').collect();
        assert_eq!(rows.len(), 2);
        for (i, row) in rows.iter().enumerate() {
            let v: serde_json::Value = serde_json::from_str(row).unwrap();
            assert_eq!(v["step"], i);
        }
        assert!(jsonl.ends_with('\n'));
    }
}
