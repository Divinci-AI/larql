//! Python bindings for the mechanistic fact-editing pipeline
//! (`crown`, `edit`, `apply_patch`, `memit`).
//!
//! Exposes the CLI operations from RFC-0001 as one-liner callables so the
//! Chapter 15–23 Python Colab experiments can invoke the Rust-native
//! implementations directly. Phase D of RFC-0001.

use std::path::PathBuf;

use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyDict, PyList};
use pyo3::exceptions::{PyRuntimeError, PyValueError};

use larql_inference::{
    edit::{apply_patch as apply_patch_rust, compute_dense, compute_rank1, read_patch,
            write_patch, EditPatch, PatchProvenance},
    forward::{capture_ffn_activation_matrix, predict, predict_with_ffn},
    forward::memit::{run_memit, MemitFact},
    InferenceModel, LastPositionAblatingFfn, LastPositionInjectingFfn, WeightFfn,
};
use larql_inference::ndarray::Array1;

// ── Helpers ─────────────────────────────────────────────────────────

fn py_err<E: ToString>(e: E) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

fn tokenize(model: &InferenceModel, text: &str) -> PyResult<Vec<u32>> {
    let enc = model
        .tokenizer()
        .encode(text, true)
        .map_err(|e| py_err(format!("tokenize error: {e}")))?;
    Ok(enc.get_ids().to_vec())
}

fn prob_of(preds: &[(String, f64)], target: &str) -> f64 {
    for (tok, prob) in preds {
        if tok.trim().eq_ignore_ascii_case(target.trim()) {
            return *prob;
        }
    }
    0.0
}

fn scan_crown(
    model: &InferenceModel,
    tokens: &[u32],
    expect: &str,
    start: usize,
    end: usize,
    top_k: usize,
) -> Vec<(usize, f64, String, f64, bool)> {
    let weights = model.weights();
    let weight_ffn = WeightFfn { weights };
    let baseline = predict(weights, model.tokenizer(), tokens, top_k);
    let baseline_expect = prob_of(&baseline.predictions, expect);
    let mut out = Vec::new();
    for layer in start..=end {
        let ffn = LastPositionAblatingFfn::new(&weight_ffn, layer);
        let r = predict_with_ffn(weights, model.tokenizer(), tokens, top_k, &ffn);
        let top = r
            .predictions
            .first()
            .map(|(t, _)| t.trim().to_string())
            .unwrap_or_default();
        let top_prob = r.predictions.first().map(|(_, p)| *p).unwrap_or(0.0);
        let expect_prob = prob_of(&r.predictions, expect);
        let flipped = !top.eq_ignore_ascii_case(expect.trim());
        out.push((layer, expect_prob - baseline_expect, top, top_prob, flipped));
    }
    let _ = baseline; // silence unused warning on some paths
    out
}

fn pick_crown(scan: &[(usize, f64, String, f64, bool)]) -> Option<usize> {
    scan.iter()
        .filter(|r| r.4)
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|r| r.0)
        .or_else(|| {
            scan.iter()
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|r| r.0)
        })
}

// ── Python-facing functions ─────────────────────────────────────────

/// Find the crown MLP layer for a (prompt, expected-token) pair.
///
/// Returns a dict:
///   { "crown_layer": int, "crown_delta_prob": float,
///     "top_after_ablation": str, "scan": [{layer, delta, top, flipped}, ...] }
#[pyfunction]
#[pyo3(signature = (model, prompt, expect, start_layer=None, end_layer=None, top_k=100))]
pub fn crown(
    py: Python<'_>,
    model: &str,
    prompt: &str,
    expect: &str,
    start_layer: Option<usize>,
    end_layer: Option<usize>,
    top_k: usize,
) -> PyResult<Py<PyDict>> {
    let m = InferenceModel::load(model).map_err(py_err)?;
    let n = m.num_layers();
    let start = start_layer.unwrap_or((n * 3) / 5);
    let end = end_layer.unwrap_or(n.saturating_sub(2));
    if start > end {
        return Err(PyValueError::new_err("start_layer must be <= end_layer"));
    }
    let tokens = tokenize(&m, prompt)?;
    let scan = scan_crown(&m, &tokens, expect, start, end, top_k);

    let crown_layer = pick_crown(&scan);
    let crown_delta = crown_layer
        .and_then(|c| scan.iter().find(|r| r.0 == c).map(|r| r.1));
    let crown_top = crown_layer
        .and_then(|c| scan.iter().find(|r| r.0 == c).map(|r| r.2.clone()));

    let out = PyDict::new(py);
    out.set_item("crown_layer", crown_layer)?;
    out.set_item("crown_delta_prob", crown_delta)?;
    out.set_item("top_after_ablation", crown_top)?;

    let scan_list = PyList::empty(py);
    for (layer, delta, top, top_prob, flipped) in &scan {
        let row = PyDict::new(py);
        row.set_item("layer", *layer)?;
        row.set_item("delta_expect_prob", *delta)?;
        row.set_item("top", top)?;
        row.set_item("top_prob", *top_prob)?;
        row.set_item("flipped", *flipped)?;
        scan_list.append(row)?;
    }
    out.set_item("scan", scan_list)?;
    Ok(out.into())
}

/// Compute and write a rank-1 `.lqpatch` that makes `src` predict `new_token`.
///
/// Parameters mirror `larql edit`:
///   model: model path or HF id
///   src / tgt: source and target prompts (target gives the desired direction)
///   new_token: token string (e.g., " Tokyo") — used by the scale search
///   output: path to write the .lqpatch file
///   layer: explicit crown layer (None = auto-discover)
///   scales: list of scales to try (None = [0.5, 1, 1.5, 2, 2.5, 3, 4])
///   fixed_scale: skip the search and use this scale exactly
///
/// Returns a dict: { "layer": int, "scale": float, "output": str, "d_norm": float }
#[pyfunction]
#[pyo3(signature = (model, src, tgt, new_token, output, layer=None, scales=None, fixed_scale=None, top_k=100, label=None))]
pub fn edit(
    py: Python<'_>,
    model: &str,
    src: &str,
    tgt: &str,
    new_token: &str,
    output: &str,
    layer: Option<usize>,
    scales: Option<Vec<f32>>,
    fixed_scale: Option<f32>,
    top_k: usize,
    label: Option<&str>,
) -> PyResult<Py<PyDict>> {
    let m = InferenceModel::load(model).map_err(py_err)?;
    let weights = m.weights();
    let hidden = weights.hidden_size;
    let intermediate = weights.intermediate_size;

    let src_tokens = tokenize(&m, src)?;
    let tgt_tokens = tokenize(&m, tgt)?;

    let chosen_layer = match layer {
        Some(l) => l,
        None => {
            let n = m.num_layers();
            let scan = scan_crown(&m, &src_tokens, new_token.trim(), (n * 3) / 5,
                                   n.saturating_sub(2), top_k);
            pick_crown(&scan)
                .ok_or_else(|| py_err("crown scan returned no candidate layer"))?
        }
    };

    let act_src = capture_ffn_activation_matrix(weights, &src_tokens, chosen_layer)
        .ok_or_else(|| py_err(format!("capture failed for src at L{chosen_layer}")))?;
    let act_tgt = capture_ffn_activation_matrix(weights, &tgt_tokens, chosen_layer)
        .ok_or_else(|| py_err(format!("capture failed for tgt at L{chosen_layer}")))?;
    let k_src = act_src.row(act_src.shape()[0] - 1).to_owned();
    let k_tgt = act_tgt.row(act_tgt.shape()[0] - 1).to_owned();
    if k_src.len() != intermediate || k_tgt.len() != intermediate {
        return Err(py_err("intermediate-size mismatch in captured keys"));
    }

    // d_base = W_down @ (k_tgt - k_src)
    let w_key = weights.arch.ffn_down_key(chosen_layer);
    let w_down = weights
        .tensors
        .get(&w_key)
        .ok_or_else(|| py_err(format!("W_down missing at {w_key}")))?;
    let k_diff: Array1<f32> = &k_tgt - &k_src;
    let w_view = w_down.view();
    let d_base: Array1<f32> = if w_down.shape() == [hidden, intermediate] {
        w_view.dot(&k_diff)
    } else if w_down.shape() == [intermediate, hidden] {
        k_diff.view().dot(&w_view)
    } else {
        return Err(py_err(format!("unexpected W_down shape {:?}", w_down.shape())));
    };
    let d_base_vec = d_base.to_vec();

    // Scale search.
    let chosen_scale = if let Some(s) = fixed_scale {
        s
    } else {
        let grid = scales.unwrap_or_else(|| vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]);
        let weight_ffn = WeightFfn { weights };
        let mut chosen: Option<f32> = None;
        for &s in &grid {
            let scaled: Vec<f32> = d_base_vec.iter().map(|&v| v * s).collect();
            let ffn = LastPositionInjectingFfn::new(&weight_ffn, chosen_layer, scaled);
            let r = predict_with_ffn(weights, m.tokenizer(), &src_tokens, 5, &ffn);
            let top = r.predictions.first()
                .map(|(t, _)| t.trim().to_string())
                .unwrap_or_default();
            if top.eq_ignore_ascii_case(new_token.trim()) {
                chosen = Some(s);
                break;
            }
        }
        chosen.ok_or_else(|| py_err("scale search exhausted without flipping to new_token"))?
    };

    let provenance = PatchProvenance {
        src_prompt: src.to_string(),
        tgt_prompt: tgt.to_string(),
        old_token: String::new(),
        new_token: new_token.to_string(),
        crown_delta: 0.0,
        created_at: format!("epoch-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs()).unwrap_or(0)),
    };
    let patch = compute_rank1(&k_src.to_vec(), &d_base_vec, chosen_scale, chosen_layer, provenance);
    write_patch(output, &patch).map_err(py_err)?;
    let d_norm: f32 = patch.d.iter().map(|v| v * v).sum::<f32>().sqrt();

    let out = PyDict::new(py);
    out.set_item("layer", chosen_layer)?;
    out.set_item("scale", chosen_scale)?;
    out.set_item("output", output)?;
    out.set_item("d_norm", d_norm as f64)?;
    if let Some(l) = label { out.set_item("label", l)?; }
    Ok(out.into())
}

/// Apply one or more patches to a model in-memory and optionally run a test prompt.
///
/// patches: list of .lqpatch paths
/// prompt: optional prompt to predict after applying
/// reverse: subtract rather than add (verifies reversibility)
///
/// Returns dict with "predictions" (list of [token, prob]) when prompt given.
#[pyfunction]
#[pyo3(signature = (model, patches, prompt=None, top_k=5, reverse=false))]
pub fn apply_patch(
    py: Python<'_>,
    model: &str,
    patches: Vec<String>,
    prompt: Option<&str>,
    top_k: usize,
    reverse: bool,
) -> PyResult<Py<PyDict>> {
    let mut m = InferenceModel::load(model).map_err(py_err)?;
    for path in &patches {
        let mut patch: EditPatch = read_patch(path).map_err(py_err)?;
        if reverse {
            for v in patch.d.iter_mut() { *v = -*v; }
            for v in patch.delta_w.iter_mut() { *v = -*v; }
        }
        apply_patch_rust(m.weights_mut(), &patch).map_err(py_err)?;
    }

    let out = PyDict::new(py);
    out.set_item("patches_applied", patches.len())?;
    out.set_item("reversed", reverse)?;

    if let Some(p) = prompt {
        let tokens = tokenize(&m, p)?;
        let r = predict(m.weights(), m.tokenizer(), &tokens, top_k);
        let preds_list = PyList::empty(py);
        for (tok, prob) in &r.predictions {
            let row = PyList::empty(py);
            row.append(tok)?;
            row.append(*prob)?;
            preds_list.append(row)?;
        }
        out.set_item("predictions", preds_list)?;
    }
    Ok(out.into())
}

/// Batch fact edit via covariance-MEMIT. Wraps `larql memit`.
///
/// `edits` is a list of dicts: [{"label": str, "src": str, "new_token": str,
///                               "layer": int (optional)}, ...]
/// Writes one dense patch per affected layer into `output_dir` + a
/// `manifest.json`. Returns dict listing emitted patches.
#[pyfunction]
#[pyo3(signature = (model, edits, output_dir, ridge=0.01, target_alpha=1.0, top_k=100))]
pub fn memit(
    py: Python<'_>,
    model: &str,
    edits: &Bound<'_, PyList>,
    output_dir: &str,
    ridge: f64,
    target_alpha: f32,
    top_k: usize,
) -> PyResult<Py<PyDict>> {
    let m = InferenceModel::load(model).map_err(py_err)?;
    let weights = m.weights();

    let mut facts: Vec<MemitFact> = Vec::with_capacity(edits.len());
    for item in edits.iter() {
        let d = item.downcast::<PyDict>()?;
        let label: String = d.get_item("label")?.ok_or_else(|| PyValueError::new_err("missing label"))?.extract()?;
        let src: String = d.get_item("src")?.ok_or_else(|| PyValueError::new_err("missing src"))?.extract()?;
        let new_token: String = d.get_item("new_token")?.ok_or_else(|| PyValueError::new_err("missing new_token"))?.extract()?;
        let layer_opt: Option<usize> = match d.get_item("layer")? {
            Some(v) => v.extract().ok(),
            None => None,
        };

        let prompt_tokens = tokenize(&m, &src)?;
        let target_tokens = m.tokenizer().encode(new_token.as_str(), false)
            .map_err(|e| py_err(format!("tokenize target: {e}")))?
            .get_ids().to_vec();
        let target_token_id = *target_tokens.first()
            .ok_or_else(|| py_err("new_token tokenised to empty list"))?;

        let layer = match layer_opt {
            Some(l) => l,
            None => {
                let n = m.num_layers();
                let scan = scan_crown(&m, &prompt_tokens, new_token.trim(),
                                      (n * 3) / 5, n.saturating_sub(2), top_k);
                pick_crown(&scan)
                    .ok_or_else(|| py_err(format!("crown scan failed for {label}")))?
            }
        };
        facts.push(MemitFact { prompt_tokens, target_token_id, layer, label });
    }

    let results = run_memit(weights, &facts, ridge, target_alpha, m.tokenizer())
        .map_err(|e| py_err(format!("run_memit: {e}")))?;

    std::fs::create_dir_all(output_dir).map_err(py_err)?;

    let patches_list = PyList::empty(py);
    for result in &results {
        let prov = PatchProvenance {
            src_prompt: String::new(),
            tgt_prompt: String::new(),
            old_token: String::new(),
            new_token: format!("MEMIT batch ({} facts @ L{})", result.fact_results.len(), result.layer),
            crown_delta: 0.0,
            created_at: format!("epoch-{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs()).unwrap_or(0)),
        };
        let patch = compute_dense(&result.delta_w, result.layer, prov);
        let path = PathBuf::from(output_dir).join(format!("memit_L{}.lqpatch", result.layer));
        write_patch(&path, &patch).map_err(py_err)?;
        patches_list.append(path.display().to_string())?;
    }

    let out = PyDict::new(py);
    out.set_item("num_edits", facts.len())?;
    out.set_item("num_layers", results.len())?;
    out.set_item("patches", patches_list)?;
    Ok(out.into())
}
