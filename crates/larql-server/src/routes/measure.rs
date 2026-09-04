//! POST /v1/patches/apply_measured — apply a patch and bracket it with a
//! before/after reading, both taken inside ONE request on ONE instance.
//!
//! # Why this exists
//!
//! The product records, for every model edit, what that edit did to the
//! model. Until now it assembled that record out of seven separate HTTP
//! requests:
//!
//! ```text
//! gateBefore probeBefore distBefore   APPLY   gateAfter probeAfter distAfter
//! ```
//!
//! A patch overlay lives in one instance's memory, and larql-service runs up
//! to three instances with no session affinity (`minScale 1`, `maxScale 3`,
//! `containerConcurrency 1`). Nothing pins those seven requests to one
//! instance. When the "after" requests land somewhere that never saw the
//! apply, they read an unpatched model and the edit is filed as having
//! changed nothing — `0.0000 TV` for an edit that landed perfectly well.
//!
//! Read-repair cannot fix this. Hydration rebuilds whichever instance a
//! request arrives at, but apply and measure are still separate requests that
//! route independently, so there is always a pairing that measures the wrong
//! instance. The fix is to stop spreading one measurement across several
//! requests: read, mutate, and read again against the same overlay, in one
//! handler.
//!
//! # Scope of the guarantee
//!
//! This makes the bracket atomic *with respect to routing*, which is the
//! failure that has actually been observed in production. It does not take a
//! lock across the whole handler, because holding the sessions guard across
//! two multi-second forward passes would stall every other handler that
//! touches the map — the hazard `infer` and `describe` already document at
//! length.
//!
//! At `containerConcurrency: 1` that distinction is academic: an instance
//! serves one request at a time, so nothing can mutate the session between
//! the two readings. The manifest notes an intent to raise concurrency to
//! 4–8 later. When that happens, two requests against the *same* session
//! could interleave here, and this handler needs a per-session lock before
//! that lands. It is called out in the response as `atomic_scope` rather than
//! left as folklore.
//!
//! # A failed measurement never loses the edit
//!
//! The apply is the durable half; the reading is commentary on it. So a
//! measurement that cannot be taken — no model weights loaded, an untokenizable
//! prompt — is reported as `null` alongside `measure_error`, and the patch is
//! applied anyway. Returning 503 instead would mean an operator asking "apply
//! this and tell me what it did" gets neither, on a service whose inference
//! half can be disabled independently (`--no-infer`).
//!
//! `null` here means "not measured", and is deliberately distinguishable from
//! a measured zero. The product has already been burned once by a probe that
//! returned `0` for an absent target, making "no effect" and "no reading"
//! indistinguishable in a compliance dossier.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::HeaderMap;
use axum::Json;
use serde::Deserialize;

use crate::error::ServerError;
use crate::routes::describe::{describe_entity_with, DescribeParams};
use crate::session::extract_session_id;
use crate::state::{AppState, LoadedModel};

/// What to read on either side of the patch.
///
/// Every field beyond `prompt` is optional so a caller can ask for only the
/// half it can interpret. The distribution is always measurable; the gate
/// score needs an `entity` to DESCRIBE and a `target` to find within it.
#[derive(Deserialize)]
pub struct MeasureSpec {
    /// Prompt for the next-token distribution, in `walk` mode — the only
    /// mode that observes a patch at all.
    pub prompt: String,
    /// How many tokens of the distribution to return.
    #[serde(default = "default_top")]
    pub top: usize,
    /// Entity to DESCRIBE for the gate-score half. Omit to skip it.
    #[serde(default)]
    pub entity: Option<String>,
    /// Target token to read a gate score for out of the DESCRIBE edges.
    /// Omit (or pass an entity with no such edge) and `gate_score` is
    /// `null` — which means "not present", never `0.0`.
    #[serde(default)]
    pub target: Option<String>,
    #[serde(default = "default_band")]
    pub band: String,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default = "default_min_score")]
    pub min_score: f32,
}

fn default_top() -> usize {
    10
}
fn default_band() -> String {
    "knowledge".to_string()
}
fn default_limit() -> usize {
    20
}
fn default_min_score() -> f32 {
    5.0
}

#[derive(Deserialize)]
pub struct ApplyMeasuredRequest {
    /// Name to file the patch under, exactly as `POST /v1/patches/apply`.
    #[serde(default)]
    pub name: Option<String>,
    /// The patch to apply. Inline only — a measured apply is a foreground
    /// operation on a caller-built patch, so there is no `url` form.
    pub patch: larql_vindex::VindexPatch,
    /// What to read either side of it.
    pub measure: MeasureSpec,
}

/// One side of the bracket.
fn read_side(
    state: &AppState,
    model: &LoadedModel,
    session_id: Option<&str>,
    spec: &MeasureSpec,
) -> Result<serde_json::Value, ServerError> {
    // Same overlay-resolution rule as `infer` and `describe`: a session with
    // no overlay has never been patched, so it reads exactly like the global
    // state — the correct answer for an unknown or expired session, and the
    // reason an expired session degrades to "unpatched" rather than to an
    // error.
    let sessions = state.sessions.sessions_blocking_read();
    if let Some(patched) = session_id
        .and_then(|sid| sessions.get(sid))
        .and_then(|s| s.patched())
    {
        read_against(model, patched, spec)
    } else {
        drop(sessions);
        let patched = model.patched.blocking_read();
        read_against(model, &patched, spec)
    }
}

fn read_against(
    model: &LoadedModel,
    patched: &larql_vindex::PatchedVindex,
    spec: &MeasureSpec,
) -> Result<serde_json::Value, ServerError> {
    let weights_guard = model
        .get_or_load_weights()
        .map_err(ServerError::InferenceUnavailable)?;
    let weights: &larql_inference::ModelWeights = &weights_guard;

    let encoding = model
        .tokenizer
        .encode(spec.prompt.as_str(), true)
        .map_err(|e| ServerError::Internal(format!("tokenize error: {e}")))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    if token_ids.is_empty() {
        return Err(ServerError::BadRequest("empty prompt".into()));
    }

    // `walk` deliberately, not `dense`. Dense inference is never handed the
    // overlay, so measuring an edit with it would report no change for every
    // edit ever made — patch-blind by construction, not by accident.
    let pred = larql_inference::infer_patched(
        weights,
        &model.tokenizer,
        patched,
        Some(&patched.knn_store),
        &token_ids,
        spec.top,
        &larql_inference::KnnRouteMode::from_env(),
    );

    let predictions: Vec<serde_json::Value> = pred
        .predictions
        .iter()
        .map(|(tok, prob)| {
            serde_json::json!({
                "token": tok,
                "probability": (prob * 10000.0).round() / 10000.0,
            })
        })
        .collect();

    let gate_score = match (&spec.entity, &spec.target) {
        (Some(entity), Some(target)) => {
            let params = DescribeParams {
                entity: entity.clone(),
                band: spec.band.clone(),
                verbose: false,
                limit: spec.limit,
                min_score: spec.min_score,
                // Deliberately off. Measurement matches an edge by its target
                // string, and relabelling would rename the very thing being
                // matched — an edit would report "no change" because the label
                // moved, not because the gate did.
                coherence: false,
                min_coherence: 0.0,
                relabel: false,
                query: "embedding".into(),
                baseline: None,
            };
            let described = describe_entity_with(model, patched, &params)?;
            gate_score_for_target(&described, target)
        }
        _ => None,
    };

    Ok(serde_json::json!({
        "predictions": predictions,
        "gate_score": gate_score,
    }))
}

/// Pull one target's gate score out of a DESCRIBE response.
///
/// Returns `None` for "no such edge", which is the whole point of a suppressed
/// feature: after a successful DELETE the edge is *gone*, and reporting that as
/// `0.0` would be indistinguishable from an edge that is present and scores
/// zero. The product has already been bitten once by a probe that conflated
/// absent with zero; this keeps the two apart at the source.
fn gate_score_for_target(described: &serde_json::Value, target: &str) -> Option<f64> {
    let wanted = target.trim().to_lowercase();
    described
        .get("edges")?
        .as_array()?
        .iter()
        .find(|edge| {
            edge.get("target")
                .and_then(|t| t.as_str())
                .map(|t| t.trim().to_lowercase() == wanted)
                .unwrap_or(false)
        })
        .and_then(|edge| edge.get("gate_score"))
        .and_then(|g| g.as_f64())
}

/// Split a reading into (value, error-message).
///
/// A reading that fails becomes `(null, Some(why))` rather than an early
/// return, so the apply still happens. See the module docs.
fn split_reading(
    r: Result<serde_json::Value, ServerError>,
) -> (serde_json::Value, Option<String>) {
    match r {
        Ok(v) => (v, None),
        Err(e) => (serde_json::Value::Null, Some(e.to_string())),
    }
}

async fn apply_measured_to_model(
    state: &Arc<AppState>,
    model_id: Option<&str>,
    headers: &HeaderMap,
    req: ApplyMeasuredRequest,
) -> Result<Json<serde_json::Value>, ServerError> {
    let model = state.model_or_err(model_id)?.clone();
    let session_id = extract_session_id(headers);

    let name = req
        .name
        .clone()
        .or_else(|| req.patch.description.clone())
        .unwrap_or_else(|| "inline-patch".to_string());
    let mut patch = req.patch;
    // The patch stack keys on `description`, not on the name echoed back to
    // the caller. Filing under one and reporting the other is what once made
    // DELETE /v1/patches/{name} 404 for callers that had done nothing wrong.
    patch.description = Some(name.clone());
    crate::routes::patches::enrich_patch_ops(&model, &mut patch);
    let op_count = patch.operations.len();

    let spec = Arc::new(req.measure);

    // ── before ────────────────────────────────────────────────────────
    let (before, before_err) = {
        let (state, model, spec, sid) = (
            Arc::clone(state),
            Arc::clone(&model),
            Arc::clone(&spec),
            session_id.clone(),
        );
        let joined =
            tokio::task::spawn_blocking(move || read_side(&state, &model, sid.as_deref(), &spec))
                .await
                .map_err(|e| ServerError::Internal(e.to_string()))?;
        split_reading(joined)
    };

    // ── the mutation being measured ───────────────────────────────────
    let (ops, active) = if let Some(sid) = session_id.as_deref() {
        state.sessions.apply_patch(sid, &model, patch).await
    } else {
        let mut patched = model.patched.write().await;
        patched.apply_patch(patch);
        (op_count, patched.num_patches())
    };

    // ── after ─────────────────────────────────────────────────────────
    let (after, after_err) = {
        let (state, model, spec, sid) = (
            Arc::clone(state),
            Arc::clone(&model),
            Arc::clone(&spec),
            session_id.clone(),
        );
        let joined =
            tokio::task::spawn_blocking(move || read_side(&state, &model, sid.as_deref(), &spec))
                .await
                .map_err(|e| ServerError::Internal(e.to_string()))?;
        split_reading(joined)
    };

    Ok(Json(serde_json::json!({
        "applied": name,
        "operations": ops,
        "active_patches": active,
        "session": session_id,
        // Names what the caller may conclude from this pair. Both readings
        // came from one overlay on one instance, so a difference between them
        // is the patch and nothing else — no routing, no expiry, no other
        // instance. See the module docs for the concurrency caveat.
        "atomic_scope": "single-instance-single-request",
        "measured": {
            "prompt": spec.prompt,
            "before": before,
            "after": after,
            // Present only when a reading could not be taken. A caller must
            // treat a null side as "unmeasured", never as "unchanged".
            "measure_error": before_err.or(after_err),
        },
    })))
}

#[utoipa::path(
    post,
    path = "/v1/patches/apply_measured",
    tag = "patches",
    request_body(
        content = crate::openapi::schemas::ApplyMeasuredBody,
        description = "Inline `patch` plus a `measure` spec. Use `X-Session-Id` to scope \
                       the apply and both readings to one session.",
    ),
    responses(
        (status = 200, body = crate::openapi::schemas::ApplyMeasuredResponse),
        (status = 400, body = crate::error::ErrorBody),
        (status = 503, body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_apply_measured(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<ApplyMeasuredRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    apply_measured_to_model(&state, None, &headers, req).await
}

#[utoipa::path(
    post,
    path = "/v1/{model_id}/patches/apply_measured",
    tag = "patches",
    params(("model_id" = String, Path, description = "Id of a loaded vindex.")),
    request_body(content = crate::openapi::schemas::ApplyMeasuredBody),
    responses(
        (status = 200, body = crate::openapi::schemas::ApplyMeasuredResponse),
        (status = 404, body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_apply_measured_multi(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<String>,
    headers: HeaderMap,
    Json(req): Json<ApplyMeasuredRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    apply_measured_to_model(&state, Some(&model_id), &headers, req).await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn described(edges: serde_json::Value) -> serde_json::Value {
        serde_json::json!({ "entity": "Paris", "edges": edges })
    }

    #[test]
    fn reads_the_named_target_score() {
        let d = described(serde_json::json!([
            {"target": "Gates", "gate_score": 18.7, "layer": 26},
            {"target": "Istanbul", "gate_score": 9.4, "layer": 26},
        ]));
        assert_eq!(gate_score_for_target(&d, "Gates"), Some(18.7));
        assert_eq!(gate_score_for_target(&d, "Istanbul"), Some(9.4));
    }

    #[test]
    fn matches_case_and_whitespace_insensitively() {
        let d = described(serde_json::json!([{"target": " Gates", "gate_score": 18.7}]));
        assert_eq!(gate_score_for_target(&d, "gates"), Some(18.7));
        assert_eq!(gate_score_for_target(&d, "  GATES "), Some(18.7));
    }

    // The distinction the whole measurement rests on. A suppressed edge is
    // absent, and absent is not zero: a caller that sees 0.0 concludes the
    // edge is still there scoring nothing, which is the opposite of what a
    // successful DELETE means.
    #[test]
    fn absent_target_is_none_not_zero() {
        let d = described(serde_json::json!([{"target": "Istanbul", "gate_score": 9.4}]));
        assert_eq!(gate_score_for_target(&d, "Gates"), None);
    }

    #[test]
    fn genuine_zero_survives_as_zero() {
        let d = described(serde_json::json!([{"target": "Gates", "gate_score": 0.0}]));
        assert_eq!(gate_score_for_target(&d, "Gates"), Some(0.0));
    }

    #[test]
    fn empty_or_malformed_describe_is_none() {
        assert_eq!(gate_score_for_target(&described(serde_json::json!([])), "Gates"), None);
        assert_eq!(
            gate_score_for_target(&serde_json::json!({"entity": "Paris"}), "Gates"),
            None
        );
    }
}
