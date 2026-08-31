//! GET /v1/describe — query all knowledge edges for an entity.

use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::header::{CACHE_CONTROL, ETAG, IF_NONE_MATCH};
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;

use crate::band_utils::{
    filter_layers_by_band, get_layer_bands, BAND_KNOWLEDGE, PROBE_RELATION_SOURCE,
};
use crate::error::ServerError;
use crate::state::{elapsed_ms, AppState, LoadedModel};

const DESCRIBE_CACHE_CONTROL: &str = "public, max-age=86400";

#[derive(Deserialize, utoipa::IntoParams)]
#[into_params(parameter_in = Query)]
pub struct DescribeParams {
    /// Entity to describe, e.g. `France`.
    pub entity: String,
    /// Layer band to scan: `knowledge` (default), `syntax`, `output`, or `all`.
    #[serde(default = "default_band")]
    pub band: String,
    /// Include low-score edges in the response.
    #[serde(default)]
    pub verbose: bool,
    /// Maximum number of edges to return.
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Minimum gate score to include an edge.
    #[serde(default = "default_min_score")]
    pub min_score: f32,
}

fn default_band() -> String {
    BAND_KNOWLEDGE.into()
}
fn default_limit() -> usize {
    20
}
fn default_min_score() -> f32 {
    5.0
}

/// Describe an entity's edges against `patched`.
///
/// The overlay is passed in rather than read from `model` so the caller
/// decides scope: a request carrying `X-Session-Id` must be answered from that
/// session's overlay, not the global one. DESCRIBE is how the product measures
/// an edit (the gate score before/after), so if it stayed global-only while
/// patches became session-scoped, every edit would report "no change" — the
/// measurement would silently stop tracking the thing it measures.
pub(crate) fn describe_entity_with(
    model: &LoadedModel,
    patched: &larql_vindex::PatchedVindex,
    params: &DescribeParams,
) -> Result<serde_json::Value, ServerError> {
    let start = std::time::Instant::now();

    let encoding = model
        .tokenizer
        .encode(params.entity.as_str(), false)
        .map_err(|e| ServerError::Internal(format!("tokenize error: {e}")))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();

    if token_ids.is_empty() {
        return Ok(serde_json::json!({
            "entity": params.entity,
            "model": model.config.model,
            "edges": [],
            "latency_ms": 0.0,
        }));
    }

    let hidden = model.embeddings.shape()[1];
    let query = if token_ids.len() == 1 {
        model
            .embeddings
            .row(token_ids[0] as usize)
            .mapv(|v| v * model.embed_scale)
    } else {
        let mut avg = larql_vindex::ndarray::Array1::<f32>::zeros(hidden);
        for &tok in &token_ids {
            avg += &model
                .embeddings
                .row(tok as usize)
                .mapv(|v| v * model.embed_scale);
        }
        avg /= token_ids.len() as f32;
        avg
    };

    let bands = get_layer_bands(model);

    let all_layers = patched.loaded_layers();

    let scan_layers = filter_layers_by_band(all_layers, &params.band, &bands);

    let trace = patched.walk(&query, &scan_layers, params.limit);

    // Aggregate edges by target token (same logic as LQL DESCRIBE).
    struct EdgeInfo {
        gate: f32,
        layers: Vec<usize>,
        count: usize,
        original: String,
        also: Vec<String>,
        best_layer: usize,
        best_feature: usize,
    }

    let entity_lower = params.entity.to_lowercase();
    let mut edges: HashMap<String, EdgeInfo> = HashMap::new();

    for (layer_idx, hits) in &trace.layers {
        for hit in hits {
            if hit.gate_score < params.min_score {
                continue;
            }

            let tok = &hit.meta.top_token;
            let tok_trimmed = tok.trim();
            if tok_trimmed.is_empty() || tok_trimmed.len() < 2 {
                continue;
            }
            if tok_trimmed.to_lowercase() == entity_lower {
                continue;
            }

            let also: Vec<String> = hit
                .meta
                .top_k
                .iter()
                .filter(|t| {
                    let tt = t.token.trim();
                    tt.to_lowercase() != tok.to_lowercase()
                        && tt.to_lowercase() != entity_lower
                        && tt.len() >= 2
                        && t.logit > 0.0
                })
                .take(3)
                .map(|t| t.token.trim().to_string())
                .collect();

            let key = tok.to_lowercase();
            let entry = edges.entry(key).or_insert_with(|| EdgeInfo {
                gate: 0.0,
                layers: Vec::new(),
                best_feature: hit.feature,
                count: 0,
                original: tok_trimmed.to_string(),
                also,
                best_layer: *layer_idx,
            });

            if hit.gate_score > entry.gate {
                entry.gate = hit.gate_score;
                entry.best_layer = *layer_idx;
                entry.best_feature = hit.feature;
            }
            if !entry.layers.contains(layer_idx) {
                entry.layers.push(*layer_idx);
            }
            entry.count += 1;
        }
    }

    let mut ranked: Vec<&EdgeInfo> = edges.values().collect();
    ranked.sort_by(|a, b| {
        b.gate
            .partial_cmp(&a.gate)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    ranked.truncate(params.limit);

    let edge_json: Vec<serde_json::Value> = ranked
        .iter()
        .map(|info| {
            let min_l = *info.layers.iter().min().unwrap_or(&0);
            let max_l = *info.layers.iter().max().unwrap_or(&0);

            let mut edge = serde_json::json!({
                "target": info.original,
                "gate_score": (info.gate * 10.0).round() / 10.0,
                "layer": info.best_layer,
            });

            // Probe-confirmed relation label.
            if let Some(label) = model
                .probe_labels
                .get(&(info.best_layer, info.best_feature))
            {
                edge["relation"] = serde_json::json!(label);
                edge["source"] = serde_json::json!(PROBE_RELATION_SOURCE);
            }

            if params.verbose {
                edge["layer_max"] = serde_json::json!(max_l);
                edge["layer_min"] = serde_json::json!(min_l);
                edge["count"] = serde_json::json!(info.count);
            }

            if !info.also.is_empty() {
                edge["also"] = serde_json::json!(info.also);
            }

            edge
        })
        .collect();

    Ok(serde_json::json!({
        "entity": params.entity,
        "model": model.config.model,
        "edges": edge_json,
        "latency_ms": elapsed_ms(start),
    }))
}

async fn describe_with_cache(
    state: &Arc<AppState>,
    model: &Arc<LoadedModel>,
    headers: &HeaderMap,
    params: DescribeParams,
) -> Result<Response, ServerError> {
    // Session scope. A request with `X-Session-Id` is answered from that
    // session's overlay; without one it reads global state, as before.
    let sid = crate::session::extract_session_id(headers);

    // Check cache.
    //
    // The session id is part of the key. It has to be: the cached value is a
    // *patched* view, so sharing one entry across sessions would serve one
    // tenant's suppressions to another — a cross-tenant leak dressed up as a
    // cache hit, and invisible in every log.
    let cache_key = if state.describe_cache.is_enabled() {
        let key = crate::cache::DescribeCache::key_scoped(
            sid.as_deref(),
            &model.id,
            &params.entity,
            &params.band,
            params.limit,
            params.min_score,
        );
        if let Some(cached) = state.describe_cache.get(&key) {
            let etag = crate::etag::compute_etag(&cached);
            let if_none_match = headers.get(IF_NONE_MATCH).and_then(|v| v.to_str().ok());
            if crate::etag::matches_etag(if_none_match, &etag) {
                return Ok((axum::http::StatusCode::NOT_MODIFIED, [(ETAG, etag)]).into_response());
            }
            return Ok((
                [(ETAG, etag), (CACHE_CONTROL, DESCRIBE_CACHE_CONTROL.into())],
                Json(cached),
            )
                .into_response());
        }
        Some(key)
    } else {
        None
    };

    let model = Arc::clone(model);
    let task_state = Arc::clone(state);
    let result = tokio::task::spawn_blocking(move || {
        // Same lock discipline as `infer`: take a reader on the sessions map,
        // and fall back to global state for an unknown or never-patched
        // session so an expired session reads like a clean model rather than
        // erroring.
        if let Some(sid) = sid.as_deref() {
            let sessions = task_state.sessions.sessions_blocking_read();
            if let Some(patched) = sessions.get(sid).and_then(|s| s.patched()) {
                return describe_entity_with(&model, patched, &params);
            }
        }
        let patched = model.patched.blocking_read();
        describe_entity_with(&model, &patched, &params)
    })
    .await
    .map_err(|e| ServerError::Internal(e.to_string()))??;

    // Store in cache.
    if let Some(key) = cache_key {
        state.describe_cache.put(key, result.clone());
    }

    let etag = crate::etag::compute_etag(&result);
    Ok((
        [(ETAG, etag), (CACHE_CONTROL, DESCRIBE_CACHE_CONTROL.into())],
        Json(result),
    )
        .into_response())
}

#[utoipa::path(
    get,
    path = "/v1/describe",
    tag = "browse",
    params(DescribeParams),
    responses(
        (status = 200, description = "Edges for the queried entity", body = crate::openapi::schemas::DescribeResponse),
        (status = 304, description = "Not modified (ETag match)"),
        (status = 400, body = crate::error::ErrorBody),
        (status = 404, body = crate::error::ErrorBody),
        (status = 500, body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_describe(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(params): Query<DescribeParams>,
) -> Result<Response, ServerError> {
    state.bump_requests();
    let model = state.model_or_err(None)?;
    describe_with_cache(&state, &model, &headers, params).await
}

#[utoipa::path(
    get,
    path = "/v1/{model_id}/describe",
    tag = "browse",
    params(
        ("model_id" = String, Path, description = "Id of a loaded vindex."),
        DescribeParams,
    ),
    responses(
        (status = 200, body = crate::openapi::schemas::DescribeResponse),
        (status = 304),
        (status = 400, body = crate::error::ErrorBody),
        (status = 404, body = crate::error::ErrorBody),
        (status = 500, body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_describe_multi(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<String>,
    headers: HeaderMap,
    Query(params): Query<DescribeParams>,
) -> Result<Response, ServerError> {
    state.bump_requests();
    let model = state.model_or_err(Some(&model_id))?;
    describe_with_cache(&state, &model, &headers, params).await
}
