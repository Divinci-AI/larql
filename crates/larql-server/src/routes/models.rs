//! `GET /v1/models` — OpenAI-compatible model listing (N0.5).
//!
//! Response shape conforms to the OpenAI Models API
//! (<https://platform.openai.com/docs/api-reference/models/list>):
//!
//! ```json
//! {
//!   "object": "list",
//!   "data": [
//!     { "id": "<model-id>", "object": "model",
//!       "created": <unix-secs>, "owned_by": "larql",
//!       /* larql-specific extras follow */
//!       "path": "/v1/<model-id>" | "/v1",
//!       "features": <total>, "loaded": true }
//!   ]
//! }
//! ```
//!
//! The OpenAI spec only requires `id`, `object`, `created`, `owned_by`;
//! every other field is an extension that compatible clients ignore.
//! This means existing OpenAI SDKs (`openai.models.list()`) work
//! unmodified, while larql-aware clients still see `path` / `features`
//! / `loaded`.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::{Path, State};
use axum::Json;

use crate::http::API_PREFIX;
use crate::routes::openai::OpenAIError;
use crate::state::{AppState, LoadedModel};
use crate::vindex3::V3Model;

const MODEL_OBJECT: &str = "model";
const LIST_OBJECT: &str = "list";
const OWNED_BY: &str = "larql";
/// larql-specific `generation` marker for VINDEX3 containers.
const V3_GENERATION: u32 = 3;

/// Returns the boot-time of this server in unix seconds. Used as the
/// `created` field for every loaded model — close enough to the
/// OpenAI semantic ("when this model became available") since `larql`
/// loads its full model set at boot.
fn server_boot_unix_secs(state: &AppState) -> u64 {
    let now_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let uptime = state.started_at.elapsed().as_secs();
    now_unix.saturating_sub(uptime)
}

#[utoipa::path(
    get,
    path = "/v1/models",
    tag = "browse",
    responses(
        (status = 200, description = "OpenAI-compatible list of loaded models", body = crate::openapi::schemas::ModelsListResponse),
    ),
)]
pub async fn handle_models(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    state.bump_requests();

    let created = server_boot_unix_secs(&state);
    let multi = state.is_multi_model();

    let mut data: Vec<serde_json::Value> = state
        .models
        .iter()
        .map(|m| v2_entry(m, created, multi))
        .collect();
    data.extend(state.v3_models.iter().map(|m| v3_entry(m, created, multi)));

    Json(serde_json::json!({
        "object": LIST_OBJECT,
        "data": data,
    }))
}

/// One list/retrieve entry for a V2 model. Extras beyond the OpenAI
/// contract (`path`, `features`, `loaded`) are larql-specific and
/// ignored by compatible clients.
fn v2_entry(m: &LoadedModel, created: u64, multi: bool) -> serde_json::Value {
    let total_features: usize = m.config.layers.iter().map(|l| l.num_features).sum();
    serde_json::json!({
        "id": m.id,
        "object": MODEL_OBJECT,
        "created": created,
        "owned_by": OWNED_BY,
        "path": model_path(&m.id, multi),
        "features": total_features,
        "loaded": true,
    })
}

/// One entry for a VINDEX3 runtime (VI3-SERVE-1). `generation` marks
/// the container class; no `features` count — a V3 container is an
/// executable program, not a feature index.
fn v3_entry(m: &V3Model, created: u64, multi: bool) -> serde_json::Value {
    serde_json::json!({
        "id": m.id,
        "object": MODEL_OBJECT,
        "created": created,
        "owned_by": OWNED_BY,
        "path": model_path(&m.id, multi),
        "generation": V3_GENERATION,
        "loaded": true,
    })
}

fn model_path(id: &str, multi: bool) -> String {
    if multi {
        format!("{API_PREFIX}/{id}")
    } else {
        API_PREFIX.to_string()
    }
}

#[utoipa::path(
    get,
    path = "/v1/models/{model}",
    tag = "browse",
    params(("model" = String, Path, description = "Model id")),
    responses(
        (status = 200, description = "OpenAI-compatible single-model entry", body = crate::openapi::schemas::ModelEntry),
        (status = 404, body = crate::routes::openai::error::OpenAIErrorBody),
    ),
)]
pub async fn handle_model_retrieve(
    State(state): State<Arc<AppState>>,
    Path(model): Path<String>,
) -> Result<Json<serde_json::Value>, OpenAIError> {
    state.bump_requests();
    let created = server_boot_unix_secs(&state);
    let multi = state.is_multi_model();

    if let Some(m) = state.models.iter().find(|m| m.id == model) {
        return Ok(Json(v2_entry(m, created, multi)));
    }
    if let Some(m) = state.v3_models.iter().find(|m| m.id == model) {
        return Ok(Json(v3_entry(m, created, multi)));
    }
    Err(OpenAIError::not_found(format!("model '{model}' not found")))
}
