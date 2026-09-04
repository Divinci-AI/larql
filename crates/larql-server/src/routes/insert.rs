//! POST /v1/insert — constellation knowledge insertion.
//!
//! Full trace-guided multi-layer insert: forward pass to capture residuals,
//! use as gate vectors, write down vector overrides with target embedding.
//! Supports session isolation via X-Session-Id header.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::HeaderMap;
use axum::Json;
use serde::Deserialize;

use crate::band_utils::{get_layer_bands, INSERT_MODE_CONSTELLATION, INSERT_MODE_EMBEDDING};
use crate::error::ServerError;
use crate::session::extract_session_id;
use crate::state::{elapsed_ms, AppState, LoadedModel};

#[derive(Deserialize, utoipa::ToSchema)]
pub struct InsertRequest {
    pub entity: String,
    pub relation: String,
    pub target: String,
    /// Name to file the resulting patch under. This is the key
    /// `GET /v1/patches` reports and `DELETE /v1/patches/{name}` accepts,
    /// so a caller that supplies it can revert the insert without listing.
    /// Defaults to `insert:<entity>:<relation>:<target>`.
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub layer: Option<usize>,
    /// How the fact is installed.
    ///
    /// - `constellation` (default): one new FFN feature per knowledge layer,
    ///   gate keyed on the prompt's residual, down vector pointed at the
    ///   target — the fact is woven into the forward pass.
    /// - `knn`: one retrieval entry keyed on the residual at the install
    ///   layer; at inference the entry overrides the prediction when the
    ///   residual matches. With `LARQL_KNN_VERIFY` set on the server, it
    ///   fires only for prompts that also name the entity — exact and
    ///   selective, and independent facts never interfere.
    #[serde(default)]
    pub mode: Option<String>,
    /// Prompt whose residual stream the new features are keyed on.
    /// Defaults to `The <relation> of <entity> is`. Pass the exact text the
    /// fact will later be asked with — for an instruction-tuned model that is
    /// the chat-templated question — so the gate fires on the same residual
    /// the probe produces.
    #[serde(default)]
    pub prompt: Option<String>,
    /// Prompts the new feature must NOT fire on. Their residuals are
    /// projected out of the capture residual before it becomes the gate, so
    /// the gate reads 0 on each of them and `gate_score` on the capture
    /// prompt. Defaults to the capture prompt with the entity swapped for a
    /// few stock entities — the questions "What is the capital of X?" share
    /// most of their residual regardless of X, and a gate keyed on the raw
    /// residual fired for every X. Pass an empty list to disable.
    #[serde(default)]
    pub reference_prompts: Option<Vec<String>>,
    #[serde(default = "default_alpha")]
    pub alpha: f32,
    /// Gate pre-activation the new feature reaches on the capture prompt's
    /// residual: the gate override is the residual scaled so that
    /// `gate · residual == gate_score`. Bounded on purpose — an unscaled
    /// residual dotted with itself is tens of thousands, and every FFN rung
    /// computes the slot's activation as `φ(gate·x)·(up·x)`.
    #[serde(default = "default_gate_score")]
    pub gate_score: f32,
    /// Up pre-activation on the capture residual, installed the same way
    /// (`up · residual == up_score`). Together with `gate_score` this pins the
    /// slot's activation on the capture prompt to `φ(gate_score)·up_score`.
    #[serde(default = "default_up_score")]
    pub up_score: f32,
    #[serde(default = "default_confidence")]
    pub confidence: f32,
}

/// Default `alpha` (down-vector scale on the target embedding) when the
/// request omits it.
const DEFAULT_INSERT_ALPHA: f32 = 0.25;
/// Default `confidence` (stored as the feature's `c_score` and top-k logit)
/// when the request omits it.
const DEFAULT_INSERT_CONFIDENCE: f32 = 0.9;

/// Default gate pre-activation for an inserted feature on its own prompt —
/// in the range of the strongest natural gate scores a walk reports (≈5–8).
const DEFAULT_INSERT_GATE_SCORE: f32 = 8.0;
/// Default up pre-activation for an inserted feature on its own prompt.
const DEFAULT_INSERT_UP_SCORE: f32 = 1.0;

fn default_alpha() -> f32 {
    DEFAULT_INSERT_ALPHA
}
fn default_gate_score() -> f32 {
    DEFAULT_INSERT_GATE_SCORE
}
fn default_up_score() -> f32 {
    DEFAULT_INSERT_UP_SCORE
}
fn default_confidence() -> f32 {
    DEFAULT_INSERT_CONFIDENCE
}

/// The prompt whose residuals key the new features.
fn capture_prompt(req: &InsertRequest) -> String {
    match &req.prompt {
        Some(p) if !p.trim().is_empty() => p.clone(),
        _ => format!(
            "The {} of {} is",
            req.relation.replace(['-', '_'], " "),
            req.entity
        ),
    }
}

/// Scale `residual` so that `scaled · residual == target`.
///
/// `None` when the residual is (numerically) zero, in which case nothing can
/// be keyed on it.
fn residual_scaled_to_dot(residual: &[f32], target: f32) -> Option<Vec<f32>> {
    let sq: f32 = residual.iter().map(|v| v * v).sum();
    if sq <= 1e-12 {
        return None;
    }
    let k = target / sq;
    Some(residual.iter().map(|v| v * k).collect())
}

/// Stock entities substituted into the capture prompt to build the default
/// reference prompts. Chosen to be common enough that the model holds facts
/// about them and unlikely to be the entity being edited.
const DEFAULT_REFERENCE_ENTITIES: [&str; 3] = ["Germany", "Japan", "Brazil"];

/// The prompts the new feature must stay silent on.
fn reference_prompts(req: &InsertRequest, capture: &str) -> Vec<String> {
    if let Some(refs) = &req.reference_prompts {
        return refs
            .iter()
            .filter(|r| !r.trim().is_empty() && r.as_str() != capture)
            .cloned()
            .collect();
    }
    if req.entity.trim().is_empty() || !capture.contains(req.entity.as_str()) {
        return Vec::new();
    }
    DEFAULT_REFERENCE_ENTITIES
        .iter()
        .filter(|e| !e.eq_ignore_ascii_case(req.entity.trim()))
        .map(|e| capture.replace(req.entity.as_str(), e))
        .collect()
}

/// Residuals at `insert_layers` for one prompt, from a forward pass.
/// Needs only read access to the patched vindex. Empty when the vindex has
/// no model weights to run.
fn residuals_for_prompt(
    model: &LoadedModel,
    patched: &larql_vindex::PatchedVindex,
    prompt: &str,
    insert_layers: &[usize],
) -> Vec<(usize, Vec<f32>)> {
    if !model.config.has_model_weights {
        return Vec::new();
    }

    let weights_guard = match model.get_or_load_weights() {
        Ok(w) => w,
        Err(_) => return Vec::new(),
    };
    let weights: &larql_inference::ModelWeights = &weights_guard;

    let encoding = match model.tokenizer.encode(prompt, true) {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();

    let walk_ffn = larql_inference::vindex::WalkFfn::new_unlimited_with_trace(weights, patched);
    let _result =
        larql_inference::predict_with_ffn(weights, &model.tokenizer, &token_ids, 1, &walk_ffn);

    walk_ffn
        .take_residuals()
        .into_iter()
        .filter(|(layer, _)| insert_layers.contains(layer))
        .collect()
}

/// What the insert is keyed on: the capture prompt's residual per layer, and
/// the reference prompts' residuals it must be orthogonal to.
struct Residuals {
    capture: Vec<(usize, Vec<f32>)>,
    references: Vec<Vec<(usize, Vec<f32>)>>,
}

fn compute_residuals(
    model: &LoadedModel,
    patched: &larql_vindex::PatchedVindex,
    req: &InsertRequest,
    insert_layers: &[usize],
) -> Residuals {
    let capture_prompt = capture_prompt(req);
    let capture = residuals_for_prompt(model, patched, &capture_prompt, insert_layers);
    let references = if capture.is_empty() {
        Vec::new()
    } else {
        reference_prompts(req, &capture_prompt)
            .iter()
            .map(|p| residuals_for_prompt(model, patched, p, insert_layers))
            .filter(|r| !r.is_empty())
            .collect()
    };
    Residuals {
        capture,
        references,
    }
}

/// The part of `v` orthogonal to every vector in `basis` (Gram–Schmidt
/// against the basis, which is orthogonalised on the fly). A gate built from
/// the result reads exactly 0 on each basis vector.
fn orthogonal_to(v: &[f32], basis: &[&[f32]]) -> Vec<f32> {
    let mut ortho_basis: Vec<Vec<f32>> = Vec::with_capacity(basis.len());
    for b in basis {
        let mut q: Vec<f32> = b.to_vec();
        for e in &ortho_basis {
            let d: f32 = q.iter().zip(e).map(|(a, b)| a * b).sum();
            for (qi, ei) in q.iter_mut().zip(e) {
                *qi -= d * ei;
            }
        }
        let n: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n > 1e-6 {
            for qi in &mut q {
                *qi /= n;
            }
            ortho_basis.push(q);
        }
    }
    let mut out = v.to_vec();
    for e in &ortho_basis {
        let d: f32 = out.iter().zip(e).map(|(a, b)| a * b).sum();
        for (oi, ei) in out.iter_mut().zip(e) {
            *oi -= d * ei;
        }
    }
    out
}

/// The patch an insert installs, plus the slots it claimed.
struct InsertPlan {
    patch: larql_vindex::VindexPatch,
    slots: Vec<(usize, usize)>,
    use_constellation: bool,
}

/// Build the patch that installs one feature per insert layer.
///
/// The insert used to write straight into the overlay's override maps,
/// which left no record in the patch stack: `GET /v1/patches` did not list
/// it and `DELETE /v1/patches/{name}` could not undo it, so a caller that
/// wanted to revert an added fact had to drop the whole session. Going
/// through a `VindexPatch` — carrying the gate and down vectors it computed —
/// makes the insert a first-class, named, revertible entry, and the returned
/// patch can be re-applied verbatim on another instance.
///
/// Needs only read access to `patched`: slots are chosen against its current
/// state and each layer receives exactly one new feature, so a plan never
/// collides with itself.
fn build_insert_patch(
    model: &LoadedModel,
    patched: &larql_vindex::PatchedVindex,
    req: &InsertRequest,
    name: &str,
    insert_layers: &[usize],
    residuals: &Residuals,
) -> InsertPlan {
    let hidden = model.embeddings.shape()[1];
    let mut ops = Vec::new();
    let mut slots = Vec::new();

    // Target embedding for the down vector: the FIRST token of the target as
    // it would be generated — with a leading space, which is how a word
    // token appears after the model's turn marker (" Atlantis" is one token;
    // "Atlantis" is "Atlant" + "is", and averaging the two made the insert
    // teach "is"). The model predicts one token next, so only the first one
    // is the fact's answer.
    let spaced = if req.target.starts_with(' ') {
        req.target.clone()
    } else {
        format!(" {}", req.target)
    };
    let target_encoding = match model.tokenizer.encode(spaced.as_str(), false) {
        Ok(e) => e,
        Err(_) => {
            return InsertPlan {
                patch: empty_patch(model, name),
                slots,
                use_constellation: false,
            }
        }
    };
    let target_ids: Vec<u32> = target_encoding.get_ids().to_vec();
    let target_id = target_ids.first().copied().unwrap_or(0);

    let mut target_embed = vec![0.0f32; hidden];
    {
        let row = model.embeddings.row(target_id as usize);
        for j in 0..hidden {
            target_embed[j] = row[j] * model.embed_scale;
        }
    }

    let use_constellation = !residuals.capture.is_empty();

    for &layer in insert_layers {
        let feature = match patched.find_free_feature(layer) {
            Some(f) => f,
            None => continue,
        };

        // Gate (and up) vectors. Constellation mode keys the slot on the
        // capture prompt's residual, scaled to a bounded pre-activation: every
        // FFN rung computes the slot's activation as φ(gate·x)·(up·x), so a
        // gate that merely matched the average gate-row norm dotted the
        // residual with itself (tens of thousands) and one insert wiped the
        // layer's output. Embedding mode (no weights to run) falls back to the
        // entity embedding as a unit gate and leaves `up` to the base slot.
        let (gate_vec, up_vec): (Vec<f32>, Option<Vec<f32>>) =
            if let Some((_, residual)) = residuals.capture.iter().find(|(l, _)| *l == layer) {
                // Selectivity: the gate is the capture residual with every
                // reference residual projected out, so it reads 0 on the
                // references and `gate_score` on the capture prompt. The up
                // vector stays residual-aligned; the gate is what gates.
                let refs: Vec<&[f32]> = residuals
                    .references
                    .iter()
                    .filter_map(|r| r.iter().find(|(l, _)| *l == layer))
                    .map(|(_, v)| v.as_slice())
                    .collect();
                let direction = orthogonal_to(residual, &refs);
                let gate_dot: f32 = direction.iter().zip(residual).map(|(a, b)| a * b).sum();
                let gate = if gate_dot.abs() > 1e-6 {
                    let k = req.gate_score / gate_dot;
                    Some(direction.iter().map(|v| v * k).collect::<Vec<f32>>())
                } else {
                    None
                };
                match (gate, residual_scaled_to_dot(residual, req.up_score)) {
                    (Some(g), Some(u)) => (g, Some(u)),
                    _ => continue,
                }
            } else {
                let enc = match model.tokenizer.encode(req.entity.as_str(), false) {
                    Ok(e) => e,
                    Err(_) => continue,
                };
                let ids = enc.get_ids();
                let mut ev = vec![0.0f32; hidden];
                for &tok in ids {
                    let row = model.embeddings.row(tok as usize);
                    for j in 0..hidden {
                        ev[j] += row[j] * model.embed_scale;
                    }
                }
                let n = ids.len().max(1) as f32;
                for v in &mut ev {
                    *v /= n;
                }
                let norm: f32 = ev.iter().map(|v| v * v).sum::<f32>().sqrt();
                if norm > 1e-8 {
                    for v in &mut ev {
                        *v /= norm;
                    }
                }
                (ev, None)
            };

        let down_vec: Vec<f32> = target_embed.iter().map(|v| v * req.alpha).collect();

        ops.push(larql_vindex::PatchOp::Insert {
            layer,
            feature,
            relation: Some(req.relation.clone()),
            entity: req.entity.clone(),
            target: req.target.clone(),
            confidence: Some(req.confidence),
            gate_vector_b64: Some(larql_vindex::patch::core::encode_gate_vector(&gate_vec)),
            up_vector_b64: up_vec
                .as_deref()
                .map(larql_vindex::patch::core::encode_gate_vector),
            down_vector_b64: Some(larql_vindex::patch::core::encode_gate_vector(&down_vec)),
            down_meta: Some(larql_vindex::patch::core::PatchDownMeta {
                top_token: req.target.clone(),
                top_token_id: target_id,
                c_score: req.confidence,
            }),
        });
        slots.push((layer, feature));
    }

    let mut patch = empty_patch(model, name);
    patch.operations = ops;
    InsertPlan {
        patch,
        slots,
        use_constellation,
    }
}

fn empty_patch(model: &LoadedModel, name: &str) -> larql_vindex::VindexPatch {
    let created_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs().to_string())
        .unwrap_or_default();
    larql_vindex::VindexPatch {
        version: 1,
        base_model: model.id.clone(),
        base_checksum: None,
        created_at,
        description: Some(name.to_string()),
        author: None,
        tags: vec!["insert".into()],
        operations: Vec::new(),
    }
}

/// Insert mode names accepted in `InsertRequest::mode`.
pub const INSERT_MODE_KNN: &str = "knn";

fn wants_knn(req: &InsertRequest) -> bool {
    req.mode
        .as_deref()
        .map(|m| m.eq_ignore_ascii_case(INSERT_MODE_KNN))
        .unwrap_or(false)
}

/// The `knn` plan: one retrieval entry at `layer`, keyed on the capture
/// prompt's L2-normalised residual there, answering with the target's first
/// token. `None` when there is no residual to key on (a vindex without model
/// weights) or the target does not tokenize.
fn build_knn_patch(
    model: &LoadedModel,
    req: &InsertRequest,
    name: &str,
    layer: usize,
    residuals: &Residuals,
) -> Option<InsertPlan> {
    let (_, residual) = residuals.capture.iter().find(|(l, _)| *l == layer)?;
    let norm: f32 = residual.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm <= 1e-8 {
        return None;
    }
    let key: Vec<f32> = residual.iter().map(|v| v / norm).collect();

    let spaced = if req.target.starts_with(' ') {
        req.target.clone()
    } else {
        format!(" {}", req.target)
    };
    let target_id = model
        .tokenizer
        .encode(spaced.as_str(), false)
        .ok()?
        .get_ids()
        .first()
        .copied()?;
    // The token the override answers with, as the tokenizer spells it.
    let target_token = model
        .tokenizer
        .decode(&[target_id], false)
        .unwrap_or_else(|_| spaced.clone());

    let mut patch = empty_patch(model, name);
    patch.tags.push(INSERT_MODE_KNN.into());
    patch.operations.push(larql_vindex::PatchOp::InsertKnn {
        layer,
        entity: req.entity.clone(),
        relation: req.relation.clone(),
        target: target_token,
        target_id,
        confidence: Some(req.confidence),
        key_vector_b64: larql_vindex::patch::core::encode_gate_vector(&key),
    });
    Some(InsertPlan {
        patch,
        slots: vec![(layer, 0)],
        use_constellation: false,
    })
}

fn default_patch_name(req: &InsertRequest) -> String {
    format!("insert:{}:{}:{}", req.entity, req.relation, req.target)
}

fn run_insert(
    state: &AppState,
    model: &LoadedModel,
    req: &InsertRequest,
    session_id: Option<&str>,
) -> Result<serde_json::Value, ServerError> {
    let start = std::time::Instant::now();
    let name = req.name.clone().unwrap_or_else(|| default_patch_name(req));

    // Determine insert layers
    let bands = get_layer_bands(model);

    let knn = wants_knn(req);
    let insert_layers: Vec<usize> = if let Some(l) = req.layer {
        vec![l]
    } else if knn {
        // One retrieval key, at the layer the LQL `INSERT ... MODE KNN` uses:
        // the last knowledge layer but one.
        vec![bands.knowledge.1.saturating_sub(1)]
    } else {
        let mid = (bands.knowledge.0 + bands.knowledge.1) / 2;
        (mid..=bands.knowledge.1).collect()
    };
    let knn_layer = insert_layers[0];

    // `knn` installs a lookup entry, not features: it needs the residual and
    // nothing else, so an index without weights cannot do it at all.
    let plan_for = |model: &LoadedModel,
                    patched: &larql_vindex::PatchedVindex,
                    residuals: &Residuals|
     -> Result<InsertPlan, ServerError> {
        if knn {
            build_knn_patch(model, req, &name, knn_layer, residuals).ok_or_else(|| {
                ServerError::BadRequest(
                    "knn insert needs a residual to key on: the vindex has no model weights, \
                     the prompt produced none at the install layer, or the target does not tokenize"
                        .into(),
                )
            })
        } else {
            Ok(build_insert_patch(
                model,
                patched,
                req,
                &name,
                &insert_layers,
                residuals,
            ))
        }
    };

    let (plan, active) = if let Some(sid) = session_id {
        // Session-scoped: read from session for residuals, write to session for insert
        let mut sessions = state.sessions.sessions_blocking_write();
        let now = std::time::Instant::now();

        let session = state
            .sessions
            .bind_in_guard(&mut sessions, sid, &model.id, now);
        // Inserting writes to the overlay, so materialise it here if this
        // is the session's first mutation.
        let patched = session.overlay_mut(|| model.patched.blocking_read().base().clone());

        let residuals = compute_residuals(model, patched, req, &insert_layers);
        let plan = plan_for(model, patched, &residuals)?;
        patched.apply_patch(plan.patch.clone());
        let active = patched.num_patches();
        (plan, active)
    } else {
        // Global: read from global for residuals, write to global for insert
        let plan = {
            let patched = model.patched.blocking_read();
            let residuals = compute_residuals(model, &patched, req, &insert_layers);
            plan_for(model, &patched, &residuals)?
        };

        let mut patched = model.patched.blocking_write();
        patched.apply_patch(plan.patch.clone());
        let active = patched.num_patches();
        (plan, active)
    };

    let slots: Vec<serde_json::Value> = plan
        .slots
        .iter()
        .map(|(layer, feature)| serde_json::json!({ "layer": layer, "feature": feature }))
        .collect();

    Ok(serde_json::json!({
        "entity": req.entity,
        "relation": req.relation,
        "target": req.target,
        "inserted": plan.slots.len(),
        "mode": if knn {
            INSERT_MODE_KNN
        } else if plan.use_constellation {
            INSERT_MODE_CONSTELLATION
        } else {
            INSERT_MODE_EMBEDDING
        },
        "alpha": req.alpha,
        "gate_score": req.gate_score,
        "up_score": req.up_score,
        "prompt": capture_prompt(req),
        "reference_prompts": reference_prompts(req, &capture_prompt(req)),
        "session": session_id,
        // The insert is filed in the patch stack under this name, so it
        // lists in `GET /v1/patches` and reverts with `DELETE /v1/patches/{name}`.
        "applied": name,
        "active_patches": active,
        "slots": slots,
        // The patch as installed, vectors included. Re-applying it through
        // `POST /v1/patches/apply` reproduces the insert exactly, on any
        // instance, without another forward pass.
        "patch": plan.patch,
        "latency_ms": elapsed_ms(start),
    }))
}

#[utoipa::path(
    post,
    path = "/v1/insert",
    tag = "inference",
    request_body = InsertRequest,
    responses(
        (status = 200, description = "Constellation insert result", body = crate::openapi::schemas::InsertResponse),
        (status = 400, body = crate::error::ErrorBody),
        (status = 500, body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_insert(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<InsertRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    let model = state.model_or_err(None)?;
    let sid = extract_session_id(&headers);
    let state2 = Arc::clone(&state);
    let result =
        tokio::task::spawn_blocking(move || run_insert(&state2, &model, &req, sid.as_deref()))
            .await
            .map_err(|e| ServerError::Internal(e.to_string()))??;
    Ok(Json(result))
}

#[utoipa::path(
    post,
    path = "/v1/{model_id}/insert",
    tag = "inference",
    params(("model_id" = String, Path, description = "Id of a loaded vindex.")),
    request_body = InsertRequest,
    responses(
        (status = 200, body = crate::openapi::schemas::InsertResponse),
        (status = 400, body = crate::error::ErrorBody),
        (status = 404, body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_insert_multi(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<String>,
    headers: HeaderMap,
    Json(req): Json<InsertRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    let model = state.model_or_err(Some(&model_id))?;
    let sid = extract_session_id(&headers);
    let state2 = Arc::clone(&state);
    let result =
        tokio::task::spawn_blocking(move || run_insert(&state2, &model, &req, sid.as_deref()))
            .await
            .map_err(|e| ServerError::Internal(e.to_string()))??;
    Ok(Json(result))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(prompt: Option<&str>) -> InsertRequest {
        InsertRequest {
            entity: "Mars".into(),
            relation: "capital".into(),
            target: "Olympus".into(),
            name: None,
            layer: None,
            mode: None,
            prompt: prompt.map(str::to_string),
            reference_prompts: None,
            alpha: default_alpha(),
            gate_score: default_gate_score(),
            up_score: default_up_score(),
            confidence: default_confidence(),
        }
    }

    #[test]
    fn capture_prompt_defaults_to_the_relation_sentence() {
        assert_eq!(capture_prompt(&req(None)), "The capital of Mars is");
        assert_eq!(capture_prompt(&req(Some("   "))), "The capital of Mars is");
    }

    #[test]
    fn capture_prompt_honours_an_explicit_prompt() {
        let p = "<|turn>user\nWhat is the capital of Mars?<turn|>\n<|turn>model\n";
        assert_eq!(capture_prompt(&req(Some(p))), p);
    }

    #[test]
    fn mode_knn_is_recognised_case_insensitively() {
        let mut r = req(None);
        assert!(!wants_knn(&r));
        r.mode = Some("KNN".into());
        assert!(wants_knn(&r));
        r.mode = Some("constellation".into());
        assert!(!wants_knn(&r));
    }

    #[test]
    fn default_patch_name_is_the_triple() {
        assert_eq!(
            default_patch_name(&req(None)),
            "insert:Mars:capital:Olympus"
        );
    }

    // The whole point of the scaling: the slot's pre-activation on its own
    // residual is the requested number, not the residual's squared norm.
    #[test]
    fn residual_scaled_to_dot_hits_the_target_dot_product() {
        let residual = vec![3.0f32, 4.0, 0.0, 12.0]; // ‖r‖² = 169
        let g = residual_scaled_to_dot(&residual, 8.0).unwrap();
        let dot: f32 = g.iter().zip(&residual).map(|(a, b)| a * b).sum();
        assert!((dot - 8.0).abs() < 1e-4, "dot = {dot}");
        // Direction preserved.
        assert!((g[0] / residual[0] - g[3] / residual[3]).abs() < 1e-7);
    }

    #[test]
    fn default_reference_prompts_swap_the_entity_for_stock_entities() {
        let r = req(None);
        let refs = reference_prompts(&r, &capture_prompt(&r));
        assert_eq!(
            refs,
            vec![
                "The capital of Germany is",
                "The capital of Japan is",
                "The capital of Brazil is",
            ]
        );
    }

    #[test]
    fn default_reference_prompts_never_include_the_entity_itself() {
        let mut r = req(None);
        r.entity = "Japan".into();
        let refs = reference_prompts(&r, &capture_prompt(&r));
        assert_eq!(refs.len(), 2);
        assert!(refs.iter().all(|p| !p.contains("Japan")));
    }

    #[test]
    fn explicit_empty_reference_list_disables_the_contrast() {
        let mut r = req(None);
        r.reference_prompts = Some(vec![]);
        assert!(reference_prompts(&r, &capture_prompt(&r)).is_empty());
    }

    #[test]
    fn no_reference_prompts_when_the_prompt_does_not_name_the_entity() {
        let r = req(Some("Tell me about the red planet"));
        assert!(reference_prompts(&r, &capture_prompt(&r)).is_empty());
    }

    // The contract the selectivity rests on: a gate built from the
    // orthogonalised residual reads exactly 0 on every reference.
    #[test]
    fn orthogonal_to_zeroes_the_dot_with_every_basis_vector() {
        let v = [1.0f32, 2.0, 3.0, 4.0];
        let b1 = [1.0f32, 0.0, 0.0, 0.0];
        let b2 = [1.0f32, 1.0, 0.0, 0.0]; // not orthogonal to b1 on purpose
        let o = orthogonal_to(&v, &[&b1, &b2]);
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        assert!(dot(&o, &b1).abs() < 1e-5);
        assert!(dot(&o, &b2).abs() < 1e-5);
        // What remains is the part outside the basis span.
        assert_eq!(o, vec![0.0, 0.0, 3.0, 4.0]);
    }

    #[test]
    fn orthogonal_to_with_no_basis_is_the_identity() {
        let v = [1.0f32, 2.0];
        assert_eq!(orthogonal_to(&v, &[]), v.to_vec());
    }

    #[test]
    fn residual_scaled_to_dot_declines_a_zero_residual() {
        assert!(residual_scaled_to_dot(&[0.0, 0.0], 8.0).is_none());
    }
}
