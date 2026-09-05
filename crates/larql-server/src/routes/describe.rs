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
    /// Maximum number of edges to return, after filtering.
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Per-layer candidate window: how many gate hits each layer contributes
    /// before coherence filtering and the final `limit`.
    ///
    /// This used to be `limit` itself, so a browse of 20 pooled the top 20 of
    /// every layer's top 20 by raw gate score. Measured 2026-09-04 on
    /// production, the Paris→France feature (L27 `French`) sits at rank 29 of
    /// a 300-wide window with score 8.2, under incoherent features scoring up
    /// to 18.7; at a window of 20 it could never appear. With the window
    /// wide and the coherence filter on it is #6.
    #[serde(default = "default_window")]
    pub window: usize,
    /// Minimum gate score to include an edge.
    #[serde(default = "default_min_score")]
    pub min_score: f32,
    /// Score each feature's top-k for coherence and report it on the edge.
    ///
    /// On by default since 2026-09-04. Scoring never changes `target` on its
    /// own; `relabel` does that. See `crate::coherence`.
    #[serde(default = "default_true")]
    pub coherence: bool,
    /// Drop edges whose coherence falls below this. Implies `coherence`.
    /// Default 0.35: measured 2026-09-01 on 80 associations, the population
    /// above it is one-concept features and the population below is noise,
    /// with the boundary around 0.3. 0.0 reports without filtering.
    #[serde(default = "default_min_coherence")]
    pub min_coherence: f32,
    /// Label each feature from its cluster (centroid-nearest, preferring the
    /// entity's script) instead of the logit argmax. Implies `coherence`.
    ///
    /// On by default since 2026-09-04. Every path that has to agree on a
    /// label — DESCRIBE, the measurement DESCRIBE in `measure.rs`, and the
    /// caller's resolution of a target — must use the same setting, or a
    /// target shown by one is unknown to another. Edges now also carry
    /// `feature`, so a caller can resolve by `(layer, feature)` and stop
    /// depending on the label at all.
    #[serde(default = "default_true")]
    pub relabel: bool,
    /// Rank by relevance — the gate score's z-score against the feature's
    /// background over a panel of unrelated queries — instead of by the raw
    /// gate score. On by default since 2026-09-04: features that fire for
    /// every input (`especially`, `either`, `role`, `mode`) are coherent, so
    /// the coherence filter keeps them, and on raw score they crowd out the
    /// entity's own. See `crate::relevance`. `gate_score` is still reported
    /// and `min_score` still applies to it.
    #[serde(default = "default_true")]
    pub relevance: bool,
    /// Which panel relevance is measured against: `corpus`, `entities` or
    /// `vocabulary`. See `crate::relevance::Background`. Absent, the
    /// deployment's default (`LARQL_RELEVANCE_BACKGROUND`, else `entities`).
    /// Kept selectable so the panels can be compared on one deployment; the
    /// response reports the one used and its size as
    /// `relevance_background` / `relevance_panel`.
    #[serde(default)]
    pub background: Option<String>,
    /// How the per-layer candidate window is chosen. `score` (default):
    /// the `window` highest raw gate scores, then relevance re-orders them.
    /// `relevance`: every feature in the layer is z-scored and the `window`
    /// most surprising are kept — retrieval by surprise, not by magnitude.
    /// Measured 2026-09-04 (research log §22), a residual query's
    /// raw-score window was junk that relevance could only re-order.
    /// Needs `relevance`.
    #[serde(default = "default_window_by")]
    pub window_by: String,
    /// What the gates are scored against.
    ///
    /// `embedding` (default): the entity's raw input embedding, averaged over
    /// its tokens, scored against every layer. That is the historical
    /// behaviour and it is lexical: it finds features that respond to the
    /// token itself, and cannot find a relation. Measured 2026-09-03 on
    /// production, "Paris" in every casing surfaces zero France features and
    /// "Eiffel" surfaces nothing Parisian, while "France" and "Paris, France"
    /// surface a French feature — only because the token is in the query.
    ///
    /// `residual`: run the model over the entity and score each layer's gates
    /// against the residual the FFN actually sees at that layer (last token),
    /// which is what inference does. Needs model weights in the vindex.
    #[serde(default = "default_query")]
    pub query: String,
    /// Residual mode only: a prompt template containing `{entity}`, run
    /// through the model in place of the bare name; the residual is taken
    /// at the template's last token. "{entity} is the capital of" puts the
    /// model in the position of *predicting* the fact, where a bare name's
    /// last token only encodes the name. The relevance panel is built on
    /// the same template, so the background matches. Default `{entity}`.
    #[serde(default = "default_prompt")]
    pub prompt: String,
    /// Residual mode only: a neutral prompt whose per-layer residual is
    /// subtracted from the entity's before scoring.
    ///
    /// Measured 2026-09-04 on production at min_score 0, the plain residual
    /// query returns the same six features for Paris, France, Einstein,
    /// Tokyo and "Paris is the capital of": the normed last-token residual
    /// carries a large component shared by every input, and the gates most
    /// aligned with it win the dot product regardless of direction. Those
    /// features genuinely fire for everything. What a browse needs is what
    /// fires *unusually* for this input, and subtracting a neutral baseline
    /// is the smallest version of that question.
    #[serde(default)]
    pub baseline: Option<String>,
}

/// Per-layer last-token residuals for `text`, tokenised as INFER does (with
/// BOS). One place, so the entity, a baseline and the relevance panel are
/// all run the same way — a panel built one way and a query another would
/// measure the difference between the two.
fn residuals_for(
    model: &LoadedModel,
    weights: &larql_inference::ModelWeights,
    patched: &larql_vindex::PatchedVindex,
    text: &str,
) -> Result<std::collections::HashMap<usize, Vec<f32>>, ServerError> {
    let enc = model
        .tokenizer
        .encode(text, true)
        .map_err(|e| ServerError::Internal(format!("tokenize error: {e}")))?;
    let ids: Vec<u32> = enc.get_ids().to_vec();
    let run = larql_inference::infer_patched(
        weights,
        &model.tokenizer,
        patched,
        Some(&patched.knn_store),
        &ids,
        1,
        &larql_inference::KnnRouteMode::from_env(),
    );
    Ok(run.residuals.into_iter().collect())
}

/// Build and install the residual relevance panel for `background` unless
/// one is already there. One forward pass per panel name; concurrent first
/// requests wait on the build lock rather than each running the panel.
fn ensure_residual_panel(
    model: &LoadedModel,
    weights: &larql_inference::ModelWeights,
    patched: &larql_vindex::PatchedVindex,
    background: crate::relevance::Background,
    template: &str,
) -> Result<(), ServerError> {
    let names =
        crate::relevance::RelevanceStats::residual_panel_names(background).ok_or_else(|| {
            ServerError::BadRequest(format!(
                "background `{}` is not offered for residual queries; use `entities`",
                background.as_str()
            ))
        })?;
    if model.relevance.has_residual_panel(background, template) {
        return Ok(());
    }
    let _build = model
        .relevance
        .residual_build
        .lock()
        .map_err(|_| ServerError::Internal("panel lock".into()))?;
    if model.relevance.has_residual_panel(background, template) {
        return Ok(());
    }
    let start = std::time::Instant::now();
    let mut rows: std::collections::HashMap<usize, Vec<Vec<f32>>> =
        std::collections::HashMap::new();
    for name in names {
        let text = template.replace("{entity}", name);
        for (layer, r) in residuals_for(model, weights, patched, &text)? {
            rows.entry(layer).or_default().push(r);
        }
    }
    let hidden = model.embeddings.ncols();
    let mut per_layer = std::collections::HashMap::new();
    for (layer, rs) in rows {
        let ok: Vec<&Vec<f32>> = rs.iter().filter(|r| r.len() == hidden).collect();
        let mut m = larql_vindex::ndarray::Array2::<f32>::zeros((ok.len(), hidden));
        for (i, r) in ok.iter().enumerate() {
            m.row_mut(i)
                .assign(&larql_vindex::ndarray::Array1::from((*r).clone()));
        }
        per_layer.insert(layer, m);
    }
    tracing::info!(
        "residual relevance panel `{}` for {:?}: {} names, {} layers, {:.1}s",
        background.as_str(),
        template,
        names.len(),
        per_layer.len(),
        start.elapsed().as_secs_f32()
    );
    model
        .relevance
        .install_residual_panel(background, template, per_layer);
    Ok(())
}

pub const QUERY_EMBEDDING: &str = "embedding";
/// Both queries, merged: every hit ranked by z against its own background.
/// The embedding side keeps its raw-score window and `min_score`; the
/// residual side takes the surprise window and no score floor, since its
/// gate scores are on another scale. Needs `relevance`.
pub const QUERY_UNION: &str = "union";
pub const QUERY_RESIDUAL: &str = "residual";

fn default_query() -> String {
    QUERY_EMBEDDING.into()
}

pub const PROMPT_BARE: &str = "{entity}";
fn default_prompt() -> String {
    PROMPT_BARE.into()
}
/// A template must name the entity, and stays short: it is a prompt the
/// panel's 114 names are each run through.
fn check_prompt(template: &str) -> Result<(), ServerError> {
    if !template.contains("{entity}") {
        return Err(ServerError::BadRequest(
            "prompt must contain `{entity}`".into(),
        ));
    }
    if template.chars().count() > 200 {
        return Err(ServerError::BadRequest(
            "prompt must be at most 200 characters".into(),
        ));
    }
    Ok(())
}

pub const WINDOW_BY_SCORE: &str = "score";
pub const WINDOW_BY_RELEVANCE: &str = "relevance";
fn default_window_by() -> String {
    WINDOW_BY_SCORE.into()
}

fn default_band() -> String {
    BAND_KNOWLEDGE.into()
}
fn default_limit() -> usize {
    20
}
fn default_window() -> usize {
    300
}
fn default_true() -> bool {
    true
}
fn default_min_coherence() -> f32 {
    0.35
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
pub fn describe_entity_with(
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

    // The same construction the relevance panel uses, so a query and its
    // background are on one footing.
    let query = crate::relevance::entity_query(
        &model.embeddings,
        model.embed_scale,
        &model.tokenizer,
        params.entity.as_str(),
    )
    .ok_or_else(|| ServerError::Internal("entity produced no query".into()))?;

    let window_by_relevance = match params.window_by.as_str() {
        WINDOW_BY_SCORE => false,
        WINDOW_BY_RELEVANCE if params.relevance => true,
        WINDOW_BY_RELEVANCE => {
            return Err(ServerError::BadRequest(
                "window_by=relevance needs relevance=true".into(),
            ))
        }
        other => {
            return Err(ServerError::BadRequest(format!(
                "window_by must be `{WINDOW_BY_SCORE}` or `{WINDOW_BY_RELEVANCE}`, got `{other}`"
            )))
        }
    };
    // Selecting by surprise means scoring the whole layer first.
    let walk_k = |layer: usize| {
        if window_by_relevance {
            patched.num_features(layer).max(params.window)
        } else {
            params.window
        }
    };

    let background = match params.background.as_deref() {
        None => model.relevance.default_background(),
        Some(s) => crate::relevance::Background::parse(s).ok_or_else(|| {
            ServerError::BadRequest(format!(
                "background must be `corpus`, `entities` or `vocabulary`, got `{s}`"
            ))
        })?,
    };

    let bands = get_layer_bands(model);

    let all_layers = patched.loaded_layers();

    let scan_layers = filter_layers_by_band(all_layers, &params.band, &bands);

    // How many layers were scored against a real residual (residual mode
    // only). Reported, because a layer with no captured residual is silently
    // absent from the answer and a caller comparing modes must be able to
    // see that.
    let mut residual_layers: Option<usize> = None;
    let mut contrasted_layers: usize = 0;

    if params.query == QUERY_EMBEDDING && params.prompt != PROMPT_BARE {
        return Err(ServerError::BadRequest(
            "prompt applies to query=residual only".into(),
        ));
    }
    let union = params.query == QUERY_UNION;
    if union && !params.relevance {
        return Err(ServerError::BadRequest(
            "query=union ranks both sides by relevance; it needs relevance=true".into(),
        ));
    }
    let want_emb = matches!(params.query.as_str(), QUERY_EMBEDDING | QUERY_UNION);
    let want_res = matches!(params.query.as_str(), QUERY_RESIDUAL | QUERY_UNION);
    if !want_emb && !want_res {
        return Err(ServerError::BadRequest(format!(
            "query must be `{QUERY_EMBEDDING}`, `{QUERY_RESIDUAL}` or `{QUERY_UNION}`, got `{}`",
            params.query
        )));
    }
    // The residual side of a union is always windowed by surprise: measured
    // 2026-09-04 (research log §22), its raw-score window is junk.
    let res_surprise = window_by_relevance || union;
    let res_k = |layer: usize| {
        if res_surprise {
            patched.num_features(layer).max(params.window)
        } else {
            params.window
        }
    };

    let mut emb_trace: Option<larql_vindex::WalkTrace> = if want_emb {
        Some({
            if window_by_relevance {
                let mut layers = Vec::with_capacity(scan_layers.len());
                for &layer in &scan_layers {
                    layers.extend(patched.walk(&query, &[layer], walk_k(layer)).layers);
                }
                larql_vindex::WalkTrace { layers }
            } else {
                patched.walk(&query, &scan_layers, params.window)
            }
        })
    } else {
        None
    };
    let mut res_trace: Option<larql_vindex::WalkTrace> = if want_res {
        Some({
            if !model.config.has_model_weights {
                return Err(ServerError::InferenceUnavailable(
                    "query=residual needs model weights in the vindex; rebuild with --include-weights"
                        .into(),
                ));
            }
            let weights_guard = model
                .get_or_load_weights()
                .map_err(ServerError::InferenceUnavailable)?;
            let weights: &larql_inference::ModelWeights = &weights_guard;

            // The relevance background for residual scores is the panel's
            // residuals, not its embeddings; built here on first use since
            // this route owns inference.
            check_prompt(&params.prompt)?;
            if params.relevance {
                ensure_residual_panel(model, weights, patched, background, &params.prompt)?;
            }

            // The production inference path. Its per-layer residuals are the
            // vector each layer's gates were scored against for the last
            // position — exactly what DESCRIBE should be scoring against.
            let text = params.prompt.replace("{entity}", &params.entity);
            let mut by_layer = residuals_for(model, weights, patched, &text)?;

            if let Some(baseline) = params.baseline.as_deref() {
                let base = residuals_for(model, weights, patched, baseline)?;
                // Contrast per layer. A layer the baseline did not capture is
                // left as-is rather than dropped: the answer is then a plain
                // residual for that layer, which the caller can see from
                // `contrasted_layers`.
                // Only the layers that will be scored count. The forward
                // pass captures every layer of the model, and counting all of
                // them reported 35 contrasted against 14 scanned in
                // production — true, and useless to a caller checking
                // whether the answer it got was fully contrasted.
                for &layer in &scan_layers {
                    let (Some(v), Some(b)) = (by_layer.get_mut(&layer), base.get(&layer)) else {
                        continue;
                    };
                    if b.len() == v.len() {
                        for (x, y) in v.iter_mut().zip(b.iter()) {
                            *x -= *y;
                        }
                        contrasted_layers += 1;
                    }
                }
            }

            let mut layers = Vec::with_capacity(scan_layers.len());
            for &layer in &scan_layers {
                let Some(res) = by_layer.get(&layer) else {
                    continue;
                };
                let r = larql_vindex::ndarray::Array1::from(res.clone());
                let t = patched.walk(&r, &[layer], res_k(layer));
                layers.extend(t.layers);
            }
            residual_layers = Some(layers.len());
            larql_vindex::WalkTrace { layers }
        })
    } else {
        None
    };

    // Window by surprise: z-score every hit against the layer's background
    // and keep the `window` largest. A layer with no background keeps its
    // raw-score window (the first `window` hits, which is what walk gave).
    let surprise_window = |trace: &mut larql_vindex::WalkTrace, residual: bool| {
        for (layer_idx, hits) in trace.layers.iter_mut() {
            let stats = if residual {
                model
                    .relevance
                    .residual_layer(patched, background, &params.prompt, *layer_idx)
            } else {
                model.relevance.layer(patched, background, *layer_idx)
            };
            if let Some(stats) = stats {
                let mut scored: Vec<(f32, larql_vindex::WalkHit)> = hits
                    .drain(..)
                    .map(|h| {
                        (
                            stats
                                .z(h.feature, h.gate_score)
                                .unwrap_or(f32::NEG_INFINITY),
                            h,
                        )
                    })
                    .collect();
                scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                hits.extend(scored.into_iter().map(|(_, h)| h));
            }
            hits.truncate(params.window);
        }
    };
    if window_by_relevance {
        if let Some(t) = emb_trace.as_mut() {
            surprise_window(t, false);
        }
    }
    if res_surprise {
        if let Some(t) = res_trace.as_mut() {
            surprise_window(t, true);
        }
    }
    let sources: Vec<(&larql_vindex::WalkTrace, bool)> = emb_trace
        .iter()
        .map(|t| (t, false))
        .chain(res_trace.iter().map(|t| (t, true)))
        .collect();

    // Per-side calibration for a union. Measured 2026-09-05 (research log
    // §26): the residual side selects its window by z from every feature in
    // the layer, so its tail is the maximum of thousands of draws and sits in
    // a band at z≈4 by selection alone, above the embedding side's true hits
    // (French 4.5, cars 3.6). Within one request each side's z is therefore
    // re-expressed against its own window — (z − median) / MAD — so a hit
    // competes across sides by how far it stands above its own noise floor.
    // Median and MAD, not mean and std: the tails are what we are measuring.
    let calibration: std::collections::HashMap<bool, (f32, f32)> = if union {
        sources
            .iter()
            .map(|(trace, residual)| {
                let mut zs: Vec<f32> = trace
                    .layers
                    .iter()
                    .flat_map(|(layer_idx, hits)| {
                        let stats = if *residual {
                            model.relevance.residual_layer(
                                patched,
                                background,
                                &params.prompt,
                                *layer_idx,
                            )
                        } else {
                            model.relevance.layer(patched, background, *layer_idx)
                        };
                        hits.iter()
                            .filter(|h| *residual || h.gate_score >= params.min_score)
                            .filter_map(move |h| {
                                stats.as_ref().and_then(|s| s.z(h.feature, h.gate_score))
                            })
                            .collect::<Vec<f32>>()
                    })
                    .filter(|z| z.is_finite())
                    .collect();
                zs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let median = |v: &[f32]| if v.is_empty() { 0.0 } else { v[v.len() / 2] };
                let med = median(&zs);
                let mut dev: Vec<f32> = zs.iter().map(|z| (z - med).abs()).collect();
                dev.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                // 1.4826·MAD estimates σ for a normal core; the floor keeps a
                // degenerate window (all equal) from dividing by zero.
                let scale = (1.4826 * median(&dev)).max(1e-3);
                (*residual, (med, scale))
            })
            .collect()
    } else {
        std::collections::HashMap::new()
    };

    // Aggregate edges by target token (same logic as LQL DESCRIBE).
    struct EdgeInfo {
        gate: f32,
        layers: Vec<usize>,
        count: usize,
        original: String,
        also: Vec<String>,
        best_layer: usize,
        best_feature: usize,
        coherence: Option<f32>,
        relabelled: bool,
        /// Best z over the hits folded into this edge; `None` when no layer
        /// background was available for any of them.
        relevance: Option<f32>,
        /// Which query found the most surprising hit: `embedding` or
        /// `residual`. Meaningful for a union; constant otherwise.
        source: &'static str,
    }

    let entity_lower = params.entity.to_lowercase();
    // Label features in the script the caller wrote the entity in. A feature
    // that is plainly "Japan" was being labelled 일본 for a caller who typed
    // "Tokyo" — right cluster, unreadable spelling. `Other` (digits, symbols)
    // expresses no preference rather than an unsatisfiable one.
    let prefer = match crate::coherence::script_of(&params.entity) {
        crate::coherence::Script::Other => None,
        s => Some(s),
    };
    let mut edges: HashMap<String, EdgeInfo> = HashMap::new();

    for (trace, residual) in &sources {
        let residual = *residual;
        let source = if residual {
            QUERY_RESIDUAL
        } else {
            QUERY_EMBEDDING
        };
        for (layer_idx, hits) in &trace.layers {
            let layer_stats = if params.relevance && residual {
                model
                    .relevance
                    .residual_layer(patched, background, &params.prompt, *layer_idx)
            } else if params.relevance {
                model.relevance.layer(patched, background, *layer_idx)
            } else {
                None
            };
            for hit in hits {
                // `min_score` is an embedding-scale floor; a union's residual
                // hits are on another scale and are ranked by z instead.
                if hit.gate_score < params.min_score && !(union && residual) {
                    continue;
                }

                // Coherence, when asked for. The candidates are the positively
                // weighted top-k: a token the feature pushes DOWN is not part of
                // what the feature is about, and letting it into the centroid would
                // drag a perfectly coherent feature's score toward zero.
                let scoring = params.coherence || params.relabel || params.min_coherence > 0.0;
                let verdict = if scoring {
                    let cands: Vec<crate::coherence::Candidate<'_>> = hit
                        .meta
                        .top_k
                        .iter()
                        .filter(|t| t.logit > 0.0 && !t.token.trim().is_empty())
                        .map(|t| crate::coherence::Candidate {
                            token: t.token.trim(),
                            token_id: t.token_id,
                        })
                        .collect();
                    crate::coherence::score_feature(&model.embeddings, &cands, prefer)
                } else {
                    None
                };

                // A feature we could not score has not passed the bar; it has not
                // been measured at all. Dropping it only when a bar was actually
                // set keeps `min_coherence = 0` a pure reporting mode.
                if params.min_coherence > 0.0 {
                    match verdict.as_ref() {
                        Some(v) if v.coherence >= params.min_coherence => {}
                        _ => continue,
                    }
                }

                // The label only moves when asked. Scoring on its own must leave
                // every `target` byte-identical, or "adopt the filter" would
                // silently mean "rename the targets" as well.
                let z_raw = layer_stats
                    .as_ref()
                    .and_then(|st| st.z(hit.feature, hit.gate_score));
                let z = match (z_raw, calibration.get(&residual)) {
                    (Some(z), Some((med, scale))) => Some((z - med) / scale),
                    (z, _) => z,
                };

                let relabelled_to = if params.relabel {
                    verdict.as_ref().and_then(|v| v.label.clone())
                } else {
                    None
                };
                let tok: &str = relabelled_to
                    .as_deref()
                    .unwrap_or(hit.meta.top_token.as_str());
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
                    coherence: verdict.as_ref().map(|v| v.coherence),
                    relabelled: relabelled_to.is_some(),
                    relevance: z,
                    source,
                });

                // Relevance is a max over the folded hits, independent of which
                // hit wins on gate: the edge is as surprising as its most
                // surprising feature — and the source follows it.
                if let Some(zz) = z {
                    if entry.relevance.is_none_or(|cur| zz > cur) {
                        entry.source = source;
                    }
                    entry.relevance = Some(entry.relevance.map_or(zz, |cur| cur.max(zz)));
                }

                if hit.gate_score > entry.gate {
                    entry.gate = hit.gate_score;
                    entry.best_layer = *layer_idx;
                    entry.best_feature = hit.feature;
                    // The reported coherence must describe the feature the rest of
                    // the edge describes. Keeping the first one seen while gate,
                    // layer and feature move to a different hit would attach a
                    // score to a feature it was never computed from.
                    entry.coherence = verdict.as_ref().map(|v| v.coherence);
                    entry.relabelled = relabelled_to.is_some();
                }
                if !entry.layers.contains(layer_idx) {
                    entry.layers.push(*layer_idx);
                }
                entry.count += 1;
            }
        }
    }

    let mut ranked: Vec<&EdgeInfo> = edges.values().collect();
    ranked.sort_by(|a, b| {
        // Relevance first when asked for; an edge without a background sorts
        // below every edge with one, and ties (including the raw-score mode)
        // fall through to gate score.
        let by_rel = if params.relevance {
            match (b.relevance, a.relevance) {
                (Some(x), Some(y)) => x.partial_cmp(&y).unwrap_or(std::cmp::Ordering::Equal),
                (Some(_), None) => std::cmp::Ordering::Less,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (None, None) => std::cmp::Ordering::Equal,
            }
        } else {
            std::cmp::Ordering::Equal
        };
        by_rel.then_with(|| {
            b.gate
                .partial_cmp(&a.gate)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
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
                // The vindex coordinate behind the label. A caller that
                // resolves an edit by this pair is immune to the label
                // changing under it — which, with relabelling on by
                // default, it can.
                "feature": info.best_feature,
            });
            if params.query == QUERY_UNION {
                edge["source"] = serde_json::json!(info.source);
            }

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

            if params.relevance {
                edge["relevance"] = match info.relevance {
                    Some(z) => serde_json::json!((z * 100.0).round() / 100.0),
                    None => serde_json::Value::Null,
                };
            }

            if params.coherence || params.relabel || params.min_coherence > 0.0 {
                edge["coherence"] = match info.coherence {
                    Some(c) => serde_json::json!((c * 1000.0).round() / 1000.0),
                    // Explicitly null rather than absent: "could not be scored"
                    // is a different statement from "scored zero", and a
                    // consumer that cannot tell them apart will treat an
                    // unmeasured feature as a measured-bad one.
                    None => serde_json::Value::Null,
                };
                edge["label_source"] = serde_json::json!(if info.relabelled {
                    "centroid"
                } else {
                    "argmax"
                });
            }

            edge
        })
        .collect();

    let mut out = serde_json::json!({
        "entity": params.entity,
        "model": model.config.model,
        "edges": edge_json,
        "latency_ms": elapsed_ms(start),
    });
    if params.relevance {
        out["relevance_background"] = serde_json::json!(background.as_str());
        out["relevance_panel"] = serde_json::json!(if params.query == QUERY_RESIDUAL {
            model
                .relevance
                .residual_panel_size(background, &params.prompt)
        } else {
            model.relevance.panel_size(background)
        });
        if params.query == QUERY_UNION {
            out["residual_panel"] = serde_json::json!(model
                .relevance
                .residual_panel_size(background, &params.prompt));
            // What each side's z was re-expressed against, so a caller can
            // see the noise floor a union hit stood above.
            let side = |residual: bool| {
                calibration.get(&residual).map(|(m, s)| {
                    serde_json::json!({
                        "median": (m * 100.0).round() / 100.0,
                        "scale": (s * 100.0).round() / 100.0,
                    })
                })
            };
            out["calibration"] = serde_json::json!({
                "embedding": side(false),
                "residual": side(true),
            });
        }
        // Which population the z is against: the panel's embeddings or its
        // residuals. A caller comparing modes must see which it got.
        out["relevance_query"] = serde_json::json!(params.query);
        out["window_by"] = serde_json::json!(params.window_by);
    }
    if params.query == QUERY_RESIDUAL && params.prompt != PROMPT_BARE {
        out["prompt"] = serde_json::json!(params.prompt);
    }
    if params.query != QUERY_EMBEDDING {
        out["query"] = serde_json::json!(params.query);
        out["residual_layers"] = serde_json::json!(residual_layers);
        out["scanned_layers"] = serde_json::json!(scan_layers.len());
        if params.baseline.is_some() {
            out["contrasted_layers"] = serde_json::json!(contrasted_layers);
        }
    }
    Ok(out)
}

async fn describe_with_cache(
    patch_set: Option<crate::overlay_cache::PatchSetRef>,
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
    // Whatever identifies the OVERLAY is part of the key. It has to be: the
    // cached value is a *patched* view, so sharing one entry across overlays
    // would serve one tenant's suppressions to another — a cross-tenant leak
    // dressed up as a cache hit, and invisible in every log.
    //
    // That used to mean the session id, and when a request began carrying its
    // own patch set instead, the session slot went empty for every such caller.
    // Two workspaces with different edits would then have collided on one key.
    // The scope is therefore the patch set's own key when there is one, and the
    // session id only when there is not.
    let scope = match patch_set.as_ref() {
        Some(ps) => Some(ps.key(&model.id, sid.as_deref())),
        None => sid.clone(),
    };
    let cache_key = if state.describe_cache.is_enabled() {
        let key = crate::cache::DescribeCache::key_scoped(
            scope.as_deref(),
            &model.id,
            &params.entity,
            &params.band,
            params.limit,
            params.window,
            params.min_score,
            params.coherence,
            params.min_coherence,
            params.relabel,
            params.relevance,
            // The RESOLVED background: an absent one takes the deployment's
            // default, and if that default changes the old entries must miss.
            params
                .background
                .as_deref()
                .and_then(crate::relevance::Background::parse)
                .unwrap_or(model.relevance.default_background())
                .as_str(),
            &params.window_by,
            &{
                let mut q = params.query.clone();
                if params.prompt != PROMPT_BARE {
                    q.push('|');
                    q.push_str(&params.prompt);
                }
                match params.baseline.as_deref() {
                    Some(b) => format!("{q}~{b}"),
                    None => q,
                }
            },
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
        crate::overlay_cache::with_overlay(
            &task_state,
            &model,
            sid.as_deref(),
            patch_set.as_ref(),
            |patched| describe_entity_with(&model, patched, &params),
        )?
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
    describe_with_cache(None, &state, &model, &headers, params).await
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
    describe_with_cache(None, &state, &model, &headers, params).await
}

// ═══════════════════════════════════════════════════════════════
// POST /v1/describe — the same browse, carrying its own patch set
// ═══════════════════════════════════════════════════════════════

/// Body for `POST /v1/describe`.
///
/// A patch set cannot travel in a query string: it is unbounded in size, and
/// putting a tenant's edits in a URL would write them into every access log and
/// proxy along the way. So the content-addressed form of DESCRIBE is a POST with
/// the same semantics as the GET, plus `patch_set`.
///
/// The GET keeps working untouched, which is what lets a client migrate one call
/// site at a time.
#[derive(Deserialize, utoipa::ToSchema)]
pub struct DescribeBody {
    pub entity: String,
    #[serde(default = "default_band")]
    pub band: String,
    #[serde(default)]
    pub verbose: bool,
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// See `DescribeParams::window`.
    #[serde(default = "default_window")]
    pub window: usize,
    #[serde(default = "default_min_score")]
    pub min_score: f32,
    /// See `DescribeParams::coherence`.
    #[serde(default = "default_true")]
    pub coherence: bool,
    /// See `DescribeParams::min_coherence`.
    #[serde(default = "default_min_coherence")]
    pub min_coherence: f32,
    /// See `DescribeParams::relabel`.
    #[serde(default = "default_true")]
    pub relabel: bool,
    /// See `DescribeParams::relevance`.
    #[serde(default = "default_true")]
    pub relevance: bool,
    /// See `DescribeParams::background`.
    #[serde(default)]
    pub background: Option<String>,
    /// See `DescribeParams::window_by`.
    #[serde(default = "default_window_by")]
    pub window_by: String,
    /// See `DescribeParams::query`.
    #[serde(default = "default_query")]
    pub query: String,
    /// See `DescribeParams::prompt`.
    #[serde(default = "default_prompt")]
    pub prompt: String,
    /// See `DescribeParams::baseline`.
    #[serde(default)]
    pub baseline: Option<String>,
    /// The overlay to answer from. Omitted, this behaves exactly like the GET.
    #[serde(default)]
    pub patch_set: Option<crate::overlay_cache::PatchSetRef>,
}

impl DescribeBody {
    fn split(self) -> (DescribeParams, Option<crate::overlay_cache::PatchSetRef>) {
        let DescribeBody {
            entity,
            band,
            verbose,
            limit,
            window,
            min_score,
            coherence,
            min_coherence,
            relabel,
            relevance,
            background,
            window_by,
            query,
            prompt,
            baseline,
            patch_set,
        } = self;
        (
            DescribeParams {
                entity,
                band,
                verbose,
                limit,
                window,
                min_score,
                coherence,
                min_coherence,
                relabel,
                relevance,
                background,
                window_by,
                query,
                prompt,
                baseline,
            },
            patch_set,
        )
    }
}

#[utoipa::path(
    post,
    path = "/v1/describe",
    tag = "browse",
    request_body = DescribeBody,
    responses(
        (status = 200, body = crate::openapi::schemas::DescribeResponse),
        (status = 400, body = crate::error::ErrorBody),
        (status = 409, description = "patch_set_unknown — retry with `patches` inline",
         body = crate::error::ErrorBody),
        (status = 500, body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_describe_post(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<DescribeBody>,
) -> Result<Response, ServerError> {
    state.bump_requests();
    let model = state.model_or_err(None)?;
    let (params, patch_set) = body.split();
    describe_with_cache(patch_set, &state, &model, &headers, params).await
}

#[utoipa::path(
    post,
    path = "/v1/{model_id}/describe",
    tag = "browse",
    params(("model_id" = String, Path, description = "Id of a loaded vindex.")),
    request_body = DescribeBody,
    responses(
        (status = 200, body = crate::openapi::schemas::DescribeResponse),
        (status = 404, body = crate::error::ErrorBody),
        (status = 409, body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_describe_post_multi(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<String>,
    headers: HeaderMap,
    Json(body): Json<DescribeBody>,
) -> Result<Response, ServerError> {
    state.bump_requests();
    let model = state.model_or_err(Some(&model_id))?;
    let (params, patch_set) = body.split();
    describe_with_cache(patch_set, &state, &model, &headers, params).await
}
