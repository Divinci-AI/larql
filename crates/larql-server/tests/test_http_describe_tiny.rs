//! The ranking parameters over HTTP against the tiny fixture, weights
//! included. `test_http_describe.rs` sends every parameter through both
//! routes but on a model without weights, so the residual paths return
//! early there; `test_patch_set_values.rs` runs them on this fixture but
//! calls the function. This is the one place a residual or union query is
//! handled end to end: serde, validation, weights, panel build, response.

mod common;
use common::*;

use std::path::Path;
use std::sync::Arc;

use axum::http::StatusCode;
use larql_server::bootstrap::load::{load_single_vindex, LoadVindexOptions};
use larql_server::state::LoadedModel;

fn tiny() -> Arc<LoadedModel> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-vindex");
    let opts = LoadVindexOptions {
        no_infer: true,
        ..Default::default()
    };
    Arc::new(load_single_vindex(root.to_str().unwrap(), opts).expect("tiny-vindex loads"))
}

/// The fixture is tracked, with weights. Forty fixture tests skip quietly
/// when it is missing; this one does not, so a lost fixture is a red build
/// rather than a green one that checked nothing.
#[test]
fn the_tiny_fixture_is_present_and_carries_weights() {
    let m = tiny();
    assert!(
        m.config.has_model_weights,
        "tiny-vindex must ship model_weights.bin"
    );
}

#[tokio::test]
async fn http_residual_query_with_relevance_is_answered_end_to_end() {
    let app = single_model_router(state(vec![tiny()]));
    let resp = get(
        app,
        // The fixture's synthetic labels score below the default coherence
        // floor, so a browse that kept it would be empty for reasons that
        // have nothing to do with what this test is about.
        "/v1/describe?entity=%5B5%5D&band=all&query=residual&min_score=0\
&window_by=relevance&coherence=false&min_coherence=0&relabel=false",
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["relevance_query"], "residual");
    assert_eq!(body["window_by"], "relevance");
    assert!(body["relevance_panel"].as_u64().unwrap() >= 2, "{body}");
    assert!(body["residual_layers"].as_u64().unwrap() >= 1, "{body}");
    assert!(!body["edges"].as_array().unwrap().is_empty(), "{body}");
}

#[tokio::test]
async fn http_union_query_over_post_reports_both_sides() {
    let app = single_model_router(state(vec![tiny()]));
    let resp = post_json(
        app,
        "/v1/describe",
        serde_json::json!({
            "entity": "[5]", "band": "all", "query": "union", "min_score": 0,
            "limit": 10000, "window": 10000,
            "coherence": false, "min_coherence": 0, "relabel": false
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["query"], "union");
    assert!(body["residual_panel"].as_u64().unwrap() >= 2, "{body}");
    assert!(body["calibration"]["embedding"]["scale"].as_f64().unwrap() > 0.0);
    assert!(body["calibration"]["residual"]["scale"].as_f64().unwrap() > 0.0);
    let sources: std::collections::BTreeSet<String> = body["edges"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["source"].as_str().unwrap().to_string())
        .collect();
    assert!(!sources.is_empty());
    assert!(
        sources.iter().all(|s| s == "embedding" || s == "residual"),
        "{sources:?}"
    );
}

#[tokio::test]
async fn http_residual_query_over_too_many_tokens_is_a_400() {
    let app = single_model_router(state(vec![tiny()]));
    let long = std::iter::repeat_n(
        "%5B5%5D",
        larql_server::routes::describe::MAX_RESIDUAL_TOKENS + 8,
    )
    .collect::<Vec<_>>()
    .join("%20");
    let resp = get(
        app,
        &format!("/v1/describe?entity={long}&query=residual&min_score=0"),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = body_json(resp.into_body()).await;
    assert!(body.to_string().contains("tokens"), "{body}");
}

#[tokio::test]
async fn http_embedding_query_over_the_same_text_is_not_capped() {
    let app = single_model_router(state(vec![tiny()]));
    let long = std::iter::repeat_n(
        "%5B5%5D",
        larql_server::routes::describe::MAX_RESIDUAL_TOKENS + 8,
    )
    .collect::<Vec<_>>()
    .join("%20");
    let resp = get(app, &format!("/v1/describe?entity={long}&min_score=0")).await;
    assert_eq!(resp.status(), StatusCode::OK);
}
