//! HTTP integration tests: describe endpoint (all band variants, verbose,
//! cache, ETag, multi-model).

mod common;
use common::*;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

// ══════════════════════════════════════════════════════════════
// GET /v1/describe
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_describe_returns_200_with_entity_field() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/describe?entity=France").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["entity"], "France");
    assert!(body["edges"].is_array());
    assert!(body["latency_ms"].as_f64().is_some());
}

#[tokio::test]
async fn http_describe_empty_vocab_returns_empty_edges() {
    // Empty BPE tokenizer → empty token_ids → graceful empty response.
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/describe?entity=Germany").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["edges"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn http_describe_missing_entity_returns_400() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/describe").await; // no entity param
                                               // axum rejects the missing required query param
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

// ══════════════════════════════════════════════════════════════
// Band variants
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_describe_band_syntax_returns_200() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/describe?entity=France&band=syntax").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["entity"], "France");
    assert!(body["edges"].is_array());
}

#[tokio::test]
async fn http_describe_band_output_returns_200() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/describe?entity=France&band=output").await;
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn http_describe_band_all_returns_200() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/describe?entity=France&band=all").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert!(body["edges"].is_array());
}

#[tokio::test]
async fn http_describe_verbose_mode_returns_200() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/describe?entity=France&verbose=true").await;
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn http_describe_empty_entity_returns_empty_edges() {
    // Empty tokenizer → empty token ids → early return with edges=[].
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/describe?entity=hello").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    // Empty BPE → no token ids → describe_entity returns edges=[].
    assert!(body["edges"].is_array());
}

// ══════════════════════════════════════════════════════════════
// ETag and cache
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_describe_has_etag_header() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/describe?entity=France").await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert!(resp.headers().contains_key("etag"));
}

#[tokio::test]
async fn http_describe_cache_hit_returns_cached_response() {
    let st = state_with_cache(vec![model("test")], 100);
    // First request populates cache.
    let app1 = single_model_router(st.clone());
    let r1 = get(app1, "/v1/describe?entity=France").await;
    assert_eq!(r1.status(), StatusCode::OK);
    let etag = r1.headers()["etag"].to_str().unwrap().to_string();

    // Second request — same key, cache enabled — returns cached with same etag.
    let app2 = single_model_router(st.clone());
    let r2 = get(app2, "/v1/describe?entity=France").await;
    assert_eq!(r2.status(), StatusCode::OK);
    assert_eq!(r2.headers()["etag"].to_str().unwrap(), etag);
}

#[tokio::test]
async fn http_describe_if_none_match_returns_304() {
    let st = state_with_cache(vec![model("test")], 100);
    // Get etag from first request.
    let app1 = single_model_router(st.clone());
    let r1 = get(app1, "/v1/describe?entity=France").await;
    let etag = r1.headers()["etag"].to_str().unwrap().to_string();

    // Second request with If-None-Match → 304.
    let app2 = single_model_router(st.clone());
    let resp = app2
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/describe?entity=France")
                .header("if-none-match", &etag)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_MODIFIED);
}

// ══════════════════════════════════════════════════════════════
// Multi-model describe
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_describe_multi_model_returns_200() {
    let app = multi_model_router(state(vec![model("a"), model("b")]));
    let resp = get(app, "/v1/a/describe?entity=France").await;
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn http_describe_multi_model_not_found_returns_404() {
    let app = multi_model_router(state(vec![model("a")]));
    let resp = get(app, "/v1/nosuchmodel/describe?entity=France").await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

// ══════════════════════════════════════════════════════════════
// The ranking parameters over HTTP: background, window_by, query, prompt,
// and the size caps. Each is a serde default plus a validation, and each
// exists in the GET query and the POST body. Two invalid measurements in
// research log §22/§25 were parameters that never reached the function;
// these pin that they do, and are refused the same way by either route.
// ══════════════════════════════════════════════════════════════

async fn describe_get(q: &str) -> (StatusCode, serde_json::Value) {
    let app = single_model_router(state(vec![model_functional("test")]));
    let resp = get(app, &format!("/v1/describe?entity=France{q}")).await;
    let status = resp.status();
    (status, body_json(resp.into_body()).await)
}

async fn describe_post(extra: serde_json::Value) -> (StatusCode, serde_json::Value) {
    let app = single_model_router(state(vec![model_functional("test")]));
    let mut body = serde_json::json!({"entity": "France"});
    for (k, v) in extra.as_object().unwrap() {
        body[k] = v.clone();
    }
    let resp = post_json(app, "/v1/describe", body).await;
    let status = resp.status();
    (status, body_json(resp.into_body()).await)
}

fn err_text(body: &serde_json::Value) -> String {
    body.to_string()
}

#[tokio::test]
async fn http_describe_background_is_forwarded_and_reported() {
    let (status, body) = describe_get("&relevance=true&background=vocabulary").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["relevance_background"], "vocabulary");
    let (status, body) =
        describe_post(serde_json::json!({"relevance": true, "background": "vocabulary"})).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["relevance_background"], "vocabulary");
}

#[tokio::test]
async fn http_describe_rejects_an_unknown_background() {
    let (status, body) = describe_get("&relevance=true&background=cities").await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(err_text(&body).contains("background"), "{body}");
    let (status, _) =
        describe_post(serde_json::json!({"relevance": true, "background": "cities"})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_describe_window_by_is_forwarded_and_reported() {
    let (status, body) = describe_get("&relevance=true&window_by=relevance").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["window_by"], "relevance");
    let (status, body) =
        describe_post(serde_json::json!({"relevance": true, "window_by": "relevance"})).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["window_by"], "relevance");
}

#[tokio::test]
async fn http_describe_rejects_window_by_relevance_without_relevance() {
    let (status, body) = describe_get("&relevance=false&window_by=relevance").await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(err_text(&body).contains("relevance=true"), "{body}");
    let (status, _) =
        describe_post(serde_json::json!({"relevance": false, "window_by": "relevance"})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_describe_rejects_an_unknown_window_by() {
    let (status, body) = describe_get("&relevance=true&window_by=magic").await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(err_text(&body).contains("window_by"), "{body}");
}

#[tokio::test]
async fn http_describe_rejects_an_unknown_query() {
    let (status, body) = describe_get("&query=both").await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(err_text(&body).contains("query must be"), "{body}");
    let (status, _) = describe_post(serde_json::json!({"query": "both"})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_describe_rejects_a_union_without_relevance() {
    let (status, body) = describe_get("&relevance=false&query=union").await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(err_text(&body).contains("relevance"), "{body}");
    let (status, _) =
        describe_post(serde_json::json!({"relevance": false, "query": "union"})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_describe_rejects_a_prompt_on_the_embedding_query() {
    let (status, body) = describe_get("&prompt=%7Bentity%7D%20is").await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(err_text(&body).contains("residual"), "{body}");
    let (status, _) = describe_post(serde_json::json!({"prompt": "{entity} is"})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_describe_rejects_a_prompt_without_the_entity_placeholder() {
    // Refused before the weights check: a bad prompt is a bad request on any model.
    let (status, body) = describe_get("&query=residual&prompt=the%20capital%20of").await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(err_text(&body).contains("{entity}"), "{body}");
    let (status, _) =
        describe_post(serde_json::json!({"query": "residual", "prompt": "the capital of"})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_describe_rejects_a_prompt_template_the_operator_did_not_enable() {
    let (status, body) =
        describe_get("&query=residual&relevance=true&prompt=%7Bentity%7D%20is%20known%20for").await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(err_text(&body).contains("not enabled"), "{body}");
    let (status, _) = describe_post(serde_json::json!({
        "query": "residual", "relevance": true, "prompt": "{entity} is known for"
    }))
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_describe_caps_limit_and_window() {
    let max_l = larql_server::routes::describe::MAX_LIMIT;
    let max_w = larql_server::routes::describe::MAX_WINDOW;
    let (status, _) = describe_get(&format!("&limit={max_l}")).await;
    assert_eq!(status, StatusCode::OK);
    let (status, body) = describe_get(&format!("&limit={}", max_l + 1)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(err_text(&body).contains("limit"), "{body}");
    let (status, _) = describe_get(&format!("&window={max_w}")).await;
    assert_eq!(status, StatusCode::OK);
    let (status, body) = describe_get(&format!("&window={}", max_w + 1)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(err_text(&body).contains("window"), "{body}");
    let (status, _) = describe_post(serde_json::json!({"limit": max_l + 1})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let (status, _) = describe_post(serde_json::json!({"window": max_w + 1})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_describe_validates_before_the_empty_entity_shortcut() {
    // An entity the tokenizer cannot encode used to return empty edges
    // before any parameter was looked at; a bad parameter is bad regardless.
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/describe?entity=France&window_by=magic").await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}
