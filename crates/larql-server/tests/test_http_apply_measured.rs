//! HTTP integration tests: POST /v1/patches/apply_measured.
//!
//! The route-shape tests here are not ceremony. LarQL already owns one route
//! that looks like an endpoint and is not: `POST /v1/patches/replay` matches
//! `/v1/patches/{name}` — a DELETE-only route — so it answers 405 rather than
//! 404. A client shipped a fallback keyed on 404, the 405 fell through, and
//! batch patch replay was dead code from the day it was written. Nobody
//! noticed until an edit silently failed to come back in production.
//!
//! `apply_measured` sits under the same prefix and would be swallowed the same
//! way if it were ever registered after the `{name}` route. These pin that it
//! is not.

mod common;
use common::*;

use axum::http::StatusCode;

fn delete_patch(layer: usize, feature: usize) -> serde_json::Value {
    serde_json::json!({
        "version": 1,
        "base_model": "test",
        "created_at": "2026-08-31T00:00:00Z",
        "operations": [{
            "op": "delete",
            "entity": "France",
            "relation": "associated-with",
            "target": "Paris",
            "weight": 1.0,
            "layer": layer,
            "feature": feature,
        }],
    })
}

fn measure_spec() -> serde_json::Value {
    serde_json::json!({ "prompt": "France", "top": 5 })
}

// ══════════════════════════════════════════════════════════════
// Route shape — the 405 class of bug
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn apply_measured_is_not_swallowed_by_the_patch_name_route() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(app, "/v1/patches/apply_measured", serde_json::json!({})).await;

    // An empty body is a *body* problem, so the route matched and axum got as
    // far as deserializing. The failure this guards against is 405 METHOD NOT
    // ALLOWED, which is what `/v1/patches/{name}` answers for a POST — that
    // would mean the handler is unreachable no matter what body is sent.
    assert_ne!(
        resp.status(),
        StatusCode::METHOD_NOT_ALLOWED,
        "apply_measured is being routed to /v1/patches/{{name}}"
    );
    assert_ne!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn the_replay_path_still_405s_which_is_why_the_test_above_exists() {
    // Documents the live collision rather than asserting a wish. If LarQL ever
    // grows a real batch-replay endpoint this test fails, which is the right
    // moment to revisit the client fallback that treats 404/405/501 alike.
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(app, "/v1/patches/replay", serde_json::json!({})).await;
    assert_eq!(resp.status(), StatusCode::METHOD_NOT_ALLOWED);
}

// ══════════════════════════════════════════════════════════════
// The bracket
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn returns_both_sides_of_the_bracket_in_one_response() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(
        app,
        "/v1/patches/apply_measured",
        serde_json::json!({
            "name": "edit-1",
            "patch": delete_patch(1, 150),
            "measure": measure_spec(),
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);

    let body = body_json(resp.into_body()).await;
    assert_eq!(body["applied"], "edit-1");
    assert_eq!(body["operations"], 1);
    assert_eq!(body["atomic_scope"], "single-instance-single-request");

    // One response carries both sides, so their difference cannot be an
    // artifact of routing. This fixture has no model weights, so the readings
    // themselves are null — see the graceful-degradation test below. The shape
    // is what is pinned here; the values are exercised against the real
    // tiny-vindex in scripts/local-infer-env.sh.
    let measured = &body["measured"];
    assert!(measured.get("before").is_some());
    assert!(measured.get("after").is_some());
    assert_eq!(measured["prompt"], "France");
}

// The apply is the durable half; the reading is commentary on it. A fixture
// with no weights cannot run a forward pass — and must still land the edit,
// because "apply this and tell me what it did" should not degrade to doing
// neither on a service whose inference half can be disabled with --no-infer.
#[tokio::test]
async fn an_unmeasurable_edit_is_still_applied_and_says_so() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json_h(
        app.clone(),
        "/v1/patches/apply_measured",
        serde_json::json!({
            "name": "edit-1",
            "patch": delete_patch(1, 150),
            "measure": measure_spec(),
        }),
        ("x-session-id", "wl:tenant-a"),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);

    let body = body_json(resp.into_body()).await;
    assert_eq!(body["active_patches"], 1, "the edit must still have landed");

    // null is "not measured". A caller that read this as "unchanged" would
    // record an edit as having done nothing — the exact confusion that a probe
    // returning 0 for an absent target once put into a compliance dossier.
    assert!(body["measured"]["before"].is_null());
    assert!(body["measured"]["after"].is_null());
    assert!(
        body["measured"]["measure_error"].is_string(),
        "an unmeasured reading must say why"
    );

    // And the patch really is resident, not merely reported.
    let listed = get_h(app, "/v1/patches", ("x-session-id", "wl:tenant-a")).await;
    let listed = body_json(listed.into_body()).await;
    assert_eq!(listed["patches"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn scopes_the_apply_and_both_readings_to_the_session() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json_h(
        app,
        "/v1/patches/apply_measured",
        serde_json::json!({
            "name": "edit-1",
            "patch": delete_patch(1, 150),
            "measure": measure_spec(),
        }),
        ("x-session-id", "wl:tenant-a"),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);

    let body = body_json(resp.into_body()).await;
    // A measured apply that wrote to global state would rewrite the overlay
    // every other tenant is served from — the cross-tenant write that session
    // scoping exists to prevent.
    assert_eq!(body["session"], "wl:tenant-a");
    assert_eq!(body["active_patches"], 1);
}

#[tokio::test]
async fn files_the_patch_under_its_name_so_it_can_be_reverted() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json_h(
        app.clone(),
        "/v1/patches/apply_measured",
        serde_json::json!({
            "name": "edit-abc",
            "patch": delete_patch(1, 150),
            "measure": measure_spec(),
        }),
        ("x-session-id", "wl:tenant-a"),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);

    // Reporting one name while filing under another is what once made
    // DELETE /v1/patches/{name} 404 for callers that had done nothing wrong.
    let listed = get_h(app, "/v1/patches", ("x-session-id", "wl:tenant-a")).await;
    let body = body_json(listed.into_body()).await;
    let names: Vec<&str> = body["patches"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|p| p["name"].as_str())
        .collect();
    assert!(names.contains(&"edit-abc"), "listed as {names:?}");
}

#[tokio::test]
async fn a_missing_measure_block_is_rejected_rather_than_silently_skipped() {
    // A measured apply with nothing measured is the silent-no-op shape this
    // whole endpoint exists to eliminate; it must not be reachable.
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(
        app,
        "/v1/patches/apply_measured",
        serde_json::json!({ "patch": delete_patch(1, 150) }),
    )
    .await;
    assert!(
        resp.status().is_client_error(),
        "expected a 4xx, got {}",
        resp.status()
    );
}
