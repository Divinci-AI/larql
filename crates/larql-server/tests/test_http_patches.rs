//! HTTP integration tests: patches apply/list/delete (global + session-scoped).

mod common;
use common::*;

use axum::http::StatusCode;

// ══════════════════════════════════════════════════════════════
// GET /v1/patches  •  DELETE /v1/patches/{name}
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_patches_list_empty_returns_empty_array() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/patches").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    let patches = body["patches"].as_array().unwrap();
    assert!(patches.is_empty());
}

#[tokio::test]
async fn http_patches_delete_nonexistent_returns_404() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = delete(app, "/v1/patches/nonexistent-patch").await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn http_patches_session_list_returns_session_field() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = get_h(app, "/v1/patches", ("x-session-id", "sess-abc")).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["session"], "sess-abc");
    assert!(body["patches"].as_array().unwrap().is_empty());
}

// ══════════════════════════════════════════════════════════════
// POST /v1/patches/apply  •  GET /v1/patches  •  DELETE /v1/patches/{name}
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn http_patches_apply_no_url_no_patch_returns_400() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(app, "/v1/patches/apply", serde_json::json!({})).await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = body_json(resp.into_body()).await;
    assert!(body["error"].as_str().unwrap().contains("url"));
}

#[tokio::test]
async fn http_patches_apply_inline_returns_200() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(app, "/v1/patches/apply", inline_delete_patch("my-patch")).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["applied"], "my-patch");
    assert!(body["active_patches"].as_u64().is_some());
}

#[tokio::test]
async fn http_patches_list_after_apply_shows_patch() {
    let st = state(vec![model("test")]);
    // Apply the patch.
    let app1 = single_model_router(st.clone());
    post_json(
        app1,
        "/v1/patches/apply",
        inline_delete_patch("visible-patch"),
    )
    .await;
    // List patches.
    let app2 = single_model_router(st.clone());
    let resp = get(app2, "/v1/patches").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    let patches = body["patches"].as_array().unwrap();
    assert!(patches.iter().any(|p| p["name"] == "visible-patch"));
}

#[tokio::test]
async fn http_patches_delete_named_returns_200() {
    let st = state(vec![model("test")]);
    // Apply, then delete.
    let app1 = single_model_router(st.clone());
    post_json(app1, "/v1/patches/apply", inline_delete_patch("to-delete")).await;
    let app2 = single_model_router(st.clone());
    let resp = delete(app2, "/v1/patches/to-delete").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["removed"], "to-delete");
    assert!(body["active_patches"].as_u64().is_some());
}

#[tokio::test]
async fn http_patches_session_apply_returns_session_field() {
    // apply_patch uses blocking_read when creating a new session inside an async
    // write-lock guard, which panics. Pre-create the session via get_or_create
    // (uses read().await, safe) so the entry already exists when the HTTP handler
    // calls apply_patch, skipping the blocking_read path entirely.
    let st = state(vec![model("test")]);
    let m = st.first_model().unwrap();
    st.sessions.get_or_create("sid-abc", &m).await;

    let app = single_model_router(st);
    let resp = post_json_h(
        app,
        "/v1/patches/apply",
        inline_delete_patch("sess-patch"),
        ("x-session-id", "sid-abc"),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["session"], "sid-abc");
    assert!(body["active_patches"].as_u64().is_some());
}

#[tokio::test]
async fn http_patches_session_list_after_session_apply() {
    let st = state(vec![model("test")]);
    let m = st.first_model().unwrap();
    st.sessions.get_or_create("sid-list", &m).await;

    let app1 = single_model_router(st.clone());
    post_json_h(
        app1,
        "/v1/patches/apply",
        inline_delete_patch("session-visible"),
        ("x-session-id", "sid-list"),
    )
    .await;
    let app2 = single_model_router(st.clone());
    let resp = get_h(app2, "/v1/patches", ("x-session-id", "sid-list")).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["session"], "sid-list");
    let patches = body["patches"].as_array().unwrap();
    assert!(patches.iter().any(|p| p["name"] == "session-visible"));
}

#[tokio::test]
async fn http_patches_multi_model_apply_not_found_returns_404() {
    let app = multi_model_router(state(vec![model("a")]));
    let resp = post_json(
        app,
        "/v1/nosuchmodel/patches/apply",
        inline_delete_patch("p"),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

// ══════════════════════════════════════════════════════════════
// Top-level `name` on POST /v1/patches/apply
//
// The body shape real clients send is `{"name": ..., "patch": {...}}` with
// no `description` inside the patch. `name` used to be dropped by serde, so
// the patch landed in the stack as "unnamed" while the response reported the
// caller's name back — and the matching DELETE then 404'd. These pin the
// round trip: whatever `applied` says is the key that lists and deletes.
// ══════════════════════════════════════════════════════════════

/// A patch body in the shape a client sends: name at the top level, no
/// `description` inside the patch itself.
fn named_patch_no_description(name: &str) -> serde_json::Value {
    serde_json::json!({
        "name": name,
        "patch": {
            "version": 1,
            "base_model": "test",
            "base_checksum": null,
            "created_at": "2026-08-28",
            "description": null,
            "author": null,
            "tags": [],
            "operations": [
                {"op": "delete", "layer": 0, "feature": 2}
            ]
        }
    })
}

#[tokio::test]
async fn http_patches_apply_honors_top_level_name() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(
        app,
        "/v1/patches/apply",
        named_patch_no_description("wl-42"),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["applied"], "wl-42");
}

#[tokio::test]
async fn http_patches_list_shows_top_level_name_not_unnamed() {
    let st = state(vec![model("test")]);
    let app1 = single_model_router(st.clone());
    post_json(
        app1,
        "/v1/patches/apply",
        named_patch_no_description("wl-listed"),
    )
    .await;

    let app2 = single_model_router(st.clone());
    let body = body_json(get(app2, "/v1/patches").await.into_body()).await;
    let patches = body["patches"].as_array().unwrap();
    assert!(
        patches.iter().any(|p| p["name"] == "wl-listed"),
        "expected the supplied name in the stack, got {patches:?}"
    );
}

#[tokio::test]
async fn http_patches_delete_by_top_level_name_succeeds() {
    let st = state(vec![model("test")]);
    let app1 = single_model_router(st.clone());
    post_json(
        app1,
        "/v1/patches/apply",
        named_patch_no_description("wl-revert"),
    )
    .await;

    // The revert a client issues: DELETE the name it was told was applied,
    // with no intervening GET /v1/patches to discover the real key.
    let app2 = single_model_router(st.clone());
    let resp = delete(app2, "/v1/patches/wl-revert").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["removed"], "wl-revert");

    let app3 = single_model_router(st.clone());
    let body = body_json(get(app3, "/v1/patches").await.into_body()).await;
    assert!(body["patches"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn http_patches_two_named_patches_delete_the_right_one() {
    // The case the caller-side "list, and if there is exactly one, delete it"
    // workaround cannot serve: two patches active at once.
    let st = state(vec![model("test")]);
    for name in ["wl-first", "wl-second"] {
        let app = single_model_router(st.clone());
        post_json(app, "/v1/patches/apply", named_patch_no_description(name)).await;
    }

    let app = single_model_router(st.clone());
    assert_eq!(
        delete(app, "/v1/patches/wl-first").await.status(),
        StatusCode::OK
    );

    let app = single_model_router(st.clone());
    let body = body_json(get(app, "/v1/patches").await.into_body()).await;
    let patches = body["patches"].as_array().unwrap();
    assert_eq!(patches.len(), 1);
    assert_eq!(patches[0]["name"], "wl-second");
}

#[tokio::test]
async fn http_patches_top_level_name_overrides_patch_description() {
    let st = state(vec![model("test")]);
    let mut body = inline_delete_patch("from-description");
    body["name"] = serde_json::json!("from-name");

    let app1 = single_model_router(st.clone());
    let resp = body_json(post_json(app1, "/v1/patches/apply", body).await.into_body()).await;
    assert_eq!(resp["applied"], "from-name");

    let app2 = single_model_router(st.clone());
    let listed = body_json(get(app2, "/v1/patches").await.into_body()).await;
    assert_eq!(listed["patches"][0]["name"], "from-name");
}

#[tokio::test]
async fn http_patches_session_delete_by_top_level_name_succeeds() {
    let st = state(vec![model("test")]);
    let m = st.model(None).unwrap();
    st.sessions.get_or_create("sid-named", &m).await;

    let app1 = single_model_router(st.clone());
    post_json_h(
        app1,
        "/v1/patches/apply",
        named_patch_no_description("sess-revert"),
        ("x-session-id", "sid-named"),
    )
    .await;

    let app2 = single_model_router(st.clone());
    let listed = body_json(
        get_h(app2, "/v1/patches", ("x-session-id", "sid-named"))
            .await
            .into_body(),
    )
    .await;
    assert!(listed["patches"]
        .as_array()
        .unwrap()
        .iter()
        .any(|p| p["name"] == "sess-revert"));
}

#[tokio::test]
async fn http_patches_apply_without_name_still_falls_back() {
    // No `name`, no `description` — the old fallback must survive so bodies
    // that never carried a name keep working.
    let app = single_model_router(state(vec![model("test")]));
    let mut body = named_patch_no_description("ignored");
    body.as_object_mut().unwrap().remove("name");
    let resp = post_json(app, "/v1/patches/apply", body).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["applied"], "inline-patch");
}
