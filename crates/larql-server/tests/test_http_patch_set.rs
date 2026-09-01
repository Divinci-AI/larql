//! HTTP integration tests: requests that carry their own patch set.
//!
//! The property under test is that an instance which has been told *nothing*
//! answers correctly, because the request carried what it needed. That is what
//! makes instances interchangeable, and it is the thing the session-based design
//! could not do: an edit applied to instance A did not exist on B or C, and
//! nothing in a later request said which instance it needed.

mod common;
use common::*;

use axum::http::StatusCode;

fn delete_op(layer: usize, feature: usize) -> serde_json::Value {
    serde_json::json!({
        "version": 1,
        "base_model": "test",
        "created_at": "2026-09-01T00:00:00Z",
        "description": "e1",
        "operations": [{ "op": "delete", "layer": layer, "feature": feature }],
    })
}

fn describe_body(patch_set: Option<serde_json::Value>) -> serde_json::Value {
    let mut b = serde_json::json!({
        "entity": "France",
        "band": "all",
        "limit": 10,
        "min_score": 0.0,
    });
    if let Some(ps) = patch_set {
        b["patch_set"] = ps;
    }
    b
}

// ══════════════════════════════════════════════════════════════
// The core property
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn a_request_carrying_a_patch_set_needs_no_prior_state() {
    let app = single_model_router(state(vec![model("test")]));

    // No apply, no session, nothing this instance was told beforehand.
    let resp = post_json(
        app,
        "/v1/describe",
        describe_body(Some(serde_json::json!({ "patches": [delete_op(1, 150)] }))),
    )
    .await;

    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn carrying_a_patch_set_leaves_the_instance_holding_nothing() {
    let app = single_model_router(state(vec![model("test")]));

    let resp = post_json(
        app.clone(),
        "/v1/describe",
        describe_body(Some(serde_json::json!({ "patches": [delete_op(1, 150)] }))),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);

    // The old design would have had to mutate something to answer that. Nothing
    // is resident, which is precisely why any instance can serve any tenant and
    // why a redeploy cannot lose an edit.
    let listed = get(app, "/v1/patches").await;
    let body = body_json(listed.into_body()).await;
    assert!(body["patches"].as_array().unwrap().is_empty());
}

// ══════════════════════════════════════════════════════════════
// Isolation — the failure a shared result cache would reintroduce
// ══════════════════════════════════════════════════════════════

// NOTE ON WHAT THIS HARNESS CAN PROVE.
//
// `common::model("test")` carries an empty BPE tokenizer, so DESCRIBE returns
// zero edges for every entity. Comparing two empty arrays and calling it
// isolation would be a test that passes for the wrong reason — the same vacuity
// that once let a fixture report 8/8 walk-vs-dense agreement while encoding
// every prompt to the same token run.
//
// So isolation is tested at two honest levels instead:
//   - here, that the cache SCOPE differs, which is the mechanism that prevents
//     the collision;
//   - against testdata/tiny-vindex, where the values actually discriminate.
//     Measured 2026-09-01 on that fixture, two patch sets over one instance,
//     neither applied and no session:
//       tenant A (delete L1·f150) -> ['[69]', '[10]', '[478]', '[99]']
//       tenant B (delete L7·f0)   -> ['[124]', '[69]', '[10]', '[478]']
//       tenant A again            -> ['[69]', '[10]', '[478]', '[99]']
//     A keeps its own answer after B's request, which is the collision failing
//     to happen.
#[tokio::test]
async fn two_patch_sets_get_distinct_cache_scopes() {
    use larql_server::overlay_cache::PatchSetRef;

    // The DESCRIBE result cache was keyed by session id. A request carrying a
    // patch set has no session, so every such caller shared one empty slot and
    // two workspaces with different edits would have been served each other's
    // suppressions — a cross-tenant leak dressed up as a cache hit.
    let a: PatchSetRef =
        serde_json::from_value(serde_json::json!({ "patches": [delete_op(1, 150)] })).unwrap();
    let b: PatchSetRef =
        serde_json::from_value(serde_json::json!({ "patches": [delete_op(7, 0)] })).unwrap();

    assert_ne!(a.key("test", None), b.key("test", None));

    // And both are served, so the scoping is exercised rather than merely
    // derivable.
    let app = single_model_router(state(vec![model("test")]));
    for ps in [
        serde_json::json!({ "patches": [delete_op(1, 150)] }),
        serde_json::json!({ "patches": [delete_op(7, 0)] }),
    ] {
        let r = post_json(app.clone(), "/v1/describe", describe_body(Some(ps))).await;
        assert_eq!(r.status(), StatusCode::OK);
    }
}

// The security boundary, exercised over HTTP rather than only at the key.
//
// Two callers send the SAME sha with DIFFERENT patches — which a caller is free
// to do, because the sha is taken as the key and the patches as the value with
// nothing tying them together. Scoped by caller, each gets its own overlay.
// Unscoped, the second would have overwritten the first for everyone.
#[tokio::test]
async fn one_caller_cannot_file_content_under_another_callers_hash() {
    let app = single_model_router(state(vec![model("test")]));
    let sha = "collide";

    let a = post_json_h(
        app.clone(),
        "/v1/describe",
        describe_body(Some(
            serde_json::json!({ "sha": sha, "patches": [delete_op(1, 150)] }),
        )),
        ("x-session-id", "wl:tenant-a"),
    )
    .await;
    assert_eq!(a.status(), StatusCode::OK);

    let b = post_json_h(
        app.clone(),
        "/v1/describe",
        describe_body(Some(
            serde_json::json!({ "sha": sha, "patches": [delete_op(7, 0)] }),
        )),
        ("x-session-id", "wl:tenant-b"),
    )
    .await;
    assert_eq!(b.status(), StatusCode::OK);

    // A asks by hash alone. It must resolve — A compiled that hash itself — and
    // it must be A's entry, not the one B filed under the same hash.
    let a2 = post_json_h(
        app.clone(),
        "/v1/describe",
        describe_body(Some(serde_json::json!({ "sha": sha }))),
        ("x-session-id", "wl:tenant-a"),
    )
    .await;
    assert_eq!(a2.status(), StatusCode::OK);

    // And a caller who never compiled that hash gets 409 rather than whatever
    // someone else happens to have left under it.
    let c = post_json_h(
        app,
        "/v1/describe",
        describe_body(Some(serde_json::json!({ "sha": sha }))),
        ("x-session-id", "wl:tenant-c"),
    )
    .await;
    assert_eq!(
        c.status(),
        StatusCode::CONFLICT,
        "tenant-c reached an overlay it never compiled"
    );
}

#[tokio::test]
async fn a_patch_set_request_does_not_touch_global_state() {
    let app = single_model_router(state(vec![model("test")]));

    let plain = get(app.clone(), "/v1/describe?entity=France&band=all&limit=10&min_score=0").await;
    let before = body_json(plain.into_body()).await;

    let resp = post_json(
        app.clone(),
        "/v1/describe",
        describe_body(Some(serde_json::json!({ "patches": [delete_op(1, 150)] }))),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);

    let plain2 = get(app.clone(), "/v1/describe?entity=France&band=all&limit=10&min_score=0").await;
    let after = body_json(plain2.into_body()).await;
    assert_eq!(before["edges"], after["edges"]);

    // The load-bearing half on this fixture: nothing was written anywhere. The
    // edge comparison above is weak here (an empty tokenizer yields no edges);
    // this is not.
    let listed = get(app, "/v1/patches").await;
    let body = body_json(listed.into_body()).await;
    assert!(body["patches"].as_array().unwrap().is_empty());
}

// ══════════════════════════════════════════════════════════════
// Hash-only negotiation
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn an_unknown_hash_is_a_409_naming_what_to_retry_with() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(
        app,
        "/v1/describe",
        describe_body(Some(serde_json::json!({ "sha": "never-seen" }))),
    )
    .await;

    // 409, not 404 and not a silent fall-through to unpatched state. A server
    // that answered an unknown hash from the base would report a tenant's model
    // as unedited — the silent-wrong-answer this whole design exists to remove.
    assert_eq!(resp.status(), StatusCode::CONFLICT);
    let body = body_json(resp.into_body()).await;
    assert!(
        body["error"].as_str().unwrap_or("").contains("patch_set_unknown"),
        "the error must say what to do: {body}"
    );
}

#[tokio::test]
async fn a_hash_resolves_once_its_patches_have_been_seen() {
    let app = single_model_router(state(vec![model("test")]));
    let sha = "wl-tenant-a-v3";

    let inline = post_json(
        app.clone(),
        "/v1/describe",
        describe_body(Some(
            serde_json::json!({ "sha": sha, "patches": [delete_op(1, 150)] }),
        )),
    )
    .await;
    assert_eq!(inline.status(), StatusCode::OK);
    let with_patches = body_json(inline.into_body()).await;

    // Now the hash alone is enough — that is the point of the negotiation: a
    // workspace with hundreds of edits stops putting them on the wire for
    // every read.
    let hash_only = post_json(
        app,
        "/v1/describe",
        describe_body(Some(serde_json::json!({ "sha": sha }))),
    )
    .await;
    assert_eq!(hash_only.status(), StatusCode::OK);
    let cached = body_json(hash_only.into_body()).await;

    assert_eq!(with_patches["edges"], cached["edges"]);
}

// ══════════════════════════════════════════════════════════════
// Compatibility
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn the_get_form_is_unchanged() {
    // The GET cannot carry a patch set — a tenant's edits in a query string
    // would be written into every access log and proxy on the path. It stays
    // exactly as it was, which is what lets a client migrate one call site at
    // a time rather than in a flag day.
    let app = single_model_router(state(vec![model("test")]));
    let resp = get(app, "/v1/describe?entity=France&band=all&limit=10&min_score=0").await;
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn a_post_without_a_patch_set_behaves_like_the_get() {
    let app = single_model_router(state(vec![model("test")]));

    let g = get(app.clone(), "/v1/describe?entity=France&band=all&limit=10&min_score=0").await;
    let gb = body_json(g.into_body()).await;
    let p = post_json(app, "/v1/describe", describe_body(None)).await;
    assert_eq!(p.status(), StatusCode::OK);
    let pb = body_json(p.into_body()).await;

    assert_eq!(gb["entity"], pb["entity"]);
    assert_eq!(gb["edges"], pb["edges"]);
    assert!(pb["latency_ms"].as_f64().is_some(), "POST must answer in the GET's shape");
}
