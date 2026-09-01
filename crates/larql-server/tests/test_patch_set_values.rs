//! The only test in this crate that can tell a right answer from a wrong one.
//!
//! # Why it exists
//!
//! Every other HTTP test here runs against `common::model("test")`, whose
//! tokenizer is empty. DESCRIBE therefore returns zero edges for every entity,
//! so those tests can assert status codes and response shapes and nothing else.
//! That is fine for routing, and useless for behaviour: if the `patch_set` →
//! overlay wiring were broken so that patches were silently ignored, every one
//! of them would still pass. Silently ignoring a patch is precisely the failure
//! this whole design exists to prevent.
//!
//! `testdata/tiny-vindex` is a real artifact with a real tokenizer, small enough
//! to load in a unit test. It discriminates: different prompts produce different
//! distributions, and deleting a feature changes what DESCRIBE returns.
//!
//! Before this file, the value-level claims about content-addressed overlays
//! were verified by hand and written into comments. Comments do not fail. This
//! turns that into coverage.
//!
//! # If this file starts failing
//!
//! Check the fixture before the code. It was rebuilt on 2026-08-30 after its
//! tokenizer was found to encode EVERY prompt to the same run of zeros, which
//! made an earlier walk-vs-dense comparison report 8/8 agreement while
//! comparing nothing at all.

use std::path::Path;
use std::sync::Arc;

use larql_server::bootstrap::load::{load_single_vindex, LoadVindexOptions};
use larql_server::overlay_cache::{resolve_overlay, PatchSetRef};
use larql_server::state::LoadedModel;

/// The fixture, or `None` when it is not present.
///
/// Skips rather than fails if the artifact is missing: a checkout without
/// testdata should not report a red suite it cannot fix. The skip is loud.
fn tiny() -> Option<Arc<LoadedModel>> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-vindex");
    if !root.join("index.json").exists() {
        eprintln!("SKIP: {} not present", root.display());
        return None;
    }
    let opts = LoadVindexOptions {
        no_infer: true,
        ..Default::default()
    };
    match load_single_vindex(root.to_str().unwrap(), opts) {
        Ok(m) => Some(Arc::new(m)),
        Err(e) => panic!("tiny-vindex present but failed to load: {e}"),
    }
}

fn delete_patch(layer: usize, feature: usize) -> larql_vindex::VindexPatch {
    larql_vindex::VindexPatch {
        version: 1,
        base_model: "test".into(),
        base_checksum: None,
        created_at: "2026-09-01T00:00:00Z".into(),
        description: Some(format!("del-L{layer}-f{feature}")),
        author: None,
        tags: vec![],
        operations: vec![larql_vindex::PatchOp::Delete {
            layer,
            feature,
            reason: None,
        }],
    }
}

fn patch_set(patches: Vec<larql_vindex::VindexPatch>) -> PatchSetRef {
    PatchSetRef {
        sha: None,
        patches: Some(patches),
    }
}

/// Targets DESCRIBE reports for `[5]`, strongest first.
///
/// Goes through `resolve_overlay` — the same function the HTTP handlers use —
/// rather than reaching into the cache, so what is exercised here is the path
/// production takes.
fn targets(model: &LoadedModel, ps: Option<&PatchSetRef>, scope: Option<&str>) -> Vec<String> {
    let params = larql_server::routes::describe::DescribeParams {
        entity: "[5]".to_string(),
        band: "all".to_string(),
        verbose: false,
        limit: 10,
        min_score: 0.0,
    };
    let extract = |patched: &larql_vindex::PatchedVindex| {
        let v = larql_server::routes::describe::describe_entity_with(model, patched, &params)
            .expect("describe must succeed");
        v["edges"]
            .as_array()
            .expect("edges array")
            .iter()
            .filter_map(|e| e["target"].as_str().map(str::to_string))
            .collect::<Vec<String>>()
    };
    match ps {
        Some(req) => {
            let overlay = resolve_overlay(model, req, scope).expect("overlay must resolve");
            extract(&overlay)
        }
        None => extract(&model.patched.blocking_read()),
    }
}

// ══════════════════════════════════════════════════════════════
// The fixture must be able to fail
// ══════════════════════════════════════════════════════════════

#[test]
fn the_fixture_discriminates_at_all() {
    let Some(m) = tiny() else { return };
    let base = targets(&m, None, None);

    // If this is empty, every assertion below is vacuous and passing means
    // nothing — the exact trap that made a previous fixture report agreement it
    // had not measured.
    assert!(
        !base.is_empty(),
        "fixture produced no edges; every value assertion here would be vacuous"
    );
    assert!(
        base.contains(&"[124]".to_string()),
        "expected [124] among {base:?}"
    );
}

// ══════════════════════════════════════════════════════════════
// The property the whole redesign rests on
// ══════════════════════════════════════════════════════════════

#[test]
fn a_carried_patch_set_actually_suppresses() {
    let Some(m) = tiny() else { return };

    let before = targets(&m, None, None);
    let after = targets(&m, Some(&patch_set(vec![delete_patch(1, 150)])), None);

    // L1·f150 is the feature behind the [124] edge. Nothing was applied, no
    // session exists, and this instance was told nothing beforehand — the
    // request carried what it needed.
    assert!(before.contains(&"[124]".to_string()));
    assert!(
        !after.contains(&"[124]".to_string()),
        "the carried patch set did not suppress: {after:?}"
    );
}

#[test]
fn carrying_a_patch_set_leaves_the_base_untouched() {
    let Some(m) = tiny() else { return };

    let before = targets(&m, None, None);
    let _ = targets(&m, Some(&patch_set(vec![delete_patch(1, 150)])), None);
    let after = targets(&m, None, None);

    // A read that mutated shared state would show up here as the unpatched view
    // having changed. This is the difference between an overlay that is a value
    // and one that is instance state.
    assert_eq!(before, after);
}

#[test]
fn an_empty_patch_set_reads_like_the_base() {
    let Some(m) = tiny() else { return };
    assert_eq!(targets(&m, None, None), targets(&m, Some(&patch_set(vec![])), None));
}

// ══════════════════════════════════════════════════════════════
// Isolation, by value rather than by cache key
// ══════════════════════════════════════════════════════════════

#[test]
fn two_callers_get_their_own_overlays() {
    let Some(m) = tiny() else { return };

    let a = patch_set(vec![delete_patch(1, 150)]); // suppresses [124]
    let b = patch_set(vec![delete_patch(7, 0)]); // unrelated

    let ta1 = targets(&m, Some(&a), Some("wl:tenant-a"));
    let tb = targets(&m, Some(&b), Some("wl:tenant-b"));
    let ta2 = targets(&m, Some(&a), Some("wl:tenant-a"));

    assert!(!ta1.contains(&"[124]".to_string()), "A: {ta1:?}");
    assert!(tb.contains(&"[124]".to_string()), "B must be unaffected: {tb:?}");
    // A after B: a shared cache entry would show up as A inheriting B's answer.
    assert_eq!(ta1, ta2, "A's second read disagreed with its first");
}

#[test]
fn the_same_patch_set_under_two_callers_gives_the_same_answer() {
    let Some(m) = tiny() else { return };
    let ps = patch_set(vec![delete_patch(1, 150)]);

    // Scoping the cache by caller must not change what an overlay MEANS. Same
    // content, same answer — the scope is a security boundary, not a semantic
    // one.
    assert_eq!(
        targets(&m, Some(&ps), Some("wl:tenant-a")),
        targets(&m, Some(&ps), Some("wl:tenant-b"))
    );
}

// ══════════════════════════════════════════════════════════════
// The cache is an optimisation, never an answer
// ══════════════════════════════════════════════════════════════

#[test]
fn a_cache_hit_answers_identically_to_a_cold_build() {
    let Some(m) = tiny() else { return };
    let ps = patch_set(vec![delete_patch(1, 150)]);

    let cold = targets(&m, Some(&ps), Some("wl:t"));
    let warm = targets(&m, Some(&ps), Some("wl:t"));
    assert_eq!(cold, warm);

    let stats = m.overlay_cache.stats();
    assert!(stats.hits >= 1, "second read should have hit: {stats:?}");
    assert!(stats.builds >= 1, "first read should have built: {stats:?}");
}

#[test]
fn eviction_costs_latency_and_not_correctness() {
    let Some(m) = tiny() else { return };
    let ps = patch_set(vec![delete_patch(1, 150)]);

    let first = targets(&m, Some(&ps), Some("wl:t"));

    // Push more distinct overlays through than the cache holds, so the entry
    // above is certainly evicted, then ask again. A rebuilt overlay must be
    // indistinguishable from the one that was dropped — that asymmetry (a miss
    // costs time, never truth) is what lets the cache be bounded at all.
    for f in 0..(m.overlay_cache.stats().capacity + 4) {
        let _ = targets(&m, Some(&patch_set(vec![delete_patch(2, f)])), Some("wl:t"));
    }

    assert_eq!(first, targets(&m, Some(&ps), Some("wl:t")));
}
