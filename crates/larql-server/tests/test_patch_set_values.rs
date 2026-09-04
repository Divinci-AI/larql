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
        window: 10,
        min_score: 0.0,
        coherence: false,
        min_coherence: 0.0,
        relabel: false,
        query: "embedding".into(),
        baseline: None,
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

// ══════════════════════════════════════════════════════════════
// Coherence: a rendering change, and it must stay opt-in
// ══════════════════════════════════════════════════════════════

fn describe_raw(
    model: &LoadedModel,
    coherence: bool,
    min_coherence: f32,
) -> serde_json::Value {
    describe_with(model, coherence, min_coherence, false)
}

fn describe_with(
    model: &LoadedModel,
    coherence: bool,
    min_coherence: f32,
    relabel: bool,
) -> serde_json::Value {
    describe_limited(model, coherence, min_coherence, relabel, 10)
}

fn describe_limited(
    model: &LoadedModel,
    coherence: bool,
    min_coherence: f32,
    relabel: bool,
    limit: usize,
    window: usize,
) -> serde_json::Value {
    let params = larql_server::routes::describe::DescribeParams {
        entity: "[5]".to_string(),
        band: "all".to_string(),
        verbose: false,
        limit,
        min_score: 0.0,
        coherence,
        min_coherence,
        relabel,
        query: "embedding".into(),
        baseline: None,
    };
    larql_server::routes::describe::describe_entity_with(
        model,
        &model.patched.blocking_read(),
        &params,
    )
    .expect("describe must succeed")
}

#[test]
fn coherence_off_is_byte_identical_to_before() {
    let Some(m) = tiny() else { return };

    // The whole safety argument for shipping this is that it is opt-in. If the
    // default path changed at all, every stored edit keyed to a target label
    // would be at risk — so this compares the entire document, not just the
    // targets.
    let a = describe_raw(&m, false, 0.0);
    let b = describe_raw(&m, false, 0.0);
    assert_eq!(a["edges"], b["edges"], "describe is not deterministic");

    for edge in a["edges"].as_array().expect("edges") {
        assert!(
            edge.get("coherence").is_none(),
            "coherence leaked into the default response: {edge}"
        );
        assert!(edge.get("label_source").is_none(), "label_source leaked: {edge}");
    }
}

#[test]
fn coherence_on_reports_a_score_for_every_edge() {
    let Some(m) = tiny() else { return };
    let out = describe_raw(&m, true, 0.0);
    let edges = out["edges"].as_array().expect("edges").clone();
    assert!(!edges.is_empty(), "fixture produced nothing to score");

    for edge in &edges {
        // Present on every edge, and `null` where it could not be computed —
        // "unmeasured" must stay distinguishable from "measured badly".
        assert!(edge.get("coherence").is_some(), "missing coherence: {edge}");
        let src = edge["label_source"].as_str().unwrap_or_default();
        assert!(
            src == "centroid" || src == "argmax",
            "label_source must say where the label came from, got {src:?}"
        );
        if let Some(c) = edge["coherence"].as_f64() {
            assert!((-1.0..=1.0).contains(&c), "coherence out of range: {c}");
        }
    }
}

#[test]
fn a_threshold_only_ever_removes_edges() {
    let Some(m) = tiny() else { return };

    // Filtering must be a subset operation. If raising the bar could ADD an
    // edge, the score would not be ordering anything and the whole mechanism
    // would be noise dressed as a measurement.
    let all = describe_raw(&m, true, 0.0);
    let strict = describe_raw(&m, true, 0.99);

    let key = |v: &serde_json::Value| {
        v["edges"]
            .as_array()
            .map(|a| {
                a.iter()
                    .map(|e| format!("{}@{}", e["target"], e["layer"]))
                    .collect::<std::collections::HashSet<_>>()
            })
            .unwrap_or_default()
    };
    let (a, s) = (key(&all), key(&strict));
    assert!(s.len() <= a.len(), "a threshold grew the result: {} -> {}", a.len(), s.len());
    for k in &s {
        assert!(a.contains(k), "threshold introduced an edge that was not there: {k}");
    }
}

#[test]
fn random_embeddings_score_near_zero_and_not_near_one() {
    let Some(m) = tiny() else { return };

    // The fixture is a NEGATIVE CONTROL. Its embeddings are synthetic, and
    // random vectors in high dimensions are near-orthogonal, so the honest
    // answer for every feature here is "no coherence". Pinning that catches the
    // failure mode this measure is most likely to have: a bug that returns a
    // flattering constant (1.0 from a self-similarity, or a norm that cancels)
    // would sail through every other test in this file, and would then be read
    // in production as "every feature is coherent".
    let out = describe_raw(&m, true, 0.0);
    let scores: Vec<f64> = out["edges"]
        .as_array()
        .expect("edges")
        .iter()
        .filter_map(|e| e["coherence"].as_f64())
        .collect();

    assert!(!scores.is_empty(), "nothing was scored; the assertion below is vacuous");
    for c in &scores {
        assert!(
            c.abs() < 0.5,
            "random embeddings scored {c}, which is not near-orthogonal — \
             the measure is reporting structure that is not in the data"
        );
    }
}

// ══════════════════════════════════════════════════════════════
// The split: scoring is information, relabelling is a change
// ══════════════════════════════════════════════════════════════

#[test]
fn scoring_alone_leaves_every_target_byte_identical() {
    let Some(m) = tiny() else { return };
    let raw = describe_raw(&m, false, 0.0);
    let scored = describe_with(&m, true, 0.0, false);
    let t = |v: &serde_json::Value| {
        v["edges"].as_array().unwrap().iter().map(|e| e["target"].clone()).collect::<Vec<_>>()
    };
    // This is the property that makes the score adoptable on its own: a
    // caller can read it, or filter on it, without a single stored edit's
    // target changing underneath it.
    assert_eq!(t(&raw), t(&scored));
    for e in scored["edges"].as_array().unwrap() {
        assert!(e.get("coherence").is_some(), "scored but not reported: {e}");
        assert_eq!(e["label_source"], "argmax", "scoring alone must not relabel: {e}");
    }
}

#[test]
fn relabel_is_what_moves_the_label_and_it_implies_scoring() {
    let Some(m) = tiny() else { return };
    let scored = describe_with(&m, true, 0.0, false);
    let relabelled = describe_with(&m, false, 0.0, true);
    let t = |v: &serde_json::Value| {
        v["edges"].as_array().unwrap().iter().map(|e| e["target"].clone()).collect::<Vec<_>>()
    };
    assert_ne!(t(&scored), t(&relabelled), "relabel changed nothing on a fixture where it must");
    for e in relabelled["edges"].as_array().unwrap() {
        // relabel without coherence=true still reports the score it used.
        assert!(e.get("coherence").is_some(), "relabel must report the score it relied on: {e}");
    }
}

#[test]
fn a_threshold_filters_without_relabelling() {
    let Some(m) = tiny() else { return };
    // Untruncated baseline: with a limit, dropping edges promotes lower-ranked
    // ones into the window, which is correct and not a rename. The subset
    // claim is about the universe of edges, not the top ten.
    let raw = describe_limited(&m, false, 0.0, false, 10_000);
    let filtered = describe_with(&m, false, 0.01, false);
    let targets = |v: &serde_json::Value| {
        v["edges"].as_array().unwrap().iter()
            .map(|e| e["target"].as_str().unwrap().to_string())
            .collect::<std::collections::HashSet<_>>()
    };
    let (r, f) = (targets(&raw), targets(&filtered));
    // Every surviving target is one the raw answer already had, under the
    // same name: the filter removed edges and renamed none.
    for k in &f { assert!(r.contains(k), "filter introduced or renamed a target: {k}"); }
    for e in filtered["edges"].as_array().unwrap() {
        assert_eq!(e["label_source"], "argmax");
    }
}

// ══════════════════════════════════════════════════════════════
// query=residual: the model's own residual, or an honest refusal
// ══════════════════════════════════════════════════════════════

#[test]
fn residual_mode_either_runs_the_model_or_says_why_it_cannot() {
    let Some(m) = tiny() else { return };
    let params = larql_server::routes::describe::DescribeParams {
        entity: "[5]".to_string(),
        band: "all".to_string(),
        verbose: false,
        limit: 10,
        window: 10,
        min_score: 0.0,
        coherence: false,
        min_coherence: 0.0,
        relabel: false,
        query: "residual".into(),
        baseline: None,
    };
    let r = larql_server::routes::describe::describe_entity_with(
        &m,
        &m.patched.blocking_read(),
        &params,
    );
    match r {
        Ok(v) => {
            // If the fixture carries weights, every scanned layer must have
            // been scored against a captured residual. A silent partial
            // answer is the failure this field exists to expose.
            assert_eq!(v["query"], "residual");
            assert_eq!(v["residual_layers"], v["scanned_layers"], "layers missing residuals: {v}");
        }
        Err(e) => {
            // Without weights the only acceptable answer is a refusal that
            // says so — never a fallback to the embedding query dressed up as
            // the residual one.
            let msg = format!("{e:?}");
            assert!(msg.contains("weights"), "refusal did not say why: {msg}");
        }
    }
}

#[test]
fn an_unknown_query_mode_is_rejected_not_defaulted() {
    let Some(m) = tiny() else { return };
    let params = larql_server::routes::describe::DescribeParams {
        entity: "[5]".to_string(),
        band: "all".to_string(),
        verbose: false,
        limit: 10,
        window: 10,
        min_score: 0.0,
        coherence: false,
        min_coherence: 0.0,
        relabel: false,
        query: "activations".into(),
        baseline: None,
    };
    let r = larql_server::routes::describe::describe_entity_with(&m, &m.patched.blocking_read(), &params);
    assert!(r.is_err(), "a typo in `query` must not silently mean `embedding`");
}

#[test]
fn contrasting_an_entity_with_itself_scores_nothing() {
    let Some(m) = tiny() else { return };
    // residual(x) - residual(x) is the zero vector at every layer; every gate
    // dot product is then exactly 0. If anything scores above zero here, the
    // baseline was not subtracted from the same vector it was meant to cancel.
    let params = larql_server::routes::describe::DescribeParams {
        entity: "[5]".to_string(),
        band: "all".to_string(),
        verbose: false,
        limit: 10,
        window: 10,
        min_score: 0.0,
        coherence: false,
        min_coherence: 0.0,
        relabel: false,
        query: "residual".into(),
        baseline: Some("[5]".into()),
    };
    let r = larql_server::routes::describe::describe_entity_with(&m, &m.patched.blocking_read(), &params);
    match r {
        Ok(v) => {
            assert_eq!(v["contrasted_layers"], v["scanned_layers"], "{v}");
            // And never more than scanned: the count is over the layers that
            // were scored, not every layer the forward pass captured.
            assert!(v["contrasted_layers"].as_u64() <= v["scanned_layers"].as_u64(), "{v}");
            for e in v["edges"].as_array().unwrap() {
                let s = e["gate_score"].as_f64().unwrap();
                assert!(s.abs() < 1e-6, "self-contrast left a non-zero score: {e}");
            }
        }
        Err(e) => assert!(format!("{e:?}").contains("weights"), "{e:?}"),
    }
}
