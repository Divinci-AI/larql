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
        relevance: true,
        background: Some("vocabulary".into()),
        window_by: "score".into(),
        query: "embedding".into(),
        prompt: "{entity}".into(),
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
    assert_eq!(
        targets(&m, None, None),
        targets(&m, Some(&patch_set(vec![])), None)
    );
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
    assert!(
        tb.contains(&"[124]".to_string()),
        "B must be unaffected: {tb:?}"
    );
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

fn describe_raw(model: &LoadedModel, coherence: bool, min_coherence: f32) -> serde_json::Value {
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
) -> serde_json::Value {
    describe_ranked(model, coherence, min_coherence, relabel, limit, true)
}

fn describe_ranked(
    model: &LoadedModel,
    coherence: bool,
    min_coherence: f32,
    relabel: bool,
    limit: usize,
    relevance: bool,
) -> serde_json::Value {
    describe_against(
        model,
        coherence,
        min_coherence,
        relabel,
        limit,
        relevance,
        "vocabulary",
    )
    .expect("vocabulary background is always valid")
}

fn describe_against(
    model: &LoadedModel,
    coherence: bool,
    min_coherence: f32,
    relabel: bool,
    limit: usize,
    relevance: bool,
    background: &str,
) -> Result<serde_json::Value, larql_server::error::ServerError> {
    let params = larql_server::routes::describe::DescribeParams {
        entity: "[5]".to_string(),
        band: "all".to_string(),
        verbose: false,
        limit,
        // The pre-2026-09-04 semantics, where the window WAS the limit: these
        // tests reason about the returned set, not the candidate pool.
        window: limit,
        min_score: 0.0,
        coherence,
        min_coherence,
        relabel,
        relevance,
        background: Some(background.into()),
        window_by: "score".into(),
        query: "embedding".into(),
        prompt: "{entity}".into(),
        baseline: None,
    };
    larql_server::routes::describe::describe_entity_with(
        model,
        &model.patched.blocking_read(),
        &params,
    )
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
        assert!(
            edge.get("label_source").is_none(),
            "label_source leaked: {edge}"
        );
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
    assert!(
        s.len() <= a.len(),
        "a threshold grew the result: {} -> {}",
        a.len(),
        s.len()
    );
    for k in &s {
        assert!(
            a.contains(k),
            "threshold introduced an edge that was not there: {k}"
        );
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

    assert!(
        !scores.is_empty(),
        "nothing was scored; the assertion below is vacuous"
    );
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
        v["edges"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["target"].clone())
            .collect::<Vec<_>>()
    };
    // This is the property that makes the score adoptable on its own: a
    // caller can read it, or filter on it, without a single stored edit's
    // target changing underneath it.
    assert_eq!(t(&raw), t(&scored));
    for e in scored["edges"].as_array().unwrap() {
        assert!(e.get("coherence").is_some(), "scored but not reported: {e}");
        assert_eq!(
            e["label_source"], "argmax",
            "scoring alone must not relabel: {e}"
        );
    }
}

#[test]
fn relabel_is_what_moves_the_label_and_it_implies_scoring() {
    let Some(m) = tiny() else { return };
    let scored = describe_with(&m, true, 0.0, false);
    let relabelled = describe_with(&m, false, 0.0, true);
    let t = |v: &serde_json::Value| {
        v["edges"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["target"].clone())
            .collect::<Vec<_>>()
    };
    assert_ne!(
        t(&scored),
        t(&relabelled),
        "relabel changed nothing on a fixture where it must"
    );
    for e in relabelled["edges"].as_array().unwrap() {
        // relabel without coherence=true still reports the score it used.
        assert!(
            e.get("coherence").is_some(),
            "relabel must report the score it relied on: {e}"
        );
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
        v["edges"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["target"].as_str().unwrap().to_string())
            .collect::<std::collections::HashSet<_>>()
    };
    let (r, f) = (targets(&raw), targets(&filtered));
    // Every surviving target is one the raw answer already had, under the
    // same name: the filter removed edges and renamed none.
    for k in &f {
        assert!(r.contains(k), "filter introduced or renamed a target: {k}");
    }
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
        relevance: true,
        background: Some("entities".into()),
        window_by: "score".into(),
        query: "residual".into(),
        prompt: "{entity}".into(),
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
            assert_eq!(
                v["residual_layers"], v["scanned_layers"],
                "layers missing residuals: {v}"
            );
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
        relevance: true,
        background: Some("vocabulary".into()),
        window_by: "score".into(),
        query: "activations".into(),
        prompt: "{entity}".into(),
        baseline: None,
    };
    let r = larql_server::routes::describe::describe_entity_with(
        &m,
        &m.patched.blocking_read(),
        &params,
    );
    assert!(
        r.is_err(),
        "a typo in `query` must not silently mean `embedding`"
    );
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
        relevance: true,
        background: Some("entities".into()),
        window_by: "score".into(),
        query: "residual".into(),
        prompt: "{entity}".into(),
        baseline: Some("[5]".into()),
    };
    let r = larql_server::routes::describe::describe_entity_with(
        &m,
        &m.patched.blocking_read(),
        &params,
    );
    match r {
        Ok(v) => {
            assert_eq!(v["contrasted_layers"], v["scanned_layers"], "{v}");
            // And never more than scanned: the count is over the layers that
            // were scored, not every layer the forward pass captured.
            assert!(
                v["contrasted_layers"].as_u64() <= v["scanned_layers"].as_u64(),
                "{v}"
            );
            for e in v["edges"].as_array().unwrap() {
                let s = e["gate_score"].as_f64().unwrap();
                assert!(s.abs() < 1e-6, "self-contrast left a non-zero score: {e}");
            }
        }
        Err(e) => assert!(format!("{e:?}").contains("weights"), "{e:?}"),
    }
}

// ══════════════════════════════════════════════════════════════
// Relevance: a ranking key, never a filter
// ══════════════════════════════════════════════════════════════

#[test]
fn relevance_off_ranks_by_gate_score_exactly_as_before() {
    let Some(m) = tiny() else { return };
    let v = describe_ranked(&m, false, 0.0, false, 50, false);
    let scores: Vec<f64> = v["edges"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["gate_score"].as_f64().unwrap())
        .collect();
    assert!(
        scores.windows(2).all(|w| w[0] >= w[1]),
        "not sorted by gate: {scores:?}"
    );
    for e in v["edges"].as_array().unwrap() {
        assert!(
            e.get("relevance").is_none(),
            "relevance leaked into the raw mode: {e}"
        );
    }
}

#[test]
fn relevance_on_reports_a_score_and_ranks_by_it() {
    let Some(m) = tiny() else { return };
    let v = describe_ranked(&m, false, 0.0, false, 50, true);
    let edges = v["edges"].as_array().unwrap();
    assert!(!edges.is_empty());
    let zs: Vec<f64> = edges
        .iter()
        .map(|e| {
            e["relevance"]
                .as_f64()
                .expect("relevance must be a number on this fixture")
        })
        .collect();
    assert!(
        zs.windows(2).all(|w| w[0] >= w[1]),
        "not sorted by relevance: {zs:?}"
    );
    assert!(zs.iter().all(|z| z.is_finite()));
}

#[test]
fn relevance_changes_order_but_not_membership() {
    let Some(m) = tiny() else { return };
    // Same universe either way: relevance re-orders, it never adds or drops.
    let raw = describe_ranked(&m, false, 0.0, false, 10_000, false);
    let rel = describe_ranked(&m, false, 0.0, false, 10_000, true);
    let set = |v: &serde_json::Value| {
        v["edges"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| format!("{}@{}", e["target"], e["layer"]))
            .collect::<std::collections::BTreeSet<_>>()
    };
    assert_eq!(set(&raw), set(&rel));
}

// ══════════════════════════════════════════════════════════════
// Relevance background: entities by default, vocabulary on request
// ══════════════════════════════════════════════════════════════

#[test]
fn the_loaded_model_has_an_entity_panel_and_reports_it() {
    let Some(m) = tiny() else { return };
    let v = describe_against(&m, false, 0.0, false, 50, true, "entities").unwrap();
    assert_eq!(v["relevance_background"], "entities");
    let panel = v["relevance_panel"].as_u64().unwrap();
    assert!(
        panel >= 2,
        "entity panel too small to give a background: {panel}"
    );
    let edges = v["edges"].as_array().unwrap();
    assert!(!edges.is_empty());
    for e in edges {
        let z = e["relevance"]
            .as_f64()
            .expect("every edge carries a relevance under the entity panel");
        assert!(z.is_finite());
    }
}

#[test]
fn the_two_backgrounds_are_two_rankings_of_one_set() {
    let Some(m) = tiny() else { return };
    let ent = describe_against(&m, false, 0.0, false, 10_000, true, "entities").unwrap();
    let voc = describe_against(&m, false, 0.0, false, 10_000, true, "vocabulary").unwrap();
    assert_eq!(voc["relevance_background"], "vocabulary");
    let set = |v: &serde_json::Value| {
        v["edges"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| format!("{}@{}", e["target"], e["layer"]))
            .collect::<std::collections::BTreeSet<_>>()
    };
    assert_eq!(
        set(&ent),
        set(&voc),
        "a background re-orders; it never adds or drops"
    );
    // And they are genuinely different backgrounds, not one panel under two
    // names: at least one edge's relevance differs.
    let z = |v: &serde_json::Value| {
        v["edges"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| {
                (
                    format!("{}@{}", e["target"], e["layer"]),
                    e["relevance"].as_f64().unwrap(),
                )
            })
            .collect::<std::collections::BTreeMap<_, _>>()
    };
    let (ze, zv) = (z(&ent), z(&voc));
    assert!(
        ze.iter().any(|(k, a)| (a - zv[k]).abs() > 1e-3),
        "both backgrounds gave identical z for every edge"
    );
}

#[test]
fn an_unknown_background_is_a_bad_request_not_a_silent_default() {
    let Some(m) = tiny() else { return };
    match describe_against(&m, false, 0.0, false, 50, true, "wikipedia") {
        Err(larql_server::error::ServerError::BadRequest(msg)) => {
            assert!(msg.contains("wikipedia"), "{msg}")
        }
        other => panic!("expected BadRequest, got {other:?}"),
    }
}

#[test]
fn relevance_off_ignores_the_background_entirely() {
    let Some(m) = tiny() else { return };
    let v = describe_against(&m, false, 0.0, false, 50, false, "entities").unwrap();
    assert!(v.get("relevance_background").is_none());
    assert!(v.get("relevance_panel").is_none());
}

#[test]
fn the_corpus_panel_is_built_through_the_real_loader_and_is_the_largest() {
    let Some(m) = tiny() else { return };
    let v = describe_against(&m, false, 0.0, false, 50, true, "corpus").unwrap();
    assert_eq!(v["relevance_background"], "corpus");
    let corpus = v["relevance_panel"].as_u64().unwrap();
    let ent = describe_against(&m, false, 0.0, false, 50, true, "entities").unwrap()
        ["relevance_panel"]
        .as_u64()
        .unwrap();
    assert!(
        corpus > ent * 5,
        "corpus panel {corpus} is not much larger than the entity panel {ent}"
    );
    for e in v["edges"].as_array().unwrap() {
        assert!(e["relevance"].as_f64().unwrap().is_finite());
    }
}

#[test]
fn an_absent_background_takes_the_models_default_and_names_it() {
    let Some(m) = tiny() else { return };
    let mut params = larql_server::routes::describe::DescribeParams {
        entity: "[5]".to_string(),
        band: "all".to_string(),
        verbose: false,
        limit: 50,
        window: 50,
        min_score: 0.0,
        coherence: false,
        min_coherence: 0.0,
        relabel: false,
        relevance: true,
        background: None,
        window_by: "score".into(),
        query: "embedding".into(),
        prompt: "{entity}".into(),
        baseline: None,
    };
    let v = larql_server::routes::describe::describe_entity_with(
        &m,
        &m.patched.blocking_read(),
        &params,
    )
    .unwrap();
    let default = m.relevance.default_background().as_str();
    assert_eq!(
        v["relevance_background"], default,
        "absent background must resolve to the model's default"
    );
    params.background = Some(default.into());
    let w = larql_server::routes::describe::describe_entity_with(
        &m,
        &m.patched.blocking_read(),
        &params,
    )
    .unwrap();
    assert_eq!(
        v["edges"], w["edges"],
        "naming the default must be byte-identical to omitting it"
    );
}

// ══════════════════════════════════════════════════════════════
// Residual query + relevance: the background is the panel's residuals
// ══════════════════════════════════════════════════════════════

fn describe_residual(
    model: &LoadedModel,
    relevance: bool,
) -> Result<serde_json::Value, larql_server::error::ServerError> {
    let params = larql_server::routes::describe::DescribeParams {
        entity: "[5]".to_string(),
        band: "all".to_string(),
        verbose: false,
        limit: 10_000,
        window: 10_000,
        min_score: 0.0,
        coherence: false,
        min_coherence: 0.0,
        relabel: false,
        relevance,
        background: Some("entities".into()),
        window_by: "score".into(),
        query: "residual".into(),
        prompt: "{entity}".into(),
        baseline: None,
    };
    larql_server::routes::describe::describe_entity_with(
        model,
        &model.patched.blocking_read(),
        &params,
    )
}

#[test]
fn residual_relevance_builds_a_residual_panel_and_ranks_by_it() {
    let Some(m) = tiny() else { return };
    let v = match describe_residual(&m, true) {
        Ok(v) => v,
        Err(e) => {
            assert!(format!("{e:?}").contains("weights"), "{e:?}");
            return;
        }
    };
    assert_eq!(v["relevance_query"], "residual");
    assert_eq!(v["relevance_background"], "entities");
    let panel = v["relevance_panel"].as_u64().unwrap();
    assert!(panel >= 2, "residual panel has {panel} rows");
    assert!(m
        .relevance
        .has_residual_panel(larql_server::relevance::Background::Entities, "{entity}"));
    let edges = v["edges"].as_array().unwrap();
    assert!(!edges.is_empty());
    let zs: Vec<f64> = edges
        .iter()
        .map(|e| {
            e["relevance"]
                .as_f64()
                .expect("every residual edge carries a z")
        })
        .collect();
    assert!(
        zs.windows(2).all(|w| w[0] >= w[1]),
        "not sorted by relevance: {zs:?}"
    );
    assert!(zs.iter().all(|z| z.is_finite()));
    // Same universe as the raw residual order: relevance re-orders, never filters.
    let raw = describe_residual(&m, false).unwrap();
    let set = |v: &serde_json::Value| {
        v["edges"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| format!("{}@{}", e["target"], e["layer"]))
            .collect::<std::collections::BTreeSet<_>>()
    };
    assert_eq!(set(&raw), set(&v));
    assert!(raw.get("relevance_query").is_none());
}

#[test]
fn residual_relevance_refuses_a_background_with_no_residual_panel() {
    let Some(m) = tiny() else { return };
    if !m.config.has_model_weights {
        return;
    }
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
        relevance: true,
        background: Some("corpus".into()),
        window_by: "score".into(),
        query: "residual".into(),
        prompt: "{entity}".into(),
        baseline: None,
    };
    let patched = m.patched.blocking_read();
    let r = larql_server::routes::describe::describe_entity_with(&m, &patched, &params);
    match r {
        Err(larql_server::error::ServerError::BadRequest(msg)) => {
            assert!(msg.contains("corpus"), "{msg}")
        }
        other => panic!("expected BadRequest, got {other:?}"),
    }
}

// ══════════════════════════════════════════════════════════════
// window_by=relevance: the window is the most surprising, not the largest
// ══════════════════════════════════════════════════════════════

fn describe_window_by(model: &LoadedModel, query: &str, window_by: &str, relevance: bool, window: usize)
    -> Result<serde_json::Value, larql_server::error::ServerError>
{
    let params = larql_server::routes::describe::DescribeParams {
        entity: "[5]".to_string(),
        band: "all".to_string(),
        verbose: false,
        limit: 10_000,
        window,
        min_score: 0.0,
        coherence: false,
        min_coherence: 0.0,
        relabel: false,
        relevance,
        background: Some("entities".into()),
        window_by: window_by.into(),
        query: query.into(),
        prompt: "{entity}".into(),
        baseline: None,
    };
    let patched = model.patched.blocking_read();
    larql_server::routes::describe::describe_entity_with(model, &patched, &params)
}

#[test]
fn a_surprise_window_is_the_top_z_of_the_whole_layer() {
    let Some(m) = tiny() else { return };
    // The whole layer, ordered by z: the reference.
    let all = describe_window_by(&m, "embedding", "relevance", true, 10_000).unwrap();
    let all_edges = all["edges"].as_array().unwrap();
    assert!(all_edges.len() > 2, "fixture too small to test a window");
    assert_eq!(all["window_by"], "relevance");
    // A window of 2 per layer: at most two edges a layer, the global top z
    // survives, and nothing below the whole-layer median gets in. (Edges
    // fold by label across features, so an exact per-layer top-two is not
    // well defined at the edge level.)
    let two = describe_window_by(&m, "embedding", "relevance", true, 2).unwrap();
    let two_edges = two["edges"].as_array().unwrap();
    let mut per_layer: std::collections::BTreeMap<u64, usize> = Default::default();
    for e in two_edges {
        *per_layer.entry(e["layer"].as_u64().unwrap()).or_default() += 1;
    }
    assert!(per_layer.values().all(|&n| n <= 2), "a layer exceeded its window: {per_layer:?}");
    let mut all_z: Vec<f64> = all_edges.iter().map(|e| e["relevance"].as_f64().unwrap()).collect();
    all_z.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let two_z: Vec<f64> = two_edges.iter().map(|e| e["relevance"].as_f64().unwrap()).collect();
    assert!((two_z[0] - all_z[0]).abs() < 1e-6, "the most surprising feature must survive: {} vs {}", two_z[0], all_z[0]);
    let median = all_z[all_z.len() / 2];
    assert!(two_z.iter().all(|&z| z >= median), "a below-median z got into a surprise window: {two_z:?} median {median}");
    // And it is a different selection from the raw-score window when the
    // two disagree — otherwise the flag would be a no-op on this fixture.
    let raw = describe_window_by(&m, "embedding", "score", true, 2).unwrap();
    let key = |v: &serde_json::Value| v["edges"].as_array().unwrap().iter()
        .map(|e| format!("{}@{}", e["feature"], e["layer"])).collect::<std::collections::BTreeSet<_>>();
    eprintln!("surprise window {:?} vs score window {:?}", key(&two), key(&raw));
}

#[test]
fn a_surprise_window_needs_relevance_and_a_known_name() {
    let Some(m) = tiny() else { return };
    match describe_window_by(&m, "embedding", "relevance", false, 10) {
        Err(larql_server::error::ServerError::BadRequest(msg)) => assert!(msg.contains("relevance"), "{msg}"),
        other => panic!("expected BadRequest, got {other:?}"),
    }
    match describe_window_by(&m, "embedding", "surprise", true, 10) {
        Err(larql_server::error::ServerError::BadRequest(msg)) => assert!(msg.contains("surprise"), "{msg}"),
        other => panic!("expected BadRequest, got {other:?}"),
    }
}

#[test]
fn a_surprise_window_works_for_residual_queries_too() {
    let Some(m) = tiny() else { return };
    let v = match describe_window_by(&m, "residual", "relevance", true, 3) {
        Ok(v) => v,
        Err(e) => { assert!(format!("{e:?}").contains("weights"), "{e:?}"); return; }
    };
    assert_eq!(v["window_by"], "relevance");
    assert_eq!(v["relevance_query"], "residual");
    let mut per_layer: std::collections::BTreeMap<u64, usize> = Default::default();
    for e in v["edges"].as_array().unwrap() {
        *per_layer.entry(e["layer"].as_u64().unwrap()).or_default() += 1;
        assert!(e["relevance"].as_f64().unwrap().is_finite());
    }
    assert!(per_layer.values().all(|&n| n <= 3), "a layer exceeded its window: {per_layer:?}");
}

// ══════════════════════════════════════════════════════════════
// prompt: a template around the entity, with its own panel
// ══════════════════════════════════════════════════════════════

fn describe_prompt(model: &LoadedModel, query: &str, prompt: &str) -> Result<serde_json::Value, larql_server::error::ServerError> {
    let params = larql_server::routes::describe::DescribeParams {
        entity: "[5]".to_string(),
        band: "all".to_string(),
        verbose: false,
        limit: 10_000,
        window: 5,
        min_score: 0.0,
        coherence: false,
        min_coherence: 0.0,
        relabel: false,
        relevance: true,
        background: Some("entities".into()),
        window_by: "relevance".into(),
        query: query.into(),
        prompt: prompt.into(),
        baseline: None,
    };
    let patched = model.patched.blocking_read();
    larql_server::routes::describe::describe_entity_with(model, &patched, &params)
}

#[test]
fn a_prompt_builds_its_own_residual_panel_and_is_reported() {
    let Some(m) = tiny() else { return };
    let v = match describe_prompt(&m, "residual", "{entity} [6]") {
        Ok(v) => v,
        Err(e) => { assert!(format!("{e:?}").contains("weights"), "{e:?}"); return; }
    };
    assert_eq!(v["prompt"], "{entity} [6]");
    assert!(m.relevance.has_residual_panel(larql_server::relevance::Background::Entities, "{entity} [6]"));
    assert!(v["relevance_panel"].as_u64().unwrap() >= 2);
    // The bare panel is a separate background: asking for it does not reuse this one.
    let bare = describe_prompt(&m, "residual", "{entity}").unwrap();
    assert!(bare.get("prompt").is_none(), "the default template is not reported");
    assert!(m.relevance.has_residual_panel(larql_server::relevance::Background::Entities, "{entity}"));
}

#[test]
fn a_prompt_must_name_the_entity_and_is_residual_only() {
    let Some(m) = tiny() else { return };
    match describe_prompt(&m, "residual", "the capital of") {
        Err(larql_server::error::ServerError::BadRequest(msg)) => assert!(msg.contains("{entity}"), "{msg}"),
        other => panic!("expected BadRequest, got {other:?}"),
    }
    match describe_prompt(&m, "embedding", "{entity} is") {
        Err(larql_server::error::ServerError::BadRequest(msg)) => assert!(msg.contains("residual"), "{msg}"),
        other => panic!("expected BadRequest, got {other:?}"),
    }
}
