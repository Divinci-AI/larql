//! The V2 ↔ V3 whole-language parity harness (V3-LQL-3A gate).
//!
//! One source checkpoint (the dense Llama-shaped fixture, LCG-seeded
//! and therefore byte-reproducible) is realised BOTH ways:
//!
//! - **V2**: loaded as `ModelWeights` and extracted with the real
//!   `build_vindex` pipeline (f32 storage, `ExtractLevel::All`);
//! - **V3**: encoded as a VINDEX3 container by `encode_system`.
//!
//! The same LQL script runs against both bindings and must produce
//! equivalent logical results. Preregistered contract for this rung:
//!
//! - **exact**: the feature space — per layer, the set and identity of
//!   `(feature id → top token)`; WALK's per-layer hit feature ids.
//! - **matched by construction**: annotation semantics (`c_score` =
//!   top logit of `embed · feature_down`) — the V3 role derivation
//!   implements the V2 extractor's contract verbatim, and the first
//!   run of this harness is what caught the original divergence
//!   (V3 initially scored against the output head).
//! - **excluded** (explicitly, not silently): relation labels (no
//!   label sidecars exist on either side here) and gate-score display
//!   strings (compared as ids/ordering, not text).
//!
//! Controls precede the parity claim: the extractor of logical rows
//! must be stable across repeated runs of one arm, and must DIFFER
//! across genuinely different models — otherwise "V2 == V3" would be
//! vacuous.

use std::collections::BTreeMap;
use std::path::Path;

/// An UNAMBIGUOUS `[N]` ↔ id N tokenizer: `unk_token` points at the
/// existing `"[0]"` entry instead of aliasing a second surface onto
/// id 0. The shared `synthetic_tokenizer_json` maps both `"[0]"` and
/// `"[UNK]"` to id 0, and the V2 down-meta reader and the V3 view
/// resolve that alias differently — a fixture artifact the first
/// parity run flagged as a false divergence. Parity fixtures must not
/// carry ambiguous vocabularies.
fn unambiguous_tokenizer_json(vocab: usize) -> String {
    let entries: Vec<String> = (0..vocab).map(|i| format!("\"[{i}]\":{i}")).collect();
    format!(
        "{{\"version\":\"1.0\",\"truncation\":null,\"padding\":null,\"added_tokens\":[],\
         \"normalizer\":null,\"pre_tokenizer\":null,\"post_processor\":null,\"decoder\":null,\
         \"model\":{{\"type\":\"WordLevel\",\"vocab\":{{{}}},\"unk_token\":\"[0]\"}}}}",
        entries.join(",")
    )
}
use larql_lql::{parse, Session};
use larql_vindex::format::vindex3::fixtures::{
    dense_f32_model, encode_fixture_container, miniature_glimmer, DENSE_LAYERS, DENSE_VOCAB,
    G_VOCAB,
};

fn run(session: &mut Session, stmt: &str) -> Vec<String> {
    let parsed = parse(stmt).unwrap_or_else(|e| panic!("parse {stmt}: {e}"));
    session
        .execute(&parsed)
        .unwrap_or_else(|e| panic!("execute {stmt}: {e}"))
}

fn session_for(dir: &Path) -> Session {
    let mut session = Session::new();
    run(&mut session, &format!("USE \"{}\";", dir.display()));
    session
}

/// The V2 realisation: checkpoint → ModelWeights → real extraction.
fn v2_vindex() -> tempfile::TempDir {
    let checkpoint = tempfile::tempdir().unwrap();
    dense_f32_model(checkpoint.path());
    let weights = larql_inference::load_model_dir(checkpoint.path()).expect("load checkpoint");

    let out = tempfile::tempdir().unwrap();
    let tok_json = unambiguous_tokenizer_json(DENSE_VOCAB);
    std::fs::write(out.path().join("tokenizer.json"), &tok_json).unwrap();
    let tokenizer = larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json.as_bytes()).unwrap();
    let mut cb = larql_vindex::SilentBuildCallbacks;
    larql_vindex::build_vindex(
        &weights,
        &tokenizer,
        "parity/dense",
        out.path(),
        8,
        larql_vindex::ExtractLevel::All,
        larql_vindex::StorageDtype::F32,
        &mut cb,
    )
    .expect("build V2 vindex");
    // build_vindex may rewrite dir contents; make sure the tokenizer
    // is present for the V2 loaders.
    std::fs::write(
        out.path().join("tokenizer.json"),
        unambiguous_tokenizer_json(DENSE_VOCAB),
    )
    .unwrap();
    out
}

/// The V3 realisation of the SAME checkpoint (LCG-seeded writer —
/// identical bytes).
fn v3_container() -> tempfile::TempDir {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(
        dense_f32_model,
        checkpoint.path(),
        container.path(),
        "parity-dense",
    );
    std::fs::write(
        container.path().join("tokenizer.json"),
        unambiguous_tokenizer_json(DENSE_VOCAB),
    )
    .unwrap();
    container
}

/// The logical feature space one binding reports: per layer, feature
/// id → top token, read from `SELECT * FROM FEATURES` rows
/// (`L<layer>  F<feat>  <token> …`).
fn feature_space(
    session: &mut Session,
    layers: usize,
    limit: usize,
) -> BTreeMap<(usize, usize), String> {
    let mut space = BTreeMap::new();
    for layer in 0..layers {
        let out = run(
            session,
            &format!("SELECT * FROM FEATURES WHERE layer = {layer} LIMIT {limit};"),
        );
        for line in &out {
            let mut parts = line.split_whitespace();
            let (Some(l), Some(f), Some(token)) = (parts.next(), parts.next(), parts.next()) else {
                continue;
            };
            let (Some(l), Some(f)) = (l.strip_prefix('L'), f.strip_prefix('F')) else {
                continue;
            };
            let (Ok(l), Ok(f)) = (l.parse::<usize>(), f.parse::<usize>()) else {
                continue;
            };
            space.insert((l, f), token.to_string());
        }
    }
    space
}

/// WALK's logical result: per layer, the hit feature ids in rank order
/// (`  L 0: F14 …`).
fn walk_hits(session: &mut Session, prompt: &str) -> Vec<(usize, usize)> {
    let out = run(session, &format!("WALK \"{prompt}\" TOP 5;"));
    let mut hits = Vec::new();
    for line in &out {
        let trimmed = line.trim_start();
        let Some(rest) = trimmed.strip_prefix('L') else {
            continue;
        };
        let Some((layer, rest)) = rest.split_once(':') else {
            continue;
        };
        let Ok(layer) = layer.trim().parse::<usize>() else {
            continue;
        };
        let Some(feat) = rest.trim_start().strip_prefix('F') else {
            continue;
        };
        let Some(feat) = feat.split_whitespace().next() else {
            continue;
        };
        let Ok(feat) = feat.parse::<usize>() else {
            continue;
        };
        hits.push((layer, feat));
    }
    hits
}

/// Control 1: the instrument is stable — one arm, twice, identical.
#[test]
fn the_parity_instrument_is_stable_across_runs() {
    let v3 = v3_container();
    let mut a = session_for(v3.path());
    let mut b = session_for(v3.path());
    assert_eq!(
        feature_space(&mut a, DENSE_LAYERS, 64),
        feature_space(&mut b, DENSE_LAYERS, 64)
    );
    assert_eq!(walk_hits(&mut a, "[3]"), walk_hits(&mut b, "[3]"));
}

/// Control 2: the instrument detects genuinely different models —
/// the dense fixture's feature space is not the miniature's.
#[test]
fn the_parity_instrument_detects_different_models() {
    let dense = v3_container();
    let mini_checkpoint = tempfile::tempdir().unwrap();
    let mini = tempfile::tempdir().unwrap();
    encode_fixture_container(
        miniature_glimmer,
        mini_checkpoint.path(),
        mini.path(),
        "parity-mini",
    );
    std::fs::write(
        mini.path().join("tokenizer.json"),
        unambiguous_tokenizer_json(G_VOCAB),
    )
    .unwrap();

    let mut a = session_for(dense.path());
    let mut b = session_for(mini.path());
    assert_ne!(
        feature_space(&mut a, DENSE_LAYERS, 64),
        feature_space(&mut b, DENSE_LAYERS, 64),
        "instrument cannot tell different models apart"
    );
}

/// THE gate: one checkpoint, two formats, one script — the same
/// logical feature space and the same walk results.
#[test]
fn v2_and_v3_report_the_same_logical_results() {
    let v2 = v2_vindex();
    let v3 = v3_container();
    let mut v2_session = session_for(v2.path());
    let mut v3_session = session_for(v3.path());

    // Feature space: identity and annotation, exact.
    let v2_space = feature_space(&mut v2_session, DENSE_LAYERS, 300);
    let v3_space = feature_space(&mut v3_session, DENSE_LAYERS, 300);
    assert!(!v2_space.is_empty(), "V2 arm reported no features");
    assert_eq!(
        v2_space, v3_space,
        "the two formats disagree about the feature space"
    );

    // WALK: same prompt, same per-layer hit ids in the same order.
    let v2_hits = walk_hits(&mut v2_session, "[3]");
    let v3_hits = walk_hits(&mut v3_session, "[3]");
    assert!(!v2_hits.is_empty(), "V2 arm walked to nothing");
    assert_eq!(v2_hits, v3_hits, "walk results diverge between formats");

    // DESCRIBE runs on both and agrees about edge presence.
    let v2_describe = run(&mut v2_session, r#"DESCRIBE "[3]";"#).join("\n");
    let v3_describe = run(&mut v3_session, r#"DESCRIBE "[3]";"#).join("\n");
    assert_eq!(
        v2_describe.contains("(no edges found)"),
        v3_describe.contains("(no edges found)"),
        "DESCRIBE disagrees about edge presence:\nV2: {v2_describe}\nV3: {v3_describe}"
    );
}
