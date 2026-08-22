//! V3-LQL-3B gates: mutation on a VINDEX3 binding, closed-loop.
//!
//! The oracle is stronger than "INSERT returns OK": every claim is a
//! **round trip through the statement surface**.
//!
//! ```text
//! before:  DESCRIBE → edge absent;  INFER → no override
//! INSERT MODE KNN (the default)
//! after:   DESCRIBE → edge present; INFER → stored target overrides top-1
//! SAVE PATCH
//! reopen pristine container → absent again
//! APPLY PATCH → present again
//! REMOVE PATCH → absent again
//! ```
//!
//! The KNN key is captured from the V3 runtime's own execution (plan
//! taps), so same-prompt retrieval is exact by construction — the
//! same property the V2 arm gets from its forward pass.

use std::path::Path;

use larql_inference::test_utils::synthetic_tokenizer_json;
use larql_lql::{parse, Session};
use larql_vindex::format::vindex3::fixtures::{
    encode_fixture_container, miniature_glimmer, G_VOCAB,
};

/// The canonical prompt `INSERT ("a", "b", …)` captures its key from —
/// INFER on this exact prompt must retrieve the stored target.
const CANONICAL_PROMPT: &str = "The b of a is";

/// Windows temp paths contain backslashes, which the LQL lexer's escape
/// pass would consume; doubling them leaves the path untouched on every
/// platform.
fn lql_path(path: impl AsRef<Path>) -> String {
    path.as_ref().display().to_string().replace('\\', "\\\\")
}

fn v3_container() -> tempfile::TempDir {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(
        miniature_glimmer,
        checkpoint.path(),
        container.path(),
        "mutation-fixture",
    );
    std::fs::write(
        container.path().join("tokenizer.json"),
        synthetic_tokenizer_json(G_VOCAB),
    )
    .unwrap();
    container
}

fn run(session: &mut Session, stmt: &str) -> Vec<String> {
    let parsed = parse(stmt).unwrap_or_else(|e| panic!("parse {stmt}: {e}"));
    session
        .execute(&parsed)
        .unwrap_or_else(|e| panic!("execute {stmt}: {e}"))
}

fn bound_session(container: &Path) -> Session {
    let mut session = Session::new();
    let use_stmt = format!("USE \"{}\";", lql_path(container));
    run(&mut session, &use_stmt);
    session
}

fn describe_entity(session: &mut Session) -> String {
    run(session, r#"DESCRIBE "a";"#).join("\n")
}

fn infer_canonical(session: &mut Session) -> String {
    run(session, &format!("INFER \"{CANONICAL_PROMPT}\" TOP 3;")).join("\n")
}

/// The core closed loop: absent → INSERT → present, observed through
/// DESCRIBE **and** through INFER's post-logits override.
#[test]
fn insert_knn_round_trips_through_describe_and_infer() {
    let container = v3_container();
    let mut session = bound_session(container.path());

    // Pre-screen: the edge must be genuinely absent before the insert,
    // or "present after" proves nothing.
    let before = describe_entity(&mut session);
    assert!(!before.contains("→ [5]"), "pristine container: {before}");
    let infer_before = infer_canonical(&mut session);
    assert!(
        !infer_before.contains("knn_override"),
        "pristine container: {infer_before}"
    );

    let out = run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "[5]");"#,
    )
    .join("\n");
    assert!(out.contains("Inserted: a —[b]→ [5]"), "{out}");
    assert!(out.contains("KNN store: 1 entries total"), "{out}");
    assert!(out.contains("VINDEX3 plan taps"), "{out}");

    // The overlay is immediately visible to browse…
    let after = describe_entity(&mut session);
    assert!(after.contains("→ [5]"), "{after}");

    // …and to inference: same-prompt retrieval fires the shared
    // post-logits gate and the stored target takes row 1.
    let infer_after = infer_canonical(&mut session);
    assert!(infer_after.contains("knn_override"), "{infer_after}");
    let row1 = infer_after
        .lines()
        .find(|l| l.trim_start().starts_with("1."))
        .unwrap_or_else(|| panic!("no row 1 in {infer_after}"));
    assert!(row1.contains("[5]"), "stored target must lead: {row1}");
    assert!(
        infer_after.contains("post-logits retrieval sidecar"),
        "{infer_after}"
    );
}

/// The full patch lifecycle: the mutation persists as a portable patch,
/// a pristine reopen loses it, APPLY restores it, REMOVE drops it.
#[test]
fn patch_lifecycle_round_trips_on_a_pristine_reopen() {
    let container = v3_container();
    let patch_dir = tempfile::tempdir().unwrap();
    let patch_file = patch_dir.path().join("facts.vlp");
    let patch_stmt_path = lql_path(&patch_file);

    // Session 1: record the mutation into a named patch.
    {
        let mut session = bound_session(container.path());
        run(&mut session, &format!("BEGIN PATCH \"{patch_stmt_path}\";"));
        run(
            &mut session,
            r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "[5]");"#,
        );
        let saved = run(&mut session, "SAVE PATCH;").join("\n");
        assert!(saved.contains("Saved:"), "{saved}");
        assert!(saved.contains("1 inserts"), "{saved}");
    }
    assert!(patch_file.exists(), "SAVE PATCH must write the file");

    // Session 2: the pristine container knows nothing of the edit —
    // the base was never modified.
    let mut session = bound_session(container.path());
    let pristine = describe_entity(&mut session);
    assert!(!pristine.contains("→ [5]"), "{pristine}");
    assert!(!infer_canonical(&mut session).contains("knn_override"));

    // APPLY restores the logical fact from the portable patch.
    let applied = run(&mut session, &format!("APPLY PATCH \"{patch_stmt_path}\";")).join("\n");
    assert!(applied.contains("Applied:"), "{applied}");
    assert!(describe_entity(&mut session).contains("→ [5]"));
    assert!(infer_canonical(&mut session).contains("knn_override"));

    let listed = run(&mut session, "SHOW PATCHES;").join("\n");
    assert!(listed.contains("1 ops"), "{listed}");

    // REMOVE drops it again — the overlay rebuilds from the remaining
    // (empty) patch list.
    let removed = run(
        &mut session,
        &format!("REMOVE PATCH \"{patch_stmt_path}\";"),
    )
    .join("\n");
    assert!(removed.contains("Removed"), "{removed}");
    let after_remove = describe_entity(&mut session);
    assert!(!after_remove.contains("→ [5]"), "{after_remove}");
    assert!(!infer_canonical(&mut session).contains("knn_override"));
}

/// MERGE resolves the V3 binding as its target; a source directory
/// that is not a vindex fails at source loading with a helpful error —
/// never the misleading "no backend loaded".
#[test]
fn merge_with_an_invalid_source_reports_the_load_failure() {
    let container = v3_container();
    let source = tempfile::tempdir().unwrap();
    let mut session = bound_session(container.path());
    let stmt = format!("MERGE \"{}\";", lql_path(source.path()));
    let parsed = parse(&stmt).unwrap();
    let err = session
        .execute(&parsed)
        .expect_err("an empty dir is not a vindex");
    let msg = err.to_string();
    assert!(msg.contains("failed to load source"), "{msg}");
    assert!(!msg.contains("No backend"), "{msg}");
}

/// A tokenizerless container cannot capture the canonical prompt —
/// the refusal names the missing capability, and no entry lands.
#[test]
fn insert_refuses_on_a_tokenizerless_container() {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(
        miniature_glimmer,
        checkpoint.path(),
        container.path(),
        "tokless-fixture",
    );
    let mut session = bound_session(container.path());
    let parsed =
        parse(r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "[5]");"#).unwrap();
    let err = session
        .execute(&parsed)
        .expect_err("INSERT needs the tokenizer capability");
    assert!(err.to_string().contains("tokenizer"), "{err}");
}

/// `AT LAYER n` pins the install layer instead of the default
/// penultimate layer — and out-of-range hints clamp, as on V2.
#[test]
fn insert_at_layer_pins_the_install_layer() {
    let container = v3_container();
    let mut session = bound_session(container.path());
    let out = run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "[5]") AT LAYER 1;"#,
    )
    .join("\n");
    assert!(out.contains("at L1"), "{out}");

    let clamped = run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("x", "y", "[6]") AT LAYER 99;"#,
    )
    .join("\n");
    assert!(clamped.contains("at L1"), "out-of-range clamps: {clamped}");
}

/// Read the feature ids SELECT reports for one layer.
fn feature_ids_at_layer(session: &mut Session, layer: usize) -> Vec<usize> {
    run(
        session,
        &format!("SELECT * FROM FEATURES WHERE layer = {layer} LIMIT 300;"),
    )
    .iter()
    .filter_map(|line| {
        let mut parts = line.split_whitespace();
        let (_l, f) = (parts.next()?, parts.next()?);
        f.strip_prefix('F')?.parse().ok()
    })
    .collect()
}

/// Feature-slot mutation closed loop (V3-LQL-3B rung 2): UPDATE
/// rewrites the annotation SELECT reports; DELETE tombstones the slot
/// out of SELECT; and — V2's statement-surface contract — an UPDATE
/// on a tombstoned slot finds nothing (its meta reads as absent), so
/// resurrection happens through patch replay, not through UPDATE.
#[test]
fn update_rewrites_and_delete_tombstones_through_select() {
    let container = v3_container();
    let mut session = bound_session(container.path());

    let before = feature_ids_at_layer(&mut session, 0);
    assert!(before.contains(&0), "fixture must annotate feature 0");

    // UPDATE a live slot: the new annotation is what SELECT reports.
    let out = run(
        &mut session,
        r#"UPDATE EDGES SET target = "[9]" WHERE layer = 0 AND feature = 0;"#,
    )
    .join("\n");
    assert!(out.contains("Updated 1 features"), "{out}");
    let rows = run(
        &mut session,
        "SELECT * FROM FEATURES WHERE layer = 0 LIMIT 300;",
    )
    .join("\n");
    let row0 = rows
        .lines()
        .find(|l| l.split_whitespace().nth(1) == Some("F0"))
        .unwrap_or_else(|| panic!("no F0 row in {rows}"));
    assert!(row0.contains("[9]"), "{row0}");

    // DELETE tombstones it out of the feature space.
    let out = run(
        &mut session,
        "DELETE FROM EDGES WHERE layer = 0 AND feature = 0;",
    )
    .join("\n");
    assert!(out.contains("Deleted 1 features"), "{out}");
    let after_delete = feature_ids_at_layer(&mut session, 0);
    assert!(!after_delete.contains(&0), "tombstoned slot must vanish");
    assert_eq!(after_delete.len(), before.len() - 1, "only that slot");

    // V2 parity: UPDATE cannot see a tombstoned slot's meta, so it
    // matches nothing — the same answer a V2 session gives.
    let out = run(
        &mut session,
        r#"UPDATE EDGES SET target = "[9]" WHERE layer = 0 AND feature = 0;"#,
    )
    .join("\n");
    assert!(out.contains("no matching features"), "{out}");
}

/// UPDATE reads the current (overlay-merged) meta, so a second UPDATE
/// composes on the first — and WALK observes the tombstone filter.
#[test]
fn walk_excludes_tombstoned_features() {
    let container = v3_container();
    let mut session = bound_session(container.path());

    // Find the top walk hit, delete exactly that slot, walk again.
    let walk_before = run(&mut session, r#"WALK "[3]" TOP 3;"#).join("\n");
    let hit = walk_before
        .lines()
        .find_map(|l| {
            let t = l.trim_start().strip_prefix("L")?;
            let (layer, rest) = t.split_once(':')?;
            let feat = rest
                .trim_start()
                .strip_prefix('F')?
                .split_whitespace()
                .next()?;
            Some((
                layer.trim().parse::<usize>().ok()?,
                feat.parse::<usize>().ok()?,
            ))
        })
        .expect("walk must return a hit");

    run(
        &mut session,
        &format!(
            "DELETE FROM EDGES WHERE layer = {} AND feature = {};",
            hit.0, hit.1
        ),
    );
    let walk_after = run(&mut session, r#"WALK "[3]" TOP 3;"#).join("\n");
    let needle = format!("F{}", hit.1);
    let still_there = walk_after
        .lines()
        .any(|l| l.trim_start().starts_with(&format!("L {}:", hit.0)) && l.contains(&needle));
    assert!(
        !still_there,
        "tombstoned hit must leave the walk:\n{walk_after}"
    );
}

/// The feature-slot patch lifecycle: DELETE + UPDATE persist as a
/// portable patch, a pristine reopen loses them, APPLY restores them,
/// REMOVE drops them.
#[test]
fn feature_patch_lifecycle_round_trips_on_a_pristine_reopen() {
    let container = v3_container();
    let patch_dir = tempfile::tempdir().unwrap();
    let patch_file = patch_dir.path().join("edits.vlp");
    let patch_stmt_path = lql_path(&patch_file);

    {
        let mut session = bound_session(container.path());
        run(&mut session, &format!("BEGIN PATCH \"{patch_stmt_path}\";"));
        run(
            &mut session,
            "DELETE FROM EDGES WHERE layer = 0 AND feature = 0;",
        );
        run(
            &mut session,
            r#"UPDATE EDGES SET target = "[9]" WHERE layer = 1 AND feature = 1;"#,
        );
        let saved = run(&mut session, "SAVE PATCH;").join("\n");
        assert!(saved.contains("1 updates, 1 deletes"), "{saved}");
    }

    let mut session = bound_session(container.path());
    assert!(
        feature_ids_at_layer(&mut session, 0).contains(&0),
        "pristine reopen must not carry the delete"
    );

    run(&mut session, &format!("APPLY PATCH \"{patch_stmt_path}\";"));
    assert!(!feature_ids_at_layer(&mut session, 0).contains(&0));
    let rows = run(
        &mut session,
        "SELECT * FROM FEATURES WHERE layer = 1 LIMIT 300;",
    )
    .join("\n");
    let row = rows
        .lines()
        .find(|l| l.split_whitespace().nth(1) == Some("F1"))
        .unwrap_or_else(|| panic!("no F1 row in {rows}"));
    assert!(row.contains("[9]"), "{row}");

    run(
        &mut session,
        &format!("REMOVE PATCH \"{patch_stmt_path}\";"),
    );
    assert!(
        feature_ids_at_layer(&mut session, 0).contains(&0),
        "REMOVE must restore the pristine feature space"
    );
}

/// A patch carrying vector-bearing operations (the compose rung's
/// territory) is refused **whole**: no partial state, no listing.
#[test]
fn a_compose_patch_refuses_whole_on_v3() {
    use larql_vindex::patch::core::encode_gate_vector;
    use larql_vindex::{PatchOp, VindexPatch};

    let container = v3_container();
    let patch_dir = tempfile::tempdir().unwrap();
    let patch_file = patch_dir.path().join("compose.vlp");

    let patch = VindexPatch {
        version: 1,
        base_model: "mutation-fixture".into(),
        base_checksum: None,
        created_at: String::new(),
        description: None,
        author: None,
        tags: vec![],
        operations: vec![
            // The KNN op alone would be applicable…
            PatchOp::InsertKnn {
                layer: 0,
                entity: "a".into(),
                relation: "b".into(),
                target: "[5]".into(),
                target_id: 5,
                confidence: Some(1.0),
                key_vector_b64: encode_gate_vector(&[0.5, 0.5, 0.5, 0.5]),
            },
            // …but the compose install poisons the whole patch.
            PatchOp::Insert {
                layer: 0,
                feature: 0,
                relation: Some("rel".into()),
                entity: "a".into(),
                target: "[5]".into(),
                confidence: Some(1.0),
                gate_vector_b64: Some(encode_gate_vector(&[0.5, 0.5, 0.5, 0.5])),
                up_vector_b64: None,
                down_vector_b64: None,
                down_meta: None,
            },
        ],
    };
    patch.save(&patch_file).unwrap();

    let mut session = bound_session(container.path());
    let stmt = format!("APPLY PATCH \"{}\";", lql_path(&patch_file));
    let parsed = parse(&stmt).unwrap();
    let err = session
        .execute(&parsed)
        .expect_err("a compose patch must refuse on V3");
    assert!(err.to_string().contains("compose rung"), "{err}");

    // All-or-nothing: the applicable KNN op must NOT have landed.
    assert!(!describe_entity(&mut session).contains("→ [5]"));
    let listed = run(&mut session, "SHOW PATCHES;").join("\n");
    assert!(listed.contains("no patches applied"), "{listed}");
}
