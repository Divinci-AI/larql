//! Gates for the V3 mutation overlay: V2's patch semantics — all-or-
//! nothing apply, replay-on-remove — over the KNN subset the overlay
//! represents today.

use crate::patch::format::{encode_gate_vector, PatchOp, VindexPatch};

use super::super::KnowledgeOverlay;

fn knn_op(entity: &str, target: &str, layer: usize) -> PatchOp {
    PatchOp::InsertKnn {
        layer,
        entity: entity.into(),
        relation: "rel".into(),
        target: target.into(),
        target_id: 5,
        confidence: Some(0.8),
        key_vector_b64: encode_gate_vector(&[1.0, 0.0, 0.0, 0.0]),
    }
}

fn patch_of(description: &str, operations: Vec<PatchOp>) -> VindexPatch {
    VindexPatch {
        version: 1,
        base_model: "overlay-fixture".into(),
        base_checksum: None,
        created_at: String::new(),
        description: Some(description.into()),
        author: None,
        tags: vec![],
        operations,
    }
}

#[test]
fn apply_populates_the_store_and_records_the_patch() {
    let mut overlay = KnowledgeOverlay::new();
    overlay
        .try_apply_patch(patch_of("facts", vec![knn_op("atlantis", "[5]", 1)]))
        .unwrap();
    assert_eq!(overlay.knn_store.len(), 1);
    assert_eq!(overlay.patches.len(), 1);
    let entries = overlay.knn_store.entries_for_entity("atlantis");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].0, 1, "install layer travels with the op");
    assert_eq!(entries[0].1.target_token, "[5]");
}

#[test]
fn delete_knn_removes_by_entity() {
    let mut overlay = KnowledgeOverlay::new();
    overlay
        .try_apply_patch(patch_of(
            "add-then-drop",
            vec![
                knn_op("atlantis", "[5]", 1),
                knn_op("lemuria", "[6]", 1),
                PatchOp::DeleteKnn {
                    entity: "atlantis".into(),
                },
            ],
        ))
        .unwrap();
    assert!(overlay.knn_store.entries_for_entity("atlantis").is_empty());
    assert_eq!(overlay.knn_store.entries_for_entity("lemuria").len(), 1);
}

/// A vector-bearing op poisons the whole patch — no state change, not
/// recorded (the compose rung owns those ops). Both shapes refuse: a
/// compose Insert, and an Update carrying vectors.
#[test]
fn a_vector_bearing_op_refuses_the_whole_patch() {
    let mut overlay = KnowledgeOverlay::new();
    let err = overlay
        .try_apply_patch(patch_of(
            "mixed",
            vec![
                knn_op("atlantis", "[5]", 1),
                PatchOp::Insert {
                    layer: 0,
                    feature: 3,
                    relation: Some("rel".into()),
                    entity: "atlantis".into(),
                    target: "[5]".into(),
                    confidence: Some(1.0),
                    gate_vector_b64: Some(encode_gate_vector(&[1.0, 0.0])),
                    up_vector_b64: None,
                    down_vector_b64: None,
                    down_meta: None,
                },
            ],
        ))
        .expect_err("compose installs are the compose rung's");
    assert!(err.to_string().contains("compose rung"), "{err}");
    assert!(overlay.knn_store.is_empty(), "all-or-nothing was violated");
    assert!(overlay.patches.is_empty());

    let err = overlay
        .try_apply_patch(patch_of(
            "update-with-vectors",
            vec![PatchOp::Update {
                layer: 0,
                feature: 3,
                gate_vector_b64: Some(encode_gate_vector(&[1.0, 0.0])),
                up_vector_b64: None,
                down_vector_b64: None,
                down_meta: None,
            }],
        ))
        .expect_err("an Update carrying vectors is a compose edit");
    assert!(err.to_string().contains("compose rung"), "{err}");
    assert!(overlay.patches.is_empty());
}

#[test]
fn a_corrupt_key_vector_refuses_the_whole_patch() {
    let mut overlay = KnowledgeOverlay::new();
    let corrupt = VindexPatch {
        operations: vec![PatchOp::InsertKnn {
            layer: 0,
            entity: "atlantis".into(),
            relation: "rel".into(),
            target: "[5]".into(),
            target_id: 5,
            confidence: None,
            key_vector_b64: "!!!not-base64!!!".into(),
        }],
        ..patch_of("corrupt", vec![])
    };
    let err = overlay
        .try_apply_patch(corrupt)
        .expect_err("corrupt vectors must refuse");
    assert!(err.to_string().contains("key_vector_b64"), "{err}");
    assert!(overlay.knn_store.is_empty());
    assert!(overlay.patches.is_empty());
}

/// Removal rebuilds by replaying the remaining list — V2's contract,
/// session-added entries outside any patch included in the reset.
#[test]
fn remove_patch_replays_the_remaining_list() {
    let mut overlay = KnowledgeOverlay::new();
    overlay
        .try_apply_patch(patch_of("first", vec![knn_op("atlantis", "[5]", 1)]))
        .unwrap();
    overlay
        .try_apply_patch(patch_of("second", vec![knn_op("lemuria", "[6]", 1)]))
        .unwrap();
    // A session entry added outside any patch is lost on rebuild —
    // exactly as `PatchedVindex::rebuild_overrides` loses anonymous
    // session overrides.
    overlay.knn_store.add(
        0,
        vec![0.0, 1.0, 0.0, 0.0],
        7,
        "[7]".into(),
        "mu".into(),
        "rel".into(),
        1.0,
    );

    overlay.remove_patch(0);
    assert_eq!(overlay.patches.len(), 1);
    assert_eq!(overlay.patches[0].description.as_deref(), Some("second"));
    assert!(overlay.knn_store.entries_for_entity("atlantis").is_empty());
    assert_eq!(overlay.knn_store.entries_for_entity("lemuria").len(), 1);
    assert!(
        overlay.knn_store.entries_for_entity("mu").is_empty(),
        "session entries outside a patch reset on rebuild, as on V2"
    );
}

#[test]
fn remove_patch_out_of_range_is_a_no_op() {
    let mut overlay = KnowledgeOverlay::new();
    overlay
        .try_apply_patch(patch_of("only", vec![knn_op("atlantis", "[5]", 1)]))
        .unwrap();
    overlay.remove_patch(9);
    assert_eq!(overlay.patches.len(), 1);
    assert_eq!(overlay.knn_store.len(), 1);
}

fn meta(token: &str, c_score: f32) -> crate::index::types::FeatureMeta {
    crate::index::types::FeatureMeta {
        top_token: token.into(),
        top_token_id: 9,
        c_score,
        top_k: vec![],
    }
}

/// The V2 read rule at the source: an override wins, a tombstone hides,
/// otherwise the base answers — and UPDATE after DELETE resurrects.
#[test]
fn tombstone_and_resurrection_follow_the_v2_contract() {
    let mut overlay = KnowledgeOverlay::new();
    let base = Some(meta("[3]", 1.0));

    // Untouched slot: base passes through.
    assert_eq!(
        overlay
            .resolve_feature_meta(0, 0, base.clone())
            .map(|m| m.top_token),
        Some("[3]".to_string()),
        "untouched slots read the base"
    );
    assert!(!overlay.has_feature_state());

    // DELETE: the slot reads absent even though the base has it.
    overlay.delete_feature(0, 0);
    assert!(overlay.resolve_feature_meta(0, 0, base.clone()).is_none());
    assert!(overlay.is_tombstoned(0, 0));
    assert_eq!(overlay.tombstones_at(0), 1);
    assert_eq!(overlay.tombstones_at(1), 0);
    assert!(overlay.has_feature_state());

    // UPDATE: resurrects with the new annotation.
    overlay.update_feature_meta(0, 0, meta("[9]", 0.5));
    let resolved = overlay.resolve_feature_meta(0, 0, base).unwrap();
    assert_eq!(resolved.top_token, "[9]");
    assert!(!overlay.is_tombstoned(0, 0));
}

/// The layer-vector merge used by `feature_metas`-shaped reads.
#[test]
fn apply_meta_overrides_edits_the_layer_in_place() {
    let mut overlay = KnowledgeOverlay::new();
    overlay.delete_feature(0, 0);
    overlay.update_feature_meta(0, 2, meta("[9]", 0.5));

    let mut metas = vec![Some(meta("[1]", 1.0)), Some(meta("[2]", 1.0)), None];
    overlay.apply_meta_overrides(0, &mut metas);
    assert!(metas[0].is_none(), "tombstoned slot hidden");
    assert_eq!(
        metas[1].as_ref().map(|m| m.top_token.as_str()),
        Some("[2]"),
        "untouched slot intact"
    );
    assert_eq!(
        metas[2].as_ref().map(|m| m.top_token.as_str()),
        Some("[9]"),
        "override lands even where the base had nothing"
    );

    // A different layer is untouched.
    let mut other = vec![Some(meta("[1]", 1.0))];
    overlay.apply_meta_overrides(1, &mut other);
    assert!(other[0].is_some());
}

/// Vector-free Delete/Update patch ops replay with V2's resolution:
/// the Update's meta is constructed exactly as `overlay_apply` does
/// (single-entry top_k), and an Update WITHOUT meta after a Delete
/// drops the pinned `None` so reads fall through to the base.
#[test]
fn feature_patch_ops_replay_with_v2_resolution() {
    let mut overlay = KnowledgeOverlay::new();
    overlay
        .try_apply_patch(patch_of(
            "edits",
            vec![
                PatchOp::Delete {
                    layer: 0,
                    feature: 0,
                    reason: None,
                },
                PatchOp::Update {
                    layer: 1,
                    feature: 1,
                    gate_vector_b64: None,
                    up_vector_b64: None,
                    down_vector_b64: None,
                    down_meta: Some(crate::patch::format::PatchDownMeta {
                        top_token: "[9]".into(),
                        top_token_id: 9,
                        c_score: 0.5,
                    }),
                },
            ],
        ))
        .unwrap();
    assert!(overlay
        .resolve_feature_meta(0, 0, Some(meta("[3]", 1.0)))
        .is_none());
    let updated = overlay.resolve_feature_meta(1, 1, None).unwrap();
    assert_eq!(updated.top_token, "[9]");
    assert_eq!(updated.top_k.len(), 1, "V2 builds a single-entry top_k");
    assert_eq!(updated.top_k[0].token_id, 9);

    // Delete then meta-less Update: the pin drops, base answers again.
    let mut overlay = KnowledgeOverlay::new();
    overlay
        .try_apply_patch(patch_of(
            "delete-then-touch",
            vec![
                PatchOp::Delete {
                    layer: 0,
                    feature: 0,
                    reason: None,
                },
                PatchOp::Update {
                    layer: 0,
                    feature: 0,
                    gate_vector_b64: None,
                    up_vector_b64: None,
                    down_vector_b64: None,
                    down_meta: None,
                },
            ],
        ))
        .unwrap();
    let base = Some(meta("[3]", 1.0));
    assert_eq!(
        overlay
            .resolve_feature_meta(0, 0, base)
            .map(|m| m.top_token),
        Some("[3]".to_string()),
        "the meta-less Update must drop the pinned None (V2 rule)"
    );
    assert!(!overlay.is_tombstoned(0, 0));
}

/// remove_patch resets feature-slot state along with the KNN store.
#[test]
fn remove_patch_clears_feature_state_too() {
    let mut overlay = KnowledgeOverlay::new();
    overlay
        .try_apply_patch(patch_of(
            "only",
            vec![PatchOp::Delete {
                layer: 0,
                feature: 0,
                reason: None,
            }],
        ))
        .unwrap();
    overlay.update_feature_meta(1, 1, meta("[9]", 0.5));
    overlay.remove_patch(0);
    assert!(!overlay.has_feature_state(), "rebuild resets slot state");
    let base = Some(meta("[3]", 1.0));
    assert_eq!(
        overlay
            .resolve_feature_meta(0, 0, base)
            .map(|m| m.top_token),
        Some("[3]".to_string())
    );
}
