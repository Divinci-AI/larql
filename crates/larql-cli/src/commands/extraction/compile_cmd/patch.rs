//! Vindex-patch compilation: read .vlp patches and land every op in the
//! model's FFN slots — `insert` installs an edge, `delete` tombstones the
//! slot — or refuse before writing anything.
//!
//! Trigger for an insert comes from the patch's stored gate vector; write
//! comes from the down_meta target token's embedding when present.
//!
//! The invariant this module holds (see `mod.rs`): the checkpoint it writes
//! carries the whole patch set. An op with no compile path fails the run by
//! name; an insert the compiler cannot place fails it unless
//! `--allow-partial`; and nothing is written until that has been decided.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

use larql_vindex::PatchOp;
use ndarray::ArcArray2;

use super::detect::{decode_f32_b64, detect_ffn_pattern, ensure_cloned};
use super::edge::{install_edge, tombstone_slot};
use super::save::{copy_model_config, merge_for_save, write_safetensors};
use super::CompileArgs;

/// Which ops the compiler will act on and which it must refuse. Pure, so the
/// refusal rule is testable without a model: a `delete` may never end up in
/// `refused`, and a KNN op may never end up anywhere else.
#[derive(Debug, Default)]
pub struct OpPlan<'a> {
    pub inserts: Vec<&'a PatchOp>,
    pub deletes: Vec<&'a PatchOp>,
    /// op name → count, for ops with no compile path.
    pub refused: BTreeMap<&'static str, usize>,
}

pub fn plan_ops(ops: &[PatchOp]) -> OpPlan<'_> {
    let mut plan = OpPlan::default();
    for op in ops {
        match op {
            PatchOp::Insert { .. } => plan.inserts.push(op),
            PatchOp::Delete { .. } => plan.deletes.push(op),
            // `update` carries raw vector overrides the overlay writes
            // verbatim; the compiler has no norm-matched path for that yet
            // and must not guess one silently.
            PatchOp::Update { .. } => *plan.refused.entry("update").or_default() += 1,
            // KNN ops live at post_logits, not in the FFN. There is no
            // weight to write.
            PatchOp::InsertKnn { .. } => *plan.refused.entry("insert_knn").or_default() += 1,
            PatchOp::DeleteKnn { .. } => *plan.refused.entry("delete_knn").or_default() += 1,
        }
    }
    plan
}

fn refusal_message(refused: &BTreeMap<&'static str, usize>) -> String {
    let parts: Vec<String> = refused
        .iter()
        .map(|(op, n)| format!("{} × `{}`", n, op))
        .collect();
    format!(
        "refusing to compile: {} — these ops have no weight-level compile path, \
         and a checkpoint that silently dropped them would misrepresent the patch set. \
         Nothing was written.",
        parts.join(", ")
    )
}

pub fn run(args: CompileArgs) -> Result<(), Box<dyn std::error::Error>> {
    let vindex_path = args.vindex.as_ref().unwrap();
    eprintln!("LARQL AOT Compiler — patch mode");
    eprintln!("  base model: {}", args.base.display());
    eprintln!("  vindex:     {}", vindex_path.display());
    eprintln!("  output:     {}", args.output.display());

    eprintln!("\nLoading patches...");
    let patch_files: Vec<PathBuf> = if vindex_path.is_file() {
        vec![vindex_path.clone()]
    } else {
        std::fs::read_dir(vindex_path)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|ext| ext == "vlp"))
            .collect()
    };

    let mut all_ops = Vec::new();
    for pf in &patch_files {
        let patch = larql_vindex::VindexPatch::load(pf)?;
        eprintln!("  patch: {} ({} ops)", pf.display(), patch.operations.len());
        all_ops.extend(patch.operations);
    }

    eprintln!("  total patch operations: {}", all_ops.len());
    if all_ops.is_empty() {
        eprintln!("  no patches found — nothing to compile");
        return Ok(());
    }

    // Decide what can land BEFORE loading a multi-GB model or writing a
    // byte: a refusal must cost seconds, not the whole run.
    let plan = plan_ops(&all_ops);
    eprintln!(
        "  planned: {} insert, {} delete, {} refused",
        plan.inserts.len(),
        plan.deletes.len(),
        plan.refused.values().sum::<usize>()
    );
    if !plan.refused.is_empty() {
        return Err(refusal_message(&plan.refused).into());
    }

    eprintln!("\nLoading base model...");
    let weights = larql_models::loading::load_model_dir(&args.base)?;
    let config = weights.arch.config();
    eprintln!(
        "  {} layers, hidden={}, ffn={}",
        config.num_layers, config.hidden_size, config.intermediate_size
    );

    let gate_pattern = detect_ffn_pattern(&weights.tensors, "gate");
    let up_pattern = detect_ffn_pattern(&weights.tensors, "up");
    let down_pattern = detect_ffn_pattern(&weights.tensors, "down");
    eprintln!("  gate pattern: {}", gate_pattern.replace("{}", "N"));
    eprintln!("  up pattern:   {}", up_pattern.replace("{}", "N"));
    eprintln!("  down pattern:  {}", down_pattern.replace("{}", "N"));

    eprintln!("\nCompiling patches into weights...");
    let mut modified: HashMap<String, ArcArray2<f32>> = HashMap::new();
    let mut n_inserted = 0usize;
    let mut n_insert_skipped = 0usize;
    let mut n_deleted = 0usize;
    let mut n_delete_already_dead = 0usize;

    for op in &plan.inserts {
        let PatchOp::Insert {
            layer,
            feature,
            gate_vector_b64,
            entity,
            target,
            down_meta,
            ..
        } = op
        else {
            unreachable!("plan_ops only puts Insert here");
        };

        let Some(b64) = gate_vector_b64 else {
            eprintln!(
                "  skip: insert at L{}[{}] has no gate vector",
                layer, feature
            );
            n_insert_skipped += 1;
            continue;
        };
        let gate_vec = decode_f32_b64(b64)?;

        let gate_key = gate_pattern.replace("{}", &layer.to_string());
        let up_key = up_pattern.replace("{}", &layer.to_string());
        let down_key = down_pattern.replace("{}", &layer.to_string());

        ensure_cloned(&mut modified, &weights.tensors, &gate_key)?;
        ensure_cloned(&mut modified, &weights.tensors, &up_key)?;
        ensure_cloned(&mut modified, &weights.tensors, &down_key)?;

        let write: Vec<f32> = match down_meta {
            Some(dm) => {
                let tid = dm.top_token_id as usize;
                if tid >= weights.embed.shape()[0] {
                    eprintln!(
                        "  skip: insert at L{}[{}] target token {} out of vocab",
                        layer, feature, tid
                    );
                    n_insert_skipped += 1;
                    continue;
                }
                weights.embed.row(tid).to_vec()
            }
            None => {
                eprintln!(
                    "  skip: insert at L{}[{}] has no down_meta target",
                    layer, feature
                );
                n_insert_skipped += 1;
                continue;
            }
        };

        let stats = install_edge(
            &mut modified,
            &gate_key,
            &up_key,
            &down_key,
            *feature,
            &gate_vec,
            &write,
            args.gate_scale,
            args.alpha,
        )?;

        n_inserted += 1;
        eprintln!(
            "  compiled: L{}[{}] {} → {} (gate ‖{:.3}‖, down ‖{:.3}‖)",
            layer, feature, entity, target, stats.g_norm, stats.d_norm
        );
    }

    for op in &plan.deletes {
        let PatchOp::Delete {
            layer,
            feature,
            reason,
        } = op
        else {
            unreachable!("plan_ops only puts Delete here");
        };

        let gate_key = gate_pattern.replace("{}", &layer.to_string());
        let up_key = up_pattern.replace("{}", &layer.to_string());
        let down_key = down_pattern.replace("{}", &layer.to_string());

        ensure_cloned(&mut modified, &weights.tensors, &gate_key)?;
        ensure_cloned(&mut modified, &weights.tensors, &up_key)?;
        ensure_cloned(&mut modified, &weights.tensors, &down_key)?;

        let stats = tombstone_slot(&mut modified, &gate_key, &up_key, &down_key, *feature)?;
        n_deleted += 1;
        if !stats.was_live {
            n_delete_already_dead += 1;
        }
        eprintln!(
            "  tombstoned: L{}[{}]{} (gate ‖{:.3}‖, up ‖{:.3}‖, down ‖{:.3}‖ → 0){}",
            layer,
            feature,
            reason
                .as_deref()
                .map(|r| format!(" — {}", r))
                .unwrap_or_default(),
            stats.g_norm_before,
            stats.u_norm_before,
            stats.d_norm_before,
            if stats.was_live {
                ""
            } else {
                "  ⚠ slot was already dead; this delete changed nothing here"
            }
        );
    }

    eprintln!(
        "\n  {} edges compiled, {} slots tombstoned ({} were already dead), {} inserts NOT placed",
        n_inserted, n_deleted, n_delete_already_dead, n_insert_skipped
    );

    if n_insert_skipped > 0 && !args.allow_partial {
        return Err(format!(
            "refusing to write: {} of {} insert ops could not be placed (see `skip:` lines above). \
             The checkpoint would carry fewer edits than its patch set. \
             Pass --allow-partial to write it anyway; the summary will still state the gap.",
            n_insert_skipped,
            plan.inserts.len()
        )
        .into());
    }
    if n_inserted + n_deleted == 0 {
        return Err("refusing to write: no op landed in the weights, so the output would be \
                    the base model under a new name."
            .into());
    }

    eprintln!("\nSaving compiled model...");
    std::fs::create_dir_all(&args.output)?;
    let merged = merge_for_save(&weights, modified);
    let output_file = args.output.join("model.safetensors");
    write_safetensors(&merged.tensors, &merged.vectors, &output_file)?;

    let file_size = std::fs::metadata(&output_file)?.len();
    eprintln!(
        "  saved: {} ({:.1} GB, {} tensors, {} vectors)",
        output_file.display(),
        file_size as f64 / 1e9,
        merged.tensors.len(),
        merged.vectors.len(),
    );
    if n_insert_skipped > 0 {
        eprintln!(
            "  ⚠ PARTIAL: {} insert op(s) are NOT in this checkpoint (--allow-partial).",
            n_insert_skipped
        );
    }

    copy_model_config(&args.base, &args.output);

    eprintln!("\nDone. The compiled model runs in any inference engine:");
    eprintln!(
        "  transformers: AutoModelForCausalLM.from_pretrained(\"{}\")",
        args.output.display()
    );
    eprintln!("  ollama:       convert to GGUF, then `ollama create`");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn insert(layer: usize, feature: usize) -> PatchOp {
        PatchOp::Insert {
            layer,
            feature,
            relation: None,
            entity: "e".into(),
            target: "t".into(),
            confidence: None,
            gate_vector_b64: None,
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: None,
        }
    }

    fn delete(layer: usize, feature: usize) -> PatchOp {
        PatchOp::Delete {
            layer,
            feature,
            reason: None,
        }
    }

    fn insert_knn() -> PatchOp {
        PatchOp::InsertKnn {
            layer: 25,
            entity: "e".into(),
            relation: "r".into(),
            target: "t".into(),
            target_id: 1,
            confidence: None,
            key_vector_b64: String::new(),
        }
    }

    #[test]
    fn a_delete_only_patch_set_is_planned_as_deletes_never_refused() {
        // The exact shape that used to compile to a base-identical checkpoint.
        let ops = vec![delete(14, 2113), delete(25, 9757), delete(25, 12)];
        let plan = plan_ops(&ops);
        assert_eq!(plan.deletes.len(), 3);
        assert!(plan.inserts.is_empty());
        assert!(plan.refused.is_empty(), "a delete has a compile path and must never be refused");
    }

    #[test]
    fn knn_ops_are_refused_by_name_and_counted() {
        let ops = vec![
            insert_knn(),
            insert_knn(),
            PatchOp::DeleteKnn { entity: "e".into() },
            insert(1, 1),
        ];
        let plan = plan_ops(&ops);
        assert_eq!(plan.refused.get("insert_knn"), Some(&2));
        assert_eq!(plan.refused.get("delete_knn"), Some(&1));
        assert_eq!(plan.inserts.len(), 1);
        let msg = refusal_message(&plan.refused);
        assert!(msg.contains("2 × `insert_knn`"), "{msg}");
        assert!(msg.contains("1 × `delete_knn`"), "{msg}");
        assert!(msg.contains("Nothing was written"), "{msg}");
    }

    #[test]
    fn update_is_refused_until_it_has_a_path() {
        let ops = vec![PatchOp::Update {
            layer: 3,
            feature: 4,
            gate_vector_b64: None,
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: None,
        }];
        let plan = plan_ops(&ops);
        assert_eq!(plan.refused.get("update"), Some(&1));
    }

    #[test]
    fn every_op_lands_in_exactly_one_bucket() {
        let ops = vec![
            insert(1, 1),
            delete(2, 2),
            insert_knn(),
            PatchOp::DeleteKnn { entity: "e".into() },
            PatchOp::Update {
                layer: 3,
                feature: 4,
                gate_vector_b64: None,
                up_vector_b64: None,
                down_vector_b64: None,
                down_meta: None,
            },
        ];
        let plan = plan_ops(&ops);
        let total = plan.inserts.len() + plan.deletes.len() + plan.refused.values().sum::<usize>();
        assert_eq!(total, ops.len(), "an op that lands nowhere is the silent-drop bug");
    }
}
