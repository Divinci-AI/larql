//! The VINDEX3 arm of `INSERT … MODE COMPOSE` (V3-LQL-3B compose).
//!
//! Same install formula as the V2 arm in `compose.rs`
//! (`install_compiled_slot`, validated on Gemma 3 4B):
//!
//! ```text
//! gate[slot]   = unit(residual) · g_ref · GATE_SCALE
//! up[slot]     = unit(residual) · u_ref
//! down[:,slot] = unit(target_embed) · d_ref · alpha_mul
//! ```
//!
//! with the layer-median reference norms sampled the same way (first
//! `min(n, 100)` features, L2, median, 1.0 fallback) — but every input
//! resolved through V3's own authorities: the residual from the plan's
//! execution taps, the target embedding from role `embedding`, the
//! norms from the plan's FFN operands via the runtime's resolver, and
//! the written vectors landing in the [`KnowledgeOverlay`], where
//! browse merges them into its scan and execution observes them
//! through the operand-source seam.
//!
//! Deliberately NOT ported this rung: the batch-refine pass, decoy
//! suppression, and cross-fact balance (`compose.rs` phases 2b/3).
//! Those are multi-fact interference science validated per-model; the
//! install here is the single-fact formula, and the report says so.
//!
//! [`KnowledgeOverlay`]: larql_vindex::format::vindex3::knowledge::KnowledgeOverlay

use larql_inference::vindex3::Vindex3Runtime;
use larql_vindex::format::vindex3::opplan::exec::production::ProductionBackend;
use larql_vindex::format::vindex3::opplan::LayerFfn;

use super::compose::{median_or, unit_vector};
use super::knn::knn_canonical_prompt;
use super::DEFAULT_INSERT_CONFIDENCE;
use crate::error::LqlError;
use crate::executor::tuning::{DEFAULT_INSERT_ALPHA_MUL, GATE_SCALE};
use crate::executor::{Backend, Session};

/// How many features the layer-median norm statistic samples — the V2
/// arm's `compute_layer_median_norms(_, _, 100)`.
const NORM_SAMPLE_SIZE: usize = 100;

/// The three layer-typical norms the install matches (g_ref / u_ref /
/// d_ref), from the plan's own FFN operands.
fn layer_median_norms(
    runtime: &Vindex3Runtime<ProductionBackend>,
    layer: usize,
) -> Result<(f32, f32, f32), LqlError> {
    let LayerFfn::Dense(ffn) = &runtime.plan().layers[layer].ffn else {
        return Err(LqlError::Execution(format!(
            "layer {layer} is routed — compose installs on MoE layers are a later role rung"
        )));
    };
    let operands = runtime.operands();
    let load = |op| {
        operands
            .load(op)
            .map_err(|e| LqlError::exec("failed to load FFN operand", e))
    };
    let gate_ref = ffn.gate.as_ref().unwrap_or(&ffn.up);
    let gate = load(gate_ref)?;
    let up = load(&ffn.up)?;
    let down = load(&ffn.down)?;

    let features = gate_ref.shape[0];
    let hidden = gate_ref.shape[1];
    let sample = features.min(NORM_SAMPLE_SIZE);
    let row_norms = |matrix: &[f32]| -> Vec<f32> {
        (0..sample)
            .filter_map(|i| {
                let row = &matrix[i * hidden..(i + 1) * hidden];
                let n: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
                (n.is_finite() && n > 0.0).then_some(n)
            })
            .collect()
    };
    let mut gate_norms = row_norms(&gate);
    let mut up_norms = row_norms(&up);
    // down is `[hidden, features]`: feature f's payload is column f.
    let down_cols = ffn.down.shape[1];
    let mut down_norms: Vec<f32> = (0..sample.min(down_cols))
        .filter_map(|f| {
            let n: f32 = (0..hidden)
                .map(|r| down[r * down_cols + f])
                .map(|v| v * v)
                .sum::<f32>()
                .sqrt();
            (n.is_finite() && n > 0.0).then_some(n)
        })
        .collect();

    Ok((
        median_or(&mut gate_norms, 1.0),
        median_or(&mut up_norms, 1.0),
        median_or(&mut down_norms, 1.0),
    ))
}

impl Session {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn exec_insert_compose_v3(
        &mut self,
        entity: &str,
        relation: &str,
        target: &str,
        layer_hint: Option<u32>,
        confidence: Option<f32>,
        alpha_override: Option<f32>,
    ) -> Result<Vec<String>, LqlError> {
        let alpha_mul = alpha_override.unwrap_or(DEFAULT_INSERT_ALPHA_MUL);
        let c_score = confidence.unwrap_or(DEFAULT_INSERT_CONFIDENCE);

        let (layer, feature, target_id, gate_vec, up_vec, down_vec);
        {
            let Backend::Vindex3 {
                runtime,
                tokenizer,
                knowledge,
                overlay,
                ..
            } = &self.backend
            else {
                unreachable!("caller matched the backend");
            };
            let tokenizer = tokenizer.as_ref().ok_or_else(|| {
                LqlError::Execution(
                    "INSERT needs a tokenizer (the canonical prompt and the target must \
                     tokenize) and this container carries no tokenizer.json"
                        .into(),
                )
            })?;
            let view = knowledge.as_ref().ok_or_else(|| {
                LqlError::Execution(
                    "a compose install needs the browse view (free-slot search reads the \
                     feature space) and this container carries no tokenizer.json"
                        .into(),
                )
            })?;

            // ── Plan: layer, slot, target embedding ──
            let num_layers = runtime.plan().layers.len();
            layer = match layer_hint {
                Some(l) => (l as usize).min(num_layers.saturating_sub(1)),
                None => num_layers.saturating_sub(2),
            };
            feature = overlay.find_free_feature(view, layer).ok_or_else(|| {
                LqlError::Execution(format!("no free feature slot at layer {layer}"))
            })?;

            let spaced_target = format!(" {target}");
            let target_encoding = tokenizer
                .encode(spaced_target.as_str(), false)
                .map_err(|e| LqlError::exec("tokenize error", e))?;
            target_id = target_encoding.get_ids().first().copied().unwrap_or(0);
            let (embed, embed_scale) = view.embedding();
            let row = embed.row((target_id as usize).min(embed.shape()[0].saturating_sub(1)));
            let target_embed: Vec<f32> = row.iter().map(|v| v * embed_scale).collect();

            // ── Capture: the canonical prompt's residual, from the
            // plan's own taps ──
            let prompt = knn_canonical_prompt(entity, relation);
            let prior = crate::executor::vindex3::compose_overrides(runtime, overlay)?;
            let residual = crate::executor::vindex3::capture_layer_residual(
                runtime,
                tokenizer,
                prompt.as_str(),
                layer,
                prior.as_ref(),
            )?;
            let gate_dir = unit_vector(&residual);

            // ── Synthesis: the validated install formula ──
            let (g_ref, u_ref, d_ref) = layer_median_norms(runtime, layer)?;
            gate_vec = gate_dir
                .iter()
                .map(|v| v * g_ref * GATE_SCALE)
                .collect::<Vec<f32>>();
            up_vec = gate_dir.iter().map(|v| v * u_ref).collect::<Vec<f32>>();
            let target_unit = unit_vector(&target_embed);
            down_vec = target_unit
                .iter()
                .map(|v| v * d_ref * alpha_mul)
                .collect::<Vec<f32>>();
        }

        // ── Install: overlay state browse merges and execution
        // observes through the operand-source seam ──
        let meta = larql_vindex::FeatureMeta {
            top_token: target.to_string(),
            top_token_id: target_id,
            c_score,
            top_k: vec![larql_models::TopKEntry {
                token: target.to_string(),
                token_id: target_id,
                logit: c_score,
            }],
        };
        let Backend::Vindex3 { overlay, .. } = &mut self.backend else {
            unreachable!("caller matched the backend");
        };
        overlay.insert_feature(layer, feature, gate_vec.clone(), meta);
        overlay.set_up_vector(layer, feature, up_vec.clone());
        overlay.set_down_vector(layer, feature, down_vec.clone());

        if let Some(ref mut recording) = self.patch_recording {
            let b64 = larql_vindex::patch::core::encode_gate_vector;
            recording.operations.push(larql_vindex::PatchOp::Insert {
                layer,
                feature,
                relation: Some(relation.to_string()),
                entity: entity.to_string(),
                target: target.to_string(),
                confidence: Some(c_score),
                gate_vector_b64: Some(b64(&gate_vec)),
                up_vector_b64: Some(b64(&up_vec)),
                down_vector_b64: Some(b64(&down_vec)),
                down_meta: Some(larql_vindex::patch::core::PatchDownMeta {
                    top_token: target.to_string(),
                    top_token_id: target_id,
                    c_score,
                }),
            });
        }

        Ok(vec![
            format!(
                "Inserted: {} —[{}]→ {} at L{}/F{} (compose overlay)",
                entity, relation, target, layer, feature,
            ),
            "  mode: COMPOSE — FFN slot install (VINDEX3 operand-source seam); \
             single-fact formula, no refine/balance pass yet"
                .into(),
            format!("  alpha {alpha_mul}, gate scale {GATE_SCALE}"),
        ])
    }
}
