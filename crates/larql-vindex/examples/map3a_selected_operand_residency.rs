//! MAP-3A: selected-operand residency (necessity + sufficiency).
//!
//! Registered as chuk-experiments programme `map`, experiment `map-3a`.
//! Built on branch `worktree-vindex2` — deliberately not merged into
//! `feat/dec-funnel-v0-4` to run this; that's a separate integration
//! decision.
//!
//! MAP-3's original design assumed literal OS-level page eviction would
//! cause execution to fail. It can't — a locally-backed `mmap`'s pages,
//! once dropped, transparently re-fault from disk on the next read, so
//! no executor built on `mmap` can observe or refuse based on that kind
//! of eviction. The real, working analogue this experiment tests instead
//! is **logical operand residency**: whether a bound execution plan's
//! expert list contains the routed expert's bytes, independent of where
//! those bytes physically live. `ExecutionError::SelectedExpertNotResident`
//! is the real, already-tested mechanism for this (`runtime/tests/placement.rs`);
//! this file exercises it against a REAL Gemma 4 26B-A4B MoE layer and a
//! REAL captured routing decision, not a synthetic 4-expert fixture.
//!
//! Acceptance ladder (each rung must pass for the experiment to close positive):
//!
//! ```text
//! 1. full population (all 128 experts bound)  }  bit-identical
//!    minimal population (only the 8 routed)   }  to each other  <- SUFFICIENCY
//! 2. one routed expert removed from the minimal population
//!    -> Err(SelectedExpertNotResident{expert, resident=7, population=128})
//!    no fallback, no numerical output                            <- NECESSITY
//! 3. same removed expert, padded back to 8 entries with an
//!    UNSELECTED substitute (byte capacity restored, correct
//!    operand still absent)
//!    -> Err(SelectedExpertNotResident{expert, resident=8, population=128})
//!                                                    <- CAPACITY IS NOT RESIDENCY
//! 4. exact missing expert restored -> bit-identical to rung 1's minimal output
//!                                                    <- closes the loop
//! ```
//!
//! Usage:
//! ```text
//! LARQL_CPU_DUMP_LAYERS=/tmp/dump larql run <vindex> "prompt" -n 1
//! cargo run --release -p larql-vindex --example map3a_selected_operand_residency -- \
//!   --vindex <path> --dump /tmp/dump [--layer 5]
//! ```

use larql_compute::cpu::ops::moe::{moe_expert_input, moe_route_from_router_input, moe_router_input};
use larql_compute::pipeline_layer::build_moe_weights;
use larql_compute::MoeLayerWeights;

use larql_vindex::format::capability::binding::{ComponentView, RepresentationIdentity};
use larql_vindex::format::capability::component::ComponentContract;
use larql_vindex::format::capability::coordinate::BankCoordinate;
use larql_vindex::format::lyrw2::region_format::RegionFormat;
use larql_vindex::format::lyrw2::region_role::RegionRole;
use larql_vindex::runtime::consts::{COL_DIM, FUSED_PROJECTION_HALVES};
use larql_vindex::runtime::{
    execute, BoundBankOperation, BoundExpert, BoundExpertScaling, BoundMoeOperation,
    BoundProjection, BoundReduction, BoundRouter, BoundTensor, ExecutionError, ExpertKernel,
    MoeInputs, RouterKernel,
};

const VARIANT: &str = "vindex2-bytes";
const ROUTER_REGION_SET: &str = "router";
const PER_EXPERT_SCALE_REGION_SET: &str = "router_per_expert_scale";
const BANK_ID: u16 = 0;
const DEFAULT_LAYER: usize = 5;

fn arg(name: &str) -> Option<String> {
    std::env::args()
        .collect::<Vec<_>>()
        .iter()
        .position(|a| a == name)
        .and_then(|i| std::env::args().nth(i + 1))
}

fn last_row(path: &str, width: usize) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    if !values.len().is_multiple_of(width) || values.is_empty() {
        return Err(format!("{path}: {} values is not a whole number of {width}-wide rows", values.len()));
    }
    let rows = values.len() / width;
    Ok(values[(rows - 1) * width..].to_vec())
}

fn tensor<'a>(region_set: &str, bytes: &'a [u8], format: RegionFormat, contract: ComponentContract) -> Result<BoundTensor<'a>, String> {
    BoundTensor::direct(RepresentationIdentity::new(region_set, VARIANT), bytes, format, contract).map_err(|e| e.to_string())
}

/// Bind a stored matrix whose trailing columns are quantisation padding —
/// same technique as `vindex3_gemma_layer_parity.rs`.
fn sliced_tensor<'a>(region_set: &str, bytes: &'a [u8], format: RegionFormat, storage: ComponentContract, keep: usize) -> Result<BoundTensor<'a>, String> {
    BoundTensor::new(
        RepresentationIdentity::new(region_set, VARIANT),
        bytes,
        format,
        storage,
        ComponentView::Slice { dim: COL_DIM, start: 0, len: keep as u32 },
    )
    .map_err(|e| e.to_string())
}

/// Bind one real expert's Q4_K bytes straight from the vindex's own mmap —
/// no dequantisation, no copy. Cheap enough to call for all 128 experts.
fn build_expert<'a>(moe: &MoeLayerWeights<'a>, hidden: usize, inter: usize, inter_padded: usize, e: usize) -> Result<BoundExpert<'a>, String> {
    let gate_up_bytes = *moe.experts_gate_up.get(e).ok_or_else(|| format!("expert {e} has no gate_up bytes"))?;
    let down_bytes = *moe.experts_down.get(e).ok_or_else(|| format!("expert {e} has no down bytes"))?;
    Ok(BoundExpert {
        expert_id: e as u32,
        projection: BoundProjection::Fused {
            gate_up: tensor(&RegionRole::GateUpFused.name(), gate_up_bytes, RegionFormat::Q4K, ComponentContract::matrix((FUSED_PROJECTION_HALVES * inter) as u32, hidden as u32))?,
        },
        down: sliced_tensor(&RegionRole::Down.name(), down_bytes, RegionFormat::Q4K, ComponentContract::matrix(hidden as u32, inter_padded as u32), inter)?,
    })
}

fn bind_operation<'a>(layer: usize, hidden: usize, moe: &MoeLayerWeights<'_>, router_bytes: &'a [u8], scale_bytes: &'a [u8], experts: Vec<BoundExpert<'a>>) -> Result<BoundMoeOperation<'a>, String> {
    Ok(BoundMoeOperation {
        router: BoundRouter {
            weight: tensor(ROUTER_REGION_SET, router_bytes, RegionFormat::F32, ComponentContract::matrix(moe.num_experts as u32, hidden as u32))?,
            top_k: moe.top_k,
            selected_weight: moe.routing_policy.selected_weight,
            scaling: if scale_bytes.is_empty() {
                BoundExpertScaling::None
            } else {
                BoundExpertScaling::PerExpert { scales: tensor(PER_EXPERT_SCALE_REGION_SET, scale_bytes, RegionFormat::F32, ComponentContract::vector(moe.num_experts as u32))? }
            },
            kernel: RouterKernel::Incumbent,
        },
        transforms: Vec::new(),
        banks: vec![BoundBankOperation {
            bank: BankCoordinate::new(layer as u32, BANK_ID),
            experts,
            intermediate_dim: moe.intermediate_size,
            hidden_dim: hidden,
            activation: moe.activation,
            kernel: ExpertKernel::IncumbentQ4kQ8k,
        }],
        reduction: BoundReduction::WeightedSum,
        residual_dim: hidden,
    })
}

fn bank_of<'a>(layer: usize, hidden: usize, moe: &MoeLayerWeights<'a>, router_bytes: &'a [u8], scale_bytes: &'a [u8], inter: usize, inter_padded: usize, ids: &[usize]) -> Result<BoundMoeOperation<'a>, String> {
    let experts = ids.iter().map(|&e| build_expert(moe, hidden, inter, inter_padded, e)).collect::<Result<_, _>>()?;
    bind_operation(layer, hidden, moe, router_bytes, scale_bytes, experts)
}

fn main() -> Result<(), String> {
    let vindex = arg("--vindex").ok_or("set --vindex <path>")?;
    let dump = arg("--dump").ok_or("set --dump <dir> (LARQL_CPU_DUMP_LAYERS output)")?;
    let layer: usize = arg("--layer").and_then(|v| v.parse().ok()).unwrap_or(DEFAULT_LAYER);

    println!("=== MAP-3A: selected-operand residency ===");
    println!("  vindex  {vindex}");
    println!("  layer   {layer}");

    let mut callbacks = larql_vindex::SilentLoadCallbacks;
    let weights = larql_vindex::load_model_weights_kquant(std::path::Path::new(&vindex), &mut callbacks).map_err(|e| format!("load weights: {e}"))?;
    let arch = &*weights.arch;
    let hidden = weights.hidden_size;
    let norm_offset = arch.norm_weight_offset();
    let eps = arch.norm_eps();
    let moe: MoeLayerWeights<'_> = build_moe_weights(&weights, arch, layer).ok_or_else(|| format!("layer {layer} is not an MoE layer"))?;
    let inter = moe.intermediate_size;
    let inter_padded = moe.inter_padded();
    println!("  shape   hidden {hidden}, {} experts, top-{}, intermediate {inter}", moe.num_experts, moe.top_k);

    let h = last_row(&larql_compute::forward::dump_config::cpu_layer_h_post_attn_path(&dump, layer), hidden)?;
    let expert_input = moe_expert_input(&h, &moe, norm_offset, eps);
    let router_in = moe_router_input(&h, &expert_input, &moe, norm_offset, eps);
    let (incumbent_ids, _incumbent_weights) = moe_route_from_router_input(&router_in, &moe);
    println!("  routed  {incumbent_ids:?} (real production routing decision)\n");

    let router_bytes: Vec<u8> = moe.router_proj.iter().flat_map(|v| v.to_le_bytes()).collect();
    let scale_bytes: Vec<u8> = moe.router_per_expert_scale.iter().flat_map(|v| v.to_le_bytes()).collect();
    let inputs = || MoeInputs::split(&expert_input, &router_in);

    let mut all_pass = true;

    // ── Rung 1: full population vs minimal population — SUFFICIENCY ────────
    println!("[rung 1] full (128 experts) vs minimal ({} routed experts) — must be bit-identical", incumbent_ids.len());
    let full_ids: Vec<usize> = (0..moe.num_experts).collect();
    let full_op = bank_of(layer, hidden, &moe, &router_bytes, &scale_bytes, inter, inter_padded, &full_ids)?;
    let minimal_op = bank_of(layer, hidden, &moe, &router_bytes, &scale_bytes, inter, inter_padded, &incumbent_ids)?;
    let full_out = execute(&full_op, inputs()).map_err(|e| format!("rung 1 full population should not error: {e}"))?;
    let minimal_out = execute(&minimal_op, inputs()).map_err(|e| format!("rung 1 minimal population should not error: {e}"))?;
    let rung1_pass = full_out == minimal_out;
    println!("  {}", if rung1_pass { "PASS — extra unselected experts in the bank change nothing" } else { "FAIL — full and minimal populations disagree" });
    all_pass &= rung1_pass;

    // ── Rung 2: remove one routed expert — NECESSITY ────────────────────────
    let removed = incumbent_ids[0];
    let without_one: Vec<usize> = incumbent_ids.iter().copied().filter(|&e| e != removed).collect();
    println!("\n[rung 2] remove routed expert {removed} (bank: {} of {} routed experts) — must refuse", without_one.len(), incumbent_ids.len());
    let op_without_one = bank_of(layer, hidden, &moe, &router_bytes, &scale_bytes, inter, inter_padded, &without_one)?;
    let result2 = execute(&op_without_one, inputs());
    let rung2_pass = matches!(
        &result2,
        Err(ExecutionError::SelectedExpertNotResident { expert, resident, population, .. })
            if *expert as usize == removed && *resident == without_one.len() && *population == moe.num_experts
    );
    println!("  result: {result2:?}");
    println!("  {}", if rung2_pass { "PASS — refused with the exact expert/resident/population it should" } else { "FAIL — wrong error, wrong metadata, or (worst case) a silent numeric result" });
    all_pass &= rung2_pass;

    // ── Rung 3: pad back to full byte capacity with an UNSELECTED expert ────
    let substitute = (0..moe.num_experts).find(|e| !incumbent_ids.contains(e)).ok_or("no unselected expert exists to substitute")?;
    let mut padded: Vec<usize> = without_one.clone();
    padded.push(substitute);
    println!("\n[rung 3] same removed expert {removed}, padded back to {} entries with unselected expert {substitute} — capacity restored, correct operand still absent", padded.len());
    let op_padded = bank_of(layer, hidden, &moe, &router_bytes, &scale_bytes, inter, inter_padded, &padded)?;
    let result3 = execute(&op_padded, inputs());
    let rung3_pass = matches!(
        &result3,
        Err(ExecutionError::SelectedExpertNotResident { expert, resident, population, .. })
            if *expert as usize == removed && *resident == padded.len() && *population == moe.num_experts
    );
    println!("  result: {result3:?}");
    println!("  {}", if rung3_pass { "PASS — capacity is not residency: still refused for the missing operand" } else { "FAIL — a byte-capacity-matched wrong substitute satisfied residency" });
    all_pass &= rung3_pass;

    // ── Rung 4: restore the exact missing expert — closes the loop ─────────
    println!("\n[rung 4] restore expert {removed} — must match rung 1's minimal output exactly");
    let op_restored = bank_of(layer, hidden, &moe, &router_bytes, &scale_bytes, inter, inter_padded, &incumbent_ids)?;
    let restored_out = execute(&op_restored, inputs()).map_err(|e| format!("rung 4 restored population should not error: {e}"))?;
    let rung4_pass = restored_out == minimal_out;
    println!("  {}", if rung4_pass { "PASS — exact parity restored" } else { "FAIL — restoring the exact operand did not restore the exact output" });
    all_pass &= rung4_pass;

    println!("\n=== verdict ===");
    if all_pass {
        println!("MAP-3A CONFIRMED: the routed expert set is necessary (rung 2), sufficient (rung 1),\nand capacity does not substitute for the specific operand (rung 3). Restoring it restores\nexact execution (rung 4).");
        Ok(())
    } else {
        Err("MAP-3A FAILED — see the rung(s) marked FAIL above".into())
    }
}
