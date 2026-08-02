//! Gemma one-layer parity — VINDEX3 bound over VINDEX2's own expert bytes.
//!
//! The first real-model step. It holds *everything* constant except the
//! execution path:
//!
//! ```text
//! one VINDEX2 index on disk
//!         ↓
//! the same Q4_K expert bytes, the same f32 router, the same activation
//!    ↓                                    ↓
//! incumbent MoE path              BoundMoeOperation bound over those bytes
//!    ↓                                    ↓
//!         compare, checkpoint by checkpoint
//! ```
//!
//! # Why not extract a VINDEX3 container first
//!
//! Because then a mismatch would have two candidate causes — the executor, or
//! the re-extraction — and telling those apart is the entire point. Binding
//! over the incumbent's own bytes makes any divergence unambiguously about
//! execution: routing policy, region interpretation, activation, or reduction.
//!
//! # What is deliberately *not* claimed here
//!
//! Bit-exactness. Two known numerical differences are structural, not faults:
//!
//! - The incumbent scores its router with BLAS `sgemv`; the reference sums in
//!   index order. Different summation order, same arithmetic.
//! - Gemma's experts are Q4_K. The incumbent's fast path quantises the
//!   activation to Q8_K and does integer dot products; this reference
//!   dequantises to f32. Binding Q4_K regions to the Q4_K kernel is kernel
//!   binding, which is a later rung.
//!
//! So the assertions are graded: **selection must match exactly** (it is a
//! discrete decision, and a difference there is a real disagreement), while
//! values are compared against a stated tolerance and reported, not asserted
//! into agreement.
//!
//! # Capturing the activation
//!
//! ```text
//! LARQL_CPU_DUMP_LAYERS=/tmp/gemma_dump \
//!   larql run <vindex> "The capital of France is" -n 1
//! ```
//!
//! writes `cpu_layer_NN_h_post_attn.f32` — the real residual entering each
//! layer's FFN/MoE block.
//!
//! Usage:
//! ```text
//! cargo run --release -p larql-vindex --example vindex3_gemma_layer_parity -- \
//!   --vindex <path> --dump /tmp/gemma_dump [--layer 5]
//! ```

use larql_compute::cpu::ops::moe::{
    moe_expert_input, moe_route_from_router_input, moe_router_input,
};
use larql_compute::cpu::ops::q4_common::dequantize_q4_k;
use larql_compute::pipeline_layer::build_moe_weights;
use larql_compute::MoeLayerWeights;

use larql_vindex::format::capability::binding::{ComponentView, RepresentationIdentity};
use larql_vindex::format::capability::component::ComponentContract;
use larql_vindex::format::capability::coordinate::BankCoordinate;
use larql_vindex::format::lyrw2::region_format::RegionFormat;
use larql_vindex::format::lyrw2::region_role::RegionRole;
use larql_vindex::runtime::{
    execute_traced, BoundBankOperation, BoundExpert, BoundMoeOperation, BoundProjection,
    BoundReduction, BoundRouter, BoundTensor, MoeInputs,
};

/// Tolerance for value comparisons. Not a pass/fail gate — the two paths use
/// different summation orders and different expert kernels, so this is the
/// band within which "the same computation" is the honest reading.
const VALUE_TOLERANCE: f32 = 1e-3;
/// Selection is discrete. Any difference is a real disagreement.
const DEFAULT_LAYER: usize = 5;
const VARIANT: &str = "vindex2-bytes";
const ROUTER_REGION_SET: &str = "router";

fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

/// Read an f32 dump and return its final row — the token being decoded.
fn last_row(path: &str, width: usize) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    if !values.len().is_multiple_of(width) || values.is_empty() {
        return Err(format!(
            "{path}: {} values is not a whole number of {width}-wide rows",
            values.len()
        ));
    }
    let rows = values.len() / width;
    Ok(values[(rows - 1) * width..].to_vec())
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn report(label: &str, a: &[f32], b: &[f32]) -> bool {
    if a.len() != b.len() {
        println!("  {label:<22} LENGTH {} vs {}", a.len(), b.len());
        return false;
    }
    let diff = max_abs_diff(a, b);
    let ok = diff <= VALUE_TOLERANCE;
    println!(
        "  {label:<22} max|Δ| = {diff:.3e}  {}",
        if ok { "within tolerance" } else { "OUTSIDE" }
    );
    ok
}

/// Bind one expert's Q4_K bytes as dequantised f32.
///
/// Materialising rather than binding Q4_K directly: the reference decoder
/// implements the directly-readable encodings only, and a quantised region is
/// a missing kernel rather than bad bytes. The incumbent's own fallback path
/// dequantises the same way, which is what makes the two comparable at all.
fn dequantised(
    role: RegionRole,
    bytes: &[u8],
    rows: usize,
    cols: usize,
) -> Result<(Vec<u8>, ComponentContract), String> {
    let values = dequantize_q4_k(bytes, rows * cols);
    if values.len() < rows * cols {
        return Err(format!(
            "{}: dequantised {} of {} expected elements",
            role.name(),
            values.len(),
            rows * cols
        ));
    }
    let raw: Vec<u8> = values[..rows * cols]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    Ok((raw, ComponentContract::matrix(rows as u32, cols as u32)))
}

fn tensor<'a>(
    region_set: &str,
    bytes: &'a [u8],
    contract: ComponentContract,
) -> Result<BoundTensor<'a>, String> {
    BoundTensor::direct(
        RepresentationIdentity::new(region_set, VARIANT),
        bytes,
        RegionFormat::F32,
        contract,
    )
    .map_err(|e| e.to_string())
}

/// Bind a stored matrix whose trailing columns are quantisation padding.
///
/// The role sees `[rows, keep]`; the bytes remain `[rows, stored_cols]`. No
/// repacking, no copy — the view resolves the difference at read time.
fn sliced_tensor<'a>(
    region_set: &str,
    bytes: &'a [u8],
    storage: ComponentContract,
    keep: usize,
) -> Result<BoundTensor<'a>, String> {
    BoundTensor::new(
        RepresentationIdentity::new(region_set, VARIANT),
        bytes,
        RegionFormat::F32,
        storage,
        ComponentView::Slice {
            dim: 1,
            start: 0,
            len: keep as u32,
        },
    )
    .map_err(|e| e.to_string())
}

/// Owns the dequantised expert buffers so bound tensors can borrow them.
struct ExpertBuffers {
    expert_id: u32,
    gate_up: Vec<u8>,
    down: Vec<u8>,
    gate_up_contract: ComponentContract,
    down_contract: ComponentContract,
}

fn main() -> Result<(), String> {
    let vindex = arg("--vindex").ok_or("set --vindex <path>")?;
    let dump = arg("--dump").ok_or("set --dump <dir> (LARQL_CPU_DUMP_LAYERS output)")?;
    let layer: usize = arg("--layer")
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_LAYER);

    println!("Gemma one-layer parity — VINDEX2 bytes, two execution paths");
    println!("  vindex  {vindex}");
    println!("  layer   {layer}");

    // ── Load the incumbent's weights ───────────────────────────────────────
    let mut callbacks = larql_vindex::SilentLoadCallbacks;
    let weights =
        larql_vindex::load_model_weights_kquant(std::path::Path::new(&vindex), &mut callbacks)
            .map_err(|e| format!("load weights: {e}"))?;
    let arch = &*weights.arch;
    let hidden = weights.hidden_size;
    let norm_offset = arch.norm_weight_offset();
    let eps = arch.norm_eps();

    let moe: MoeLayerWeights<'_> = build_moe_weights(&weights, arch, layer)
        .ok_or_else(|| format!("layer {layer} is not an MoE layer"))?;
    println!(
        "  shape   hidden {hidden}, {} experts, top-{}, intermediate {}",
        moe.num_experts, moe.top_k, moe.intermediate_size
    );

    // ── The real activation ────────────────────────────────────────────────
    let h = last_row(
        &larql_compute::forward::dump_config::cpu_layer_h_post_attn_path(&dump, layer),
        hidden,
    )?;
    println!("  input   real h_post_attn, last token of the prompt");

    // ── Incumbent: the two inputs and the routing decision ─────────────────
    let expert_input = moe_expert_input(&h, &moe, norm_offset, eps);
    let router_in = moe_router_input(&h, &expert_input, &moe, norm_offset, eps);
    let (incumbent_ids, incumbent_weights) = moe_route_from_router_input(&router_in, &moe);
    println!(
        "\n  incumbent routes on a {} vector",
        if expert_input == router_in {
            "shared"
        } else {
            "separate"
        }
    );

    // ── Bind the selected experts' bytes as a VINDEX3 operation ────────────
    //
    // Only the selected experts: a bank holding a subset of the population is
    // a legitimate shard, and dequantising all of them would cost gigabytes to
    // no purpose. If the VINDEX3 router disagrees about the selection it will
    // ask for an expert that is not bound and fail loudly, which is the right
    // failure.
    let inter = moe.intermediate_size;
    let mut buffers = Vec::new();
    for &e in &incumbent_ids {
        let gate_up_bytes = moe
            .experts_gate_up
            .get(e)
            .ok_or_else(|| format!("expert {e} has no gate_up bytes"))?;
        let down_bytes = moe
            .experts_down
            .get(e)
            .ok_or_else(|| format!("expert {e} has no down bytes"))?;
        let (gate_up, gate_up_contract) =
            dequantised(RegionRole::GateUpFused, gate_up_bytes, 2 * inter, hidden)?;
        // `down` is stored at the *padded* intermediate width: Q4_K rounds 704
        // up to the next 256-multiple (768), while `gate_up` is unpadded
        // because `hidden` is already a multiple. So the two regions disagree
        // about the intermediate axis, and the padding columns are inert.
        //
        // This is precisely what a slice view is for: bind the stored
        // [hidden, 768] and let the role see [hidden, 704]. The incumbent
        // reaches the same place by zero-padding the activation instead.
        let (down, down_contract) =
            dequantised(RegionRole::Down, down_bytes, hidden, moe.inter_padded())?;
        buffers.push(ExpertBuffers {
            expert_id: e as u32,
            gate_up,
            down,
            gate_up_contract,
            down_contract,
        });
    }

    let router_bytes: Vec<u8> = moe
        .router_proj
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let experts: Vec<BoundExpert<'_>> = buffers
        .iter()
        .map(|b| -> Result<BoundExpert<'_>, String> {
            Ok(BoundExpert {
                expert_id: b.expert_id,
                projection: BoundProjection::Fused {
                    gate_up: tensor(
                        &RegionRole::GateUpFused.name(),
                        &b.gate_up,
                        b.gate_up_contract.clone(),
                    )?,
                },
                down: sliced_tensor(
                    &RegionRole::Down.name(),
                    &b.down,
                    b.down_contract.clone(),
                    inter,
                )?,
            })
        })
        .collect::<Result<_, _>>()?;

    let operation = BoundMoeOperation {
        router: BoundRouter {
            weight: tensor(
                ROUTER_REGION_SET,
                &router_bytes,
                ComponentContract::matrix(moe.num_experts as u32, hidden as u32),
            )?,
            top_k: moe.top_k,
            selected_weight: moe.routing_policy.selected_weight,
            expert_scale: moe.routing_policy.expert_scale,
            per_expert_scale: None,
        },
        transforms: Vec::new(),
        banks: vec![BoundBankOperation {
            bank: BankCoordinate::new(layer as u32, 0),
            experts,
            intermediate_dim: inter,
            hidden_dim: hidden,
            activation: moe.activation,
        }],
        reduction: BoundReduction::WeightedSum,
        residual_dim: hidden,
    };
    operation.validate().map_err(|e| format!("bind: {e}"))?;
    println!("  bound   {}", operation.describe());
    println!(
        "  shard   holds {} of {} experts (full population: {})",
        operation.banks[0].population(),
        operation.router.population(),
        operation.holds_full_population()
    );

    // ── Execute the VINDEX3 route on the identical inputs ──────────────────
    let (_, trace) = execute_traced(&operation, MoeInputs::split(&expert_input, &router_in))
        .map_err(|e| format!("execute: {e}"))?;

    // ── Compare ────────────────────────────────────────────────────────────
    println!("\n== selection (exact match required) ==");
    let vindex3_ids: Vec<usize> = trace.selected_ids().iter().map(|i| *i as usize).collect();
    println!("  incumbent  {incumbent_ids:?}");
    println!("  vindex3    {vindex3_ids:?}");
    let selection_matches = incumbent_ids == vindex3_ids;
    println!(
        "  {}",
        if selection_matches {
            "PASS identical experts, identical order"
        } else {
            "FAIL the two paths route differently"
        }
    );
    if let Some(margin) = trace.selection_margin {
        println!(
            "  margin     {margin:.6}{}",
            if margin == 0.0 {
                "  (an exact tie decided the boundary)"
            } else {
                ""
            }
        );
    }

    println!("\n== values (tolerance {VALUE_TOLERANCE:.0e}) ==");
    let weights_match = report("gate weights", &incumbent_weights, &trace.gate_weights());

    println!("\n== notes ==");
    println!("  Router scoring differs in summation order (BLAS sgemv vs index order).");
    println!("  Expert values differ by kernel: incumbent Q4_K x Q8_K integer dot,");
    println!("  reference dequantised f32. Kernel binding is a later rung.");

    if selection_matches && weights_match {
        println!("\nPARITY: selection identical, routing weights within tolerance.");
        Ok(())
    } else {
        Err("parity not established — see above".into())
    }
}
