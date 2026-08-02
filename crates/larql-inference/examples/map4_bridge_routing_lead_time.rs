//! MAP-4 bridge: does routing have usable lead time?
//!
//! Registered as chuk-experiments programme `map`, experiment `map-4`
//! (recorded in its next_action as the "MAP-3<->MAP-4 bridge").
//!
//! MAP-4 (via the independent GLA-4 spike-attribution thread) found that
//! the DANGER signal for an FFN approximation error has zero early-warning
//! lead time — a corrupted position's causal effect only becomes visible
//! at the moment of failure, not before. That's a real headwind for using
//! MAP-4's local-frame machinery to drive prefetch. But it leaves a
//! DIFFERENT question untested: does the ROUTING decision itself (which
//! experts a later MoE layer will select — the thing MAP-3A/3B showed is
//! necessary, sufficient, and the load-bearing prefetch lever) have usable
//! lead time, even if failure-danger doesn't?
//!
//! Method: for a real captured forward pass, take layer L's OWN router
//! weights and feed them an EARLIER layer's fully-computed residual (L-k,
//! k layers before L actually runs its own attention) instead of L's
//! true, current-time post-attention residual. Compare the resulting
//! top-8 expert selection against L's TRUE selection (computed from L's
//! real post-attention residual — the actual input its router consumes).
//! High overlap at large k would mean the routing decision is
//! substantially "baked in" well before layer L is reached — real,
//! prefetchable lead time. Overlap collapsing toward chance quickly would
//! mirror GLA-4's danger-signal finding: no early-warning, at least not
//! from this cheap a probe.
//!
//! This reuses ONLY real, already-loaded production functions
//! (`larql_compute::cpu::ops::moe::{moe_expert_input, moe_router_input,
//! moe_route_from_router_input}`) against REAL captured activations from
//! `LARQL_CPU_DUMP_LAYERS` dumps (the same production, MoE-aware forward
//! pass MAP-3A/3B used) — no new routing logic, no approximation of the
//! router itself, and critically, NOT an in-process `trace_forward_*` +
//! `WeightFfn` call: `WeightFfn` is dense-only and cannot process this
//! model's per-layer MoE FFN, which is why this reads real dumped
//! activations from the CLI's own MoE-aware forward pass instead.
//!
//! Usage:
//! ```text
//! for i, prompt in prompts: LARQL_CPU_DUMP_LAYERS=/tmp/dump_p{i} larql run <vindex> "prompt" -n 1
//! cargo run --release -p larql-inference --example map4_bridge_routing_lead_time -- \
//!   --vindex <path> --dump-prefix /tmp/dump_p
//! ```

use larql_compute::cpu::ops::moe::{moe_expert_input, moe_route_from_router_input, moe_router_input};
use larql_compute::forward::dump_config::{cpu_layer_h_post_attn_path, cpu_layer_path};
use larql_compute::pipeline_layer::build_moe_weights;
use larql_compute::MoeLayerWeights;

const HOME_LAYERS: [usize; 3] = [10, 20, 29];
const LEAD_TIMES: [usize; 5] = [1, 2, 4, 8, 16];
const NUM_PROMPTS: usize = 8;

fn arg(name: &str) -> Option<String> {
    std::env::args()
        .collect::<Vec<_>>()
        .iter()
        .position(|a| a == name)
        .and_then(|i| std::env::args().nth(i + 1))
}

fn last_row(path: &str, width: usize) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let values: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    assert!(values.len().is_multiple_of(width) && !values.is_empty(), "{path}: {} values not a whole number of {width}-wide rows", values.len());
    let rows = values.len() / width;
    values[(rows - 1) * width..].to_vec()
}

struct Cell {
    home_layer: usize,
    lead: usize,
    overlap: usize,
}

fn route_at(moe: &MoeLayerWeights<'_>, h: &[f32], norm_offset: f32, eps: f32) -> Vec<usize> {
    let expert_input = moe_expert_input(h, moe, norm_offset, eps);
    let router_in = moe_router_input(h, &expert_input, moe, norm_offset, eps);
    let (ids, _weights) = moe_route_from_router_input(&router_in, moe);
    ids
}

fn overlap_count(a: &[usize], b: &[usize]) -> usize {
    a.iter().filter(|x| b.contains(x)).count()
}

fn run_prompt(weights: &larql_models::ModelWeights, dump_dir: &str, hidden: usize, norm_offset: f32, eps: f32) -> Vec<Cell> {
    let arch = &*weights.arch;
    let mut cells = Vec::new();
    for &l in &HOME_LAYERS {
        let moe = match build_moe_weights(weights, arch, l) {
            Some(m) => m,
            None => continue,
        };
        // The real, current-time input layer L's own router actually
        // consumes: its post-attention residual, pre-FFN.
        let h_true = last_row(&cpu_layer_h_post_attn_path(dump_dir, l), hidden);
        let true_ids = route_at(&moe, &h_true, norm_offset, eps);
        for &k in &LEAD_TIMES {
            if k > l {
                continue;
            }
            // The earlier stand-in: layer (L-k)'s fully-computed residual
            // (post-attention + its own FFN + PLE/scalar) — the actual,
            // complete checkpoint available k layers of real compute
            // before L is reached, not a partial/inconsistent snapshot.
            let h_early = last_row(&cpu_layer_path(dump_dir, l - k), hidden);
            let pred_ids = route_at(&moe, &h_early, norm_offset, eps);
            cells.push(Cell { home_layer: l, lead: k, overlap: overlap_count(&true_ids, &pred_ids) });
        }
    }
    cells
}

fn main() {
    println!("=== MAP-4 bridge: does routing have usable lead time? ===");
    let vindex = match arg("--vindex") {
        Some(v) => v,
        None => {
            println!("skipped — set --vindex <path> (a real MoE vindex, e.g. Gemma 4 26B-A4B) and --dump-prefix <dir prefix>");
            return;
        }
    };
    let dump_prefix = arg("--dump-prefix").expect("set --dump-prefix <dir prefix> (dirs <prefix>0 .. <prefix>{N-1}, one LARQL_CPU_DUMP_LAYERS dump per prompt)");

    println!("loading real vindex (weights only, no forward pass — activations come from real dumps): {vindex}");
    let mut cb = larql_vindex::SilentLoadCallbacks;
    let weights = larql_vindex::load_model_weights_kquant(std::path::Path::new(&vindex), &mut cb).expect("load real vindex weights");
    let arch = &*weights.arch;
    let hidden = weights.hidden_size;
    let norm_offset = arch.norm_weight_offset();
    let eps = arch.norm_eps();

    let sample_moe = build_moe_weights(&weights, arch, HOME_LAYERS[0]).expect("home layer must be MoE");
    let top_k = sample_moe.top_k;
    let num_experts = sample_moe.num_experts;
    let chance_overlap = (top_k * top_k) as f64 / num_experts as f64;
    println!("top-{top_k} of {num_experts} experts; chance-level overlap ~= {chance_overlap:.2}\n");

    let mut all_cells = Vec::new();
    for i in 0..NUM_PROMPTS {
        let dump_dir = format!("{dump_prefix}{i}");
        println!("--- prompt {i}: {dump_dir} ---");
        let cells = run_prompt(&weights, &dump_dir, hidden, norm_offset, eps);
        for c in &cells {
            println!("  L{:>2} <- lead {:>2}: overlap {}/{}", c.home_layer, c.lead, c.overlap, top_k);
        }
        all_cells.extend(cells);
    }

    println!("\n=== mean overlap by lead time (across {NUM_PROMPTS} prompts x {} home layers) ===", HOME_LAYERS.len());
    for &k in &LEAD_TIMES {
        let matching: Vec<&Cell> = all_cells.iter().filter(|c| c.lead == k).collect();
        if matching.is_empty() {
            continue;
        }
        let mean = matching.iter().map(|c| c.overlap as f64).sum::<f64>() / matching.len() as f64;
        println!("  lead {k:>2}: mean overlap {mean:.2}/{top_k}  (n={}, chance level {chance_overlap:.2})", matching.len());
    }

    println!(
        "\nInterpretation: overlap well above chance ({chance_overlap:.2}) even at large lead times would mean\n\
         routing is substantially decided before layer L is reached -- real, prefetchable lead time.\n\
         Overlap collapsing toward chance quickly would mirror GLA-4's danger-signal finding: no\n\
         early-warning, at least not from this cheap a probe (L-k's fully-computed residual through L's router)."
    );
}
