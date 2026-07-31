//! Rendering for `larql k3-ledger` — I/O, excluded from coverage.
//!
//! Every quoted ratio names its convention (R0). Anything resting on an
//! unmeasured input says so in the output rather than in a footnote.

use serde_json::json;

use super::args::{BlockArgs, BudgetArgs, FrontierArgs, TouchArgs};
use super::block::{self, DraftProfile, PhysicalReuse, StateTraffic};
use super::budget::{self, LinkPremises};
use super::frontier::{self, ServingPremises};
use super::geometry::{BitConvention, K3Geometry};
use super::touch::{self, SliceTier};

type R = Result<(), Box<dyn std::error::Error>>;

fn header(geom: &K3Geometry) {
    println!(
        "K3 geometry: {} layers ({} KDA + {} MLA), {}-of-{} ({:.2}% activation), \
         expert {:.2} MB",
        geom.n_layers,
        geom.n_kda_layers,
        geom.n_mla_layers,
        geom.top_k,
        geom.n_experts,
        100.0 * geom.activation_fraction(),
        geom.expert_bytes as f64 / 1e6,
    );
    println!(
        "activated {:.2} B params (dense {:.2} B + routed {:.2} B); vision {}",
        geom.activated_params() as f64 / 1e9,
        geom.dense_params() as f64 / 1e9,
        geom.routed_activated_params() as f64 / 1e9,
        if geom.vision_included {
            "included"
        } else {
            "EXCLUDED"
        },
    );
    if geom.branches_are_equal() {
        println!(
            "branches w1/w2/w3 equal to the byte -> dropping one caps at {:.2}x (retention {:.3})",
            1.0 / geom.up_fold_retention(),
            geom.up_fold_retention(),
        );
    }
}

pub fn budget(geom: &K3Geometry, a: &BudgetArgs, as_json: bool) -> R {
    let link = LinkPremises {
        gb_s: a.link_gb_s,
        target_tok_s: a.target_tok_s,
        ports: a.ports,
    };
    let m = budget::miss_budget(geom, &link);
    let g = budget::granularity(geom, &link, a.feature_fraction, 2.0, a.locality);

    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({"budget": m, "granularity": g}))?
        );
        return Ok(());
    }
    header(geom);
    println!();
    println!(
        "miss budget   {:.1} MB/token ({} GB/s x {} port(s) / {} tok/s)",
        m.miss_budget_bytes / 1e6,
        a.link_gb_s,
        a.ports,
        a.target_tok_s
    );
    println!("required      {:.2} GB/token", m.required_fetch_bytes / 1e9);
    println!("GAP           {:.1}x", m.gap);
    println!(
        "  {} expert-visits/token, {:.1} KB allowed at each = {:.3}% of an expert",
        m.expert_visits,
        m.bytes_allowed_per_visit / 1e3,
        100.0 * m.fraction_of_expert_allowed,
    );
    println!();
    println!(
        "--- read granularity at {:.1}% of features ---",
        100.0 * a.feature_fraction
    );
    println!(
        "  feature row {:.0} B, {:.0} rows/visit",
        g.feature_row_bytes, g.features_selected
    );
    println!(
        "  ideal {:.0} KB/visit -> paged {:.2} MB/visit ({:.2}x amplification)",
        g.ideal_bytes_per_visit / 1e3,
        g.paged_bytes_per_visit / 1e6,
        g.read_amplification
    );
    println!(
        "  {:.2}M IOPS required vs {:.2}M bandwidth-equivalent -> {:.1}x short",
        g.iops_required / 1e6,
        g.iops_bandwidth_equivalent / 1e6,
        g.iops_shortfall
    );
    println!("  (request rate binds before bandwidth does)");
    Ok(())
}

pub fn touch(geom: &K3Geometry, p: &ServingPremises, a: &TouchArgs, as_json: bool) -> R {
    let l = touch::touch_ledger(geom, p, a.target_tok_s);
    let slices: Vec<_> = a
        .slice_gb
        .iter()
        .map(|gb| {
            touch::slice_composition(
                geom,
                p,
                &SliceTier {
                    size_bytes: gb * 1e9,
                },
            )
        })
        .collect();

    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({"ledger": l, "slices": slices}))?
        );
        return Ok(());
    }
    header(geom);
    println!();
    println!(
        "--- per-token touch at {:.2} all-in bits ---",
        p.dense_all_in_bits
    );
    println!(
        "  dense (irreducible) {:>7.2} GB -> {:>5.1} tok/s",
        l.dense_bytes / 1e9,
        l.tok_s_dense_only
    );
    println!("  routed experts      {:>7.2} GB", l.routed_bytes / 1e9);
    println!(
        "  TOTAL               {:>7.2} GB -> {:>5.1} tok/s",
        l.total_bytes / 1e9,
        l.tok_s_total
    );
    println!();
    println!(
        "  dense is {:.1}% of activated params — the LARGER half",
        100.0 * l.dense_share
    );
    println!(
        "  expert-side levers cap at {:.2}x; {:.2}x needed for {} tok/s",
        l.expert_side_ceiling,
        l.reduction_needed.unwrap_or(f64::NAN),
        a.target_tok_s
    );
    println!(
        "  >> even a perfect expert lever leaves {:.1} tok/s",
        l.tok_s_dense_only
    );
    println!(
        "  at CPU-attainable {:.0} GB/s the same read gives {:.1} tok/s",
        frontier::BW_CPU_GB_S,
        frontier::BW_CPU_GB_S * 1e9 * p.dequant_efficiency / l.total_bytes,
    );
    println!();
    println!("--- slice composition (down-row payloads, up folded) ---");
    for s in &slices {
        println!(
            "  {:>3.0} GB: {:>6.0} experts ({:.1}% of bank) -> {:.2} of {} hit/layer uniform",
            s.slice_bytes / 1e9,
            s.resident_experts,
            100.0 * s.resident_fraction,
            s.uniform_hits_per_layer,
            geom.top_k
        );
    }
    Ok(())
}

pub fn frontier(geom: &K3Geometry, p: &ServingPremises, a: &FrontierArgs, as_json: bool) -> R {
    let p = ServingPremises {
        routed_retention: a.routed_retention.unwrap_or(geom.up_fold_retention()),
        ..*p
    };
    let rows = frontier::frontier(geom, &p, &a.targets);
    let ceiling = frontier::expert_side_ceiling_tok_s(geom, &p);

    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "rows": rows, "expert_side_ceiling_tok_s": ceiling,
                "routed_retention": p.routed_retention,
            }))?
        );
        return Ok(());
    }
    header(geom);
    println!();
    println!(
        "routed retention {:.3} ({:.2}x reduction); bandwidth {:.0} GB/s x {:.2} efficiency",
        p.routed_retention,
        1.0 / p.routed_retention,
        p.bandwidth_gb_s,
        p.dequant_efficiency
    );
    if a.routed_retention.is_none() {
        println!("  (banked up-fold ceiling; anything lower is row sparsity, UNMEASURED)");
    }
    println!();
    println!(
        "{:>7} {:>12} {:>13} {:>8} {:>8}  verdict",
        "target", "budget", "dense budget", "payload", "all-in"
    );
    for r in &rows {
        println!(
            "{:>7.1} {:>11.2}G {:>12.2}G {:>8.2} {:>8.2}  {:?}",
            r.target_tok_s,
            r.budget_bytes / 1e9,
            r.dense_budget_bytes / 1e9,
            r.payload_bits,
            r.all_in_bits,
            r.verdict
        );
    }
    println!(
        "  (bits are per weight; {} excludes block scales, {} includes them)",
        BitConvention::Payload.label(),
        BitConvention::AllIn.label(),
    );
    let undecidable: Vec<_> = rows
        .iter()
        .filter(|r| !frontier::expert_side_can_decide(geom, &p, r.target_tok_s))
        .map(|r| format!("{:.1}", r.target_tok_s))
        .collect();
    if !undecidable.is_empty() {
        println!(
            "  R4: no expert-side experiment can decide {} tok/s",
            undecidable.join(", ")
        );
    }
    if a.zero_out_check {
        println!();
        println!("--- R4 zero-out: routed traffic deleted entirely ---");
        println!(
            "  dense at {:.2} bits caps decode at {:.1} tok/s",
            p.dense_all_in_bits, ceiling
        );
        println!("  >> no expert-side experiment can decide any target above {ceiling:.1}");
    }
    Ok(())
}

pub fn block(geom: &K3Geometry, p: &ServingPremises, a: &BlockArgs, as_json: bool) -> R {
    let p = ServingPremises {
        routed_retention: a.routed_retention.unwrap_or(geom.up_fold_retention()),
        ..*p
    };
    let draft = DraftProfile::new(a.proposal_width, a.mean_accepted);
    let state = match a.state_prefix_ladder_bytes {
        Some(w) => StateTraffic {
            bytes_per_pass: a.state_bytes_per_pass,
            ..StateTraffic::prefix_state_ladder(geom.n_kda_layers, w)
        },
        None => StateTraffic {
            bytes_per_pass: a.state_bytes_per_pass,
            bytes_per_position: a.state_bytes_per_position,
            measured: a.state_bytes_per_pass > 0.0 || a.state_bytes_per_position > 0.0,
        },
    };

    if draft.assumes_perfect() {
        eprintln!(
            "WARNING: proposal width == mean accepted. That is the R5 error \
             (perfect acceptance assumed). Costs key on positions EVALUATED, \
             throughput on positions COMMITTED."
        );
    }

    let rows: Vec<_> = if a.token_loop {
        a.widths
            .iter()
            .map(|&t| {
                let reuse = PhysicalReuse::token_loop(t);
                block::evaluate(geom, &p, &draft, t, reuse, state, a.target_tok_s)
            })
            .collect()
    } else {
        block::sweep(geom, &p, &draft, &a.widths, state, a.target_tok_s)
    };

    let refused: Vec<_> = rows
        .iter()
        .filter(|r| {
            let reuse = PhysicalReuse::assumed_ideal(r.union_equivalents);
            block::dense_alone_refuses(geom, &p, r.accepted, r.width, reuse, state, a.target_tok_s)
        })
        .map(|r| r.width.to_string())
        .collect();

    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "rows": rows, "fitted_alpha": draft.fitted_alpha(),
                "provisional": "alpha fitted from ONE observed (width, accepted) pair \
                                under a constant-alpha geometric model; optima are \
                                design points, not performance bars",
            }))?
        );
        return Ok(());
    }
    header(geom);
    println!();
    println!(
        "drafter: {} proposed / {:.2} committed, fitted alpha {:.4}",
        a.proposal_width,
        a.mean_accepted,
        draft.fitted_alpha()
    );
    println!(
        "execution: {}",
        if a.token_loop {
            "TOKEN LOOP (no grouping)"
        } else {
            "ideal grouping ASSUMED (R6, unmeasured)"
        }
    );
    println!(
        "state traffic: {}",
        match (state.bytes_per_pass, state.bytes_per_position) {
            (0.0, 0.0) => "OMITTED — no state term in this budget (owner M1-B)".to_string(),
            (pass, pos) => format!(
                "{:.2} GB/pass + {:.2} GB/position [{}]",
                pass / 1e9,
                pos / 1e9,
                if state.measured { "measured" } else { "DERIVED, not measured" }
            ),
        }
    );
    println!();
    println!(
        "{:>3} {:>8} {:>8} {:>10} {:>9} {:>9}",
        "T", "A(T)", "u(T)", "GB/token", "tok/s", "G_r need"
    );
    for r in &rows {
        let need = r
            .routed_reduction_needed
            .map(|v| format!("{v:>9.2}"))
            .unwrap_or_else(|| "      inf".into());
        println!(
            "{:>3} {:>8.2} {:>8.2} {:>10.2} {:>9.2} {}",
            r.width,
            r.accepted,
            r.union_equivalents,
            r.bytes_per_committed_token / 1e9,
            r.tok_s,
            need
        );
    }
    if let Some(best) = rows
        .iter()
        .filter(|r| r.rho_max.is_some())
        .max_by(|x, y| x.rho_max.partial_cmp(&y.rho_max).unwrap())
    {
        println!();
        println!(
            "  interior optimum at T={} (needs routed reduction {:.2}x)",
            best.width,
            best.routed_reduction_needed.unwrap_or(f64::NAN)
        );
        println!("  >> a drafter's SHIPPED width is not its best width");
    }
    if !refused.is_empty() {
        println!();
        println!(
            "  R4: dense traffic alone refuses width(s) {} regardless of the routed side",
            refused.join(", ")
        );
    }
    println!();
    println!("  PROVISIONAL: alpha fitted from one observed pair under a constant-alpha");
    println!("  geometric model. These optima are design points, not performance bars.");
    if rows.iter().any(|r| r.rests_on_assumptions) {
        println!("  RESTS ON ASSUMPTIONS: physical reuse and/or state traffic unmeasured.");
    }
    Ok(())
}
