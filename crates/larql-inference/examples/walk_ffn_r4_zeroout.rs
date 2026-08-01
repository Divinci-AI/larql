//! R4 zero-out for the sparse-FFN / ANN routing programme.
//!
//! **The only question:** with perfect routes supplied for free, is the
//! remaining sparse execution cheaper than dense execution?
//!
//! If sparse execution with **free** routing still can't turn a faithful
//! row reduction into proportional speed, the bottleneck is the sparse
//! kernel, not selection — and there is no reason to repair HNSW or
//! pursue better routing for the all-layer route.
//!
//! ## Two levers, deliberately separated
//!
//! A known route does two things here, and collapsing them into one
//! number is how you get an unexplainable result:
//!
//! 1. **Selection disappears** — the O(N·d) gate sweep becomes O(K·d).
//! 2. **A different kernel unlocks** — `sparse.rs:158` fires the
//!    contiguous-gather kernel (`sparse:gather_q4k`) only when the route
//!    is known in advance AND unranked (`!rank_within_pool`). Production
//!    (which must search) can never reach it.
//!
//! `rank_within_pool = true` on a K-sized pool forces the production
//! kernel with a known route, isolating lever 1 from lever 2.
//!
//! ## Arms
//!
//! | Arm | Pool | Rank | Kernel | Selection cost |
//! |---|---|---|---|---|
//! | `dense` | — | — | dense FFN | none |
//! | `exact-pool` | all N | yes | `parallel_q4k_down` | N row-dots |
//! | `oracle-par` | top-K | yes | `parallel_q4k_down` | K row-dots |
//! | `oracle-gath` | top-K | no | `gather_q4k` | K row-dots |
//!
//! - **R4 decision:** `oracle-par` vs `dense` — routing free, production kernel.
//! - **Kernel bonus:** `oracle-par` vs `oracle-gath` — selection-identical.
//!
//! `exact-pool` is retained as a kernel-matched upper bound on selection
//! cost, NOT as the router budget: it pays N *scattered* row-dots where
//! production pays a batched gemv, so differencing it overstates
//! recoverable production time by ~30%. The router budget is quoted
//! against production's own ~3 ms/layer selection cost.
//!
//! ## What is *not* zeroed
//!
//! The oracle arms still pay the K gate row-dots. That is correct: the
//! sparse kernel consumes `gate_score` as its activation input, and any
//! real router emits candidate IDs, not exact scores. Those K dots are
//! execution work every arm must do. What disappears is the search over N.
//!
//! ## Measurement protocol (paired, interleaved)
//!
//! Arms are **not** run in long separate blocks — thermal and scheduler
//! drift over a multi-minute block would be indistinguishable from an arm
//! effect. Instead every repeat measures all four arms back-to-back in a
//! **rotated order**, and the reported statistic is the distribution of
//! **per-repeat paired ratios** (`arm / dense` within the same repeat), so
//! drift cancels. Dense wall-time per repeat is retained as a throttle
//! sentinel; a drifting sentinel invalidates the run rather than being
//! averaged away.
//!
//! Routes are captured once per cell and held fixed across every repeat.
//!
//! ## Guardrail (access-pattern faithfulness)
//!
//! Routes are captured from `exact-pool`'s runtime trace in executed visit
//! order and replayed in that order, so the oracle arms issue the same
//! scattered row reads a real stage-1 router would produce — the search is
//! zeroed, the gather disorder is not. Capturing from `exact-pool` rather
//! than production also keeps the gate-score source identical (interleaved
//! Q4K bytes), so a parity failure means a real divergence.
//!
//! Usage: `cargo run --release --example walk_ffn_r4_zeroout -- [VINDEX_DIR]`
//! Refuses to run off AC power or on a loaded box; `--allow-dirty` overrides
//! (and stamps every number as provisional).

use larql_inference::vindex::{
    insert_q4k_layer_tensors_resident, LayerTraceRecord, PhaseTimingsHandle, WalkFfn, WalkFfnConfig,
};
use larql_inference::{load_tokenizer, predict_with_ffn};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

/// Untimed forwards per arm before the repeat loop, so every kernel and
/// shape is warm before anything is recorded.
const WARMUP: usize = 8;
/// Forwards averaged into one arm's measurement inside a single repeat.
/// Small on purpose: a repeat must stay short enough that all four arms
/// see the same thermal conditions.
const BLOCK: usize = 3;
/// Paired repeats per cell.
const REPEATS: usize = 16;

/// Fraction of drift in the dense sentinel (last quartile vs first) above
/// which the cell is reported as thermally invalid rather than believed.
const SENTINEL_DRIFT_LIMIT: f64 = 0.10;

/// 1-minute load average above which the box is treated as contended.
/// A concurrent build is as fatal to a ratio experiment as throttling.
const LOAD_LIMIT: f64 = 3.0;

/// Relative tolerance for residual-delta parity. Arms executing the same
/// feature set in a different accumulation order differ in the last float
/// bits; that is a summation-order artifact. Set equality is exact.
const PARITY_REL_TOL: f32 = 1e-4;

/// Decisive cells only — `(band, K)`. Densely sweeping accuracy-dead
/// cells buys nothing. `usize::MAX` band = all layers.
const CELLS: [(usize, usize); 7] = [
    (usize::MAX, 2048),
    (usize::MAX, 4096),
    (usize::MAX, 6144),
    (4, 2048),
    (4, 4096),
    (9, 2048),
    (9, 4096),
];

/// Decode prompt. A single trailing token gives the seq_len=1 shape where
/// the FFN is a matvec — the regime the sparse walk targets. A multi-token
/// prompt would measure prefill (batched gemm), where dense wins by
/// construction and the comparison is meaningless.
const PROMPT: &str = "The capital of France is the city of";

const ARM_DENSE: usize = 0;
const ARM_EXACT: usize = 1;
const ARM_ORACLE_PAR: usize = 2;
const ARM_ORACLE_GATH: usize = 3;
const ARM_NAMES: [&str; 4] = ["dense", "exact-pool", "oracle-par", "oracle-gath"];

/// Per-layer routes in executed visit order.
type Routes = Arc<Vec<Vec<usize>>>;

// ── Environment gates ────────────────────────────────────────────────

fn on_ac_power() -> bool {
    std::process::Command::new("pmset")
        .args(["-g", "ps"])
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).contains("AC Power"))
        .unwrap_or(false)
}

fn load_average() -> f64 {
    std::process::Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
        .ok()
        .and_then(|o| {
            String::from_utf8_lossy(&o.stdout)
                .split_whitespace()
                .nth(1)
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or(f64::NAN)
}

// ── Statistics ───────────────────────────────────────────────────────

fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
    }
}

/// Median, p25, p75, min, max of a sample.
struct Summary {
    median: f64,
    p25: f64,
    p75: f64,
    min: f64,
    max: f64,
}

fn summarise(values: &[f64]) -> Summary {
    let mut s = values.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Summary {
        median: quantile(&s, 0.5),
        p25: quantile(&s, 0.25),
        p75: quantile(&s, 0.75),
        min: *s.first().unwrap_or(&f64::NAN),
        max: *s.last().unwrap_or(&f64::NAN),
    }
}

/// Drift in the throttle sentinel: mean of the last quarter of repeats
/// against the first quarter. Large positive = the box slowed mid-run.
fn sentinel_drift(dense_us: &[f64]) -> f64 {
    let q = (dense_us.len() / 4).max(1);
    let head: f64 = dense_us[..q].iter().sum::<f64>() / q as f64;
    let tail: f64 = dense_us[dense_us.len() - q..].iter().sum::<f64>() / q as f64;
    (tail - head) / head
}

// ── Trace helpers ────────────────────────────────────────────────────

fn sparse_from(num_layers: usize, band: usize) -> usize {
    if band == usize::MAX {
        0
    } else {
        num_layers.saturating_sub(band)
    }
}

fn band_label(band: usize) -> String {
    if band == usize::MAX {
        "all".to_string()
    } else {
        format!("last-{band}")
    }
}

/// Fold a runtime trace into per-layer routes, preserving executed visit
/// order. Dense layers contribute nothing — they never consult a pool.
fn routes_from_trace(records: &[LayerTraceRecord], num_layers: usize) -> Routes {
    let mut per_layer: Vec<Vec<usize>> = vec![Vec::new(); num_layers];
    for rec in records {
        if rec.features.is_empty() || rec.layer >= num_layers {
            continue;
        }
        let mut ranked: Vec<(usize, usize)> =
            rec.features.iter().map(|f| (f.rank, f.feature)).collect();
        ranked.sort_unstable();
        per_layer[rec.layer] = ranked.into_iter().map(|(_, feat)| feat).collect();
    }
    Arc::new(per_layer)
}

fn kernels(trace: &[LayerTraceRecord]) -> String {
    let mut names: Vec<&str> = trace
        .iter()
        .filter(|r| !r.features.is_empty())
        .map(|r| r.path)
        .collect();
    names.sort_unstable();
    names.dedup();
    names.join("+")
}

/// Parity: identical executed feature SET per layer (order-insensitive)
/// and residual deltas within [`PARITY_REL_TOL`].
fn trace_parity(a: &[LayerTraceRecord], b: &[LayerTraceRecord]) -> Result<(), String> {
    if a.len() != b.len() {
        return Err(format!("record count {} vs {}", a.len(), b.len()));
    }
    for (ra, rb) in a.iter().zip(b.iter()) {
        if ra.layer != rb.layer || ra.position != rb.position {
            return Err(format!("identity L{}p{}", ra.layer, ra.position));
        }
        let mut fa: Vec<usize> = ra.features.iter().map(|f| f.feature).collect();
        let mut fb: Vec<usize> = rb.features.iter().map(|f| f.feature).collect();
        fa.sort_unstable();
        fb.sort_unstable();
        if fa != fb {
            return Err(format!(
                "L{} feature set differs ({} vs {})",
                ra.layer,
                fa.len(),
                fb.len()
            ));
        }
        let (x, y) = (ra.residual_delta_norm, rb.residual_delta_norm);
        let rel = (x - y).abs() / x.abs().max(y.abs()).max(f32::MIN_POSITIVE);
        if rel > PARITY_REL_TOL {
            return Err(format!("L{} residual {x} vs {y} (rel {rel:.2e})", ra.layer));
        }
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let allow_dirty = args.iter().any(|a| a == "--allow-dirty");
    let vindex = args
        .iter()
        .skip(1)
        .find(|a| !a.starts_with("--"))
        .cloned()
        .unwrap_or_else(|| "output/gemma3-4b-q4k-v2.vindex".to_string());

    // ── Refuse to emit a bad number ──────────────────────────────────
    let ac = on_ac_power();
    let load = load_average();
    let dirty = !ac || !(load < LOAD_LIMIT);
    if dirty {
        eprintln!("REFUSING TO MEASURE:");
        if !ac {
            eprintln!("  - not on AC power (throttled; sparse-vs-dense is exactly the");
            eprintln!("    bandwidth/compute tradeoff throttling distorts)");
        }
        if !(load < LOAD_LIMIT) {
            eprintln!("  - 1-min load average {load:.2} exceeds {LOAD_LIMIT:.1} (contended box)");
        }
        if !allow_dirty {
            eprintln!(
                "\n  Fix the environment, or pass --allow-dirty to record PROVISIONAL numbers."
            );
            std::process::exit(2);
        }
        eprintln!("\n  --allow-dirty set: every number below is PROVISIONAL.\n");
    }

    let dir = std::path::PathBuf::from(&vindex);
    let mut cb = larql_vindex::SilentLoadCallbacks;
    eprintln!("Loading {vindex} ...");
    let mut weights = larql_vindex::load_model_weights_kquant(&dir, &mut cb).expect("weights");
    let mut index = larql_vindex::VectorIndex::load_vindex(&dir, &mut cb).expect("index");
    index.load_interleaved_kquant(&dir).expect("interleaved");
    index.load_attn_kquant(&dir).expect("attn");
    let _ = index.load_lm_head_kquant(&dir);
    index.load_down_features_q4k(&dir).expect("down sidecar");
    let _ = index.load_gate_vectors_q4(&dir);
    assert!(
        index.has_down_features_kquant(),
        "feature-major down sidecar required"
    );
    let tok = load_tokenizer(&dir).expect("tok");
    for layer in 0..weights.num_layers {
        insert_q4k_layer_tensors_resident(&mut weights, &index, layer).expect("dequant attn");
    }

    let nl = weights.num_layers;
    let features = index.num_features(nl.saturating_sub(1));
    let full = tok.encode(PROMPT, true).expect("enc").get_ids().to_vec();
    let ids = vec![*full.last().unwrap()];

    println!(
        "\nR4 zero-out (paired, interleaved) — {nl} layers, {features} features/layer, seq_len=1\n\
         {REPEATS} repeats × {BLOCK}-forward blocks, rotated arm order, {WARMUP} warmup/arm\n\
         AC={ac} load={load:.2}{}\n",
        if dirty { "  [PROVISIONAL]" } else { "" }
    );

    for (band, k) in CELLS {
        let sf = sparse_from(nl, band);
        let base = WalkFfnConfig::hybrid(nl, sf, k);

        let all_pool: Routes = Arc::new(
            (0..nl)
                .map(|l| {
                    if l >= sf {
                        (0..index.num_features(l)).collect()
                    } else {
                        Vec::new()
                    }
                })
                .collect(),
        );
        let exact_cfg = base
            .clone()
            .with_pool_per_layer(all_pool)
            .with_precomputed_routing(true)
            .with_rank_within_pool(true);

        // Routes captured ONCE per cell and held fixed across repeats.
        let cap_ffn = WalkFfn::from_config(&weights, &index, exact_cfg.clone()).with_trace();
        let cap_token = predict_with_ffn(&weights, &tok, &ids, 1, &cap_ffn)
            .token_ids
            .first()
            .copied()
            .unwrap_or(0);
        let exact_trace = cap_ffn.take_runtime_trace();
        let routes = routes_from_trace(&exact_trace, nl);
        let routed_layers = routes.iter().filter(|r| !r.is_empty()).count();

        let oracle_par_cfg = base
            .clone()
            .with_pool_per_layer(routes.clone())
            .with_precomputed_routing(true)
            .with_rank_within_pool(true);
        let oracle_gath_cfg = base
            .clone()
            .with_pool_per_layer(routes)
            .with_precomputed_routing(true);

        // ── Parity, untimed, before any timing is believed ───────────
        let mut parity = Vec::new();
        for (name, cfg) in [
            ("oracle-par", &oracle_par_cfg),
            ("oracle-gath", &oracle_gath_cfg),
        ] {
            let pf = WalkFfn::from_config(&weights, &index, cfg.clone()).with_trace();
            let tkn = predict_with_ffn(&weights, &tok, &ids, 1, &pf)
                .token_ids
                .first()
                .copied()
                .unwrap_or(0);
            let tr = pf.take_runtime_trace();
            let verdict = match trace_parity(&exact_trace, &tr) {
                Ok(()) if tkn == cap_token => "parity ✓".to_string(),
                Ok(()) => format!("token {cap_token}≠{tkn}"),
                Err(e) => format!("DIFF {e}"),
            };
            parity.push(format!("{name}[{}] {verdict}", kernels(&tr)));
        }

        // ── Build all arms once; caches persist across repeats ───────
        let timings: Vec<Arc<PhaseTimingsHandle>> = (0..4)
            .map(|_| Arc::new(PhaseTimingsHandle::default()))
            .collect();
        let dense_ffn =
            WalkFfn::new_unlimited(&weights, &index).with_phase_timings(timings[ARM_DENSE].clone());
        let exact_ffn = WalkFfn::from_config(&weights, &index, exact_cfg)
            .with_phase_timings(timings[ARM_EXACT].clone());
        let par_ffn = WalkFfn::from_config(&weights, &index, oracle_par_cfg)
            .with_phase_timings(timings[ARM_ORACLE_PAR].clone());
        let gath_ffn = WalkFfn::from_config(&weights, &index, oracle_gath_cfg)
            .with_phase_timings(timings[ARM_ORACLE_GATH].clone());
        let arms: [&dyn larql_inference::ffn::FfnBackend; 4] =
            [&dense_ffn, &exact_ffn, &par_ffn, &gath_ffn];

        let run_block = |arm: &dyn larql_inference::ffn::FfnBackend| -> (f64, u32) {
            let t = Instant::now();
            let mut last = 0u32;
            for _ in 0..BLOCK {
                last = predict_with_ffn(&weights, &tok, &ids, 1, arm)
                    .token_ids
                    .first()
                    .copied()
                    .unwrap_or(0);
            }
            (t.elapsed().as_micros() as f64 / BLOCK as f64, last)
        };

        // Warm every arm and shape before recording anything.
        for arm in arms {
            for _ in 0..WARMUP {
                let _ = predict_with_ffn(&weights, &tok, &ids, 1, arm);
            }
        }
        for t in &timings {
            t.gate_knn_ns.store(0, Ordering::Relaxed);
            t.parallel_scan_ns.store(0, Ordering::Relaxed);
            t.calls.store(0, Ordering::Relaxed);
        }

        // ── Paired, rotated repeats ──────────────────────────────────
        let mut samples: [Vec<f64>; 4] = Default::default();
        let mut tokens = [0u32; 4];
        for repeat in 0..REPEATS {
            for slot in 0..4 {
                let arm = (slot + repeat) % 4; // rotate order every repeat
                let (us, tkn) = run_block(arms[arm]);
                samples[arm].push(us);
                tokens[arm] = tkn;
            }
        }

        let drift = sentinel_drift(&samples[ARM_DENSE]);
        let dense_sum = summarise(&samples[ARM_DENSE]);

        println!(
            "── {} K={k}  ({routed_layers} routed layers)  dense {:.0} µs [{:.0}–{:.0}]  \
             {:.1} tok/s  sentinel drift {:+.1}%{}",
            band_label(band),
            dense_sum.median,
            dense_sum.min,
            dense_sum.max,
            1e6 / dense_sum.median,
            drift * 100.0,
            if drift.abs() > SENTINEL_DRIFT_LIMIT {
                "  ⚠ THERMALLY INVALID"
            } else {
                ""
            }
        );
        for p in &parity {
            println!("     {p}");
        }
        println!(
            "     {:<12} {:>9} {:>28} {:>9} {:>5}",
            "arm", "median µs", "paired ratio vs dense (med [IQR])", "sel µs/L", "top1"
        );
        for arm in [ARM_EXACT, ARM_ORACLE_PAR, ARM_ORACLE_GATH] {
            // Paired ratios: computed within a repeat so drift cancels.
            let ratios: Vec<f64> = samples[arm]
                .iter()
                .zip(samples[ARM_DENSE].iter())
                .map(|(a, d)| d / a)
                .collect();
            let r = summarise(&ratios);
            let s = summarise(&samples[arm]);
            let calls = timings[arm].calls.load(Ordering::Relaxed);
            let sel = if calls == 0 {
                f64::NAN
            } else {
                timings[arm].gate_knn_ns.load(Ordering::Relaxed) as f64 / calls as f64 / 1000.0
            };
            println!(
                "     {:<12} {:>9.0} {:>18.3}× [{:.3}–{:.3}] {:>9.1} {:>5}",
                ARM_NAMES[arm],
                s.median,
                r.median,
                r.p25,
                r.p75,
                sel,
                if tokens[arm] == tokens[ARM_DENSE] {
                    "✓"
                } else {
                    "✗"
                },
            );
        }
        println!();
    }

    println!("  Analytic per-layer work (N = {features} features):");
    println!(
        "  {:<8} {:>12} {:>16} {:>14}",
        "K", "exec rows", "selection rows", "exec/dense"
    );
    for k in [2048usize, 4096, 6144] {
        println!(
            "  {:<8} {:>12} {:>16} {:>13.2}×",
            k,
            3 * k,
            features,
            (3 * k) as f64 / (3 * features) as f64
        );
    }
    println!(
        "\n  Row-count ceiling on the sparse band: {:.2}× at K=2048, {:.2}× at K=4096,\n  \
         {:.2}× at K=6144. A paired ratio far below its ceiling is gather\n  \
         inefficiency, not selection cost — which no router can recover.",
        features as f64 / 2048.0,
        features as f64 / 4096.0,
        features as f64 / 6144.0
    );
}
