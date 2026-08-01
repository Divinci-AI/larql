//! Promote the NOT-DECISION-GRADE efficiency classes by repeating the census.
//!
//! `k3_ledger::classes` banks `down` at 0.78 [0.73-0.82] and the ungrouped
//! expert shape at 0.67 [0.63-0.70] over three repeats. Both exceed the 5%
//! decision-grade bar, and both feed DEC-8.7b's target-row selection — which,
//! now that the eta programme is closed as a throughput lever, is the only live
//! throughput rung. A 10% swing on either can pick a different bit-width.
//!
//! This calls `profile_shape_census` **verbatim**, so the samples are drawn from
//! the identical protocol the bank was drawn from. A faster focused loop would
//! measure a different thing and could not be banked beside the existing rows.
//!
//! ## What it reports, and what it deliberately does not conclude
//!
//! Mean and **standard error**, not spread — they are different quantities and
//! the second gets read as the first. But the mean is only meaningful if the
//! samples are unimodal, and 0.73 / 0.78 / 0.82 is as consistent with two modes
//! (page alignment, thermal state, allocator luck) as with noise. So this also
//! prints the sorted samples, a coarse histogram, and a largest-gap ratio, and
//! leaves the modality call to a human. **If bimodal, the honest bank is the
//! pessimistic mode, not the mean.**
//!
//! Usage:
//!   cargo run --release -p larql-compute-metal --example diag_eta_repeats [-- N]

extern crate blas_src;

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("Requires macOS + Metal");
}

#[cfg(target_os = "macos")]
fn main() {
    use larql_compute_metal::diag::kernel_profile::profile_shape_census;

    /// Cells to promote, plus attention as a stability control — it banked at
    /// 2% spread, so if it widens here the harness moved, not the kernels.
    const TARGETS: [(&str, &str); 3] = [
        ("K3 latent up 7168x3584", "q6k"),
        ("K3 expert w2 3584x3072", "q6k"),
        ("K3 KDA attn 12288x7168", "q6k"),
    ];

    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(15);

    println!("eta repeats: {n} runs of the census protocol (34 layers, 5 warmup, 30 iters)");
    println!();

    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); TARGETS.len()];
    for run in 0..n {
        let cells = profile_shape_census(34, 5, 30);
        for (i, (shape, kernel)) in TARGETS.iter().enumerate() {
            if let Some(c) = cells
                .iter()
                .find(|c| c.shape == *shape && c.kernel == *kernel)
            {
                samples[i].push(c.eta);
            }
        }
        eprint!("\r  run {}/{n}", run + 1);
    }
    eprintln!();
    println!();

    for (i, (shape, kernel)) in TARGETS.iter().enumerate() {
        let mut v = samples[i].clone();
        if v.is_empty() {
            println!("{shape} {kernel}: NO SAMPLES (cell not found)");
            continue;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n_f = v.len() as f64;
        let mean = v.iter().sum::<f64>() / n_f;
        let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n_f - 1.0).max(1.0);
        let se = (var / n_f).sqrt();
        let (lo, hi) = (v[0], v[v.len() - 1]);

        // Largest gap between consecutive order statistics, against the median
        // gap. A big interior gap is the cheap smell of two modes; it is a
        // smell, not a test.
        let gaps: Vec<f64> = v.windows(2).map(|w| w[1] - w[0]).collect();
        let mut sorted_gaps = gaps.clone();
        sorted_gaps.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med_gap = sorted_gaps[sorted_gaps.len() / 2].max(1e-9);
        let max_gap = gaps.iter().cloned().fold(0.0f64, f64::max);
        let gap_at = gaps.iter().position(|g| *g == max_gap).unwrap_or(0);

        println!("{shape}  {kernel}   n={}", v.len());
        println!(
            "  mean {mean:.4}  SE {se:.4}   observed {lo:.3}-{hi:.3}  spread {:.1}%",
            100.0 * (hi - lo) / mean
        );
        println!("  samples: {:?}", v.iter().map(|x| (x * 1000.0).round() / 1000.0).collect::<Vec<_>>());

        // Coarse histogram over the observed range.
        const BINS: usize = 10;
        let width = ((hi - lo) / BINS as f64).max(1e-9);
        let mut counts = [0usize; BINS];
        for x in &v {
            let b = (((x - lo) / width) as usize).min(BINS - 1);
            counts[b] += 1;
        }
        for (b, c) in counts.iter().enumerate() {
            println!(
                "    {:.3}-{:.3} {}",
                lo + b as f64 * width,
                lo + (b + 1) as f64 * width,
                "#".repeat(*c)
            );
        }
        println!(
            "  largest gap {max_gap:.4} at position {}/{} = {:.1}x the median gap",
            gap_at + 1,
            gaps.len(),
            max_gap / med_gap
        );
        let decision_grade = (hi - lo) / mean <= 0.05;
        println!(
            "  => {} at the 5% bar{}",
            if decision_grade { "DECISION-GRADE" } else { "still NOT decision-grade" },
            if max_gap / med_gap > 3.0 {
                "; interior gap is large — inspect for two modes before banking a mean"
            } else {
                ""
            }
        );
        println!();
    }
    println!("Bank mean +/- SE only if the histogram is unimodal.");
    println!("If two modes, bank the pessimistic one — a mean between modes describes nothing.");
}
