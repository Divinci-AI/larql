//! `vindex3 sensitivity` — SENSITIVITY-1A, the cheap local screen.
//!
//! Q-BANK is the promotion gate and costs 1,622 teacher-forced positions
//! per candidate. That does not scale to role x depth combinatorics, and it
//! certainly does not scale to K3, where no one will evaluate every
//! expert x layer x representation decision globally.
//!
//! So this asks a much cheaper question, from the weights alone and with no
//! forward pass anywhere:
//!
//! ```text
//! e(t) = || W - dequant(quant(W)) ||^2 / || W ||^2
//! ```
//!
//! the relative error quantising tensor `t` introduces. One pass over the
//! weights scores every tensor, and any candidate precision map is then a
//! sum over the tensors it protects — so hundreds of candidates cost one
//! screen rather than one screen each.
//!
//! **This may not work, and the response is fixed in advance** (see
//! `bench/prompts/quality-bank-1/SENSITIVITY-1.md`). Weight error measures
//! how far the weights move, not how strongly the model uses the directions
//! they move in. A clean failure says weight geometry alone does not
//! predict semantic sensitivity, which is the argument for the
//! activation-weighted rung, not a reason to tune this one until it agrees
//! with fifteen known answers.
//!
//! The quantiser here is the *same* `quantize_nvfp4` the compiler and the
//! loader use, so the screen cannot drift from the thing it is screening.

use std::path::PathBuf;

use clap::Args;
use larql_vindex::format::vindex3::inspect::inspect_container;
use larql_vindex::format::vindex3::opplan::exec::operands::{OperandStore, RepresentationSource};
use larql_vindex::format::vindex3::opplan::OperandRef;
use larql_vindex::format::vindex3::represent::policy::{classify_in, Role};

#[derive(Args)]
pub struct SensitivityArgs {
    /// Canonical container to screen.
    pub container: PathBuf,

    /// Write per-tensor scores here as JSON.
    #[arg(long)]
    pub output: PathBuf,

    /// SENSITIVITY-1B: capture per-feature activation second moments over
    /// a calibration set instead of scoring weight error alone.
    ///
    /// The file is JSON lines of `{"id": ..., "ids": [...]}`. 1A needs no
    /// forward pass; 1B needs one per calibration prompt, which is still
    /// far cheaper than a Q-BANK run per candidate.
    #[arg(long, value_name = "JSONL")]
    pub calibration: Option<PathBuf>,

    /// Where `--calibration` writes the captured moments and the
    /// reconstruction control.
    #[arg(long, value_name = "JSON")]
    pub moments: Option<PathBuf>,
}

#[derive(serde::Serialize)]
struct TensorScore {
    object: String,
    tensor: String,
    role: String,
    shape: Vec<usize>,
    /// Bytes this tensor occupies compiled.
    compiled_bytes: u64,
    /// Bytes it occupies at source precision.
    source_bytes: u64,
    /// Relative quantisation error — the screen's whole signal.
    rel_error: f64,
    /// Weight energy, so a caller can re-weight by magnitude rather than
    /// by the normalised score if it wants to.
    energy: f64,
}

pub fn run(args: SensitivityArgs) -> Result<(), Box<dyn std::error::Error>> {
    use larql_models::quant::nvfp4::{round_trip, NVFP4_GROUP_ELEMS};
    use larql_vindex::format::vindex3::represent::nvfp4_pack::PackLayout;

    let inspection = inspect_container(&args.container, false)?;
    // Canonical bytes: the screen scores what quantisation would do to the
    // source, so it must read the source.
    let store = OperandStore::open_for(
        &args.container,
        &inspection,
        None,
        RepresentationSource::Transient,
    )?;

    let text: std::collections::BTreeSet<&str> = inspection
        .graph
        .components
        .iter()
        .filter(|c| {
            c.role == larql_vindex::format::vindex3::graph::component::ComponentRole::PrimaryText
        })
        .map(|c| c.id.as_str())
        .collect();
    let primary: std::collections::BTreeSet<String> = inspection
        .graph
        .objects
        .iter()
        .filter(|o| text.contains(o.component.as_str()))
        .map(|o| o.id.clone())
        .collect();

    let mut scores = Vec::new();
    let started = std::time::Instant::now();

    for entry in inspection.index.representations.values() {
        let (header, _) = larql_vindex::format::vindex3::encode::segment::read_segment_header(
            &args.container.join(&entry.segment),
        )?;
        for t in &header.tensors {
            let role = classify_in(
                primary.contains(&entry.object),
                &entry.object,
                &t.name,
                &t.shape,
            );
            // Only tensors an encoding could apply to are worth scoring;
            // a norm has no candidate map to appear in.
            if !matches!(role, Role::DecoderLinear | Role::ExpertWeight) {
                continue;
            }
            let Ok(layout) = PackLayout::derive(&t.shape, &t.name) else {
                continue;
            };
            let values = store.load(&OperandRef {
                object: entry.object.clone(),
                tensor: t.name.clone(),
                dtype: t.dtype.clone(),
                shape: t.shape.clone(),
            })?;
            let back = round_trip(&values, layout.rows, layout.k)
                .map_err(|e| format!("{}: {e}", t.name))?;

            let mut num = 0f64;
            let mut den = 0f64;
            for (a, b) in values.iter().zip(&back) {
                let d = (*a - *b) as f64;
                num += d * d;
                den += (*a as f64) * (*a as f64);
            }
            scores.push(TensorScore {
                object: entry.object.clone(),
                tensor: t.name.clone(),
                role: role.name().to_string(),
                shape: t.shape.clone(),
                compiled_bytes: layout.total_len as u64,
                source_bytes: t.len,
                rel_error: if den > 0.0 { num / den } else { 0.0 },
                energy: den,
            });
            if scores.len() % 40 == 0 {
                println!(
                    "  scored {} tensors ({:.0}s)",
                    scores.len(),
                    started.elapsed().as_secs_f64()
                );
            }
        }
    }

    let _ = NVFP4_GROUP_ELEMS;
    let n = scores.len();
    let mean = scores.iter().map(|s| s.rel_error).sum::<f64>() / n.max(1) as f64;
    std::fs::write(&args.output, serde_json::to_string(&scores)?)?;
    println!(
        "scored {n} tensors in {:.0}s  (mean relative error {mean:.6})\n-> {}",
        started.elapsed().as_secs_f64(),
        args.output.display()
    );
    Ok(())
}

/// Accumulates per-feature second moments per (layer, site), plus one
/// sample of the FFN output for the reconstruction control.
#[derive(Default)]
pub struct MomentCollector {
    /// (layer, site) -> running sum of x_j^2, and the count.
    pub sums: std::collections::BTreeMap<(usize, u8), (Vec<f64>, u64)>,
    /// One captured (ffn input, ffn output) pair, for the control.
    pub control: Option<(usize, Vec<f32>, Vec<f32>)>,
    pending_ffn_input: Option<(usize, Vec<f32>)>,
}

impl larql_vindex::format::vindex3::opplan::exec::observe::StepObserver for MomentCollector {
    fn event(&mut self, _e: larql_vindex::format::vindex3::opplan::exec::observe::StepEvent) {}

    fn operand_input(
        &mut self,
        layer: usize,
        site: larql_vindex::format::vindex3::opplan::exec::observe::InputSite,
        values: &[f32],
    ) {
        use larql_vindex::format::vindex3::opplan::exec::observe::InputSite;
        let code = match site {
            InputSite::Attention => 0u8,
            InputSite::Ffn => 1,
            InputSite::FfnOutput => 2,
        };
        if site == InputSite::Ffn {
            self.pending_ffn_input = Some((layer, values.to_vec()));
        }
        if site == InputSite::FfnOutput {
            // Keep exactly one pair: the control needs a single position
            // where both sides are known, not a corpus.
            if self.control.is_none() {
                if let Some((l, input)) = self.pending_ffn_input.take() {
                    if l == layer {
                        self.control = Some((layer, input, values.to_vec()));
                    }
                }
            }
            // The FFN output is not an input site; it carries no moments.
            return;
        }
        let entry = self
            .sums
            .entry((layer, code))
            .or_insert_with(|| (vec![0.0; values.len()], 0));
        for (acc, v) in entry.0.iter_mut().zip(values) {
            *acc += (*v as f64) * (*v as f64);
        }
        entry.1 += 1;
    }
}
