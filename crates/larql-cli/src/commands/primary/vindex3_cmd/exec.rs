//! `larql vindex3 exec` — run a container's own program (V3-G5b-3c).
//!
//! Research-oriented on purpose. The first useful mode is not chat: it is
//! a layer-by-layer hidden-state dump in exactly the format
//! `larql shannon layer-dump` writes, so `larql shannon layer-diff`
//! compares a VINDEX3 execution against an upstream `transformers` trace
//! with **no new comparator**. A divergence localises to a layer before
//! anyone asks what the model said.
//!
//! Token ids are given explicitly rather than tokenised here. A tokenizer
//! is part of the fixture, and only one side of a parity comparison may
//! choose it — `scripts/capture_glimmer_oracle.py` already recorded the
//! ids this reads back.
//!
//! The backend is a flag over the same plan. That is the point of the
//! seam: `--backend reference` and `--backend production` execute one
//! program through two numerical realisations, and their dumps are
//! directly diffable against each other as well as against upstream.

use std::path::Path;

use larql_vindex::format::vindex3::inspect::inspect_container;
use larql_vindex::format::vindex3::opplan::exec::operands::OperandStore;
use larql_vindex::format::vindex3::opplan::exec::production::ProductionBackend;
use larql_vindex::format::vindex3::opplan::exec::reference::ReferenceBackend;
use larql_vindex::format::vindex3::opplan::exec::{execute_plan, ExecutionTrace};
use larql_vindex::format::vindex3::opplan::plan_component_ops;
use ndarray::Array2;

use super::super::shannon_trace::dump::{
    plane_name, write_plane, LayerDumpManifest, MANIFEST_NAME, PLANE_DTYPE,
};
use super::{ExecArgs, ExecBackend};

/// Extra planes beyond the layer table, matching
/// `scripts/capture_glimmer_oracle.py`.
const FINAL_NORM_PLANE: &str = "final_norm.f32";
const LOGITS_PLANE: &str = "logits.f32";

/// Engine tag prefix; the backend name completes it so a dump can never
/// be mistaken for one produced by the other realisation.
const ENGINE_PREFIX: &str = "vindex3";

pub fn run_exec(args: ExecArgs) -> Result<(), Box<dyn std::error::Error>> {
    let tokens = parse_tokens(&args.tokens)?;
    let inspection = inspect_container(&args.container, false)?;
    let outcome = plan_component_ops(&inspection, &args.container, &args.component)?;
    if !outcome.defects.is_empty() {
        for defect in &outcome.defects {
            eprintln!("defect: {defect}");
        }
        return Err(format!(
            "component `{}` does not close: {} defect(s)",
            args.component,
            outcome.defects.len()
        )
        .into());
    }
    let plan = outcome
        .plan
        .ok_or_else(|| format!("component `{}` produced no plan", args.component))?;
    let store = OperandStore::open(&args.container, &inspection)?;

    let (engine, trace) = match args.backend {
        ExecBackend::Reference => {
            let backend = ReferenceBackend::new();
            let name = format!("{ENGINE_PREFIX}-{}", backend_name(&backend));
            (name, execute_plan(&plan, &store, &tokens, &backend)?)
        }
        ExecBackend::Production => {
            let backend = ProductionBackend::new();
            let name = format!("{ENGINE_PREFIX}-{}", backend_name(&backend));
            (name, execute_plan(&plan, &store, &tokens, &backend)?)
        }
    };

    match &args.dump_layers {
        Some(dir) => {
            write_dump(dir, &engine, &args, &tokens, &trace)?;
            eprintln!(
                "wrote {} planes + final norm + logits to {}",
                trace.layers.len() + 1,
                dir.display()
            );
        }
        None => summarise(&engine, &trace),
    }
    Ok(())
}

/// Read the backend's own name through the trait, so the engine tag
/// cannot drift from the implementation that produced the numbers.
fn backend_name<B: larql_vindex::format::vindex3::opplan::exec::backend::PlanBackend>(
    backend: &B,
) -> String {
    backend.name().to_string()
}

/// Parse a comma-separated token list.
fn parse_tokens(spec: &str) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let tokens: Result<Vec<u32>, _> = spec
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::parse::<u32>)
        .collect();
    let tokens = tokens.map_err(|e| format!("--tokens must be comma-separated ids: {e}"))?;
    if tokens.is_empty() {
        return Err("--tokens is empty".into());
    }
    Ok(tokens)
}

/// One `[seq, hidden]` plane from a per-position row list.
fn plane_of(rows: &[Vec<f32>]) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
    let seq = rows.len();
    let hidden = rows.first().map(Vec::len).unwrap_or(0);
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();
    Ok(Array2::from_shape_vec((seq, hidden), flat)?)
}

/// Write the dump directory: plane 000 is the residual entering layer 0,
/// plane `i + 1` the residual leaving layer `i` — the same convention
/// `forward_hidden_all_layers` and `dump_layers_hf.py` use.
fn write_dump(
    dir: &Path,
    engine: &str,
    args: &ExecArgs,
    tokens: &[u32],
    trace: &ExecutionTrace,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(dir)?;
    let mut planes = Vec::with_capacity(trace.layers.len() + 1);

    let entering = plane_of(&trace.embedded)?;
    let (seq_len, hidden_size) = (entering.shape()[0], entering.shape()[1]);
    let name = plane_name(0);
    write_plane(&dir.join(&name), &entering)?;
    planes.push(name);

    for (index, layer) in trace.layers.iter().enumerate() {
        let name = plane_name(index + 1);
        write_plane(&dir.join(&name), &plane_of(&layer.post_layer)?)?;
        planes.push(name);
    }

    // Extras beyond the layer table: the final norm and the head. Written
    // as `[1, n]` because both are last-position only.
    write_plane(
        &dir.join(FINAL_NORM_PLANE),
        &plane_of(std::slice::from_ref(&trace.final_hidden))?,
    )?;
    if let Some(logits) = &trace.logits {
        write_plane(
            &dir.join(LOGITS_PLANE),
            &plane_of(std::slice::from_ref(logits))?,
        )?;
    }

    let manifest = LayerDumpManifest {
        engine: engine.to_string(),
        model: args.container.display().to_string(),
        num_layers: trace.layers.len(),
        seq_len,
        hidden_size,
        token_ids: tokens.to_vec(),
        planes,
        dtype: PLANE_DTYPE.to_string(),
    };
    std::fs::write(
        dir.join(MANIFEST_NAME),
        serde_json::to_string_pretty(&manifest)?,
    )?;
    Ok(())
}

/// Without `--dump-layers`, print enough to see the forward ran.
fn summarise(engine: &str, trace: &ExecutionTrace) {
    println!("engine: {engine}");
    println!(
        "layers: {}  seq: {}  hidden: {}",
        trace.layers.len(),
        trace.embedded.len(),
        trace.embedded.first().map(Vec::len).unwrap_or(0),
    );
    match &trace.logits {
        Some(logits) => {
            let (best, value) =
                logits
                    .iter()
                    .enumerate()
                    .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, v)| {
                        if *v > bv {
                            (i, *v)
                        } else {
                            (bi, bv)
                        }
                    });
            println!("logits: {}, argmax {best} ({value:+.4})", logits.len());
        }
        None => println!("logits: none (plan carries no output head)"),
    }
}
