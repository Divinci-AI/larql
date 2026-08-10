//! `larql vindex3` — VINDEX3 container programme verbs.
//!
//! `plan` is the G1 gate: a semantic representability check over one or
//! more artifacts, run *before* any conversion. It prints the full
//! [`larql_vindex::format::vindex3::plan::SystemPlan`] as JSON and exits
//! non-zero when the plan is inadmissible, so scripts can gate on it.

use std::path::{Path, PathBuf};

use clap::{Args, Subcommand};

use larql_models::inventory::{build_inventory, ArchitectureInventory};
use larql_vindex::format::vindex3::plan::plan_system;

/// Extension distinguishing a saved inventory JSON from a checkpoint dir.
const INVENTORY_EXT: &str = "json";

#[derive(Subcommand)]
pub enum Vindex3Command {
    /// Semantic representability plan over HF checkpoint dirs and/or saved
    /// `inspect-hf` inventory JSONs, treated together as one model system.
    /// Exits non-zero when the plan is inadmissible.
    Plan(PlanArgs),
}

#[derive(Args)]
pub struct PlanArgs {
    /// Checkpoint directories or inventory JSON files (one per artifact).
    #[arg(required = true)]
    pub artifacts: Vec<PathBuf>,

    /// Write the plan JSON here instead of stdout.
    #[arg(long)]
    pub output: Option<PathBuf>,
}

pub fn run(cmd: Vindex3Command) -> Result<(), Box<dyn std::error::Error>> {
    match cmd {
        Vindex3Command::Plan(args) => run_plan(args),
    }
}

/// Artifact display name: the file/directory stem.
fn artifact_name(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

/// Load one artifact: a `.json` file deserialises as a saved inventory;
/// anything else is inspected as a checkpoint directory.
fn load_artifact(path: &Path) -> Result<ArchitectureInventory, Box<dyn std::error::Error>> {
    if path.extension().is_some_and(|ext| ext == INVENTORY_EXT) {
        let text = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&text)?)
    } else {
        Ok(build_inventory(path)?)
    }
}

fn run_plan(args: PlanArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut named = Vec::new();
    for path in &args.artifacts {
        named.push((artifact_name(path), load_artifact(path)?));
    }
    let plan = plan_system(&named);
    let json = serde_json::to_string_pretty(&plan)?;
    match &args.output {
        Some(path) => {
            std::fs::write(path, &json)?;
            eprintln!("plan written to {}", path.display());
        }
        None => println!("{json}"),
    }
    let summary = &plan.summary;
    eprintln!(
        "plan: {} representable, {} mismatched, {} unrepresented, {} interfaces — {} blocking",
        summary.representable,
        summary.mismatched,
        summary.unrepresented,
        summary.interfaces,
        summary.blocking,
    );
    if plan.admissible {
        Ok(())
    } else {
        Err(format!(
            "plan not admissible: {} blocking finding(s); \
             every one is a schema or resolution gap to close before conversion",
            summary.blocking
        )
        .into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// `judged` controls whether the config carries a key no registry has
    /// seen — the difference between an admissible and a blocked plan.
    fn fixture_dir(judged: bool) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let mut text_config = serde_json::json!({
            "model_type": "muse_glimmer_text",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "intermediate_size": 256,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "qk_scale_factor": 3.87
        });
        if !judged {
            text_config["some_future_field_nobody_reviewed"] = serde_json::json!(1.5);
        }
        std::fs::write(
            dir.path().join("config.json"),
            serde_json::json!({
                "model_type": "muse_glimmer",
                "text_config": text_config
            })
            .to_string(),
        )
        .unwrap();
        let header = serde_json::json!({
            "model.layers.0.mlp.gate_proj.weight":
                {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 8]}
        });
        let header_bytes = serde_json::to_vec(&header).unwrap();
        let mut f = std::fs::File::create(dir.path().join("model.safetensors")).unwrap();
        f.write_all(&(header_bytes.len() as u64).to_le_bytes())
            .unwrap();
        f.write_all(&header_bytes).unwrap();
        dir
    }

    #[test]
    fn inadmissible_plan_is_an_error_and_still_writes_the_report() {
        let dir = fixture_dir(false);
        let out = dir.path().join("plan.json");
        let result = run(Vindex3Command::Plan(PlanArgs {
            artifacts: vec![dir.path().to_path_buf()],
            output: Some(out.clone()),
        }));
        assert!(result.is_err(), "an unjudged key must block");
        let plan: larql_vindex::format::vindex3::plan::SystemPlan =
            serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
        assert!(!plan.admissible);
        assert!(plan.summary.blocking > 0);
    }

    #[test]
    fn admissible_plan_exits_clean() {
        let dir = fixture_dir(true);
        let out = dir.path().join("plan.json");
        let result = run(Vindex3Command::Plan(PlanArgs {
            artifacts: vec![dir.path().to_path_buf()],
            output: Some(out.clone()),
        }));
        assert!(
            result.is_ok(),
            "fully judged artifact must pass: {result:?}"
        );
        let plan: larql_vindex::format::vindex3::plan::SystemPlan =
            serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
        assert!(plan.admissible);
        assert_eq!(plan.summary.blocking, 0);
    }

    #[test]
    fn accepts_a_saved_inventory_json() {
        let dir = fixture_dir(false);
        let inventory = larql_models::inventory::build_inventory(dir.path()).unwrap();
        let saved = dir.path().join("inventory.json");
        std::fs::write(&saved, serde_json::to_string(&inventory).unwrap()).unwrap();
        let out = dir.path().join("plan.json");
        let result = run(Vindex3Command::Plan(PlanArgs {
            artifacts: vec![saved],
            output: Some(out.clone()),
        }));
        assert!(result.is_err()); // same artifact, same verdict
        assert!(out.exists());
    }

    #[test]
    fn missing_artifact_errors_before_planning() {
        let result = run(Vindex3Command::Plan(PlanArgs {
            artifacts: vec![PathBuf::from("/nonexistent/model")],
            output: None,
        }));
        assert!(result.is_err());
    }
}
