//! `larql apply-patch` — load a `.lqpatch` file and apply it to a model.
//!
//! Non-destructive: modifies the in-memory `ModelWeights`, does not write
//! back to the model directory. Optional `--prompt` runs a prediction with
//! the patch active so users can verify the edit.

use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use larql_inference::{
    edit::{apply_patch, read_patch},
    forward::predict,
    InferenceModel,
};

#[derive(Args)]
pub struct ApplyPatchArgs {
    /// Model path or HuggingFace model ID.
    model: String,

    /// One or more `.lqpatch` files to apply in order. Later patches sum
    /// atop earlier ones — safe when each edit targets a different key.
    #[arg(short, long, num_args = 1.., required = true)]
    patch: Vec<PathBuf>,

    /// Optional prompt — run `predict` after applying and print the top-k.
    #[arg(long)]
    prompt: Option<String>,

    /// Top-k for optional prediction.
    #[arg(short = 'k', long, default_value = "5")]
    top_k: usize,

    /// Reverse the patch(es) (subtract instead of add). Verifies the
    /// edit is reversible and produces the original behaviour.
    #[arg(long)]
    reverse: bool,
}

pub fn run(args: ApplyPatchArgs) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Loading model: {}", args.model);
    let t0 = Instant::now();
    let mut model = InferenceModel::load(&args.model)?;
    eprintln!(
        "  {} layers ({:.1}s)",
        model.num_layers(),
        t0.elapsed().as_secs_f64()
    );

    for patch_path in &args.patch {
        eprintln!("Reading patch: {}", patch_path.display());
        let mut patch = read_patch(patch_path)?;
        if args.reverse {
            for v in patch.d.iter_mut() {
                *v = -*v;
            }
        }
        eprintln!(
            "  layer=L{}  module={}  scale={:.2}  hidden={}  intermediate={}",
            patch.layer, patch.module, patch.scale, patch.hidden_size, patch.intermediate_size
        );

        // SAFETY: we mutate ModelWeights in-place via the public field.
        apply_patch(model.weights_mut(), &patch).map_err(|e| format!("apply_patch: {e}"))?;
        eprintln!("  applied{}.", if args.reverse { " (reversed)" } else { "" });
    }

    if let Some(prompt) = args.prompt {
        eprintln!("\nPrediction under applied patch(es):");
        let encoding = model
            .tokenizer()
            .encode(prompt.as_str(), true)
            .map_err(|e| format!("tokenize: {e}"))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();
        let result = predict(model.weights(), model.tokenizer(), &token_ids, args.top_k);
        for (i, (tok, prob)) in result.predictions.iter().enumerate() {
            eprintln!("  {:>2}. {:<20} {:.3}", i + 1, tok, prob);
        }
    }

    Ok(())
}
