//! `larql compile` — AOT compilation of vindex patches or single facts to
//! standard safetensors checkpoints. Output runs in any inference engine
//! without LARQL.
//!
//! Three modes:
//! - **Single** (`--prompt` + `--answer`): one compiled edge from a prompt's
//!   residual at `--layer`, writing the answer token. CLI-driven; used for
//!   the pi/Gauss demos and any prompt→answer pair.
//! - **Menu** (`--menu path.json`): batch of prompt/answer pairs, each gets
//!   its own edge at auto-incrementing slots starting from `--slot`. One
//!   compile command, K edges. Gives the "variable answer per prompt"
//!   demo when each entry's answer comes from a bounded-compute kernel run
//!   at menu-generation time.
//! - **Patch** (`--vindex`): replays .vlp patch files into the model's FFN
//!   slots. `insert` installs an edge; `delete` tombstones the slot (zeroes
//!   gate row, up row, down column — the weight-level twin of the served
//!   overlay's tombstone, so the checkpoint behaves as the overlay does in
//!   any engine). Vindex-driven; many edges per run.
//!
//! ## The compiled checkpoint must carry the whole patch set, or refuse
//!
//! Until 2026-09-15 patch mode matched `PatchOp::Insert` and fell through
//! every other op with a bare `continue` — no message — then printed
//! "Compiling patches into weights..." and wrote the file. A patch set of
//! only DELETE ops therefore compiled to a checkpoint byte-identical to the
//! base and exited 0. That checkpoint is the artifact a third party is
//! handed to test an erasure independently, so it was the unmodified model
//! presented as the erased one.
//!
//! The rule now: every op either lands in the weights or fails the run,
//! **before** anything is written. `insert_knn` / `delete_knn` are a
//! post-logits mechanism with no FFN-weight expression and are refused by
//! name; `update` is refused until it has a compile path. An `insert` the
//! compiler cannot place (no gate vector, target out of vocab) fails too,
//! unless `--allow-partial` is passed — and then the summary says how many
//! ops the checkpoint does NOT carry.
//!
//! The install primitive in [`edge::install_edge`] mirrors the convention
//! described in `~/chris-source/chris-experiments/foundations/07_wasm_compute/WASM_GATE_ARCHITECTURE.md` §3.1.2.

use std::path::PathBuf;

use clap::Args;

mod byte_patch;
mod chat;
mod constraints;
mod detect;
mod edge;
mod patch;
mod save;
mod single;

#[derive(Args)]
pub struct CompileArgs {
    /// Path to the base model (directory with safetensors, or HF model ID).
    #[arg(long)]
    pub base: PathBuf,

    /// Path to the vindex (with patches to compile). Not needed for fact mode.
    #[arg(long)]
    pub vindex: Option<PathBuf>,

    /// Output directory for the compiled model safetensors.
    #[arg(short, long)]
    pub output: PathBuf,

    /// Gate scale for compiled edges (default: 1.0).
    /// Previous default 30.0 saturated silu on every question prompt and
    /// leaked the edge into unrelated queries; 1.0 keeps natural usage
    /// clean on Gemma 3 4B. See ~/chris-source/chris-experiments/foundations/07_wasm_compute/RESULTS.md.
    #[arg(long, default_value = "1.0")]
    pub gate_scale: f32,

    /// Alpha multiplier for initial write magnitude (default: 0.3).
    /// The balancer (single mode) refines this after install by scaling
    /// the down vector up/down until the target-token probability lands
    /// in [--floor, --ceiling].
    #[arg(long, default_value = "0.3")]
    pub alpha: f32,

    // ── Balancer options (single mode only) ─────────────────────
    /// Minimum probability the target token must reach before the
    /// balancer stops scaling up the down vector.
    #[arg(long, default_value = "0.40")]
    pub floor: f64,

    /// Maximum probability the target token may reach before the
    /// balancer starts scaling down. Too-confident installs over-ride
    /// context and regress unrelated prompts.
    #[arg(long, default_value = "0.85")]
    pub ceiling: f64,

    /// Maximum balancer iterations. Default 0 — the balancer is opt-in
    /// because `larql_inference::forward::predict` is systematically
    /// "peakier" than HF transformers' forward pass on the same weights,
    /// so scaling the down vector to reach [floor, ceiling] in Rust's
    /// simulation over-dampens the edge relative to deployed inference.
    /// Leaving this at 0 installs at --alpha / --gate-scale and trusts
    /// the caller's pre-tuned defaults (the paraphrase-sweep sweet spot:
    /// g=1.0, α=0.3). Set --max-iters >0 only if you have reason to
    /// believe Rust's predict tracks HF for your model.
    #[arg(long, default_value = "0")]
    pub max_iters: u32,

    /// Skip applying the base model's `tokenizer_config.json::chat_template`
    /// to the prompt before tokenising. By default the template is loaded
    /// from the base model and rendered (so the trigger residual captured
    /// here matches what a served/chat-wrapped deployment will produce).
    /// Only set this for raw-prompt experiments.
    #[arg(long, default_value = "false")]
    pub no_chat_template: bool,

    // ── Fact compilation mode ─────────────────────────────────
    /// Prompt text whose residual becomes the trigger direction.
    #[arg(long)]
    pub prompt: Option<String>,

    /// Correct answer token to compile into the weights.
    #[arg(long)]
    pub answer: Option<String>,

    /// Layer to install the compiled edge at (default: 30).
    #[arg(long, default_value = "30")]
    pub layer: usize,

    /// FFN slot to install the compiled edge at (default: 9000).
    #[arg(long, default_value = "9000")]
    pub slot: usize,

    /// A control prompt and the answer it must still give AFTER the edge is installed,
    /// written `"prompt=>expected"`. Repeatable.
    ///
    /// `install_edge` writes at whatever magnitude --alpha asks for and nothing downstream used
    /// to ask what that cost. Measured 2026-09-20 on gemma-4-E2B-it: at alpha 3 the compiled
    /// checkpoint answers the installed fact AND has forgotten the capital of Italy, and this
    /// command reported success. Each control is measured before and after the install; if any
    /// of them changes its top-1 answer, NOTHING IS WRITTEN and the command exits non-zero.
    ///
    /// Top-1 rather than a probability threshold on purpose: `forward::predict` is systematically
    /// peakier than HF's forward pass (see --max-iters), and argmax survives that calibration gap
    /// where a probability bar does not.
    #[arg(long = "control", value_name = "PROMPT=>ANSWER")]
    pub controls: Vec<String>,

    /// Write the compiled checkpoint as a BYTE-PATCHED COPY of the base file instead of
    /// re-serialising a standalone text-only model.
    ///
    /// The default writer produces an artifact to SERVE: multimodal towers dropped, config
    /// rewritten to a text architecture, every tensor re-serialised from the in-memory f32
    /// representation. It shares no tensor names with its base, so nothing about it can be
    /// compared byte for byte with the checkpoint it came from.
    ///
    /// This produces an artifact to AUDIT: the base file with only the edited slot's gate row,
    /// up row and down column overwritten. Header, tensor set, names, dtypes and every other
    /// byte unchanged, so the compiled checkpoint is the base plus a named, bounded, reversible
    /// difference — which is what lets a third party verify a later deletion by recomputing one
    /// hash instead of trusting the tool that performed it.
    #[arg(long, default_value = "false")]
    pub byte_patch: bool,

    /// Patch mode only. Write the checkpoint even if some `insert` ops
    /// could not be placed (no gate vector, target token out of vocab).
    /// Off by default: a checkpoint that carries fewer edits than its
    /// patch set is a misrepresentation of what was compiled, and the
    /// default refuses to produce one. Ops the compiler has NO path for
    /// (`insert_knn`, `delete_knn`, `update`) are refused regardless.
    #[arg(long, default_value = "false")]
    pub allow_partial: bool,
}

pub fn run(args: CompileArgs) -> Result<(), Box<dyn std::error::Error>> {
    if args.prompt.is_some() && args.answer.is_some() {
        return single::run(args);
    }
    if args.vindex.is_none() {
        return Err("either --vindex or --prompt + --answer required".into());
    }
    patch::run(args)
}
