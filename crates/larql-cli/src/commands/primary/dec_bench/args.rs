//! Clap args for `larql dec-bench` — the DEC residual-replay loadgen
//! (docs/dec-funnel.md §3, DEC-0). Two sub-modes:
//!
//!   * `capture` — run single-stream decode over N prompts against a live
//!     expert server, recording each step's pre-normed per-layer residuals
//!     into a portable pool.
//!   * `replay`  — sweep batch × wire × dispatch against the server using
//!     B-row frames built from the pool; emit the `dec/*` schema.

use clap::{Args, Subcommand};

#[derive(Args, Clone)]
pub struct DecBenchArgs {
    #[command(subcommand)]
    pub cmd: DecBenchCmd,
}

#[derive(Subcommand, Clone)]
pub enum DecBenchCmd {
    /// Capture a residual pool from live decode over N prompts.
    Capture(CaptureArgs),
    /// Replay the pool as B-row requests, sweeping batch × wire × dispatch.
    Replay(ReplayArgs),
}

#[derive(Args, Clone)]
pub struct CaptureArgs {
    /// Vindex directory, `hf://owner/name`, or cache shorthand.
    pub model: String,

    /// Expert-server URL (loopback for DEC-0 arm M).
    #[arg(long)]
    pub ffn: String,

    /// Prompts file, one prompt per line (fixed set, checked in).
    #[arg(long)]
    pub prompts_file: std::path::PathBuf,

    /// Number of prompts to capture (= max replay batch size).
    #[arg(long, default_value = "64")]
    pub num_prompts: usize,

    /// Decode steps to capture per prompt.
    #[arg(long, default_value = "16")]
    pub steps: usize,

    /// Use the Metal GPU attention backend (required on macOS for the
    /// remote-FFN decode path).
    #[arg(long)]
    pub metal: bool,

    /// Output pool directory (`manifest.json` + `residuals.bin`, plus
    /// `raw.bin`/`normed.bin`/`routing.bin` with `--routing`).
    #[arg(long)]
    pub out: std::path::PathBuf,

    /// Also capture the routed-experts sidecars (raw + pre-experts-normed
    /// residual planes and per-layer top-k expert routing) for the DEC
    /// routed replay arm. `top_k` comes from the model arch; errors if the
    /// model has no MoE. Without this flag the capture is byte-identical to
    /// the walk-ffn-only pool format.
    #[arg(long)]
    pub routing: bool,

    /// Per-request timeout for the expert server.
    #[arg(long, default_value = "60")]
    pub ffn_timeout_secs: u64,

    #[arg(short, long)]
    pub verbose: bool,
}

#[derive(Args, Clone)]
pub struct ReplayArgs {
    /// Expert-server URL.
    #[arg(long)]
    pub ffn: String,

    /// Capture pool directory (from `dec-bench capture`).
    #[arg(long)]
    pub capture: std::path::PathBuf,

    /// Server endpoint family: `walk-ffn` (dense/shared-expert FFN path) or
    /// `experts` (routed multi-layer expert path — needs a pool captured
    /// with `--routing`; wire axis limited to f32/q8k, mapping to
    /// /v1/experts/multi-layer-batch[-q8k]).
    #[arg(long, default_value = "walk-ffn")]
    pub endpoint: String,

    /// Comma-separated batch sizes (rows per request).
    #[arg(long, default_value = "1,8,16,32,64")]
    pub batch: String,

    /// Comma-separated wire arms: f32, f16, i8, q8k (symmetric — unchanged
    /// behaviour: f32 request frames, Accept per arm), or asymmetric
    /// in/return pairs like `f16/i8`, `i8/f16` (request Content-Type and
    /// Accept set independently; walk-ffn endpoint only — q8k is its own
    /// endpoint and cannot pair). i8 returns need LARQL_I8_WIRE=1 on the
    /// server or the return direction falls back (recorded in
    /// served_wire_out).
    #[arg(long, default_value = "f32,f16,i8,q8k")]
    pub wire: String,

    /// Comma-separated dispatch modes: streaming (sequential per-layer),
    /// batch (all layers fired in parallel, mirrors predispatch).
    #[arg(long, default_value = "streaming,batch")]
    pub dispatch: String,

    /// Inclusive layer range `A-B` to replay (default: all layers).
    #[arg(long)]
    pub layers: Option<String>,

    /// Steps to replay per repeat (default: all captured steps).
    #[arg(long)]
    pub steps: Option<usize>,

    /// Repeats per sweep point.
    #[arg(long, default_value = "3")]
    pub repeats: usize,

    /// Warmup passes over all replayed layers per sweep point (discarded).
    /// Full passes, not single requests — mmap-backed layer weights pay
    /// first-touch page faults per layer.
    #[arg(long, default_value = "1")]
    pub warmup_passes: usize,

    /// Per-request timeout.
    #[arg(long, default_value = "60")]
    pub timeout_secs: u64,

    /// Write the full JSON run record here.
    #[arg(long)]
    pub output_file: Option<std::path::PathBuf>,

    /// Write `dec/*` pulse JSONL here (rig `$CHUK_METRICS` format).
    #[arg(long)]
    pub pulse_file: Option<std::path::PathBuf>,

    /// Include per-layer p50/p99 keys in the pulse lines (always present in
    /// the JSON run record).
    #[arg(long)]
    pub pulse_per_layer: bool,

    /// Measured link RTT to record alongside the run (spec §6 gate).
    #[arg(long)]
    pub net_rtt_ms: Option<f64>,

    /// Measured link bandwidth to record alongside the run (spec §6 gate).
    #[arg(long)]
    pub net_gbps: Option<f64>,

    #[arg(short, long)]
    pub verbose: bool,
}
