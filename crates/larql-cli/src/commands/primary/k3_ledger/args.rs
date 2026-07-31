//! `larql k3-ledger` clap surface.

use clap::{Args, Subcommand};

#[derive(Debug, Args)]
pub struct K3LedgerArgs {
    #[command(subcommand)]
    pub cmd: K3LedgerCmd,

    /// Hugging Face repo to read geometry from.
    #[arg(long, global = true, default_value = "moonshotai/Kimi-K3")]
    pub repo: String,

    /// Shard holding a KDA (linear-attention) layer.
    #[arg(long, global = true, default_value = "model-00050-of-000096.safetensors")]
    pub kda_shard: String,

    /// Shard holding a full-attention (MLA) layer. NOTE the config's layer lists
    /// are 1-indexed against 0-indexed tensor names — config layer 4 is tensor
    /// layer 3. Picking wrong yields another KDA layer; the loader rejects it.
    #[arg(long, global = true, default_value = "model-00004-of-000096.safetensors")]
    pub mla_shard: String,

    /// Attainable memory bandwidth, GB/s (measured, not spec sheet).
    #[arg(long, global = true, default_value_t = 367.0)]
    pub bandwidth_gb_s: f64,

    /// Fraction of that bandwidth the kernel realises. Owner: the MXFP4 bench.
    #[arg(long, global = true, default_value_t = 1.0)]
    pub dequant_efficiency: f64,

    /// Emit the full record as JSON instead of a table.
    #[arg(long, global = true)]
    pub json: bool,
}

#[derive(Debug, Subcommand)]
pub enum K3LedgerCmd {
    /// DEC-8.0 — per-token fetch budget when experts live on external storage.
    Budget(BudgetArgs),
    /// DEC-8.6 — weight touch once resident, split reducible vs irreducible.
    Touch(TouchArgs),
    /// DEC-8.7a — dense-precision frontier; the target generator.
    Frontier(FrontierArgs),
    /// DEC-9.2 — speculative block economics, work-width separate from commit.
    Block(BlockArgs),
}

#[derive(Debug, Args)]
pub struct BudgetArgs {
    /// Link throughput per port, GB/s.
    #[arg(long, default_value_t = 3.5)]
    pub link_gb_s: f64,
    #[arg(long, default_value_t = 20.0)]
    pub target_tok_s: f64,
    #[arg(long, default_value_t = 1)]
    pub ports: u32,
    /// Feature fraction to price row-granular access at.
    #[arg(long, default_value_t = 0.064)]
    pub feature_fraction: f64,
    /// Measured routing-locality discount on the request rate.
    #[arg(long, default_value_t = 1.5)]
    pub locality: f64,
}

#[derive(Debug, Args)]
pub struct TouchArgs {
    #[arg(long, default_value_t = 20.0)]
    pub target_tok_s: f64,
    /// Demo-tier slice sizes to compose, in GB.
    #[arg(long, value_delimiter = ',', default_value = "55,60,65")]
    pub slice_gb: Vec<f64>,
}

#[derive(Debug, Args)]
pub struct FrontierArgs {
    #[arg(long, value_delimiter = ',', default_value = "6.6,8,10,12,14,20")]
    pub targets: Vec<f64>,
    /// Routed bytes still read. Defaults to the banked up-fold ceiling (2/3);
    /// anything lower is row sparsity, which is not yet measured.
    #[arg(long)]
    pub routed_retention: Option<f64>,
    /// Also run the R4 zero-out check: what the dense half alone permits.
    #[arg(long, default_value_t = true)]
    pub zero_out_check: bool,
}

#[derive(Debug, Args)]
pub struct BlockArgs {
    /// Positions the target evaluates per pass.
    #[arg(long, default_value_t = 7)]
    pub proposal_width: u32,
    /// Positions that commit, at that width. Must differ from the width unless
    /// perfect acceptance is genuinely intended (R5).
    #[arg(long, default_value_t = 3.85)]
    pub mean_accepted: f64,
    #[arg(long, value_delimiter = ',', default_value = "1,2,3,4,5,6,7,8,9,10,11,12")]
    pub widths: Vec<u32>,
    #[arg(long)]
    pub routed_retention: Option<f64>,
    #[arg(long, default_value_t = 20.0)]
    pub target_tok_s: f64,
    /// KDA state and transaction bytes per pass. Owner M1-B; zero means
    /// unmeasured, and the record says so rather than omitting the term.
    #[arg(long, default_value_t = 0.0)]
    pub state_bytes_per_pass: f64,
    /// KDA state bytes per *proposed* position (prefix-state retention). This is
    /// the term that grows with width while acceptance saturates, so omitting it
    /// inverts the sign of state traffic's effect on the optimum.
    #[arg(long, default_value_t = 0.0)]
    pub state_bytes_per_position: f64,
    /// Model execution as a naive token loop instead of assuming ideal grouping.
    #[arg(long)]
    pub token_loop: bool,
}
