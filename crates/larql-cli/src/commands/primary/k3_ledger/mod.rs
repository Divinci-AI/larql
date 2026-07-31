//! `larql k3-ledger` — checkpoint-derived serving arithmetic (docs/dec-funnel.md).
//!
//! Answers what a K3-class model costs to serve, from the checkpoint's own
//! tensor table rather than from estimates. Every rung is zero-compute: two HTTP
//! range requests against a 1.5 TB repository, then division.
//!
//! Module map:
//!
//!   args      — clap surface (`budget` / `touch` / `frontier` / `block`).
//!   geometry  — measured checkpoint shape; nothing derived (pure, gated).
//!   budget    — DEC-8.0 miss budget and read-granularity (pure, gated).
//!   touch     — DEC-8.6 resident weight-touch ledger (pure, gated).
//!   frontier  — DEC-8.7a dense-precision frontier, R4 ceiling (pure, gated).
//!   block     — DEC-9.2 speculative block economics, R5/R6 (pure, gated).
//!   fetch     — HTTP range-GET geometry loading (I/O, excluded).
//!   report    — human-readable rendering (I/O-adjacent, excluded).
//!
//! The pure/runtime split is load-bearing here. Every error this ladder caught
//! was arithmetic over correct measurements, so the arithmetic carries unit
//! tests and the network does not.

pub mod args;
pub mod block;
pub mod budget;
pub mod classes;
pub mod fetch;
pub mod frontier;
pub mod geometry;
mod report;
pub mod touch;
pub mod transcode;

pub use args::K3LedgerArgs;

pub fn run(args: K3LedgerArgs) -> Result<(), Box<dyn std::error::Error>> {
    let repo = fetch::Repo::new(&args.repo)?;
    eprintln!(
        "reading geometry from {} (headers only, no weights)",
        args.repo
    );
    let geom = fetch::load_geometry(&repo, &args.kda_shard, &args.mla_shard)?;

    let premises = frontier::ServingPremises {
        bandwidth_gb_s: args.bandwidth_gb_s,
        dequant_efficiency: args.dequant_efficiency,
        ..Default::default()
    };

    match &args.cmd {
        args::K3LedgerCmd::Budget(a) => report::budget(&geom, a, args.json),
        args::K3LedgerCmd::Touch(a) => report::touch(&geom, &premises, a, args.json),
        args::K3LedgerCmd::Frontier(a) => report::frontier(&geom, &premises, a, args.json),
        args::K3LedgerCmd::Block(a) => report::block(&geom, &premises, a, args.json),
        args::K3LedgerCmd::Ceilings(a) => report::ceilings(&geom, a, args.json),
        args::K3LedgerCmd::TranscodeScan(a) => {
            report::transcode_scan(&repo, &args.kda_shard, a, args.json)
        }
    }
}
