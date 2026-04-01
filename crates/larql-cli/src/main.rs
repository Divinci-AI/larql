use clap::{Parser, Subcommand};

mod commands;
mod formatting;
mod utils;

use commands::extraction::*;
use commands::query::*;
use commands::surreal::*;

#[derive(Parser)]
#[command(
    name = "larql",
    version,
    about = "LARQL knowledge graph extraction and querying"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    // ── Extraction ──
    /// Extract edges from FFN weights. Zero forward passes.
    WeightExtract(weight_walk_cmd::WeightWalkArgs),

    /// Extract routing edges from attention OV circuits. Zero forward passes.
    AttentionExtract(attention_walk_cmd::AttentionWalkArgs),

    /// Extract full vectors from model weights to NDJSON files.
    VectorExtract(vector_extract_cmd::VectorExtractArgs),

    /// Capture residual stream vectors for entities via forward passes.
    Residuals(residuals_cmd::ResidualsArgs),

    /// Run full forward pass and predict next token.
    Predict(predict_cmd::PredictArgs),

    /// Build gate index for graph-based FFN (offline, run once per model).
    IndexGates(index_gates_cmd::IndexGatesArgs),

    /// Extract attention routing patterns from forward passes.
    ExtractRoutes(extract_routes_cmd::ExtractRoutesArgs),

    /// Walk the model as a local vector index — gate KNN + down token lookup.
    Walk(walk_cmd::WalkArgs),

    /// Capture and compare attention patterns across prompts.
    AttentionCapture(attention_capture_cmd::AttentionCaptureArgs),

    /// Extract attention template circuits from QK weight decomposition.
    QkTemplates(qk_templates_cmd::QkTemplatesArgs),

    /// SVD rank analysis of attention QK products — how many modes per head.
    QkRank(qk_rank_cmd::QkRankArgs),

    /// Extract interpretable modes from low-rank QK heads via SVD → gate projection.
    QkModes(qk_modes_cmd::QkModesArgs),

    /// Map attention OV circuits to FFN gate features (what each head activates).
    OvGate(ov_gate_cmd::OvGateArgs),

    /// Discover attention→FFN circuits from weight decomposition. No forward passes.
    CircuitDiscover(circuit_discover_cmd::CircuitDiscoverArgs),

    /// Bottleneck analysis of attention components.
    AttnBottleneck(attn_bottleneck_cmd::AttnBottleneckArgs),

    /// Benchmark FFN performance: dense vs sparse at various K values.
    FfnBench(ffn_bench_cmd::FfnBenchArgs),

    /// Bottleneck analysis of FFN components.
    FfnBottleneck(ffn_bottleneck_cmd::FfnBottleneckArgs),

    /// Measure overlap between entity-routed and ground-truth gate features.
    FfnOverlap(ffn_overlap_cmd::FfnOverlapArgs),

    /// Knowledge graph retrieval benchmark — zero matmul entity lookup.
    KgBench(kg_bench_cmd::KgBenchArgs),

    /// Measure FFN throughput: tokens/second at various access patterns.
    FfnThroughput(ffn_throughput_cmd::FfnThroughputArgs),

    /// Build a .vindex — the model decompiled to a standalone vector index.
    ExtractIndex(extract_index_cmd::ExtractIndexArgs),

    /// Build a custom model from a Vindexfile (declarative: FROM + PATCH + INSERT).
    Build(build_cmd::BuildArgs),

    /// Graph-routed walk with banded layer strategy (graph FFN + highway skip).
    GraphWalk(graph_walk_cmd::GraphWalkArgs),

    /// Trace residual stream trajectories on the sphere across layers.
    TrajectoryTrace(trajectory_trace_cmd::TrajectoryTraceArgs),

    /// Benchmark vindex walk: accuracy vs dense, throughput.
    VindexBench(vindex_bench_cmd::VindexBenchArgs),

    /// Test rank-k projection: replace L0→L_inject with a linear map, run the rest dense.
    ProjectionTest(projection_test_cmd::ProjectionTestArgs),

    /// Extract OV fingerprint basis from attention weights (zero forward passes).
    FingerprintExtract(fingerprint_extract_cmd::FingerprintExtractArgs),

    /// Test rule-based bottleneck: 9 if-else rules replace L0-13, run L14-33 dense.
    BottleneckTest(bottleneck_test_cmd::BottleneckTestArgs),

    /// Embedding jump: raw token embeddings → projected L13 → decoder. Zero layers for L0-13.
    EmbeddingJump(embedding_jump_cmd::EmbeddingJumpArgs),

    /// BFS extraction from a model endpoint.
    Bfs(bfs_cmd::BfsArgs),

    // ── SurrealDB ──
    /// Load vectors into SurrealDB with HNSW indexes (small tables, HTTP).
    VectorLoad(vector_load_cmd::VectorLoadArgs),

    /// Import vectors into SurrealDB via batched `surreal import` (large tables).
    VectorImport(vector_import_cmd::VectorImportArgs),

    /// Export vectors to .surql files for manual import.
    VectorExportSurql(vector_export_surql_cmd::VectorExportSurqlArgs),

    /// Load OV→gate coupling edges into SurrealDB for circuit discovery.
    CouplingLoad(coupling_load_cmd::CouplingLoadArgs),

    // ── Query ──
    /// Query a graph for facts.
    Query(query_cmd::QueryArgs),

    /// Describe an entity (all edges).
    Describe(describe_cmd::DescribeArgs),

    /// Show graph statistics.
    Stats(stats_cmd::StatsArgs),

    /// Validate a graph file.
    Validate(validate_cmd::ValidateArgs),

    /// Merge multiple graph files.
    Merge(merge_cmd::MergeArgs),

    /// Filter graph edges by confidence, layer, selectivity, relation, source, etc.
    Filter(filter_cmd::FilterArgs),

    // ── LQL ──
    /// Launch the LQL interactive REPL.
    Repl,

    /// Execute an LQL statement.
    Lql(LqlArgs),
}

#[derive(clap::Args)]
struct LqlArgs {
    /// LQL statement to execute (e.g., 'WALK "The capital of France is" TOP 5;')
    statement: String,
}

fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        // Extraction
        Commands::WeightExtract(args) => weight_walk_cmd::run(args),
        Commands::AttentionExtract(args) => attention_walk_cmd::run(args),
        Commands::VectorExtract(args) => vector_extract_cmd::run(args),
        Commands::Residuals(args) => residuals_cmd::run(args),
        Commands::Predict(args) => predict_cmd::run(args),
        Commands::IndexGates(args) => index_gates_cmd::run(args),
        Commands::AttentionCapture(args) => attention_capture_cmd::run(args),
        Commands::QkTemplates(args) => qk_templates_cmd::run(args),
        Commands::QkRank(args) => qk_rank_cmd::run(args),
        Commands::QkModes(args) => qk_modes_cmd::run(args),
        Commands::OvGate(args) => ov_gate_cmd::run(args),
        Commands::CircuitDiscover(args) => circuit_discover_cmd::run(args),
        Commands::ExtractRoutes(args) => extract_routes_cmd::run(args),
        Commands::Walk(args) => walk_cmd::run(args),
        Commands::AttnBottleneck(args) => attn_bottleneck_cmd::run(args),
        Commands::FfnBench(args) => ffn_bench_cmd::run(args),
        Commands::FfnBottleneck(args) => ffn_bottleneck_cmd::run(args),
        Commands::FfnOverlap(args) => ffn_overlap_cmd::run(args),
        Commands::KgBench(args) => kg_bench_cmd::run(args),
        Commands::FfnThroughput(args) => ffn_throughput_cmd::run(args),
        Commands::ExtractIndex(args) => extract_index_cmd::run(args),
        Commands::Build(args) => build_cmd::run(args),
        Commands::GraphWalk(args) => graph_walk_cmd::run(args),
        Commands::TrajectoryTrace(args) => trajectory_trace_cmd::run(args),
        Commands::VindexBench(args) => vindex_bench_cmd::run(args),
        Commands::ProjectionTest(args) => projection_test_cmd::run(args),
        Commands::FingerprintExtract(args) => fingerprint_extract_cmd::run(args),
        Commands::BottleneckTest(args) => bottleneck_test_cmd::run(args),
        Commands::EmbeddingJump(args) => embedding_jump_cmd::run(args),
        Commands::Bfs(args) => bfs_cmd::run(args),
        // SurrealDB
        Commands::VectorLoad(args) => vector_load_cmd::run(args),
        Commands::VectorImport(args) => vector_import_cmd::run(args),
        Commands::VectorExportSurql(args) => vector_export_surql_cmd::run(args),
        Commands::CouplingLoad(args) => coupling_load_cmd::run(args),
        // Query
        Commands::Query(args) => query_cmd::run(args),
        Commands::Describe(args) => describe_cmd::run(args),
        Commands::Stats(args) => stats_cmd::run(args),
        Commands::Validate(args) => validate_cmd::run(args),
        Commands::Merge(args) => merge_cmd::run(args),
        Commands::Filter(args) => filter_cmd::run(args),
        // LQL
        Commands::Repl => {
            larql_lql::run_repl();
            Ok(())
        }
        Commands::Lql(args) => {
            match larql_lql::run_statement(&args.statement) {
                Ok(lines) => {
                    for line in &lines {
                        println!("{line}");
                    }
                    Ok(())
                }
                Err(e) => Err(e),
            }
        }
    };

    if let Err(e) = result {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
