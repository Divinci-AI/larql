#!/usr/bin/env bash
# DEC-0.5 — x86 expert-tier kernel gate (docs/dec-funnel.md v0.5 §3).
#
# Runs ON the rented x86 box (Ubuntu, ≥128GB RAM). Turnkey: clone/build,
# fetch the expert-server vindex slice + replay pools from HF, run the
# per-core kernel bench + the serving-level replay points, print the
# gate inputs. Designed to fit inside a ≤3h lease with time to spare.
#
# Prereqs on the box: git, curl, build-essential, pkg-config, libssl-dev,
# python3-pip (for huggingface_hub). Rust installed by this script.
#
# Usage:
#   HF_TOKEN=hf_... ./scripts/dec0p5-x86.sh [COMMIT]
#
# Outputs under ./dec0p5-out/: kernel bench log, replay run records +
# pulses (dense q8k B8 + routed experts-ml-q8k B8), kernel_class line,
# lscpu + core count — everything the registry run record needs.

set -euo pipefail

COMMIT="${1:-main}"
OUT="$PWD/dec0p5-out"
URL=http://127.0.0.1:8080
mkdir -p "$OUT"

echo "[dec0p5] host: $(uname -mo); cores: $(nproc)"
lscpu > "$OUT/lscpu.txt" 2>/dev/null || true

if ! command -v cargo >/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    . "$HOME/.cargo/env"
fi

if [ ! -d larql ]; then git clone --depth 50 https://github.com/chrishayuk/larql.git; fi
cd larql && git fetch --depth 50 origin "$COMMIT" && git checkout "$COMMIT"
git rev-parse HEAD > "$OUT/commit.txt"

pip3 install -q huggingface_hub
python3 - << 'PYEOF'
from huggingface_hub import snapshot_download
snapshot_download("chrishayuk/gemma-4-26b-a4b-it-vindex-expert-server",
                  local_dir="vindex-expert-server")
snapshot_download("chrishayuk/dec0-residual-pools", repo_type="dataset",
                  local_dir="pools")
PYEOF

echo "[dec0p5] building release…"
cargo build --release -p larql-cli -p larql-server 2>&1 | tail -1

# ── 1. Per-core kernel bench (single-thread GiB/s; AVX2-or-scalar on x86,
#      the startup kernel_class line pins which) ─────────────────────────
echo "[dec0p5] kernel bench (single-thread)…"
cargo bench -p larql-compute --bench q4k_q8k_matvec 2>&1 | tee "$OUT/kernel-bench.log" | grep -E "GiB|time:|thrpt" | head -20

# ── 2. Serving-level points (loopback on this box) ──────────────────────
./target/release/larql-server vindex-expert-server --ffn-only --port 8080 \
    > "$OUT/server.log" 2>&1 &
SRV=$!
trap 'kill "$SRV" 2>/dev/null || true' EXIT
for i in $(seq 1 180); do
    curl -sf $URL/v1/health >/dev/null 2>&1 && break
    kill -0 "$SRV" 2>/dev/null || { echo "server died — see $OUT/server.log"; exit 1; }
    sleep 1
done
grep -E "kernel class|decode options" "$OUT/server.log" | tee "$OUT/kernel-class.txt"

echo "[dec0p5] dense q8k replay point (B ∈ {1,8}, batch dispatch)…"
./target/release/larql dec-bench replay --ffn $URL --capture pools/dense \
    --endpoint walk-ffn --wire q8k --batch 1,8 --dispatch batch --repeats 3 \
    --net-rtt-ms 0.05 --net-gbps 0 \
    --output-file "$OUT/dense-q8k.json" --pulse-file "$OUT/dense-q8k.jsonl"

echo "[dec0p5] routed experts-ml-q8k replay point (B ∈ {1,8}, batch dispatch)…"
./target/release/larql dec-bench replay --ffn $URL --capture pools/routed \
    --endpoint experts --wire q8k --batch 1,8 --dispatch batch --repeats 3 \
    --net-rtt-ms 0.05 --net-gbps 0 \
    --output-file "$OUT/routed-q8k.json" --pulse-file "$OUT/routed-q8k.jsonl"

echo
echo "[dec0p5] done. Gate inputs in $OUT:"
echo "  kernel-bench.log (single-thread GiB/s vs Mac baseline)"
echo "  kernel-class.txt (which kernel arm actually served)"
echo "  {dense,routed}-q8k.jsonl (step p50 per point; divide by cores for per-core)"
grep -h "step_ms_p50" "$OUT"/*.jsonl 2>/dev/null | head -8 || true
