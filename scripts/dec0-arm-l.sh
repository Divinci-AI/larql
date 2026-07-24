#!/usr/bin/env bash
# DEC-0 arm L — loopback batch curve, Linux/Colab (docs/dec-funnel.md §3).
#
# Sibling of scripts/dec0-loopback.sh (arm M). Differs in three ways:
#   1. No Metal — this box has no GPU attention path pre-G-ladder, and
#      dec-bench replay never invokes attention anyway (it replays captured
#      residuals straight at the expert/FFN server).
#   2. No capture — arm L's whole point is host-portability: it replays the
#      *same* 64-prompt pool captured on the Mac (arm M), fetched from
#      wherever it was staged, not a fresh Linux-side capture.
#   3. No e2e single-stream anchor by default — that needs the FULL vindex
#      (attention + embed + FFN weights, ~tens of GB) plus a working CPU
#      attention decode path. The expert-server-only vindex slice
#      (`hf://chrishayuk/gemma-4-26b-a4b-it-vindex-expert-server`) is enough
#      for the replay sweep, which is the claim-bearing measurement (batch
#      *shape*, not absolute tok/s — dec-funnel.md §3 DEC-0). Set
#      DEC0L_RUN_ANCHOR=1 with DEC0L_ANCHOR_VINDEX pointing at a full vindex
#      to also run the anchor arm.
#
# Usage:
#   POOL_DENSE_URL=https://.../residuals-gemma4-26b-a4b-q4k.tar.gz \
#   POOL_ROUTED_URL=https://.../residuals-gemma4-26b-a4b-q4k-routed.tar.gz \
#   ./scripts/dec0-arm-l.sh
#
# Optional env vars:
#   DEC0L_VINDEX         — expert-server vindex source (default: the
#                           published HF expert-server slice; `larql-server`
#                           auto-downloads hf:// on first launch)
#   DEC0L_PORT            (default: 8080)
#   DEC0L_OUT_DIR          (default: bench/dec0)
#   DEC0L_POOL_DIR         (default: bench/dec0, pools land in
#                           $DEC0L_POOL_DIR/residuals-gemma4-26b-a4b-q4k[-routed])
#   POOL_DENSE_URL        — required unless the dense pool dir already exists
#   POOL_ROUTED_URL       — required unless the routed pool dir already exists
#   DEC0L_STEPS            (default: 16, must match the captured pool)
#   DEC0L_REPEATS          (default: 3)
#   DEC0L_BATCHES           (default: 1,8,16,32,64)
#   DEC0L_DENSE_WIRES       (default: f32,f16,i8,q8k)
#   DEC0L_ROUTED_WIRES      (default: f32,q8k — the experts endpoint's only
#                           supported wires, per `larql dec-bench replay --help`)
#   DEC0L_RUN_ANCHOR=1     — also run the e2e single-stream anchor arm
#   DEC0L_ANCHOR_VINDEX    — full vindex for the anchor arm (required if
#                           DEC0L_RUN_ANCHOR=1)
#   DEC0L_SKIP_BUILD=1     — skip `cargo build` (reuse an existing binary)

set -euo pipefail

PORT="${DEC0L_PORT:-8080}"
OUT_DIR="${DEC0L_OUT_DIR:-bench/dec0}"
POOL_DIR="${DEC0L_POOL_DIR:-bench/dec0}"
VINDEX="${DEC0L_VINDEX:-hf://chrishayuk/gemma-4-26b-a4b-it-vindex-expert-server}"
STEPS="${DEC0L_STEPS:-16}"
REPEATS="${DEC0L_REPEATS:-3}"
BATCHES="${DEC0L_BATCHES:-1,8,16,32,64}"
DENSE_WIRES="${DEC0L_DENSE_WIRES:-f32,f16,i8,q8k}"
ROUTED_WIRES="${DEC0L_ROUTED_WIRES:-f32,q8k}"
URL="http://127.0.0.1:${PORT}"
STAMP="$(date +%Y%m%d-%H%M%S)"

DENSE_POOL="${POOL_DIR}/residuals-gemma4-26b-a4b-q4k"
ROUTED_POOL="${POOL_DIR}/residuals-gemma4-26b-a4b-q4k-routed"

mkdir -p "${OUT_DIR}" "${POOL_DIR}"

fetch_pool() {
    local dir="$1" url_var="$2"
    if [ -f "${dir}/manifest.json" ]; then
        echo "[dec0-arm-l] pool already present: ${dir}"
        return
    fi
    local url="${!url_var:-}"
    if [ -z "${url}" ]; then
        echo "[dec0-arm-l] ${dir} missing and ${url_var} not set — set it to a" \
             "fetchable tar.gz of the arm-M pool (dec-funnel.md: pools are" \
             "host-portable, capture is model-free)." >&2
        exit 1
    fi
    echo "[dec0-arm-l] fetching pool: ${url}"
    local tmp
    tmp="$(mktemp)"
    curl -fsSL "${url}" -o "${tmp}"
    mkdir -p "${dir}"
    tar -xzf "${tmp}" -C "${dir}" --strip-components=1
    rm -f "${tmp}"
}

fetch_pool "${DENSE_POOL}" POOL_DENSE_URL
fetch_pool "${ROUTED_POOL}" POOL_ROUTED_URL

if [ "${DEC0L_SKIP_BUILD:-0}" != "1" ]; then
    echo "[dec0-arm-l] building release binaries…"
    # --no-default-features mirrors larql-cli.yml's Linux CI job: default
    # features enable Metal (macOS-only) plus a wider dependency graph
    # (aws-lc-sys among others) this CPU-only expert-server path doesn't need.
    cargo build --release -p larql-cli -p larql-server --no-default-features
fi

echo "[dec0-arm-l] launching expert server (${URL}, --ffn-only, vindex=${VINDEX})…"
echo "[dec0-arm-l] first launch downloads the vindex from HuggingFace if not cached — allow a few minutes."
./target/release/larql-server "${VINDEX}" --ffn-only --port "${PORT}" \
    >"${OUT_DIR}/server-arml-${STAMP}.log" 2>&1 &
SERVER_PID=$!
trap 'kill "${SERVER_PID}" 2>/dev/null || true' EXIT

for i in $(seq 1 600); do
    if curl -sf "${URL}/v1/health" >/dev/null 2>&1; then break; fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[dec0-arm-l] server died during startup — see ${OUT_DIR}/server-arml-${STAMP}.log" >&2
        exit 1
    fi
    sleep 1
    if [ "$i" = 600 ]; then
        echo "[dec0-arm-l] server did not become healthy in 600s (cold HF download can be slow)" >&2
        exit 1
    fi
done
echo "[dec0-arm-l] server healthy."

# ── Optional anchor arm (off by default — see header) ───────────────────────
if [ "${DEC0L_RUN_ANCHOR:-0}" = "1" ]; then
    ANCHOR_VINDEX="${DEC0L_ANCHOR_VINDEX:?set DEC0L_ANCHOR_VINDEX for the anchor arm}"
    for dispatch in streaming batch; do
        echo "[dec0-arm-l] anchor arm: --ffn-dispatch ${dispatch}…"
        ./target/release/larql bench "${ANCHOR_VINDEX}" \
            --ffn "${URL}" \
            --ffn-dispatch "${dispatch}" \
            --wire f32,f16,i8 \
            --tokens 50 \
            --warmup 3 \
            --output json \
            --output-file "${OUT_DIR}/anchor_arml_${dispatch}_${STAMP}.json"
    done
fi

# ── Dense replay: walk-ffn endpoint ──────────────────────────────────────────
DENSE_PULSE="${OUT_DIR}/dec0_arml_dense_pulse_${STAMP}.jsonl"
DENSE_RECORD="${OUT_DIR}/dec0_arml_dense_replay_${STAMP}.json"
echo "[dec0-arm-l] dense replay sweep: batch {${BATCHES}} × wire {${DENSE_WIRES}} × dispatch {streaming,batch}…"
./target/release/larql dec-bench replay \
    --ffn "${URL}" \
    --capture "${DENSE_POOL}" \
    --endpoint walk-ffn \
    --batch "${BATCHES}" \
    --wire "${DENSE_WIRES}" \
    --dispatch streaming,batch \
    --repeats "${REPEATS}" \
    --steps "${STEPS}" \
    --net-rtt-ms 0.05 \
    --net-gbps 0 \
    --output-file "${DENSE_RECORD}" \
    --pulse-file "${DENSE_PULSE}"

# ── Routed replay: experts endpoint (needs the --routing sidecars) ──────────
ROUTED_PULSE="${OUT_DIR}/dec0_arml_routed_pulse_${STAMP}.jsonl"
ROUTED_RECORD="${OUT_DIR}/dec0_arml_routed_replay_${STAMP}.json"
echo "[dec0-arm-l] routed replay sweep: batch {${BATCHES}} × wire {${ROUTED_WIRES}} × dispatch {streaming,batch}…"
./target/release/larql dec-bench replay \
    --ffn "${URL}" \
    --capture "${ROUTED_POOL}" \
    --endpoint experts \
    --batch "${BATCHES}" \
    --wire "${ROUTED_WIRES}" \
    --dispatch streaming,batch \
    --repeats "${REPEATS}" \
    --steps "${STEPS}" \
    --net-rtt-ms 0.05 \
    --net-gbps 0 \
    --output-file "${ROUTED_RECORD}" \
    --pulse-file "${ROUTED_PULSE}"

echo
echo "[dec0-arm-l] done."
echo "  dense  run record : ${DENSE_RECORD}"
echo "  dense  pulse       : ${DENSE_PULSE}"
echo "  routed run record : ${ROUTED_RECORD}"
echo "  routed pulse       : ${ROUTED_PULSE}"
echo
echo "  C1 verdict input: step p50 vs batch, both pools — compare against the"
echo "  arm-M numbers in docs/dec-funnel.md §3 DEC-0. Kill condition: either"
echo "  pool saturates below batch 16 on this box while arm M does not."
echo
