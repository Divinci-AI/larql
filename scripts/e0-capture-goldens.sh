#!/usr/bin/env bash
# E0 corpus C3 — capture pinned golden outputs from the PRE-V2 binary.
#
# E0 (docs/vindex2-experiments.md) asserts that adding VINDEX2 support to the
# binary changes nothing observable on any VINDEX1 path. That assertion needs a
# fixed record to compare against.
#
# WHY THIS EXISTS AT ALL: without committed goldens, "zero behavioural
# regression" silently degrades into "the two binaries agree with each other" —
# a condition any bug present in BOTH satisfies. Since the whole point of E0 is
# to catch a v1 path damaged by v2 work sharing one binary, comparing two builds
# of that binary to each other is close to circular. The baseline has to predate
# the v2 code and has to be committed.
#
# WHEN: run this against a binary built from a commit BEFORE any lyrw2 work.
# Once VINDEX2 merges to the main line, the baseline stops being a checkout and
# starts being an archaeology exercise — and a reconstructed baseline is exactly
# the artefact that drifts without anyone noticing.
#
# Usage:
#   E0_BIN=<pre-v2 larql> E0_VINDEX=output/gemma.vindex ./scripts/e0-capture-goldens.sh
#
# Env vars:
#   E0_BIN              — larql binary to capture from   (required; must be pre-v2)
#   E0_VINDEX           — v1 vindex to exercise          (required)
#   E0_MODEL            — checkpoint the vindex came from (recorded in the recipe)
#   E0_MODEL_REVISION   — that checkpoint's revision hash (recorded in the recipe)
#   E0_EXTRACT_FLAGS    — flags used to extract it        (recorded in the recipe)
#   E0_OUT              — golden output dir              (default: tests/goldens/e0/<vindex name>)
#   E0_TOKENS           — tokens to decode per prompt    (default: 24)
#   E0_WALK_K           — WALK top-K per layer           (default: 20)
#
# The vindex itself is NOT committed — it is regenerable. The recipe is.
#
# Determinism: `larql run` samples greedily (SamplingConfig::greedy), so decode
# is reproducible without a seed flag. Every command's output is normalised for
# paths and timings before writing, so a diff means a behavioural change rather
# than a different working directory or a faster machine.

set -uo pipefail

readonly TOKENS="${E0_TOKENS:-24}"
readonly WALK_K="${E0_WALK_K:-20}"
readonly SLICE_PRESETS=(client attn embed server browse router expert-server all)

# Fixed prompt set. Short, deterministic, and spanning factual recall, code and
# multi-token continuation so a regression in any one does not hide in the others.
readonly PROMPTS=(
  "The capital of France is"
  "def fibonacci(n):"
  "Water boils at"
)

: "${E0_BIN:?set E0_BIN to a larql binary built BEFORE any lyrw2 commit}"
: "${E0_VINDEX:?set E0_VINDEX to a VINDEX1 directory}"

readonly OUT_DIR="${E0_OUT:-tests/goldens/e0/$(basename "$E0_VINDEX")}"
readonly SLICE_TMP="$(mktemp -d)"
trap 'rm -rf "$SLICE_TMP"' EXIT

mkdir -p "$OUT_DIR"

# Strip anything that varies between runs or machines but is not behaviour:
# absolute paths, wall-clock durations, throughput rates, ISO timestamps.
normalise() {
  sed -E \
    -e "s#${E0_VINDEX}#<VINDEX>#g" \
    -e "s#${SLICE_TMP}#<TMP>#g" \
    -e 's#/[^ ]*/(larql|target)[^ ]*#<BIN>#g' \
    -e 's/[0-9]+(\.[0-9]+)? *(ms|s|tok\/s|MB\/s|GB\/s)\b/<TIME>/g' \
    -e 's/[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:.]+Z?/<TS>/g'
}

capture() {
  local name="$1"; shift
  printf '  → %-28s' "$name"
  # A non-zero exit is itself part of the golden: E0 must notice if a path that
  # used to fail starts succeeding, or vice versa.
  local out status
  out="$("$@" 2>&1)"; status=$?
  { printf '%s\n' "$out" | normalise; echo "exit_status=${status}"; } > "${OUT_DIR}/${name}.txt"
  echo "exit=${status}"
}

echo "E0 C3 golden capture"
echo "  binary: ${E0_BIN}"
echo "  vindex: ${E0_VINDEX}"
echo "  out:    ${OUT_DIR}"
echo

# Provenance + regeneration recipe.
#
# The corpus is reproducible, so it is pinned rather than stored: the multi-GB
# vindex is not committed, this recipe is. Two things must be here or the golden
# set is unfalsifiable — the baseline commit (without it, a reader cannot tell
# whether these goldens predate the v2 work they police) and the exact extract
# flags (without them, "re-extract it" is not a reproducible instruction).
{
  echo "captured_from_commit=$(git rev-parse HEAD)"
  echo "captured_from_describe=$(git describe --always --dirty)"
  echo "vindex=$(basename "$E0_VINDEX")"
  echo "tokens=${TOKENS}"
  echo "walk_k=${WALK_K}"
  echo
  echo "# ── C1 regeneration recipe ──"
  echo "# rebuild the baseline binary:"
  echo "#   git checkout $(git rev-parse HEAD) && cargo build --release -p larql-cli"
  echo "# re-extract the vindex:"
  echo "model_source=${E0_MODEL:-<unrecorded — set E0_MODEL next capture>}"
  echo "model_revision=${E0_MODEL_REVISION:-<unrecorded — set E0_MODEL_REVISION next capture>}"
  echo "extract_flags=${E0_EXTRACT_FLAGS:-<unrecorded — set E0_EXTRACT_FLAGS next capture>}"
} > "${OUT_DIR}/PROVENANCE.txt"

echo "── identity and integrity ──"
capture verify "$E0_BIN" verify "$E0_VINDEX"
capture show   "$E0_BIN" show   "$E0_VINDEX"

# index.json is the generation discriminator (spec §12.1). Pin the version
# fields explicitly so a stray bump is a diff rather than a load-time surprise.
printf '  → %-28s' index_version
python3 - "$E0_VINDEX/index.json" > "${OUT_DIR}/index_version.txt" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for k in ("version", "vindex_spec_version", "extract_level", "quant_format"):
    print(f"{k}={d.get(k)}")
PY
echo "ok"

echo "── query surface (WALK top-K rankings) ──"
for i in "${!PROMPTS[@]}"; do
  capture "walk_p${i}" "$E0_BIN" walk \
    --index "$E0_VINDEX" --prompt "${PROMPTS[$i]}" --top-k "$WALK_K"
done

echo "── decode (greedy, token-identical assertion) ──"
for i in "${!PROMPTS[@]}"; do
  capture "decode_p${i}" "$E0_BIN" run \
    "$E0_VINDEX" "${PROMPTS[$i]}" -n "$TOKENS"
done

echo "── slice presets ──"
for preset in "${SLICE_PRESETS[@]}"; do
  capture "slice_${preset}" "$E0_BIN" slice \
    "$E0_VINDEX" -o "${SLICE_TMP}/${preset}" --preset "$preset" --dry-run
done

echo
echo "captured $(find "$OUT_DIR" -name '*.txt' | wc -l | tr -d ' ') golden files into ${OUT_DIR}"
echo "COMMIT THESE. They are the only fixed record E0 has."
