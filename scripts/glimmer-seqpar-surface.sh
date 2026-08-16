#!/usr/bin/env bash
# Explicit slice surface for the lowering at Glimmer geometry (hd 128).
# Per context: off, 2, 4, 6, 8, off — bracket = the two off arms.
W="${LARQL_LOWERED_TREE:-.}"

CTXS="${CTXS:-512 1024 2048 4096}"
ARMS="${ARMS:-off 2 4 6 8 off}"
for n in $CTXS; do
  for arm in $ARMS; do
    out=$(LARQL_KV_SEQPAR=$arm $W/target/release/larql vindex3 exec /Users/christopherhay/chris-models/muse-glimmer-s5.vindex3 --tokens "$(cat bench/prompts/glimmer/span-$n.ids)" --backend metal-lowered --generate 64 2>&1)
    steady=$(echo "$out" | grep -E "^steady" | sed 's/steady (last half): //')
    ids=$(echo "$out" | grep "generated ids" | md5 | cut -c1-8)
    wit=$(echo "$out" | grep "attention dispatches" | sed 's/attention dispatches: //')
    metal=$(echo "$out" | grep -c "\[metal\]")
    batt=$(pmset -g batt | grep -oE '[0-9]+%; [a-z ]+' | head -1)
    printf '%s ctx=%-5s arm=%-4s %-30s ids=%s  %s  metal=%s  [%s]\n' "$(date +%H:%M:%S)" "$n" "$arm" "$steady" "$ids" "$wit" "$metal" "$batt"
  done
done
