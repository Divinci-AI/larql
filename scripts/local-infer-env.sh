#!/usr/bin/env bash
# Local walk-vs-dense test environment.
#
# Serves the synthetic testdata/tiny-vindex on localhost and exercises both
# inference engines through the real server, with no model download and no
# cloud dependency. `/v1/infer` with `mode: "compare"` runs walk and dense
# over the SAME loaded weights in one request, which is what makes the
# comparison trustworthy: same process, same tokenization, same residuals.
#
#   ./scripts/local-infer-env.sh serve     # start the server (foreground)
#   ./scripts/local-infer-env.sh compare   # walk-vs-dense sweep
#   ./scripts/local-infer-env.sh patch     # DELETE-vs-UPDATE effect on walk
#
# NOTE ON PORTS: OrbStack, if installed, squats on a wide range of common
# ports (8080, 8787, 8791...) and answers with its own 403/404 rather than
# refusing the connection — so a health check can "succeed" against something
# that is not this server at all. Hence the high default and 127.0.0.1 bind.
set -euo pipefail

PORT="${LARQL_TEST_PORT:-18791}"
BASE="http://127.0.0.1:${PORT}"
VINDEX="${LARQL_TEST_VINDEX:-testdata/tiny-vindex}"
BIN=./target/release/larql-server

case "${1:-serve}" in
  serve)
    [ -x "$BIN" ] || cargo build --release -p larql-server
    exec "$BIN" "$VINDEX" --host 127.0.0.1 --port "$PORT"
    ;;
  compare) exec python3 scripts/local_infer_probe.py compare "$BASE" ;;
  patch)   exec python3 scripts/local_infer_probe.py patch   "$BASE" ;;
  *) echo "usage: $0 {serve|compare|patch}" >&2; exit 2 ;;
esac
