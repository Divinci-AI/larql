# Roadmap — larql-server / larql-router

For shipped work, see [CHANGELOG.md](CHANGELOG.md) — including the 2026-05-07
state-of-the-server snapshot and perf tables that used to live at the top of
this file, and the full 2026-05-10 code review. Design rationale is in
[THESIS.md](THESIS.md); protocol detail in [docs/server-spec.md](docs/server-spec.md).

## Current state (verified 2026-08-04)

**Tests.** 336 lib tests passing (`cargo test -p larql-server --lib`), plus the
integration suites under `tests/`.

**Route surface.** 20 modules under `src/routes/`: the OpenAI-compatible group
(`openai/` — chat, completions, embeddings, models), the LQL group (`describe`,
`explain`, `insert`, `patches`, `relations`, `select`, `walk`), the grid group
(`expert/`, `shard`, `topology`, `walk_ffn/`), and operational endpoints
(`health`, `stats`, `stream`, `warmup`, `embed`, `infer`).

**Coverage.** Enforced: 90% per file, **65% crate total**, 9 debt baselines.
The low total is deliberate and structural — the `routes/expert/*` modules need
a live grid harness to exercise, so they sit outside what a unit test can
reach. Run `make larql-server-coverage-summary` to re-measure.

**Performance.** The last full measurement is the 2026-05-07 M3 Max 2-shard
grid snapshot in [`CHANGELOG.md`](CHANGELOG.md), which also holds the remote
MoE expert path table. Nothing in this crate has been re-benched since; treat
those numbers as a baseline, not a current claim.

---

## Open defects

- **P1 — unbounded in-memory growth with dead eviction logic.** Raised
  2026-05-28, **confirmed still open 2026-08-04** and slightly worse than
  written. `ratelimit.rs:83` defines `evict_stale()` and **nothing in
  production calls it** — the only call site is its own unit test, which
  asserts merely that it does not panic. `session.rs` is worse still: it has no
  eviction function to wire up at all, so the session map grows without bound
  for the process lifetime. Memory/DoS class. Fixing this means writing the
  session eviction, not just calling it.

---

## Great new functionality (next big-ticket items)

The numbered F0..F23 items below are mostly **incremental polish**
(metrics, shutdown drain, RBAC, OpenAPI, etc.) — necessary but not
load-bearing for new use cases. The items in this section are
**new capabilities** that would unlock production deployment shapes
the server can't currently serve. Ranked by how much they expand
the addressable surface, not by implementation effort.

### N0. OpenAI API compatibility (Chat Completions, Completions, Responses, Embeddings)

**Status**: **Slices 1 + 2 shipped 2026-05-02** — `/v1/models`,
`/v1/embeddings`, `/v1/completions`, `/v1/chat/completions` (all
non-streaming) live and OpenAI-shape-conformant on `larql-server`.
Live-validated against `output/gemma3-4b-q4k-streaming.vindex`. Chat
templates auto-detected from `arch.family()` (Gemma / Llama / ChatML
/ Mistral / Plain).

Slice 3 (SSE streaming on completions + chat completions) + slice 4
(tools / JSON mode / `response_format: json_schema`) + slice 5
(Responses API) remain; per-item **Status** lines below.

Supersedes the older F10 ("OpenAI-compat `/v1/chat/completions`")
which scoped only the chat endpoint shallowly. **Highest-leverage
item in this section** — every existing OpenAI client (Python `openai`
SDK, JS `openai`, LangChain, LlamaIndex, Cursor, Continue, Aider, eval
harnesses, dashboards) becomes a larql client the day all slices ship.
With slices 1+2 every chat client today already works; slices 3+4 add
the polish (streaming, tools, structured output).

**Router-side parity (N0-router)**: `larql-router` should also serve
the OpenAI surface so clients can hit the grid as a single endpoint
and the router fans out to shards. `/v1/models` aggregates from
registered shards; `/v1/embeddings`, `/v1/completions`, and
`/v1/chat/completions` proxy to shards owning the relevant compute.
Tracked under "Router-side OpenAI surface" in P1.

**Scope** — five endpoints, mapped onto our existing inference path:

#### N0.1 `POST /v1/chat/completions` (Chat Completions API)

**Status**: Slice 2 shipped 2026-05-02 (non-streaming). Live-validated
against `output/gemma3-4b-q4k-streaming.vindex`. Wire conforms to the
OpenAI shape; chat templates auto-detected from `arch.family()` (Gemma
/ Llama / ChatML / Mistral) with id-string fallback and a Plain
template for unknown / non-instruct models. SSE streaming → slice 3.
`tools` / `tool_choice` / `response_format: json_*` → slice 4 (returns
400 with a clear "see ROADMAP" message). `n>1` → 400.

Original spec preserved below for context on the streaming + tools work
that remains.

```
Request:  {model, messages: [{role, content, tool_calls?, tool_call_id?}],
           temperature?, top_p?, max_tokens?, stream?, tools?, tool_choice?,
           response_format?, seed?, stop?, n?, frequency_penalty?,
           presence_penalty?, logprobs?, top_logprobs?, user?}
Response: {id, object: "chat.completion", created, model,
           choices: [{index, message: {role: "assistant", content,
                       tool_calls?}, finish_reason, logprobs?}],
           usage: {prompt_tokens, completion_tokens, total_tokens}}
SSE chunk: {id, object: "chat.completion.chunk", created, model,
            choices: [{index, delta: {role?, content?, tool_calls?},
                       finish_reason?}]}
SSE terminator: `data: [DONE]\n\n`
```

Translation layer:
- `messages` → render via existing `chat::render_user_prompt` (per
  family chat template) → `encode_prompt` → `generate_streaming`.
- `stream: true` → wrap `generate_streaming`'s `on_token` callback in
  an SSE encoder; emit one chunk per token.
- `tools` → constrained-decoding mask routing the model toward valid
  tool-call JSON. Depends on N0.6 (JSON schema → grammar).
- `response_format: {type: "json_object"}` or
  `response_format: {type: "json_schema", json_schema: {...}}` → same
  constrained-decoding hook.
- `stop` strings → augment the existing `EosConfig` for the duration
  of the call.
- `seed` → pass through to `SamplingConfig` (already supported).

#### N0.2 `POST /v1/completions` (Legacy Completions API)

Older but still widely used (especially by older eval harnesses and
embedding/reranker pipelines that haven't migrated). Simpler shape:

```
Request:  {model, prompt: string | string[], max_tokens?, temperature?,
           top_p?, stream?, logprobs?, echo?, stop?, n?, best_of?,
           seed?, suffix?}
Response: {id, object: "text_completion", created, model,
           choices: [{text, index, finish_reason, logprobs?}],
           usage: {...}}
```

Strictly easier than N0.1 — no chat template, no tool calls, no
multi-message context. Maps directly to `encode_prompt` +
`generate_streaming`. Could ship first as a smoke-test of the
overall translation layer.

#### N0.3 `POST /v1/responses` (Responses API — newer, stateful)

OpenAI's 2025 successor to chat completions. Designed for stateful
multi-turn agents with built-in tool execution + reasoning content.
Pairs naturally with **N1 (stateful chat sessions)** — the
`previous_response_id` field references prior turns whose KV-cache
the server kept resident.

```
Request:  {model, input: string | InputItem[], previous_response_id?,
           instructions?, tools?, tool_choice?, response_format?,
           reasoning?, store?, metadata?, parallel_tool_calls?}

InputItem variants: text input ({type: "message", role, content}),
                    function-call output ({type: "function_call_output",
                    call_id, output}), file references, etc.

Response: {id, object: "response", created_at, status: "completed"|...,
           model, output: [
             {type: "message", role: "assistant", content: [{type: "output_text", text}]},
             {type: "function_call", call_id, name, arguments},
             {type: "reasoning", content},  // for o1 / DeepSeek-R1 style models
           ],
           usage: {input_tokens, output_tokens, reasoning_tokens, total_tokens},
           previous_response_id}
```

Implementation path:
- Without N1: each call is a fresh prefill (server-side response storage
  optional via `store: true` — return `id` for retrieval but don't
  reuse KV-cache).
- With N1: `previous_response_id` → look up the session's KV-cache,
  continue from that state (zero re-prefill on the prior turns).
- Reasoning content (DeepSeek-R1 / Gemma-thinking-style models): emit
  thinking traces as a separate `output[]` entry.

#### N0.4 `POST /v1/embeddings` (Embeddings API)

Existing `/v1/embed` endpoint already does this work; just needs an
OpenAI-shape wrapper.

```
Request:  {model, input: string | string[] | int[] | int[][],
           encoding_format?: "float" | "base64", dimensions?}
Response: {object: "list", data: [{object: "embedding", embedding: [...],
           index}], model, usage: {prompt_tokens, total_tokens}}
```

Two important nuances:
- `input` accepts strings (we tokenise) or pre-tokenised arrays
  (we embed directly via existing `/v1/embed`).
- `encoding_format: "base64"` returns embeddings as
  base64-encoded f32 little-endian bytes — ~33% smaller wire than
  the JSON float array form. Many production clients default to
  base64.

#### N0.5 `GET /v1/models` (already exists, needs OpenAI shape)

Current response shape doesn't match OpenAI's. Reshape:

```
{object: "list", data: [
   {id, object: "model", created, owned_by: "larql", parent?, ...}
]}
```

Trivial — existing route just needs the wrapper.

#### N0.6 Constrained decoding (JSON schema + GBNF grammar)

`response_format: {type: "json_schema"}` and `tools` both require
the decoder to emit only tokens that keep the output grammar-valid.
Today the inference-side decoder has a regex/grammar hook
(`EosConfig` / sampling pipeline already supports "stop strings");
need to extend with a real GBNF parser + JSON Schema → GBNF compiler.

Implementation is well-trodden — port from llama.cpp's `grammar.cpp` /
`grammar-parser.cpp` (well-defined spec; ~1000 LOC). Tracked
separately as F17 in this ROADMAP, but N0 makes it load-bearing.

#### Cross-cutting concerns

- **Streaming framing**: SSE format is `data: {json}\n\n` per chunk,
  terminated by `data: [DONE]\n\n`. axum has `axum::response::sse`
  out of the box.
- **Authentication**: the existing `--api-key` Bearer token mechanism
  works as-is; OpenAI clients send `Authorization: Bearer sk-...`.
- **Model identity**: `model` field in the request maps to a vindex
  ID. For single-model servers, ignore. For multi-model, route via
  the existing model-id mux.
- **Usage tokens**: track `prompt_tokens` (count from
  `encode_prompt`'s output) and `completion_tokens` (count tokens
  generated). Trivial bookkeeping.
- **Error envelope**: OpenAI uses `{error: {message, type, param,
  code}}` — slightly different from our `{error: "..."}`. Add an
  OpenAI-shape error mapper at the route layer.
- **Rate limit headers**: `x-ratelimit-limit-requests`,
  `x-ratelimit-remaining-requests`, etc. — pairs with our existing
  `--rate-limit` machinery.

#### Build order recommendation

1. **N0.5 + N0.4 + N0.2** (Models + Embeddings + Completions) —
   smallest, no streaming, validates the OpenAI shape + auth.
   Makes the server immediately usable for embedding-only and
   text-completion workloads.
2. **N0.1 non-streaming** (Chat Completions, no tools, no
   constrained output yet) — covers ~80% of real chat usage.
3. **N0.1 streaming** (SSE) — every chat UI assumes this.
4. **N0.6** (constrained decoding) — unblocks tools + structured
   output.
5. **N0.1 with tools + JSON mode** — production-grade chat.
6. **N0.3 (Responses API)** — pairs with N1 for stateful continuation.

#### Implementation surface (rough)

- N0.5: ~30 LOC (just a wrapper)
- N0.4: ~150 LOC (translate input format, base64 encoding)
- N0.2: ~250 LOC (legacy completions, simpler)
- N0.1 non-streaming: ~400 LOC
- N0.1 streaming SSE: +200 LOC
- N0.6 GBNF + JSON Schema: ~1200 LOC (port from llama.cpp)
- N0.1 with tools + JSON mode: +300 LOC (depends on N0.6)
- N0.3 Responses API (stateless): ~500 LOC
- N0.3 stateful (with N1): +200 LOC on top

**Total**: ~3200 LOC, shippable in slices. The first 5-day slice
(items 1-3 above) is enough to make larql-server a viable target for
most existing clients.

#### Files

New `routes/openai/` directory — one file per endpoint. Shared
`routes/openai/types.rs` for the request/response schemas (use
`serde` to match the OpenAI shape exactly; let serde-rename do the
heavy lifting for camelCase conversions). Wire into
`routes/mod.rs::single_model_router` alongside the existing routes;
multi-model routing via `model` field in the request body.

#### Why this beats every other N item on leverage

- N1 (sessions) is great but only useful if you have a client to use
  it with. **N0 brings every existing client.**
- N4 (multimodal) is an addressable-market expansion, not a
  client-acquisition unlock.
- N5 (federated knowledge graph) is unique but needs a custom
  client until OpenAI adds federated DESCRIBE to their spec (never).
- N0 is the move that makes everything else discoverable. Ship it
  first.

---

### N1. Stateful chat sessions (KV-cache as a first-class resource)

**Why this is the biggest gap.** Every production LLM API today is
session-aware: client sends the new turn, server remembers prior context
via KV-cache. larql-server's `/v1/infer` is single-shot — every request
re-prefills from scratch. For a 4 K context that's ~100 ms of wasted
compute per turn; for 16 K it's seconds. We're not competitive with
vLLM / TGI / OpenAI for any chat workload.

The pieces exist or are tracked piecemeal — F7 (KV-cache prefix
sharing), F22 (persistent patches as a precedent for session
persistence), the chat session machinery already in
`larql-inference::layer_graph::generate::chat_session` — but no
end-to-end story.

**Proposal**:
- `POST /v1/sessions` → returns `{session_id}` + initial state
- `POST /v1/sessions/{id}/append` → adds user message, generates assistant
  reply, returns SSE stream. KV-cache stays resident.
- `GET /v1/sessions/{id}` → describes current state (msg count, token
  count, model, adapter, last activity).
- `DELETE /v1/sessions/{id}` → frees KV-cache.
- Eviction policy: per-session TTL, total-RSS budget, LRU under
  pressure. Surfaces in `/v1/stats.sessions`.
- Pairs with **N3 (LoRA hot-load)** — sessions can pin a specific adapter.

**Implementation surface**: ~600 LOC. New `routes/sessions.rs`,
new `state::SessionStore`, hook into the existing `generate_streaming`
+ `Detokenizer` machinery. Roughly half the work is the eviction /
budget management — non-trivial but well-scoped.

### N2. Asynchronous batch inference job queue

**Why**: Real-time chat is one model; **bulk inference** (RAG document
processing, embedding pre-compute, reranker scoring, evaluation
harnesses) is another. They have very different SLOs. A batch job
submitter doesn't care about per-token latency; it cares about
throughput, cost, and being able to run while the cluster is otherwise
idle. Today users have to wrap `/v1/infer` in their own retry/queue
glue.

**Proposal**:
- `POST /v1/jobs` → submit `{prompts: [...], model_id, params}` →
  returns `{job_id}`.
- `GET /v1/jobs/{id}` → status + partial results.
- `POST /v1/jobs/{id}/cancel`.
- Optional `webhook_url` in the submit body for completion callback.
- Worker pool: independent rayon thread pool, capped concurrency,
  prioritises real-time `/v1/infer` traffic (job worker yields when a
  real-time request arrives).
- Persistence: jobs survive restarts (write-ahead log to disk).

**Pairs with**: F12 (batched infer in same request), F22 (persistent
state). Together those two are the building blocks; this item is the
asynchronous wrapper.

**Implementation surface**: ~800 LOC. New `routes/jobs.rs`, new
`worker::Pool`, persistence to a `jobs/` directory. The hardest piece
is the priority scheduler — getting it wrong means batch starves
real-time or vice versa.

### N3. LoRA / adapter hot-loading per session

**Why**: Multi-tenant production. Today every tenant either gets the
same base model or has to spin up a separate process. Real production
serving (Anthropic, OpenAI, Together, Replicate) supports per-request
adapter swap. Adapters are 10-100 MB vs the 16 GB base model —
hot-loading hundreds of them is feasible if we have the surface.

**Proposal**:
- `POST /v1/adapters/load` → `{adapter_id, source: "hf://..."|"file://..."|"http://...",
  model_id}` → loads into RAM.
- `GET /v1/adapters` → list loaded adapters with size + last-used.
- `DELETE /v1/adapters/{id}` → evict.
- Inference / sessions take an optional `adapter_id` field — applies
  the LoRA delta to gate/up/down/q/k/v/o matmuls per layer per call.
- Eviction: LRU + total-RSS budget, configurable.

**Pairs with**: N1 (sessions pin adapters). Independent enough to ship
first if N1 is too heavy.

**Implementation surface**: ~500 LOC. The LoRA forward-pass plumbing
already exists at the inference-crate level (per
`larql-inference/ROADMAP.md` § F4 LoRA loading). The server piece is
the lifecycle + RSS management.

### N4. Multimodal API surface (vision tower, mixed image+text infer)

**Why**: Gemma 3/4 ships vision variants; Llama 3.2 too. The vindex
extractor already handles vision tower weights (per
`larql-inference/ROADMAP.md → vision`). We're missing the API
surface — there's no way to send an image to the server today.

**Proposal**:
- `POST /v1/embed/image` → multipart upload → vision tower forward →
  returns `{embedding: [...], hidden_size}`.
- `POST /v1/infer` accepts `images: [base64, ...]` field; server
  routes through the vision tower then concatenates with text tokens
  for the language decoder.
- `POST /v1/sessions/{id}/append` accepts images for multimodal chat.

**Implementation surface**: ~400 LOC server-side once the inference
crate's vision forward path is exposed (currently tracked separately).
Big use-case unlock: docVQA, ChartQA, image classification, image
embedding service.

### N5. Federated knowledge graph over multiple vindexes

**Why**: The DESCRIBE/WALK/SELECT trio makes a vindex a queryable
knowledge graph. Multi-model serving (`--dir`) puts multiple
graphs side-by-side — but each is queried independently. There's no
way to ask "describe France using Gemma's knowledge AND Llama's
knowledge AND my custom vindex". This is a unique capability the
larql architecture enables that nothing else (vLLM, TGI, OpenAI) can
do, and it's invisible.

**Proposal**:
- `GET /v1/federated/describe?entity=X&models=gemma,llama,custom` →
  merges edges across vindexes, sourcing each edge with its origin
  model.
- `POST /v1/federated/select` with cross-model joins ("entities
  Gemma calls capitals AND Llama calls capitals").
- New LQL syntax: `DESCRIBE "France" USING gemma, llama;` already
  hinted in the REPL doc (`USE REMOTE`); the server-side surface is
  the missing half.
- Surfacing model disagreement is a research-grade capability:
  "Gemma says Paris is the capital of France with score 1436;
  Llama says Lyon with score 320. Confidence-weighted merge?"

**Implementation surface**: ~600 LOC. New `routes/federated.rs`,
extends multi-model serving to do cross-model fan-out + merge.

### N6. Live blue-green vindex deployment

**Why**: Production model rollouts. Today swapping a vindex requires
restart (modulo F8 hot-swap, which is admin-only and atomic). True
blue-green wants: load v2 alongside v1, route X% of traffic, observe
metric drift, ramp or rollback.

**Proposal**:
- `POST /v1/admin/deploy` → load `v2.vindex` alongside the active
  `v1.vindex`, returns `{green_id}`.
- `POST /v1/admin/traffic` → set weighted routing
  (`{"v1": 0.9, "v2": 0.1}`).
- `GET /v1/stats.deployment` → per-vindex per-endpoint p50/p99/error
  rate side-by-side. Pairs with F3 metrics.
- `POST /v1/admin/promote/{id}` → atomically swap routing to 100%
  green; old vindex becomes stale-evictable.

**Pairs with**: F8 (admin endpoints), F3 (metrics for traffic
comparison). N6 is the **product** built on top of those primitives.

**Implementation surface**: ~700 LOC. New `routes/admin/deploy.rs`,
extends `AppState` to hold multiple model versions, weighted routing
logic in the request entry points.

---


## P0: Active

### G-TRANSPORT. Wire format evolution + WebSocket streaming + QUIC (ADR-0009, ADR-0010)

All work here is architecture-agnostic: no hardcoded layer counts, hidden
sizes, or model-family assumptions. Sizes and dtypes are read from vindex
config at runtime.

#### GT1 — f16 wire default

**Status**: ✅ **Shipped 2026-05-07.**

Added `FFN_F16_CT = "application/x-larql-ffn-f16"` in `wire.rs`; `encode_binary_output_f16` in `walk_ffn.rs`; `preferred_response_ct` selects f16 when client sends `Accept: application/x-larql-ffn-f16`. Client (`ffn/remote/http.rs`) sends `Accept: i8, f16, f32` on every grid request. `LARQL_F16_WIRE_DISABLE` opt-out. `half = "2"` added to both crates.

**Spec**: ADR-0009 §Decision, §Wire Layout (f16).

Wire format is currently f32-only (4 bytes/value). For a model with
hidden_size=H and seq_len=1, one round-trip costs `H × 4 × 2` bytes
(request + response). f16 halves this with no accuracy loss for all tested
architectures.

- Add `F16_WIRE = "LARQL_F16_WIRE"` to `env_flags.rs` (present = opt-out,
  i.e. `LARQL_F16_WIRE=0` forces f32).
- Add `F16_CT = "application/x-larql-ffn-f16"` to `wire.rs`.
- In `routes/walk_ffn.rs`: inspect `Accept` header; if client sends
  `Accept: application/x-larql-ffn-f16`, encode response as f16.
- In `larql-inference/src/ffn/remote/http.rs`: set
  `Accept: application/x-larql-ffn-f16` by default (opt-out via flag).
- Accuracy gate: `larql bench <vindex> --wire f32,f16 --assert-topk-match 5`
  must pass for each model family before enabling as default.

**Acceptance**: `larql bench <vindex> --ffn URL --wire f32,f16` shows <1%
tok/s difference and identical top-5 tokens. Wire bytes column shows 50% reduction.

#### GT2 — i8 quantised residuals (opt-in)

**Status**: ✅ **Shipped 2026-05-07.**

Added `FFN_I8_CT`; `encode_binary_output_i8` (per-position symmetric scale, zero_point=0) in `walk_ffn.rs`; `decode_binary_single/batch_i8` in `codec.rs`. Client advertises i8 in Accept header; server honours when `LARQL_I8_WIRE=1`. `preferred_response_ct` checks i8 before f16.

**Spec**: ADR-0009 §Wire Layout (i8), §Negotiation Protocol.

Per-position symmetric quantisation: `scale = max(|x|)/127`, `zero_point = 0`.
Wire: `[scale f32 LE][zero_point f32 LE][data i8[] × hidden_size]` per position.

- Add `I8_WIRE = "LARQL_I8_WIRE"` to `env_flags.rs` (opt-in, default off).
- Add `I8_CT = "application/x-larql-ffn-i8"` to `wire.rs`.
- Add `encode_i8_request`, `decode_i8_single/batch` to `ffn/remote/codec.rs`.
- Add `encode_i8_output` to `routes/walk_ffn.rs`.
- Accuracy gate: `--wire f32,i8 --assert-topk-match 1` must pass before
  enabling i8 as opt-out on any model family.

**Acceptance**: 75% bandwidth reduction vs f32; top-1 token identical on
≥95% of decode steps across tested architectures.

#### GT3 — Per-layer latency in HeartbeatMsg

**Status**: ✅ **Shipped 2026-05-07.**

`LayerLatency { layer, avg_ms, p99_ms }` added to grid.proto (`HeartbeatMsg.layer_stats` + `ServerInfo.layer_stats`). New `metrics::LayerLatencyTracker` (EMA α=0.1, p99 ring-buffer per layer, thread-safe Mutex). `LoadedModel.layer_latency_tracker` populated at construction; `walk_ffn.rs` records timing per layer after each FFN forward. `announce.rs` heartbeat sender calls `tracker.snapshot()`. Router `grid.rs` stores `layer_latencies: HashMap<u32, (avg_ms, p99_ms)>` in `ServerEntry`; `route()` prefers lowest `avg_ms` for the requested layer.

**Spec**: ADR-0011 §HeartbeatMsg Extension.

Current heartbeat sends `cpu_pct`, `ram_used`, `requests_in_flight` — all
global. Router uses `requests_in_flight` for load balancing. This is blind to
per-layer compute bottlenecks (e.g. a sparse MoE model where layer 15 is 3×
slower than others due to expert placement).

Proto change (`grid.proto`):
```protobuf
message LayerLatency {
  uint32 layer  = 1;
  float  avg_ms = 2;  // EMA α=0.1
  float  p99_ms = 3;  // ring-buffer p99 over last 100 requests
}
message HeartbeatMsg {
  // existing fields unchanged
  repeated LayerLatency layer_stats = 4;
}
```

Server changes:
- `LayerLatencyTracker` struct in new `src/metrics.rs`: one EMA + `VecDeque`
  per layer, updated in `routes/walk_ffn.rs` after each layer forward.
- `announce.rs`: populate `layer_stats` in the heartbeat sender.

Router change:
- `grid.rs::update_heartbeat`: store `layer_stats` in `ServerEntry`.
- `grid.rs::route`: prefer server with lowest `layer_stats[layer].avg_ms`
  when multiple replicas cover the same layer.

**Acceptance**: `larql serve --join ... --log-level debug` logs per-layer
latency in each heartbeat. Router `/grid-status` response includes
`layer_stats` per server.

#### GT4 — WebSocket token streaming (Q1.10 completion + N0.1 SSE)

**Status**: ✅ **Shipped 2026-05-07.**

`handle_stream_generate` added to `routes/stream.rs`: accepts `{"type":"generate","prompt":"...","max_tokens":N}` WebSocket message, calls `generate_streaming` in a `spawn_blocking` task, streams `{"type":"token","text":"...","index":N}` per token, emits `{"type":"done","tokens":N,"latency_ms":M}` on completion. Client cancel supported via `{"type":"cancel"}` frame. SSE on `/v1/chat/completions` (`stream:true`) was confirmed already fully wired (N0.1 slice 3 complete).

`routes/stream.rs` previously had a working WebSocket handler for `describe` and `infer`
commands but lacked a streaming token generation path. This is the missing
piece for N0.1 slice 3 (SSE on `POST /v1/chat/completions`).

- Complete `handle_stream_infer` in `routes/stream.rs`:
  - Accept `{"type": "generate", "prompt": "..."}` WS message.
  - Call `generate_streaming` (already exists in larql-inference).
  - Emit one `{"type": "token", "text": "..."}` frame per token.
  - Emit `{"type": "done", "tokens": N, "ms": M}` on completion.
  - Handle `{"type": "cancel"}` to abort generation.
- Add binary frame support: client can send
  `{"type": "generate", "format": "binary"}` to receive token IDs as u32 LE
  instead of JSON (lower overhead for embedding clients).
- Wire SSE for N0.1: in `routes/chat.rs`, when `stream: true`, use
  `axum::response::Sse` to wrap the same `generate_streaming` callback.
  Emit OpenAI-format `data: {...}\n\n` chunks; terminate with `data: [DONE]\n\n`.

**Acceptance**: `wscat -c ws://localhost:8080/v1/stream` receives one JSON
frame per token. `curl -N -H "Accept: text/event-stream" \
-d '{"model":"...","messages":[...],"stream":true}' \
http://localhost:8080/v1/chat/completions` streams tokens in SSE format.

#### GT7 — QUIC transport for grid

**Status**: ✅ **Shipped 2026-05-15 (router) + earlier on the server side; ROADMAP entry was stale.**

Feature-gated by `--features quic` on both `larql-server` and
`larql-router`. The transport wrapper lives in
`crates/larql-router-protocol/src/transport/quic.rs` (shared between
both crates so client + server code paths stay in sync).

Server side (`crates/larql-server/`):
- `connect_grid_channel` (`src/announce.rs:282-339`) parses `quic://`
  scheme on `--join` URLs and dispatches to the QUIC client endpoint;
  fingerprint pinning via `--quic-cert-fingerprint <SHA-256>`. Falls
  through to plain TCP gRPC for `http://` URLs.
- `--quic-cert-fingerprint` flag wired through to both `AnnounceConfig`
  and `AvailableConfig` (`src/bootstrap.rs:662, 1125-1128, 1176`).

Router side (`crates/larql-router/`):
- `--quic-port`, `--quic-cert`, `--quic-key`, `--quic-server-name`
  flags accept QUIC `Join` connections via the same QUIC endpoint.
- Self-signed TLS cert auto-generated when `--quic-cert`/`--quic-key`
  aren't passed; server logs the SHA-256 fingerprint for clients to
  pin.

Acceptance test:
`crates/larql-router-protocol/tests/test_quic_roundtrip.rs` — opens a
real QUIC endpoint, runs `Join` over the wrapper, asserts streaming
announce/heartbeat semantics survive the transport swap.

**Limitation (clarified in ADR-0019):** This is QUIC-as-TCP-replacement
(HTTP/2 over a single QUIC bi-stream). True HTTP/3 with per-stream
independence shipped separately under ADR-0019 (router) for the MoE
expert fan-out path, behind `--http3-shards` / `--http3-port`.

---

### G-MODEB. Self-assembling grid Mode B (ADR-0011)

#### GT5 — Gap-fill assignment

**Status**: ✅ **Shipped 2026-05-13 (router) + 2026-05-16 (server end-to-end test).**

Server-side `run_available_loop` in `crates/larql-server/src/announce.rs`
sends `AvailableMsg` → handles `AssignMsg` by calling
`shard_loader::download_and_load_shard` (atomic tar-then-rename, SHA-256
verification when a real content hash is provided) → sends `ReadyMsg`
or `RefuseMsg(reason="download_failed")` → loops until `AckMsg` from
the router. Public `try_once_available` entry point lets integration
tests drive a full handshake end-to-end. Router-side serves
`GET /v1/shard/{model_id}/{start}-{end}` as a streamed tar
(`crates/larql-server/src/routes/shard.rs`; documented in
[`docs/router-spec.md`](docs/router-spec.md) §4).

Wired end-to-end + tested:
- `tests/test_grid_mode_b.rs::mode_b_full_vertical_handoff` — protocol-level
  drive of the gRPC stream + direct `shard_loader` call (covers AssignMsg
  shape, hash propagation, tar unpack).
- `tests/test_grid_mode_b.rs::mode_b_try_once_available_drives_full_handshake`
  — exercises the production `try_once_available` loop end-to-end (Available
  → Assign → download → Ready → Ack) against an in-process router.
- `tests/test_grid_mode_b.rs::no_assign_when_gap_has_no_surviving_origin`
  — router declines to assign when no live replica can be origin.

**Known follow-up — GT5 hash semantics mismatch (P1):**
`vindex_identity_hash` (announce.rs:183) emits a 16-hex model-identity
tag (`u64.hash`-based), but `shard_loader` verifies SHA-256 of the
downloaded tar bytes against `AssignMsg.shard_hash`. Today this
"works" only because deployments pass an empty/placeholder hash so
the verification is skipped (see the `skip_hash` branch at
`shard_loader.rs:62`). Real hash verification — meaning the donor
hashes its on-disk shard at announce time and the spare verifies the
download against that — is a follow-up. ADR-0011 left this implicit;
the right shape is probably a new optional `shard_content_sha256`
field on `AnnounceMsg` distinct from `vindex_hash`.

**Mode A AssignMsg edge case:** `announce.rs:413-428` now logs a
descriptive warning when an already-serving Mode A stream receives an
unexpected AssignMsg (router bug — AssignMsg should target Mode B
available pool only). Previously logged "Mode B not implemented",
which was misleading because Mode B *is* implemented in
`run_available_loop`; the stub was for a different code path.

#### GT6 — Dynamic rebalancing

**Status**: ✅ **Shipped 2026-05-13 (router) + earlier on the server side; ROADMAP entry was stale.**

Server-side `announce.rs:416-442` handles `UnassignMsg` by polling
`requests_in_flight` for up to 30 s (`DRAIN_TIMEOUT`), then sending
`DroppingMsg(reason="reassigned")` and either exiting cleanly or
re-entering Mode B on the same gRPC stream via `run_available_loop`
when `available_after_drain` is configured (ADR-0011 §Phase B2).
Router-side rebalancer task lives at
`crates/larql-router/src/tasks/rebalancer/` (6-module folder shipped
in ADR-0016) with periodic ticks for replication, eviction,
imbalance detection, and hot-shard elevation. Latency-driven
rebalancing reads `LayerLatency.avg_ms` from heartbeats (GT3); under-
replication tick pulls spares from the available pool.

Tested:
- `tests/test_grid_drain_reassign.rs::drain_then_reassign_via_available_after_drain`
  — drives the full UnassignMsg → drain → DroppingMsg → re-enter Mode B path.
- Router-side replication + rebalancer covered in
  `crates/larql-router/tests/test_admin_rpcs.rs` and the chaos test.

---

### G-BENCH. Grid benchmarking (ADR-0012)

#### GT8 — `larql bench` grid/wire/transport extensions

**Status**: ✅ **Shipped 2026-05-15.** All flags except `--transport` (which
waits on GT7 QUIC) are live. The CLI now lives under `crates/larql-cli/src/commands/primary/bench/`
as a folder of single-responsibility modules with per-file 90%+ test coverage
gated by `crates/larql-cli/coverage-policy.json`.

**What shipped:**

- `--bench-grid` — 1..N shard sweep over a `--moe-shards` map; emits
  `shard_efficiency = tok/s / (N × single_shard_tok/s)` per row.
- `--wire f32,f16,i8` — one row per format against `--ffn`; the parity
  guarantee is at the codec level (`larql-inference/WirePreference` chooses
  the best mutually-supported format).
- `--concurrent N` — spawns N parallel client threads per backend; aggregate
  tok/s = sum(client.tok_per_s), p99 = max(client.p99). Production wire path
  is `std::thread::spawn` over the existing sync bench fn — no async refactor.
- `--output json` / `--output-file PATH` — emits the ADR-0012 envelope:
  `{timestamp, model, prompt, tokens, wire, concurrent, results[...]}`.

**Module layout** (`commands/primary/bench/`):
- `args.rs` — clap `BenchArgs`.
- `row.rs` — `BenchRow` + `BenchJsonRow` + `BenchJsonResult` + percentile helpers.
- `helpers.rs` — wire-list parser, concurrent aggregator, shard-efficiency math.
- `output.rs` — table renderer split into pure `Vec<String>` formatters.
- `ollama.rs` — Ollama side-by-side bench (curl wrapper isolated behind a
  `Fetcher` indirection so the orchestration is unit-testable).
- `engine.rs` — KV-engine post-processing helpers.
- `local.rs` — local Metal/CPU post-processing helpers.
- `remote_ffn.rs` — concurrent-row aggregation, FFN summary, label composer.
- `remote_moe.rs` — shard-map parser, MoE summary, label composer.
- `*_runtime.rs` — I/O wrappers (`run_larql`, `run_engine*`, `run_remote_ffn_bench`, `run_remote_moe_bench`). Excluded from the per-file coverage gate.
- `run.rs` — top-level dispatch. Excluded from the per-file coverage gate.

`--transport http,quic` is documented but deferred to GT7 (ADR-0010 QUIC).

**Acceptance**: `larql bench <vindex> --ffn URL --wire f32,f16 --output json --output-file out.json`
writes a JSON envelope containing both wire format results with their
`wire_bytes_per_tok` and `ms_per_tok.{mean,p50,p99}` fields populated.

#### GT9 — Criterion micro-benchmarks

**Status**: ✅ **Shipped 2026-05-07.**

`larql-inference/benches/wire_codec.rs` (encode f32 request, decode f32/f16 response, 30-layer batch) and `larql-router/benches/routing.rs` (route single layer, route_all 30/62 layers, update_heartbeat, rebuild_route_table) — both parameterised over server counts and hidden sizes with no hardcoded model names. `larql-router` gained `src/lib.rs` re-exporting `pub mod grid`. Makefile: `bench-wire`, `bench-routing`, `bench-grid`, `bench-all`.

**Spec**: ADR-0012 §Layer 2.

- `crates/larql-inference/benches/wire_codec.rs`: encode/decode throughput
  (MB/s) for f32/f16/i8 at hidden_size ∈ {2560, 4096, 5120}, seq_len ∈ {1, 32, 256}.
  Parameters read as `criterion::BenchmarkId` — no hardcoded model names.
- `crates/larql-router/benches/routing.rs`: `route()` hot path (ns/op at
  1/10/100 servers), `rebuild_route_table()` cold path, `update_heartbeat()`.

Run with: `make bench-wire` / `make bench-routing`.

#### GT10 — CI regression gate

**Status**: ✅ **Shipped 2026-05-15.** Scripts + comparator + baselines
directory all live; the script writes the first run as the baseline and
compares subsequent runs against it.

**Files:**
- `scripts/bench-grid-regress.sh` — wraps `larql bench ... --wire f32,f16 --output json`,
  compares against `bench/baselines/grid-<model>.json`. Saves the current
  run as baseline when none exists. Env vars: `LARQL_BENCH_VINDEX`,
  `LARQL_BENCH_FFN_URL`, optional `LARQL_TOK_PER_S_THRESHOLD` (default 0.05),
  `LARQL_P99_THRESHOLD` (default 0.10).
- `scripts/bench_compare.py` — pure-stdlib JSON diff. Fails if any `backend`
  in the baseline regresses tok/s by more than the threshold or rises p99
  by more than the threshold.
- `bench/baselines/README.md` — workflow for updating baselines after a
  deliberate perf improvement.

**Acceptance**: `LARQL_BENCH_VINDEX=… LARQL_BENCH_FFN_URL=… ./scripts/bench-grid-regress.sh gemma3-4b-q4k`
exits 0 on a clean run; exits 1 with a per-backend failure list if any
threshold trips.

---

### F-COLLECT. Parallelize shard collection in `forward_moe_stream_collect_with_timing`

**Status**: ✅ **Shipped 2026-05-02.** Both halves of the gRPC dispatch are
now parallel across shards:
- `forward_moe_stream_collect_with_timing` uses `std::thread::scope`,
  one OS thread per stream, joined into a single result vector.
  `ShardStream::result_rx` was wrapped in `std::sync::Mutex` to make
  `ShardStream: Sync` (the type-system requirement for parallel borrow).
- `forward_moe_stream_fire` uses `rayon::par_iter().enumerate().try_for_each(...)`
  with a single-shard fast path. The blocking residual-bytes / post-norm-bytes
  clones now happen across rayon workers instead of serially.

Verified on 2-shard local-loopback: per-layer collect ≈ 21 ms (~ equal to
1-shard collect time), confirming `collect ≈ max(per_shard.wall)` rather
than `sum` — the structural win. Real-network validation pending under
**F-FLY** below; loopback can't show the absolute tok/s improvement
because both shards finish nearly simultaneously and the savings sit
under M3 Max P-core saturation noise.

**Driver**: 2026-05-02 bottleneck analysis on the local Metal MoE path
vs the CPU/grid path (single shard, colocated). Both land at ~19 tok/s
because the grid sequentially blocks on each shard's `collect_with_timing()?`
in `crates/larql-inference/src/ffn/moe_remote.rs:1984`. With one shard,
sequential = max. With 2+ shards over real network, the per-layer
collect time stacks instead of overlapping.

**Concrete impact** (Gemma 4 26B-A4B, 30 MoE layers, top_k=8):

| Topology | Per-shard wall (RTT) | Collect/layer today (sequential) | Collect/layer fixed (parallel) | Saved per token |
|---|---|---|---|---|
| 1 shard local | ~8 ms | ~8 ms | ~8 ms (no change) | 0 |
| 2 shards LAN (~5 ms RTT) | ~5–10 ms | sum ≈ 10–20 ms | max ≈ 5–10 ms | ~5–10 ms × 30 layers = **150–300 ms/tok** |
| 4 shards LAN | ~5–10 ms | sum ≈ 20–40 ms | max ≈ 5–10 ms | ~15–30 ms × 30 layers = **450–900 ms/tok** |
| 4 shards cross-region (~50 ms RTT) | ~50 ms | sum ≈ 200 ms | max ≈ 50 ms | ~150 ms × 30 layers = **4500 ms/tok** |

The `fire` half of `forward_moe_stream_fire` already pushes to all
streams' channels in a non-blocking loop — concurrency exists at the
wire layer; the bug is the blocking serial collect on top.

**Fix**: change the collect loop from

```rust
for stream in streams.iter().take(n_streams) {
    let (partial, server_compute_ms) = stream.collect_with_timing()?;
    // accumulate into out
}
```

to a concurrent join. `tokio::join_all` if the call site is async, or
`std::thread::scope` / `rayon::par_iter().map(...)` if not (each
`collect_with_timing` blocks on a condvar inside `ShardStream`, so
parallelism comes from holding multiple condvars in flight). Picking
between these depends on whether `ShardStream::collect_with_timing` is
`Send + Sync`; check before deciding.

**Acceptance**: `LARQL_MOE_TIMING=1` summary line on a 2-shard run
reports `collect ≈ max(per_shard)`, not `sum(per_shard)`. End-to-end
tok/s on a 2-shard local-loopback run improves measurably.

**Strategic context**: this is the load-bearing primitive for the
"split in grids" axis of LARQL — the future Kimi K2.6 / DeepSeek V4
deployment shapes will need 8+ shards. Without this fix, the grid
scales backwards: more shards = more sequential collect time.

### F-LOCAL-MOE. Local Metal MoE optimisations (CPU staging + batched dispatch)

**Status**: Not started.

**Driver**: same 2026-05-02 bottleneck analysis. On the local Metal
MoE path, **67% of wall is CPU work**, only 33% is GPU active (51 ms
wall = 17 ms GPU + 33 ms CPU + sync). The GPU is barely loaded — the
CPU-side per-layer router + memcpy of 8 expert Q4_K byte slices into
staging buffers + commit/wait sync is dominating.

For the "run large models on consumer hardware" axis, every ms here
matters — the user runs LARQL on a single M3 Max, the grid isn't
available.

**Two levers, both CPU-path-safe**:

1. **Zero-copy expert byte aliasing**: today
   `gpu_moe_dispatch_with_scratch` memcpys ~300 KB per expert × 8 ×
   30 layers = ~72 MB of Q4_K bytes per token into pre-allocated
   staging buffers. The infra already exists —
   `MetalBackend::cached_buffer_for_bytes` does
   `new_buffer_with_bytes_no_copy` for the shard server's pre-staged
   path. Wiring it for the local path eliminates the per-layer
   memcpy entirely; experts alias the model's mmap directly.
   **Estimated win: 5–10 ms/tok.**

2. **Batched expert GPU dispatch**: today each MoE layer issues 24
   GPU dispatches (8 × `q4k_ffn_gate_up` + 8 × `geglu` + 8 ×
   `q4k_matvec` for down). Batching these into ~3 dispatches/layer
   using per-expert offsets into the already-staged buffers reduces
   dispatch overhead from ~720 calls/token to ~90.
   **Estimated win: 3–5 ms/tok.**

Combined: **8–15 ms/tok off the local path → 23–28 tok/s** on Gemma 4
26B-A4B Metal MoE (from 19.4 tok/s today).

**Acceptance**: `LARQL_GPU_TIMING=1` shows `cpu` shrunk by ~10 ms/tok;
`larql bench gemma4-26b-a4b-q4k-v2` shows ≥23 tok/s warm-state on
M3 Max with output unchanged.

### F-FLY. Remote multi-shard deployment on fly.io

**Status**: Not started — next session.

**Goal**: validate the HTTP CPU-path optimisations from the 2026-05-01 session
on a real network (LAN-class RTT ≥ 100 µs), not just M3 Max loopback. Most
of what we shipped is designed to win on real links but is invisible on
loopback (TCP_NODELAY, f16 wire). This is the apples-to-apples test that
tells us whether the in-room engineering translates to a deployable grid.

**Setup target (~2 hosts, then 4-8 if Phase 1 looks good)**:

- 1× client host (Mac dev box or fly.io VM): runs `larql run --moe-shards`
  with attention + dense FFN compute. Holds the 2 GB attention/router/dense
  weight set.
- N× shard hosts (fly.io VMs, ~16 GB RAM each): each runs
  `larql-server --experts START-END --grpc-port 9081 --uds-path ...`
  on a slice of the expert table. 26B-A4B has 128 experts × 30 layers;
  e.g., 4 shards × 32 experts × 30 layers ≈ 4 GB Q4_K + 2 GB working set
  per shard.
- Network: same fly.io region (intra-DC ~0.5 ms RTT) for Phase 1; a second
  region (cross-region ~30-100 ms RTT) for Phase 2 to stress the streaming
  overlap.

**What we expect to learn from this**:

1. Whether the **f16 wire** opt-in actually wins on real links (estimate:
   +3-5% on 1 Gbps, more on slower). On loopback it was within noise; we
   need real RTT to see the wire-bytes saving translate.
2. Whether **gRPC SPLIT default** (now on by default for gRPC) holds its
   ~12% steady-state win when the network leg is bigger than the dense
   FFN GPU leg (instead of comparable). The overlap math says the win
   grows when RTT > dense_FFN_time.
3. End-to-end tok/s ceiling on a real grid — we currently know loopback
   is ~19.7 tok/s; a multi-host grid should be slower per-token but
   throughput-scalable (more shards per host = more concurrent expert work).
4. Whether **predispatch (`batch` dispatch mode)** actually breaks
   generation on every multi-host setup or just on M3 Max loopback. We
   saw garbage output on loopback; might be a different story with real
   network timing.

**Prerequisites already in place** (from this session):

- gRPC streaming default-on for gRPC shards (~12% loopback gain,
  expected to grow on RTT-heavier links)
- TCP_NODELAY on accepted connections (defensive against tail-packet
  stalls on real LAN)
- f16 wire as opt-in (`LARQL_MOE_WIRE_F16=1`)
- Unix domain sockets (`--uds-path`, `unix:///path` URL) for same-host
  shard collocation
- `LARQL_HTTP_TIMING=1` per-call instrumentation (encode / send_total /
  recv_body / decode breakdown)
- `LARQL_MOE_TIMING=1` per-token MoE summary (route / collect / server
  compute / network estimate)
- 9.6× CPU MoE speedup on the shard side (bench: 30-layer sweep
  221 → 22.9 ms; production: 2.3 → ~19.7 tok/s end-to-end on M3 Max
  loopback)

**fly.io specifics worth pinning down before deploy**:

- VM size for shards: 26B-A4B vindex is ~16 GB on disk; needs ~10 GB
  RSS at warmup. `performance-cpu-2x` (~7 GB RAM) won't fit a full
  shard; need `performance-cpu-4x` (~14 GB) at minimum, or shard the
  vindex finer.
- Vindex distribution: cheapest is to ship the full 16 GB to each shard
  and let `--experts START-END` cap working set; alternative is per-shard
  vindex slicing (`larql slice` exists but needs a per-shard variant).
- Persistent volume vs in-memory: with `--warmup-walk-ffn` the boot
  cost is ~6-7 s; if VMs reboot per deploy, that adds up. Consider
  fly.io persistent volumes for the vindex.
- Health check: `/v1/health` is already there.
- Authentication: the existing `--api-key` flag works but a multi-tenant
  fly.io setup probably wants per-shard token rotation (out of scope for
  Phase 1).

### F0. CPU MoE correctness — RESOLVED ✅

**Status**: Closed 2026-05-01.

Smoke-test `larql run output/gemma4-26b-a4b-q4k.vindex "The capital of
France is" --max-tokens 5` (no `--moe-shards`, no `--metal`) returns
**"Paris."** End-to-end CPU path on the per-layer Q4_K hybrid-MoE
vindex now produces the correct answer; the M-CPU kernel work
(NEON SDOT direct-Q4K + scratch reuse + correct hybrid-combine
ordering, see `larql-inference/ROADMAP.md → M-CPU-1..6`) shared the
code path with the server-side fix that landed 2026-04-30, so the
local route inherited the correctness for free.

The historical analysis below is preserved as forensics for future
CPU-vs-Metal divergence debugging — the diff-and-localise pattern
generalised better than the specific bug.

**Historical context (2026-04-27, pre-M-CPU work):**

The per-expert refactor + `experts_packed.bin` removal landed without a
correctness end-to-end check. `larql run` on the 26B-A4B vindex via the CPU
MoE path produces incoherent text ("ever own로 el"), while `larql run --metal`
on the same vindex produces "Paris." The server-side remote-expert endpoint
inherits the same bug because `run_single_expert` and `cpu_moe_forward` share
the same per-expert compute.

**What I tried that did not help:**
- Aligning `cpu_moe_forward`'s router-norm input to `h_norm` (matching Metal's
  `cpu_moe_route(&h_norm, ...)` convention) — different garbage, not "Paris".
- Swapping gate/up row order in the `[2*inter, hidden]` slice — different
  garbage, not "Paris".
- Verified `dequantize_q4_k` is bit-identical to the `larql_models` reference
  via `tests/test_q4k_parity.rs` on synthetic ramp data (3 super-blocks of
  varied content, plus round-trip-within-noise).
- Verified `inter_padded` handling matches Metal's convention (zero-pad
  hidden_state to `inter_padded`, dequant down at `hidden * inter_padded`).

**What's still suspect:**
- Q4_K dequant on the **real per-layer file's bytes** has not been compared
  against Metal's GPU dequant. Synthetic parity ≠ real-data parity.
- The **gate/up convention in HF Gemma 4** could differ from what
  `quantize_moe_entries` assumes about the source BF16 layout.
- BLAS `sgemv` on Apple Accelerate vs Metal's `q4k_matvec` shader could have
  precision drift at 26B scale, though both should be IEEE-754 correct.

**Why the bench numbers were misleading:**
`bench_expert_server` measured `forward_moe` warm at 1.91 ms and the
`cpu_moe_forward` floor at 0.10 ms. Post-fix the floor jumped to 1.81 ms (18×).
The 0.10 ms number was the buggy old code silently returning empty buffers
when the dequant length didn't match the bytes — fast because no work was
happening. This was not flagged because no test compared **output values**,
only latency.

**Diagnosis status (2026-04-27, via `larql parity` + dump-and-diff):**

Layer-by-layer cosine-similarity diff between CPU `predict_q4k` and Metal
`predict_q4k_metal` on the 26B-A4B vindex, using `LARQL_CPU_DUMP_LAYERS` +
`LARQL_DUMP_RESIDUALS`:

| Stage at layer 0 | cos(cpu, metal) |
|---|---|
| h_embed (input to layer 0) | 1.000000 |
| h_post_attn (post-attention) | 1.000000 |
| layer_out (post-FFN+MoE+combine) | **0.626708** ← divergence |

Attention is correct on layer 0; the divergence is in the **FFN + MoE +
combine** between `h_post_attn` and `layer_out`. The CPU MoE block routes
to the same top-K experts as Metal at layer 0 (verified via `MOE_DEBUG=1`:
both pick `[79, 114, 16, 92, 89, 101, 67, 46]` with the same `moe_out_rms`).
Per-expert math is provably correct (parity test). The bug is therefore in
how `run_moe_layer_cpu` composes h1 (dense), h2 (MoE), the outer
post-FFN norm, and `layer_scalar` — and it has drifted from Metal's
`metal/decode/moe_combine.rs::apply_outer_combine`.

`larql parity` v1 shipped (CLI subcommand, `larql-cli/src/commands/diagnostics/parity.rs`)
with `--component moe-expert` + `--component moe-block` and `--backends reference,cpu`.
Run on the 26B-A4B vindex the tool reports:

| Component | reference vs cpu max abs diff | Verdict |
|---|---|---|
| `moe-expert` layer 0 / expert 0 | 4.3 × 10⁻⁶ | within fp32+BLAS noise |
| `moe-block` layer 0 (router → top-K → K experts → sum → post-norm) | 8.4 × 10⁻⁵ | within fp32+BLAS noise |

So the entire MoE expert pathway — Q4_K dequant, gate matmul, up matmul,
activation, down matmul, router, top-K, weighted sum, post-experts norm — is
mathematically correct end-to-end. The bug producing garbage on `larql run`
is **outside** the MoE block. Suspect surface area:

- attention block (Q/K/V proj, RoPE, softmax, O proj) — Metal vs CPU
- hybrid combine: `h1 + h2 → moe_post_outer_norm → + h_post_attn` in
  `larql-inference/src/vindex/q4k_forward.rs::layer_step`
- `apply_layer_scalar` and PLE (`apply_per_layer_embedding`) afterwards
- per-position iteration loop on prefill (`for pos in 0..seq_len`)

**Root cause (further localised 2026-04-27):**

The CPU and Metal paths use **two different forward implementations** for
hybrid-MoE Q4_K vindexes — they have drifted:

- **Metal**: `predict_q4k_metal` builds `FullPipelineLayer` per layer and
  calls `backend.decode_token(&layers, ...)`. Hybrid MoE handled by
  `decode_token_with_moe` → `gpu_moe_dispatch`. This works.
- **CPU**: legacy `q4k_forward.rs::predict_q4k_step` →
  `run_moe_layer_cpu` (hand-rolled) → `cpu_moe_forward` per position +
  hand-rolled hybrid combine (`combined = h1 + h2`,
  `combined_normed = outer_norm(combined)`, `h_out = h_post_attn + combined_normed`).
  Doc comment in that function says it's "verified against HF bf16 via
  residual-cosine diff in the Metal `diag.rs` dumps" — but the file has
  since drifted from Metal and the verification is stale. This produces
  garbage end-to-end on Gemma 4 26B-A4B.

Routing-convention fix (apply router_norm to `h_norm`, not raw `h`,
matching Metal's `cpu_moe_route(&h_norm, ...)`) was applied to
`cpu_moe_forward` and `MoeRouterWeights::route`, with regression tests in
`larql-compute/src/cpu/ops/moe/mod.rs`. Necessary but not sufficient — the
hybrid combine in `run_moe_layer_cpu` is still wrong.

**Next steps for F0 (proper fix):**

The cleanest path is to **delete `run_moe_layer_cpu` and route CPU
predictions through the same `FullPipelineLayer` + `decode_token` pipeline
Metal uses**, swapping `MetalBackend` for `CpuBackend`. That requires
`CpuBackend::decode_token` to support Q4 layers (it currently doesn't —
`predict_q4k_metal` literally `expect()`s "need Metal with Q4 kernels").

Either:
- Implement `CpuBackend::decode_token` for Q4 layers — substantial work
  porting the Metal kernels' algorithm to CPU + BLAS, but unifies the two
  paths and resolves all class-of-bug drifts at once.
- Patch `run_moe_layer_cpu` to match Metal's exact hybrid combine. Faster
  but leaves the dual-path drift surface in place; another knob will go
  out of sync next session.

A `larql parity --component layer` (parity v2) component would catch this
class of bug going forward — diffing the **full hybrid layer output**
between CPU and Metal would have surfaced the combine drift immediately.
That's the right next investment.

**Implication for the remote-MoE story:**
The wire format, `--experts` shard ownership (with the off-by-one fix),
the per-expert byte-table API, and the per-layer Q4_K layout all work
correctly. What does **not** work is the CPU numerical compute on the
server side. Until F0 is closed, "remote MoE on Gemma 4 26B-A4B" is
plumbing-correct but inference-incorrect — clients pointing at a remote
larql-server shard will get garbage output. Workaround: use `--metal` for
all-local generation; remote-MoE is on hold.

---

Functional gaps from the 2026-04-27 server review. Numbering is stable so we
can reference items in commits and reviews.

### F1. Router-side expert-shard fan-out
**Files**: `crates/larql-router/src/main.rs`, `crates/larql-router/src/grid.rs`,
`crates/larql-router-protocol/proto/*.proto`.
The grid router fans out `walk-ffn` by layer ranges only. For MoE, the
remote-expert client (`RemoteMoeBackend` in `larql-inference`) carries the
expert→shard map itself; nothing on the router side. Means clients can't just
point at the router for MoE. Add `POST /v1/expert/{layer}/{id}` and
`POST /v1/expert/batch` to the router, with shard discovery via the existing
gRPC announce stream. Pairs with **F11** (topology endpoint).

### F2. Streaming HTTP infer (SSE)
**Files**: `crates/larql-server/src/routes/infer.rs` (new sibling
`infer_stream.rs`).
`/v1/infer` is single-shot — full output buffered, no incremental tokens. WS
has it (`WS_CMD_INFER`) but most chat UIs talk SSE. Add
`POST /v1/infer/stream` with `text/event-stream`. Same generation loop, yield
each token. Mid-generation cancellation on client disconnect (see **F16**).

### F3. `/metrics` (Prometheus)
**Files**: `crates/larql-server/src/main.rs`, new `crates/larql-server/src/metrics.rs`.
No latency histograms, no per-endpoint counters, no rate-limit drops, no
shard-call durations today. Wire `metrics` + `metrics-exporter-prometheus` (or
hand-rolled). Histograms for: `walk-ffn` per `layer_count`, `forward_moe` per
`top_k`, queue wait, auth failures, rate-limit drops, shard-call latency.

### F4. Graceful shutdown with in-flight drain
**Files**: `crates/larql-server/src/main.rs`.
SIGTERM today probably cuts long-running walks. Standard axum + tokio shutdown
signal: stop accepting, drain N seconds (configurable), hard-kill. Important
for grid rolling restarts.

### F5. Readiness vs liveness split
**Files**: `crates/larql-server/src/routes/health.rs`, `routes/mod.rs`.
`/v1/health` returns `{status, uptime, requests_served}`. Add `GET /v1/ready`
returning 503 until weights are loaded (under `--warmup-walk-ffn` or first
lazy load); include `model_id`, `mode`, `version`, `git_sha`, `format`
(per-layer vs legacy) in the readiness payload. Standard k8s liveness/readiness
split.

---

## P1: Active

### Q1.10 Reduce `routes/stream.rs::handle_stream_infer` (327 LOC) — deferred

The remaining open code-quality item from the 2026-05-01 audit. The other
nine (Q1.1–Q1.9) shipped — see "Completed → 2026-05-01 (continued) — Q1
code-quality cleanup". Q1.10 is deferred until N0.1 (OpenAI Chat
Completions SSE) forces a similar streaming state-machine shape; the
two should share infrastructure. Effort estimate: ~3 hours when picked up.

---

### F6. Replica round-robin + retry on shard failure
**Files**: `crates/larql-router/src/grid.rs`.
Router picks first owning shard; no load-balancing across replicas, no retry
on 5xx. `--shards "0-15=A,0-15=B"` doesn't fan evenly today.

### F7. KV-cache prefix sharing for chat
**Files**: `crates/larql-inference/src/layer_graph/generate/*`,
`crates/larql-server/src/routes/infer.rs`.
Every `/v1/infer` call is fresh prefill. For chat (long shared system prompt +
short user turn) prefix-caching is a 5–10× decode-time win. Needs a
`session_id`-keyed KV cache.

### F8. Vindex hot-swap admin endpoints
**Files**: `crates/larql-server/src/routes/` (new `admin.rs`),
`crates/larql-server/src/state.rs` (mutable model registry).
`POST /v1/admin/vindex/load`, `DELETE /v1/admin/vindex/{id}`,
`POST /v1/admin/vindex/reload`. Admin-key-gated (see **F14**). Otherwise every
model swap is a process restart.

### F9. Binary wire format for `expert/batch`
**Files**: `crates/larql-server/src/routes/expert.rs`,
`crates/larql-inference/src/ffn/moe_remote.rs`.
A K=8 batch on Gemma 4 26B-A4B is ~90 KB JSON per call. The
`application/x-larql-ffn` binary format already exists for `walk-ffn`; mirror
it for `expert/batch`. Expected 3–5× wire reduction.

### F10. OpenAI-compat `/v1/chat/completions` — superseded by N0

This item scoped only the chat completions endpoint shallowly. See
**N0** in the "Great new functionality" section above for the full
plan: chat completions + completions + responses + embeddings +
models, with streaming, tools, structured output, and constrained
decoding. F10 is left here for cross-references; the work happens
under N0.

### F11. Expert topology endpoint
**Files**: new `crates/larql-server/src/routes/topology.rs`.
`GET /v1/expert/topology` returns `{model_id, layers, num_experts, owned: [start,end]}`.
Lets clients build the shard map dynamically instead of having it baked in.
Pairs with **F1** (router fan-out).

### F12. Batched infer
**Files**: `crates/larql-server/src/routes/infer.rs`.
`/v1/infer` takes one prompt today. RAG workloads send N prompts; one batched
call across them amortises router/dispatch overhead. Either accept
`prompts: [...]` or new `/v1/infer/batch`.

### T3. Review follow-up — server hygiene ✅ done 2026-04-26

**Scope**: follow-up from review of `larql-server` focused on magic strings,
modularity, cleanliness, tests, and clippy.

Shipped:
- `X-Forwarded-For` is ignored by default for rate limiting; new
  `--trust-forwarded-for` opt-in is for deployments behind a trusted proxy.
- HTTP protocol constants added for shared health path, API prefix,
  bearer prefix, and binary FFN content type.
- Route path literals in `routes/mod.rs` centralized as named constants so
  single-model and multi-model routing drift is easier to spot.
- `load_single_vindex` now takes a `LoadVindexOptions` struct instead of
  an 11-argument call and repeated `too_many_arguments` clippy allows.
- Embed endpoints now return the standard `{"error": ...}` JSON envelope
  for errors instead of a mix of plain text and JSON.
- Server-local clippy cleanup removed the repeated `too_many_arguments`
  exemptions from the vindex loading path.

Follow-up worth keeping open:
- Consider a route-registration macro/table if route count keeps growing.

### T1. Test coverage — functional tokenizer + uncovered routes ✅ done 2026-04-26

**Outcome**: 49.1% → **58.0% line**, 56.4% → **65.3% function**. 345 → 402 tests.

**Root cause fixed**: added `functional_tokenizer()` (WordLevel, France→0 etc.) to
`tests/common/mod.rs`. The empty BPE tokenizer that previously blocked all
tokenize-dependent routes is now supplemented by a real in-memory tokenizer that
maps test words to embeddings with known KNN hits.

**Files moved:**

| File | Before | After |
|---|---|---|
| `band_utils.rs` | 35% | **100%** |
| `routes/describe.rs` | 48% | **95%** |
| `routes/walk.rs` | 38% | **96%** |
| `ratelimit.rs` | 70% | **98%** |
| `routes/walk_ffn.rs` | 54% | **77%** |
| `routes/patches.rs` | 63% | **91%** |
| `routes/relations.rs` | 83% | **91%** |

**Remaining hard ceiling** (no path forward without real weights or real sockets):

| File | Coverage | Reason |
|---|---|---|
| `grpc.rs` | 0% | Needs full gRPC server+client; defer |
| `routes/stream.rs` | 0% | WebSocket — needs `tokio-tungstenite`; defer |
| `routes/explain.rs` | 11% | Calls `get_or_load_weights()`; rest gated on real model |
| `embed_store.rs` | 25% | Reads real f16 embedding files |
| `main.rs` | 0% | CLI entrypoint; skip |

### T2. Test coverage — remaining reachable paths ✅ done 2026-04-26

**Current**: 74.2% line / 81.2% function. 478 tests.

**Completed this pass:**
- `grpc.rs` 0% → **65%** — 28 direct gRPC handler tests (health, stats, describe, walk, select, relations, walk_ffn, infer, stream_describe)
- Magic strings: `"probe"` → `PROBE_RELATION_SOURCE`; `"ok"` → `HEALTH_STATUS_OK`; infer mode strings in grpc.rs; WebSocket message types in stream.rs (`WS_TYPE_*`, `WS_CMD_*`)
- `embed_store.rs` 25% → **98% line** — tiny f16 mmap fixtures cover open, size validation, lookup, L1 cap, out-of-range, subnormal/inf/nan conversion.
- `announce.rs` 6% → **56% line** — extracted deterministic message builders for announce, heartbeat, dropping, and grid bearer metadata.
- `main.rs` boot/loading/discovery helpers moved into `bootstrap.rs`; `bootstrap.rs` has **92% function** coverage for parse/discovery/serve-alias/options behavior.
- `routes/stream.rs` 0% → **65% line** — WebSocket JSON message builders plus pure describe-message planning cover missing-entity, no-model, and functional edge streaming cases.
- `routes/infer.rs` 32% → **56% line** and `routes/explain.rs` 18% → **46% line** via request/default deserialization tests and response-formatting helpers.
- `routes/embed.rs` 67% → **87% line** — binary embed/logits parsing extracted into helpers; HTTP tests cover binary success, malformed JSON, truncated binary input, hidden-size mismatches, no-model errors, and cacheable single-token JSON/binary responses.
- `routes/walk_ffn.rs` 77% → **80% line** — validation helpers now cover layer selection precedence, missing layers, seq_len handling, overflow, and latency rounding.

**Remaining hard ceiling:**

| File | Current | Gap | What to add |
|---|---|---|---|
| `main.rs` | 0% | 237 lines | Tokio binary entrypoint; boot orchestration is covered through `bootstrap.rs` |
| `bootstrap.rs` | 43% | 134 lines | Real vindex load path still requires filesystem fixtures with full vindex assets |
| `routes/stream.rs` | 65% | 148 lines | Full WebSocket socket loop still needs a client harness such as `tokio-tungstenite` |
| `routes/explain.rs` | 46% | 167 lines | Main path gated on `get_or_load_weights()` and real inference trace |
| `routes/infer.rs` | 56% | 82 lines | Prediction paths need real or injectable inference backend |
| `routes/embed.rs` | 87% | 74 lines | Remaining positive logits path requires loadable weights/lm_head fixture |
| `routes/walk_ffn.rs` | 80% | 125 lines | Remaining full-output path requires loadable weights/FFN fixture |
| `routes/warmup.rs` | 80% | ~15 lines | `warmup_hnsw=true` warn path (HNSW not enabled) |
| `announce.rs` | 56% | ~78 lines | Remaining gap is live gRPC stream lifecycle and retry loop |

### G1. Cold-start profile ✅ done 2026-04-26
**Findings**: walk-ffn cold cost decomposes into two distinct phases:

1. **First walk-ffn ever**: ~1.27 s + ~2.9 GB RSS — lazy
   `get_or_load_weights` builds the f32-decoded gate-vector cache,
   loads `lm_head.bin` + `norms.bin`. One-shot regardless of which
   layer was requested. Confirmed not Metal init: a prior gate-KNN
   walk only adds 2 MB.
2. **First touch of each new layer**: ~17 ms + ~11 MB RSS — kernel
   page-fault for the layer's `interleaved_q4k.bin` slice (gate +
   up + down, ~22 MB on disk). Linear in number of cold layers.

Warm steady state is **0.2–0.3 ms/layer**. The 50× cold:warm ratio
is mostly phase 1; phase 2 is ~50× cheaper.

Conclusion: the win lives in phase 1 — pre-load weights at boot.
Mmap prefetch is a 12 ms one-shot for all 30 layers (negligible).
Both wired in **G2** below.

### G2. `/v1/warmup` endpoint + `--warmup-walk-ffn` flag ✅ done 2026-04-26
**Impact (measured on Gemma 26B)**: first walk-ffn **1247 ms → 12.6 ms (99×)** at the cost of +3.2 GB pre-allocated RSS and ~1.3 s boot delay.

Shipped:
- `POST /v1/warmup` accepting `{layers, skip_weights, warmup_hnsw}`
  (all optional). Returns `{weights_loaded, weights_load_ms,
  layers_prefetched, prefetch_ms, hnsw_built, hnsw_warmup_ms,
  total_ms}`.
- `larql-server --warmup-walk-ffn` boot flag — calls the same code
  path before the listener binds. Goes through
  `warmup_model_async` (`spawn_blocking`) because the boot point
  is already inside the tokio runtime.
- The endpoint runs the work on a blocking pool so the runtime
  stays responsive.

### G3. Dual-host gRPC self-assembling grid ✅ done 2026-04-26
**Live-validated** (single-host two-port simulation, exercises the
same code path as a real LAN-distributed grid):

- Shards launched with `--join http://router:50052 --grid-key <s>
  --public-url http://shard:port` register automatically; router
  logs `Grid: server joined layers=0-14` and updates coverage.
- `total_layers_covered` field on the router is the operator's
  view of grid completeness.
- Killed shard A → router logs `Grid: server left`, coverage drops.
  Layer-5 request returns HTTP 400 `"layer 5 has no owning shard"`
  (clean error, not hang). Layer 22 (live shard B) stays at 0.3 ms.
- Restart killed shard → it auto-rejoins, coverage returns to 30,
  layer 5 routes successfully (cold-page first request: 13.9 ms).
- README "Recommended setup" updated with the `--grid-port` /
  `--join` recipe (separate edit pending).

The gRPC mechanism is production-ready as of this validation.
True cross-host RTT measurement is forward-looking (G3a below).

### G3a. Cross-host RTT measurement *(forward-looking)*
**Status**: open. Requires two physical machines on the same LAN.
The same-host validation establishes correctness; cross-host
measures the additional TCP overhead per fan-out.

## P2: Forward-looking

### G-SCALE. Run T-class models on grid (Kimi K2.6, DeepSeek V4 scale)

**Driver**: LARQL's strategic axis is "run large models on consumer
hardware OR split across grids." T-class MoE models (Kimi K2 ≈ 1T total
params, top-K ≈ 8; DeepSeek V3 ≈ 671B, top-K=2; future K2.6 / V4 likely
similar shape) can't fit on any single consumer machine — the grid
deployment shape is **the only way** to run them locally.

**What changes vs Gemma 4 26B A4B (today's reference)**:

| Dimension | Gemma 4 26B-A4B | Kimi K2 (~1T) | DeepSeek V3 (~671B) |
|---|---|---|---|
| Total params | 26B | ~1T | 671B |
| Layers | 30 | ~60 | 61 |
| Experts/layer | 128 | ~384 | 256 |
| Top-K active | 8 | 8 | 8 |
| Active params/token | ~5B | ~37B | ~37B |
| Q4_K vindex size (estimate) | 16 GB | ~600 GB | ~400 GB |

**Implications for the grid primitives**:

1. **Memory-conscious shard layout**. A T-class model's expert table is
   100× our current. With 16 GB consumer-class RAM per shard, K2 needs
   ~40 shards just to fit. Per-shard memory targeting matters: each
   shard owns a tight `(layer, expert_id)` set of mmap pages and never
   loads the rest. The `--units PATH` JSON manifest already supports
   per-(layer, expert) ownership; **G5 below** (per-shard expert routing
   in router-protocol) lights it up at the router layer.
2. **Parallel shard collect is non-negotiable**. With 40+ shards,
   sequential collect would compound to seconds/token. **F-COLLECT**
   above is the prerequisite.
3. **Streaming expert byte transfer**. T-class expert weights per layer
   may not fit in RAM even on a fat shard if it owns many experts. The
   shard's mmap+page-fault behaviour does the right thing today (only
   active expert pages are paged in), but **G4 mmap residency control**
   below becomes operationally important — long-running shards need
   `madvise(DONTNEED)` after a layer to reclaim RSS.
4. **Router-side fan-out batching**. With 40+ shards and 30+ layers,
   per-layer round-trips dominate. Multi-layer `forward_moe_predispatch`
   (already exists) becomes the default rather than an opt-in; the
   pass-1 approximation cost is negligible compared to 40-shard ×
   30-layer sequential RTT.

**Status**: Forward-looking. **F-COLLECT** + **G5** + **G4** are the
direct prerequisites; once those land we should attempt a multi-shard
deployment of one T-class model end-to-end as a capability check, even
if perf is exploratory rather than production-tuned.

### G4. mmap residency control endpoint
**Impact**: For long-running shards under memory pressure, expose
`POST /v1/mmap/advise {layers, advice: "willneed"|"dontneed"}` so
operators can trim RSS or pre-warm specific layer ranges without
restarting.

### G5. Per-shard expert routing
**Impact**: For DeepSeek-V3+/Kimi K-class models (1k+ experts), shard
by expert ID within a layer rather than by layer range. Needs an
`ExpertRoute` message type in `larql-router-protocol` and
GridState dispatch updates. Mentioned in larql-vindex P2. Subsumed by
**F1** (router-side expert fan-out) at the router layer; G5 covers the
router-protocol changes specifically.

### G6. Live router-shard topology change
**Impact**: Today shards are static (`--shards` flag at router boot).
For ops convenience, expose `POST /v1/router/shards` (admin-gated)
to add/remove a shard without restarting the router. Pair with
`--grid-port` health checks.

### F13. OpenTelemetry tracing exporter
**Files**: `crates/larql-server/src/main.rs`.
Per-request spans across HTTP→shard fan-out. `tracing_subscriber::fmt` is the
only output today. Wire `tracing-opentelemetry` + OTLP exporter, configurable
via `--otel-endpoint`. Pairs with **F3** (metrics).

### F14. Per-key quotas + audit log
**Files**: `crates/larql-server/src/auth.rs`, `crates/larql-server/src/main.rs`.
Single API key today; no per-key quotas, no rotation, no scoped tokens. Add
`--api-keys keys.toml` (name + role + per-key rate). Structured audit on
patches + admin ops to a configurable sink (file / stdout / OTel).

### F15. RBAC (read-only vs admin keys)
**Files**: `crates/larql-server/src/auth.rs`, all mutating routes.
Today any key can patch the loaded model. Add `role` per key
(read / infer / patch / admin). Mutating endpoints (`patches/apply`,
`insert`, future `admin/*`) require the matching role.

### F16. Mid-generation cancellation on HTTP infer
**Files**: `crates/larql-server/src/routes/infer.rs`.
Client disconnect on `/v1/infer` waits for the full max_tokens. Wire
`tokio::select!` against an axum `OnUpgrade`-style cancellation token (or just
poll the connection on each decode step) to abort early.

### F17. Structured-output / grammar-constrained generation
**Files**: `crates/larql-inference/src/layer_graph/generate/*`,
`crates/larql-server/src/routes/infer.rs`.
`{format: "json", schema: ...}` or `{grammar: "gbnf:..."}` on `/v1/infer`.
Constrains decoding by masking the logits to grammar-valid tokens. Standard
ML-server feature; missing today.

### F18. Log-prob / perplexity endpoint
**Files**: new `crates/larql-server/src/routes/logprobs.rs`.
`POST /v1/logprobs {prompt, top_k}` — return per-token log-probabilities.
Needed for ranking, classification, and eval workflows.

### F19. OpenAPI schema route
**Files**: new derive macro setup using `utoipa` (or hand-rolled).
`GET /openapi.json`. Required for SDK codegen, `kubectl explain`-style
tooling, and external API consumers. Today external consumers read the
README.

### F20. Compression negotiation
**Files**: `crates/larql-server/src/main.rs`.
No `Content-Encoding: gzip|zstd` advertised; relies on a reverse proxy. Wire
`tower-http::compression`. Particularly useful for `walk-ffn` JSON responses
on slow links.

### F21. `/v1/stats` per-layer mmap residency
**Files**: `crates/larql-server/src/routes/stats.rs`.
Existing `q4k_ffn` block exposes cache slots/bytes; extend with per-layer
hot/cold (resident vs paged-out) so operators can see what `--release-mmap-after-request`
actually buys them.

### F22. Persistent patches
**Files**: `crates/larql-server/src/session.rs`,
`crates/larql-server/src/routes/patches.rs`.
Patches are session-scoped today; no on-disk overlay. Add a durable
`POST /v1/patches/save` + auto-apply on boot. Pairs with **F8** (hot-swap)
so a patched model survives restart.

### F23. Python HTTP client SDK
**Files**: new `crates/larql-python/src/http_client.rs` (or new crate).
`larql-python` is walk-only against a local vindex; no HTTP client. Add a
`pip install larql` package speaking the server's HTTP API (sync + async),
mirroring the OpenAI Python SDK shape. Pairs with **F10** (OpenAI compat) so
the SDK is a thin wrapper over the OpenAI client.

---
