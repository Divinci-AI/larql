# Runtime model lifecycle — design notes (pre-implementation)

Status: **design only** — no endpoints, no state-machine code, no event
stream. This document exists to settle the state machine and inventory
the seams a future `POST/DELETE /v1/runtime/model` would need, before
any of it gets built. See `/v1/runtime` (`routes/runtime.rs`,
`runtime_stats.rs`) for the read-only surface this lifecycle work sits
underneath.

## 1. What exists today (grounded in code, not assumption)

- `AppState.models: Vec<Arc<LoadedModel>>` and `v3_models: Vec<Arc<V3Model>>`
  are populated **once**, in `bootstrap::serve` (`bootstrap/mod.rs:68-190`),
  before `AppState` is constructed (`bootstrap/mod.rs:276`). They are
  plain `Vec`s — not `RwLock`, not `ArcSwap`. There is no code path,
  anywhere in the crate, that mutates them after boot. **This is the
  seam**: nothing about routing, request dispatch, or the OpenAI
  handlers assumes an immutable model list, but the storage itself is
  immutable. Every other lifecycle question is downstream of fixing
  this one fact.
- Route dispatch already goes through `AppState::model(id)` /
  `AppState::served(id)` (`state.rs`) at request time, searching the
  Vec by id. So once the Vecs are interior-mutable, **no router or
  handler code needs to change to pick up a newly-loaded or
  newly-removed model** — the seam is narrow.
- **The router topology is a separate, bigger seam.** `bootstrap/mod.rs:340-353`
  picks `single_model_router` or `multi_model_router` **once**, based
  on `state.is_multi_model()` (`models.len() + v3_models.len() > 1`)
  at boot. Those two routers have different route tables (multi-model
  adds `/v1/{model_id}/...` prefixed paths). Consequences:
  - 0 models and 1 model are **both** "single" mode — so an idle
    server picking up its first model, or a single model being
    swapped for a different one, never crosses the single/multi
    boundary. **This is the tractable first slice.**
  - Going from 1 loaded model to 2 **would** flip which router
    variant is correct. Axum's `Router` isn't swappable in place
    without extra machinery. **Loading a second concurrent model into
    an already-bound single-model server is a materially bigger
    change than "swap the bound model" — don't scope them together.**
- `LoadedModel.weights: OnceLock<RwLock<ModelWeights>>` +
  `weights_init: Mutex<()>` solve *lazy first-load* single-flighting
  within one already-registered model. They have no "unset" — a
  `LoadedModel` cannot have its weights freed while keeping the rest
  of the struct alive. **Unload is necessarily "drop the whole
  `Arc<LoadedModel>`/`Arc<V3Model>`", not a finer-grained operation.**
- Because generation handlers already clone the model's `Arc` for the
  duration of a request (e.g. `model_arc = model.clone()` before
  `spawn_blocking` throughout `routes/openai/*`), **removing a model
  from `AppState.models` is memory-safe immediately** — Rust's
  refcounting keeps any in-flight request's own clone alive until that
  request finishes, with no coordination required. The drain pattern
  below exists for *policy* (don't claim "unloaded" — or start loading
  a replacement — while the old one might still be resident), not for
  safety.
- **In-flight accounting is inconsistent across the two things that
  would need it.** `LoadedModel.requests_in_flight: AtomicU32` exists
  but is walk-ffn/grid-shard-scoped only (its own doc comment says so;
  it's what GT6 drain reads — see below). My new
  `RuntimeRecorder.active_requests` is OpenAI-generation-scoped but
  **server-wide**, not per-model — fine for today's single-model-focus
  `/v1/runtime`, but it cannot answer "is *this* model still serving a
  request" once two models can be bound at once. **`V3Model` has no
  in-flight counter of any kind today** — zero fields for it, the same
  gap `runtime_stats` just closed for *timing* on V3 (no instrumentation
  existed; it got added once at the `generate_v3_request` choke point).
  Unload-safety accounting for V3 needs the identical move: one counter
  at that same choke point, not one per route.
- **A drain-then-signal pattern already ships** — GT6 in `announce.rs`
  (`drain_requests`, `DRAIN_TIMEOUT`, `DroppingMsg`): on `UnassignMsg`
  from the grid router, stop accepting new shard work, poll
  `requests_in_flight` every 100 ms up to a timeout, then announce
  `Dropping`. **This is the right shape to imitate for local unload**
  (stop resolving new requests → poll a counter with a timeout →
  proceed) but it is not directly reusable code: it's about leaving a
  *distributed grid's* routing table, not about freeing a *local*
  `Arc`. There is no equivalent "stop resolving new local requests"
  primitive today — that's new.
- **No graceful shutdown exists at all.** No SIGTERM/SIGHUP handling,
  no quiescing, anywhere in `bootstrap/`. The server runs until killed.
  A lifecycle endpoint would be the *first* thing in this codebase
  that needs "stop routing new work to X, then wait" outside the grid
  context.
- **Session and N1 KV state are keyed by model-id string, not by
  `Arc` identity.** `ResponseKvCache::take` (`response_kv/mod.rs:202-215`)
  already refuses a resume when `entry.model_id != model_id` — a real,
  existing guard. But it can't detect "same id, reloaded with
  different weights" (a swapped quantization under the same name,
  say) — the guard only compares strings. **Any lifecycle design that
  lets a model id be reloaded (not just removed) must sweep session +
  KV-cache entries for that id as part of the transition**, or a
  resumed KV state can silently pair with the wrong weights.
- `/v1/runtime`'s `memory.resident_bytes` is `getrusage`'s **peak**
  RSS (documented as such in `runtime_stats.rs`), which is monotonic
  for the process lifetime. **Once unload exists, this becomes
  materially misleading**: unloading a model will not move this number
  down, ever, even if the memory really was freed. Not fixing this
  now — flagging it because it's a direct, foreseeable consequence of
  work already merged, not a new speculative gap.

## 2. Scope for the *first* lifecycle cut

Given the router-topology seam above, the first tractable slice is
**single-bound-model lifecycle only** — matches the "tiny Mac app"
target anyway (one local model, not a fleet):

- load into an idle (zero-model) server
- swap the bound model for a different one (implies unload-then-load)
- unload back to idle

**Explicitly out of scope for the first cut** (bigger, separable
seams): holding two independently-loadable models at once on one
server (router topology change), any multi-model `/v1/{id}/...`
lifecycle, and anything about the grid/router protocol.

## 3. State machine

One state machine per **bound-model slot** (today: exactly one slot,
since multi-model dynamic loading is out of scope). `/v1/runtime`'s
`model` field already reports `null` in every state except `ready` and
`generating` — that's unchanged.

```text
                                   ┌────────────────────────────┐
                                   │                            │
                                   ▼                            │
                              ┌─────────┐   load ok        ┌─────────┐
             load requested   │ loading │ ───────────────► │  ready  │
        ┌─────────────────────┤         │                  │         │◄───┐
        │                     └────┬────┘                  └────┬────┘    │
        │                          │ load failed                │         │ generation
        │                          ▼                             generation  completes
   ┌─────────┐               ┌─────────┐                        starts    │
   │  idle   │◄──────────────┤ failed  │                         │        │
   │         │  surfaced,    │(logged, │                         ▼        │
   └────┬────┘  slot freed   │ no slot)│                   ┌────────────┐ │
        │                    └─────────┘                   │ generating │─┘
        │ (already idle —                                  └─────┬──────┘
        │  no-op / 409,                                          │
        │  see §4)                                     unload requested
        │                                                        │
        │                          unload requested (from ready) │
        │                                        ┌───────────────┴──┐
        │                                        ▼                  ▼
        │                                  ┌────────────┐    ┌────────────┐
        └──────────────────────────────────┤ unloading  │◄───┤ unloading  │
                     drain complete,        │ (draining) │    │ (draining, │
                     Arc dropped            └────────────┘    │  cancel    │
                                                                │  requested)│
                                                                └────────────┘
```

Two notes the diagram can't carry on its own:

- **`unloading` while `generating`** is not a separate state — it's
  `unloading` with the in-flight-generation drain still counting down.
  The distinction the user's edge-case list asks about
  ("cancelled generation" vs "let it finish") is a **policy knob on
  the unload call**, not an extra state: drain-to-completion (default,
  matches GT6) vs. best-effort cancel. Today's generation loops have
  no cooperative-cancel hook (the `/v1/infer` timeout path documents
  exactly this: on timeout it drops the `JoinHandle` and lets the
  blocking thread finish in the background regardless — see
  `routes/openai/completions.rs`'s timeout comment). So "cancelled
  generation" on unload would, for now, mean the same thing the
  infer-timeout already means: stop *waiting* for it, not stop it
  running. Actually killing an in-flight blocking generation thread
  is a separate, harder problem this design does not solve.
- **`failed` is not sticky.** A failed load returns the slot straight
  to `idle` (with the error surfaced to the caller) — there is no
  persistent "broken" state to recover from, because nothing was
  committed to `AppState.models` on failure.

## 4. Edge cases, resolved explicitly

| Case | Resolution |
|---|---|
| Load B while A is loaded (single-slot scope) | Rejected as a **swap request**, not a raw load — the caller says "replace," the server does unload(A)-then-load(B) as one sequenced operation, never holding both. If the caller instead calls plain "load B" while a slot is occupied: reject (409-shaped), point at the swap operation. |
| Unload while generation active | `ready → unloading`: stop resolving *new* requests to this slot immediately (an instant Vec/slot mutation); poll the model's in-flight counter with a timeout (GT6 shape); drop the `Arc` once it hits zero or the timeout elapses. See §1 on V3's missing counter — this is the blocking prerequisite for V3 unload specifically. |
| Load while a load is already active | Single-flighted per slot, same shape as `LoadedModel.weights_init` today but at the *admin* layer, not the weights layer — a second concurrent load call on the same slot is rejected outright, not queued (queuing hides operator mistakes; better to fail loud). |
| Failed load | Slot returns to `idle`; nothing was written to `AppState.models`; error surfaced. No retry state to manage. |
| Failed unload (drain timeout) | Two legitimate policies, must pick one explicitly rather than let it be implicit: (a) force-drop the `Arc` anyway once the timeout elapses (in-flight requests keep working off their own clone, per §1 — this is safe, just not clean), or (b) refuse and stay in `unloading`, requiring an explicit force-unload. Recommend (a) with a logged warning, since it mirrors GT6's own choice (`drain_requests` logs and proceeds past its timeout rather than blocking forever). |
| Cancelled generation (mid-unload) | See §3 — "stop waiting," not "stop running," matches the existing infer-timeout precedent. Don't promise more than the codebase can deliver today. |
| VINDEX2 vs VINDEX3 lifecycle | Same state machine, different mechanics: V2 unload drops `Arc<LoadedModel>` (weights may or may not be loaded yet, per `weights: OnceLock`); V3 drops `Arc<V3Model>` (operands are lowered at bind time — `Vindex3Runtime::prepare`, see `vindex3.rs` module docs — so a V3 "load" is heavier up front and a V3 "unload" has nothing lazy left to *not* free). The important asymmetry is the in-flight counter gap noted in §1, not the drop itself. |

## 5. Explicit non-goals for this pass

- No `POST`/`DELETE` endpoints yet.
- No event stream (`runtime.model.loading` etc.) — polling
  `/v1/runtime` is sufficient until a real client (the Mac app)
  demonstrates it isn't.
- No multi-model dynamic loading (router-topology change, §1/§2).
- No memory-accounting fix for the peak-RSS-after-unload confusion
  (§1, last bullet) — noted for whoever picks this up next, not solved
  here.
- No cooperative generation cancellation — "drain or force-drop after
  timeout" is the ceiling of what's honestly deliverable right now.

## 6. What actually has to land first, in order

1. Make `AppState.models` / `v3_models` interior-mutable for a single
   slot (the narrow seam from §1) — this alone is real, reviewable,
   low-risk work with no user-visible behavior change yet.
2. Add the missing V3 in-flight counter at the `generate_v3_request`
   choke point (mirrors exactly how V3 timing was added for
   `/v1/runtime`) — needed before "unload while generating" can be
   honest for V3, independent of whether the endpoints exist yet.
3. Only then: the two endpoints, built directly against the state
   machine in §3 and the edge-case table in §4 — at which point they
   should be close to mechanical.
