//! Compiled patch overlays, addressed by the content hash of their patch set.
//!
//! # Why a cache rather than a session
//!
//! A patch overlay used to be *state*: built by `POST /v1/patches/apply`, held
//! in one instance's session map, and read by whatever request happened to land
//! on that instance. larql-service runs several instances with no session
//! affinity, so an edit applied on instance A did not exist on B or C, and
//! nothing about the request said which instance it needed. Losing that state —
//! to a TTL, a redeploy, or a recycle — was silent data loss dressed up as a
//! model that had never been edited.
//!
//! Here the overlay is a *value*: a pure function of (base artifact, patch set),
//! and the request carries the patch set that identifies it. The cache only ever
//! saves the work of rebuilding one. A miss costs latency; it can never cost
//! correctness, and that asymmetry is the whole design.
//!
//! # Why the cache is nevertheless load-bearing
//!
//! Building an overlay clones the model's base index. The mmap'd artifact is
//! shared by refcount, so the ~10 GiB does not move — but `GateStore::clone`
//! deliberately resets `f16_decode_cache`, `warmed_gates` and `hnsw_cache`,
//! because caches are working memory rather than durable state. Rebuilding per
//! request would therefore discard the HNSW index and every decoded layer each
//! time, on a 35-layer model with 337k features.
//!
//! So the cache keeps overlays *warm*, which is exactly what the session map was
//! quietly doing. The change is what the entry is keyed by: content, not tenant.
//! One tenant's overlay is now reusable by any instance that is handed the same
//! patch set, and an evicted entry is rebuilt rather than lost.
//!
//! # Sizing
//!
//! Bounded by entry count, and small by default. The memory is dominated by
//! those warm caches, not by the handful of operations in a patch set, so an
//! overlay costs roughly what a patched session used to cost. The cache lives on
//! [`LoadedModel`], so unloading a model drops its overlays with it — a reloaded
//! model cannot be served a stale overlay compiled against the previous
//! artifact, which is the one invalidation bug this design must not have.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use larql_vindex::{PatchedVindex, VindexPatch};

/// Default number of compiled overlays held per model.
///
/// Derived, not guessed. It was 8 by judgement until the footprint was measured
/// on 2026-09-01, and 8 does not survive the arithmetic.
///
/// What an overlay costs, measured against testdata/tiny-vindex with
/// `cargo test --test mem_probe -- --ignored --nocapture`:
///
///   - COLD: ~28 KB. Gate data is mmap-backed — 0 of 8 layers heap-resident —
///     so `base().clone()` shares the artifact by refcount and copies nothing
///     of consequence. This is the reassuring half.
///   - WARM: unbounded per overlay, and this is the half that matters.
///     `GateStore::clone` resets `f16_decode_cache` and `warmed_gates`, and each
///     refills as layers are touched. One warmed layer costs
///     `features_in_layer × hidden × 4` bytes.
///
/// For the served Gemma-4-E2B artifact (337,920 features over 35 layers, hidden
/// 1536) that is ~59 MB per warmed layer and ~2.08 GB for a fully warmed
/// overlay. Against a 32 GiB limit already holding a ~10 GiB base:
///
///   8 overlays -> ~27.3 GB   no headroom; this service has been OOM-killed before
///   4 overlays -> ~19.0 GB   room to spare
///
/// Hence 4. Nothing enforces the per-overlay bound today: the container runs
/// without `--max-gate-cache-layers`, so `gate_cache_max_layers` is 0, meaning
/// unlimited. Capping that is the better lever if this ever gets tight, because
/// it bounds each overlay rather than counting them — raise this constant only
/// alongside it.
pub const DEFAULT_OVERLAY_CACHE_ENTRIES: usize = 4;

pub fn cache_entries_from_env() -> usize {
    std::env::var("LARQL_OVERLAY_CACHE_ENTRIES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_OVERLAY_CACHE_ENTRIES)
}

/// A tiny string-keyed LRU.
///
/// Generic over the value only so it can be exercised without constructing a
/// PatchedVindex. Eviction and touch ordering are the parts of this file most
/// likely to be quietly wrong, and "obviously correct by inspection" is how the
/// 404-only replay fallback survived for its entire life.
struct Lru<V> {
    map: HashMap<String, V>,
    /// Most-recently-used at the back.
    order: VecDeque<String>,
    capacity: usize,
}

impl<V: Clone> Lru<V> {
    fn new(capacity: usize) -> Self {
        Self {
            map: HashMap::new(),
            order: VecDeque::new(),
            capacity,
        }
    }

    fn touch(&mut self, key: &str) {
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            let k = self.order.remove(pos).expect("position just found");
            self.order.push_back(k);
        }
    }

    fn get(&mut self, key: &str) -> Option<V> {
        let found = self.map.get(key).cloned();
        if found.is_some() {
            // Order must reflect USE, not insertion. Without this the entry a
            // busy tenant is hammering is evicted as readily as one nobody has
            // asked for in an hour.
            self.touch(key);
        }
        found
    }

    fn put(&mut self, key: String, value: V) {
        if self.capacity == 0 {
            return;
        }
        if self.map.insert(key.clone(), value).is_none() {
            self.order.push_back(key);
        } else {
            self.touch(&key);
        }
        while self.order.len() > self.capacity {
            if let Some(evicted) = self.order.pop_front() {
                self.map.remove(&evicted);
            }
        }
    }

    fn len(&self) -> usize {
        self.map.len()
    }
}

pub struct OverlayCache {
    inner: Mutex<Lru<Arc<PatchedVindex>>>,
    capacity: usize,
    hits: AtomicU64,
    misses: AtomicU64,
    builds: AtomicU64,
}

impl OverlayCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Mutex::new(Lru::new(capacity)),
            capacity,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            builds: AtomicU64::new(0),
        }
    }

    pub fn with_env_capacity() -> Self {
        Self::new(cache_entries_from_env())
    }

    pub fn get(&self, key: &str) -> Option<Arc<PatchedVindex>> {
        let found = self.inner.lock().unwrap().get(key);
        if found.is_some() {
            self.hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
        }
        found
    }

    pub fn put(&self, key: String, overlay: Arc<PatchedVindex>) {
        self.inner.lock().unwrap().put(key, overlay);
    }

    pub fn note_build(&self) {
        self.builds.fetch_add(1, Ordering::Relaxed);
    }

    pub fn stats(&self) -> OverlayCacheStats {
        OverlayCacheStats {
            entries: self.inner.lock().unwrap().len(),
            capacity: self.capacity,
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            builds: self.builds.load(Ordering::Relaxed),
        }
    }
}

impl Default for OverlayCache {
    fn default() -> Self {
        Self::with_env_capacity()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct OverlayCacheStats {
    pub entries: usize,
    pub capacity: usize,
    pub hits: u64,
    pub misses: u64,
    pub builds: u64,
}

/// Cache key for one compiled overlay.
///
/// Three parts, and each excludes a specific wrong answer.
///
/// **model** — an overlay is operations applied to a specific base. Without
/// this, a reloaded artifact could be served an overlay compiled against the one
/// it replaced: same operations, wrong base, no symptom but plausible output.
///
/// **scope** — the caller. This is a SECURITY boundary, not a nicety. The sha a
/// request supplies is taken as the key while the patches it supplies become the
/// value, and nothing ties the two together, so a caller can file arbitrary
/// content under any hash it likes. Unscoped, that is cross-tenant cache
/// poisoning: `sha256("[]")` is the same constant every workspace with no edits
/// sends, so poisoning that one key once would serve one caller's overlay to all
/// of them. Scoping by caller confines a poisoned entry to whoever poisoned it.
///
/// Residual, stated rather than hidden: within one scope a caller that pairs a
/// stale sha with fresh patches still files the wrong content under its own key
/// and will read it back. That is a caller bug with a blast radius of one
/// tenant, which is a different class from the above.
///
/// **sha** — the patch set itself.
pub fn overlay_key(model_id: &str, scope: Option<&str>, patch_set_sha: &str) -> String {
    format!("{model_id}:{}:{patch_set_sha}", scope.unwrap_or("-"))
}

/// Canonical hash of a patch set.
///
/// Computed over the patches **in the order given**, because that order is
/// semantically load-bearing: `apply_patch` accumulates, so the same operations
/// applied in a different sequence are not guaranteed to be the same overlay.
/// The caller (which holds the write-ahead log and its sequence numbers) is the
/// only party that can canonicalise the order, so this hashes what it is handed
/// rather than sorting behind its back.
///
/// Callers normally send their own `sha`; this exists so the server can verify
/// one, and so tests do not have to hand-roll the same hash.
pub fn patch_set_sha(patches: &[VindexPatch]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    for p in patches {
        hasher.update(p.description.as_deref().unwrap_or("").as_bytes());
        hasher.update([0u8]);
        for op in &p.operations {
            // Serialise the op rather than hand-picking fields: a new field on
            // PatchOp that this forgot would produce equal hashes for unequal
            // overlays, which is the one failure a content-addressed cache
            // cannot tolerate.
            let bytes = serde_json::to_vec(op).unwrap_or_default();
            hasher.update((bytes.len() as u64).to_le_bytes());
            hasher.update(&bytes);
        }
        hasher.update([1u8]);
    }
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn patch(name: &str, feature: usize) -> VindexPatch {
        VindexPatch {
            version: 1,
            base_model: "test".into(),
            base_checksum: None,
            created_at: "2026-08-31T00:00:00Z".into(),
            description: Some(name.to_string()),
            author: None,
            tags: vec![],
            operations: vec![larql_vindex::PatchOp::Delete {
                layer: 26,
                feature,
                reason: None,
            }],
        }
    }

    #[test]
    fn same_patches_same_hash() {
        assert_eq!(
            patch_set_sha(&[patch("a", 1), patch("b", 2)]),
            patch_set_sha(&[patch("a", 1), patch("b", 2)])
        );
    }

    #[test]
    fn a_different_feature_is_a_different_hash() {
        assert_ne!(patch_set_sha(&[patch("a", 1)]), patch_set_sha(&[patch("a", 2)]));
    }

    // Order is part of the identity, not noise to be normalised away:
    // apply_patch accumulates, so a reordered set is not the same overlay.
    #[test]
    fn order_changes_the_hash() {
        assert_ne!(
            patch_set_sha(&[patch("a", 1), patch("b", 2)]),
            patch_set_sha(&[patch("b", 2), patch("a", 1)])
        );
    }

    #[test]
    fn an_empty_set_is_stable_and_distinct() {
        assert_eq!(patch_set_sha(&[]), patch_set_sha(&[]));
        assert_ne!(patch_set_sha(&[]), patch_set_sha(&[patch("a", 1)]));
    }

    // The bug this key exists to prevent: two models, same edits, one overlay.
    #[test]
    fn the_key_separates_models() {
        assert_ne!(
            overlay_key("gemma-4", None, "abc"),
            overlay_key("gemma-3", None, "abc")
        );
    }

    // Cross-tenant cache poisoning, excluded structurally.
    //
    // The supplied sha becomes the key and the supplied patches become the
    // value, with nothing tying them together — so a caller CAN file arbitrary
    // content under any hash. Scoping is what stops that being everyone else's
    // problem.
    #[test]
    fn the_key_separates_callers_sharing_a_hash() {
        assert_ne!(
            overlay_key("m", Some("wl:tenant-a"), "abc"),
            overlay_key("m", Some("wl:tenant-b"), "abc")
        );
    }

    // The sharpest case: every workspace with no edits sends the hash of an
    // empty set, which is one fixed constant. Unscoped, poisoning that single
    // key would reach every un-edited tenant at once.
    #[test]
    fn the_empty_patch_set_is_not_one_shared_key() {
        let empty = patch_set_sha(&[]);
        assert_ne!(
            overlay_key("m", Some("wl:tenant-a"), &empty),
            overlay_key("m", Some("wl:tenant-b"), &empty)
        );
    }

    #[test]
    fn a_caller_scope_is_distinct_from_no_scope() {
        assert_ne!(overlay_key("m", None, "abc"), overlay_key("m", Some("wl:a"), "abc"));
    }

    // ── LRU ───────────────────────────────────────────────────────────
    // An evicted overlay is rebuilt, never lost, so none of this is a
    // correctness boundary. It is a cost boundary, and a cache that evicts the
    // wrong entry rebuilds a 35-layer overlay on a request that should have
    // been a hit.

    fn lru(capacity: usize) -> Lru<&'static str> {
        Lru::new(capacity)
    }

    #[test]
    fn evicts_the_least_recently_used() {
        let mut c = lru(2);
        c.put("a".into(), "A");
        c.put("b".into(), "B");
        c.put("c".into(), "C");
        assert_eq!(c.get("a"), None, "a was oldest and should be gone");
        assert_eq!(c.get("b"), Some("B"));
        assert_eq!(c.get("c"), Some("C"));
    }

    #[test]
    fn a_read_protects_an_entry_from_eviction() {
        let mut c = lru(2);
        c.put("a".into(), "A");
        c.put("b".into(), "B");
        // Reading `a` makes `b` the least recently used, so `b` goes.
        assert_eq!(c.get("a"), Some("A"));
        c.put("c".into(), "C");
        assert_eq!(c.get("a"), Some("A"), "a was just read and must survive");
        assert_eq!(c.get("b"), None);
    }

    #[test]
    fn re_putting_a_key_does_not_grow_the_order() {
        let mut c = lru(2);
        c.put("a".into(), "A");
        c.put("a".into(), "A2");
        c.put("b".into(), "B");
        // If a duplicate put pushed a second order entry, this would evict "a"
        // while its map entry stayed — a slow leak plus a wrong eviction.
        assert_eq!(c.get("a"), Some("A2"));
        assert_eq!(c.get("b"), Some("B"));
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn capacity_zero_stores_nothing_rather_than_leaking() {
        let mut c = lru(0);
        c.put("a".into(), "A");
        assert_eq!(c.get("a"), None);
        assert_eq!(c.len(), 0);
    }

    #[test]
    fn counters_distinguish_a_hit_from_a_miss() {
        let cache = OverlayCache::new(4);
        assert!(cache.get("nope").is_none());
        let s = cache.stats();
        assert_eq!((s.hits, s.misses, s.entries), (0, 1, 0));
    }
}

// ═══════════════════════════════════════════════════════════════
// Resolving the overlay a request should be answered from
// ═══════════════════════════════════════════════════════════════

/// The patch set a request carries.
///
/// Sending this on every read is what makes an instance interchangeable: the
/// request states which overlay it wants, so no instance has to have been told
/// beforehand, and no instance can answer from a stale one it happens to hold.
#[derive(serde::Deserialize, Clone, Default, utoipa::ToSchema)]
pub struct PatchSetRef {
    /// Content hash of `patches`. Optional — the server computes it when
    /// omitted, and verifies nothing: a caller that sends a wrong hash gets its
    /// own overlay filed under its own key, which is wrong only for that caller.
    #[serde(default)]
    pub sha: Option<String>,
    /// The patches themselves, in the order they must be applied.
    ///
    /// May be empty in two quite different senses. `patches: []` with no `sha`
    /// means "this tenant has no edits" — a real, cacheable overlay identical to
    /// the base. Omitting `patches` while giving a `sha` is the negotiated form:
    /// see [`resolve_overlay`].
    #[serde(default)]
    #[schema(value_type = Option<Vec<serde_json::Value>>)]
    pub patches: Option<Vec<VindexPatch>>,
}

impl PatchSetRef {
    /// The key this request's overlay is filed under.
    pub fn key(&self, model_id: &str, scope: Option<&str>) -> String {
        let sha = match (&self.sha, &self.patches) {
            (Some(s), _) => s.clone(),
            (None, Some(p)) => patch_set_sha(p),
            (None, None) => patch_set_sha(&[]),
        };
        overlay_key(model_id, scope, &sha)
    }
}

/// Resolve the overlay for a request that carries a patch set.
///
/// Hit: serve the compiled overlay. Miss with patches inline: build, file, and
/// serve it. Miss with only a hash: answer `409` naming the hash, because the
/// server genuinely cannot know what that caller meant — the caller then retries
/// with the patches inline. That is the whole of the hash-only negotiation, and
/// it exists so a workspace with hundreds of edits need not put them on the wire
/// for every read. Callers should keep sending patches inline until they are
/// large enough to be worth a round trip; a few hundred bytes per op means most
/// never will.
///
/// The overlay is built from the model's **base**, not from its global
/// `patched` state. The patch set is a complete statement of what this caller's
/// model should be, so honouring global applies underneath it would make the
/// result depend on instance history again — the exact property this removes.
pub fn resolve_overlay(
    model: &crate::state::LoadedModel,
    req: &PatchSetRef,
    scope: Option<&str>,
) -> Result<Arc<PatchedVindex>, crate::error::ServerError> {
    let key = req.key(&model.id, scope);

    if let Some(hit) = model.overlay_cache.get(&key) {
        return Ok(hit);
    }

    let patches = match &req.patches {
        Some(p) => p,
        None => {
            return Err(crate::error::ServerError::Conflict(format!(
                "patch_set_unknown: no compiled overlay for {key}; retry with `patches` inline"
            )));
        }
    };

    model.overlay_cache.note_build();
    let base = model.patched.blocking_read().base().clone();
    let mut overlay = PatchedVindex::new(base);
    for p in patches {
        overlay.apply_patch(p.clone());
    }
    let overlay = Arc::new(overlay);
    model.overlay_cache.put(key, Arc::clone(&overlay));
    Ok(overlay)
}

/// Run `f` against the overlay this request should be answered from.
///
/// Precedence, and why:
///
/// 1. **`patch_set`** — the request said what it wants. Instance-independent,
///    so it must win over anything this instance happens to be holding.
/// 2. **session** — the legacy path, kept so a client can be migrated without a
///    flag day. An unpatched or unknown session reads like the global state,
///    which is why an expired session degrades to "unedited" rather than to an
///    error.
/// 3. **global** — no session, no patch set.
///
/// The guard discipline matches `infer` and `describe`: a read guard on the
/// sessions map, never a write, and never held across an `.await`.
pub fn with_overlay<R>(
    state: &crate::state::AppState,
    model: &crate::state::LoadedModel,
    session_id: Option<&str>,
    patch_set: Option<&PatchSetRef>,
    f: impl FnOnce(&PatchedVindex) -> R,
) -> Result<R, crate::error::ServerError> {
    if let Some(req) = patch_set {
        // The session id doubles as the cache scope — see `overlay_key`. It is
        // the same value that already isolates the legacy session path, so a
        // caller cannot reach another caller's entries by either route.
        let overlay = resolve_overlay(model, req, session_id)?;
        return Ok(f(&overlay));
    }

    let sessions = state.sessions.sessions_blocking_read();
    if let Some(patched) = session_id
        .and_then(|sid| sessions.get(sid))
        .and_then(|s| s.patched())
    {
        return Ok(f(patched));
    }
    drop(sessions);
    let patched = model.patched.blocking_read();
    Ok(f(&patched))
}
