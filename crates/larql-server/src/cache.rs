//! TTL cache for DESCRIBE results.

use std::collections::HashMap;
use std::sync::RwLock;
use std::time::{Duration, Instant};

struct CacheEntry {
    value: serde_json::Value,
    inserted_at: Instant,
}

/// Simple in-memory TTL cache for DESCRIBE responses.
pub struct DescribeCache {
    entries: RwLock<HashMap<String, CacheEntry>>,
    ttl: Duration,
}

impl DescribeCache {
    pub fn new(ttl_secs: u64) -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
            ttl: Duration::from_secs(ttl_secs),
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.ttl.as_secs() > 0
    }

    /// Build a cache key from describe parameters.
    pub fn key(model_id: &str, entity: &str, band: &str, limit: usize, min_score: f32) -> String {
        Self::key_scoped(None, model_id, entity, band, limit, min_score, false, 0.0)
    }

    /// Cache key including the session scope.
    ///
    /// DESCRIBE answers from a session's patch overlay when one is supplied,
    /// so the session is part of the identity of the cached value. Omitting it
    /// would let a hit computed under one session answer another session's
    /// request — one tenant's suppressions served to another, with no error
    /// and nothing in the logs to show it happened.
    // Every argument is a dimension of the answer; folding them into a struct
    // would move the same eight names one line down without removing any.
    #[allow(clippy::too_many_arguments)]
    pub fn key_scoped(
        session_id: Option<&str>,
        model_id: &str,
        entity: &str,
        band: &str,
        limit: usize,
        min_score: f32,
        coherence: bool,
        min_coherence: f32,
    ) -> String {
        // Anything that changes the ANSWER belongs in the key. Coherence
        // renames targets and can drop edges, so a key without it would serve a
        // relabelled answer to a caller that asked for the raw one — the same
        // class of bug as sharing one entry across two overlays, and just as
        // invisible in a log.
        format!(
            "{}:{}:{}:{}:{}:{}:{}:{}",
            session_id.unwrap_or("-"),
            model_id,
            entity,
            band,
            limit,
            min_score as u32,
            coherence as u8,
            (min_coherence * 1000.0) as i32
        )
    }

    /// Get a cached value if it exists and hasn't expired.
    pub fn get(&self, key: &str) -> Option<serde_json::Value> {
        let entries = self.entries.read().ok()?;
        let entry = entries.get(key)?;
        if entry.inserted_at.elapsed() < self.ttl {
            Some(entry.value.clone())
        } else {
            None
        }
    }

    /// Insert a value into the cache.
    pub fn put(&self, key: String, value: serde_json::Value) {
        if let Ok(mut entries) = self.entries.write() {
            // Evict expired entries if the cache is getting large.
            if entries.len() > 10000 {
                let now = Instant::now();
                entries.retain(|_, e| now.duration_since(e.inserted_at) < self.ttl);
            }
            entries.insert(
                key,
                CacheEntry {
                    value,
                    inserted_at: Instant::now(),
                },
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_when_ttl_zero() {
        let cache = DescribeCache::new(0);
        assert!(!cache.is_enabled());
    }

    #[test]
    fn enabled_when_ttl_nonzero() {
        let cache = DescribeCache::new(60);
        assert!(cache.is_enabled());
    }

    #[test]
    fn put_and_get() {
        let cache = DescribeCache::new(60);
        let key = DescribeCache::key("model", "France", "knowledge", 20, 5.0);
        let value = serde_json::json!({"entity": "France"});
        cache.put(key.clone(), value.clone());
        assert_eq!(cache.get(&key), Some(value));
    }

    #[test]
    fn miss_on_unknown_key() {
        let cache = DescribeCache::new(60);
        assert_eq!(cache.get("nonexistent"), None);
    }

    #[test]
    fn expired_entry_returns_none() {
        let cache = DescribeCache::new(0); // 0 → disabled, but let's test with 1ns TTL
                                           // Can't easily test TTL expiration in a unit test without sleeping,
                                           // so we test the disabled path instead.
        let key = "test".to_string();
        cache.put(key.clone(), serde_json::json!("val"));
        // With TTL=0, is_enabled() is false, so caller won't even check cache.
        // But internally get() will return None because elapsed >= 0s TTL.
        assert_eq!(cache.get(&key), None);
    }

    #[test]
    fn key_format() {
        // The leading `-` is the session slot, held by a placeholder for the
        // global (sessionless) entry. It is not decoration: without a slot,
        // a global lookup for model "x" and a session named "x" asking about
        // the same entity would collide, and DESCRIBE would answer one
        // tenant's browse out of another's overlay.
        let key = DescribeCache::key("gemma-3-4b-it", "France", "knowledge", 20, 5.0);
        // Trailing `0:0` is coherence-off with no threshold, i.e. the raw
        // argmax rendering. It is pinned here so that turning coherence on can
        // never quietly reuse this entry.
        assert_eq!(key, "-:gemma-3-4b-it:France:knowledge:20:5:0:0");
    }

    #[test]
    fn different_params_different_keys() {
        let k1 = DescribeCache::key("model", "France", "knowledge", 20, 5.0);
        let k2 = DescribeCache::key("model", "Germany", "knowledge", 20, 5.0);
        let k3 = DescribeCache::key("model", "France", "syntax", 20, 5.0);
        assert_ne!(k1, k2);
        assert_ne!(k1, k3);
    }

    /// The session is part of the cached value's identity, because DESCRIBE
    /// answers from that session's patch overlay. Two sessions asking the same
    /// question must not share an entry: the first tenant to warm the cache
    /// would otherwise decide what every other tenant sees, and a suppression
    /// applied by one would surface in another's browse view.
    #[test]
    #[test]
    fn coherence_is_part_of_the_key() {
        // Same question, different rendering of the answer. Sharing one entry
        // would hand a relabelled result to a caller that asked for the raw one.
        let raw = DescribeCache::key_scoped(None, "m", "Paris", "all", 20, 0.0, false, 0.0);
        let coh = DescribeCache::key_scoped(None, "m", "Paris", "all", 20, 0.0, true, 0.0);
        let filt = DescribeCache::key_scoped(None, "m", "Paris", "all", 20, 0.0, true, 0.3);
        assert_ne!(raw, coh, "coherence on/off must not share an entry");
        assert_ne!(coh, filt, "two thresholds must not share an entry");
    }

    #[test]
    fn key_scoped_separates_sessions_and_global() {
        let global = DescribeCache::key_scoped(None, "m", "Paris", "all", 20, 0.0, false, 0.0);
        let a = DescribeCache::key_scoped(Some("tenant-a"), "m", "Paris", "all", 20, 0.0, false, 0.0);
        let b = DescribeCache::key_scoped(Some("tenant-b"), "m", "Paris", "all", 20, 0.0, false, 0.0);
        assert_ne!(a, b, "two sessions must not share a cache entry");
        assert_ne!(a, global, "a session must not share the global entry");
        assert_ne!(b, global);
        // Same session + same question is still a hit, or the cache is pointless.
        assert_eq!(
            a,
            DescribeCache::key_scoped(Some("tenant-a"), "m", "Paris", "all", 20, 0.0, false, 0.0)
        );
    }

    /// `key` is the unscoped spelling of `key_scoped(None, ..)`; if they ever
    /// diverge, existing global callers silently miss the cache forever.
    #[test]
    fn key_matches_key_scoped_with_no_session() {
        assert_eq!(
            DescribeCache::key("m", "Paris", "all", 20, 0.0),
            DescribeCache::key_scoped(None, "m", "Paris", "all", 20, 0.0, false, 0.0)
        );
    }
}
