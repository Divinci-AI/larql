//! Per-session PatchedVindex management.
//!
//! Each session gets its own PatchedVindex overlay. The base vindex is shared
//! (readonly). Patches applied via the session API are isolated to that session.
//!
//! Sessions are identified by a `X-Session-Id` header. If no header is present,
//! patches go to the global (shared) PatchedVindex.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use larql_vindex::PatchedVindex;
use tokio::sync::RwLock;

use crate::state::LoadedModel;

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Per-session state — an isolated PatchedVindex overlay.
pub struct SessionState {
    pub patched: PatchedVindex,
    last_accessed: AtomicU64,
}

impl SessionState {
    pub fn new(base: larql_vindex::VectorIndex, _now: Instant) -> Self {
        Self {
            patched: PatchedVindex::new(base),
            last_accessed: AtomicU64::new(now_millis()),
        }
    }

    /// Update last-accessed timestamp; takes &self so read-lock holders can call it.
    pub fn touch(&self) {
        self.last_accessed.store(now_millis(), Ordering::Relaxed);
    }

    pub fn last_accessed_millis(&self) -> u64 {
        self.last_accessed.load(Ordering::Relaxed)
    }
}

/// Manages per-session PatchedVindex instances.
#[allow(dead_code)]
pub struct SessionManager {
    sessions: RwLock<HashMap<String, SessionState>>,
    ttl: Duration,
}

impl SessionManager {
    pub fn new(ttl_secs: u64) -> Self {
        Self {
            sessions: RwLock::new(HashMap::new()),
            ttl: Duration::from_secs(if ttl_secs == 0 { 3600 } else { ttl_secs }),
        }
    }

    /// Get or create a session's PatchedVindex.
    #[allow(dead_code)]
    pub async fn get_or_create(
        &self,
        session_id: &str,
        model: &Arc<LoadedModel>,
    ) -> PatchedVindex {
        let mut sessions = self.sessions.write().await;

        // Evict expired sessions opportunistically (max 10 per call).
        let now_ms = now_millis();
        let ttl_ms = self.ttl.as_millis() as u64;
        let expired: Vec<String> = sessions
            .iter()
            .filter(|(_, s)| now_ms.saturating_sub(s.last_accessed_millis()) > ttl_ms)
            .take(10)
            .map(|(k, _)| k.clone())
            .collect();
        for k in expired {
            sessions.remove(&k);
        }

        if let Some(session) = sessions.get_mut(session_id) {
            session.touch();
            // Clone the base and replay patches for isolation.
            let base = model.patched.read().await;
            let mut cloned = PatchedVindex::new(base.base().clone());
            for patch in &session.patched.patches {
                cloned.apply_patch(patch.clone());
            }
            return cloned;
        }

        // New session — start from the global patched state.
        let base = model.patched.read().await;
        let patched = PatchedVindex::new(base.base().clone());
        sessions.insert(
            session_id.to_string(),
            SessionState {
                patched: PatchedVindex::new(base.base().clone()),
                last_accessed: AtomicU64::new(now_millis()),
            },
        );
        patched
    }

    /// Apply a patch to a session (not global).
    pub async fn apply_patch(
        &self,
        session_id: &str,
        model: &Arc<LoadedModel>,
        patch: larql_vindex::VindexPatch,
    ) -> (usize, usize) {
        // Pre-acquire base outside the write lock to avoid blocking_read inside async.
        let base_for_new_session = {
            let existing = self.sessions.read().await;
            if existing.contains_key(session_id) {
                None
            } else {
                drop(existing);
                let base = model.patched.read().await;
                Some(base.base().clone())
            }
        };

        let mut sessions = self.sessions.write().await;
        let session = sessions
            .entry(session_id.to_string())
            .or_insert_with(|| {
                let base = base_for_new_session
                    .clone()
                    .unwrap_or_else(|| model.patched.blocking_read().base().clone());
                SessionState::new(base, Instant::now())
            });

        session.touch();
        let op_count = patch.operations.len();
        session.patched.apply_patch(patch);
        (op_count, session.patched.num_patches())
    }

    /// List patches for a session.
    pub async fn list_patches(&self, session_id: &str) -> Vec<serde_json::Value> {
        let sessions = self.sessions.read().await;
        match sessions.get(session_id) {
            Some(session) => session
                .patched
                .patches
                .iter()
                .map(|p| {
                    serde_json::json!({
                        "name": p.description.as_deref().unwrap_or("unnamed"),
                        "operations": p.operations.len(),
                        "base_model": p.base_model,
                    })
                })
                .collect(),
            None => vec![],
        }
    }

    /// Remove a patch from a session.
    pub async fn remove_patch(
        &self,
        session_id: &str,
        name: &str,
    ) -> Result<usize, String> {
        let mut sessions = self.sessions.write().await;
        let session = sessions
            .get_mut(session_id)
            .ok_or_else(|| format!("session '{}' not found", session_id))?;

        let idx = session
            .patched
            .patches
            .iter()
            .position(|p| p.description.as_deref().unwrap_or("unnamed") == name)
            .ok_or_else(|| format!("patch '{}' not found in session", name))?;

        session.patched.remove_patch(idx);
        Ok(session.patched.num_patches())
    }

    /// Blocking read access to sessions map — safe for concurrent INFER calls.
    pub fn sessions_blocking_read(&self) -> tokio::sync::RwLockReadGuard<'_, HashMap<String, SessionState>> {
        self.sessions.blocking_read()
    }

    /// Blocking write access to sessions map (for use in spawn_blocking / patch ops).
    pub fn sessions_blocking_write(&self) -> tokio::sync::RwLockWriteGuard<'_, HashMap<String, SessionState>> {
        self.sessions.blocking_write()
    }

    /// Number of active sessions.
    #[allow(dead_code)]
    pub async fn session_count(&self) -> usize {
        self.sessions.read().await.len()
    }
}
