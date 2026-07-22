//! [`KvHandle`] / [`ResidualHandle`] opaque handle types and their
//! backend-side inner traits. Split from `kv_dispatch/mod.rs` — see the
//! module-level doc there.

/// Opaque handle to a K/V cache allocation. Layout is backend-specific;
/// engines pass these around without observing structure beyond the
/// queries the trait exposes.
///
/// Backends ship their own inner type (`CpuKvHandle`, `MetalKvHandle`,
/// `VulkanKvHandle`) implementing [`KvHandleInner`]. Engines hold
/// `KvHandle` opaquely and call backend methods to manipulate it.
pub struct KvHandle {
    inner: Box<dyn KvHandleInner>,
}

impl KvHandle {
    /// Construct from a backend-specific inner. Backend implementations
    /// call this; engines never do.
    pub fn new<I: KvHandleInner + 'static>(inner: I) -> Self {
        Self {
            inner: Box::new(inner),
        }
    }

    /// Number of K/V rows currently cached.
    pub fn cached_len(&self) -> usize {
        self.inner.cached_len()
    }

    /// Hidden dim per K/V row (kv_dim, not full hidden — already
    /// accounts for GQA head count).
    pub fn kv_dim(&self) -> usize {
        self.inner.kv_dim()
    }

    /// Which backend allocated this handle. Used for sanity checks
    /// when handles cross backend boundaries (which normally
    /// shouldn't happen — read out to host first via
    /// [`super::KvDispatch::read_kv_to_host`]).
    pub fn backend_name(&self) -> &'static str {
        self.inner.backend_name()
    }

    /// Downcast access for backend implementations. Engines never call
    /// this; only the backend that allocated the handle should.
    pub fn as_inner(&self) -> &dyn KvHandleInner {
        &*self.inner
    }

    /// Mutable downcast for backend impls.
    pub fn as_inner_mut(&mut self) -> &mut dyn KvHandleInner {
        &mut *self.inner
    }
}

/// Backend-side trait for K/V handle inner types. Backends implement
/// this on whatever GPU-side or host-side allocation they manage
/// (`MTLBuffer`, `VkBuffer`, `Vec<f32>`, or a wrapper over an engine's
/// `KvCache` from `larql-kv`).
pub trait KvHandleInner: Send + Sync + std::any::Any {
    fn cached_len(&self) -> usize;
    fn kv_dim(&self) -> usize;
    fn backend_name(&self) -> &'static str;
    fn as_any(&self) -> &dyn std::any::Any;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

/// Opaque handle to a residual upload (used by `apollo` for boundary
/// residuals). Same pattern as [`KvHandle`].
pub struct ResidualHandle {
    pub(super) inner: Box<dyn ResidualHandleInner>,
}

impl ResidualHandle {
    pub fn new<I: ResidualHandleInner + 'static>(inner: I) -> Self {
        Self {
            inner: Box::new(inner),
        }
    }

    pub fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    pub fn backend_name(&self) -> &'static str {
        self.inner.backend_name()
    }

    pub fn as_inner(&self) -> &dyn ResidualHandleInner {
        &*self.inner
    }
}

pub trait ResidualHandleInner: Send + Sync + std::any::Any {
    fn shape(&self) -> (usize, usize);
    fn backend_name(&self) -> &'static str;
    fn as_any(&self) -> &dyn std::any::Any;
}
