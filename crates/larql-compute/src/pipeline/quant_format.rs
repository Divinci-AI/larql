/// Bytes per Q4_KF pre-baked super-block. Q4_KF keeps the 256-element
/// Q4_K block shape but expands packed scale/min metadata for faster decode.
pub const Q4_KF_BLOCK_BYTES: usize = 160;

/// Quantization format for a weight tensor.
/// Names match GGUF conventions (Q4_K, Q6_K, etc.).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum QuantFormat {
    Q4_0,  // 18 bytes per 32 values (one f16 scale)
    Q4_K,  // 144 bytes per 256 values (GGUF-canonical, Ollama-compatible)
    Q4_KF, // 160 bytes per 256 values (pre-baked half scales — fast decode)
    Q6_K,  // 210 bytes per 256 values (6-bit with sub-block scales)
    Q8_0,  // int8 values + separate f32 scales
    BF16,  // raw bfloat16 (2 bytes per value, no quantization scales)
    F16,   // raw float16  (2 bytes per value)
    F32,   // raw float32  (4 bytes per value)
    /// BitNet 1.58-bit ternary (GGML I2_S, type 36): 4 trits/byte packed
    /// row-major (`cols/4` bytes per row) plus a separate per-channel f32
    /// scale array. Unlike the block-quant formats, the weight is NOT a
    /// flat `&[u8]` block stream — it is carried by
    /// [`crate::cpu::ops::ternary_matvec::BitLinearWeight`] (bytes + scales)
    /// and served by the dedicated `ternary_matvec` dispatch, not the
    /// block-quant `quant_matvec` path (which has no per-channel-scale input).
    I2S,
}

impl QuantFormat {
    /// Packed block geometry as `(elements_per_block, bytes_per_block)`.
    ///
    /// This is the compute-side mirror of the GGML layout constants used by
    /// the quantizers. Callers that need byte offsets should ask the format
    /// instead of spelling `256 * 144` or `32 * 18` locally.
    pub fn packed_block_layout(self) -> Option<(usize, usize)> {
        use larql_models::quant::ggml;

        match self {
            Self::Q4_0 => Some((ggml::Q4_0_BLOCK_ELEMS, ggml::Q4_0_BLOCK_BYTES)),
            Self::Q4_K => Some((ggml::Q4_K_BLOCK_ELEMS, ggml::Q4_K_BLOCK_BYTES)),
            Self::Q4_KF => Some((ggml::Q4_K_BLOCK_ELEMS, Q4_KF_BLOCK_BYTES)),
            Self::Q6_K => Some((ggml::Q6_K_BLOCK_ELEMS, ggml::Q6_K_BLOCK_BYTES)),
            _ => None,
        }
    }

    /// Byte length for a packed row-major matrix with `rows * cols` values.
    ///
    /// Current interleaved FFN fallback stores each matrix contiguously, so
    /// this intentionally preserves the historical flat packing calculation.
    /// Manifest-aware paths should prefer recorded offsets and lengths.
    pub fn packed_matrix_bytes(self, rows: usize, cols: usize) -> Option<usize> {
        let elems = rows.checked_mul(cols)?;
        let (block_elems, block_bytes) = self.packed_block_layout()?;
        Some(elems.div_ceil(block_elems) * block_bytes)
    }

    /// Whether this format uses the GGUF k-quant 256-element super-block
    /// layout that flows through the dedicated Q4_K / Q4_KF / Q6_K matvec
    /// dispatchers (vs the legacy block-32 Q4_0 / Q8_0 path). Used to gate
    /// the "skip Q8 quantize" fast path in `residual_norm` and FFN routing.
    ///
    /// Adding a future k-quant format (e.g. Q5_K) extends this one method,
    /// not the ~10 OR-chains it currently replaces. Roadmap #7
    /// (`FormatRoute` enum) is the fuller version of this idea; this helper
    /// is the contained step that addresses the user-visible code-duplication
    /// cost without rippling through 49 files.
    pub fn is_kquant_family(self) -> bool {
        matches!(self, Self::Q4_K | Self::Q4_KF | Self::Q6_K)
    }

    /// Whether this format uses the llama.cpp-exact "Q4_KF" pre-baked
    /// half-scale fast path (`q4kf_proj` shader). Distinct from the
    /// canonical `Q4_K` GGUF layout used by Ollama extracts.
    pub fn is_q4kf(self) -> bool {
        matches!(self, Self::Q4_KF)
    }

    /// Whether this format uses the legacy block-32 Q8 dispatch path
    /// (`q4_matvec` / `q8_matvec` against pre-quantised Q8 input). The
    /// inverse of [`Self::is_kquant_family`] for the dense matvec dispatch
    /// (the float-input `BF16` / `F16` / `F32` branches don't run on
    /// these dispatchers, so `is_legacy_q8` covers exactly the rest).
    pub fn is_legacy_q8(self) -> bool {
        matches!(self, Self::Q4_0 | Self::Q8_0)
    }

    /// Parse a GGUF-convention registry tag (`"Q4_K"`, `"Q6_K"`, …) into a
    /// `QuantFormat`. The canonical inverse of the names the extractor and
    /// weight manifests record; `None` for any tag with no compute mapping.
    ///
    /// This is the contained version of Roadmap #7's `from_registry_tag`:
    /// it lets the string-keyed matvec dispatchers (`q4k_q8k_matvec_parallel`,
    /// `kquant_forward::cached`) ask the format for its packed layout instead
    /// of re-spelling `(cols/256)*144` locally, without changing their `&str`
    /// call-site signatures.
    pub fn from_registry_tag(tag: &str) -> Option<Self> {
        Some(match tag {
            "Q4_0" => Self::Q4_0,
            "Q4_K" => Self::Q4_K,
            "Q4_KF" => Self::Q4_KF,
            "Q6_K" => Self::Q6_K,
            "Q8_0" => Self::Q8_0,
            "BF16" => Self::BF16,
            "F16" => Self::F16,
            "F32" => Self::F32,
            // BitNet ternary (GGML type 36). The vindex bitnet sidecar tags
            // its I2_S weight stream with this so the registry recognises it.
            "I2_S" => Self::I2S,
            _ => return None,
        })
    }

    /// Inverse of [`Self::from_registry_tag`] — the canonical registry tag
    /// string for this format. `from_registry_tag(f.registry_tag()) == Some(f)`
    /// for every variant. Used by writers that record the per-tensor format
    /// tag into the weight manifest / index.
    pub fn registry_tag(self) -> &'static str {
        match self {
            Self::Q4_0 => "Q4_0",
            Self::Q4_K => "Q4_K",
            Self::Q4_KF => "Q4_KF",
            Self::Q6_K => "Q6_K",
            Self::Q8_0 => "Q8_0",
            Self::BF16 => "BF16",
            Self::F16 => "F16",
            Self::F32 => "F32",
            Self::I2S => "I2_S",
        }
    }

    /// Whether this format is BitNet ternary (I2_S). Served by the dedicated
    /// `ternary_matvec` path with a [`crate::cpu::ops::ternary_matvec::BitLinearWeight`],
    /// never the block-quant `quant_matvec` dispatch.
    pub fn is_ternary(self) -> bool {
        matches!(self, Self::I2S)
    }

    /// Where this format keeps its dequantisation scales.
    ///
    /// Exhaustive by construction: a new format must answer this, rather
    /// than inheriting a default that happens to be right for the formats
    /// that existed when it was added.
    pub fn scale_storage(self) -> ScaleStorage {
        match self {
            // Block-packed: the scale rides inside each block.
            Self::Q4_0 | Self::Q4_K | Self::Q4_KF | Self::Q6_K => ScaleStorage::Inline,
            // "int8 values + separate f32 scales", per this enum's own doc.
            Self::Q8_0 => ScaleStorage::External(ExternalScaleKind::PerBlockF32),
            // Ternary carries a separate per-channel f32 array.
            Self::I2S => ScaleStorage::External(ExternalScaleKind::PerChannelF32),
            Self::BF16 | Self::F16 | Self::F32 => ScaleStorage::None,
        }
    }
}

/// How a format stores its dequantisation scales.
///
/// Distinct from [`QuantFormat::packed_block_layout`] on purpose. That
/// describes *representation geometry*; this describes the *auxiliary
/// storage contract*. They correlate perfectly today — every block-packed
/// format carries its scales inline — but they are not the same property,
/// and a future packed format with external scales would make
/// `packed_block_layout().is_some()` the wrong discriminator. Keeping them
/// apart is the point; conflating them is how the caller ended up
/// reconstructing the format's own rules.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScaleStorage {
    /// Scales live inside the packed blocks. No external array exists —
    /// not "an empty one".
    Inline,
    /// Scales live in a separate f32 array the caller must supply.
    External(ExternalScaleKind),
    /// Unquantised — there are no scales at all.
    None,
}

/// Shape of an external scale array, for formats that have one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalScaleKind {
    /// One f32 per 32-element block (Q8_0).
    PerBlockF32,
    /// One f32 per output channel (I2_S / BitNet ternary).
    PerChannelF32,
}

/// Auxiliary material a caller supplies alongside the packed bytes.
///
/// Deliberately cannot express *how* scales are stored — only whether the
/// caller is handing over an external array. The format owns the rest, so
/// there is no second description of the same truth to keep in sync.
#[derive(Clone, Copy, Debug, Default)]
pub enum QuantAux<'a> {
    /// No external scale array — either the format packs them inline or it
    /// is unquantised. Which of those it is, is the format's business.
    #[default]
    None,
    /// An external scale array, required by `Q8_0` and `I2S`.
    ExternalScales(&'a [f32]),
}

/// A quantized weight matrix — raw bytes with format tag.
///
/// Construct via [`QuantWeight::new`]; the fields are private so a caller
/// cannot assert an auxiliary-storage arrangement the format disagrees
/// with. Two states this previously permitted, both of which occurred in
/// this repository:
///
/// - `Q4_K` with an external scale buffer. Q4_K packs scales inline, so
///   the buffer was fabricated — bound as a zero-length resource at 24
///   sites, which destroys the distinction between "no scales are
///   required" and "scales are required and there are none".
/// - `Q4_0` with an external scale array (a test fixture did exactly
///   this, and passed). Q4_0's 18-byte block *is* an f16 scale plus 16
///   bytes of nibbles.
#[derive(Clone, Copy)]
pub struct QuantWeight<'a> {
    pub data: &'a [u8],
    pub format: QuantFormat,
    /// Private on purpose: this is the field a caller must not be able to
    /// set independently of `format`. One private field is enough to
    /// force construction through [`QuantWeight::new`], while `data` and
    /// `format` stay readable without ceremony.
    aux: QuantAux<'a>,
}

impl<'a> QuantWeight<'a> {
    /// Build a weight, checking the auxiliary material against what the
    /// format actually requires.
    ///
    /// # Panics
    /// If `aux` disagrees with `format.scale_storage()`. This is a
    /// programming error in the same class as an unknown format tag, not
    /// a runtime condition — the alternative is a kernel reading a
    /// fabricated buffer, which is how this became worth enforcing.
    pub fn new(format: QuantFormat, data: &'a [u8], aux: QuantAux<'a>) -> Self {
        match (format.scale_storage(), aux) {
            (ScaleStorage::External(_), crate::QuantAux::ExternalScales(_))
            | (ScaleStorage::Inline, crate::QuantAux::None)
            | (ScaleStorage::None, crate::QuantAux::None) => {}
            (ScaleStorage::External(kind), crate::QuantAux::None) => panic!(
                "{format:?} stores scales externally ({kind:?}) but none were supplied"
            ),
            (ScaleStorage::Inline, crate::QuantAux::ExternalScales(_)) => panic!(
                "{format:?} packs its scales inline; an external scale array is not a                  thing it has"
            ),
            (ScaleStorage::None, crate::QuantAux::ExternalScales(_)) => {
                panic!("{format:?} is unquantised and has no scales")
            }
        }
        Self { data, format, aux }
    }

    /// The external scale array, when the format has one.
    ///
    /// Returns `None` for inline and unquantised formats — and callers
    /// must bind no scale resource in that case rather than fabricating an
    /// empty one.
    pub fn external_scales(&self) -> Option<&'a [f32]> {
        match self.aux {
            crate::QuantAux::ExternalScales(s) => Some(s),
            crate::QuantAux::None => None,
        }
    }
}

impl Default for QuantWeight<'_> {
    fn default() -> Self {
        // Q4_0 is inline, so the empty default needs no aux.
        Self {
            data: &[],
            format: QuantFormat::Q4_0,
            aux: crate::QuantAux::None,
        }
    }
}

#[cfg(test)]
mod scale_storage_tests {
    use super::*;

    /// Every format answers, and the answer matches the enum's own
    /// documentation of how it stores scales.
    #[test]
    fn scale_storage_is_exhaustive_and_matches_the_documented_layout() {
        use ExternalScaleKind::*;
        use ScaleStorage::*;
        let table = [
            (QuantFormat::Q4_0, Inline),
            (QuantFormat::Q4_K, Inline),
            (QuantFormat::Q4_KF, Inline),
            (QuantFormat::Q6_K, Inline),
            (QuantFormat::Q8_0, External(PerBlockF32)),
            (QuantFormat::I2S, External(PerChannelF32)),
            (QuantFormat::BF16, None),
            (QuantFormat::F16, None),
            (QuantFormat::F32, None),
        ];
        for (f, expected) in table {
            assert_eq!(f.scale_storage(), expected, "{f:?}");
        }
    }

    /// Inline formats are exactly the block-packed ones *today*. Asserted
    /// as an observation, not used as the discriminator — representation
    /// geometry and auxiliary storage are different properties that
    /// merely correlate, and a future packed format with external scales
    /// must break this test rather than silently misbehave.
    #[test]
    fn inline_and_block_packed_coincide_today() {
        for f in [QuantFormat::Q4_0, QuantFormat::Q4_K, QuantFormat::Q4_KF, QuantFormat::Q6_K] {
            assert!(f.packed_block_layout().is_some());
            assert_eq!(f.scale_storage(), ScaleStorage::Inline);
        }
    }

    // ── the Phase A specification, as a matrix ──────────────────────
    //
    //                     no aux      external scales
    //   Q4_0 / Q4_K         ok             panic
    //   Q4_KF / Q6_K        ok             panic
    //   Q8_0 / I2S        panic              ok
    //   F16 / F32 / BF16    ok             panic

    #[test]
    fn inline_formats_accept_no_aux() {
        for f in [QuantFormat::Q4_0, QuantFormat::Q4_K, QuantFormat::Q4_KF, QuantFormat::Q6_K] {
            let w = QuantWeight::new(f, &[0u8; 4], QuantAux::None);
            assert!(w.external_scales().is_none(), "{f:?}");
        }
    }

    #[test]
    fn unquantised_formats_accept_no_aux() {
        for f in [QuantFormat::BF16, QuantFormat::F16, QuantFormat::F32] {
            let w = QuantWeight::new(f, &[0u8; 4], QuantAux::None);
            assert!(w.external_scales().is_none(), "{f:?}");
        }
    }

    #[test]
    fn external_formats_accept_and_expose_their_scales() {
        let s = [1.0f32, 2.0];
        for f in [QuantFormat::Q8_0, QuantFormat::I2S] {
            let w = QuantWeight::new(f, &[0u8; 4], QuantAux::ExternalScales(&s));
            assert_eq!(w.external_scales(), Some(&s[..]), "{f:?}");
        }
    }

    /// The state that produced the dead O-projection fixture and the 24
    /// fabricated buffers: an inline format carrying external scales.
    #[test]
    #[should_panic(expected = "packs its scales inline")]
    fn q4k_cannot_carry_external_scales() {
        let s = [1.0f32];
        let _ = QuantWeight::new(QuantFormat::Q4_K, &[0u8; 4], QuantAux::ExternalScales(&s));
    }

    /// Q4_0's 18-byte block *is* an f16 scale plus 16 bytes of nibbles.
    /// A test fixture in this repository supplied external scales for it
    /// and passed; that is now unrepresentable.
    #[test]
    #[should_panic(expected = "packs its scales inline")]
    fn q4_0_cannot_carry_external_scales() {
        let s = [1.0f32];
        let _ = QuantWeight::new(QuantFormat::Q4_0, &[0u8; 4], QuantAux::ExternalScales(&s));
    }

    #[test]
    #[should_panic(expected = "unquantised")]
    fn unquantised_formats_cannot_carry_scales() {
        let s = [1.0f32];
        let _ = QuantWeight::new(QuantFormat::F32, &[0u8; 4], QuantAux::ExternalScales(&s));
    }

    /// The other half: a format that genuinely needs scales cannot exist
    /// without them. Previously `scales: None` on a Q8_0 weight was a
    /// silently-constructible state.
    #[test]
    #[should_panic(expected = "stores scales externally")]
    fn q8_0_cannot_exist_without_scales() {
        let _ = QuantWeight::new(QuantFormat::Q8_0, &[0u8; 4], QuantAux::None);
    }

    #[test]
    #[should_panic(expected = "stores scales externally")]
    fn i2s_cannot_exist_without_scales() {
        let _ = QuantWeight::new(QuantFormat::I2S, &[0u8; 4], QuantAux::None);
    }

    /// The default is a valid state, not merely a zeroed one.
    #[test]
    fn default_weight_is_internally_consistent() {
        let w = QuantWeight::default();
        assert_eq!(w.format.scale_storage(), ScaleStorage::Inline);
        assert!(w.external_scales().is_none());
    }
}
