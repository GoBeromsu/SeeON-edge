//! Per-call accelerator receipts derived from native execution counters.
//! A receipt proves one synchronous call ran on the configured device with both
//! copies observed. It does not prove numeric equivalence or authenticate the
//! engine; the caller supplies the digest of the engine it admitted.

use seeon_deepstream_native::GpuMetrics;

pub const PROVIDER: &str = "tensorrt";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Precision {
    Fp32,
}

impl Precision {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32",
        }
    }
}

/// SHA-256 of the serialized engine bytes the caller admitted.
#[derive(Clone, Copy, Eq, PartialEq)]
pub struct EngineDigest([u8; 32]);

impl EngineDigest {
    pub const fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl std::fmt::Display for EngineDigest {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0
            .iter()
            .try_for_each(|byte| write!(formatter, "{byte:02x}"))
    }
}

impl std::fmt::Debug for EngineDigest {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "EngineDigest({self})")
    }
}

/// One refusal per violated field; no counter values or device names.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvidenceError {
    DeviceMismatch,
    CounterRegressed,
    AttemptedNotOne,
    SucceededNotOne,
    FailedNonzero,
    NoHostToDevice,
    NoDeviceToHost,
}

impl std::fmt::Display for EvidenceError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::DeviceMismatch => "accelerator evidence names another device",
            Self::CounterRegressed => "accelerator counters moved backwards",
            Self::AttemptedNotOne => "accelerator evidence must cover exactly one attempt",
            Self::SucceededNotOne => "accelerator evidence must cover exactly one success",
            Self::FailedNonzero => "accelerator evidence includes a failed execution",
            Self::NoHostToDevice => "accelerator evidence shows no host-to-device copy",
            Self::NoDeviceToHost => "accelerator evidence shows no device-to-host copy",
        })
    }
}
impl std::error::Error for EvidenceError {}

/// Constructed only from a validated counter delta around one `run`.
#[must_use]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AcceleratorEvidence {
    precision: Precision,
    device_ordinal: i32,
    engine_sha256: EngineDigest,
    call_seq: u64,
    attempted: u64,
    succeeded: u64,
    failed: u64,
    h2d_bytes: u64,
    d2h_bytes: u64,
    elapsed_ns: u64,
}

impl AcceleratorEvidence {
    /// `before` and `after` bracket exactly one call on an exclusively owned model.
    /// `call_seq` is the model's cumulative attempt count after that call.
    pub fn from_delta(
        before: &GpuMetrics,
        after: &GpuMetrics,
        device_ordinal: i32,
        engine_sha256: EngineDigest,
        precision: Precision,
    ) -> Result<Self, EvidenceError> {
        if before.device != device_ordinal || after.device != device_ordinal {
            return Err(EvidenceError::DeviceMismatch);
        }
        let delta = |before: u64, after: u64| {
            after
                .checked_sub(before)
                .ok_or(EvidenceError::CounterRegressed)
        };
        let attempted = delta(before.attempted, after.attempted)?;
        let succeeded = delta(before.succeeded, after.succeeded)?;
        let failed = delta(before.failed, after.failed)?;
        let h2d_bytes = delta(before.host_to_device_bytes, after.host_to_device_bytes)?;
        let d2h_bytes = delta(before.device_to_host_bytes, after.device_to_host_bytes)?;
        let elapsed_ns = delta(before.elapsed_ns, after.elapsed_ns)?;
        if attempted != 1 {
            return Err(EvidenceError::AttemptedNotOne);
        }
        if succeeded != 1 {
            return Err(EvidenceError::SucceededNotOne);
        }
        if failed != 0 {
            return Err(EvidenceError::FailedNonzero);
        }
        if h2d_bytes == 0 {
            return Err(EvidenceError::NoHostToDevice);
        }
        if d2h_bytes == 0 {
            return Err(EvidenceError::NoDeviceToHost);
        }
        Ok(Self {
            precision,
            device_ordinal,
            engine_sha256,
            call_seq: after.attempted,
            attempted,
            succeeded,
            failed,
            h2d_bytes,
            d2h_bytes,
            elapsed_ns,
        })
    }

    pub const fn provider(&self) -> &'static str {
        PROVIDER
    }
    pub const fn precision(&self) -> Precision {
        self.precision
    }
    pub const fn device_ordinal(&self) -> i32 {
        self.device_ordinal
    }
    pub const fn engine_sha256(&self) -> EngineDigest {
        self.engine_sha256
    }
    pub const fn call_seq(&self) -> u64 {
        self.call_seq
    }
    pub const fn attempted(&self) -> u64 {
        self.attempted
    }
    pub const fn succeeded(&self) -> u64 {
        self.succeeded
    }
    pub const fn failed(&self) -> u64 {
        self.failed
    }
    pub const fn h2d_bytes(&self) -> u64 {
        self.h2d_bytes
    }
    pub const fn d2h_bytes(&self) -> u64 {
        self.d2h_bytes
    }
    pub const fn elapsed_ns(&self) -> u64 {
        self.elapsed_ns
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIGEST: EngineDigest = EngineDigest::new([0xab; 32]);

    fn before() -> GpuMetrics {
        GpuMetrics {
            attempted: 4,
            succeeded: 4,
            failed: 0,
            host_to_device_bytes: 400,
            device_to_host_bytes: 40,
            elapsed_ns: 1_000,
            device: 0,
        }
    }

    fn after() -> GpuMetrics {
        GpuMetrics {
            attempted: 5,
            succeeded: 5,
            failed: 0,
            host_to_device_bytes: 500,
            device_to_host_bytes: 50,
            elapsed_ns: 1_250,
            device: 0,
        }
    }

    #[test]
    fn complete_receipt_carries_the_call_delta() {
        let evidence =
            AcceleratorEvidence::from_delta(&before(), &after(), 0, DIGEST, Precision::Fp32)
                .expect("a complete receipt is admitted");
        assert_eq!(evidence.provider(), "tensorrt");
        assert_eq!(evidence.precision().as_str(), "fp32");
        assert_eq!(evidence.device_ordinal(), 0);
        assert_eq!(evidence.engine_sha256(), DIGEST);
        assert_eq!(evidence.call_seq(), 5);
        assert_eq!(
            (
                evidence.attempted(),
                evidence.succeeded(),
                evidence.failed()
            ),
            (1, 1, 0)
        );
        assert_eq!((evidence.h2d_bytes(), evidence.d2h_bytes()), (100, 10));
        assert_eq!(evidence.elapsed_ns(), 250);
    }

    #[test]
    fn each_incomplete_proof_names_its_own_violation() {
        type Mutation = fn(&mut GpuMetrics, &mut GpuMetrics);
        let cases: [(Mutation, EvidenceError); 9] = [
            (|_, after| after.device = 1, EvidenceError::DeviceMismatch),
            (|before, _| before.device = 1, EvidenceError::DeviceMismatch),
            (
                |before, _| before.elapsed_ns = 2_000,
                EvidenceError::CounterRegressed,
            ),
            (
                |_, after| after.attempted = 6,
                EvidenceError::AttemptedNotOne,
            ),
            (
                |_, after| after.succeeded = 4,
                EvidenceError::SucceededNotOne,
            ),
            (|_, after| after.failed = 1, EvidenceError::FailedNonzero),
            (
                |_, after| after.host_to_device_bytes = 400,
                EvidenceError::NoHostToDevice,
            ),
            (
                |_, after| after.device_to_host_bytes = 40,
                EvidenceError::NoDeviceToHost,
            ),
            (
                |before, after| {
                    before.attempted = 5;
                    after.attempted = 5;
                },
                EvidenceError::AttemptedNotOne,
            ),
        ];
        for (index, (mutate, expected)) in cases.into_iter().enumerate() {
            let (mut before, mut after) = (before(), after());
            mutate(&mut before, &mut after);
            assert_eq!(
                AcceleratorEvidence::from_delta(&before, &after, 0, DIGEST, Precision::Fp32),
                Err(expected),
                "case {index}"
            );
        }
    }

    #[test]
    fn engine_digest_renders_lowercase_hex() {
        let mut bytes = [0_u8; 32];
        bytes[0] = 0x0f;
        bytes[31] = 0xa0;
        let text = EngineDigest::new(bytes).to_string();
        assert_eq!(text.len(), 64);
        assert!(text.starts_with("0f00"));
        assert!(text.ends_with("00a0"));
        assert_eq!(
            format!("{:?}", EngineDigest::new(bytes)),
            format!("EngineDigest({text})")
        );
    }
}
