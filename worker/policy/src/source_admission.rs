//! Single-source admission for one native capture handle.
//!
//! Binding and high-water checks follow `worker/runtime/flow/metadata_slot.py`.
//! Logical generation and stream epoch follow `SourceTable.rebuild` in
//! `worker/adapters/deepstream/sources.py`: both advance by one. Those logical
//! counters are not the native handle's immutable binding generation and epoch.
//! The transport id passed to [`SourceAdmission::new`] is that handle identity
//! and is never rewritten. This owner stores no frames and has no mailbox,
//! thread, clock, or UUID source.
//!
//! A later native adapter may copy one packet sequence into both
//! `canonical_sequence` and `native_publish_sequence`. Admission still checks
//! those ordinals independently, so a malformed or replayed pair cannot pass on
//! the strength of the other field. PTS regression never rotates a source.
//!
//! [`SourceAdmission::rotate`] installs a discard-through floor `F` from the
//! caller's already validated native `status.frames` cutover. That floor is not
//! an RTSP capture boundary: rotation does not open or close a capture, and it
//! does not reset native transport identity or the native publication counter.
//! Packets still queued with a native publication ordinal `<= F` are rejected.
//! This owner does not re-derive `F` from accepted watermarks.
//!
//! [`SourceAdmission::accepts_policy_work`] is only the incarnation predicate
//! for uncommitted policy results. Older accepted work from the same incarnation
//! stays eligible; the latest frame is not required. The predicate does not
//! govern durable event or recording obligations that were already admitted.

use std::fmt;
use std::sync::Arc;

/// Caller-supplied canonical binding. Boot, camera, and transform are exact,
/// untruncated strings; the canonical contract supplies no byte-length limit.
/// Child identity is the lowercase hyphenated spelling of a UUID, matching
/// Python's `str(metadata.child_instance_id)`. No UUID is generated here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceBinding {
    pub worker_boot_id: String,
    pub child_instance_id: String,
    pub camera_id: String,
    pub source_generation: u64,
    pub stream_epoch: u64,
    pub transform_id: String,
}

/// Borrowed scalar header, not a frame or a retained native resource. Strings
/// borrow packet/caller buffers; constructing a header needs no owned binding.
/// Generation and epoch are canonical logical counters, not immutable SDK
/// counters. The adapter supplies the already-normalized child UUID spelling.
#[derive(Debug, Clone, Copy)]
pub struct FrameMetadata<'a> {
    pub transport_id: &'a str,
    pub worker_boot_id: &'a str,
    pub child_instance_id: &'a str,
    pub camera_id: &'a str,
    pub source_generation: u64,
    pub stream_epoch: u64,
    pub transform_id: &'a str,
    pub source_pts: Option<i64>,
    pub canonical_sequence: u64,
    pub native_publish_sequence: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HighWater {
    pub source_pts: i64,
    pub canonical_sequence: u64,
    pub native_publish_sequence: u64,
}

/// Issued only by successful admission. Cloning shares immutable incarnation
/// identity without copying strings or allocating. Retaining a stamp keeps that
/// binding alive; callers own the number and lifetime of outstanding stamps.
#[derive(Debug, Clone)]
pub struct AcceptedWork {
    incarnation: Arc<SourceBinding>,
    high_water: HighWater,
}
impl AcceptedWork {
    pub fn binding(&self) -> &SourceBinding {
        &self.incarnation
    }

    pub fn high_water(&self) -> HighWater {
        self.high_water
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmissionError {
    MalformedCamera,
    InvalidChildInstanceId,
    UnknownSource,
    BootMismatch,
    ChildMismatch,
    GenerationMismatch,
    EpochMismatch,
    TransformMismatch,
    TransportMismatch,
    PtsMissing,
    DiscardedPublication,
    NonIncreasingPts,
    NonIncreasingCanonicalSequence,
    NonIncreasingNativePublicationSequence,
    RegressingFence,
    GenerationExhausted,
    EpochExhausted,
}
impl fmt::Display for AdmissionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::MalformedCamera => "camera identity is empty",
            Self::InvalidChildInstanceId => "binding child identity is not a normalized UUID",
            Self::UnknownSource => "camera is not registered with this owner",
            Self::BootMismatch => "worker boot identity does not match",
            Self::ChildMismatch => "normalized child identity does not match",
            Self::GenerationMismatch => "source generation does not match",
            Self::EpochMismatch => "stream epoch does not match",
            Self::TransformMismatch => "transform identity does not match",
            Self::TransportMismatch => "native transport identity does not match",
            Self::PtsMissing => "source PTS is missing",
            Self::DiscardedPublication => "publication is at or below the rotation fence",
            Self::NonIncreasingPts => "source PTS did not strictly increase",
            Self::NonIncreasingCanonicalSequence => "canonical sequence did not strictly increase",
            Self::NonIncreasingNativePublicationSequence => {
                "native publication sequence did not strictly increase"
            }
            Self::RegressingFence => "fence precedes known native publication progress",
            Self::GenerationExhausted => "logical source generation is exhausted",
            Self::EpochExhausted => "logical stream epoch is exhausted",
        })
    }
}
impl std::error::Error for AdmissionError {}

fn validate_binding(binding: &SourceBinding) -> Result<(), AdmissionError> {
    if binding.camera_id.is_empty() {
        return Err(AdmissionError::MalformedCamera);
    }
    let child = binding.child_instance_id.as_bytes();
    if child.len() != 36
        || child.iter().enumerate().any(|(index, &byte)| {
            if matches!(index, 8 | 13 | 18 | 23) {
                byte != b'-'
            } else {
                !matches!(byte, b'0'..=b'9' | b'a'..=b'f')
            }
        })
    {
        return Err(AdmissionError::InvalidChildInstanceId);
    }
    Ok(())
}

/// One registered source on one native handle. Storage is one binding, one
/// transport identity, and fixed-size scalars; there is no history or mailbox.
/// Identity storage scales with caller-supplied string lengths, not poll count.
/// Only lifecycle operations allocate; admission merely clones an Arc stamp.
/// No domain state is owned or reset here, including on rejected input.
#[derive(Debug)]
pub struct SourceAdmission {
    transport_id: String,
    incarnation: Arc<SourceBinding>,
    high_water: Option<HighWater>,
    discard_through: Option<u64>,
    known_published_through: Option<u64>,
}
impl SourceAdmission {
    /// Registers without an initial publication floor or accepted-frame readiness.
    /// A different native handle needs a new owner, not re-registration.
    pub fn new(transport_id: String, binding: SourceBinding) -> Result<Self, AdmissionError> {
        validate_binding(&binding)?;
        Ok(Self {
            transport_id,
            incarnation: Arc::new(binding),
            high_water: None,
            discard_through: None,
            known_published_through: None,
        })
    }

    pub fn binding(&self) -> &SourceBinding {
        &self.incarnation
    }

    pub fn transport_id(&self) -> &str {
        &self.transport_id
    }

    pub fn high_water(&self) -> Option<HighWater> {
        self.high_water
    }

    pub fn is_ready(&self) -> bool {
        self.high_water.is_some()
    }

    pub fn discard_through(&self) -> Option<u64> {
        self.discard_through
    }

    /// Lower bound proved by accepted headers or caller-validated rotation fences,
    /// not the handle's current status.frames, a last-poll cursor, or readiness.
    /// Knowledge survives re-registration; it is not an implicit admission floor.
    pub fn known_published_through(&self) -> Option<u64> {
        self.known_published_through
    }

    /// Mirrors explicit canonical registration: clears all three accepted
    /// watermarks and any discard floor, even for an equal binding. Old policy
    /// work becomes ineligible. Native transport identity/progress is unchanged.
    /// Use rotate, not this method, for fenced same-handle outage recovery.
    pub fn reregister(&mut self, binding: SourceBinding) -> Result<(), AdmissionError> {
        validate_binding(&binding)?;
        self.incarnation = Arc::new(binding);
        self.high_water = None;
        self.discard_through = None;
        Ok(())
    }

    /// `publication_fence` MUST be an already validated status.frames value for
    /// this handle, supplied by the caller; this primitive cannot validate its
    /// provenance. Accepted progress is only a regression check, never a derived F.
    /// Equal fences are valid. Every error leaves the complete owner unchanged.
    pub fn rotate(&mut self, publication_fence: u64) -> Result<(), AdmissionError> {
        if self
            .known_published_through
            .is_some_and(|known| publication_fence < known)
        {
            return Err(AdmissionError::RegressingFence);
        }
        let generation = self
            .incarnation
            .source_generation
            .checked_add(1)
            .ok_or(AdmissionError::GenerationExhausted)?;
        let epoch = self
            .incarnation
            .stream_epoch
            .checked_add(1)
            .ok_or(AdmissionError::EpochExhausted)?;
        let mut binding = self.binding().clone();
        binding.source_generation = generation;
        binding.stream_epoch = epoch;
        self.incarnation = Arc::new(binding);
        self.high_water = None;
        self.discard_through = Some(publication_fence);
        self.known_published_through = Some(publication_fence);
        Ok(())
    }

    /// Canonical mismatch order precedes missing PTS, the explicit floor, and
    /// independent strict watermark checks. Rejections have no state effects.
    pub fn admit(&mut self, metadata: FrameMetadata<'_>) -> Result<AcceptedWork, AdmissionError> {
        let actual = &metadata;
        let expected = self.binding();
        let mismatch = if actual.camera_id.is_empty() {
            Some(AdmissionError::MalformedCamera)
        } else if actual.camera_id != expected.camera_id {
            Some(AdmissionError::UnknownSource)
        } else if actual.worker_boot_id != expected.worker_boot_id {
            Some(AdmissionError::BootMismatch)
        } else if actual.child_instance_id != expected.child_instance_id {
            Some(AdmissionError::ChildMismatch)
        } else if actual.source_generation != expected.source_generation {
            Some(AdmissionError::GenerationMismatch)
        } else if actual.stream_epoch != expected.stream_epoch {
            Some(AdmissionError::EpochMismatch)
        } else if actual.transform_id != expected.transform_id {
            Some(AdmissionError::TransformMismatch)
        } else if metadata.transport_id != self.transport_id {
            Some(AdmissionError::TransportMismatch)
        } else {
            None
        };
        if let Some(error) = mismatch {
            return Err(error);
        }
        let source_pts = metadata.source_pts.ok_or(AdmissionError::PtsMissing)?;
        if self
            .discard_through
            .is_some_and(|floor| metadata.native_publish_sequence <= floor)
        {
            return Err(AdmissionError::DiscardedPublication);
        }
        if let Some(previous) = self.high_water {
            if source_pts <= previous.source_pts {
                return Err(AdmissionError::NonIncreasingPts);
            }
            if metadata.canonical_sequence <= previous.canonical_sequence {
                return Err(AdmissionError::NonIncreasingCanonicalSequence);
            }
            if metadata.native_publish_sequence <= previous.native_publish_sequence {
                return Err(AdmissionError::NonIncreasingNativePublicationSequence);
            }
        }
        let high_water = HighWater {
            source_pts,
            canonical_sequence: metadata.canonical_sequence,
            native_publish_sequence: metadata.native_publish_sequence,
        };
        self.high_water = Some(high_water);
        self.known_published_through = Some(
            self.known_published_through
                .unwrap_or(0)
                .max(metadata.native_publish_sequence),
        );
        Ok(AcceptedWork {
            incarnation: Arc::clone(&self.incarnation),
            high_water,
        })
    }

    /// Eligibility for UNCOMMITTED policy work, not latest-frame equality.
    /// Already-admitted durable events/recordings MUST NOT use this predicate.
    /// Pointer identity is exact while stamps retain their Arc: no hash, global
    /// registry, counter wrap, or equal-binding ABA across owners/registrations.
    pub fn accepts_policy_work(&self, work: &AcceptedWork) -> bool {
        Arc::ptr_eq(&self.incarnation, &work.incarnation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TRANSPORT: &str = "caller-owned-native-handle";

    fn binding() -> SourceBinding {
        SourceBinding {
            worker_boot_id: "worker-boot".into(),
            child_instance_id: "abcdef01-2345-6789-abcd-ef0123456789".into(),
            camera_id: "camera-a".into(),
            source_generation: 7,
            stream_epoch: 11,
            transform_id: "transform-a".into(),
        }
    }

    fn frame(
        binding: &SourceBinding,
        source_pts: Option<i64>,
        canonical_sequence: u64,
        native_publish_sequence: u64,
    ) -> FrameMetadata<'_> {
        FrameMetadata {
            transport_id: TRANSPORT,
            worker_boot_id: &binding.worker_boot_id,
            child_instance_id: &binding.child_instance_id,
            camera_id: &binding.camera_id,
            source_generation: binding.source_generation,
            stream_epoch: binding.stream_epoch,
            transform_id: &binding.transform_id,
            source_pts,
            canonical_sequence,
            native_publish_sequence,
        }
    }

    #[derive(Debug, PartialEq, Eq)]
    struct Snapshot {
        binding: SourceBinding,
        transport_id: String,
        high_water: Option<HighWater>,
        discard_through: Option<u64>,
        known_published_through: Option<u64>,
        ready: bool,
    }
    impl Snapshot {
        fn of(owner: &SourceAdmission) -> Self {
            Self {
                binding: owner.binding().clone(),
                transport_id: owner.transport_id().into(),
                high_water: owner.high_water(),
                discard_through: owner.discard_through(),
                known_published_through: owner.known_published_through(),
                ready: owner.is_ready(),
            }
        }
    }

    fn assert_rejected(
        owner: &mut SourceAdmission,
        metadata: FrameMetadata<'_>,
        error: AdmissionError,
    ) {
        let before = Snapshot::of(owner);
        let incarnation = Arc::clone(&owner.incarnation);
        assert_eq!(owner.admit(metadata).unwrap_err(), error);
        assert_eq!(Snapshot::of(owner), before);
        assert!(Arc::ptr_eq(&owner.incarnation, &incarnation));
    }

    fn assert_rotation_rejected(owner: &mut SourceAdmission, fence: u64, error: AdmissionError) {
        let before = Snapshot::of(owner);
        let incarnation = Arc::clone(&owner.incarnation);
        assert_eq!(owner.rotate(fence), Err(error));
        assert_eq!(Snapshot::of(owner), before);
        assert!(Arc::ptr_eq(&owner.incarnation, &incarnation));
    }

    #[test]
    fn initial_acceptance_has_no_floor_and_preserves_signed_pts() {
        let binding = binding();
        let mut owner = SourceAdmission::new(TRANSPORT.into(), binding.clone()).unwrap();
        assert_rejected(
            &mut owner,
            frame(&binding, None, 100, 100),
            AdmissionError::PtsMissing,
        );
        assert!(!owner.is_ready());
        assert_eq!(owner.high_water(), None);
        assert_eq!(owner.known_published_through(), None);
        let first = owner.admit(frame(&binding, Some(i64::MIN), 0, 0)).unwrap();
        assert!(owner.is_ready());
        assert_eq!(owner.discard_through(), None);
        assert_eq!(owner.known_published_through(), Some(0));
        assert_eq!(first.binding(), &binding);
        assert_eq!(
            first.high_water(),
            HighWater {
                source_pts: i64::MIN,
                canonical_sequence: 0,
                native_publish_sequence: 0,
            }
        );
        assert_eq!(owner.high_water(), Some(first.high_water()));
        let second = owner.admit(frame(&binding, Some(-1), 1, 1)).unwrap();
        assert!(owner.accepts_policy_work(&first));
        assert!(owner.accepts_policy_work(&second));
        assert_eq!(owner.high_water().unwrap().source_pts, -1);
    }

    #[test]
    fn every_binding_mismatch_preserves_ready_and_unready_owners() {
        for prime in [false, true] {
            let binding = binding();
            let mut owner = SourceAdmission::new(TRANSPORT.into(), binding.clone()).unwrap();
            let accepted = if prime {
                Some(owner.admit(frame(&binding, Some(10), 20, 30)).unwrap())
            } else {
                None
            };
            for error in [
                AdmissionError::MalformedCamera,
                AdmissionError::UnknownSource,
                AdmissionError::BootMismatch,
                AdmissionError::ChildMismatch,
                AdmissionError::GenerationMismatch,
                AdmissionError::EpochMismatch,
                AdmissionError::TransformMismatch,
            ] {
                let mut wrong = binding.clone();
                match error {
                    AdmissionError::MalformedCamera => wrong.camera_id.clear(),
                    AdmissionError::UnknownSource => wrong.camera_id = "other-camera".into(),
                    AdmissionError::BootMismatch => wrong.worker_boot_id = "other-boot".into(),
                    AdmissionError::ChildMismatch => {
                        wrong.child_instance_id = "00000000-0000-0000-0000-000000000000".into();
                    }
                    AdmissionError::GenerationMismatch => wrong.source_generation += 1,
                    AdmissionError::EpochMismatch => wrong.stream_epoch += 1,
                    AdmissionError::TransformMismatch => {
                        wrong.transform_id = "other-transform".into()
                    }
                    _ => unreachable!("only canonical binding mismatches"),
                }
                assert_rejected(&mut owner, frame(&wrong, Some(999), 999, 999), error);
            }
            let mut uppercase_child = binding.clone();
            uppercase_child.child_instance_id.make_ascii_uppercase();
            assert_rejected(
                &mut owner,
                frame(&uppercase_child, Some(999), 999, 999),
                AdmissionError::ChildMismatch,
            );
            let other_transport = FrameMetadata {
                transport_id: "other-handle",
                ..frame(&binding, Some(999), 999, 999)
            };
            assert_rejected(
                &mut owner,
                other_transport,
                AdmissionError::TransportMismatch,
            );
            if let Some(accepted) = accepted {
                assert!(owner.accepts_policy_work(&accepted));
            }
            owner.admit(frame(&binding, Some(11), 21, 31)).unwrap();
            assert_eq!(owner.high_water().unwrap().native_publish_sequence, 31);
        }
    }

    #[test]
    fn canonical_mismatch_precedence_distinguishes_unknown_from_malformed() {
        let binding = binding();
        let mut owner = SourceAdmission::new(TRANSPORT.into(), binding.clone()).unwrap();
        let mut wrong = SourceBinding {
            camera_id: String::new(),
            worker_boot_id: "wrong-boot".into(),
            child_instance_id: "not-a-uuid".into(),
            source_generation: 99,
            stream_epoch: 99,
            transform_id: "wrong-transform".into(),
        };
        assert_rejected(
            &mut owner,
            frame(&wrong, None, 0, 0),
            AdmissionError::MalformedCamera,
        );
        wrong.camera_id = "unregistered".into();
        assert_rejected(
            &mut owner,
            frame(&wrong, None, 0, 0),
            AdmissionError::UnknownSource,
        );
        wrong.camera_id = binding.camera_id.clone();
        assert_rejected(
            &mut owner,
            frame(&wrong, None, 0, 0),
            AdmissionError::BootMismatch,
        );
        wrong.worker_boot_id = binding.worker_boot_id.clone();
        assert_rejected(
            &mut owner,
            frame(&wrong, None, 0, 0),
            AdmissionError::ChildMismatch,
        );
        wrong.child_instance_id = binding.child_instance_id.clone();
        assert_rejected(
            &mut owner,
            frame(&wrong, None, 0, 0),
            AdmissionError::GenerationMismatch,
        );
        wrong.source_generation = binding.source_generation;
        assert_rejected(
            &mut owner,
            frame(&wrong, None, 0, 0),
            AdmissionError::EpochMismatch,
        );
        wrong.stream_epoch = binding.stream_epoch;
        assert_rejected(
            &mut owner,
            frame(&wrong, None, 0, 0),
            AdmissionError::TransformMismatch,
        );
        wrong.transform_id = binding.transform_id.clone();
        assert_rejected(
            &mut owner,
            frame(&wrong, None, 0, 0),
            AdmissionError::PtsMissing,
        );
    }

    #[test]
    fn missing_pts_and_each_equal_or_regressing_watermark_are_independent() {
        let binding = binding();
        let mut owner = SourceAdmission::new(TRANSPORT.into(), binding.clone()).unwrap();
        let accepted = owner.admit(frame(&binding, Some(-10), 50, 100)).unwrap();
        for (pts, seq, native, error) in [
            (None, 51, 101, AdmissionError::PtsMissing),
            (Some(-10), 51, 101, AdmissionError::NonIncreasingPts),
            (Some(-11), 51, 101, AdmissionError::NonIncreasingPts),
            (
                Some(-9),
                50,
                101,
                AdmissionError::NonIncreasingCanonicalSequence,
            ),
            (
                Some(-9),
                49,
                101,
                AdmissionError::NonIncreasingCanonicalSequence,
            ),
            (
                Some(-9),
                51,
                100,
                AdmissionError::NonIncreasingNativePublicationSequence,
            ),
            (
                Some(-9),
                51,
                99,
                AdmissionError::NonIncreasingNativePublicationSequence,
            ),
        ] {
            assert_rejected(&mut owner, frame(&binding, pts, seq, native), error);
            assert!(owner.accepts_policy_work(&accepted));
        }
        let next = owner.admit(frame(&binding, Some(-9), 51, 101)).unwrap();
        assert_eq!(owner.high_water(), Some(next.high_water()));
        assert!(owner.accepts_policy_work(&accepted));
    }

    #[test]
    fn identity_storage_is_exact_and_has_no_invented_byte_limit() {
        let mut binding = binding();
        binding.camera_id = "카메라 ".repeat(300);
        binding.worker_boot_id = "boot".repeat(300);
        binding.transform_id = "transform".repeat(300);
        let transport = "transport".repeat(300);
        let mut owner = SourceAdmission::new(transport.clone(), binding.clone()).unwrap();
        let metadata = FrameMetadata {
            transport_id: &transport,
            ..frame(&binding, Some(1), 1, 1)
        };
        let work = owner.admit(metadata).unwrap();
        assert_eq!(work.binding(), &binding);
        assert_eq!(owner.transport_id(), transport);
        for field in 0..3 {
            let mut different_suffix = binding.clone();
            let error = match field {
                0 => {
                    different_suffix.camera_id.push('b');
                    AdmissionError::UnknownSource
                }
                1 => {
                    different_suffix.worker_boot_id.push('b');
                    AdmissionError::BootMismatch
                }
                _ => {
                    different_suffix.transform_id.push('b');
                    AdmissionError::TransformMismatch
                }
            };
            assert_rejected(
                &mut owner,
                FrameMetadata {
                    transport_id: &transport,
                    ..frame(&different_suffix, Some(2), 2, 2)
                },
                error,
            );
        }
        let other_transport = format!("{transport}b");
        assert_rejected(
            &mut owner,
            FrameMetadata {
                transport_id: &other_transport,
                ..frame(&binding, Some(2), 2, 2)
            },
            AdmissionError::TransportMismatch,
        );
    }

    #[test]
    fn registration_validates_only_canonical_camera_and_child_requirements() {
        let mut empty_camera = binding();
        empty_camera.camera_id.clear();
        assert_eq!(
            SourceAdmission::new(TRANSPORT.into(), empty_camera).unwrap_err(),
            AdmissionError::MalformedCamera,
        );
        for child in [
            "",
            "abcdef0123456789abcdef0123456789",
            "ABCDEF01-2345-6789-ABCD-EF0123456789",
            "abcdef01_2345-6789-abcd-ef0123456789",
            "gbcdef01-2345-6789-abcd-ef0123456789",
            "abcdef01-2345-6789-abcd-ef01234567890",
            "abcdef01-2345-6789-abcd-ef012345678가",
        ] {
            let mut invalid = binding();
            invalid.child_instance_id = child.into();
            assert_eq!(
                SourceAdmission::new(TRANSPORT.into(), invalid).unwrap_err(),
                AdmissionError::InvalidChildInstanceId,
            );
        }
        // Equality, not a guessed nonempty/trim rule, owns the other strings.
        let mut valid = binding();
        valid.worker_boot_id.clear();
        valid.transform_id.clear();
        valid.camera_id = " ".into();
        valid.child_instance_id = "00000000-0000-0000-0000-000000000000".into();
        let mut owner = SourceAdmission::new(TRANSPORT.into(), valid.clone()).unwrap();
        assert!(owner.admit(frame(&valid, Some(0), 0, 0)).is_ok());
    }

    #[test]
    fn same_handle_rotation_uses_the_supplied_publication_fence() {
        let binding = binding();
        let mut owner = SourceAdmission::new(TRANSPORT.into(), binding.clone()).unwrap();
        let old_work = owner.admit(frame(&binding, Some(100), 10, 5)).unwrap();
        // Native status is ahead of the last accepted/polled header.
        owner.rotate(12).unwrap();
        let rotated = owner.binding().clone();
        assert_eq!(
            rotated,
            SourceBinding {
                source_generation: binding.source_generation + 1,
                stream_epoch: binding.stream_epoch + 1,
                ..binding.clone()
            }
        );
        assert_eq!(owner.transport_id(), TRANSPORT);
        assert_eq!(owner.high_water(), None);
        assert_eq!(owner.discard_through(), Some(12));
        assert_eq!(owner.known_published_through(), Some(12));
        assert!(!owner.is_ready());
        assert!(!owner.accepts_policy_work(&old_work));
        assert_rejected(
            &mut owner,
            frame(&binding, Some(101), 11, 13),
            AdmissionError::GenerationMismatch,
        );
        for native in [0, 5, 11, 12] {
            assert_rejected(
                &mut owner,
                frame(&rotated, Some(i64::MIN), 0, native),
                AdmissionError::DiscardedPublication,
            );
        }
        let new_work = owner.admit(frame(&rotated, Some(i64::MIN), 0, 13)).unwrap();
        assert!(owner.is_ready());
        assert_eq!(owner.high_water(), Some(new_work.high_water()));
        assert_eq!(owner.known_published_through(), Some(13));
        assert!(owner.accepts_policy_work(&new_work));
        assert!(!owner.accepts_policy_work(&old_work));
        assert_eq!(owner.transport_id(), TRANSPORT);
    }

    #[test]
    fn zero_and_equal_fences_are_explicit_not_registration_defaults() {
        let binding = binding();
        let mut owner = SourceAdmission::new(TRANSPORT.into(), binding).unwrap();
        owner.rotate(0).unwrap();
        owner.rotate(0).unwrap();
        let rotated = owner.binding().clone();
        assert!(!owner.is_ready());
        assert_eq!(rotated.source_generation, 9);
        assert_eq!(rotated.stream_epoch, 13);
        assert_rejected(
            &mut owner,
            frame(&rotated, Some(0), 0, 0),
            AdmissionError::DiscardedPublication,
        );
        owner.admit(frame(&rotated, Some(0), 0, 1)).unwrap();
        assert_rotation_rejected(&mut owner, 0, AdmissionError::RegressingFence);
    }

    #[test]
    fn explicit_reregistration_clears_floor_and_invalidates_even_equal_binding_work() {
        let binding = binding();
        let mut owner = SourceAdmission::new(TRANSPORT.into(), binding.clone()).unwrap();
        let initial = owner.admit(frame(&binding, Some(100), 100, 100)).unwrap();
        owner.rotate(200).unwrap();
        let current = owner.binding().clone();
        let before = owner.admit(frame(&current, Some(101), 101, 201)).unwrap();
        owner.reregister(current.clone()).unwrap();
        assert_eq!(owner.binding(), &current);
        assert_eq!(owner.transport_id(), TRANSPORT);
        assert_eq!(owner.high_water(), None);
        assert_eq!(owner.discard_through(), None);
        assert_eq!(owner.known_published_through(), Some(201));
        assert!(!owner.is_ready());
        assert!(!owner.accepts_policy_work(&initial));
        assert!(!owner.accepts_policy_work(&before));
        let low = owner.admit(frame(&current, Some(i64::MIN), 0, 0)).unwrap();
        assert!(owner.accepts_policy_work(&low));
        assert_eq!(owner.high_water(), Some(low.high_water()));
        assert_eq!(owner.known_published_through(), Some(201));
        assert_rotation_rejected(&mut owner, 200, AdmissionError::RegressingFence);
        owner.rotate(201).unwrap();
        assert!(!owner.accepts_policy_work(&low));
        assert!(!owner.is_ready());
    }

    #[test]
    fn explicit_reregistration_replaces_binding_without_inferred_counters() {
        let first = binding();
        let mut owner = SourceAdmission::new(TRANSPORT.into(), first.clone()).unwrap();
        let old = owner.admit(frame(&first, Some(10), 20, 30)).unwrap();
        let replacement = SourceBinding {
            worker_boot_id: "different-boot".into(),
            child_instance_id: "00000000-0000-0000-0000-000000000001".into(),
            camera_id: "camera-b".into(),
            source_generation: 0,
            stream_epoch: 0,
            transform_id: "different-transform".into(),
        };
        owner.reregister(replacement.clone()).unwrap();
        assert!(!owner.accepts_policy_work(&old));
        assert_rejected(
            &mut owner,
            frame(&first, Some(11), 21, 31),
            AdmissionError::UnknownSource,
        );
        let accepted = owner.admit(frame(&replacement, Some(-100), 0, 0)).unwrap();
        assert_eq!(accepted.binding(), &replacement);
        assert_eq!(owner.discard_through(), None);
    }

    #[test]
    fn invalid_reregistration_is_atomic() {
        let binding = binding();
        let mut owner = SourceAdmission::new(TRANSPORT.into(), binding).unwrap();
        owner.rotate(10).unwrap();
        let current = owner.binding().clone();
        let accepted = owner.admit(frame(&current, Some(100), 100, 11)).unwrap();
        for error in [
            AdmissionError::MalformedCamera,
            AdmissionError::InvalidChildInstanceId,
        ] {
            let mut invalid = current.clone();
            if error == AdmissionError::MalformedCamera {
                invalid.camera_id.clear();
            } else {
                invalid.child_instance_id = "malformed".into();
            }
            let before = Snapshot::of(&owner);
            assert_eq!(owner.reregister(invalid), Err(error));
            assert_eq!(Snapshot::of(&owner), before);
            assert!(owner.accepts_policy_work(&accepted));
        }
    }

    #[test]
    fn policy_work_is_incarnation_scoped_not_latest_frame_or_binding_equality() {
        let binding = binding();
        let mut owner = SourceAdmission::new(TRANSPORT.into(), binding.clone()).unwrap();
        let older = owner.admit(frame(&binding, Some(1), 1, 1)).unwrap();
        let newer = owner.admit(frame(&binding, Some(2), 2, 2)).unwrap();
        assert!(owner.accepts_policy_work(&older.clone()));
        assert!(owner.accepts_policy_work(&newer));
        let independent = SourceAdmission::new(TRANSPORT.into(), binding.clone()).unwrap();
        assert!(!independent.accepts_policy_work(&older));
        owner.reregister(binding.clone()).unwrap();
        let repeated = owner.admit(frame(&binding, Some(1), 1, 1)).unwrap();
        assert_eq!(older.high_water(), repeated.high_water());
        assert!(!owner.accepts_policy_work(&older));
        assert!(!owner.accepts_policy_work(&newer));
        assert!(owner.accepts_policy_work(&repeated));
    }

    #[test]
    fn multiple_rotations_never_revive_old_work_or_reset_native_progress() {
        let original = binding();
        let mut owner = SourceAdmission::new(TRANSPORT.into(), original.clone()).unwrap();
        let mut work = vec![owner.admit(frame(&original, Some(100), 100, 1)).unwrap()];
        for turn in 1..=3 {
            let fence = turn * 10;
            owner.rotate(fence).unwrap();
            let current = owner.binding().clone();
            assert_eq!(current.source_generation, original.source_generation + turn);
            assert_eq!(current.stream_epoch, original.stream_epoch + turn);
            assert_eq!(owner.transport_id(), TRANSPORT);
            assert_eq!(owner.known_published_through(), Some(fence));
            assert!(!owner.is_ready());
            for old in &work {
                assert!(!owner.accepts_policy_work(old));
            }
            assert_rejected(
                &mut owner,
                frame(&current, Some(0), 0, fence),
                AdmissionError::DiscardedPublication,
            );
            let accepted = owner.admit(frame(&current, Some(0), 0, fence + 1)).unwrap();
            assert!(owner.accepts_policy_work(&accepted));
            work.push(accepted);
        }
        for old in &work[..work.len() - 1] {
            assert!(!owner.accepts_policy_work(old));
        }
        assert!(owner.accepts_policy_work(work.last().unwrap()));
        assert_eq!(owner.known_published_through(), Some(31));
    }

    #[test]
    fn regressing_fence_is_atomic_against_both_accepted_progress_and_prior_fence() {
        let binding = binding();
        let mut owner = SourceAdmission::new(TRANSPORT.into(), binding.clone()).unwrap();
        let accepted = owner.admit(frame(&binding, Some(10), 10, 20)).unwrap();
        assert_rotation_rejected(&mut owner, 19, AdmissionError::RegressingFence);
        assert!(owner.accepts_policy_work(&accepted));
        owner.rotate(30).unwrap();
        assert_rotation_rejected(&mut owner, 29, AdmissionError::RegressingFence);
        // A rejected large native ordinal cannot advance publication knowledge.
        let current = owner.binding().clone();
        assert_rejected(
            &mut owner,
            frame(&current, None, 1000, 1000),
            AdmissionError::PtsMissing,
        );
        owner.rotate(30).unwrap();
        assert!(!owner.is_ready());
        let current = owner.binding().clone();
        owner.admit(frame(&current, Some(0), 0, 31)).unwrap();
        assert_rotation_rejected(&mut owner, 30, AdmissionError::RegressingFence);
    }

    #[test]
    fn logical_counter_exhaustion_never_partially_rotates() {
        for (generation, epoch, error) in [
            (u64::MAX, 0, AdmissionError::GenerationExhausted),
            (0, u64::MAX, AdmissionError::EpochExhausted),
            (u64::MAX, u64::MAX, AdmissionError::GenerationExhausted),
        ] {
            let binding = SourceBinding {
                source_generation: generation,
                stream_epoch: epoch,
                ..binding()
            };
            let mut owner = SourceAdmission::new(TRANSPORT.into(), binding.clone()).unwrap();
            let accepted = owner.admit(frame(&binding, Some(1), 1, 1)).unwrap();
            assert_rotation_rejected(&mut owner, 50, error);
            assert!(owner.accepts_policy_work(&accepted));
            owner.admit(frame(&binding, Some(2), 2, 2)).unwrap();
        }
        let binding = SourceBinding {
            source_generation: u64::MAX - 1,
            stream_epoch: u64::MAX - 1,
            ..binding()
        };
        let mut owner = SourceAdmission::new(TRANSPORT.into(), binding).unwrap();
        owner.rotate(10).unwrap();
        assert_eq!(owner.binding().source_generation, u64::MAX);
        assert_eq!(owner.binding().stream_epoch, u64::MAX);
        assert_rotation_rejected(&mut owner, 11, AdmissionError::GenerationExhausted);
        assert_eq!(owner.discard_through(), Some(10));
    }

    #[test]
    fn full_width_pts_and_ordinals_compare_without_arithmetic_or_wraparound() {
        let binding = binding();
        let mut owner = SourceAdmission::new(TRANSPORT.into(), binding.clone()).unwrap();
        owner
            .admit(frame(&binding, Some(i64::MIN), u64::MAX - 1, u64::MAX - 1))
            .unwrap();
        let last = owner
            .admit(frame(&binding, Some(i64::MAX), u64::MAX, u64::MAX))
            .unwrap();
        assert_rejected(
            &mut owner,
            frame(&binding, Some(i64::MIN), 0, 0),
            AdmissionError::NonIncreasingPts,
        );
        assert_eq!(owner.high_water(), Some(last.high_water()));
        assert_rotation_rejected(&mut owner, u64::MAX - 1, AdmissionError::RegressingFence);
        // A max-valued fence is representable but admits no later u64 ordinal.
        // The native handle's exhaustion/replacement is outside this primitive.
        owner.rotate(u64::MAX).unwrap();
        let current = owner.binding().clone();
        assert_rejected(
            &mut owner,
            frame(&current, Some(0), 0, u64::MAX),
            AdmissionError::DiscardedPublication,
        );
        assert_rejected(
            &mut owner,
            frame(&current, Some(0), 0, 0),
            AdmissionError::DiscardedPublication,
        );
        assert!(!owner.is_ready());
    }
}
