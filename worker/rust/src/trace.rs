//! Owned decision evidence. Wire encoding and content identity belong elsewhere.
use std::collections::BTreeMap;
use std::fmt;

macro_rules! tokens {
    ($name:ident { $($variant:ident => $token:literal),+ $(,)? }) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
        pub enum $name { $($variant),+ }
        impl $name {
            pub fn as_str(self) -> &'static str {
                match self { $(Self::$variant => $token),+ }
            }
            pub fn from_token(token: &str) -> Option<Self> {
                match token { $($token => Some(Self::$variant),)+ _ => None }
            }
        }
        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(self.as_str())
            }
        }
    };
}

tokens!(DecisionTraceReason {
    ScoreMissing => "score-missing",
    TransitionConfirmed => "transition-confirmed",
    TransitionCandidate => "transition-candidate",
    FallActive => "fall-active",
    FallRecovered => "fall-recovered",
    BelowThreshold => "below-threshold",
    EpisodeAlreadyOpen => "episode-already-open",
    EpisodeReassociated => "episode-reassociated",
    EpisodeResolvedHold => "episode-resolved-hold",
    EpisodeCandidate => "episode-candidate",
    OutsideDetectionWindow => "outside-detection-window",
    BedRegionUnavailable => "bed-region-unavailable",
    BedObservationMissing => "bed-observation-missing",
    PersonObservationMissing => "person-observation-missing",
    IdentityHandoff => "identity-handoff",
    StaleTrackClear => "stale-track-clear",
    Assigned => "assigned",
    AssignmentHold => "assignment-hold",
    BelowContainment => "below-containment",
    Contained => "contained",
    ContainedPostureUnconfirmed => "contained-posture-unconfirmed",
    ContainedInOtherBed => "contained-in-other-bed",
    OutsideDwellExit => "outside-dwell-exit",
    OutsideDwell => "outside-dwell",
    OutsideNotArmed => "outside-not-armed",
});
tokens!(DecisionTraceState {
    Unknown => "unknown",
    Clear => "clear",
    TransitionCandidate => "transition-candidate",
    TransitionConfirmed => "transition-confirmed",
    Fallen => "fallen",
    NoDecision => "no-decision",
    Armed => "armed",
    Arming => "arming",
    Retired => "retired",
    Unassigned => "unassigned",
    Contained => "contained",
    OtherBed => "other-bed",
    Triggered => "triggered",
});
tokens!(DecisionTraceValueName {
    FallTransitionProbability => "fall_transition_probability",
    FallenProbability => "fallen_probability",
    TransitionThreshold => "transition_threshold",
    TransitionVotes => "transition_votes",
    TransitionWindow => "transition_window",
    ContainmentRatio => "containment_ratio",
    BedId => "bed_id",
    MinContainment => "min_containment",
    CandidateFrames => "candidate_frames",
    HoldFramesThreshold => "hold_frames_threshold",
    InBedDwellSec => "in_bed_dwell_sec",
    OutsideDwellSec => "outside_dwell_sec",
    InBedDwellThresholdSec => "in_bed_dwell_threshold_sec",
    OutsideDwellThresholdSec => "outside_dwell_threshold_sec",
    MaxOtherContainmentRatio => "max_other_containment_ratio",
    HipDepth => "hip_depth",
    TimeSec => "time_sec",
});
tokens!(DecisionTraceMissingReason {
    AdapterNotProvided => "adapter-not-provided",
    AdapterReturnedNoData => "adapter-returned-no-data",
    OutsideDetectionWindow => "outside-detection-window",
    NoLiveClassifiedTrack => "no-live-classified-track",
    ClassifierWarmup => "classifier-warmup",
    ClassifierStrideNotDue => "classifier-stride-not-due",
    ResampleGap => "resample-gap",
    BedRegionUnavailable => "bed-region-unavailable",
    BedObservationMissing => "bed-observation-missing",
    TrackNoLongerLive => "track-no-longer-live",
    NoObservedPerson => "no-observed-person",
    PoseUnavailable => "pose-unavailable",
    BedPolygonInvalid => "bed-polygon-invalid",
    NoPoseEvidence => "no-pose-evidence",
    TimeNotProvided => "time-not-provided",
});

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NonFiniteTraceValue;
impl fmt::Display for NonFiniteTraceValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("decision trace values must be finite")
    }
}
impl std::error::Error for NonFiniteTraceValue {}

/// Canonical six-decimal scalar, without changing the probabilities used for decisions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TraceFloat(f64);
impl TraceFloat {
    pub fn new(value: f64) -> Result<Self, NonFiniteTraceValue> {
        if !value.is_finite() {
            return Err(NonFiniteTraceValue);
        }
        // Fixed decimal formatting rounds the original binary value ties-to-even.
        // Multiplication by 1e6 followed by rounding introduces a second rounding
        // and disagrees with Python round(value, 6) at decimal half boundaries.
        let rounded: f64 = format!("{value:.6}")
            .parse()
            .expect("a formatted finite f64 parses as f64");
        Ok(Self(if rounded == 0.0 { 0.0 } else { rounded }))
    }
    pub fn get(self) -> f64 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NumericTraceValue {
    Integer(usize),
    Float(TraceFloat),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionTraceSnapshotError {
    OverlappingValue(DecisionTraceValueName),
}
impl fmt::Display for DecisionTraceSnapshotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OverlappingValue(name) => {
                write!(
                    f,
                    "decision trace value {name} cannot be both known and missing"
                )
            }
        }
    }
}
impl std::error::Error for DecisionTraceSnapshotError {}

/// The same known/missing value vocabulary as the policy's Python snapshot.
/// Enum keys bound both maps; neither keys nor reason/state tokens retain strings.
/// Maps are owned, disjoint, and accessible only through shared references.
///
/// Direct mutation of either map is unavailable outside this module:
///
/// ```compile_fail
/// use seeon_worker::trace::DecisionTraceSnapshot;
///
/// fn clear_values(snapshot: &mut DecisionTraceSnapshot) {
///     snapshot.values.clear();
/// }
/// ```
///
/// ```compile_fail
/// use seeon_worker::trace::DecisionTraceSnapshot;
///
/// fn clear_missing_values(snapshot: &mut DecisionTraceSnapshot) {
///     snapshot.missing_values.clear();
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionTraceSnapshot {
    pub reason: DecisionTraceReason,
    pub previous_state: DecisionTraceState,
    pub current_state: DecisionTraceState,
    pub triggered: bool,
    pub track_id: Option<u64>,
    pub bed_id: Option<u64>,
    values: BTreeMap<DecisionTraceValueName, NumericTraceValue>,
    missing_values: BTreeMap<DecisionTraceValueName, DecisionTraceMissingReason>,
}
impl DecisionTraceSnapshot {
    pub fn new(
        reason: DecisionTraceReason,
        states: (DecisionTraceState, DecisionTraceState),
        triggered: bool,
        track_id: Option<u64>,
        bed_id: Option<u64>,
        values: BTreeMap<DecisionTraceValueName, NumericTraceValue>,
        missing_values: BTreeMap<DecisionTraceValueName, DecisionTraceMissingReason>,
    ) -> Result<Self, DecisionTraceSnapshotError> {
        let (previous_state, current_state) = states;
        for &name in values.keys() {
            if missing_values.contains_key(&name) {
                return Err(DecisionTraceSnapshotError::OverlappingValue(name));
            }
        }
        Ok(Self {
            reason,
            previous_state,
            current_state,
            triggered,
            track_id,
            bed_id,
            values,
            missing_values,
        })
    }

    /// Known values cannot be mutated, even through a mutable snapshot reference.
    ///
    /// ```compile_fail
    /// use seeon_worker::trace::DecisionTraceSnapshot;
    ///
    /// fn clear_values(snapshot: &mut DecisionTraceSnapshot) {
    ///     snapshot.values().clear();
    /// }
    /// ```
    pub fn values(&self) -> &BTreeMap<DecisionTraceValueName, NumericTraceValue> {
        &self.values
    }

    /// Missing values cannot be mutated, even through a mutable snapshot reference.
    ///
    /// ```compile_fail
    /// use seeon_worker::trace::DecisionTraceSnapshot;
    ///
    /// fn clear_missing_values(snapshot: &mut DecisionTraceSnapshot) {
    ///     snapshot.missing_values().clear();
    /// }
    /// ```
    pub fn missing_values(&self) -> &BTreeMap<DecisionTraceValueName, DecisionTraceMissingReason> {
        &self.missing_values
    }

    pub(crate) fn missing(
        track_id: u64,
        current: DecisionTraceState,
        reason: DecisionTraceMissingReason,
    ) -> Self {
        Self::new(
            DecisionTraceReason::ScoreMissing,
            (current, current),
            false,
            Some(track_id),
            None,
            BTreeMap::new(),
            BTreeMap::from([(DecisionTraceValueName::FallTransitionProbability, reason)]),
        )
        .expect("missing-only trace has no known values to overlap")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_reasons_round_trip_every_compiled_token() {
        for (reason, token) in [
            (
                DecisionTraceMissingReason::AdapterNotProvided,
                "adapter-not-provided",
            ),
            (
                DecisionTraceMissingReason::AdapterReturnedNoData,
                "adapter-returned-no-data",
            ),
            (
                DecisionTraceMissingReason::OutsideDetectionWindow,
                "outside-detection-window",
            ),
            (
                DecisionTraceMissingReason::NoLiveClassifiedTrack,
                "no-live-classified-track",
            ),
            (
                DecisionTraceMissingReason::ClassifierWarmup,
                "classifier-warmup",
            ),
            (
                DecisionTraceMissingReason::ClassifierStrideNotDue,
                "classifier-stride-not-due",
            ),
            (DecisionTraceMissingReason::ResampleGap, "resample-gap"),
            (
                DecisionTraceMissingReason::BedRegionUnavailable,
                "bed-region-unavailable",
            ),
            (
                DecisionTraceMissingReason::BedObservationMissing,
                "bed-observation-missing",
            ),
            (
                DecisionTraceMissingReason::TrackNoLongerLive,
                "track-no-longer-live",
            ),
            (
                DecisionTraceMissingReason::NoObservedPerson,
                "no-observed-person",
            ),
            (
                DecisionTraceMissingReason::PoseUnavailable,
                "pose-unavailable",
            ),
            (
                DecisionTraceMissingReason::BedPolygonInvalid,
                "bed-polygon-invalid",
            ),
            (
                DecisionTraceMissingReason::NoPoseEvidence,
                "no-pose-evidence",
            ),
            (
                DecisionTraceMissingReason::TimeNotProvided,
                "time-not-provided",
            ),
        ] {
            assert_eq!(DecisionTraceMissingReason::from_token(token), Some(reason));
            assert_eq!(reason.as_str(), token);
            assert_eq!(reason.to_string(), token);
        }
    }

    #[test]
    fn missing_reasons_reject_unknown_or_nonexact_tokens() {
        for token in [
            "",
            "arbitrary-text",
            "score-missing",
            "POSE_UNAVAILABLE",
            "pose_unavailable",
            "Pose-Unavailable",
            " pose-unavailable",
            "pose-unavailable ",
            "pose-unavailable\n",
            "pose-unavailable\0",
            "pose-unavailable-extra",
        ] {
            assert_eq!(DecisionTraceMissingReason::from_token(token), None);
        }
    }

    #[test]
    fn checked_snapshot_rejects_each_overlapping_key() {
        let values: BTreeMap<_, _> = [
            DecisionTraceValueName::FallTransitionProbability,
            DecisionTraceValueName::FallenProbability,
            DecisionTraceValueName::TransitionThreshold,
            DecisionTraceValueName::TransitionVotes,
            DecisionTraceValueName::TransitionWindow,
            DecisionTraceValueName::ContainmentRatio,
            DecisionTraceValueName::BedId,
            DecisionTraceValueName::MinContainment,
            DecisionTraceValueName::CandidateFrames,
            DecisionTraceValueName::HoldFramesThreshold,
            DecisionTraceValueName::InBedDwellSec,
            DecisionTraceValueName::OutsideDwellSec,
            DecisionTraceValueName::InBedDwellThresholdSec,
            DecisionTraceValueName::OutsideDwellThresholdSec,
            DecisionTraceValueName::MaxOtherContainmentRatio,
            DecisionTraceValueName::HipDepth,
            DecisionTraceValueName::TimeSec,
        ]
        .into_iter()
        .map(|name| (name, NumericTraceValue::Integer(1)))
        .collect();
        for &name in values.keys() {
            let result = DecisionTraceSnapshot::new(
                DecisionTraceReason::ScoreMissing,
                (DecisionTraceState::Unknown, DecisionTraceState::Unknown),
                false,
                Some(7),
                None,
                values.clone(),
                BTreeMap::from([(name, DecisionTraceMissingReason::PoseUnavailable)]),
            );
            assert_eq!(
                result,
                Err(DecisionTraceSnapshotError::OverlappingValue(name))
            );
        }
    }

    #[test]
    fn checked_snapshot_preserves_disjoint_maps_and_metadata() {
        let values = BTreeMap::from([
            (
                DecisionTraceValueName::FallTransitionProbability,
                NumericTraceValue::Float(TraceFloat::new(0.1234567).unwrap()),
            ),
            (
                DecisionTraceValueName::TransitionVotes,
                NumericTraceValue::Integer(usize::MAX),
            ),
        ]);
        let missing_values = BTreeMap::from([(
            DecisionTraceValueName::FallenProbability,
            DecisionTraceMissingReason::PoseUnavailable,
        )]);
        for (values, missing_values) in [
            (values.clone(), BTreeMap::new()),
            (BTreeMap::new(), missing_values.clone()),
            (values, missing_values),
            (BTreeMap::new(), BTreeMap::new()),
        ] {
            let snapshot = DecisionTraceSnapshot::new(
                DecisionTraceReason::TransitionConfirmed,
                (
                    DecisionTraceState::TransitionCandidate,
                    DecisionTraceState::TransitionConfirmed,
                ),
                true,
                Some(7),
                Some(9),
                values.clone(),
                missing_values.clone(),
            )
            .unwrap();
            assert_eq!(snapshot.reason, DecisionTraceReason::TransitionConfirmed);
            assert_eq!(
                snapshot.previous_state,
                DecisionTraceState::TransitionCandidate
            );
            assert_eq!(
                snapshot.current_state,
                DecisionTraceState::TransitionConfirmed
            );
            assert!(snapshot.triggered);
            assert_eq!(snapshot.track_id, Some(7));
            assert_eq!(snapshot.bed_id, Some(9));
            assert_eq!(snapshot.values(), &values);
            assert_eq!(snapshot.missing_values(), &missing_values);
        }
    }

    #[test]
    fn trace_rounding_uses_binary_value_and_normalizes_negative_zero() {
        for (input, expected) in [
            (0.1234564, 0.123456),
            (0.1234566, 0.123457),
            (0.0078125, 0.007812),
            (0.0234375, 0.023438),
            (0.0000005, 0.0),
            (0.0000015, 0.000002),
            (-0.0000001, 0.0),
            (f64::MAX, f64::MAX),
        ] {
            assert_eq!(TraceFloat::new(input).unwrap().get(), expected);
        }
        assert!(!TraceFloat::new(-0.0).unwrap().get().is_sign_negative());
        for input in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(TraceFloat::new(input), Err(NonFiniteTraceValue));
        }
    }

    #[test]
    fn missing_snapshot_has_no_invented_score_or_policy_values() {
        let snapshot = DecisionTraceSnapshot::missing(
            7,
            DecisionTraceState::TransitionConfirmed,
            DecisionTraceMissingReason::ClassifierStrideNotDue,
        );
        assert_eq!(snapshot.previous_state, snapshot.current_state);
        assert!(!snapshot.triggered);
        assert!(snapshot.values().is_empty());
        assert_eq!(snapshot.missing_values().len(), 1);
        let (key, reason) = snapshot.missing_values().first_key_value().unwrap();
        assert_eq!(key.as_str(), "fall_transition_probability");
        assert_eq!(reason.as_str(), "classifier-stride-not-due");
        assert_eq!(DecisionTraceReason::from_token("arbitrary-text"), None);
    }
}

#[cfg(test)]
mod bed_vocabulary_tests {
    use super::*;

    #[test]
    fn bed_and_window_tokens_round_trip_without_legacy_fall_renames() {
        use DecisionTraceReason as R;
        for (reason, token) in [
            (R::OutsideDetectionWindow, "outside-detection-window"),
            (R::BedRegionUnavailable, "bed-region-unavailable"),
            (R::BedObservationMissing, "bed-observation-missing"),
            (R::PersonObservationMissing, "person-observation-missing"),
            (R::IdentityHandoff, "identity-handoff"),
            (R::StaleTrackClear, "stale-track-clear"),
            (R::Assigned, "assigned"),
            (R::AssignmentHold, "assignment-hold"),
            (R::BelowContainment, "below-containment"),
            (R::Contained, "contained"),
            (
                R::ContainedPostureUnconfirmed,
                "contained-posture-unconfirmed",
            ),
            (R::ContainedInOtherBed, "contained-in-other-bed"),
            (R::OutsideDwellExit, "outside-dwell-exit"),
            (R::OutsideDwell, "outside-dwell"),
            (R::OutsideNotArmed, "outside-not-armed"),
            (R::FallActive, "fall-active"),
            (R::TransitionConfirmed, "transition-confirmed"),
        ] {
            assert_eq!(R::from_token(token), Some(reason));
            assert_eq!(reason.as_str(), token);
        }
        use DecisionTraceState as S;
        for (state, token) in [
            (S::NoDecision, "no-decision"),
            (S::Armed, "armed"),
            (S::Arming, "arming"),
            (S::Retired, "retired"),
            (S::Unassigned, "unassigned"),
            (S::Contained, "contained"),
            (S::OtherBed, "other-bed"),
            (S::Triggered, "triggered"),
        ] {
            assert_eq!(S::from_token(token), Some(state));
            assert_eq!(state.as_str(), token);
        }
        use DecisionTraceValueName as V;
        for (name, token) in [
            (V::ContainmentRatio, "containment_ratio"),
            (V::BedId, "bed_id"),
            (V::MinContainment, "min_containment"),
            (V::CandidateFrames, "candidate_frames"),
            (V::HoldFramesThreshold, "hold_frames_threshold"),
            (V::InBedDwellSec, "in_bed_dwell_sec"),
            (V::OutsideDwellSec, "outside_dwell_sec"),
            (V::InBedDwellThresholdSec, "in_bed_dwell_threshold_sec"),
            (V::OutsideDwellThresholdSec, "outside_dwell_threshold_sec"),
            (V::MaxOtherContainmentRatio, "max_other_containment_ratio"),
            (V::HipDepth, "hip_depth"),
            (V::TimeSec, "time_sec"),
        ] {
            assert_eq!(V::from_token(token), Some(name));
            assert_eq!(name.as_str(), token);
        }
        for token in [
            "armed ",
            "outside_dwell_exit",
            "bed-exit-onset",
            "stale-track-exit",
        ] {
            assert_eq!(R::from_token(token), None);
        }
    }
}
