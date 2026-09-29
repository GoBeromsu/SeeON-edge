//! CPU decisions from worker/domains/bed_exit/{detector,schema,latch}.py and
//! worker/domains/staleness.py, layered on the existing EpisodeAuthority.
//!
//! This is a SCORED NUMERIC seam, not a geometry/inference implementation.
//! The producer must supply the actual Python-equivalent containment ratios
//! (including polygon rasterization) and current-person/prior-box overlap
//! ratios. Missing geometry is rejected, never synthesized from an AABB.
//! Assignment snapshots expose the exact prior boxes needed by that producer.
//!
//! Bounded admission is deliberately narrower than Python's unbounded ints
//! and permissive floats: IDs are u64, frame/coordinates i64, counters checked,
//! all supplied scalars finite, retained collections explicitly budgeted. Signed
//! PTS and clock reversals are supported; no monotonic-time requirement or
//! arbitrary duration cap is added. Pose scalars and scored ratios are NOT
//! clamped to [0,1]. Unused pose fields and duplicate feature/observation IDs
//! are preserved (last feature wins, observation order matters).
use std::collections::{BTreeMap, BTreeSet};

use crate::detection_window::{AwareDateTime, DetectionWindow};
use crate::episode::{BusinessEvent, EpisodeAuthority, EpisodeProposal, EpisodeState};
use crate::trace::DecisionTraceSnapshot;

pub use errors::{BedExitError, BedExitFailure, BedExitPhase};
use evidence::{armed_state, missing_trace, trace};
pub use vocabulary::*;

pub const DEFAULT_STALE_AFTER_SEC: f64 = 3.0;

mod vocabulary {
    use super::{AwareDateTime, BusinessEvent, DetectionWindow};

    #[derive(Debug, Clone, PartialEq)]
    pub struct BoundingBox {
        pub x1: i64,
        pub y1: i64,
        pub x2: i64,
        pub y2: i64,
        pub confidence: f64,
        pub polygon: Option<Vec<(i64, i64)>>,
    }
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum BedRegionCacheState {
        Fresh,
        Cached,
        Empty,
        Expired,
    }
    impl BedRegionCacheState {
        pub fn as_str(self) -> &'static str {
            match self {
                Self::Fresh => "fresh",
                Self::Cached => "cached",
                Self::Empty => "empty",
                Self::Expired => "expired",
            }
        }
        pub(super) fn usable(self) -> bool {
            matches!(self, Self::Fresh | Self::Cached)
        }
    }
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct BedRegionDebugSnapshot {
        pub source: BedRegionCacheState,
        pub empty_cycles: u64,
    }
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct BedPoseFeatures {
        pub track_id: u64,
        pub bed_id: Option<u64>,
        pub torso_in_frac: f64,
        pub lower_in_frac: f64,
        pub keypoint_in_frac: f64,
        pub hip_depth: f64,
        pub torso_angle: f64,
        pub centroid_displacement: f64,
        pub hip_x_rel: f64,
        pub hip_y_rel: f64,
        pub observability: f64,
        pub bed_polygon_valid: bool,
    }
    impl BedPoseFeatures {
        pub(super) fn confirms_in_bed(self) -> bool {
            self.bed_polygon_valid && self.observability >= 0.35 && self.hip_depth >= 0.10
        }
    }
    /// `ratios[i]` = containment_ratio(current person box i, previous_box).
    /// Bind evidence to BOTH the previous track and its exact last observed box.
    /// Supply a row for every stale, armed, mid-exit assignment with a current
    /// own bed and last box when there is any unclaimed live observation.
    #[derive(Debug, Clone, PartialEq)]
    pub struct PriorBoxContainments {
        pub previous_track_id: u64,
        pub previous_box: BoundingBox,
        pub ratios: Vec<f64>,
    }
    #[derive(Debug, Clone, PartialEq)]
    pub struct BedExitInput {
        pub person_boxes: Vec<BoundingBox>,
        pub bed_boxes: Vec<BoundingBox>,
        /// Empty means Python's positional-ID mode, NOT all-unknown tracking.
        pub track_ids: Vec<Option<u64>>,
        pub live_track_ids: Vec<u64>,
        pub bed_pose_features: Vec<BedPoseFeatures>,
        /// Row i contains containment_ratio(person i, bed j), in bed order.
        /// Required only when the region is usable and bed_boxes is nonempty.
        pub containments: Vec<Vec<f64>>,
        pub prior_box_containments: Vec<PriorBoxContainments>,
        pub time_sec: Option<f64>,
        pub frame_index: i64,
        pub bed_region: BedRegionDebugSnapshot,
    }
    /// Separate samples model observe() and snapshot() calling the injected
    /// monotonic clock independently. Wall time is consulted only with a window.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct BedExitClocks {
        pub wall_time: Option<AwareDateTime>,
        pub observed_at: f64,
        pub snapshot_at: f64,
    }
    #[derive(Debug, Clone, PartialEq)]
    pub struct BedExitConfig {
        pub camera_id: String,
        pub facility_id: String,
        pub min_containment: f64,
        pub hold_frames: usize,
        /// Wire metadata only; elapsed outside_dwell_sec owns exit timing.
        pub grace_frames: u64,
        pub night_window: Option<DetectionWindow>,
        pub in_bed_dwell_sec: f64,
        pub outside_dwell_sec: f64,
    }
    /// No eviction, defaults, or hidden unlimited identity cache. Traces are
    /// bounded by tracks + observations (or one missing-input row); raw events
    /// and recovery rows by observations, lost rows by tracks. Geometry storage
    /// is bounded by observations*beds + tracks*observations. Every box has at
    /// most polygon_points vertices. Authority rows remain until owner teardown.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct BedExitCapacities {
        pub tracks: usize,
        pub observations: usize,
        pub beds: usize,
        pub pose_features: usize,
        pub polygon_points: usize,
        pub episodes: usize,
    }
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum BedOccupancy {
        Empty,
        Occupied,
        Exit,
        Covered,
        Unknown,
    }
    impl BedOccupancy {
        pub fn as_str(self) -> &'static str {
            match self {
                Self::Empty => "empty",
                Self::Occupied => "occupied",
                Self::Exit => "exit",
                Self::Covered => "covered",
                Self::Unknown => "unknown",
            }
        }
    }
    #[derive(Debug, Clone, PartialEq)]
    pub struct BedStatus {
        pub bed_id: usize,
        pub box_value: BoundingBox,
        pub occupancy: BedOccupancy,
        pub person_id: Option<u64>,
    }
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct BedExitEvent {
        pub person_id: u64,
        pub bed_id: usize,
    }
    #[derive(Debug, Clone, PartialEq)]
    pub struct BedExitFrame {
        pub statuses: Vec<BedStatus>,
        pub events: Vec<BedExitEvent>,
    }
    #[derive(Debug, Clone, PartialEq)]
    pub struct BedExitDebugSnapshot {
        pub frame_index: Option<i64>,
        pub person_boxes: Vec<BoundingBox>,
        pub bed_boxes: Vec<BoundingBox>,
        pub statuses: Vec<BedStatus>,
        pub events: Vec<BedExitEvent>,
        pub bed_region: Option<BedRegionDebugSnapshot>,
        pub stale: bool,
        pub observation_age_sec: Option<f64>,
    }
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct FreshnessSnapshot {
        pub stale: bool,
        pub observation_age_sec: Option<f64>,
    }
    #[derive(Debug, Clone, Copy, Default, PartialEq)]
    pub struct BedExitScoring {
        pub max_containment_observed: f64,
        pub grace_positive_transitions: u64,
        pub assignments_made: u64,
    }
    /// Telemetry is data, never a fallible callback on the emission path.
    #[derive(Debug, Clone, PartialEq)]
    pub struct BedExitOutcome {
        pub events: Vec<BusinessEvent>,
        pub scoring_observation: Option<BedExitScoring>,
    }
    #[derive(Debug, Clone, Default, PartialEq)]
    pub struct AssignmentSnapshot {
        pub bed_id: Option<usize>,
        pub candidate_bed_id: Option<usize>,
        pub candidate_frames: usize,
        pub in_bed_dwell_sec: f64,
        pub outside_dwell_sec: f64,
        pub armed: bool,
        pub last_time_sec: Option<f64>,
        pub last_box: Option<BoundingBox>,
    }
}

mod errors {
    use crate::detection_window::DetectionWindowError;
    use crate::episode::{BusinessEvent, EpisodeError};
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum BedExitFailure {
        InvalidConfig(&'static str),
        InvalidIdentity,
        InvalidCapacities,
        Capacity(&'static str),
        NonFinite(&'static str),
        InvalidShape(&'static str),
        MissingPriorOverlap(u64),
        MismatchedPriorBox(u64),
        Overflow(&'static str),
        MissingWallClock,
        Window(DetectionWindowError),
        Episode(EpisodeError),
    }
    impl From<EpisodeError> for BedExitFailure {
        fn from(value: EpisodeError) -> Self {
            Self::Episode(value)
        }
    }
    impl std::fmt::Display for BedExitFailure {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "bed-exit failure: {self:?}")
        }
    }
    impl std::error::Error for BedExitFailure {}
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum BedExitPhase {
        Frame,
        Expire,
        Onset,
        TrackLost,
        Recovery,
    }
    /// Rejected means no state changed. Fatal means STOP this camera owner;
    /// inspection/release remain available but update/coast/window changes do
    /// not. Previously accepted events MUST be delivered or explicitly released,
    /// including the ordered prefix returned on this error. Never retry/reset
    /// silently: partial assignment/authority changes are not rolled back.
    #[derive(Debug, Clone, PartialEq)]
    pub enum BedExitError {
        Rejected(BedExitFailure),
        FatalPartialState {
            phase: BedExitPhase,
            track_id: Option<u64>,
            cause: BedExitFailure,
            emitted_events: Vec<BusinessEvent>,
        },
        Poisoned(BedExitFailure),
    }
    impl From<BedExitFailure> for BedExitError {
        fn from(value: BedExitFailure) -> Self {
            Self::Rejected(value)
        }
    }
    impl std::fmt::Display for BedExitError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::Rejected(cause) => write!(f, "bed-exit input rejected: {cause}"),
                Self::FatalPartialState {
                    phase,
                    track_id,
                    cause,
                    ..
                } => write!(
                    f,
                    "fatal partial bed-exit update in {phase:?} at {track_id:?}: {cause}"
                ),
                Self::Poisoned(cause) => write!(f, "bed-exit owner stopped: {cause}"),
            }
        }
    }
    impl std::error::Error for BedExitError {}
}

/// One owner per camera, boot, stream epoch and source generation.
#[derive(Debug)]
pub struct BedExitMonitor {
    config: BedExitConfig,
    night_window: Option<DetectionWindow>,
    capacities: BedExitCapacities,
    assignments: BTreeMap<u64, AssignmentSnapshot>,
    episodes: EpisodeAuthority,
    freshness: ObservationFreshness,
    scoring: BedExitScoring,
    last_debug_snapshot: Option<BedExitDebugSnapshot>,
    last_trace_snapshots: Vec<DecisionTraceSnapshot>,
    recovery_events: Vec<BedExitEvent>,
    lost_track_ids: Vec<u64>,
    poisoned: Option<BedExitFailure>,
    last_update_complete: bool,
    phase: BedExitPhase,
    active_track: Option<u64>,
}
impl BedExitMonitor {
    pub fn new(
        config: BedExitConfig,
        boot_id: impl Into<String>,
        stream_epoch: impl Into<String>,
        source_generation: u64,
        capacities: BedExitCapacities,
        stale_after_sec: f64,
    ) -> Result<Self, BedExitFailure> {
        admission::config(&config, capacities)?;
        let freshness = ObservationFreshness::new(stale_after_sec)?;
        let episodes = EpisodeAuthority::new(
            boot_id,
            stream_epoch,
            source_generation,
            capacities.episodes,
            1,
        )?;
        Ok(Self {
            night_window: config.night_window.clone(),
            config,
            capacities,
            assignments: BTreeMap::new(),
            episodes,
            freshness,
            scoring: BedExitScoring::default(),
            last_debug_snapshot: None,
            last_trace_snapshots: Vec::new(),
            recovery_events: Vec::new(),
            lost_track_ids: Vec::new(),
            poisoned: None,
            last_update_complete: false,
            phase: BedExitPhase::Frame,
            active_track: None,
        })
    }
    pub fn config(&self) -> &BedExitConfig {
        &self.config
    }
    pub fn night_window(&self) -> Option<&DetectionWindow> {
        self.night_window.as_ref()
    }
    pub fn assignments(&self) -> &BTreeMap<u64, AssignmentSnapshot> {
        &self.assignments
    }
    pub fn last_debug_snapshot(&self) -> Option<&BedExitDebugSnapshot> {
        self.last_debug_snapshot.as_ref()
    }
    pub fn last_trace_snapshots(&self) -> &[DecisionTraceSnapshot] {
        &self.last_trace_snapshots
    }
    pub fn last_shadow_trace_count(&self) -> usize {
        0
    }
    pub fn last_recovery_events(&self) -> &[BedExitEvent] {
        &self.recovery_events
    }
    pub fn last_lost_track_ids(&self) -> &[u64] {
        &self.lost_track_ids
    }
    pub fn last_update_complete(&self) -> bool {
        self.last_update_complete
    }
    pub fn scoring(&self) -> BedExitScoring {
        self.scoring
    }
    pub fn track_id_switch_absorbed_total(&self) -> u64 {
        self.episodes.track_id_switch_absorbed_total()
    }
    pub fn freshness_snapshot(
        &self,
        snapshot_at: f64,
    ) -> Result<FreshnessSnapshot, BedExitFailure> {
        self.freshness.snapshot(snapshot_at)
    }
    pub fn last_episode_disposition(&self) -> Option<crate::episode::ProposalDisposition> {
        self.episodes.last_disposition()
    }
    pub fn episode_state(&self, track_id: u64, bed_id: usize) -> EpisodeState {
        self.episodes.state_for(
            &self.config.camera_id,
            "bed-exit",
            Some(bed_id as u64),
            track_id,
        )
    }
    pub fn update_night_window(
        &mut self,
        window: Option<DetectionWindow>,
    ) -> Result<(), BedExitError> {
        self.ensure_running()?;
        self.night_window = window;
        Ok(())
    }
    /// Release only the authority's exact accepted identity. Does NOT re-arm.
    /// Available even after a fatal error so failed durable staging can be named.
    pub fn release_onset(&mut self, event_identity: &str) -> bool {
        self.episodes.release(event_identity)
    }
    fn ensure_running(&self) -> Result<(), BedExitError> {
        match self.poisoned {
            Some(cause) => Err(BedExitError::Poisoned(cause)),
            None => Ok(()),
        }
    }
    pub fn coast(
        &mut self,
        frame_index: Option<i64>,
        snapshot_at: f64,
    ) -> Result<BedExitOutcome, BedExitError> {
        self.ensure_running()?;
        let freshness = self.freshness.snapshot(snapshot_at)?;
        let previous = self.last_debug_snapshot.take();
        let (person_boxes, bed_boxes, mut statuses, bed_region) = match previous {
            Some(previous) => (
                previous.person_boxes,
                previous.bed_boxes,
                previous.statuses,
                previous.bed_region,
            ),
            None => (Vec::new(), Vec::new(), Vec::new(), None),
        };
        for status in &mut statuses {
            status.occupancy = BedOccupancy::Covered;
        }
        self.last_debug_snapshot = Some(BedExitDebugSnapshot {
            frame_index,
            person_boxes,
            bed_boxes,
            statuses,
            events: Vec::new(),
            bed_region,
            stale: freshness.stale,
            observation_age_sec: freshness.observation_age_sec,
        });
        // Python coast deliberately retains the previous traces and PTS anchor.
        self.last_update_complete = true;
        Ok(BedExitOutcome {
            events: Vec::new(),
            scoring_observation: None,
        })
    }
    pub fn update(
        &mut self,
        input: &BedExitInput,
        clocks: BedExitClocks,
    ) -> Result<BedExitOutcome, BedExitError> {
        self.ensure_running()?;
        admission::input(self, input)?;
        if !input.bed_region.source.usable() || input.bed_boxes.is_empty() {
            self.record_unavailable(input);
            self.last_update_complete = true;
            return Ok(BedExitOutcome {
                events: Vec::new(),
                scoring_observation: None,
            });
        }
        let prepared = admission::prepare(self, input)?;
        let in_window = match &self.night_window {
            Some(window) => window
                .contains(clocks.wall_time.ok_or(BedExitFailure::MissingWallClock)?)
                .map_err(BedExitFailure::Window)?,
            None => true,
        };
        // Preflight all fallible clock admission before any mutable decisions.
        let fresh = freshness_at(
            Some(clocks.observed_at),
            clocks.snapshot_at,
            self.freshness.stale_after_sec,
        )?;
        self.last_update_complete = false;
        self.phase = BedExitPhase::Frame;
        self.active_track = None;
        self.recovery_events.clear();
        self.lost_track_ids.clear();
        self.last_trace_snapshots.clear();
        let mut emitted = Vec::new();
        if let Err(cause) = self.process(
            input,
            prepared,
            in_window,
            clocks.observed_at,
            fresh,
            &mut emitted,
        ) {
            self.poisoned = Some(cause);
            return Err(BedExitError::FatalPartialState {
                phase: self.phase,
                track_id: self.active_track,
                cause,
                emitted_events: emitted,
            });
        }
        self.last_update_complete = true;
        Ok(BedExitOutcome {
            events: emitted,
            scoring_observation: Some(self.scoring),
        })
    }
    fn process(
        &mut self,
        input: &BedExitInput,
        prepared: admission::Prepared,
        in_window: bool,
        observed_at: f64,
        fresh: FreshnessSnapshot,
        emitted: &mut Vec<BusinessEvent>,
    ) -> Result<(), BedExitFailure> {
        let frame = self.update_frame(input, prepared)?;
        self.freshness.last_observed_at = Some(observed_at);
        self.last_debug_snapshot = Some(BedExitDebugSnapshot {
            frame_index: Some(input.frame_index),
            person_boxes: input.person_boxes.clone(),
            bed_boxes: input.bed_boxes.clone(),
            statuses: frame.statuses,
            events: frame.events.clone(),
            bed_region: Some(input.bed_region),
            stale: fresh.stale,
            observation_age_sec: fresh.observation_age_sec,
        });
        if !in_window {
            for event in frame.events {
                self.mark_suppressed(
                    event.person_id,
                    crate::trace::DecisionTraceReason::OutsideDetectionWindow,
                );
            }
            return Ok(());
        }
        self.phase = BedExitPhase::Expire;
        self.active_track = None;
        self.episodes
            .expire(input.frame_index, input.time_sec.unwrap_or(0.0))?;
        self.phase = BedExitPhase::Onset;
        for event in frame.events {
            self.active_track = Some(event.person_id);
            let proposal = self.proposal(input, event, true, false);
            if let Some(event) = self.episodes.propose(&proposal)? {
                emitted.push(event);
            } else if let Some(reason) =
                crate::episode::suppression_reason(self.episodes.last_disposition())
            {
                self.mark_suppressed(
                    event.person_id,
                    crate::trace::DecisionTraceReason::from_token(reason)
                        .expect("episode suppression vocabulary is shared"),
                );
            }
        }
        self.phase = BedExitPhase::TrackLost;
        for &track in &self.lost_track_ids {
            self.active_track = Some(track);
            self.episodes.track_lost(
                &self.config.camera_id,
                input.frame_index,
                input.time_sec.unwrap_or(0.0),
                Some(track),
            )?;
        }
        self.phase = BedExitPhase::Recovery;
        for &event in &self.recovery_events {
            self.active_track = Some(event.person_id);
            let proposal = self.proposal(input, event, false, true);
            // A nonqualifying recovery cannot create an onset in this authority.
            let recovery = self.episodes.propose(&proposal)?;
            if let Some(event) = recovery {
                emitted.push(event);
            }
        }
        Ok(())
    }
    fn proposal(
        &self,
        input: &BedExitInput,
        event: BedExitEvent,
        qualifying: bool,
        confirmed_recovery: bool,
    ) -> EpisodeProposal {
        EpisodeProposal {
            camera_id: self.config.camera_id.clone(),
            facility_id: self.config.facility_id.clone(),
            event_type: "bed-exit".into(),
            track_id: event.person_id,
            bed_id: Some(event.bed_id as u64),
            frame_index: input.frame_index,
            time_sec: input.time_sec.unwrap_or(0.0),
            qualifying,
            confirmed_recovery,
            probability: Some(1.0),
            domain: Some("bed_exit".into()),
            generation: 0,
            confirmation_votes: 1,
            confirmation_window: 1,
        }
    }
    fn mark_suppressed(&mut self, track_id: u64, reason: crate::trace::DecisionTraceReason) {
        for snapshot in &mut self.last_trace_snapshots {
            if snapshot.triggered && snapshot.track_id == Some(track_id) {
                snapshot.triggered = false;
                snapshot.reason = reason;
            }
        }
    }
    fn record_unavailable(&mut self, input: &BedExitInput) {
        use crate::trace::{
            DecisionTraceMissingReason as M, DecisionTraceReason as R, DecisionTraceValueName as V,
        };
        let (reason, missing) = if input.bed_region.source.usable() {
            (R::BedObservationMissing, M::BedObservationMissing)
        } else {
            (R::BedRegionUnavailable, M::BedRegionUnavailable)
        };
        self.last_trace_snapshots = vec![missing_trace(
            reason,
            &[(V::ContainmentRatio, missing), (V::BedId, missing)],
        )];
        // Matches Python's early-return defaults, NOT a freshness sample.
        self.last_debug_snapshot = Some(BedExitDebugSnapshot {
            frame_index: Some(input.frame_index),
            person_boxes: input.person_boxes.clone(),
            bed_boxes: Vec::new(),
            statuses: Vec::new(),
            events: Vec::new(),
            bed_region: Some(input.bed_region),
            stale: false,
            observation_age_sec: None,
        });
    }
}

/// BedExitLatch owns only this freshness state; it never owns episode latching.
#[derive(Debug)]
pub struct ObservationFreshness {
    stale_after_sec: f64,
    last_observed_at: Option<f64>,
}
impl ObservationFreshness {
    pub fn new(stale_after_sec: f64) -> Result<Self, BedExitFailure> {
        if !stale_after_sec.is_finite() || stale_after_sec <= 0.0 {
            return Err(BedExitFailure::InvalidConfig("stale_after_sec"));
        }
        Ok(Self {
            stale_after_sec,
            last_observed_at: None,
        })
    }
    pub fn observe(&mut self, now: f64) -> Result<(), BedExitFailure> {
        finite(now, "observation_clock")?;
        self.last_observed_at = Some(now);
        Ok(())
    }
    pub fn snapshot(&self, now: f64) -> Result<FreshnessSnapshot, BedExitFailure> {
        freshness_at(self.last_observed_at, now, self.stale_after_sec)
    }
}
fn finite(value: f64, field: &'static str) -> Result<(), BedExitFailure> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(BedExitFailure::NonFinite(field))
    }
}
fn freshness_at(
    last: Option<f64>,
    now: f64,
    threshold: f64,
) -> Result<FreshnessSnapshot, BedExitFailure> {
    finite(now, "snapshot_clock")?;
    let Some(last) = last else {
        return Ok(FreshnessSnapshot {
            stale: true,
            observation_age_sec: None,
        });
    };
    finite(last, "observation_clock")?;
    let age = (now - last).max(0.0);
    if !age.is_finite() {
        return Err(BedExitFailure::Overflow("observation_age_sec"));
    }
    Ok(FreshnessSnapshot {
        stale: age >= threshold,
        observation_age_sec: Some(age),
    })
}

mod admission {
    use super::*;
    pub(super) struct Prepared {
        pub ids: Vec<Option<u64>>,
        pub live: BTreeSet<u64>,
    }
    pub(super) fn config(
        config: &BedExitConfig,
        caps: BedExitCapacities,
    ) -> Result<(), BedExitFailure> {
        if config.camera_id.len() > crate::episode::MAX_AUTHORITY_ID_BYTES
            || config.facility_id.len() > crate::episode::MAX_AUTHORITY_ID_BYTES
        {
            return Err(BedExitFailure::InvalidIdentity);
        }
        if !config.min_containment.is_finite()
            || config.min_containment <= 0.0
            || config.min_containment > 1.0
        {
            return Err(BedExitFailure::InvalidConfig("min_containment"));
        }
        if config.hold_frames == 0 {
            return Err(BedExitFailure::InvalidConfig("hold_frames"));
        }
        for (name, value) in [
            ("in_bed_dwell_sec", config.in_bed_dwell_sec),
            ("outside_dwell_sec", config.outside_dwell_sec),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(BedExitFailure::InvalidConfig(name));
            }
        }
        if [
            caps.tracks,
            caps.observations,
            caps.beds,
            caps.pose_features,
            caps.polygon_points,
            caps.episodes,
        ]
        .contains(&0)
            || caps.tracks.checked_add(caps.observations).is_none()
            || caps.tracks.checked_mul(caps.observations).is_none()
            || caps.beds.checked_mul(caps.observations).is_none()
        {
            return Err(BedExitFailure::InvalidCapacities);
        }
        Ok(())
    }
    fn bound(length: usize, limit: usize, name: &'static str) -> Result<(), BedExitFailure> {
        if length > limit {
            Err(BedExitFailure::Capacity(name))
        } else {
            Ok(())
        }
    }
    fn box_value(value: &BoundingBox, caps: BedExitCapacities) -> Result<(), BedExitFailure> {
        finite(value.confidence, "box.confidence")?;
        if let Some(points) = &value.polygon {
            bound(points.len(), caps.polygon_points, "polygon_points")?;
        }
        Ok(())
    }
    pub(super) fn input(
        owner: &BedExitMonitor,
        input: &BedExitInput,
    ) -> Result<(), BedExitFailure> {
        let caps = owner.capacities;
        for (length, limit, name) in [
            (input.person_boxes.len(), caps.observations, "observations"),
            (input.track_ids.len(), caps.observations, "track_ids"),
            (input.bed_boxes.len(), caps.beds, "beds"),
            (input.live_track_ids.len(), caps.tracks, "live_track_ids"),
            (
                input.bed_pose_features.len(),
                caps.pose_features,
                "pose_features",
            ),
            (
                input.containments.len(),
                caps.observations,
                "containment_rows",
            ),
            (
                input.prior_box_containments.len(),
                caps.tracks,
                "prior_box_rows",
            ),
        ] {
            bound(length, limit, name)?;
        }
        if let Some(time) = input.time_sec {
            finite(time, "time_sec")?;
        }
        for value in input.person_boxes.iter().chain(&input.bed_boxes) {
            box_value(value, caps)?;
        }
        for features in &input.bed_pose_features {
            for value in [
                features.torso_in_frac,
                features.lower_in_frac,
                features.keypoint_in_frac,
                features.hip_depth,
                features.torso_angle,
                features.centroid_displacement,
                features.hip_x_rel,
                features.hip_y_rel,
                features.observability,
            ] {
                finite(value, "bed_pose_features")?;
            }
        }
        for ratios in &input.containments {
            bound(ratios.len(), caps.beds, "containment_columns")?;
            for &ratio in ratios {
                finite(ratio, "containment_ratio")?;
            }
        }
        for row in &input.prior_box_containments {
            box_value(&row.previous_box, caps)?;
            bound(row.ratios.len(), caps.observations, "overlap_columns")?;
            for &ratio in &row.ratios {
                finite(ratio, "prior_box_containment")?;
            }
        }
        Ok(())
    }
    pub(super) fn prepare(
        owner: &BedExitMonitor,
        input: &BedExitInput,
    ) -> Result<Prepared, BedExitFailure> {
        let count = input.person_boxes.len();
        if !input.track_ids.is_empty() && input.track_ids.len() != count {
            return Err(BedExitFailure::InvalidShape("track_ids"));
        }
        if input.containments.len() != count
            || input
                .containments
                .iter()
                .any(|row| row.len() != input.bed_boxes.len())
        {
            return Err(BedExitFailure::InvalidShape("containments"));
        }
        let (ids, live) = if input.track_ids.is_empty() {
            (
                (0..count).map(|id| Some(id as u64)).collect(),
                (0..count).map(|id| id as u64).collect(),
            )
        } else {
            (
                input.track_ids.clone(),
                input.live_track_ids.iter().copied().collect(),
            )
        };
        let prepared = Prepared { ids, live };
        bound(
            prepared.live.len(),
            owner.capacities.tracks,
            "live_track_ids",
        )?;
        // Stale rows are removed before admitting observations. Every retained
        // or new assignment is in this bounded live set; no second set needed.
        let mut rows = BTreeSet::new();
        for row in &input.prior_box_containments {
            if !rows.insert(row.previous_track_id) || row.ratios.len() != count {
                return Err(BedExitFailure::InvalidShape("prior_box_containments"));
            }
            if owner
                .assignments
                .get(&row.previous_track_id)
                .and_then(|a| a.last_box.as_ref())
                != Some(&row.previous_box)
            {
                return Err(BedExitFailure::MismatchedPriorBox(row.previous_track_id));
            }
        }
        let has_unclaimed = prepared
            .ids
            .iter()
            .flatten()
            .any(|id| prepared.live.contains(id) && !owner.assignments.contains_key(id));
        if has_unclaimed {
            for (&track, assignment) in &owner.assignments {
                if !prepared.live.contains(&track)
                    && assignment.armed
                    && assignment.outside_dwell_sec > 0.0
                    && assignment
                        .bed_id
                        .is_some_and(|bed| bed < input.bed_boxes.len())
                    && assignment.last_box.is_some()
                    && !rows.contains(&track)
                {
                    return Err(BedExitFailure::MissingPriorOverlap(track));
                }
            }
        }
        Ok(prepared)
    }
}

mod evidence {
    use super::*;
    use crate::trace::{
        DecisionTraceMissingReason as M, DecisionTraceReason as R, DecisionTraceState as S,
        DecisionTraceValueName as V, NumericTraceValue as N, TraceFloat,
    };
    pub(super) fn armed_state(armed: bool) -> S {
        if armed { S::Armed } else { S::Arming }
    }
    pub(super) fn trace(
        reason: R,
        states: (S, S),
        triggered: bool,
        subject: (Option<u64>, Option<usize>),
        floats: &[(V, f64)],
        integers: &[(V, usize)],
        missing: &[(V, M)],
    ) -> Result<DecisionTraceSnapshot, BedExitFailure> {
        let (track, bed) = subject;
        let mut values = BTreeMap::new();
        for &(name, value) in floats {
            values.insert(
                name,
                N::Float(TraceFloat::new(value).map_err(|_| BedExitFailure::NonFinite("trace"))?),
            );
        }
        for &(name, value) in integers {
            values.insert(name, N::Integer(value));
        }
        Ok(DecisionTraceSnapshot::new(
            reason,
            states,
            triggered,
            track,
            bed.map(|id| id as u64),
            values,
            missing.iter().copied().collect(),
        )
        .expect("static bed trace fields are disjoint"))
    }
    pub(super) fn missing_trace(reason: R, missing: &[(V, M)]) -> DecisionTraceSnapshot {
        trace(
            reason,
            (S::Unknown, S::NoDecision),
            false,
            (None, None),
            &[],
            &[],
            missing,
        )
        .expect("missing-only trace is finite")
    }
}

mod handoff {
    use super::*;
    use crate::trace::{
        DecisionTraceMissingReason as M, DecisionTraceReason as R, DecisionTraceState as S,
        DecisionTraceValueName as V,
    };
    impl BedExitMonitor {
        pub(super) fn retire_stale(
            &mut self,
            input: &BedExitInput,
            prepared: &admission::Prepared,
            poses: &BTreeMap<u64, BedPoseFeatures>,
        ) -> Result<(), BedExitFailure> {
            // Keep observation order, including duplicates. Containment ties pick
            // the LAST eligible row, unlike best-bed ties which pick lowest bed.
            let mut unclaimed: Vec<(usize, u64)> = prepared
                .ids
                .iter()
                .enumerate()
                .filter_map(|(index, id)| {
                    id.filter(|id| prepared.live.contains(id) && !self.assignments.contains_key(id))
                        .map(|id| (index, id))
                })
                .collect();
            let stale: Vec<u64> = self
                .assignments
                .keys()
                .copied()
                .filter(|id| !prepared.live.contains(id))
                .collect();
            for stale_id in stale {
                self.active_track = Some(stale_id);
                let assignment = self
                    .assignments
                    .get(&stale_id)
                    .expect("stale assignment exists");
                let inside = assignment.bed_id.and_then(|bed| {
                    inside_recipient(input, bed, &unclaimed, poses, self.config.min_containment)
                });
                let outside =
                    if inside.is_none() && assignment.armed && assignment.outside_dwell_sec > 0.0 {
                        assignment.bed_id.and_then(|bed| {
                            outside_recipient(
                                input,
                                bed,
                                stale_id,
                                assignment,
                                &unclaimed,
                                self.config.min_containment,
                            )
                        })
                    } else {
                        None
                    };
                if let Some(recipient) = inside.or(outside) {
                    let bed = assignment.bed_id.expect("handoff requires an assigned bed");
                    let successor = AssignmentSnapshot {
                        bed_id: Some(bed),
                        candidate_bed_id: Some(bed),
                        candidate_frames: self.config.hold_frames,
                        in_bed_dwell_sec: if inside.is_some() {
                            assignment.in_bed_dwell_sec
                        } else {
                            0.0
                        },
                        outside_dwell_sec: if inside.is_some() {
                            0.0
                        } else {
                            assignment.outside_dwell_sec
                        },
                        armed: assignment.armed,
                        last_time_sec: assignment.last_time_sec,
                        last_box: None,
                    };
                    let state = armed_state(successor.armed);
                    let snapshot = trace(
                        R::IdentityHandoff,
                        (state, state),
                        false,
                        (Some(recipient), Some(bed)),
                        &[
                            (V::InBedDwellSec, successor.in_bed_dwell_sec),
                            (V::OutsideDwellSec, successor.outside_dwell_sec),
                        ],
                        &[],
                        &[],
                    )?;
                    // Remove first so transient state also respects the track budget.
                    self.assignments.remove(&stale_id);
                    self.assignments.insert(recipient, successor);
                    unclaimed.retain(|&(_, id)| id != recipient);
                    let proposal = self.proposal(
                        input,
                        BedExitEvent {
                            person_id: recipient,
                            bed_id: bed,
                        },
                        false,
                        false,
                    );
                    self.episodes.reassociate_bed_exit(&proposal)?;
                    self.last_trace_snapshots.push(snapshot);
                } else {
                    self.last_trace_snapshots.push(trace(
                        R::StaleTrackClear,
                        (armed_state(assignment.armed), S::Retired),
                        false,
                        (Some(stale_id), assignment.bed_id),
                        &[
                            (V::InBedDwellSec, assignment.in_bed_dwell_sec),
                            (V::OutsideDwellSec, assignment.outside_dwell_sec),
                        ],
                        &[],
                        &[(V::ContainmentRatio, M::TrackNoLongerLive)],
                    )?);
                    if assignment.bed_id.is_some() {
                        self.lost_track_ids.push(stale_id);
                    }
                    self.assignments.remove(&stale_id);
                }
            }
            Ok(())
        }
    }
    fn inside_recipient(
        input: &BedExitInput,
        bed: usize,
        unclaimed: &[(usize, u64)],
        poses: &BTreeMap<u64, BedPoseFeatures>,
        threshold: f64,
    ) -> Option<u64> {
        if bed >= input.bed_boxes.len() {
            return None;
        }
        let mut best = None;
        let mut ratio = threshold;
        for &(index, id) in unclaimed {
            if poses.get(&id).is_some_and(|pose| pose.confirms_in_bed())
                && input.containments[index][bed] >= ratio
            {
                best = Some(id);
                ratio = input.containments[index][bed];
            }
        }
        best
    }
    fn outside_recipient(
        input: &BedExitInput,
        bed: usize,
        stale: u64,
        assignment: &AssignmentSnapshot,
        unclaimed: &[(usize, u64)],
        threshold: f64,
    ) -> Option<u64> {
        if assignment.last_box.is_none() || bed >= input.bed_boxes.len() || unclaimed.is_empty() {
            return None;
        }
        if unclaimed
            .iter()
            .any(|&(index, _)| input.containments[index][bed] >= threshold)
        {
            return None;
        }
        let overlaps = &input
            .prior_box_containments
            .iter()
            .find(|row| row.previous_track_id == stale)
            .expect("admission requires exact prior-box evidence")
            .ratios;
        let mut candidates = unclaimed.iter().filter(|&&(index, _)| {
            overlaps[index] > 0.0
                && !input.containments[index]
                    .iter()
                    .any(|&ratio| ratio >= threshold)
        });
        let first = candidates.next().map(|&(_, id)| id);
        if candidates.next().is_none() {
            first
        } else {
            None
        }
    }
}

mod decisions {
    use super::*;
    use crate::trace::{
        DecisionTraceMissingReason as M, DecisionTraceReason as R, DecisionTraceState as S,
        DecisionTraceValueName as V,
    };
    impl BedExitMonitor {
        pub(super) fn update_frame(
            &mut self,
            input: &BedExitInput,
            prepared: admission::Prepared,
        ) -> Result<BedExitFrame, BedExitFailure> {
            let poses = input
                .bed_pose_features
                .iter()
                .map(|pose| (pose.track_id, *pose))
                .collect();
            self.retire_stale(input, &prepared, &poses)?;
            let mut occupied = BTreeMap::new();
            let mut exit_beds = BTreeSet::new();
            let mut events = Vec::new();
            for (index, id) in prepared.ids.into_iter().enumerate() {
                let Some(id) = id.filter(|id| prepared.live.contains(id)) else {
                    continue;
                };
                self.active_track = Some(id);
                let assignment = self.assignments.entry(id).or_default();
                assignment.last_box = Some(input.person_boxes[index].clone());
                let ratios = &input.containments[index];
                let maximum = ratios
                    .iter()
                    .copied()
                    .reduce(f64::max)
                    .expect("usable region has beds");
                self.scoring.max_containment_observed =
                    self.scoring.max_containment_observed.max(maximum);
                if assignment.bed_id.is_none() {
                    let candidate = best_bed(ratios, self.config.min_containment);
                    update_candidate(assignment, candidate)?;
                    if assignment.candidate_frames >= self.config.hold_frames {
                        assignment.bed_id = assignment.candidate_bed_id;
                        assignment.armed = false;
                        assignment.in_bed_dwell_sec = 0.0;
                        assignment.outside_dwell_sec = 0.0;
                        assignment.last_time_sec = input.time_sec;
                        self.scoring.assignments_made =
                            increment(self.scoring.assignments_made, "assignments_made")?;
                    }
                    let bed = assignment.bed_id;
                    let snapshot = trace(
                        if bed.is_some() {
                            R::Assigned
                        } else if candidate.is_some() {
                            R::AssignmentHold
                        } else {
                            R::BelowContainment
                        },
                        (
                            S::Unassigned,
                            if bed.is_some() {
                                S::Contained
                            } else {
                                S::Unassigned
                            },
                        ),
                        false,
                        (Some(id), bed.or(candidate)),
                        &[
                            (V::ContainmentRatio, maximum),
                            (V::MinContainment, self.config.min_containment),
                        ],
                        &[
                            (V::CandidateFrames, assignment.candidate_frames),
                            (V::HoldFramesThreshold, self.config.hold_frames),
                        ],
                        &[],
                    )?;
                    if let Some(bed) = bed {
                        occupied.insert(bed, id);
                        let proposal = self.proposal(
                            input,
                            BedExitEvent {
                                person_id: id,
                                bed_id: bed,
                            },
                            false,
                            false,
                        );
                        self.episodes.reassociate_bed_exit(&proposal)?;
                    }
                    self.last_trace_snapshots.push(snapshot);
                    continue;
                }
                let bed = assignment.bed_id.expect("assigned branch");
                let dt = match input.time_sec {
                    None => 0.0,
                    Some(now) => {
                        let dt = assignment
                            .last_time_sec
                            .map_or(0.0, |last| (now - last).max(0.0));
                        assignment.last_time_sec = Some(now);
                        dt
                    }
                };
                let features = poses.get(&id);
                let confirms = features.is_some_and(|pose| pose.confirms_in_bed());
                let own_ratio = ratios.get(bed).copied().unwrap_or(0.0);
                if own_ratio >= self.config.min_containment {
                    assignment.outside_dwell_sec = 0.0;
                    assignment.in_bed_dwell_sec = if confirms {
                        add_dwell(assignment.in_bed_dwell_sec, dt, "in_bed_dwell_sec")?
                    } else {
                        0.0
                    };
                    if !assignment.armed
                        && assignment.in_bed_dwell_sec >= self.config.in_bed_dwell_sec
                    {
                        assignment.armed = true;
                        self.scoring.grace_positive_transitions = increment(
                            self.scoring.grace_positive_transitions,
                            "grace_positive_transitions",
                        )?;
                        self.recovery_events.push(BedExitEvent {
                            person_id: id,
                            bed_id: bed,
                        });
                    }
                    occupied.insert(bed, id);
                    let mut missing = Vec::new();
                    if features.is_none() {
                        missing.push((V::HipDepth, M::NoPoseEvidence));
                    }
                    if input.time_sec.is_none() {
                        missing.push((V::TimeSec, M::TimeNotProvided));
                    }
                    let state = armed_state(assignment.armed); // Source traces the POST-arm state on both sides.
                    self.last_trace_snapshots.push(trace(
                        if confirms {
                            R::Contained
                        } else {
                            R::ContainedPostureUnconfirmed
                        },
                        (state, state),
                        false,
                        (Some(id), Some(bed)),
                        &[
                            (V::ContainmentRatio, own_ratio),
                            (V::MinContainment, self.config.min_containment),
                            (V::InBedDwellSec, assignment.in_bed_dwell_sec),
                            (V::InBedDwellThresholdSec, self.config.in_bed_dwell_sec),
                        ],
                        &[],
                        &missing,
                    )?);
                } else if ratios
                    .iter()
                    .enumerate()
                    .any(|(other, &ratio)| other != bed && ratio >= self.config.min_containment)
                {
                    assignment.outside_dwell_sec = 0.0;
                    assignment.in_bed_dwell_sec = 0.0;
                    self.last_trace_snapshots.push(trace(
                        R::ContainedInOtherBed,
                        (armed_state(assignment.armed), S::OtherBed),
                        false,
                        (Some(id), Some(bed)),
                        &[
                            (V::ContainmentRatio, own_ratio),
                            (
                                V::MaxOtherContainmentRatio,
                                ratios
                                    .iter()
                                    .enumerate()
                                    .filter_map(|(other, &ratio)| (other != bed).then_some(ratio))
                                    .reduce(f64::max)
                                    .expect("other bed exists"),
                            ),
                            (V::MinContainment, self.config.min_containment),
                        ],
                        &[],
                        &[],
                    )?);
                } else {
                    assignment.outside_dwell_sec =
                        add_dwell(assignment.outside_dwell_sec, dt, "outside_dwell_sec")?;
                    let triggered = assignment.armed
                        && assignment.outside_dwell_sec >= self.config.outside_dwell_sec;
                    let state = armed_state(assignment.armed);
                    self.last_trace_snapshots.push(trace(
                        if triggered {
                            R::OutsideDwellExit
                        } else if assignment.armed {
                            R::OutsideDwell
                        } else {
                            R::OutsideNotArmed
                        },
                        (state, if triggered { S::Triggered } else { state }),
                        triggered,
                        (Some(id), Some(bed)),
                        &[
                            (V::ContainmentRatio, own_ratio),
                            (V::MinContainment, self.config.min_containment),
                            (V::OutsideDwellSec, assignment.outside_dwell_sec),
                            (V::OutsideDwellThresholdSec, self.config.outside_dwell_sec),
                        ],
                        &[],
                        &[],
                    )?);
                    // Deliberately no in-bed reset on a mere outside dip: this is
                    // what the current Python owner does, despite its dwell prose.
                    if triggered {
                        events.push(BedExitEvent {
                            person_id: id,
                            bed_id: bed,
                        });
                        exit_beds.insert(bed);
                        assignment.armed = false;
                        assignment.in_bed_dwell_sec = 0.0;
                        assignment.outside_dwell_sec = 0.0;
                    }
                }
            }
            let statuses = input
                .bed_boxes
                .iter()
                .enumerate()
                .map(|(bed, box_value)| BedStatus {
                    bed_id: bed,
                    box_value: box_value.clone(),
                    person_id: occupied.get(&bed).copied(),
                    occupancy: if exit_beds.contains(&bed) {
                        BedOccupancy::Exit
                    } else if occupied.contains_key(&bed) {
                        BedOccupancy::Occupied
                    } else {
                        BedOccupancy::Empty
                    },
                })
                .collect();
            if self.last_trace_snapshots.is_empty() {
                self.last_trace_snapshots.push(missing_trace(
                    R::PersonObservationMissing,
                    &[(V::ContainmentRatio, M::NoObservedPerson)],
                ));
            }
            Ok(BedExitFrame { statuses, events })
        }
    }
    fn best_bed(ratios: &[f64], threshold: f64) -> Option<usize> {
        let mut best = None;
        for (bed, &ratio) in ratios.iter().enumerate() {
            if ratio >= threshold && best.is_none_or(|previous| ratio > ratios[previous]) {
                best = Some(bed);
            }
        }
        best
    }
    fn update_candidate(
        assignment: &mut AssignmentSnapshot,
        bed: Option<usize>,
    ) -> Result<(), BedExitFailure> {
        match bed {
            None => {
                assignment.candidate_bed_id = None;
                assignment.candidate_frames = 0;
            }
            Some(_) if assignment.candidate_bed_id == bed => {
                assignment.candidate_frames = assignment
                    .candidate_frames
                    .checked_add(1)
                    .ok_or(BedExitFailure::Overflow("candidate_frames"))?;
            }
            Some(_) => {
                assignment.candidate_bed_id = bed;
                assignment.candidate_frames = 1;
            }
        }
        Ok(())
    }
    fn increment(value: u64, field: &'static str) -> Result<u64, BedExitFailure> {
        value.checked_add(1).ok_or(BedExitFailure::Overflow(field))
    }
    fn add_dwell(value: f64, dt: f64, field: &'static str) -> Result<f64, BedExitFailure> {
        let total = value + dt;
        if total.is_finite() {
            Ok(total)
        } else {
            Err(BedExitFailure::Overflow(field))
        }
    }
}

#[cfg(test)]
mod fixtures {
    use super::*;
    use crate::detection_window::{ClockRelation, DateTime};
    use crate::trace::{DecisionTraceValueName, NumericTraceValue};

    pub(super) fn config() -> BedExitConfig {
        BedExitConfig {
            camera_id: "camera-bed-exit".into(),
            facility_id: "facility-bed-exit".into(),
            min_containment: 0.5,
            hold_frames: 1,
            grace_frames: 0,
            night_window: None,
            in_bed_dwell_sec: 1.0,
            outside_dwell_sec: 1.0,
        }
    }
    pub(super) fn caps() -> BedExitCapacities {
        BedExitCapacities {
            tracks: 8,
            observations: 8,
            beds: 4,
            pose_features: 8,
            polygon_points: 16,
            episodes: 32,
        }
    }
    pub(super) fn monitor() -> BedExitMonitor {
        custom(config(), caps())
    }
    pub(super) fn custom(config: BedExitConfig, capacities: BedExitCapacities) -> BedExitMonitor {
        BedExitMonitor::new(config, "test-boot", "test-epoch", 0, capacities, 3.0).unwrap()
    }
    pub(super) fn bbox(x1: i64, y1: i64, x2: i64, y2: i64, confidence: f64) -> BoundingBox {
        BoundingBox {
            x1,
            y1,
            x2,
            y2,
            confidence,
            polygon: None,
        }
    }
    // Exact facts from tests_support/bed_pose_fixtures.py, not a geometry producer.
    pub(super) fn lying(track_id: u64) -> BedPoseFeatures {
        BedPoseFeatures {
            track_id,
            bed_id: Some(0),
            torso_in_frac: 1.0,
            lower_in_frac: 1.0,
            keypoint_in_frac: 1.0,
            hip_depth: 0.257,
            torso_angle: 1.4,
            centroid_displacement: 0.0,
            hip_x_rel: 0.5,
            hip_y_rel: 0.5,
            observability: 0.9,
            bed_polygon_valid: true,
        }
    }
    pub(super) fn frame(time: f64, ids: &[u64], ratios: &[&[f64]], poses: &[u64]) -> BedExitInput {
        assert_eq!(ids.len(), ratios.len());
        let beds = ratios.first().map_or(1, |row| row.len());
        BedExitInput {
            // IN_BED and OUTSIDE_BED from test_worker_domains_bed_exit_assignment.py.
            person_boxes: ratios
                .iter()
                .map(|row| {
                    if row.iter().any(|&ratio| ratio >= 0.5) {
                        bbox(10, 10, 70, 90, 0.95)
                    } else {
                        bbox(100, 10, 160, 90, 0.94)
                    }
                })
                .collect(),
            bed_boxes: (0..beds)
                .map(|bed| {
                    if bed == 0 {
                        bbox(0, 0, 80, 100, 0.99)
                    } else {
                        bbox(200, 10, 260, 90, 0.94)
                    }
                })
                .collect(),
            track_ids: ids.iter().copied().map(Some).collect(),
            live_track_ids: ids.to_vec(),
            bed_pose_features: poses.iter().copied().map(lying).collect(),
            containments: ratios.iter().map(|row| row.to_vec()).collect(),
            prior_box_containments: Vec::new(),
            time_sec: Some(time),
            frame_index: time as i64,
            bed_region: BedRegionDebugSnapshot {
                source: BedRegionCacheState::Fresh,
                empty_cycles: 0,
            },
        }
    }
    pub(super) fn clocks(time: f64, hour: i8) -> BedExitClocks {
        BedExitClocks {
            wall_time: Some(
                AwareDateTime::new(
                    DateTime::new(2026, 7, 31, hour, 0, 0, 0).unwrap(),
                    Some(9 * 3600),
                    ClockRelation::DifferentTzinfo,
                )
                .unwrap(),
            ),
            observed_at: time,
            snapshot_at: time,
        }
    }
    pub(super) fn send(owner: &mut BedExitMonitor, input: BedExitInput) -> BedExitOutcome {
        owner
            .update(&input, clocks(input.time_sec.unwrap_or(0.0), 22))
            .unwrap()
    }
    pub(super) fn one(time: f64, ratio: f64, pose: bool) -> BedExitInput {
        frame(time, &[7], &[&[ratio]], if pose { &[7] } else { &[] })
    }
    pub(super) fn arm(owner: &mut BedExitMonitor) {
        assert!(send(owner, one(0.0, 1.0, false)).events.is_empty());
        assert!(send(owner, one(1.0, 1.0, true)).events.is_empty());
        assert!(owner.assignments()[&7].armed);
    }
    pub(super) fn overlap(
        owner: &BedExitMonitor,
        input: &mut BedExitInput,
        track: u64,
        ratios: &[f64],
    ) {
        input.prior_box_containments.push(PriorBoxContainments {
            previous_track_id: track,
            previous_box: owner.assignments()[&track].last_box.clone().unwrap(),
            ratios: ratios.to_vec(),
        });
    }
    pub(super) fn window() -> DetectionWindow {
        let root = std::env::var_os("SEEON_TEST_ZONEINFO_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from("/usr/share/zoneinfo"));
        DetectionWindow::from_zoneinfo_dir("21:00", "05:00", "Asia/Seoul", &root).unwrap()
    }
    pub(super) fn float(snapshot: &DecisionTraceSnapshot, name: DecisionTraceValueName) -> f64 {
        match snapshot.values()[&name] {
            NumericTraceValue::Float(value) => value.get(),
            NumericTraceValue::Integer(_) => panic!("expected a floating-point trace"),
        }
    }
}

#[cfg(test)]
mod assignment_tests {
    use super::fixtures::*;
    use super::*;
    use crate::trace::{
        DecisionTraceReason as R, DecisionTraceState as S, DecisionTraceValueName as V,
    };

    #[test]
    fn hold_resets_on_candidate_switch_or_missing_and_ties_choose_lowest_bed() {
        let mut owner = custom(
            BedExitConfig {
                hold_frames: 2,
                ..config()
            },
            caps(),
        );
        for (time, ratios, bed, count) in [
            (0.0, [0.5, 0.5], Some(0), 1),
            (1.0, [0.499999, 0.5], Some(1), 1),
            (2.0, [0.0, 0.0], None, 0),
            (3.0, [0.5, 0.5], Some(0), 1),
            (4.0, [0.5, 0.5], Some(0), 2),
        ] {
            assert!(
                send(&mut owner, frame(time, &[7], &[&ratios], &[7]))
                    .events
                    .is_empty()
            );
            let assignment = &owner.assignments()[&7];
            assert_eq!(
                (assignment.candidate_bed_id, assignment.candidate_frames),
                (bed, count)
            );
            assert_eq!(assignment.bed_id, if time == 4.0 { Some(0) } else { None });
        }
        let statuses = &owner.last_debug_snapshot().unwrap().statuses;
        assert_eq!(
            (statuses[0].occupancy, statuses[1].occupancy),
            (BedOccupancy::Occupied, BedOccupancy::Empty)
        );
        assert_eq!(owner.last_trace_snapshots()[0].reason, R::Assigned);
        assert_eq!(owner.assignments()[&7].in_bed_dwell_sec, 0.0);
        assert!(!owner.assignments()[&7].armed); // Assignment frame never earns dwell.
    }

    #[test]
    fn every_posture_gate_is_inclusive_and_never_uses_torso_angle_or_feature_bed_id() {
        for (valid, observability, hip, expected) in [
            (true, 0.35, 0.10, true),
            (true, 0.349999, 0.257, false),
            (true, 0.9, 0.099999, false),
            (true, 0.9, 0.043, false),
            (true, 0.9, -0.289, false),
            (false, 0.9, 0.257, false),
        ] {
            let mut owner = monitor();
            send(&mut owner, one(0.0, 1.0, true));
            let mut input = one(1.0, 0.5, true);
            let pose = &mut input.bed_pose_features[0];
            pose.bed_polygon_valid = valid;
            pose.observability = observability;
            pose.hip_depth = hip;
            pose.bed_id = Some(999);
            pose.torso_angle = -100.0;
            send(&mut owner, input);
            assert_eq!(owner.assignments()[&7].armed, expected);
            let snapshot = &owner.last_trace_snapshots()[0];
            assert_eq!(
                snapshot.reason,
                if expected {
                    R::Contained
                } else {
                    R::ContainedPostureUnconfirmed
                }
            );
            assert_eq!(
                snapshot.previous_state,
                if expected { S::Armed } else { S::Arming }
            );
            assert!(snapshot.missing_values().is_empty()); // Invalid pose != absent pose.
            assert_eq!(
                send(&mut owner, one(2.0, 0.0, false)).events.len(),
                usize::from(expected)
            );
        }
    }

    #[test]
    fn standing_for_ten_seconds_never_arms_and_missing_pose_is_explicit() {
        let mut owner = custom(
            BedExitConfig {
                in_bed_dwell_sec: 3.0,
                outside_dwell_sec: 2.0,
                ..config()
            },
            caps(),
        );
        send(&mut owner, one(0.0, 1.0, false));
        for t in 1..=10 {
            let mut input = one(f64::from(t), 1.0, true);
            input.bed_pose_features[0].hip_depth = -0.289;
            assert!(send(&mut owner, input).events.is_empty());
        }
        assert!(send(&mut owner, one(13.0, 0.0, false)).events.is_empty());
        assert_eq!(owner.last_trace_snapshots()[0].reason, R::OutsideNotArmed);
        send(&mut owner, one(14.0, 1.0, false));
        assert_eq!(
            owner.last_trace_snapshots()[0].missing_values()[&V::HipDepth].as_str(),
            "no-pose-evidence"
        );
        assert_eq!(owner.scoring().grace_positive_transitions, 0);
    }

    #[test]
    fn posture_resets_climb_but_outside_dip_does_not_and_armed_latch_survives_both() {
        let mut owner = monitor();
        send(&mut owner, one(0.0, 1.0, false));
        send(&mut owner, one(0.5, 1.0, true));
        send(&mut owner, one(0.75, 0.0, false));
        assert_eq!(owner.assignments()[&7].in_bed_dwell_sec, 0.5);
        send(&mut owner, one(1.0, 1.0, true));
        assert_eq!(owner.assignments()[&7].in_bed_dwell_sec, 0.75);
        send(&mut owner, one(1.25, 1.0, false));
        assert_eq!(owner.assignments()[&7].in_bed_dwell_sec, 0.0);
        send(&mut owner, one(2.25, 1.0, true));
        send(&mut owner, one(2.5, 1.0, false));
        assert!(owner.assignments()[&7].armed);
        send(&mut owner, one(3.0, 0.0, false));
        assert_eq!(owner.assignments()[&7].outside_dwell_sec, 0.5);
        send(&mut owner, one(3.25, 1.0, false));
        assert_eq!(owner.assignments()[&7].outside_dwell_sec, 0.0);
        assert_eq!(send(&mut owner, one(4.25, 0.0, false)).events.len(), 1);
    }

    #[test]
    fn other_bed_is_neutral_sticky_and_not_occupied_in_status() {
        let mut owner = monitor();
        for time in [0.0, 1.0] {
            send(&mut owner, frame(time, &[7], &[&[1.0, 0.0]], &[7]));
        }
        send(&mut owner, frame(2.0, &[7], &[&[0.0, 0.5]], &[7]));
        let assignment = &owner.assignments()[&7];
        assert_eq!(assignment.bed_id, Some(0));
        assert!(assignment.armed);
        assert_eq!(
            (assignment.in_bed_dwell_sec, assignment.outside_dwell_sec),
            (0.0, 0.0)
        );
        let snapshot = &owner.last_trace_snapshots()[0];
        assert_eq!(snapshot.reason, R::ContainedInOtherBed);
        assert_eq!(float(snapshot, V::MaxOtherContainmentRatio), 0.5);
        assert!(
            owner
                .last_debug_snapshot()
                .unwrap()
                .statuses
                .iter()
                .all(|s| s.occupancy == BedOccupancy::Empty)
        );
        assert_eq!(
            send(&mut owner, frame(3.0, &[7], &[&[0.0, 0.0]], &[]))
                .events
                .len(),
            1
        );
    }

    #[test]
    fn removed_own_bed_uses_zero_ratio_but_keeps_original_event_identity() {
        let mut owner = monitor();
        for time in [0.0, 1.0] {
            send(&mut owner, frame(time, &[7], &[&[0.0, 1.0]], &[7]));
        }
        let output = send(&mut owner, one(2.0, 0.0, false));
        assert_eq!(output.events[0].bed_id, Some(1));
        assert_eq!(
            owner.last_debug_snapshot().unwrap().statuses[0].occupancy,
            BedOccupancy::Empty
        );
        assert_eq!(
            float(&owner.last_trace_snapshots()[0], V::ContainmentRatio),
            0.0
        );
    }

    #[test]
    fn scoring_is_cumulative_last_duplicate_pose_wins_and_numeric_domain_is_not_clipped() {
        let mut owner = monitor();
        send(&mut owner, one(0.0, -2.0, false));
        assert_eq!(owner.scoring().max_containment_observed, 0.0);
        send(&mut owner, one(1.0, 2.5, false));
        let mut input = one(2.0, 0.75, true);
        input.bed_pose_features.push(BedPoseFeatures {
            hip_depth: -0.289,
            ..lying(7)
        });
        send(&mut owner, input);
        assert!(!owner.assignments()[&7].armed);
        let output = send(&mut owner, one(3.0, 0.75, true));
        assert_eq!(
            output.scoring_observation,
            Some(BedExitScoring {
                max_containment_observed: 2.5,
                assignments_made: 1,
                grace_positive_transitions: 1,
            })
        );
        assert_eq!(
            owner.last_recovery_events(),
            &[BedExitEvent {
                person_id: 7,
                bed_id: 0
            }]
        );
    }
}

#[cfg(test)]
mod time_tests {
    use super::fixtures::*;
    use super::*;
    use crate::trace::{DecisionTraceReason as R, DecisionTraceValueName as V};

    #[test]
    fn missing_pts_keeps_anchor_and_next_real_pts_spans_the_gap() {
        let mut owner = custom(
            BedExitConfig {
                outside_dwell_sec: 3.0,
                ..config()
            },
            caps(),
        );
        arm(&mut owner);
        send(&mut owner, one(2.0, 0.0, false));
        let mut missing = one(3.0, 0.0, false);
        missing.time_sec = None;
        assert!(send(&mut owner, missing).events.is_empty());
        assert_eq!(owner.assignments()[&7].last_time_sec, Some(2.0));
        assert_eq!(owner.assignments()[&7].outside_dwell_sec, 1.0);
        // The source names missing time only on a contained row, not outside.
        assert!(owner.last_trace_snapshots()[0].missing_values().is_empty());
        let output = send(&mut owner, one(4.0, 0.0, false));
        assert_eq!(output.events[0].time_sec, 4.0);
        assert_eq!(
            float(&owner.last_trace_snapshots()[0], V::OutsideDwellSec),
            3.0
        );
    }

    #[test]
    fn first_missing_time_and_reversed_pts_never_fabricate_elapsed_seconds() {
        let mut owner = monitor();
        let mut missing = one(100.0, 1.0, true);
        missing.time_sec = None;
        send(&mut owner, missing.clone());
        send(&mut owner, missing);
        assert_eq!(owner.assignments()[&7].last_time_sec, None);
        assert_eq!(
            owner.last_trace_snapshots()[0].missing_values()[&V::TimeSec].as_str(),
            "time-not-provided"
        );
        send(&mut owner, one(100.0, 1.0, true));
        assert_eq!(owner.assignments()[&7].in_bed_dwell_sec, 0.0);
        send(&mut owner, one(-10.0, 1.0, true));
        assert_eq!(owner.assignments()[&7].last_time_sec, Some(-10.0));
        assert!(!owner.assignments()[&7].armed);
        send(&mut owner, one(-9.0, 1.0, true));
        assert!(owner.assignments()[&7].armed);
        let event = send(&mut owner, one(-8.0, 0.0, false)).events.remove(0);
        assert_eq!(event.time_sec, -8.0);
    }

    #[test]
    fn freshness_first_observation_exact_staleness_threshold_and_clock_reversal() {
        let mut freshness = ObservationFreshness::new(3.0).unwrap();
        assert_eq!(
            freshness.snapshot(10.0).unwrap(),
            FreshnessSnapshot {
                stale: true,
                observation_age_sec: None
            }
        );
        freshness.observe(10.0).unwrap();
        for (time, age, stale) in [
            (12.999999, 2.999999, false),
            (13.0, 3.0, true),
            (9.0, 0.0, false),
        ] {
            let snapshot = freshness.snapshot(time).unwrap();
            assert_eq!(snapshot.stale, stale);
            assert!((snapshot.observation_age_sec.unwrap() - age).abs() < 1e-12);
        }
        let mut owner = monitor();
        owner.coast(None, 0.0).unwrap();
        assert!(owner.last_debug_snapshot().unwrap().stale);
        assert_eq!(owner.last_debug_snapshot().unwrap().bed_region, None);
        // Independent observe/snapshot samples, with a gap crossing the limit.
        owner
            .update(
                &one(0.0, 1.0, false),
                BedExitClocks {
                    observed_at: 0.0,
                    snapshot_at: 3.0,
                    wall_time: None,
                },
            )
            .unwrap();
        assert!(owner.last_debug_snapshot().unwrap().stale);
        let traces = owner.last_trace_snapshots().to_vec();
        owner.coast(Some(99), 2.0).unwrap();
        let snapshot = owner.last_debug_snapshot().unwrap();
        assert_eq!(snapshot.statuses[0].occupancy, BedOccupancy::Covered);
        assert_eq!(snapshot.statuses[0].person_id, Some(7));
        assert_eq!(snapshot.frame_index, Some(99));
        assert_eq!(snapshot.observation_age_sec, Some(2.0));
        assert!(!snapshot.stale);
        assert_eq!(owner.last_trace_snapshots(), traces);
        assert_eq!(owner.assignments()[&7].last_time_sec, Some(0.0));
    }

    #[test]
    fn region_cache_early_returns_preserve_state_and_do_not_sample_freshness_or_window() {
        for (source, no_beds, reason) in [
            (BedRegionCacheState::Expired, false, R::BedRegionUnavailable),
            (BedRegionCacheState::Empty, false, R::BedRegionUnavailable),
            (BedRegionCacheState::Fresh, true, R::BedObservationMissing),
            (BedRegionCacheState::Cached, true, R::BedObservationMissing),
        ] {
            let mut owner = custom(
                BedExitConfig {
                    outside_dwell_sec: 3.0,
                    ..config()
                },
                caps(),
            );
            arm(&mut owner);
            owner.update_night_window(Some(window())).unwrap();
            let before = owner.assignments().clone();
            let mut input = one(10.0, 0.0, false);
            input.bed_region = BedRegionDebugSnapshot {
                source,
                empty_cycles: 2,
            };
            input.live_track_ids.clear();
            if no_beds {
                input.bed_boxes.clear();
                input.containments.clear();
            }
            let output = owner
                .update(
                    &input,
                    BedExitClocks {
                        wall_time: None,
                        observed_at: f64::NAN,
                        snapshot_at: f64::NAN,
                    },
                )
                .unwrap();
            assert!(output.events.is_empty());
            assert_eq!(output.scoring_observation, None);
            assert_eq!(owner.assignments(), &before);
            let snapshot = owner.last_debug_snapshot().unwrap();
            assert!(!snapshot.stale);
            assert_eq!(snapshot.observation_age_sec, None);
            assert!(snapshot.bed_boxes.is_empty());
            assert_eq!(snapshot.person_boxes, input.person_boxes);
            assert_eq!(snapshot.bed_region, Some(input.bed_region));
            assert_eq!(owner.last_trace_snapshots()[0].reason, reason);
            assert_eq!(owner.last_trace_snapshots()[0].missing_values().len(), 2);
            owner.coast(None, 4.0).unwrap();
            assert!(owner.last_debug_snapshot().unwrap().stale); // Last real observation was t=1.
            let mut recovered = one(1.5, 0.0, false);
            recovered.bed_region.source = BedRegionCacheState::Cached;
            assert!(send(&mut owner, recovered).events.is_empty());
            assert!(send(&mut owner, one(2.0, 0.0, false)).events.is_empty());
            assert_eq!(send(&mut owner, one(4.5, 0.0, false)).events.len(), 1);
        }
    }

    #[test]
    fn coast_covers_but_preserves_an_onset_trace_and_does_not_move_dwell_anchor() {
        let mut owner = custom(
            BedExitConfig {
                outside_dwell_sec: 2.0,
                ..config()
            },
            caps(),
        );
        arm(&mut owner);
        owner.coast(Some(2), 4.0).unwrap();
        assert!(owner.last_debug_snapshot().unwrap().stale);
        assert_eq!(send(&mut owner, one(4.0, 0.0, false)).events.len(), 1);
        let traces = owner.last_trace_snapshots().to_vec();
        assert!(traces[0].triggered);
        owner.coast(Some(3), 100.0).unwrap();
        assert_eq!(owner.last_trace_snapshots(), traces);
        assert!(owner.last_debug_snapshot().unwrap().events.is_empty());
        assert_eq!(
            owner.last_debug_snapshot().unwrap().statuses[0].occupancy,
            BedOccupancy::Covered
        );
    }

    #[test]
    fn daytime_raw_exit_consumes_arm_but_not_authority_and_window_update_keeps_original_config() {
        let mut owner = custom(
            BedExitConfig {
                night_window: Some(window()),
                ..config()
            },
            caps(),
        );
        for (time, ratio, pose) in [(0.0, 1.0, false), (1.0, 1.0, true), (2.0, 0.0, false)] {
            assert!(
                owner
                    .update(&one(time, ratio, pose), clocks(time, 13))
                    .unwrap()
                    .events
                    .is_empty()
            );
        }
        assert!(!owner.assignments()[&7].armed);
        assert_eq!(owner.episode_state(7, 0), EpisodeState::Normal);
        let snapshot = owner.last_debug_snapshot().unwrap();
        assert_eq!(snapshot.statuses[0].occupancy, BedOccupancy::Exit);
        assert_eq!(
            snapshot.events,
            vec![BedExitEvent {
                person_id: 7,
                bed_id: 0
            }]
        );
        assert_eq!(snapshot.observation_age_sec, Some(0.0));
        let trace = &owner.last_trace_snapshots()[0];
        assert!(!trace.triggered);
        assert_eq!(trace.reason, R::OutsideDetectionWindow);
        owner.update_night_window(None).unwrap();
        assert!(owner.night_window().is_none());
        assert!(owner.config().night_window.is_some());
        send(&mut owner, one(3.0, 1.0, false));
        send(&mut owner, one(4.0, 1.0, true));
        let output = owner
            .update(
                &one(5.0, 0.0, false),
                BedExitClocks {
                    wall_time: None,
                    observed_at: 0.0,
                    snapshot_at: 0.0,
                },
            )
            .unwrap();
        assert_eq!(
            output.events[0].identity,
            "test-boot:test-epoch:bed-exit:0:7:0:0:1"
        );
    }

    #[test]
    fn elapsed_dwell_not_grace_frames_or_frame_rate_controls_exit() {
        for fps in [15, 30] {
            let mut owner = custom(
                BedExitConfig {
                    in_bed_dwell_sec: 3.0,
                    outside_dwell_sec: 2.0,
                    grace_frames: u64::MAX,
                    ..config()
                },
                caps(),
            );
            let mut events = Vec::new();
            // Same 3.2s occupancy / 2.2s departure facts as the Python FPS case.
            for index in 0..=(fps * 32 / 10) {
                let time = f64::from(index) / f64::from(fps);
                events.extend(send(&mut owner, one(time, 1.0, true)).events);
            }
            for index in 1..=(fps * 22 / 10) {
                let time = 3.2 + f64::from(index) / f64::from(fps);
                events.extend(send(&mut owner, one(time, 0.0, false)).events);
            }
            assert_eq!(events.len(), 1);
            assert_eq!(events[0].person_id, Some(7));
            assert!(events[0].time_sec >= 5.2 && events[0].time_sec <= 5.3);
            assert_eq!(owner.scoring().assignments_made, 1);
            assert_eq!(owner.scoring().grace_positive_transitions, 1);
        }
    }
}

#[cfg(test)]
mod handoff_tests {
    use super::fixtures::*;
    use super::*;
    use crate::trace::{
        DecisionTraceReason as R, DecisionTraceState as S, DecisionTraceValueName as V,
    };

    #[test]
    fn occupancy_churn_preserves_partial_and_armed_dwell_without_new_assignment_count() {
        for arm_time in [0.5, 1.0] {
            let mut owner = monitor();
            send(&mut owner, one(0.0, 1.0, false));
            send(&mut owner, one(arm_time, 1.0, true));
            let output = send(&mut owner, frame(2.0, &[8], &[&[0.5]], &[8]));
            assert!(output.events.is_empty());
            assert!(!owner.assignments().contains_key(&7));
            assert!(owner.assignments()[&8].armed);
            assert_eq!(owner.assignments()[&8].in_bed_dwell_sec, 2.0);
            assert_eq!(owner.assignments()[&8].candidate_frames, 1);
            assert_eq!(owner.scoring().assignments_made, 1);
            assert!(owner.last_lost_track_ids().is_empty());
            let trace = &owner.last_trace_snapshots()[0];
            assert_eq!(trace.reason, R::IdentityHandoff);
            assert_eq!(
                trace.previous_state,
                if arm_time == 1.0 { S::Armed } else { S::Arming }
            );
            assert_eq!(float(trace, V::InBedDwellSec), arm_time);
            let output = send(&mut owner, frame(3.0, &[8], &[&[0.0]], &[]));
            assert_eq!(output.events[0].person_id, Some(8));
            assert_eq!(
                output.events[0].identity,
                "test-boot:test-epoch:bed-exit:0:8:0:0:1"
            );
        }
    }

    #[test]
    fn inside_handoff_ties_use_last_eligible_observation_not_track_sort() {
        let mut owner = monitor();
        arm(&mut owner);
        let mut input = frame(2.0, &[8, 12, 9], &[&[0.5], &[0.5], &[1.0]], &[8, 12, 9]);
        input.bed_pose_features[2].observability = 0.349999;
        send(&mut owner, input);
        assert_eq!(owner.last_trace_snapshots()[0].track_id, Some(12));
        assert!(owner.assignments()[&12].armed);
        assert!(!owner.assignments()[&8].armed);
        assert!(!owner.assignments()[&9].armed);
        // Status occupancy is overwritten in observation order, not handoff order.
        assert_eq!(
            owner.last_debug_snapshot().unwrap().statuses[0].person_id,
            Some(9)
        );
    }

    #[test]
    fn stale_order_is_numeric_and_each_successor_is_claimed_at_most_once() {
        let mut owner = monitor();
        send(&mut owner, frame(0.0, &[9, 7], &[&[1.0], &[1.0]], &[]));
        send(&mut owner, frame(1.0, &[9, 7], &[&[1.0], &[1.0]], &[7, 9]));
        send(&mut owner, frame(2.0, &[8], &[&[1.0]], &[8]));
        let traces = owner.last_trace_snapshots();
        assert_eq!(
            (traces[0].reason, traces[0].track_id),
            (R::IdentityHandoff, Some(8))
        );
        assert_eq!(
            (traces[1].reason, traces[1].track_id),
            (R::StaleTrackClear, Some(9))
        );
        assert_eq!(
            (traces[2].reason, traces[2].track_id),
            (R::Contained, Some(8))
        );
        assert_eq!(owner.last_lost_track_ids(), &[9]);
        assert_eq!(owner.assignments().len(), 1);
    }

    #[test]
    fn previously_seen_unassigned_track_cannot_inherit_and_missing_bed_cannot_handoff() {
        let mut owner = monitor();
        send(&mut owner, frame(0.0, &[7, 8], &[&[1.0], &[0.0]], &[]));
        send(&mut owner, frame(1.0, &[7, 8], &[&[1.0], &[0.0]], &[7]));
        send(&mut owner, frame(2.0, &[8], &[&[1.0]], &[8]));
        assert_eq!(owner.last_trace_snapshots()[0].reason, R::StaleTrackClear);
        assert!(!owner.assignments()[&8].armed);
        let mut owner = monitor();
        send(&mut owner, frame(0.0, &[7], &[&[0.0, 1.0]], &[]));
        send(&mut owner, frame(1.0, &[7], &[&[0.0, 1.0]], &[7]));
        send(&mut owner, frame(2.0, &[8], &[&[1.0]], &[8]));
        assert_eq!(owner.last_trace_snapshots()[0].reason, R::StaleTrackClear);
        assert!(!owner.assignments()[&8].armed);
        assert_eq!(owner.assignments()[&8].bed_id, Some(0));
    }

    #[test]
    fn mid_exit_handoff_requires_positive_overlap_and_completes_exact_dwell_once() {
        for overlap_ratio in [0.0, f64::MIN_POSITIVE, 1.0] {
            let mut owner = custom(
                BedExitConfig {
                    outside_dwell_sec: 2.0,
                    ..config()
                },
                caps(),
            );
            arm(&mut owner);
            assert!(send(&mut owner, one(2.0, 0.0, false)).events.is_empty());
            let mut input = frame(3.0, &[9], &[&[0.0]], &[]);
            overlap(&owner, &mut input, 7, &[overlap_ratio]);
            let output = send(&mut owner, input);
            assert_eq!(output.events.len(), usize::from(overlap_ratio > 0.0));
            assert!(!owner.assignments().contains_key(&7));
            assert!(!owner.assignments()[&9].armed);
            if overlap_ratio > 0.0 {
                assert_eq!(output.events[0].person_id, Some(9));
                assert_eq!(
                    float(&owner.last_trace_snapshots()[0], V::OutsideDwellSec),
                    1.0
                );
                assert_eq!(owner.assignments()[&9].outside_dwell_sec, 0.0);
                assert_eq!(owner.last_trace_snapshots()[1].reason, R::OutsideDwellExit);
            } else {
                assert_eq!(owner.last_trace_snapshots()[0].reason, R::StaleTrackClear);
                assert_eq!(owner.last_lost_track_ids(), &[7]);
            }
            assert!(
                send(&mut owner, frame(4.0, &[9], &[&[0.0]], &[]))
                    .events
                    .is_empty()
            );
        }
    }

    #[test]
    fn other_bed_reoccupation_and_ambiguous_bodies_block_outside_handoff() {
        for case in 0..3 {
            let mut owner = custom(
                BedExitConfig {
                    outside_dwell_sec: 2.0,
                    ..config()
                },
                caps(),
            );
            arm(&mut owner);
            send(&mut owner, one(2.0, 0.0, false));
            let mut input = match case {
                0 => frame(3.0, &[9], &[&[0.0, 0.5]], &[]),
                1 => frame(3.0, &[10, 11], &[&[0.5], &[0.0]], &[]),
                _ => frame(3.0, &[10, 11], &[&[0.0], &[0.0]], &[]),
            };
            let overlaps = vec![1.0; input.person_boxes.len()];
            overlap(&owner, &mut input, 7, &overlaps);
            assert!(send(&mut owner, input).events.is_empty());
            assert_eq!(owner.last_trace_snapshots()[0].reason, R::StaleTrackClear);
            assert!(
                owner
                    .assignments()
                    .values()
                    .all(|assignment| !assignment.armed && assignment.outside_dwell_sec == 0.0)
            );
        }
    }

    #[test]
    fn absence_retires_without_emission_even_mid_exit_and_dead_observations_are_ignored() {
        let mut owner = custom(
            BedExitConfig {
                outside_dwell_sec: 3.0,
                grace_frames: 3,
                ..config()
            },
            caps(),
        );
        arm(&mut owner);
        send(&mut owner, one(2.0, 0.0, false));
        let mut dead = one(3.0, 0.0, false);
        dead.live_track_ids.clear();
        assert!(send(&mut owner, dead).events.is_empty());
        assert!(owner.assignments().is_empty());
        assert_eq!(owner.last_trace_snapshots()[0].reason, R::StaleTrackClear);
        assert_eq!(
            float(&owner.last_trace_snapshots()[0], V::OutsideDwellSec),
            1.0
        );
        assert!(send(&mut owner, one(100.0, 0.0, false)).events.is_empty());
        assert_eq!(owner.last_trace_snapshots()[0].reason, R::BelowContainment);
    }

    #[test]
    fn positional_ids_unknown_ids_and_live_but_unobserved_tracks_are_distinct() {
        let mut positional = monitor();
        let mut input = one(0.0, 1.0, false);
        input.track_ids.clear();
        input.live_track_ids.clear();
        send(&mut positional, input);
        assert!(positional.assignments().contains_key(&0));
        assert!(!positional.assignments().contains_key(&7));
        let mut owner = monitor();
        arm(&mut owner);
        let mut unknown = one(2.0, 0.0, false);
        unknown.track_ids = vec![None];
        // ID 7 is live, even though no box is associated to it this frame.
        assert!(send(&mut owner, unknown.clone()).events.is_empty());
        assert!(owner.assignments()[&7].armed);
        assert_eq!(owner.assignments()[&7].last_time_sec, Some(1.0));
        assert_eq!(
            owner.last_trace_snapshots()[0].reason,
            R::PersonObservationMissing
        );
        unknown.live_track_ids.clear();
        send(&mut owner, unknown);
        assert!(owner.assignments().is_empty());
        assert_eq!(owner.last_trace_snapshots()[0].reason, R::StaleTrackClear);
        send(&mut owner, frame(3.0, &[], &[], &[]));
        assert_eq!(
            owner.last_trace_snapshots()[0].reason,
            R::PersonObservationMissing
        );
        assert_eq!(
            owner.last_debug_snapshot().unwrap().statuses[0].occupancy,
            BedOccupancy::Empty
        );
    }
}

#[cfg(test)]
mod episode_tests {
    use super::fixtures::*;
    use super::*;
    use crate::trace::{
        DecisionTraceReason as R, DecisionTraceState as S, DecisionTraceValueName as V,
    };

    #[test]
    fn onset_release_and_positive_recovery_preserve_once_only_sequence_and_full_metadata() {
        let mut owner = custom(
            BedExitConfig {
                outside_dwell_sec: 3.0,
                grace_frames: 2,
                ..config()
            },
            caps(),
        );
        arm(&mut owner);
        for time in [2.0, 3.0] {
            assert!(send(&mut owner, one(time, 0.0, false)).events.is_empty());
            assert_eq!(owner.last_trace_snapshots()[0].reason, R::OutsideDwell);
        }
        let event = send(&mut owner, one(4.0, 0.0, false)).events.remove(0);
        assert_eq!(
            event,
            BusinessEvent {
                domain: "bed_exit".into(),
                event_type: "bed-exit".into(),
                identity: "test-boot:test-epoch:bed-exit:0:7:0:0:1".into(),
                camera_id: "camera-bed-exit".into(),
                facility_id: "facility-bed-exit".into(),
                time_sec: 4.0,
                probability: Some(1.0),
                person_id: Some(7),
                bed_id: Some(0),
            }
        );
        assert!(!owner.release_onset("other-boot:test-epoch:bed-exit:0:7:0:0:1"));
        assert_eq!(owner.episode_state(7, 0), EpisodeState::Open);
        assert!(owner.release_onset(&event.identity));
        assert!(!owner.release_onset(&event.identity));
        assert_eq!(owner.episode_state(7, 0), EpisodeState::Normal);
        assert!(send(&mut owner, one(5.0, 0.0, false)).events.is_empty());
        assert_eq!(owner.last_trace_snapshots()[0].reason, R::OutsideNotArmed);
        send(&mut owner, one(6.0, 1.0, true));
        let retry = send(&mut owner, one(9.0, 0.0, false)).events.remove(0);
        assert_eq!(retry.identity, "test-boot:test-epoch:bed-exit:0:7:0:0:2");
        assert!(!owner.release_onset(&event.identity));
        send(&mut owner, one(10.0, 1.0, true)); // Actual recovery, no release.
        assert_eq!(owner.episode_state(7, 0), EpisodeState::Normal);
        assert_eq!(
            send(&mut owner, one(13.0, 0.0, false)).events[0].identity,
            "test-boot:test-epoch:bed-exit:0:7:0:0:3"
        );
    }

    #[test]
    fn candidate_promotion_reassociates_unknown_episode_and_release_follows_identity() {
        let mut owner = custom(
            BedExitConfig {
                hold_frames: 2,
                ..config()
            },
            caps(),
        );
        for time in [0.0, 1.0, 2.0] {
            send(&mut owner, one(time, 1.0, true));
        }
        let first = send(&mut owner, one(3.0, 0.0, false)).events.remove(0);
        assert!(
            send(&mut owner, frame(4.0, &[], &[], &[]))
                .events
                .is_empty()
        );
        assert_eq!(owner.episode_state(7, 0), EpisodeState::Unknown);
        send(&mut owner, frame(5.0, &[8], &[&[1.0]], &[8]));
        assert_eq!(owner.last_trace_snapshots()[0].reason, R::AssignmentHold);
        assert_eq!(owner.episode_state(7, 0), EpisodeState::Unknown);
        send(&mut owner, frame(6.0, &[8], &[&[1.0]], &[8]));
        assert_eq!(owner.last_trace_snapshots()[0].reason, R::Assigned);
        assert_eq!(owner.episode_state(7, 0), EpisodeState::Normal);
        assert_eq!(owner.episode_state(8, 0), EpisodeState::Open);
        assert_eq!(owner.track_id_switch_absorbed_total(), 1);
        assert!(owner.release_onset(&first.identity));
        assert!(!owner.release_onset(&first.identity));
        send(&mut owner, frame(7.0, &[8], &[&[1.0]], &[8]));
        assert_eq!(
            send(&mut owner, frame(8.0, &[8], &[&[0.0]], &[])).events[0].identity,
            "test-boot:test-epoch:bed-exit:0:8:0:0:2"
        );
    }

    #[test]
    fn window_skips_lost_and_expiry_notifications_but_not_frame_retirement() {
        let mut owner = monitor();
        arm(&mut owner);
        send(&mut owner, one(2.0, 0.0, false));
        owner.update_night_window(Some(window())).unwrap();
        owner
            .update(&frame(3.0, &[], &[], &[]), clocks(3.0, 13))
            .unwrap();
        assert!(owner.assignments().is_empty());
        assert_eq!(owner.last_lost_track_ids(), &[7]);
        assert_eq!(owner.episode_state(7, 0), EpisodeState::Open);
        // Source resets the pending-lost list next frame, not a deferred queue.
        owner.update_night_window(None).unwrap();
        send(&mut owner, frame(100.0, &[], &[], &[]));
        assert_eq!(owner.episode_state(7, 0), EpisodeState::Open);

        let mut owner = monitor();
        arm(&mut owner);
        send(&mut owner, one(2.0, 0.0, false));
        send(&mut owner, frame(3.0, &[], &[], &[]));
        assert_eq!(owner.episode_state(7, 0), EpisodeState::Unknown);
        owner.update_night_window(Some(window())).unwrap();
        owner
            .update(&frame(100.0, &[], &[], &[]), clocks(100.0, 13))
            .unwrap();
        assert_eq!(owner.episode_state(7, 0), EpisodeState::Unknown);
        owner.update_night_window(None).unwrap();
        send(&mut owner, frame(100.0, &[], &[], &[]));
        assert_eq!(owner.episode_state(7, 0), EpisodeState::Resolved);
    }

    #[test]
    fn suppressed_recovery_leaves_open_episode_and_rewrites_only_triggered_rows() {
        let mut owner = monitor();
        arm(&mut owner);
        send(&mut owner, one(2.0, 0.0, false));
        owner.update_night_window(Some(window())).unwrap();
        owner.update(&one(3.0, 1.0, true), clocks(3.0, 13)).unwrap();
        assert!(owner.assignments()[&7].armed);
        assert_eq!(owner.episode_state(7, 0), EpisodeState::Open);
        owner.update_night_window(None).unwrap();
        let input = frame(4.0, &[7, 8], &[&[0.0], &[0.0]], &[]);
        assert!(send(&mut owner, input).events.is_empty());
        let trace = &owner.last_trace_snapshots()[0];
        assert_eq!(trace.reason, R::EpisodeAlreadyOpen);
        assert_eq!(
            (trace.previous_state, trace.current_state),
            (S::Armed, S::Triggered)
        );
        assert!(!trace.triggered);
        assert_eq!(float(trace, V::OutsideDwellSec), 1.0);
        assert_eq!(owner.last_trace_snapshots()[1].reason, R::BelowContainment);
        assert_eq!(
            owner.last_debug_snapshot().unwrap().statuses[0].occupancy,
            BedOccupancy::Exit
        );
    }

    #[test]
    fn emission_trace_and_occupancy_order_are_observation_order_not_sorted_ids() {
        let mut owner = monitor();
        for time in [0.0, 1.0] {
            send(&mut owner, frame(time, &[9, 7], &[&[1.0], &[1.0]], &[9, 7]));
        }
        let events = send(&mut owner, frame(2.0, &[9, 7], &[&[0.0], &[0.0]], &[])).events;
        assert_eq!(
            events
                .iter()
                .map(|event| event.person_id)
                .collect::<Vec<_>>(),
            vec![Some(9), Some(7)]
        );
        assert_eq!(
            events[0].identity,
            "test-boot:test-epoch:bed-exit:0:9:0:0:1"
        );
        assert_eq!(
            events[1].identity,
            "test-boot:test-epoch:bed-exit:0:7:0:0:2"
        );
        assert_eq!(
            owner
                .last_trace_snapshots()
                .iter()
                .map(|trace| trace.track_id)
                .collect::<Vec<_>>(),
            vec![Some(9), Some(7)]
        );
        send(&mut owner, frame(3.0, &[9, 7], &[&[1.0], &[1.0]], &[9, 7]));
        send(&mut owner, frame(4.0, &[9, 7], &[&[1.0], &[0.0]], &[9]));
        let status = &owner.last_debug_snapshot().unwrap().statuses[0];
        assert_eq!(status.occupancy, BedOccupancy::Exit); // Exit outranks occupied.
        assert_eq!(status.person_id, Some(9));
    }

    #[test]
    fn duplicate_observations_are_processed_and_cameras_do_not_share_assignment_state() {
        let mut a = custom(
            BedExitConfig {
                hold_frames: 2,
                camera_id: "camera-a".into(),
                ..config()
            },
            caps(),
        );
        let mut b = custom(
            BedExitConfig {
                camera_id: "camera-b".into(),
                ..config()
            },
            caps(),
        );
        send(&mut a, frame(0.0, &[7, 7], &[&[1.0], &[1.0]], &[]));
        assert_eq!(
            a.last_trace_snapshots()
                .iter()
                .map(|trace| trace.reason)
                .collect::<Vec<_>>(),
            vec![R::AssignmentHold, R::Assigned]
        );
        send(&mut a, one(1.0, 1.0, true));
        assert!(send(&mut b, one(2.0, 0.0, false)).events.is_empty());
        let event = send(&mut a, one(2.0, 0.0, false)).events.remove(0);
        assert_eq!(event.camera_id, "camera-a");
        assert!(!b.release_onset(&event.identity));
        assert_eq!(b.episode_state(7, 0), EpisodeState::Normal);
    }
}

#[cfg(test)]
mod admission_tests {
    use super::fixtures::*;
    use super::*;

    fn rejects_unchanged(
        owner: &mut BedExitMonitor,
        input: &BedExitInput,
        clocks: BedExitClocks,
        cause: BedExitFailure,
    ) {
        let assignments = owner.assignments().clone();
        let debug = owner.last_debug_snapshot().cloned();
        let traces = owner.last_trace_snapshots().to_vec();
        let scoring = owner.scoring();
        let observed = owner.freshness.last_observed_at;
        let disposition = owner.last_episode_disposition();
        let complete = owner.last_update_complete();
        assert_eq!(
            owner.update(input, clocks),
            Err(BedExitError::Rejected(cause))
        );
        assert_eq!(owner.assignments(), &assignments);
        assert_eq!(owner.last_debug_snapshot(), debug.as_ref());
        assert_eq!(owner.last_trace_snapshots(), traces);
        assert_eq!(owner.scoring(), scoring);
        assert_eq!(owner.freshness.last_observed_at, observed);
        assert_eq!(owner.last_episode_disposition(), disposition);
        assert_eq!(owner.last_update_complete(), complete);
        assert_eq!(owner.poisoned, None);
    }

    #[test]
    fn configuration_rejects_invalid_domains_and_identity_byte_overflow() {
        for (field, value) in [
            ("min_containment", 0.0),
            ("min_containment", 1.000001),
            ("min_containment", f64::NAN),
            ("in_bed_dwell_sec", 0.0),
            ("in_bed_dwell_sec", f64::INFINITY),
            ("outside_dwell_sec", -1.0),
            ("outside_dwell_sec", f64::NEG_INFINITY),
        ] {
            let mut config = config();
            match field {
                "min_containment" => config.min_containment = value,
                "in_bed_dwell_sec" => config.in_bed_dwell_sec = value,
                _ => config.outside_dwell_sec = value,
            }
            assert_eq!(
                BedExitMonitor::new(config, "boot", "epoch", 0, caps(), 3.0).unwrap_err(),
                BedExitFailure::InvalidConfig(field)
            );
        }
        assert_eq!(
            BedExitMonitor::new(
                BedExitConfig {
                    hold_frames: 0,
                    ..config()
                },
                "boot",
                "epoch",
                0,
                caps(),
                3.0
            )
            .unwrap_err(),
            BedExitFailure::InvalidConfig("hold_frames")
        );
        for stale in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(
                ObservationFreshness::new(stale).unwrap_err(),
                BedExitFailure::InvalidConfig("stale_after_sec")
            );
        }
        for field in ["camera", "facility"] {
            let mut config = config();
            if field == "camera" {
                config.camera_id = "가".repeat(86);
            } else {
                config.facility_id = "x".repeat(257);
            }
            assert_eq!(
                BedExitMonitor::new(config, "boot", "epoch", 0, caps(), 3.0).unwrap_err(),
                BedExitFailure::InvalidIdentity
            );
        }
        for (boot, epoch) in [("", "epoch"), ("boot", "")] {
            assert_eq!(
                BedExitMonitor::new(config(), boot, epoch, 0, caps(), 3.0).unwrap_err(),
                BedExitFailure::Episode(crate::episode::EpisodeError::InvalidIdentity)
            );
        }
        let mut exact = custom(
            BedExitConfig {
                min_containment: 1.0,
                camera_id: "x".repeat(256),
                ..config()
            },
            caps(),
        );
        send(&mut exact, one(0.0, 0.999999, false));
        assert_eq!(exact.assignments()[&7].bed_id, None);
        send(&mut exact, one(1.0, 1.0, false));
        assert_eq!(exact.assignments()[&7].bed_id, Some(0));
    }

    #[test]
    fn all_capacity_dimensions_reject_overflow_before_mutation() {
        let small = BedExitCapacities {
            tracks: 1,
            observations: 1,
            beds: 1,
            pose_features: 1,
            polygon_points: 3,
            episodes: 1,
        };
        let mut owner = custom(config(), small);
        send(&mut owner, one(0.0, 1.0, false));
        for dimension in [
            "observations",
            "track_ids",
            "beds",
            "live_track_ids",
            "pose_features",
            "containment_rows",
            "containment_columns",
            "prior_box_rows",
            "overlap_columns",
            "polygon_points",
        ] {
            let mut input = one(1.0, 1.0, true);
            match dimension {
                "observations" => input.person_boxes.push(input.person_boxes[0].clone()),
                "track_ids" => input.track_ids.push(Some(8)),
                "beds" => input.bed_boxes.push(input.bed_boxes[0].clone()),
                "live_track_ids" => input.live_track_ids.push(8),
                "pose_features" => input.bed_pose_features.push(lying(8)),
                "containment_rows" => input.containments.push(vec![1.0]),
                "containment_columns" => input.containments[0].push(1.0),
                "prior_box_rows" => {
                    overlap(&owner, &mut input, 7, &[1.0]);
                    overlap(&owner, &mut input, 7, &[1.0]);
                }
                "overlap_columns" => overlap(&owner, &mut input, 7, &[1.0, 1.0]),
                "polygon_points" => input.person_boxes[0].polygon = Some(vec![(0, 0); 4]),
                _ => panic!("test dimension"),
            }
            rejects_unchanged(
                &mut owner,
                &input,
                clocks(1.0, 22),
                BedExitFailure::Capacity(dimension),
            );
        }
        for dimension in 0..6 {
            let mut limits = caps();
            match dimension {
                0 => limits.tracks = 0,
                1 => limits.observations = 0,
                2 => limits.beds = 0,
                3 => limits.pose_features = 0,
                4 => limits.polygon_points = 0,
                _ => limits.episodes = 0,
            }
            assert_eq!(
                BedExitMonitor::new(config(), "boot", "epoch", 0, limits, 3.0).unwrap_err(),
                BedExitFailure::InvalidCapacities
            );
        }
        assert_eq!(
            BedExitMonitor::new(
                config(),
                "boot",
                "epoch",
                0,
                BedExitCapacities {
                    observations: usize::MAX,
                    ..caps()
                },
                3.0
            )
            .unwrap_err(),
            BedExitFailure::InvalidCapacities
        );
    }

    #[test]
    fn geometry_shape_binding_and_missing_overlap_never_use_a_fallback() {
        let mut owner = custom(
            BedExitConfig {
                outside_dwell_sec: 2.0,
                ..config()
            },
            caps(),
        );
        arm(&mut owner);
        send(&mut owner, one(2.0, 0.0, false));
        let mut input = frame(3.0, &[9], &[&[0.0]], &[]);
        rejects_unchanged(
            &mut owner,
            &input,
            clocks(3.0, 22),
            BedExitFailure::MissingPriorOverlap(7),
        );
        overlap(&owner, &mut input, 7, &[1.0]);
        for case in 0..6 {
            let mut bad = input.clone();
            let cause = match case {
                0 => {
                    bad.track_ids.push(None);
                    BedExitFailure::InvalidShape("track_ids")
                }
                1 => {
                    bad.containments.clear();
                    BedExitFailure::InvalidShape("containments")
                }
                2 => {
                    bad.containments[0].clear();
                    BedExitFailure::InvalidShape("containments")
                }
                3 => {
                    bad.prior_box_containments[0].previous_box.x1 += 1;
                    BedExitFailure::MismatchedPriorBox(7)
                }
                4 => {
                    bad.prior_box_containments[0].ratios.clear();
                    BedExitFailure::InvalidShape("prior_box_containments")
                }
                _ => {
                    bad.prior_box_containments
                        .push(bad.prior_box_containments[0].clone());
                    BedExitFailure::InvalidShape("prior_box_containments")
                }
            };
            rejects_unchanged(&mut owner, &bad, clocks(3.0, 22), cause);
        }
        assert_eq!(send(&mut owner, input).events.len(), 1);
    }

    #[test]
    fn nonfinite_inputs_and_clocks_reject_atomically_without_changing_finite_domain() {
        let mut owner = monitor();
        arm(&mut owner);
        for field in [
            "time_sec",
            "box.confidence",
            "bed_pose_features",
            "containment_ratio",
            "prior_box_containment",
        ] {
            let mut input = one(2.0, 1.0, true);
            match field {
                "time_sec" => input.time_sec = Some(f64::NAN),
                "box.confidence" => input.bed_boxes[0].confidence = f64::INFINITY,
                "bed_pose_features" => input.bed_pose_features[0].hip_depth = f64::NEG_INFINITY,
                "containment_ratio" => input.containments[0][0] = f64::NAN,
                _ => overlap(&owner, &mut input, 7, &[f64::NAN]),
            }
            rejects_unchanged(
                &mut owner,
                &input,
                clocks(2.0, 22),
                BedExitFailure::NonFinite(field),
            );
        }
        for (observed_at, snapshot_at, cause) in [
            (
                f64::NAN,
                2.0,
                BedExitFailure::NonFinite("observation_clock"),
            ),
            (2.0, f64::NAN, BedExitFailure::NonFinite("snapshot_clock")),
            (
                -f64::MAX,
                f64::MAX,
                BedExitFailure::Overflow("observation_age_sec"),
            ),
        ] {
            rejects_unchanged(
                &mut owner,
                &one(2.0, 1.0, true),
                BedExitClocks {
                    wall_time: None,
                    observed_at,
                    snapshot_at,
                },
                cause,
            );
        }
        owner.update_night_window(Some(window())).unwrap();
        rejects_unchanged(
            &mut owner,
            &one(2.0, 0.0, false),
            BedExitClocks {
                wall_time: None,
                observed_at: 2.0,
                snapshot_at: 2.0,
            },
            BedExitFailure::MissingWallClock,
        );
        let debug = owner.last_debug_snapshot().cloned();
        assert!(matches!(
            owner.coast(None, f64::NAN),
            Err(BedExitError::Rejected(BedExitFailure::NonFinite(
                "snapshot_clock"
            )))
        ));
        assert_eq!(owner.last_debug_snapshot(), debug.as_ref());
    }
}

#[cfg(test)]
mod fatal_tests {
    use super::fixtures::*;
    use super::*;
    use crate::episode::{CapacityLimit, EpisodeError};

    #[test]
    fn fatal_onset_capacity_preserves_accepted_prefix_in_observation_order_and_stops_owner() {
        let mut owner = custom(
            BedExitConfig {
                night_window: Some(window()),
                ..config()
            },
            BedExitCapacities {
                episodes: 1,
                ..caps()
            },
        );
        // Arm outside the window so recoveries do not occupy authority rows.
        for time in [0.0, 1.0] {
            owner
                .update(
                    &frame(time, &[9, 7], &[&[1.0], &[1.0]], &[9, 7]),
                    clocks(time, 13),
                )
                .unwrap();
        }
        owner.update_night_window(None).unwrap();
        let input = frame(2.0, &[9, 7], &[&[0.0], &[0.0]], &[]);
        let cause = BedExitFailure::Episode(EpisodeError::Capacity(CapacityLimit::Episodes));
        let error = owner.update(&input, clocks(2.0, 22)).unwrap_err();
        let BedExitError::FatalPartialState {
            phase,
            track_id,
            cause: actual,
            emitted_events,
        } = error
        else {
            panic!("fatal error required")
        };
        assert_eq!(
            (phase, track_id, actual),
            (BedExitPhase::Onset, Some(7), cause)
        );
        assert_eq!(emitted_events.len(), 1);
        assert_eq!(
            emitted_events[0].identity,
            "test-boot:test-epoch:bed-exit:0:9:0:0:1"
        );
        assert_eq!(owner.last_debug_snapshot().unwrap().events.len(), 2);
        assert!(!owner.last_update_complete());
        assert_eq!(owner.episode_state(9, 0), EpisodeState::Open);
        assert_eq!(owner.episode_state(7, 0), EpisodeState::Normal);
        let assignments = owner.assignments().clone();
        assert_eq!(
            owner.update(&input, clocks(3.0, 22)),
            Err(BedExitError::Poisoned(cause))
        );
        assert_eq!(owner.coast(None, 3.0), Err(BedExitError::Poisoned(cause)));
        assert_eq!(
            owner.update_night_window(None),
            Err(BedExitError::Poisoned(cause))
        );
        assert_eq!(owner.assignments(), &assignments);
        assert!(owner.release_onset(&emitted_events[0].identity));
        assert!(!owner.release_onset(&emitted_events[0].identity));
        assert_eq!(
            owner.update(&input, clocks(3.0, 22)),
            Err(BedExitError::Poisoned(cause))
        );
    }

    #[test]
    fn later_recovery_capacity_failure_cannot_discard_an_earlier_onset() {
        let mut owner = custom(
            config(),
            BedExitCapacities {
                episodes: 1,
                ..caps()
            },
        );
        send(&mut owner, frame(0.0, &[7, 8], &[&[1.0], &[1.0]], &[]));
        send(&mut owner, frame(1.0, &[7, 8], &[&[1.0], &[1.0]], &[7]));
        let error = owner
            .update(
                &frame(2.0, &[7, 8], &[&[0.0], &[1.0]], &[8]),
                clocks(2.0, 22),
            )
            .unwrap_err();
        let BedExitError::FatalPartialState {
            phase,
            track_id,
            cause,
            emitted_events,
        } = error
        else {
            panic!("fatal error required")
        };
        assert_eq!((phase, track_id), (BedExitPhase::Recovery, Some(8)));
        assert_eq!(
            cause,
            BedExitFailure::Episode(EpisodeError::Capacity(CapacityLimit::Episodes))
        );
        assert_eq!(emitted_events.len(), 1);
        assert_eq!(emitted_events[0].person_id, Some(7));
        assert!(owner.release_onset(&emitted_events[0].identity));
        assert!(!owner.last_update_complete());
    }

    #[test]
    fn integer_and_elapsed_overflow_are_fatal_not_saturation_or_fake_empty_success() {
        for field in [
            "candidate_frames",
            "assignments_made",
            "grace_positive_transitions",
            "in_bed_dwell_sec",
            "outside_dwell_sec",
        ] {
            let mut owner = monitor();
            let input = match field {
                "candidate_frames" => {
                    owner.config.hold_frames = usize::MAX;
                    send(&mut owner, one(0.0, 1.0, false));
                    owner.assignments.get_mut(&7).unwrap().candidate_frames = usize::MAX;
                    one(1.0, 1.0, true)
                }
                "assignments_made" => {
                    owner.scoring.assignments_made = u64::MAX;
                    one(0.0, 1.0, false)
                }
                "grace_positive_transitions" => {
                    send(&mut owner, one(0.0, 1.0, false));
                    owner.scoring.grace_positive_transitions = u64::MAX;
                    one(1.0, 1.0, true)
                }
                _ => {
                    send(&mut owner, one(-f64::MAX, 1.0, false));
                    one(
                        f64::MAX,
                        if field == "in_bed_dwell_sec" {
                            1.0
                        } else {
                            0.0
                        },
                        true,
                    )
                }
            };
            let error = owner.update(&input, clocks(0.0, 22)).unwrap_err();
            let BedExitError::FatalPartialState {
                phase,
                cause,
                emitted_events,
                ..
            } = error
            else {
                panic!("fatal error required")
            };
            assert_eq!(phase, BedExitPhase::Frame);
            assert_eq!(cause, BedExitFailure::Overflow(field));
            assert!(emitted_events.is_empty());
            assert!(!owner.last_update_complete());
            assert_eq!(owner.coast(None, 1.0), Err(BedExitError::Poisoned(cause)));
        }
    }

    #[test]
    fn full_track_capacity_handoff_replaces_without_transient_overflow_or_eviction() {
        let mut owner = custom(
            config(),
            BedExitCapacities {
                tracks: 1,
                ..caps()
            },
        );
        arm(&mut owner);
        send(&mut owner, frame(2.0, &[8], &[&[1.0]], &[8]));
        assert_eq!(
            owner.assignments().keys().copied().collect::<Vec<_>>(),
            vec![8]
        );
        assert!(owner.assignments()[&8].armed);
        assert_eq!(
            send(&mut owner, frame(3.0, &[8], &[&[0.0]], &[]))
                .events
                .len(),
            1
        );
    }
}
