//! Scored policy port of worker/domains/fall/policy.py, not its classifier adapter.
//! EpisodeAuthority alone owns episode votes, reassociation, suppression, and identity.
use std::collections::{BTreeMap, BTreeSet};

use crate::episode::{BusinessEvent, EpisodeAuthority, EpisodeError, EpisodeState};
use crate::trace::{DecisionTraceMissingReason, DecisionTraceSnapshot};

pub use errors::{FallCapacity, FallCounter, FallError, FallFailure};
pub use parameters::{FallCapacities, FallPolicy, FallPolicyParameters, FallProbabilities};
use track::TrackState;

mod errors {
    use crate::episode::{BusinessEvent, EpisodeError};
    use std::fmt;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum FallCapacity {
        TrackStates,
        LiveTracks,
        GenerationIdentities,
        VoteWindow,
    }
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum FallCounter {
        Generation,
        FallenStreak,
        RecoveryStreak,
    }
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum FallFailure {
        InvalidPolicy(&'static str),
        InvalidProbability(&'static str),
        InvalidIdentity,
        InvalidCapacities,
        NonFiniteTime,
        Capacity(FallCapacity),
        Overflow(FallCounter),
        Episode(EpisodeError),
    }
    impl From<EpisodeError> for FallFailure {
        fn from(value: EpisodeError) -> Self {
            Self::Episode(value)
        }
    }
    impl fmt::Display for FallFailure {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::InvalidPolicy(field) => write!(f, "invalid fall policy field: {field}"),
                Self::InvalidProbability(field) => write!(f, "invalid fall probability: {field}"),
                Self::InvalidIdentity => {
                    f.write_str("fall camera/facility identity exceeds byte bound")
                }
                Self::InvalidCapacities => f.write_str("fall capacities must all be positive"),
                Self::NonFiniteTime => f.write_str("fall update time must be finite"),
                Self::Capacity(limit) => write!(f, "fall capacity exceeded: {limit:?}"),
                Self::Overflow(counter) => write!(f, "fall counter overflow: {counter:?}"),
                Self::Episode(error) => write!(f, "{error}"),
            }
        }
    }
    impl std::error::Error for FallFailure {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            match self {
                Self::Episode(error) => Some(error),
                _ => None,
            }
        }
    }

    /// Rejections preserve the prior call. A fatal partial update must stop this owner:
    /// earlier tracks may already have emitted identities, which are returned here,
    /// never disguised as a successful empty score. No implicit reset/retry is safe.
    #[derive(Debug, Clone, PartialEq)]
    pub enum FallError {
        Rejected(FallFailure),
        FatalPartialState {
            track_id: u64,
            cause: FallFailure,
            emitted_events: Vec<BusinessEvent>,
        },
        Poisoned(FallFailure),
    }
    impl From<FallFailure> for FallError {
        fn from(value: FallFailure) -> Self {
            Self::Rejected(value)
        }
    }
    impl fmt::Display for FallError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::Rejected(cause) => write!(f, "fall update rejected: {cause}"),
                Self::FatalPartialState {
                    track_id, cause, ..
                } => {
                    write!(f, "fatal partial fall update at track {track_id}: {cause}")
                }
                Self::Poisoned(cause) => {
                    write!(f, "fall owner stopped after partial update: {cause}")
                }
            }
        }
    }
    impl std::error::Error for FallError {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            Some(match self {
                Self::Rejected(cause) | Self::Poisoned(cause) => cause,
                Self::FatalPartialState { cause, .. } => cause,
            })
        }
    }
}

mod parameters {
    use super::{FallCapacity, FallFailure};

    /// All FallPolicyV2 fields, including the currently unused cooldown_frames.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct FallPolicyParameters {
        pub transition_threshold: f64,
        pub transition_votes: usize,
        pub transition_window: usize,
        pub fallen_threshold: f64,
        pub fallen_consecutive: u64,
        pub recovery_transition_max: f64,
        pub recovery_fallen_max: f64,
        pub recovery_consecutive: u64,
        pub track_ttl_frames: u64,
        pub cooldown_frames: u64,
    }
    impl Default for FallPolicyParameters {
        fn default() -> Self {
            Self {
                transition_threshold: 0.5,
                transition_votes: 3,
                transition_window: 5,
                fallen_threshold: 0.8,
                fallen_consecutive: 3,
                recovery_transition_max: 0.4,
                recovery_fallen_max: 0.5,
                recovery_consecutive: 5,
                track_ttl_frames: 45,
                cooldown_frames: 90,
            }
        }
    }
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct FallPolicy(FallPolicyParameters);
    impl FallPolicy {
        pub fn new(parameters: FallPolicyParameters) -> Result<Self, FallFailure> {
            for (name, value) in [
                ("transition_threshold", parameters.transition_threshold),
                ("fallen_threshold", parameters.fallen_threshold),
                (
                    "recovery_transition_max",
                    parameters.recovery_transition_max,
                ),
                ("recovery_fallen_max", parameters.recovery_fallen_max),
            ] {
                if !valid_probability(value) {
                    return Err(FallFailure::InvalidPolicy(name));
                }
            }
            for (name, positive) in [
                ("transition_votes", parameters.transition_votes > 0),
                ("transition_window", parameters.transition_window > 0),
                ("fallen_consecutive", parameters.fallen_consecutive > 0),
                ("recovery_consecutive", parameters.recovery_consecutive > 0),
                ("track_ttl_frames", parameters.track_ttl_frames > 0),
                ("cooldown_frames", parameters.cooldown_frames > 0),
            ] {
                if !positive {
                    return Err(FallFailure::InvalidPolicy(name));
                }
            }
            if parameters.transition_votes > parameters.transition_window {
                return Err(FallFailure::InvalidPolicy("transition_votes"));
            }
            Ok(Self(parameters))
        }
        pub fn parameters(&self) -> &FallPolicyParameters {
            &self.0
        }
    }
    impl Default for FallPolicy {
        fn default() -> Self {
            Self::new(FallPolicyParameters::default()).expect("canonical policy is valid")
        }
    }

    /// The source calls the normal class background. No sum-to-one constraint or
    /// calibration/evidence reconstruction is imposed by this scored-policy seam.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct FallProbabilities {
        background: f64,
        fall_transition: f64,
        fallen: f64,
    }
    impl FallProbabilities {
        pub fn new(
            background: f64,
            fall_transition: f64,
            fallen: f64,
        ) -> Result<Self, FallFailure> {
            for (name, value) in [
                ("background", background),
                ("fall_transition", fall_transition),
                ("fallen", fallen),
            ] {
                if !valid_probability(value) {
                    return Err(FallFailure::InvalidProbability(name));
                }
            }
            Ok(Self {
                background,
                fall_transition,
                fallen,
            })
        }
        pub fn background(self) -> f64 {
            self.background
        }
        pub fn fall_transition(self) -> f64 {
            self.fall_transition
        }
        pub fn fallen(self) -> f64 {
            self.fallen
        }
    }
    fn valid_probability(value: f64) -> bool {
        value.is_finite() && (0.0..=1.0).contains(&value)
    }

    /// No defaults: composition must deliberately bound every retained collection.
    /// retained_tracks also bounds live IDs and the last evaluated trace batch,
    /// including live IDs that have never received a score. vote_window bounds
    /// the authority window AND the source's fixed five-entry trace-only history.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct FallCapacities {
        pub retained_tracks: usize,
        pub generation_identities: usize,
        pub episodes: usize,
        pub vote_window: usize,
    }
    impl FallCapacities {
        pub(super) fn validate(self, policy: &FallPolicy) -> Result<(), FallFailure> {
            if [
                self.retained_tracks,
                self.generation_identities,
                self.episodes,
                self.vote_window,
            ]
            .contains(&0)
            {
                return Err(FallFailure::InvalidCapacities);
            }
            if self.vote_window < 5 || self.vote_window < policy.parameters().transition_window {
                return Err(FallFailure::Capacity(FallCapacity::VoteWindow));
            }
            Ok(())
        }
    }
}

mod track {
    use super::{FallCounter, FallFailure, FallPolicyParameters, FallProbabilities};

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(super) struct TrackState {
        pub generation: u64,
        pub last_seen_frame: i64,
        // Python's deque(maxlen=5), independent of policy.transition_window.
        // Used ONLY to name trace state; never to decide episode confirmation.
        pub transition_history: [bool; 5],
        pub fallen_streak: u64,
        pub recovery_streak: u64,
        pub fallen: bool,
        pub initialized: bool,
        pub proposed_this_call: bool,
    }
    impl TrackState {
        pub fn new(generation: u64, frame_index: i64) -> Self {
            Self {
                generation,
                last_seen_frame: frame_index,
                transition_history: [false; 5],
                fallen_streak: 0,
                recovery_streak: 0,
                fallen: false,
                initialized: false,
                proposed_this_call: false,
            }
        }
        pub fn has_transition(self) -> bool {
            self.transition_history.contains(&true)
        }
        pub fn stale(self, frame_index: i64, ttl: u64) -> bool {
            i128::from(frame_index) - i128::from(self.last_seen_frame) >= i128::from(ttl)
        }
        /// None means the initialized-already-fallen early return (no proposal).
        /// A local copy reserves both streak increments before touching authority state.
        pub fn scored(
            mut self,
            probability: FallProbabilities,
            policy: &FallPolicyParameters,
        ) -> Result<(Self, Option<bool>), FallFailure> {
            if !self.initialized {
                self.initialized = true;
                if probability.fallen() >= policy.fallen_threshold {
                    self.fallen = true;
                    self.proposed_this_call = false;
                    return Ok((self, None));
                }
            }
            let qualifying = probability.fall_transition() >= policy.transition_threshold;
            self.transition_history.rotate_left(1);
            self.transition_history[4] = qualifying;
            self.fallen_streak = if probability.fallen() >= policy.fallen_threshold {
                self.fallen_streak
                    .checked_add(1)
                    .ok_or(FallFailure::Overflow(FallCounter::FallenStreak))?
            } else {
                0
            };
            if self.fallen_streak >= policy.fallen_consecutive {
                self.fallen = true;
            }
            let recovering = probability.fall_transition() < policy.recovery_transition_max
                && probability.fallen() < policy.recovery_fallen_max;
            self.recovery_streak = if recovering {
                self.recovery_streak
                    .checked_add(1)
                    .ok_or(FallFailure::Overflow(FallCounter::RecoveryStreak))?
            } else {
                0
            };
            let confirmed_recovery =
                recovering && self.recovery_streak >= policy.recovery_consecutive;
            if self.recovery_streak >= policy.recovery_consecutive {
                self.fallen = false;
                self.fallen_streak = 0;
                self.transition_history.fill(false);
            }
            self.proposed_this_call = qualifying && !confirmed_recovery;
            Ok((self, Some(confirmed_recovery)))
        }
    }
}

/// One camera/boot/source epoch. Identity tables are never LRU-evicted.
/// On fatal partial failure, all mutating entry points reject further calls;
/// getters remain available for diagnostics but traces are explicitly stale.
#[derive(Debug)]
pub struct FallPolicyDecider {
    camera_id: String,
    facility_id: String,
    policy: FallPolicy,
    capacities: FallCapacities,
    states: BTreeMap<u64, TrackState>,
    next_generations: BTreeMap<u64, u64>,
    episodes: EpisodeAuthority,
    last_trace_snapshots: Vec<DecisionTraceSnapshot>,
    last_update_evaluated: bool,
    fatal_failure: Option<FallFailure>,
}
impl FallPolicyDecider {
    pub fn new(
        camera_id: impl Into<String>,
        facility_id: impl Into<String>,
        boot_id: impl Into<String>,
        stream_epoch: impl Into<String>,
        source_generation: u64,
        policy: FallPolicy,
        capacities: FallCapacities,
    ) -> Result<Self, FallFailure> {
        capacities.validate(&policy)?;
        let (camera_id, facility_id) = (camera_id.into(), facility_id.into());
        // Python does not reject empty camera/facility strings. Boot/epoch validation
        // remains the authority's responsibility; none of these inputs are truncated.
        if [&camera_id, &facility_id]
            .iter()
            .any(|id| id.len() > crate::episode::MAX_AUTHORITY_ID_BYTES)
        {
            return Err(FallFailure::InvalidIdentity);
        }
        let episodes = EpisodeAuthority::new(
            boot_id,
            stream_epoch,
            source_generation,
            capacities.episodes,
            capacities.vote_window,
        )?;
        Ok(Self {
            camera_id,
            facility_id,
            policy,
            capacities,
            states: BTreeMap::new(),
            next_generations: BTreeMap::new(),
            episodes,
            last_trace_snapshots: Vec::new(),
            last_update_evaluated: false,
            fatal_failure: None,
        })
    }

    /// Only this call's live scores are consumed; absent scores never become empty-room
    /// observations. Live IDs are deduplicated and evaluated in numeric track order.
    /// Signed frame/time rollback is NOT an epoch reset at this layer.
    pub fn update(
        &mut self,
        frame_index: i64,
        time_sec: f64,
        probabilities_by_track: &BTreeMap<u64, FallProbabilities>,
        live_track_ids: impl IntoIterator<Item = u64>,
        missing_score_reasons: Option<&BTreeMap<u64, DecisionTraceMissingReason>>,
    ) -> Result<Vec<BusinessEvent>, FallError> {
        self.ensure_healthy()?;
        if !time_sec.is_finite() {
            return Err(FallFailure::NonFiniteTime.into());
        }
        let mut live = BTreeSet::new();
        for track_id in live_track_ids {
            if !live.contains(&track_id) && live.len() == self.capacities.retained_tracks {
                return Err(FallFailure::Capacity(FallCapacity::LiveTracks).into());
            }
            live.insert(track_id);
        }
        self.preflight(&live, frame_index, probabilities_by_track)?;
        if let Err((track_id, cause)) = self.evict_stale(&live, frame_index, time_sec) {
            return Err(self.fail(track_id, cause.into(), Vec::new()));
        }
        let mut emitted = Vec::new();
        let mut snapshots = Vec::with_capacity(live.len());
        for track_id in live {
            if let Some(state) = self.states.get_mut(&track_id) {
                state.last_seen_frame = frame_index;
            }
            let Some(&probability) = probabilities_by_track.get(&track_id) else {
                snapshots.push(DecisionTraceSnapshot::missing(
                    track_id,
                    self.trace_state(track_id),
                    missing_score_reasons
                        .and_then(|reasons| reasons.get(&track_id))
                        .copied()
                        .unwrap_or(DecisionTraceMissingReason::NoLiveClassifiedTrack),
                ));
                continue;
            };
            match self.score_track(track_id, probability, frame_index, time_sec) {
                Ok((event, snapshot)) => {
                    snapshots.push(snapshot);
                    if let Some(event) = event {
                        emitted.push(event);
                    }
                }
                Err(cause) => return Err(self.fail(track_id, cause, emitted)),
            }
        }
        self.last_trace_snapshots = snapshots;
        self.last_update_evaluated = true;
        Ok(emitted)
    }

    pub fn coast(&mut self) -> Result<Vec<BusinessEvent>, FallError> {
        self.ensure_healthy()?;
        self.last_update_evaluated = false;
        Ok(Vec::new())
    }
    /// Like Python, exact emitted identity is the release key, not event metadata.
    pub fn release_onset(&mut self, event: &BusinessEvent) -> Result<bool, FallError> {
        self.ensure_healthy()?;
        Ok(self.episodes.release(&event.identity))
    }
    pub fn generation_for(&self, track_id: u64) -> Option<u64> {
        self.states.get(&track_id).map(|state| state.generation)
    }
    pub fn is_fallen(&self, track_id: u64) -> bool {
        self.states.get(&track_id).is_some_and(|state| state.fallen)
    }
    pub fn last_trace_snapshots(&self) -> &[DecisionTraceSnapshot] {
        &self.last_trace_snapshots
    }
    pub fn last_update_evaluated(&self) -> bool {
        self.last_update_evaluated
    }
    pub fn track_id_switch_absorbed_total(&self) -> u64 {
        self.episodes.track_id_switch_absorbed_total()
    }
    pub fn policy(&self) -> &FallPolicy {
        &self.policy
    }

    fn episode_state(&self, track_id: u64) -> EpisodeState {
        self.episodes
            .state_for(&self.camera_id, "fall", None, track_id)
    }
    fn ensure_healthy(&self) -> Result<(), FallError> {
        match self.fatal_failure {
            Some(cause) => Err(FallError::Poisoned(cause)),
            None => Ok(()),
        }
    }
    fn fail(
        &mut self,
        track_id: u64,
        cause: FallFailure,
        emitted_events: Vec<BusinessEvent>,
    ) -> FallError {
        self.fatal_failure = Some(cause);
        self.last_update_evaluated = false;
        FallError::FatalPartialState {
            track_id,
            cause,
            emitted_events,
        }
    }
}

// Separate admission and scoring stages while keeping a single lifecycle owner.
mod admission {
    use super::*;

    impl FallPolicyDecider {
        pub(super) fn preflight(
            &self,
            live: &BTreeSet<u64>,
            frame_index: i64,
            probabilities: &BTreeMap<u64, FallProbabilities>,
        ) -> Result<(), FallFailure> {
            let retained = self
                .states
                .iter()
                .filter(|(id, state)| {
                    live.contains(*id)
                        || !state.stale(frame_index, self.policy.parameters().track_ttl_frames)
                })
                .count();
            let new_states = live
                .iter()
                .filter(|&&id| probabilities.contains_key(&id) && !self.states.contains_key(&id))
                .count();
            if new_states > self.capacities.retained_tracks - retained {
                return Err(FallFailure::Capacity(FallCapacity::TrackStates));
            }
            let new_identities = live
                .iter()
                .filter(|&&id| {
                    probabilities.contains_key(&id) && !self.next_generations.contains_key(&id)
                })
                .count();
            if new_identities > self.capacities.generation_identities - self.next_generations.len()
            {
                return Err(FallFailure::Capacity(FallCapacity::GenerationIdentities));
            }
            for track_id in live {
                let Some(&probability) = probabilities.get(track_id) else {
                    continue;
                };
                if let Some(state) = self.states.get(track_id) {
                    state.scored(probability, self.policy.parameters())?;
                } else {
                    self.next_generations
                        .get(track_id)
                        .copied()
                        .unwrap_or(0)
                        .checked_add(1)
                        .ok_or(FallFailure::Overflow(FallCounter::Generation))?;
                }
            }
            Ok(())
        }

        pub(super) fn state_for(
            &mut self,
            track_id: u64,
            frame_index: i64,
        ) -> Result<&mut TrackState, FallFailure> {
            if !self.states.contains_key(&track_id) {
                if self.states.len() == self.capacities.retained_tracks {
                    return Err(FallFailure::Capacity(FallCapacity::TrackStates));
                }
                if !self.next_generations.contains_key(&track_id)
                    && self.next_generations.len() == self.capacities.generation_identities
                {
                    return Err(FallFailure::Capacity(FallCapacity::GenerationIdentities));
                }
                let generation = self.next_generations.get(&track_id).copied().unwrap_or(0);
                let next = generation
                    .checked_add(1)
                    .ok_or(FallFailure::Overflow(FallCounter::Generation))?;
                self.next_generations.insert(track_id, next);
                self.states
                    .insert(track_id, TrackState::new(generation, frame_index));
            }
            let state = self
                .states
                .get_mut(&track_id)
                .expect("state exists after admission");
            state.last_seen_frame = frame_index;
            Ok(state)
        }

        pub(super) fn evict_stale(
            &mut self,
            live: &BTreeSet<u64>,
            frame_index: i64,
            time_sec: f64,
        ) -> Result<(), (u64, EpisodeError)> {
            for &track_id in self.states.keys() {
                if !live.contains(&track_id) {
                    self.episodes
                        .track_lost(&self.camera_id, frame_index, time_sec, Some(track_id))
                        .map_err(|error| (track_id, error))?;
                }
            }
            let ttl = self.policy.parameters().track_ttl_frames;
            self.states.retain(|track_id, state| {
                live.contains(track_id) || !state.stale(frame_index, ttl)
            });
            Ok(())
        }
    }
}

mod scoring {
    use super::*;
    use crate::episode::{EpisodeProposal, suppression_reason};
    use crate::trace::{
        DecisionTraceReason, DecisionTraceState, DecisionTraceValueName, NumericTraceValue,
        TraceFloat,
    };

    impl FallPolicyDecider {
        pub(super) fn trace_state(&self, track_id: u64) -> DecisionTraceState {
            let Some(state) = self.states.get(&track_id) else {
                return DecisionTraceState::Unknown;
            };
            let episode = self.episode_state(track_id);
            if episode == EpisodeState::Open {
                DecisionTraceState::TransitionConfirmed
            } else if state.fallen {
                DecisionTraceState::Fallen
            } else if episode == EpisodeState::Candidate || state.has_transition() {
                DecisionTraceState::TransitionCandidate
            } else {
                DecisionTraceState::Clear
            }
        }

        pub(super) fn score_track(
            &mut self,
            track_id: u64,
            probability: FallProbabilities,
            frame_index: i64,
            time_sec: f64,
        ) -> Result<(Option<BusinessEvent>, DecisionTraceSnapshot), FallFailure> {
            let state = *self.state_for(track_id, frame_index)?;
            let previous = self.trace_state(track_id);
            let (next, recovery) = state.scored(probability, self.policy.parameters())?;
            let mut proposal = EpisodeProposal {
                camera_id: self.camera_id.clone(),
                facility_id: self.facility_id.clone(),
                event_type: "fall".into(),
                track_id,
                bed_id: None,
                frame_index,
                time_sec,
                qualifying: probability.fall_transition()
                    >= self.policy.parameters().transition_threshold,
                confirmed_recovery: false,
                probability: Some(probability.fall_transition()),
                domain: Some("fall".into()),
                generation: state.generation,
                confirmation_votes: self.policy.parameters().transition_votes,
                confirmation_window: self.policy.parameters().transition_window,
            };
            if !state.initialized {
                self.episodes.reassociate_fall(&proposal)?;
            }
            let event = match recovery {
                None => None,
                Some(confirmed_recovery) => {
                    proposal.confirmed_recovery = confirmed_recovery;
                    self.episodes.propose(&proposal)?
                }
            };
            self.states.insert(track_id, next);
            let snapshot =
                self.scored_snapshot(track_id, next, previous, probability, event.is_some());
            Ok((event, snapshot))
        }

        fn scored_snapshot(
            &self,
            track_id: u64,
            state: TrackState,
            previous: DecisionTraceState,
            probability: FallProbabilities,
            triggered: bool,
        ) -> DecisionTraceSnapshot {
            let suppressed = if state.proposed_this_call {
                // The authority owns the mapping. Only convert its static token to our enum.
                suppression_reason(self.episodes.last_disposition()).map(|token| {
                    DecisionTraceReason::from_token(token)
                        .expect("authority suppression is compiled vocabulary")
                })
            } else {
                None
            };
            let reason = if triggered {
                DecisionTraceReason::TransitionConfirmed
            } else if let Some(reason) = suppressed {
                reason
            } else if state.fallen && previous == DecisionTraceState::Fallen {
                DecisionTraceReason::FallActive
            } else if !state.fallen && previous == DecisionTraceState::Fallen {
                DecisionTraceReason::FallRecovered
            } else if state.has_transition() {
                DecisionTraceReason::TransitionCandidate
            } else {
                DecisionTraceReason::BelowThreshold
            };
            // Inputs and policy are immutable validated types. Trace rounding never
            // participates in threshold comparisons or event probabilities.
            let number = |value| {
                NumericTraceValue::Float(
                    TraceFloat::new(value).expect("validated policy/score is finite"),
                )
            };
            let policy = self.policy.parameters();
            DecisionTraceSnapshot::new(
                reason,
                (previous, self.trace_state(track_id)),
                triggered,
                Some(track_id),
                None,
                BTreeMap::from([
                    (
                        DecisionTraceValueName::FallTransitionProbability,
                        number(probability.fall_transition()),
                    ),
                    (
                        DecisionTraceValueName::FallenProbability,
                        number(probability.fallen()),
                    ),
                    (
                        DecisionTraceValueName::TransitionThreshold,
                        number(policy.transition_threshold),
                    ),
                    (
                        DecisionTraceValueName::TransitionVotes,
                        NumericTraceValue::Integer(policy.transition_votes),
                    ),
                    (
                        DecisionTraceValueName::TransitionWindow,
                        NumericTraceValue::Integer(policy.transition_window),
                    ),
                ]),
                BTreeMap::new(),
            )
            .expect("scored trace has no missing values to overlap")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::{
        DecisionTraceReason as Reason, DecisionTraceState as State,
        DecisionTraceValueName as ValueName, NumericTraceValue,
    };

    fn capacities() -> FallCapacities {
        FallCapacities {
            retained_tracks: 8,
            generation_identities: 16,
            episodes: 16,
            vote_window: 8,
        }
    }
    fn decider(parameters: FallPolicyParameters) -> FallPolicyDecider {
        FallPolicyDecider::new(
            "camera",
            "facility",
            "boot",
            "epoch",
            7,
            FallPolicy::new(parameters).unwrap(),
            capacities(),
        )
        .unwrap()
    }
    fn probability(transition: f64, fallen: f64) -> FallProbabilities {
        FallProbabilities::new(0.0, transition, fallen).unwrap()
    }
    fn update(
        d: &mut FallPolicyDecider,
        track: u64,
        frame: i64,
        transition: f64,
        fallen: f64,
    ) -> Vec<BusinessEvent> {
        d.update(
            frame,
            frame as f64,
            &BTreeMap::from([(track, probability(transition, fallen))]),
            [track],
            None,
        )
        .unwrap()
    }
    fn reason(d: &FallPolicyDecider) -> Reason {
        d.last_trace_snapshots()[0].reason
    }

    mod scored_branches {
        use super::*;

        #[test]
        fn initialized_fallen_has_no_onset_even_when_transition_qualifies() {
            for (fallen, initially_fallen) in
                [(0.799999999, false), (0.8, true), (0.800000001, true)]
            {
                let mut d = decider(FallPolicyParameters::default());
                assert!(update(&mut d, 7, 0, 0.9, fallen).is_empty());
                assert_eq!(d.is_fallen(7), initially_fallen);
                let state = d.states[&7];
                assert!(state.initialized);
                assert_eq!(state.proposed_this_call, !initially_fallen);
                assert_eq!(state.fallen_streak, 0);
                assert_eq!(d.last_trace_snapshots()[0].previous_state, State::Clear);
                if initially_fallen {
                    assert_eq!(d.episode_state(7), EpisodeState::Normal);
                    assert_eq!(reason(&d), Reason::BelowThreshold);
                    assert_eq!(d.last_trace_snapshots()[0].current_state, State::Fallen);
                    assert!(update(&mut d, 7, 1, 0.0, 0.8).is_empty());
                    assert_eq!(reason(&d), Reason::FallActive);
                    assert_eq!(d.states[&7].fallen_streak, 1);
                } else {
                    assert_eq!(d.episode_state(7), EpisodeState::Candidate);
                    assert_eq!(reason(&d), Reason::EpisodeCandidate);
                }
            }
        }

        #[test]
        fn transition_threshold_is_inclusive_and_uses_unrounded_score() {
            for (score, qualifies) in [(0.499999999, false), (0.5, true), (0.500000001, true)] {
                let mut d = decider(FallPolicyParameters {
                    transition_votes: 1,
                    ..Default::default()
                });
                let events = update(&mut d, 7, 0, score, 0.0);
                assert_eq!(events.len(), usize::from(qualifies));
                assert_eq!(d.states[&7].proposed_this_call, qualifies);
                let snapshot = &d.last_trace_snapshots()[0];
                assert_eq!(snapshot.triggered, qualifies);
                assert_eq!(snapshot.values().len(), 5);
                assert_eq!(
                    snapshot.values()[&ValueName::TransitionVotes],
                    NumericTraceValue::Integer(1)
                );
                assert_eq!(
                    snapshot.values()[&ValueName::TransitionWindow],
                    NumericTraceValue::Integer(5)
                );
                let NumericTraceValue::Float(traced) =
                    snapshot.values()[&ValueName::FallTransitionProbability]
                else {
                    panic!("probability must remain a float");
                };
                assert_eq!(traced.get(), 0.5);
                if qualifies {
                    assert_eq!(events[0].probability, Some(score));
                    assert_eq!(events[0].identity, "boot:epoch:fall:none:7:7:0:1");
                } else {
                    assert_eq!(reason(&d), Reason::BelowThreshold);
                }
            }
        }

        #[test]
        fn score_gaps_coast_but_nonqualifying_scores_reset_episode_votes() {
            let mut d = decider(FallPolicyParameters::default());
            assert!(update(&mut d, 7, 0, 0.7, 0.0).is_empty());
            assert!(
                d.update(1, 1.0, &BTreeMap::new(), [7], None)
                    .unwrap()
                    .is_empty()
            );
            assert!(update(&mut d, 7, 2, 0.7, 0.0).is_empty());
            // Nonlive scores are ignored, not a false vote or an empty-room reset.
            assert!(
                d.update(
                    3,
                    3.0,
                    &BTreeMap::from([(7, probability(0.0, 0.0))]),
                    [],
                    None
                )
                .unwrap()
                .is_empty()
            );
            assert_eq!(update(&mut d, 7, 4, 0.7, 0.0).len(), 1);
            assert_eq!(reason(&d), Reason::TransitionConfirmed);
            assert!(update(&mut d, 7, 5, 0.7, 0.0).is_empty());
            assert_eq!(reason(&d), Reason::EpisodeAlreadyOpen);
            assert!(update(&mut d, 7, 6, 0.1, 0.0).is_empty());
            assert!(!d.states[&7].proposed_this_call);
            assert_eq!(reason(&d), Reason::TransitionCandidate);
            assert_eq!(
                d.last_trace_snapshots()[0].current_state,
                State::TransitionConfirmed
            );

            let mut d = decider(FallPolicyParameters::default());
            for (frame, score) in [(0, 0.7), (1, 0.7), (2, 0.49), (3, 0.7), (4, 0.7)] {
                assert!(update(&mut d, 7, frame, score, 0.0).is_empty());
            }
            assert_eq!(update(&mut d, 7, 5, 0.7, 0.0).len(), 1);
        }

        #[test]
        fn local_trace_history_is_five_entries_not_configured_episode_window() {
            let mut d = decider(FallPolicyParameters {
                transition_votes: 2,
                transition_window: 2,
                ..Default::default()
            });
            assert!(update(&mut d, 7, 0, 0.7, 0.6).is_empty());
            for frame in 1..5 {
                assert!(update(&mut d, 7, frame, 0.49, 0.6).is_empty());
                assert_eq!(d.episode_state(7), EpisodeState::Normal);
                assert_eq!(reason(&d), Reason::TransitionCandidate);
            }
            update(&mut d, 7, 5, 0.49, 0.6);
            assert_eq!(reason(&d), Reason::BelowThreshold);
            assert_eq!(d.last_trace_snapshots()[0].current_state, State::Clear);
        }

        #[test]
        fn fallen_streak_is_consecutive_and_threshold_inclusive() {
            for (fallen, qualifies) in [(0.799999999, false), (0.8, true), (0.800000001, true)] {
                let mut d = decider(FallPolicyParameters {
                    fallen_consecutive: 2,
                    ..Default::default()
                });
                update(&mut d, 7, 0, 0.0, 0.0);
                update(&mut d, 7, 1, 0.0, fallen);
                assert!(!d.is_fallen(7));
                update(&mut d, 7, 2, 0.0, 0.79);
                assert_eq!(d.states[&7].fallen_streak, 0);
                update(&mut d, 7, 3, 0.0, fallen);
                update(&mut d, 7, 4, 0.0, fallen);
                assert_eq!(d.is_fallen(7), qualifies);
            }
        }

        #[test]
        fn recovery_requires_joint_strict_thresholds_then_immediately_rearms() {
            for (transition, fallen, recovering) in [
                (0.399999999, 0.499999999, true),
                (0.4, 0.49, false),
                (0.400000001, 0.49, false),
                (0.39, 0.5, false),
                (0.39, 0.500000001, false),
            ] {
                let mut d = decider(FallPolicyParameters {
                    recovery_consecutive: 2,
                    ..Default::default()
                });
                update(&mut d, 7, 0, 0.0, 0.8);
                update(&mut d, 7, 1, 0.0, 0.0);
                update(&mut d, 7, 2, transition, fallen);
                assert_eq!(d.is_fallen(7), !recovering);
                assert_eq!(d.states[&7].recovery_streak, if recovering { 2 } else { 0 });
                if recovering {
                    assert_eq!(reason(&d), Reason::FallRecovered);
                    assert_eq!(d.states[&7].fallen_streak, 0);
                    assert!(!d.states[&7].has_transition());
                }
            }
            let mut d = decider(FallPolicyParameters {
                transition_votes: 1,
                recovery_consecutive: 2,
                ..Default::default()
            });
            let first = update(&mut d, 7, 0, 0.7, 0.0).remove(0);
            for (frame, transition, fallen) in [(1, 0.0, 0.0), (2, 0.4, 0.0), (3, 0.0, 0.0)] {
                update(&mut d, 7, frame, transition, fallen);
                assert_eq!(d.episode_state(7), EpisodeState::Open);
            }
            update(&mut d, 7, 4, 0.0, 0.0);
            assert_eq!(d.episode_state(7), EpisodeState::Normal);
            // cooldown_frames=90 is retained config, NOT an extra alert gate in source.
            let second = update(&mut d, 7, 5, 0.7, 0.0).remove(0);
            assert_eq!(second.identity, "boot:epoch:fall:none:7:7:0:2");
            assert_ne!(first.identity, second.identity);
        }

        #[test]
        fn qualifying_recovery_keeps_authority_precedence_and_no_suppression_claim() {
            let mut d = decider(FallPolicyParameters {
                transition_threshold: 0.1,
                transition_votes: 1,
                recovery_consecutive: 1,
                ..Default::default()
            });
            // Overlapping thresholds are legal: NORMAL can emit despite recovery.
            assert_eq!(update(&mut d, 7, 0, 0.2, 0.0).len(), 1);
            assert!(!d.states[&7].proposed_this_call);
            assert_eq!(reason(&d), Reason::TransitionConfirmed);
            assert!(update(&mut d, 7, 1, 0.2, 0.0).is_empty());
            assert_eq!(reason(&d), Reason::BelowThreshold);
            assert_eq!(d.episode_state(7), EpisodeState::Normal);
        }

        #[test]
        fn simultaneous_events_and_traces_are_sorted_without_camera_or_reduction() {
            let mut d = decider(FallPolicyParameters {
                transition_votes: 1,
                ..Default::default()
            });
            let scores = BTreeMap::from([
                (9, probability(0.9, 0.0)),
                (2, probability(0.8, 0.0)),
                (99, probability(1.0, 0.0)),
            ]);
            let events = d.update(0, 0.0, &scores, [9, 2, 9, 4], None).unwrap();
            assert_eq!(
                events.iter().map(|e| e.person_id).collect::<Vec<_>>(),
                [Some(2), Some(9)]
            );
            assert_eq!(events[0].identity, "boot:epoch:fall:none:2:7:0:1");
            assert_eq!(events[1].identity, "boot:epoch:fall:none:9:7:0:2");
            assert_eq!(
                d.last_trace_snapshots()
                    .iter()
                    .map(|s| s.track_id)
                    .collect::<Vec<_>>(),
                [Some(2), Some(4), Some(9)]
            );
            assert_eq!(d.last_trace_snapshots()[1].current_state, State::Unknown);
            assert_eq!(d.generation_for(4), None);
            assert_eq!(d.generation_for(99), None);
        }
    }

    mod missing_and_identity {
        use super::*;

        #[test]
        fn pose_unavailable_unknown_track_keeps_reason_without_generation_or_event() {
            let mut d = decider(FallPolicyParameters::default());
            let missing = DecisionTraceMissingReason::from_token("pose-unavailable").unwrap();
            let reasons = BTreeMap::from([(7, missing)]);
            let events = d
                .update(0, 0.0, &BTreeMap::new(), [7], Some(&reasons))
                .unwrap();
            assert!(events.is_empty());
            assert!(d.last_update_evaluated());
            assert_eq!(d.last_trace_snapshots().len(), 1);
            let snapshot = &d.last_trace_snapshots()[0];
            assert_eq!(snapshot.reason, Reason::ScoreMissing);
            assert_eq!(snapshot.previous_state, State::Unknown);
            assert_eq!(snapshot.current_state, State::Unknown);
            assert!(!snapshot.triggered);
            assert_eq!(snapshot.track_id, Some(7));
            assert_eq!(snapshot.bed_id, None);
            assert!(snapshot.values().is_empty());
            assert_eq!(
                snapshot.missing_values(),
                &BTreeMap::from([(
                    ValueName::FallTransitionProbability,
                    DecisionTraceMissingReason::PoseUnavailable,
                )])
            );
            assert_eq!(d.generation_for(7), None);
            assert!(d.states.is_empty());
            assert!(d.next_generations.is_empty());
        }

        #[test]
        fn missing_reasons_are_current_call_only_and_never_create_or_reuse_a_score() {
            let mut d = decider(FallPolicyParameters::default());
            update(&mut d, 7, 0, 0.7, 0.0);
            let before = d.states[&7];
            for (frame, missing) in [
                (1, Some(DecisionTraceMissingReason::ClassifierWarmup)),
                (2, Some(DecisionTraceMissingReason::ClassifierStrideNotDue)),
                (3, Some(DecisionTraceMissingReason::ResampleGap)),
                (4, None),
            ] {
                let reasons = missing.map(|reason| BTreeMap::from([(7, reason)]));
                assert!(
                    d.update(
                        frame,
                        frame as f64,
                        &BTreeMap::new(),
                        [7, 8],
                        reasons.as_ref()
                    )
                    .unwrap()
                    .is_empty()
                );
                assert!(d.last_update_evaluated());
                let snapshot = &d.last_trace_snapshots()[0];
                assert_eq!(snapshot.reason, Reason::ScoreMissing);
                assert_eq!(snapshot.previous_state, State::TransitionCandidate);
                assert_eq!(snapshot.current_state, State::TransitionCandidate);
                assert!(snapshot.values().is_empty());
                assert_eq!(
                    snapshot.missing_values()[&ValueName::FallTransitionProbability],
                    missing.unwrap_or(DecisionTraceMissingReason::NoLiveClassifiedTrack)
                );
                assert_eq!(
                    d.states[&7],
                    TrackState {
                        last_seen_frame: frame,
                        ..before
                    }
                );
                assert!(d.states[&7].proposed_this_call); // Missing does not advance/reset this flag.
                assert_eq!(d.generation_for(8), None);
                assert!(!d.next_generations.contains_key(&8));
            }
            let snapshots = d.last_trace_snapshots().to_vec();
            let states = d.states.clone();
            assert!(d.coast().unwrap().is_empty());
            assert!(!d.last_update_evaluated());
            assert_eq!(d.last_trace_snapshots(), snapshots);
            assert_eq!(d.states, states);
            assert_eq!(d.episode_state(7), EpisodeState::Candidate);
            assert!(update(&mut d, 7, 5, 0.7, 0.0).is_empty());
            assert_eq!(update(&mut d, 7, 6, 0.7, 0.0).len(), 1);
            let reasons = BTreeMap::from([(7, DecisionTraceMissingReason::ClassifierWarmup)]);
            let open_state = d.states[&7];
            assert!(
                d.update(7, 7.0, &BTreeMap::new(), [7], Some(&reasons))
                    .unwrap()
                    .is_empty()
            );
            assert_eq!(reason(&d), Reason::ScoreMissing);
            assert_eq!(
                d.last_trace_snapshots()[0].current_state,
                State::TransitionConfirmed
            );
            assert_eq!(d.episode_state(7), EpisodeState::Open);
            assert_eq!(
                d.states[&7],
                TrackState {
                    last_seen_frame: 7,
                    ..open_state
                }
            );
            d.update(
                8,
                8.0,
                &BTreeMap::from([(7, probability(0.9, 0.0))]),
                [7],
                Some(&reasons),
            )
            .unwrap();
            assert_eq!(reason(&d), Reason::EpisodeAlreadyOpen);
            assert!(d.last_trace_snapshots()[0].missing_values().is_empty());
        }

        #[test]
        fn missing_scores_coast_both_streaks_and_refresh_liveness() {
            let mut d = decider(FallPolicyParameters {
                fallen_consecutive: 2,
                recovery_consecutive: 2,
                ..Default::default()
            });
            update(&mut d, 7, 0, 0.0, 0.6);
            update(&mut d, 7, 1, 0.0, 0.8);
            d.update(1000, 1000.0, &BTreeMap::new(), [7], None).unwrap();
            assert_eq!(d.generation_for(7), Some(0));
            assert_eq!(d.states[&7].fallen_streak, 1);
            update(&mut d, 7, 1001, 0.0, 0.8);
            assert!(d.is_fallen(7));
            update(&mut d, 7, 1002, 0.0, 0.0);
            d.update(1003, 1003.0, &BTreeMap::new(), [7], None).unwrap();
            assert_eq!(d.states[&7].recovery_streak, 1);
            assert!(d.is_fallen(7));
            update(&mut d, 7, 1004, 0.0, 0.0);
            assert_eq!(reason(&d), Reason::FallRecovered);
            assert!(!d.is_fallen(7));
            d.update(1048, 1048.0, &BTreeMap::new(), [], None).unwrap();
            assert_eq!(d.generation_for(7), Some(0));
            d.update(1049, 1049.0, &BTreeMap::new(), [], None).unwrap();
            assert_eq!(d.generation_for(7), None);
        }

        #[test]
        fn ttl_below_equal_above_and_generation_reuse_do_not_erase_identity_history() {
            for (absent_frame, expired) in [(44, false), (45, true), (46, true)] {
                let mut d = decider(FallPolicyParameters {
                    transition_votes: 1,
                    ..Default::default()
                });
                update(&mut d, 7, 0, 0.0, 0.0);
                d.update(
                    absent_frame,
                    absent_frame as f64,
                    &BTreeMap::new(),
                    [],
                    None,
                )
                .unwrap();
                assert_eq!(d.generation_for(7), if expired { None } else { Some(0) });
                assert_eq!(d.next_generations[&7], 1);
                let event = update(&mut d, 7, absent_frame + 1, 0.7, 0.0).remove(0);
                assert_eq!(d.generation_for(7), Some(u64::from(expired)));
                assert_eq!(
                    event.identity,
                    format!("boot:epoch:fall:none:7:7:{}:1", u64::from(expired))
                );
            }
            let mut d = decider(FallPolicyParameters::default());
            for frame in 0..3 {
                update(&mut d, 7, frame, 0.7, 0.0);
            }
            d.update(47, 47.0, &BTreeMap::new(), [], None).unwrap();
            // New generation does not bypass the existing accepted episode's recovery.
            for frame in 48..53 {
                assert!(update(&mut d, 7, frame, 0.1, 0.1).is_empty());
            }
            assert_eq!(d.generation_for(7), Some(1));
            assert_eq!(d.episode_state(7), EpisodeState::Normal);
            assert!(update(&mut d, 7, 53, 0.7, 0.0).is_empty());
            assert!(update(&mut d, 7, 54, 0.7, 0.0).is_empty());
            assert_eq!(
                update(&mut d, 7, 55, 0.7, 0.0)[0].identity,
                "boot:epoch:fall:none:7:7:1:2"
            );
        }

        #[test]
        fn failed_staging_release_uses_exact_identity_once_and_never_rewinds_sequence() {
            let mut d = decider(FallPolicyParameters::default());
            for frame in 0..2 {
                update(&mut d, 7, frame, 0.7, 0.0);
            }
            let event = update(&mut d, 7, 2, 0.7, 0.0).remove(0);
            let mut foreign = event.clone();
            foreign.identity.push('x');
            assert!(!d.release_onset(&foreign).unwrap());
            assert_eq!(d.episode_state(7), EpisodeState::Open);
            let traces = d.last_trace_snapshots().to_vec();
            let mut same_identity = event.clone();
            same_identity.camera_id = "other-camera".into();
            same_identity.person_id = Some(99);
            assert!(d.release_onset(&same_identity).unwrap());
            assert_eq!(d.last_trace_snapshots(), traces);
            assert!(d.last_update_evaluated());
            assert!(update(&mut d, 7, 3, 0.7, 0.0).is_empty());
            assert!(!d.release_onset(&event).unwrap());
            assert!(update(&mut d, 7, 4, 0.7, 0.0).is_empty());
            let retried = update(&mut d, 7, 5, 0.7, 0.0).remove(0);
            assert_eq!(retried.identity, "boot:epoch:fall:none:7:7:0:2");
            assert!(!d.release_onset(&event).unwrap());
            assert_eq!(d.episode_state(7), EpisodeState::Open);
        }
    }

    mod reassociation {
        use super::*;

        #[test]
        fn new_id_is_absorbed_immediately_and_initial_fallen_still_reassociates() {
            for fallen in [0.0, 0.8] {
                let mut d = decider(FallPolicyParameters {
                    transition_votes: 1,
                    ..Default::default()
                });
                update(&mut d, 7, 0, 0.7, 0.0);
                assert!(update(&mut d, 8, 1, 0.7, fallen).is_empty());
                assert_eq!(d.generation_for(7), Some(0)); // TTL has not expired.
                assert_eq!(d.track_id_switch_absorbed_total(), 1);
                assert_eq!(d.episode_state(7), EpisodeState::Normal);
                assert_eq!(d.episode_state(8), EpisodeState::Open);
                assert_eq!(d.last_trace_snapshots()[0].previous_state, State::Clear);
                assert_eq!(
                    d.last_trace_snapshots()[0].current_state,
                    State::TransitionConfirmed
                );
                if fallen == 0.8 {
                    assert!(d.is_fallen(8));
                    assert!(!d.states[&8].proposed_this_call);
                    assert_eq!(reason(&d), Reason::BelowThreshold);
                } else {
                    // reassociate_fall precedes propose, which now sees OPEN.
                    assert_eq!(reason(&d), Reason::EpisodeAlreadyOpen);
                }
            }
        }

        #[test]
        fn existing_id_return_maps_reassociation_and_resolved_hold_without_reimplementing_timers() {
            for (frame, time, expected) in [
                (75, 5.999, Reason::EpisodeReassociated),
                (76, 6.0, Reason::EpisodeReassociated),
                (77, 6.0, Reason::EpisodeResolvedHold),
                (76, 6.000001, Reason::EpisodeResolvedHold),
                (-100, -100.0, Reason::EpisodeReassociated),
            ] {
                let mut d = decider(FallPolicyParameters {
                    transition_votes: 1,
                    track_ttl_frames: 1000,
                    recovery_consecutive: 1,
                    ..Default::default()
                });
                update(&mut d, 7, 0, 0.7, 0.0);
                d.update(1, 1.0, &BTreeMap::new(), [], None).unwrap();
                assert_eq!(d.episode_state(7), EpisodeState::Unknown);
                assert!(
                    d.update(
                        frame,
                        time,
                        &BTreeMap::from([(7, probability(0.7, 0.0))]),
                        [7],
                        None
                    )
                    .unwrap()
                    .is_empty()
                );
                assert_eq!(reason(&d), expected);
                assert_eq!(d.track_id_switch_absorbed_total(), 0);
                assert_eq!(d.generation_for(7), Some(0));
                update(&mut d, 7, frame + 1, 0.0, 0.0);
                assert_eq!(d.episode_state(7), EpisodeState::Normal);
                assert_eq!(
                    update(&mut d, 7, frame + 2, 0.7, 0.0)[0].identity,
                    "boot:epoch:fall:none:7:7:0:2"
                );
            }
        }

        #[test]
        fn reassociation_selects_numeric_lowest_unknown_and_release_follows_moved_identity() {
            let mut d = decider(FallPolicyParameters {
                transition_votes: 1,
                ..Default::default()
            });
            let events = d
                .update(
                    0,
                    0.0,
                    &BTreeMap::from([(9, probability(0.8, 0.0)), (2, probability(0.7, 0.0))]),
                    [9, 2],
                    None,
                )
                .unwrap();
            assert!(update(&mut d, 8, 1, 0.7, 0.0).is_empty());
            assert_eq!(d.episode_state(2), EpisodeState::Normal);
            assert_eq!(d.episode_state(9), EpisodeState::Unknown);
            assert_eq!(d.episode_state(8), EpisodeState::Open);
            assert!(d.release_onset(&events[0]).unwrap());
            assert_eq!(d.episode_state(8), EpisodeState::Normal);
            assert!(!d.release_onset(&events[0]).unwrap());
            assert!(d.release_onset(&events[1]).unwrap());
            assert_eq!(d.episode_state(9), EpisodeState::Normal);
        }
    }

    mod validation {
        use super::*;

        #[test]
        fn probabilities_validate_each_class_without_normalization_or_model_evidence() {
            for (index, name) in ["background", "fall_transition", "fallen"]
                .into_iter()
                .enumerate()
            {
                for invalid in [
                    f64::NAN,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    -f64::EPSILON,
                    1.0 + f64::EPSILON,
                ] {
                    let mut values = [0.0, 0.5, 1.0];
                    values[index] = invalid;
                    assert_eq!(
                        FallProbabilities::new(values[0], values[1], values[2]),
                        Err(FallFailure::InvalidProbability(name))
                    );
                }
            }
            let non_normalized = FallProbabilities::new(1.0, 1.0, 1.0).unwrap();
            assert_eq!(non_normalized.background(), 1.0);
            assert_eq!(non_normalized.fall_transition(), 1.0);
            assert_eq!(non_normalized.fallen(), 1.0);
            let mut d = decider(FallPolicyParameters {
                transition_votes: 1,
                ..Default::default()
            });
            assert!(
                d.update(0, 0.0, &BTreeMap::from([(7, non_normalized)]), [7], None)
                    .unwrap()
                    .is_empty()
            );
            assert!(d.is_fallen(7)); // Valid three-class inputs are not forced into a simplex.
            assert!(
                FallProbabilities::new(-0.0, 0.0, 0.0)
                    .unwrap()
                    .background()
                    .is_sign_negative()
            );
        }

        #[test]
        fn every_policy_field_is_validated_and_confirmation_is_cross_checked() {
            for (index, name) in [
                "transition_threshold",
                "fallen_threshold",
                "recovery_transition_max",
                "recovery_fallen_max",
            ]
            .into_iter()
            .enumerate()
            {
                for value in [
                    f64::NAN,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    -0.01,
                    1.01,
                    0.0,
                    1.0,
                ] {
                    let mut p = FallPolicyParameters::default();
                    match index {
                        0 => p.transition_threshold = value,
                        1 => p.fallen_threshold = value,
                        2 => p.recovery_transition_max = value,
                        _ => p.recovery_fallen_max = value,
                    }
                    if value == 0.0 || value == 1.0 {
                        assert!(FallPolicy::new(p).is_ok());
                    } else {
                        assert_eq!(FallPolicy::new(p), Err(FallFailure::InvalidPolicy(name)));
                    }
                }
            }
            for (index, name) in [
                "transition_votes",
                "transition_window",
                "fallen_consecutive",
                "recovery_consecutive",
                "track_ttl_frames",
                "cooldown_frames",
            ]
            .into_iter()
            .enumerate()
            {
                let mut p = FallPolicyParameters::default();
                match index {
                    0 => p.transition_votes = 0,
                    1 => p.transition_window = 0,
                    2 => p.fallen_consecutive = 0,
                    3 => p.recovery_consecutive = 0,
                    4 => p.track_ttl_frames = 0,
                    _ => p.cooldown_frames = 0,
                }
                assert_eq!(FallPolicy::new(p), Err(FallFailure::InvalidPolicy(name)));
            }
            assert_eq!(
                FallPolicy::new(FallPolicyParameters {
                    transition_votes: 6,
                    transition_window: 5,
                    ..Default::default()
                }),
                Err(FallFailure::InvalidPolicy("transition_votes"))
            );
            let mut d = decider(FallPolicyParameters {
                transition_votes: 6,
                transition_window: 6,
                ..Default::default()
            });
            for frame in 0..5 {
                assert!(update(&mut d, 7, frame, 0.7, 0.0).is_empty());
            }
            // Confirmation can exceed five: the local trace history is not a vote owner.
            assert_eq!(update(&mut d, 7, 5, 0.7, 0.0).len(), 1);
        }

        #[test]
        fn capacities_are_explicit_and_bound_both_windows_without_changing_policy() {
            for index in 0..4 {
                let mut limits = capacities();
                match index {
                    0 => limits.retained_tracks = 0,
                    1 => limits.generation_identities = 0,
                    2 => limits.episodes = 0,
                    _ => limits.vote_window = 0,
                }
                assert_eq!(
                    FallPolicyDecider::new("c", "f", "b", "e", 0, FallPolicy::default(), limits)
                        .unwrap_err(),
                    FallFailure::InvalidCapacities
                );
            }
            for (window, budget) in [(2, 4), (6, 5)] {
                let policy = FallPolicy::new(FallPolicyParameters {
                    transition_votes: 1,
                    transition_window: window,
                    ..Default::default()
                })
                .unwrap();
                assert_eq!(
                    FallPolicyDecider::new(
                        "c",
                        "f",
                        "b",
                        "e",
                        0,
                        policy,
                        FallCapacities {
                            vote_window: budget,
                            ..capacities()
                        }
                    )
                    .unwrap_err(),
                    FallFailure::Capacity(FallCapacity::VoteWindow)
                );
            }
        }

        #[test]
        fn identities_are_bounded_in_bytes_not_characters_and_bind_events() {
            let at_limit = "é".repeat(128);
            let too_long = "é".repeat(129);
            let policy = FallPolicy::new(FallPolicyParameters {
                transition_votes: 1,
                ..Default::default()
            })
            .unwrap();
            for (camera, facility) in [(&too_long, &at_limit), (&at_limit, &too_long)] {
                assert_eq!(
                    FallPolicyDecider::new(
                        camera.clone(),
                        facility.clone(),
                        "boot",
                        "epoch",
                        0,
                        policy,
                        capacities()
                    )
                    .unwrap_err(),
                    FallFailure::InvalidIdentity
                );
            }
            for (boot, epoch) in [
                ("", "epoch"),
                ("boot", ""),
                (too_long.as_str(), "epoch"),
                ("boot", too_long.as_str()),
            ] {
                assert_eq!(
                    FallPolicyDecider::new("c", "f", boot, epoch, 0, policy, capacities())
                        .unwrap_err(),
                    FallFailure::Episode(EpisodeError::InvalidIdentity)
                );
            }
            let mut d = FallPolicyDecider::new(
                at_limit.clone(),
                at_limit.clone(),
                "boot-2",
                "epoch-2",
                u64::MAX,
                policy,
                capacities(),
            )
            .unwrap();
            let event = update(&mut d, 7, -1, 1.0, 0.0).remove(0);
            assert_eq!(event.camera_id, at_limit);
            assert_eq!(event.facility_id, at_limit);
            assert_eq!(
                event.identity,
                format!("boot-2:epoch-2:fall:none:7:{}:0:1", u64::MAX)
            );
            assert_eq!(event.time_sec, -1.0);
            assert_eq!(event.domain, "fall");
        }

        #[test]
        fn nonfinite_time_rejects_even_empty_calls_without_staling_prior_evidence() {
            let mut d = decider(FallPolicyParameters::default());
            update(&mut d, 7, 0, 0.7, 0.0);
            let states = d.states.clone();
            let traces = d.last_trace_snapshots().to_vec();
            for time in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                assert_eq!(
                    d.update(45, time, &BTreeMap::new(), [], None),
                    Err(FallError::Rejected(FallFailure::NonFiniteTime))
                );
                assert_eq!(d.states, states);
                assert_eq!(d.last_trace_snapshots(), traces);
                assert!(d.last_update_evaluated());
                assert_eq!(d.episode_state(7), EpisodeState::Candidate);
            }
        }
    }

    mod bounded_failures {
        use super::*;

        #[test]
        fn capacity_rejection_precedes_track_loss_and_never_lru_evicts_generations() {
            let mut d = FallPolicyDecider::new(
                "camera",
                "facility",
                "boot",
                "epoch",
                7,
                FallPolicy::new(FallPolicyParameters {
                    transition_votes: 1,
                    ..Default::default()
                })
                .unwrap(),
                FallCapacities {
                    retained_tracks: 1,
                    generation_identities: 1,
                    ..capacities()
                },
            )
            .unwrap();
            update(&mut d, 7, 0, 0.7, 0.0);
            let scores = BTreeMap::from([(8, probability(0.7, 0.0))]);
            assert_eq!(
                d.update(1, 1.0, &scores, [8], None),
                Err(FallError::Rejected(FallFailure::Capacity(
                    FallCapacity::TrackStates
                )))
            );
            assert_eq!(d.episode_state(7), EpisodeState::Open);
            assert_eq!(d.states[&7].last_seen_frame, 0);
            // Expiry would free a state slot, but cannot free an accepted generation ID.
            assert_eq!(
                d.update(45, 45.0, &scores, [8], None),
                Err(FallError::Rejected(FallFailure::Capacity(
                    FallCapacity::GenerationIdentities
                )))
            );
            assert_eq!(d.episode_state(7), EpisodeState::Open);
            assert_eq!(d.generation_for(7), Some(0));
            assert_eq!(d.generation_for(8), None);
            d.update(45, 45.0, &BTreeMap::new(), [], None).unwrap();
            assert_eq!(d.generation_for(7), None);
            update(&mut d, 7, 46, 0.0, 0.0);
            assert_eq!(d.generation_for(7), Some(1));
            assert_eq!(d.next_generations.len(), 1);
            let before = d.states.clone();
            assert_eq!(
                d.update(47, 47.0, &BTreeMap::new(), [7, 7, 8], None),
                Err(FallError::Rejected(FallFailure::Capacity(
                    FallCapacity::LiveTracks
                )))
            );
            assert_eq!(d.states, before);
            d.update(47, 47.0, &BTreeMap::new(), [7, 7], None).unwrap();
            assert_eq!(d.last_trace_snapshots().len(), 1);
        }

        #[test]
        fn ttl_subtraction_is_widened_and_rollback_never_silently_resets_state() {
            let mut d = decider(FallPolicyParameters {
                track_ttl_frames: u64::MAX,
                ..Default::default()
            });
            update(&mut d, 7, i64::MIN, 0.0, 0.0);
            d.update(i64::MAX - 1, 0.0, &BTreeMap::new(), [], None)
                .unwrap();
            assert_eq!(d.generation_for(7), Some(0));
            d.update(i64::MAX, 0.0, &BTreeMap::new(), [], None).unwrap();
            assert_eq!(d.generation_for(7), None);
            assert_eq!(d.next_generations[&7], 1);

            let mut d = decider(FallPolicyParameters {
                transition_votes: 1,
                ..Default::default()
            });
            update(&mut d, 7, i64::MAX, 0.7, 0.0);
            d.update(i64::MIN, -1.0, &BTreeMap::new(), [], None)
                .unwrap();
            assert_eq!(d.generation_for(7), Some(0));
            assert!(
                d.update(
                    i64::MIN + 1,
                    -2.0,
                    &BTreeMap::from([(7, probability(0.7, 0.0))]),
                    [7],
                    None
                )
                .unwrap()
                .is_empty()
            );
            assert_eq!(reason(&d), Reason::EpisodeReassociated);
            assert_eq!(d.generation_for(7), Some(0));
            assert_eq!(d.track_id_switch_absorbed_total(), 0);
        }

        #[test]
        fn checked_generation_and_streak_overflow_reject_the_entire_call() {
            let mut d = decider(FallPolicyParameters::default());
            d.next_generations.insert(7, u64::MAX);
            assert_eq!(
                d.update(
                    0,
                    0.0,
                    &BTreeMap::from([(7, probability(0.7, 0.0))]),
                    [7],
                    None
                ),
                Err(FallError::Rejected(FallFailure::Overflow(
                    FallCounter::Generation
                )))
            );
            assert!(d.states.is_empty());
            assert_eq!(d.next_generations[&7], u64::MAX);
            for (counter, transition, fallen) in [
                (FallCounter::FallenStreak, 0.7, 0.8),
                (FallCounter::RecoveryStreak, 0.0, 0.0),
            ] {
                let mut d = decider(FallPolicyParameters::default());
                update(&mut d, 7, 0, 0.7, 0.0);
                let state = d.states.get_mut(&7).unwrap();
                match counter {
                    FallCounter::FallenStreak => state.fallen_streak = u64::MAX,
                    FallCounter::RecoveryStreak => state.recovery_streak = u64::MAX,
                    FallCounter::Generation => unreachable!(),
                }
                let states = d.states.clone();
                let traces = d.last_trace_snapshots().to_vec();
                let scores = BTreeMap::from([
                    (1, probability(0.7, 0.0)),
                    (7, probability(transition, fallen)),
                ]);
                assert_eq!(
                    d.update(1, 1.0, &scores, [1, 7], None),
                    Err(FallError::Rejected(FallFailure::Overflow(counter)))
                );
                assert_eq!(d.states, states);
                assert_eq!(d.last_trace_snapshots(), traces);
                assert_eq!(d.episode_state(7), EpisodeState::Candidate);
                assert_eq!(d.episode_state(1), EpisodeState::Normal);
                assert!(!d.next_generations.contains_key(&1));
                assert!(d.last_update_evaluated());
            }
        }

        #[test]
        fn last_representable_generation_and_streak_increments_are_not_saturated_or_reset() {
            let mut d = decider(FallPolicyParameters {
                transition_votes: 1,
                ..Default::default()
            });
            d.next_generations.insert(7, u64::MAX - 1);
            let event = update(&mut d, 7, 0, 0.7, 0.0).remove(0);
            assert_eq!(
                event.identity,
                format!("boot:epoch:fall:none:7:7:{}:1", u64::MAX - 1)
            );
            assert_eq!(d.next_generations[&7], u64::MAX);
            d.update(45, 45.0, &BTreeMap::new(), [], None).unwrap();
            assert_eq!(
                d.update(
                    46,
                    46.0,
                    &BTreeMap::from([(7, probability(0.7, 0.0))]),
                    [7],
                    None
                ),
                Err(FallError::Rejected(FallFailure::Overflow(
                    FallCounter::Generation
                )))
            );
            assert_eq!(d.generation_for(7), None);
            for (counter, fallen) in [
                (FallCounter::FallenStreak, 0.8),
                (FallCounter::RecoveryStreak, 0.0),
            ] {
                let mut d = decider(FallPolicyParameters::default());
                update(&mut d, 7, 0, 0.0, 0.6);
                let state = d.states.get_mut(&7).unwrap();
                if counter == FallCounter::FallenStreak {
                    state.fallen_streak = u64::MAX - 1;
                } else {
                    state.recovery_streak = u64::MAX - 1;
                }
                update(&mut d, 7, 1, 0.0, fallen);
                let state = d.states[&7];
                assert_eq!(
                    if counter == FallCounter::FallenStreak {
                        state.fallen_streak
                    } else {
                        state.recovery_streak
                    },
                    u64::MAX
                );
                assert_eq!(
                    d.update(
                        2,
                        2.0,
                        &BTreeMap::from([(7, probability(0.0, fallen))]),
                        [7],
                        None
                    ),
                    Err(FallError::Rejected(FallFailure::Overflow(counter)))
                );
                assert_eq!(d.states[&7], state);
            }
        }

        #[test]
        fn authority_capacity_failure_exposes_partial_events_and_stops_all_mutations() {
            let mut d = FallPolicyDecider::new(
                "camera",
                "facility",
                "boot",
                "epoch",
                7,
                FallPolicy::new(FallPolicyParameters {
                    transition_votes: 1,
                    ..Default::default()
                })
                .unwrap(),
                FallCapacities {
                    episodes: 1,
                    ..capacities()
                },
            )
            .unwrap();
            let scores = BTreeMap::from([(1, probability(0.7, 0.0)), (2, probability(0.8, 0.0))]);
            let error = d.update(0, 0.0, &scores, [2, 1], None).unwrap_err();
            let FallError::FatalPartialState {
                track_id,
                cause,
                emitted_events,
            } = error
            else {
                panic!("authority failure must be explicit partial state");
            };
            assert_eq!(track_id, 2);
            assert_eq!(
                cause,
                FallFailure::Episode(EpisodeError::Capacity(
                    crate::episode::CapacityLimit::Episodes
                ))
            );
            assert_eq!(emitted_events.len(), 1);
            assert_eq!(emitted_events[0].identity, "boot:epoch:fall:none:1:7:0:1");
            assert_eq!(d.episode_state(1), EpisodeState::Open);
            assert_eq!(d.episode_state(2), EpisodeState::Normal);
            assert!(!d.last_update_evaluated());
            assert!(d.last_trace_snapshots().is_empty());
            let states = d.states.clone();
            assert_eq!(
                d.update(1, 1.0, &BTreeMap::new(), [], None),
                Err(FallError::Poisoned(cause))
            );
            assert_eq!(d.coast(), Err(FallError::Poisoned(cause)));
            assert_eq!(
                d.release_onset(&emitted_events[0]),
                Err(FallError::Poisoned(cause))
            );
            assert_eq!(d.states, states);
        }

        #[test]
        fn evicting_track_state_does_not_reclaim_authority_episode_capacity() {
            let mut d = FallPolicyDecider::new(
                "camera",
                "facility",
                "boot",
                "epoch",
                7,
                FallPolicy::default(),
                FallCapacities {
                    episodes: 1,
                    ..capacities()
                },
            )
            .unwrap();
            update(&mut d, 7, 0, 0.0, 0.0); // NORMAL still owns an authority row.
            d.update(45, 45.0, &BTreeMap::new(), [], None).unwrap();
            let error = d
                .update(
                    46,
                    46.0,
                    &BTreeMap::from([(8, probability(0.7, 0.0))]),
                    [8],
                    None,
                )
                .unwrap_err();
            assert!(matches!(error, FallError::FatalPartialState {
                track_id: 8, cause: FallFailure::Episode(EpisodeError::Capacity(crate::episode::CapacityLimit::Episodes)),
                ref emitted_events,
            } if emitted_events.is_empty()));
            assert!(!d.last_update_evaluated());
        }
    }
}
