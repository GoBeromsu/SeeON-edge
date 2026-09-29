//! Pure lifecycle port of worker/domains/episode/authority.py.
//! No inference, persistence, evidence enrichment, or wire-identity conversion.
use std::collections::{BTreeMap, VecDeque};

pub use vocabulary::{
    BusinessEvent, CapacityLimit, EpisodeError, EpisodeProposal, EpisodeState,
    MAX_AUTHORITY_ID_BYTES, OverflowCounter, ProposalDisposition, suppression_reason,
};

// Keep the value/error vocabulary separate from the mutable lifecycle owner.
mod vocabulary {
    use std::fmt;

    /// Byte bound for each constructor-owned boot/epoch identity, without truncation.
    pub const MAX_AUTHORITY_ID_BYTES: usize = 256;

    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    pub enum EpisodeState {
        #[default]
        Normal,
        Candidate,
        Open,
        Unknown,
        Resolved,
    }
    impl EpisodeState {
        pub fn as_str(self) -> &'static str {
            match self {
                Self::Normal => "normal",
                Self::Candidate => "candidate",
                Self::Open => "open",
                Self::Unknown => "unknown",
                Self::Resolved => "resolved",
            }
        }
    }
    impl fmt::Display for EpisodeState {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(self.as_str())
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ProposalDisposition {
        Emitted,
        Reassociated,
        AlreadyOpen,
        ResolvedHold,
        Candidate,
        NotQualifying,
        Recovery,
    }
    impl ProposalDisposition {
        pub fn as_str(self) -> &'static str {
            match self {
                Self::Emitted => "emitted",
                Self::Reassociated => "reassociated",
                Self::AlreadyOpen => "already-open",
                Self::ResolvedHold => "resolved-hold",
                Self::Candidate => "candidate",
                Self::NotQualifying => "not-qualifying",
                Self::Recovery => "recovery",
            }
        }
    }
    impl fmt::Display for ProposalDisposition {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(self.as_str())
        }
    }

    pub fn suppression_reason(disposition: Option<ProposalDisposition>) -> Option<&'static str> {
        match disposition {
            Some(ProposalDisposition::AlreadyOpen) => Some("episode-already-open"),
            Some(ProposalDisposition::Reassociated) => Some("episode-reassociated"),
            Some(ProposalDisposition::ResolvedHold) => Some("episode-resolved-hold"),
            Some(ProposalDisposition::Candidate) => Some("episode-candidate"),
            _ => None,
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    pub struct EpisodeProposal {
        pub camera_id: String,
        pub facility_id: String,
        pub event_type: String,
        /// The admitting owner rejects SDK sentinels; this owner never remaps IDs.
        pub track_id: u64,
        pub bed_id: Option<u64>,
        pub frame_index: i64,
        pub time_sec: f64,
        pub qualifying: bool,
        pub confirmed_recovery: bool,
        pub probability: Option<f64>,
        pub domain: Option<String>,
        /// Track generation, named as in the Python proposal.
        pub generation: u64,
        pub confirmation_votes: usize,
        pub confirmation_window: usize,
    }

    /// Only the fields populated by the authority; downstream owns audit and media.
    #[derive(Debug, Clone, PartialEq)]
    pub struct BusinessEvent {
        pub domain: String,
        pub event_type: String,
        pub identity: String,
        pub camera_id: String,
        pub facility_id: String,
        pub time_sec: f64,
        pub probability: Option<f64>,
        pub person_id: Option<u64>,
        pub bed_id: Option<u64>,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum CapacityLimit {
        Episodes,
        ConfirmationWindow,
    }
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum OverflowCounter {
        EventSequence,
        TrackIdSwitchAbsorbedTotal,
    }
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum EpisodeError {
        InvalidIdentity,
        InvalidBudget,
        InvalidConfirmation,
        NonFiniteTime,
        NonFiniteProbability,
        Capacity(CapacityLimit),
        Overflow(OverflowCounter),
    }
    impl fmt::Display for EpisodeError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(match self {
                Self::InvalidIdentity => {
                    "boot and epoch identities must be nonempty and within the byte limit"
                }
                Self::InvalidBudget => "episode and confirmation-window budgets must be positive",
                Self::InvalidConfirmation => {
                    "confirmation votes must be positive and covered by the window"
                }
                Self::NonFiniteTime => "episode time must be finite",
                Self::NonFiniteProbability => "episode probability must be finite",
                Self::Capacity(CapacityLimit::Episodes) => "episode capacity exceeded",
                Self::Capacity(CapacityLimit::ConfirmationWindow) => {
                    "confirmation-window capacity exceeded"
                }
                Self::Overflow(OverflowCounter::EventSequence) => "event sequence overflow",
                Self::Overflow(OverflowCounter::TrackIdSwitchAbsorbedTotal) => {
                    "track-switch counter overflow"
                }
            })
        }
    }
    impl std::error::Error for EpisodeError {}
}

type EpisodeKey = (String, String, Option<u64>, u64);

#[derive(Debug, Clone, Default, PartialEq)]
struct Episode {
    state: EpisodeState,
    votes: VecDeque<bool>,
    confirmation_window: usize,
    unknown_frame: Option<i64>,
    unknown_time: Option<f64>,
    sequence: Option<u64>,
    emitted_identity: Option<String>,
}
impl Episode {
    fn within(&self, frame_index: i64, time_sec: f64) -> bool {
        let (Some(frame), Some(time)) = (self.unknown_frame, self.unknown_time) else {
            return false;
        };
        // An i64 difference fits i128. Deliberately no lower bound on either delta.
        // Finite f64 operands may produce an infinite delta, just as in Python.
        i128::from(frame_index) - i128::from(frame) <= 75 && time_sec - time <= 5.0
    }

    /// Assess without mutation so sequence overflow cannot partially apply recovery.
    fn disposition(&self, proposal: &EpisodeProposal) -> ProposalDisposition {
        use EpisodeState::{Candidate, Open, Resolved, Unknown};
        if self.state == Unknown && self.within(proposal.frame_index, proposal.time_sec) {
            return ProposalDisposition::Reassociated;
        }
        // Expired UNKNOWN acts as RESOLVED; recovery falls through to voting.
        if matches!(self.state, Unknown | Resolved) && !proposal.confirmed_recovery {
            return ProposalDisposition::ResolvedHold;
        }
        if self.state == Open {
            return if proposal.confirmed_recovery {
                ProposalDisposition::Recovery
            } else {
                ProposalDisposition::AlreadyOpen
            };
        }
        if !proposal.qualifying {
            return ProposalDisposition::NotQualifying;
        }
        let votes = if self.state == Candidate {
            // Only true votes are appended. Below the stored bound, +1 cannot overflow.
            let count = self.votes.len();
            if count < self.confirmation_window {
                count + 1
            } else {
                count
            }
        } else {
            1
        };
        if votes >= proposal.confirmation_votes {
            ProposalDisposition::Emitted
        } else {
            ProposalDisposition::Candidate
        }
    }

    fn apply(&mut self, proposal: &EpisodeProposal, disposition: ProposalDisposition) {
        match disposition {
            ProposalDisposition::Candidate | ProposalDisposition::Emitted => {
                if self.state != EpisodeState::Candidate {
                    self.votes.clear();
                    self.confirmation_window = proposal.confirmation_window;
                }
                if self.votes.len() == self.confirmation_window {
                    self.votes.pop_front();
                }
                self.votes.push_back(true);
                self.state = if disposition == ProposalDisposition::Emitted {
                    EpisodeState::Open
                } else {
                    EpisodeState::Candidate
                };
            }
            ProposalDisposition::Recovery | ProposalDisposition::NotQualifying => {
                self.state = EpisodeState::Normal;
                self.votes.clear();
            }
            ProposalDisposition::AlreadyOpen | ProposalDisposition::Reassociated => {
                self.state = EpisodeState::Open;
            }
            ProposalDisposition::ResolvedHold => self.state = EpisodeState::Resolved,
        }
    }
}

/// Bounded authority with no eviction: forgetting accepted identities would re-arm.
#[derive(Debug)]
pub struct EpisodeAuthority {
    boot_id: String,
    stream_epoch: String,
    source_generation: u64,
    max_episodes: usize,
    max_confirmation_window: usize,
    episodes: BTreeMap<EpisodeKey, Episode>,
    next_sequence: u64,
    track_id_switch_absorbed_total: u64,
    last_disposition: Option<ProposalDisposition>,
}
impl EpisodeAuthority {
    pub fn new(
        boot_id: impl Into<String>,
        stream_epoch: impl Into<String>,
        source_generation: u64,
        max_episodes: usize,
        max_confirmation_window: usize,
    ) -> Result<Self, EpisodeError> {
        let (boot_id, stream_epoch) = (boot_id.into(), stream_epoch.into());
        if [&boot_id, &stream_epoch]
            .iter()
            .any(|id| id.is_empty() || id.len() > MAX_AUTHORITY_ID_BYTES)
        {
            return Err(EpisodeError::InvalidIdentity);
        }
        if max_episodes == 0 || max_confirmation_window == 0 {
            return Err(EpisodeError::InvalidBudget);
        }
        Ok(Self {
            boot_id,
            stream_epoch,
            source_generation,
            max_episodes,
            max_confirmation_window,
            episodes: BTreeMap::new(),
            next_sequence: 1,
            track_id_switch_absorbed_total: 0,
            last_disposition: None,
        })
    }

    pub fn state_for(
        &self,
        camera_id: &str,
        event_type: &str,
        bed_id: Option<u64>,
        track_id: u64,
    ) -> EpisodeState {
        self.episodes
            .get(&(camera_id.into(), event_type.into(), bed_id, track_id))
            .map_or(EpisodeState::Normal, |episode| episode.state)
    }

    pub fn last_disposition(&self) -> Option<ProposalDisposition> {
        self.last_disposition
    }

    pub fn track_id_switch_absorbed_total(&self) -> u64 {
        self.track_id_switch_absorbed_total
    }

    /// Zero or one event, corresponding to the Python tuple. Errors leave all state intact.
    pub fn propose(
        &mut self,
        proposal: &EpisodeProposal,
    ) -> Result<Option<BusinessEvent>, EpisodeError> {
        self.validate_proposal(proposal)?;
        let key = episode_key(proposal, proposal.track_id);
        let existing = self.episodes.get(&key);
        if existing.is_none() && self.episodes.len() >= self.max_episodes {
            return Err(EpisodeError::Capacity(CapacityLimit::Episodes));
        }
        let empty = Episode::default();
        let disposition = existing.unwrap_or(&empty).disposition(proposal);
        let emission = if disposition == ProposalDisposition::Emitted {
            let next = self
                .next_sequence
                .checked_add(1)
                .ok_or(EpisodeError::Overflow(OverflowCounter::EventSequence))?;
            Some((next, self.event(proposal, self.next_sequence)))
        } else {
            None
        };
        // All fallible validation and counter reservation precede insertion or transition.
        let episode = self.episodes.entry(key).or_default();
        episode.apply(proposal, disposition);
        let event = emission.map(|(next, event)| {
            episode.sequence = Some(self.next_sequence);
            episode.emitted_identity = Some(event.identity.clone());
            self.next_sequence = next;
            event
        });
        self.last_disposition = Some(disposition);
        Ok(event)
    }

    /// Reopen only the matching failed-staging identity, at most once; never rewind sequence.
    pub fn release(&mut self, event_identity: &str) -> bool {
        for episode in self.episodes.values_mut() {
            if episode.emitted_identity.as_deref() == Some(event_identity) {
                episode.state = EpisodeState::Normal;
                episode.votes.clear();
                episode.emitted_identity = None;
                return true;
            }
        }
        false
    }

    pub fn track_lost(
        &mut self,
        camera_id: &str,
        frame_index: i64,
        time_sec: f64,
        track_id: Option<u64>,
    ) -> Result<(), EpisodeError> {
        validate_time(time_sec)?;
        for ((camera, _, _, track), episode) in &mut self.episodes {
            if camera == camera_id
                && track_id.is_none_or(|id| id == *track)
                && episode.state == EpisodeState::Open
            {
                episode.state = EpisodeState::Unknown;
                episode.unknown_frame = Some(frame_index);
                episode.unknown_time = Some(time_sec);
            }
        }
        Ok(())
    }

    pub fn reassociate(
        &mut self,
        proposal: &EpisodeProposal,
        previous_track_id: u64,
    ) -> Result<bool, EpisodeError> {
        self.validate_proposal(proposal)?;
        let old_key = episode_key(proposal, previous_track_id);
        let Some(episode) = self.episodes.get(&old_key) else {
            return Ok(false);
        };
        if episode.state != EpisodeState::Unknown
            || !episode.within(proposal.frame_index, proposal.time_sec)
        {
            return Ok(false);
        }
        let next =
            self.track_id_switch_absorbed_total
                .checked_add(1)
                .ok_or(EpisodeError::Overflow(
                    OverflowCounter::TrackIdSwitchAbsorbedTotal,
                ))?;
        let new_key = episode_key(proposal, proposal.track_id);
        let mut episode = self
            .episodes
            .remove(&old_key)
            .expect("episode checked above");
        episode.state = EpisodeState::Open;
        // Canonical collision behavior: overwrite the target, even an accepted OPEN row.
        // Review risk: the displaced row's accepted identity is no longer releasable.
        self.episodes.insert(new_key, episode);
        self.track_id_switch_absorbed_total = next;
        Ok(true)
    }

    pub fn reassociate_bed_exit(
        &mut self,
        proposal: &EpisodeProposal,
    ) -> Result<bool, EpisodeError> {
        self.reassociate_matching(proposal, "bed-exit", proposal.bed_id)
    }

    pub fn reassociate_fall(&mut self, proposal: &EpisodeProposal) -> Result<bool, EpisodeError> {
        self.reassociate_matching(proposal, "fall", None)
    }

    fn reassociate_matching(
        &mut self,
        proposal: &EpisodeProposal,
        event_type: &str,
        bed_id: Option<u64>,
    ) -> Result<bool, EpisodeError> {
        self.validate_proposal(proposal)?;
        let previous = self
            .episodes
            .iter()
            .filter_map(|((camera, event, bed, track), episode)| {
                (camera == &proposal.camera_id
                    && event == event_type
                    && *bed == bed_id
                    && episode.state == EpisodeState::Unknown
                    && episode.within(proposal.frame_index, proposal.time_sec))
                .then_some(*track)
            })
            .min(); // Explicit numeric lowest-ID policy, never insertion order.
        // Python scans literal event/bed filters, then looks up the proposal's own key.
        match previous {
            Some(track) => self.reassociate(proposal, track),
            None => Ok(false),
        }
    }

    pub fn expire(&mut self, frame_index: i64, time_sec: f64) -> Result<(), EpisodeError> {
        validate_time(time_sec)?;
        for episode in self.episodes.values_mut() {
            if episode.state == EpisodeState::Unknown && !episode.within(frame_index, time_sec) {
                episode.state = EpisodeState::Resolved;
            }
        }
        Ok(())
    }

    fn validate_proposal(&self, proposal: &EpisodeProposal) -> Result<(), EpisodeError> {
        if proposal.confirmation_votes == 0
            || proposal.confirmation_window < proposal.confirmation_votes
        {
            return Err(EpisodeError::InvalidConfirmation);
        }
        if proposal.confirmation_window > self.max_confirmation_window {
            return Err(EpisodeError::Capacity(CapacityLimit::ConfirmationWindow));
        }
        validate_time(proposal.time_sec)?;
        if proposal.probability.is_some_and(|value| !value.is_finite()) {
            return Err(EpisodeError::NonFiniteProbability);
        }
        Ok(())
    }

    fn event(&self, proposal: &EpisodeProposal, sequence: u64) -> BusinessEvent {
        let domain = proposal
            .domain
            .as_deref()
            .filter(|domain| !domain.is_empty())
            .unwrap_or_else(|| {
                if proposal.event_type == "bed-exit" {
                    "bed_exit"
                } else {
                    proposal.event_type.as_str()
                }
            });
        let bed = proposal
            .bed_id
            .map_or_else(|| "none".to_owned(), |id| id.to_string());
        BusinessEvent {
            domain: domain.to_owned(),
            event_type: proposal.event_type.clone(),
            identity: format!(
                "{}:{}:{}:{}:{}:{}:{}:{}",
                self.boot_id,
                self.stream_epoch,
                proposal.event_type,
                bed,
                proposal.track_id,
                self.source_generation,
                proposal.generation,
                sequence
            ),
            camera_id: proposal.camera_id.clone(),
            facility_id: proposal.facility_id.clone(),
            time_sec: proposal.time_sec,
            probability: proposal.probability,
            person_id: Some(proposal.track_id),
            bed_id: proposal.bed_id,
        }
    }
}

fn episode_key(proposal: &EpisodeProposal, track_id: u64) -> EpisodeKey {
    (
        proposal.camera_id.clone(),
        proposal.event_type.clone(),
        proposal.bed_id,
        track_id,
    )
}

fn validate_time(time_sec: f64) -> Result<(), EpisodeError> {
    if time_sec.is_finite() {
        Ok(())
    } else {
        Err(EpisodeError::NonFiniteTime)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn authority() -> EpisodeAuthority {
        EpisodeAuthority::new("boot", "epoch", 7, 32, 8).unwrap()
    }

    fn proposal(track_id: u64) -> EpisodeProposal {
        EpisodeProposal {
            camera_id: "camera".into(),
            facility_id: "facility".into(),
            event_type: "fall".into(),
            track_id,
            bed_id: None,
            frame_index: 100,
            time_sec: 10.0,
            qualifying: true,
            confirmed_recovery: false,
            probability: Some(0.8),
            domain: None,
            generation: 11,
            confirmation_votes: 1,
            confirmation_window: 4,
        }
    }

    fn state(authority: &EpisodeAuthority, proposal: &EpisodeProposal) -> EpisodeState {
        authority.state_for(
            &proposal.camera_id,
            &proposal.event_type,
            proposal.bed_id,
            proposal.track_id,
        )
    }

    fn emit(authority: &mut EpisodeAuthority, proposal: &EpisodeProposal) -> BusinessEvent {
        authority.propose(proposal).unwrap().expect("onset")
    }

    fn snapshot(
        authority: &EpisodeAuthority,
    ) -> (
        BTreeMap<EpisodeKey, Episode>,
        u64,
        u64,
        Option<ProposalDisposition>,
    ) {
        (
            authority.episodes.clone(),
            authority.next_sequence,
            authority.track_id_switch_absorbed_total(),
            authority.last_disposition(),
        )
    }

    #[test]
    fn votes_reset_and_open_requires_scored_recovery() {
        let mut authority = authority();
        let mut p = proposal(9);
        p.confirmation_votes = 3;
        for _ in 0..2 {
            assert_eq!(authority.propose(&p).unwrap(), None);
            assert_eq!(state(&authority, &p), EpisodeState::Candidate);
        }
        p.qualifying = false;
        assert_eq!(authority.propose(&p).unwrap(), None);
        assert_eq!(state(&authority, &p), EpisodeState::Normal);
        assert_eq!(
            authority.last_disposition(),
            Some(ProposalDisposition::NotQualifying)
        );
        p.qualifying = true;
        for _ in 0..2 {
            assert_eq!(authority.propose(&p).unwrap(), None);
        }
        // A recovery flag does not reset an existing CANDIDATE in the canonical owner.
        p.confirmed_recovery = true;
        let first = emit(&mut authority, &p);
        assert_eq!(first.identity, "boot:epoch:fall:none:9:7:11:1");
        p.confirmed_recovery = false;
        for qualifying in [true, false, true] {
            p.qualifying = qualifying;
            assert_eq!(authority.propose(&p).unwrap(), None);
            assert_eq!(state(&authority, &p), EpisodeState::Open);
            assert_eq!(
                authority.last_disposition(),
                Some(ProposalDisposition::AlreadyOpen)
            );
        }
        p.confirmed_recovery = true;
        assert_eq!(authority.propose(&p).unwrap(), None);
        assert_eq!(state(&authority, &p), EpisodeState::Normal);
        assert_eq!(
            authority.last_disposition(),
            Some(ProposalDisposition::Recovery)
        );
        p.confirmed_recovery = false;
        for _ in 0..2 {
            assert_eq!(authority.propose(&p).unwrap(), None);
        }
        assert_eq!(
            emit(&mut authority, &p).identity,
            "boot:epoch:fall:none:9:7:11:2"
        );
    }

    #[test]
    fn same_id_return_and_expiry_use_both_inclusive_boundaries() {
        for (frame_delta, time_delta, within) in [
            (74, 4.999, true),
            (75, 5.0, true),
            (76, 5.0, false),
            (75, 5.000001, false),
            (-1, -1.0, true),
            (76, -1.0, false),
            (-1, 5.000001, false),
        ] {
            for use_expire in [false, true] {
                let mut authority = authority();
                let mut p = proposal(3);
                let first = emit(&mut authority, &p);
                authority.track_lost("camera", 100, 10.0, Some(3)).unwrap();
                p.frame_index = 100 + frame_delta;
                p.time_sec = 10.0 + time_delta;
                if use_expire {
                    authority.expire(p.frame_index, p.time_sec).unwrap();
                    assert_eq!(
                        state(&authority, &p),
                        if within {
                            EpisodeState::Unknown
                        } else {
                            EpisodeState::Resolved
                        }
                    );
                }
                assert_eq!(authority.propose(&p).unwrap(), None);
                assert_eq!(
                    state(&authority, &p),
                    if within {
                        EpisodeState::Open
                    } else {
                        EpisodeState::Resolved
                    }
                );
                assert_eq!(
                    authority.last_disposition(),
                    Some(if within {
                        ProposalDisposition::Reassociated
                    } else {
                        ProposalDisposition::ResolvedHold
                    })
                );
                assert_eq!(authority.track_id_switch_absorbed_total(), 0);
                assert_eq!(authority.next_sequence, 2);
                assert!(authority.release(&first.identity));
            }
        }
    }

    #[test]
    fn unknown_return_precedes_recovery_but_expired_recovery_falls_through() {
        let mut authority = authority();
        let mut p = proposal(1);
        emit(&mut authority, &p);
        authority.track_lost("camera", 100, 10.0, None).unwrap();
        p.confirmed_recovery = true;
        p.qualifying = false;
        assert_eq!(authority.propose(&p).unwrap(), None);
        assert_eq!(state(&authority, &p), EpisodeState::Open);
        assert_eq!(
            authority.last_disposition(),
            Some(ProposalDisposition::Reassociated)
        );
        authority.track_lost("camera", 100, 10.0, None).unwrap();
        p.frame_index = 176;
        p.confirmed_recovery = false;
        assert_eq!(authority.propose(&p).unwrap(), None);
        assert_eq!(state(&authority, &p), EpisodeState::Resolved);
        p.qualifying = true;
        for _ in 0..4 {
            assert_eq!(authority.propose(&p).unwrap(), None);
            assert_eq!(
                authority.last_disposition(),
                Some(ProposalDisposition::ResolvedHold)
            );
        }
        p.qualifying = false;
        p.confirmed_recovery = true;
        assert_eq!(authority.propose(&p).unwrap(), None);
        assert_eq!(state(&authority, &p), EpisodeState::Normal);
        assert_eq!(
            authority.last_disposition(),
            Some(ProposalDisposition::NotQualifying)
        );
        p.qualifying = true;
        assert_eq!(
            emit(&mut authority, &p).identity,
            "boot:epoch:fall:none:1:7:11:2"
        );
        authority.track_lost("camera", 176, 10.0, None).unwrap();
        p.frame_index = 252;
        // UNKNOWN -> RESOLVED -> NORMAL -> OPEN can happen in one proposal.
        assert_eq!(
            emit(&mut authority, &p).identity,
            "boot:epoch:fall:none:1:7:11:3"
        );
        authority.track_lost("camera", 252, 10.0, None).unwrap();
        authority.expire(328, 10.0).unwrap();
        p.frame_index = 328;
        p.confirmation_votes = 2;
        assert_eq!(authority.propose(&p).unwrap(), None);
        assert_eq!(state(&authority, &p), EpisodeState::Candidate);
        assert_eq!(
            emit(&mut authority, &p).identity,
            "boot:epoch:fall:none:1:7:11:4"
        );
    }

    #[test]
    fn track_loss_is_camera_scoped_optional_track_and_open_only() {
        let mut authority = authority();
        let fall = proposal(1);
        let mut bed = proposal(1);
        bed.event_type = "bed-exit".into();
        bed.bed_id = Some(4);
        let other_track = proposal(2);
        let mut other_camera = proposal(1);
        other_camera.camera_id = "other-camera".into();
        for p in [&fall, &bed, &other_track, &other_camera] {
            emit(&mut authority, p);
        }
        let mut candidate = proposal(3);
        candidate.confirmation_votes = 2;
        assert_eq!(authority.propose(&candidate).unwrap(), None);
        let mut normal = proposal(4);
        normal.qualifying = false;
        assert_eq!(authority.propose(&normal).unwrap(), None);
        authority.track_lost("camera", 100, 10.0, Some(1)).unwrap();
        assert_eq!(state(&authority, &fall), EpisodeState::Unknown);
        assert_eq!(state(&authority, &bed), EpisodeState::Unknown);
        assert_eq!(state(&authority, &other_track), EpisodeState::Open);
        assert_eq!(state(&authority, &other_camera), EpisodeState::Open);
        authority.track_lost("camera", 150, 12.0, None).unwrap();
        assert_eq!(state(&authority, &candidate), EpisodeState::Candidate);
        assert_eq!(state(&authority, &normal), EpisodeState::Normal);
        assert_eq!(state(&authority, &other_track), EpisodeState::Unknown);
        assert_eq!(state(&authority, &other_camera), EpisodeState::Open);
        authority.expire(176, 13.0).unwrap();
        // Repeated loss did not refresh already-UNKNOWN rows.
        assert_eq!(state(&authority, &fall), EpisodeState::Resolved);
        assert_eq!(state(&authority, &bed), EpisodeState::Resolved);
        assert_eq!(state(&authority, &other_track), EpisodeState::Unknown);
    }

    #[test]
    fn scans_choose_lowest_numeric_id_with_exact_camera_event_and_bed_filters() {
        let mut authority = authority();
        let mut rows = Vec::new();
        for (camera, event, bed, track) in [
            ("camera", "bed-exit", Some(8), 30),
            ("camera", "bed-exit", Some(8), 2),
            ("camera", "bed-exit", Some(9), 1),
            ("other-camera", "bed-exit", Some(8), 1),
            ("camera", "fall", None, 30),
            ("camera", "fall", None, 2),
            ("camera", "fall", Some(8), 1),
            ("other-camera", "fall", None, 1),
        ] {
            let mut p = proposal(track);
            p.camera_id = camera.into();
            p.event_type = event.into();
            p.bed_id = bed;
            let event = emit(&mut authority, &p);
            rows.push((p, event));
        }
        authority.track_lost("camera", 100, 10.0, None).unwrap();
        authority
            .track_lost("other-camera", 100, 10.0, None)
            .unwrap();
        let mut bed = rows[1].0.clone();
        bed.track_id = 90;
        bed.generation = 99;
        assert!(authority.reassociate_bed_exit(&bed).unwrap());
        let mut fall = rows[5].0.clone();
        fall.track_id = 91;
        assert!(authority.reassociate_fall(&fall).unwrap());
        for (index, (p, _)) in rows.iter().enumerate() {
            assert_eq!(
                state(&authority, p),
                if [1, 5].contains(&index) {
                    EpisodeState::Normal
                } else {
                    EpisodeState::Unknown
                }
            );
        }
        assert_eq!(authority.track_id_switch_absorbed_total(), 2);
        // Explicit reassociation does not update last_disposition or event sequence.
        assert_eq!(
            authority.last_disposition(),
            Some(ProposalDisposition::Emitted)
        );
        assert_eq!(authority.next_sequence, 9);
        assert_eq!(authority.episodes[&episode_key(&bed, 90)].sequence, Some(2));
        assert_eq!(
            authority.episodes[&episode_key(&fall, 91)].sequence,
            Some(6)
        );
        assert_eq!(authority.propose(&bed).unwrap(), None);
        assert_eq!(authority.propose(&fall).unwrap(), None);
        assert!(authority.release(&rows[1].1.identity));
        assert!(authority.release(&rows[5].1.identity));
        assert_eq!(state(&authority, &bed), EpisodeState::Normal);
        assert_eq!(state(&authority, &fall), EpisodeState::Normal);
    }

    #[test]
    fn scan_and_lookup_filters_are_not_silently_normalized() {
        let mut authority = authority();
        let p = proposal(1);
        emit(&mut authority, &p);
        authority.track_lost("camera", 100, 10.0, None).unwrap();
        let mut with_bed = proposal(1);
        with_bed.bed_id = Some(9);
        let mut replacement = with_bed.clone();
        replacement.track_id = 90;
        // Scan finds fall/None/1, but the subsequent lookup uses proposal.bed_id.
        assert!(!authority.reassociate_fall(&replacement).unwrap());
        let bed_event = emit(&mut authority, &with_bed);
        authority.track_lost("camera", 100, 10.0, Some(1)).unwrap();
        assert!(authority.reassociate_fall(&replacement).unwrap());
        assert_eq!(state(&authority, &p), EpisodeState::Unknown);
        assert_eq!(state(&authority, &with_bed), EpisodeState::Normal);
        assert!(authority.release(&bed_event.identity));

        let mut bed_exit = proposal(1);
        bed_exit.event_type = "bed-exit".into();
        // bed-exit scanning accepts None beds exactly as Python does.
        emit(&mut authority, &bed_exit);
        authority.track_lost("camera", 100, 10.0, None).unwrap();
        let mut wrong_event = proposal(92);
        wrong_event.event_type = "custom".into();
        assert!(!authority.reassociate_bed_exit(&wrong_event).unwrap());
        wrong_event.event_type = "bed-exit".into();
        assert!(authority.reassociate_bed_exit(&wrong_event).unwrap());
    }

    #[test]
    fn explicit_reassociation_preserves_canonical_target_overwrite() {
        let mut authority = EpisodeAuthority::new("boot", "epoch", 7, 2, 4).unwrap();
        let old = proposal(3);
        let new = proposal(8);
        let first = emit(&mut authority, &old);
        let displaced = emit(&mut authority, &new);
        assert!(!authority.reassociate(&new, 3).unwrap()); // OPEN is not eligible.
        authority.track_lost("camera", 100, 10.0, Some(3)).unwrap();
        assert!(authority.reassociate(&new, 3).unwrap());
        assert_eq!(authority.episodes.len(), 1);
        assert_eq!(state(&authority, &old), EpisodeState::Normal);
        assert_eq!(state(&authority, &new), EpisodeState::Open);
        assert_eq!(authority.episodes[&episode_key(&new, 8)].sequence, Some(1));
        assert!(!authority.release(&displaced.identity));
        assert!(authority.release(&first.identity));
        assert_eq!(
            emit(&mut authority, &new).identity,
            "boot:epoch:fall:none:8:7:11:3"
        );
        authority.track_lost("camera", 100, 10.0, Some(8)).unwrap();
        // Explicit same-ID reassociation increments the counter, unlike propose.
        assert!(authority.reassociate(&new, 8).unwrap());
        assert_eq!(authority.track_id_switch_absorbed_total(), 2);
        assert_eq!(authority.episodes.len(), 1);
    }

    #[test]
    fn failed_staging_release_is_identity_exact_once_and_never_rewinds() {
        let mut authority = authority();
        let first_p = proposal(1);
        let other_p = proposal(2);
        let first = emit(&mut authority, &first_p);
        let other = emit(&mut authority, &other_p);
        let before = snapshot(&authority);
        assert!(!authority.release("unrelated"));
        assert_eq!(snapshot(&authority), before);
        authority.track_lost("camera", 100, 10.0, Some(1)).unwrap();
        let mut replacement = proposal(10);
        replacement.confirmation_votes = 2;
        assert!(authority.reassociate(&replacement, 1).unwrap());
        assert!(authority.release(&first.identity));
        assert!(!authority.release(&first.identity));
        assert_eq!(state(&authority, &other_p), EpisodeState::Open);
        assert_eq!(authority.propose(&replacement).unwrap(), None);
        let retry = emit(&mut authority, &replacement);
        assert_eq!(retry.identity, "boot:epoch:fall:none:10:7:11:3");
        assert!(!authority.release(&first.identity));
        assert_eq!(state(&authority, &replacement), EpisodeState::Open);
        assert!(authority.release(&other.identity));
        assert!(authority.release(&retry.identity));
    }

    #[test]
    fn release_can_match_identity_retained_across_recovery() {
        let mut authority = authority();
        let mut p = proposal(1);
        let event = emit(&mut authority, &p);
        p.confirmed_recovery = true;
        authority.propose(&p).unwrap();
        p.confirmed_recovery = false;
        p.confirmation_votes = 3;
        authority.propose(&p).unwrap();
        assert_eq!(state(&authority, &p), EpisodeState::Candidate);
        assert!(authority.release(&event.identity));
        assert_eq!(state(&authority, &p), EpisodeState::Normal);
        assert!(authority.episodes[&episode_key(&p, 1)].votes.is_empty());
        assert_eq!(authority.episodes[&episode_key(&p, 1)].sequence, Some(1));
    }

    #[test]
    fn event_identity_and_owned_fields_match_domain_fallback_and_override() {
        for (event_type, bed_id, domain, expected_domain, expected_identity) in [
            (
                "bed-exit",
                Some(0),
                None,
                "bed_exit",
                "boot:epoch:bed-exit:0:23:7:11:1",
            ),
            (
                "bed-exit",
                Some(4),
                Some("custom"),
                "custom",
                "boot:epoch:bed-exit:4:23:7:11:1",
            ),
            (
                "bed-exit",
                None,
                Some(""),
                "bed_exit",
                "boot:epoch:bed-exit:none:23:7:11:1",
            ),
            ("fall", None, None, "fall", "boot:epoch:fall:none:23:7:11:1"),
            (
                "custom-event",
                None,
                Some(""),
                "custom-event",
                "boot:epoch:custom-event:none:23:7:11:1",
            ),
        ] {
            let mut authority = authority();
            let mut p = proposal(23);
            p.event_type = event_type.into();
            p.bed_id = bed_id;
            p.domain = domain.map(str::to_owned);
            p.frame_index = -8;
            p.time_sec = -1.25;
            p.probability = None;
            assert_eq!(
                emit(&mut authority, &p),
                BusinessEvent {
                    domain: expected_domain.into(),
                    event_type: event_type.into(),
                    identity: expected_identity.into(),
                    camera_id: "camera".into(),
                    facility_id: "facility".into(),
                    time_sec: -1.25,
                    probability: None,
                    person_id: Some(23),
                    bed_id,
                }
            );
        }
        let mut authority =
            EpisodeAuthority::new("boot:part", "epoch:part", u64::MAX, 1, 4).unwrap();
        let mut p = proposal(u64::MAX - 1);
        p.generation = u64::MAX;
        p.bed_id = Some(u64::MAX);
        p.probability = Some(-0.25); // Python does not clamp or range-check finite scores.
        let event = emit(&mut authority, &p);
        assert_eq!(
            event.identity,
            "boot:part:epoch:part:fall:18446744073709551615:18446744073709551614:18446744073709551615:18446744073709551615:1"
        );
        assert_eq!(event.person_id, Some(u64::MAX - 1));
        assert_eq!(event.probability, Some(-0.25));
    }

    #[test]
    fn candidate_window_changes_only_after_returning_to_normal() {
        let mut authority = authority();
        let mut p = proposal(1);
        p.confirmation_votes = 2;
        p.confirmation_window = 2;
        assert_eq!(authority.propose(&p).unwrap(), None);
        p.confirmation_votes = 3;
        p.confirmation_window = 5;
        for _ in 0..12 {
            assert_eq!(authority.propose(&p).unwrap(), None);
        }
        assert_eq!(authority.episodes[&episode_key(&p, 1)].votes.len(), 2);
        assert_eq!(
            authority.last_disposition(),
            Some(ProposalDisposition::Candidate)
        );
        p.qualifying = false;
        authority.propose(&p).unwrap();
        p.qualifying = true;
        for _ in 0..2 {
            assert_eq!(authority.propose(&p).unwrap(), None);
        }
        assert_eq!(
            emit(&mut authority, &p).identity,
            "boot:epoch:fall:none:1:7:11:1"
        );
        assert_eq!(
            authority.episodes[&episode_key(&p, 1)].confirmation_window,
            5
        );
    }

    #[test]
    fn capacity_allows_existing_rows_and_moves_but_never_evicts() {
        let mut authority = EpisodeAuthority::new("boot", "epoch", 7, 1, 4).unwrap();
        let mut p = proposal(1);
        p.confirmation_votes = 2;
        assert_eq!(authority.propose(&p).unwrap(), None);
        let mut new = proposal(2);
        for qualifying in [true, false] {
            new.qualifying = qualifying;
            let before = snapshot(&authority);
            assert_eq!(
                authority.propose(&new),
                Err(EpisodeError::Capacity(CapacityLimit::Episodes))
            );
            assert_eq!(snapshot(&authority), before);
        }
        let first = emit(&mut authority, &p);
        assert_eq!(authority.propose(&p).unwrap(), None);
        authority.track_lost("camera", 100, 10.0, None).unwrap();
        assert!(authority.reassociate(&new, 1).unwrap()); // Full-budget move, not insertion.
        assert_eq!(state(&authority, &new), EpisodeState::Open);
        assert!(authority.release(&first.identity));
        new.qualifying = true;
        assert_eq!(
            emit(&mut authority, &new).identity,
            "boot:epoch:fall:none:2:7:11:2"
        );
        authority.track_lost("camera", 100, 10.0, None).unwrap();
        authority.expire(176, 10.0).unwrap();
        assert_eq!(state(&authority, &new), EpisodeState::Resolved);
        assert_eq!(
            authority.propose(&p),
            Err(EpisodeError::Capacity(CapacityLimit::Episodes))
        );
        assert_eq!(authority.propose(&new).unwrap(), None);
        assert_eq!(authority.episodes.len(), 1);
    }

    #[test]
    fn constructor_bounds_and_proposal_validation_do_not_mutate_state() {
        for (episodes, window) in [(0, 1), (1, 0)] {
            assert_eq!(
                EpisodeAuthority::new("boot", "epoch", 0, episodes, window).unwrap_err(),
                EpisodeError::InvalidBudget
            );
        }
        for (boot, epoch) in [
            (String::new(), "epoch".into()),
            ("boot".into(), String::new()),
            ("x".repeat(MAX_AUTHORITY_ID_BYTES + 1), "epoch".into()),
            ("boot".into(), "é".repeat(MAX_AUTHORITY_ID_BYTES / 2 + 1)),
        ] {
            let error = EpisodeAuthority::new(boot, epoch, 0, 1, 4).unwrap_err();
            assert_eq!(error, EpisodeError::InvalidIdentity);
            assert_eq!(
                error.to_string(),
                "boot and epoch identities must be nonempty and within the byte limit"
            );
        }
        let bound = "x".repeat(MAX_AUTHORITY_ID_BYTES);
        let mut at_bound = EpisodeAuthority::new(bound.clone(), bound.clone(), 7, 1, 4).unwrap();
        assert_eq!(
            emit(&mut at_bound, &proposal(1)).identity,
            format!("{bound}:{bound}:fall:none:1:7:11:1")
        );
        let mut authority = authority();
        let mut p = proposal(1);
        p.confirmation_votes = 2;
        authority.propose(&p).unwrap();
        for (votes, window, expected) in [
            (0, 4, EpisodeError::InvalidConfirmation),
            (1, 0, EpisodeError::InvalidConfirmation),
            (3, 2, EpisodeError::InvalidConfirmation),
            (
                1,
                9,
                EpisodeError::Capacity(CapacityLimit::ConfirmationWindow),
            ),
            (
                1,
                usize::MAX,
                EpisodeError::Capacity(CapacityLimit::ConfirmationWindow),
            ),
        ] {
            for track in [1, 2] {
                let mut invalid = p.clone();
                invalid.track_id = track;
                invalid.confirmation_votes = votes;
                invalid.confirmation_window = window;
                let before = snapshot(&authority);
                assert_eq!(authority.propose(&invalid), Err(expected));
                assert_eq!(authority.reassociate(&invalid, 1), Err(expected));
                assert_eq!(snapshot(&authority), before);
            }
        }
        assert_eq!(
            emit(&mut authority, &p).identity,
            "boot:epoch:fall:none:1:7:11:1"
        );
    }

    #[test]
    fn nonfinite_inputs_are_rejected_before_any_mutation() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut authority = authority();
            let p = proposal(1);
            emit(&mut authority, &p);
            authority.track_lost("camera", 100, 10.0, Some(1)).unwrap();
            let mut other = proposal(2);
            other.confirmation_votes = 2;
            authority.propose(&other).unwrap();
            for track in [1, 2, 3] {
                for invalid_time in [true, false] {
                    let mut invalid = proposal(track);
                    let expected = if invalid_time {
                        invalid.time_sec = value;
                        EpisodeError::NonFiniteTime
                    } else {
                        invalid.probability = Some(value);
                        EpisodeError::NonFiniteProbability
                    };
                    let before = snapshot(&authority);
                    assert_eq!(authority.propose(&invalid), Err(expected));
                    assert_eq!(authority.reassociate(&invalid, 1), Err(expected));
                    assert_eq!(authority.reassociate_fall(&invalid), Err(expected));
                    assert_eq!(authority.reassociate_bed_exit(&invalid), Err(expected));
                    assert_eq!(snapshot(&authority), before);
                }
            }
            let before = snapshot(&authority);
            assert_eq!(
                authority.track_lost("camera", 100, value, None),
                Err(EpisodeError::NonFiniteTime)
            );
            assert_eq!(
                authority.expire(176, value),
                Err(EpisodeError::NonFiniteTime)
            );
            assert_eq!(snapshot(&authority), before);
            assert_eq!(authority.propose(&p).unwrap(), None);
            assert_eq!(state(&authority, &p), EpisodeState::Open);
        }
    }

    #[test]
    fn signed_frame_edges_and_finite_time_rollback_preserve_python_comparisons() {
        for (lost_frame, lost_time, returned_frame, returned_time, within) in [
            (i64::MIN, -f64::MAX, i64::MAX, f64::MAX, false),
            (i64::MAX, f64::MAX, i64::MIN, -f64::MAX, true),
            (i64::MIN, -10.0, i64::MIN + 75, -5.0, true),
            (i64::MIN, -10.0, i64::MIN + 76, -5.0, false),
            (i64::MAX, 0.0, i64::MAX - 75, -5.0, true),
            (0, -f64::MAX, 0, f64::MAX, false),
            (0, f64::MAX, 0, -f64::MAX, true),
        ] {
            let mut authority = authority();
            let mut p = proposal(1);
            p.frame_index = lost_frame;
            p.time_sec = lost_time;
            p.probability = Some(f64::MAX);
            let event = emit(&mut authority, &p);
            assert_eq!(event.probability, Some(f64::MAX));
            authority
                .track_lost("camera", lost_frame, lost_time, None)
                .unwrap();
            p.frame_index = returned_frame;
            p.time_sec = returned_time;
            assert_eq!(authority.propose(&p).unwrap(), None);
            assert_eq!(
                state(&authority, &p),
                if within {
                    EpisodeState::Open
                } else {
                    EpisodeState::Resolved
                }
            );
        }
    }

    #[test]
    fn sequence_overflow_never_inserts_votes_recovers_or_promotes_partially() {
        let mut authority = authority();
        authority.next_sequence = u64::MAX - 1;
        let mut p = proposal(1);
        let event = emit(&mut authority, &p);
        assert!(event.identity.ends_with(":18446744073709551614"));
        assert_eq!(authority.next_sequence, u64::MAX);
        assert_eq!(authority.propose(&p).unwrap(), None); // OPEN does not need sequence.
        authority.track_lost("camera", 100, 10.0, None).unwrap();
        p.frame_index = 176;
        p.confirmed_recovery = true;
        for explicit_expiry in [false, true] {
            if explicit_expiry {
                authority.expire(176, 10.0).unwrap();
            }
            let before = snapshot(&authority);
            assert_eq!(
                authority.propose(&p),
                Err(EpisodeError::Overflow(OverflowCounter::EventSequence))
            );
            assert_eq!(snapshot(&authority), before);
        }
        let before = snapshot(&authority);
        assert_eq!(
            authority.propose(&proposal(2)),
            Err(EpisodeError::Overflow(OverflowCounter::EventSequence))
        );
        assert_eq!(snapshot(&authority), before);
        p.confirmation_votes = 2;
        assert_eq!(authority.propose(&p).unwrap(), None); // Recovery into CANDIDATE is allowed.
        let before = snapshot(&authority);
        assert_eq!(
            authority.propose(&p),
            Err(EpisodeError::Overflow(OverflowCounter::EventSequence))
        );
        assert_eq!(snapshot(&authority), before);
        assert_eq!(authority.episodes[&episode_key(&p, 1)].votes.len(), 1);
        assert!(authority.release(&event.identity));
        assert_eq!(authority.next_sequence, u64::MAX);
    }

    #[test]
    fn reassociation_counter_overflow_does_not_remove_or_overwrite_rows() {
        let mut authority = authority();
        let p = proposal(1);
        let new = proposal(2);
        emit(&mut authority, &p);
        let target = emit(&mut authority, &new);
        authority.track_lost("camera", 100, 10.0, Some(1)).unwrap();
        authority.track_id_switch_absorbed_total = u64::MAX;
        let before = snapshot(&authority);
        let error = EpisodeError::Overflow(OverflowCounter::TrackIdSwitchAbsorbedTotal);
        assert_eq!(authority.reassociate(&new, 1), Err(error));
        assert_eq!(authority.reassociate_fall(&new), Err(error));
        assert_eq!(authority.reassociate(&p, 1), Err(error));
        assert!(!authority.reassociate(&new, 99).unwrap());
        assert_eq!(snapshot(&authority), before);
        authority.track_id_switch_absorbed_total = u64::MAX - 1;
        assert!(authority.reassociate(&new, 1).unwrap());
        assert_eq!(authority.track_id_switch_absorbed_total(), u64::MAX);
        assert!(!authority.release(&target.identity));
        authority.track_lost("camera", 100, 10.0, Some(2)).unwrap();
        // Same-ID propose does not increment the explicit reassociation counter.
        assert_eq!(authority.propose(&new).unwrap(), None);
        assert_eq!(state(&authority, &new), EpisodeState::Open);
        assert_eq!(authority.track_id_switch_absorbed_total(), u64::MAX);
    }

    #[test]
    fn disposition_labels_and_suppression_mapping_match_the_trace_contract() {
        for (disposition, label, reason) in [
            (ProposalDisposition::Emitted, "emitted", None),
            (
                ProposalDisposition::Reassociated,
                "reassociated",
                Some("episode-reassociated"),
            ),
            (
                ProposalDisposition::AlreadyOpen,
                "already-open",
                Some("episode-already-open"),
            ),
            (
                ProposalDisposition::ResolvedHold,
                "resolved-hold",
                Some("episode-resolved-hold"),
            ),
            (
                ProposalDisposition::Candidate,
                "candidate",
                Some("episode-candidate"),
            ),
            (ProposalDisposition::NotQualifying, "not-qualifying", None),
            (ProposalDisposition::Recovery, "recovery", None),
        ] {
            assert_eq!(disposition.to_string(), label);
            assert_eq!(suppression_reason(Some(disposition)), reason);
        }
        assert_eq!(suppression_reason(None), None);
        for (state, label) in [
            (EpisodeState::Normal, "normal"),
            (EpisodeState::Candidate, "candidate"),
            (EpisodeState::Open, "open"),
            (EpisodeState::Unknown, "unknown"),
            (EpisodeState::Resolved, "resolved"),
        ] {
            assert_eq!(state.to_string(), label);
        }
    }
}
