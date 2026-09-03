from worker.domains.episode import EpisodeAuthority, EpisodeProposal, EpisodeState


def proposal(*, track_id=7, frame=0, time=0.0, qualifying=True, recovery=False):
    return EpisodeProposal(
        camera_id="camera-1",
        facility_id="facility-1",
        event_type="fall",
        track_id=track_id,
        bed_id=None,
        frame_index=frame,
        time_sec=time,
        qualifying=qualifying,
        confirmed_recovery=recovery,
        probability=0.9,
        domain="fall",
        generation=0,
    )


def open_episode(authority, *, track_id=7, start=0):
    events = ()
    for offset in range(3):
        frame = start + offset
        events = authority.propose(proposal(track_id=track_id, frame=frame, time=frame / 15))
    return events


def test_stable_episode_emits_once_and_recovery_rearms_distinct_identity():
    authority = EpisodeAuthority(boot_id="boot", stream_epoch="epoch", source_generation=1)
    first = open_episode(authority)
    assert len(first) == 1
    assert authority.propose(proposal(frame=3, time=0.2)) == ()
    assert authority.propose(proposal(frame=4, time=0.3, qualifying=False, recovery=True)) == ()
    second = open_episode(authority, start=5)
    assert len(second) == 1
    assert second[0].identity != first[0].identity


def test_track_switch_inside_window_is_absorbed_once():
    authority = EpisodeAuthority(boot_id="boot", stream_epoch="epoch", source_generation=1)
    assert open_episode(authority)
    authority.track_lost(camera_id="camera-1", frame_index=3, time_sec=0.2)
    assert authority.reassociate(proposal(track_id=8, frame=10, time=0.6), previous_track_id=7)
    assert authority.track_id_switch_absorbed_total == 1
    assert authority.propose(proposal(track_id=8, frame=11, time=0.7)) == ()


def test_unknown_window_expiry_resolves_and_timeout_does_not_rearm():
    authority = EpisodeAuthority(boot_id="boot", stream_epoch="epoch", source_generation=1)
    assert open_episode(authority)
    authority.track_lost(camera_id="camera-1", frame_index=3, time_sec=0.2)
    authority.expire(frame_index=79, time_sec=5.3)
    assert (
        authority.state_for(camera_id="camera-1", event_type="fall", bed_id=None, track_id=7)
        is EpisodeState.RESOLVED
    )
    assert authority.propose(proposal(frame=80, time=5.4)) == ()
