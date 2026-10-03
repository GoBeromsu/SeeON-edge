//! Admission and the growing native MP4. Publication oracles live next door.

use std::fs;
use std::os::unix::fs::MetadataExt;
use std::path::Path;
use std::time::Instant;

use super::wait::{poll_until, wall_seconds};
use seeon_deepstream_native::RecordTicket;

use super::support::{BOOT_ID, EVENT_IDENTITY, SOURCE_ID};
use crate::clips::time::Utc;
use crate::seam::Clock;

pub(super) struct Admitted {
    pub ticket: RecordTicket,
    pub detected_at: Utc,
}

pub(super) struct GrownFile {
    pub dev: u64,
    pub ino: u64,
    pub len: u64,
}

pub(super) fn admit(started: &mut super::support::Started) -> Admitted {
    let event = seeon_worker::episode::BusinessEvent {
        domain: "fall".to_owned(),
        event_type: "fall_detected".to_owned(),
        identity: EVENT_IDENTITY.to_owned(),
        camera_id: super::support::CAMERA_ID.to_owned(),
        facility_id: super::support::FACILITY_ID.to_owned(),
        time_sec: wall_seconds(started.clock.wall()),
        probability: None,
        person_id: None,
        bed_id: None,
    };
    let stream = crate::records::builder::Stream {
        camera_id: super::support::CAMERA_ID.to_owned(),
        worker_boot_id: BOOT_ID.to_owned(),
        source_generation: 7,
        stream_epoch: 11,
    };
    let mut prepared = started
        .publications
        .prepare(
            0,
            &event,
            &stream,
            crate::records::builder::Frame {
                frame_seq: 1,
                source_pts_ns: None,
            },
            None,
        )
        .expect("prepare test event");
    let staged = started
        .publications
        .stage(0, &mut prepared, &mut |_| {})
        .expect("stage test event");
    let detected_at = Utc::parse(&staged.detected_at).expect("staged detection time");
    let ticket = match started
        .publications
        .admit_recording(0, &event, &staged)
        .expect("admit recording")
    {
        crate::clips::recorder::Admit::Started(ticket) => ticket,
        other => panic!("production admission did not start native recording: {other:?}"),
    };
    assert_eq!(
        (ticket.source_id, ticket.binding.token, ticket.coalesced),
        (SOURCE_ID, 73, 0)
    );
    assert_eq!(ticket.request_id, 1);
    // record_start can return before the SDK assigns a session. Do not require
    // session_valid here and do not rewrite the ticket the plane returned.
    Admitted {
        ticket,
        detected_at,
    }
}

pub(super) fn growing_mp4(record_dir: &Path, deadline: Instant) -> GrownFile {
    let first = poll_until(deadline, "native MP4 appearance deadline", || {
        mp4_identity(record_dir)
    });
    assert!(first.len > 0, "native MP4 is empty");
    poll_until(deadline, "native MP4 growth deadline", || {
        let again = mp4_identity(record_dir)?;
        (again.dev == first.dev && again.ino == first.ino && again.len > first.len).then_some(again)
    })
}

fn mp4_identity(record_dir: &Path) -> Option<GrownFile> {
    let mut found = Vec::new();
    for entry in fs::read_dir(record_dir).expect("record directory") {
        let path = entry.expect("record entry").path();
        if path.extension().is_some_and(|extension| extension == "mp4") {
            found.push(path);
        }
    }
    let [path] = found.as_slice() else {
        return None;
    };
    let meta = fs::symlink_metadata(path).ok()?;
    if !meta.is_file() || meta.len() == 0 {
        return None;
    }
    Some(GrownFile {
        dev: meta.dev(),
        ino: meta.ino(),
        len: meta.len(),
    })
}
