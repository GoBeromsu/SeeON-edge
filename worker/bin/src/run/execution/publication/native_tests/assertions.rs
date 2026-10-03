//! Admission and the growing native MP4. Publication oracles live next door.

use std::collections::BTreeSet;
use std::fs;
use std::os::unix::fs::MetadataExt;
use std::path::Path;
use std::time::Instant;

use super::wait::{poll_until, wall_seconds};
use seeon_deepstream_native::RecordTicket;

use super::support::{BOOT_ID, CAMERA_ZERO, Camera, EVENT_IDENTITY};
use crate::clips::time::Utc;
use crate::seam::Clock;

pub(super) struct Admitted {
    pub source_index: usize,
    pub ticket: RecordTicket,
    pub detected_at: Utc,
    pub event_ref: &'static str,
    pub camera_id: &'static str,
    pub record_prefix: &'static str,
}

#[derive(Clone, Copy)]
pub(super) struct GrownFile {
    pub dev: u64,
    pub ino: u64,
    pub len: u64,
}

pub(super) fn admit(started: &mut super::support::Started) -> Admitted {
    let admitted = admit_event(started, EVENT_IDENTITY);
    assert_eq!(admitted.ticket.request_id, 1);
    admitted
}

pub(super) fn admit_event(
    started: &mut super::support::Started,
    event_ref: &'static str,
) -> Admitted {
    admit_for_source(started, 0, &CAMERA_ZERO, event_ref)
}

pub(super) fn admit_for_source(
    started: &mut super::support::Started,
    source_index: usize,
    camera: &Camera,
    event_ref: &'static str,
) -> Admitted {
    assert_eq!(
        u32::try_from(source_index).expect("source index"),
        camera.source_id
    );
    let event = seeon_worker::episode::BusinessEvent {
        domain: "fall".to_owned(),
        event_type: "fall_detected".to_owned(),
        identity: event_ref.to_owned(),
        camera_id: camera.camera_id.to_owned(),
        facility_id: camera.facility_id.to_owned(),
        time_sec: wall_seconds(started.clock.wall()),
        probability: None,
        person_id: None,
        bed_id: None,
    };
    let stream = crate::records::builder::Stream {
        camera_id: camera.camera_id.to_owned(),
        worker_boot_id: BOOT_ID.to_owned(),
        source_generation: camera.binding.generation,
        stream_epoch: camera.binding.epoch,
    };
    let mut prepared = started
        .publications
        .prepare(
            source_index,
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
        .stage(source_index, &mut prepared, &mut |_| {})
        .expect("stage test event");
    let detected_at = Utc::parse(&staged.detected_at).expect("staged detection time");
    let ticket = match started
        .publications
        .admit_recording(source_index, &event, &staged)
        .expect("admit recording")
    {
        crate::clips::recorder::Admit::Started(ticket) => ticket,
        other => panic!("production admission did not start native recording: {other:?}"),
    };
    assert_eq!(
        (ticket.source_id, ticket.binding, ticket.coalesced),
        (camera.source_id, camera.binding, 0)
    );
    assert_ne!(ticket.request_id, 0);
    // record_start can return before the SDK assigns a session. Do not require
    // session_valid here and do not rewrite the ticket the plane returned.
    Admitted {
        source_index,
        ticket,
        detected_at,
        event_ref,
        camera_id: camera.camera_id,
        record_prefix: camera.record_prefix,
    }
}

pub(super) fn growing_mp4(record_dir: &Path, deadline: Instant) -> GrownFile {
    growing_mp4_for(record_dir, CAMERA_ZERO.record_prefix, deadline)
}

pub(super) fn growing_mp4_for(
    record_dir: &Path,
    record_prefix: &str,
    deadline: Instant,
) -> GrownFile {
    let first = poll_until(deadline, "native MP4 appearance deadline", || {
        mp4_identity(record_dir, record_prefix)
    });
    assert!(first.len > 0, "native MP4 is empty");
    poll_until(deadline, "native MP4 growth deadline", || {
        let again = mp4_identity(record_dir, record_prefix)?;
        (again.dev == first.dev && again.ino == first.ino && again.len > first.len).then_some(again)
    })
}

pub(super) fn growing_started_mp4_for(
    started: &super::support::Started,
    record_prefix: &str,
    deadline: Instant,
) -> GrownFile {
    let first = poll_until(deadline, "native MP4 appearance deadline", || {
        started.drain_pose_packets();
        mp4_identity(&started.record_dir, record_prefix)
    });
    assert!(first.len > 0, "native MP4 is empty");
    poll_until(deadline, "native MP4 growth deadline", || {
        started.drain_pose_packets();
        let again = mp4_identity(&started.record_dir, record_prefix)?;
        (again.dev == first.dev && again.ino == first.ino && again.len > first.len).then_some(again)
    })
}

fn mp4_identity(record_dir: &Path, record_prefix: &str) -> Option<GrownFile> {
    let mut found = Vec::new();
    for entry in fs::read_dir(record_dir).expect("record directory") {
        let path = entry.expect("record entry").path();
        if path.extension().is_some_and(|extension| extension == "mp4")
            && path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| matches_record_prefix(name, record_prefix))
        {
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

fn matches_record_prefix(name: &str, prefix: &str) -> bool {
    name.strip_prefix(prefix)
        .is_some_and(|suffix| suffix.starts_with('_'))
}

#[test]
fn source_prefix_must_end_at_the_native_filename_separator() {
    assert!(matches_record_prefix("synthetic_0_00000.mp4", "synthetic"));
    assert!(!matches_record_prefix(
        "synthetic-two_1_00001.mp4",
        "synthetic"
    ));
    assert!(matches_record_prefix(
        "synthetic-two_1_00001.mp4",
        "synthetic-two"
    ));
}

pub(super) fn assert_exact_native_files(record_dir: &Path, expected: &[&str]) {
    let actual: BTreeSet<_> = fs::read_dir(record_dir)
        .expect("record directory")
        .map(|entry| entry.expect("record entry").path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "mp4"))
        .map(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .expect("MP4 filename")
                .to_owned()
        })
        .collect();
    let expected: BTreeSet<_> = expected.iter().map(|name| (*name).to_owned()).collect();
    assert_eq!(
        actual, expected,
        "native MP4 files do not match the receipts"
    );
    assert_eq!(expected.len(), 2);
}
