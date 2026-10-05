use super::support::{CLIP_ID, INVALID_MEDIA, SaveFixture, assert_absent};
use crate::clips::publish::{MANIFEST_FILE, MEDIA_FILE, PublishError, Publisher, TERMINAL_MARKER};
use crate::clips::reserve::FINALIZE_FAILED;
use std::os::unix::fs::MetadataExt;

#[test]
fn failed_codec_probe_refuses_existing_manifest_and_orphan_marker() {
    for evidence in ["manifest", "marker", "both"] {
        let mut fixture = SaveFixture::new(30_000);
        let sidecar = fixture.persist_observation();
        let sidecar_bytes = std::fs::read(&sidecar).unwrap();
        let reservation = fixture.store.reserve("camera-real", CLIP_ID).unwrap();
        // Identical existing negatives would otherwise be resumable, not conflicting.
        let first = Publisher::new(&fixture.queue)
            .publish_unavailable(&reservation, &fixture.metadata(), FINALIZE_FAILED, None)
            .unwrap();
        assert!(
            fixture
                .queue
                .acknowledge_backend(first.entry.entry_id(), 204)
                .unwrap()
        );
        let manifest = reservation.final_dir.join(MANIFEST_FILE);
        let marker = reservation.final_dir.join(TERMINAL_MARKER);
        let marker_bytes = std::fs::read(&marker).unwrap();
        if evidence == "marker" {
            std::fs::remove_file(&manifest).unwrap();
        } else if evidence == "manifest" {
            std::fs::remove_file(&marker).unwrap();
        }
        let artifact = fixture
            .store
            .reserve("camera-real", CLIP_ID)
            .unwrap()
            .artifact_path();
        let video = reservation.final_dir.join(MEDIA_FILE);
        std::fs::write(&artifact, INVALID_MEDIA).unwrap();
        std::fs::write(&video, INVALID_MEDIA).unwrap();

        let saved = fixture.save();

        assert!(
            saved.is_err(),
            "existing {evidence} must refuse conversion: {saved:?}"
        );
        assert_eq!(std::fs::read(&sidecar).unwrap(), sidecar_bytes);
        fixture.assert_retained();
        assert_eq!(std::fs::read(artifact).unwrap(), INVALID_MEDIA);
        assert_eq!(std::fs::read(video).unwrap(), INVALID_MEDIA);
        assert!(fixture.queue.entries().unwrap().is_empty());
        if evidence == "marker" {
            assert_absent(&manifest);
        } else {
            assert_eq!(std::fs::read(manifest).unwrap(), first.manifest_bytes);
        }
        if evidence == "manifest" {
            assert_absent(&marker);
        } else {
            assert_eq!(std::fs::read(marker).unwrap(), marker_bytes);
        }
    }
}

#[test]
fn failed_codec_probe_refuses_malformed_symlink_and_nonregular_terminal_evidence() {
    for name in [MANIFEST_FILE, TERMINAL_MARKER] {
        for kind in ["malformed", "symlink", "dangling-symlink", "directory"] {
            let mut fixture = SaveFixture::new(30_000);
            let sidecar = fixture.persist_observation();
            let bytes = std::fs::read(&sidecar).unwrap();
            let directory = fixture.store.clip_dir(CLIP_ID);
            std::fs::create_dir_all(&directory).unwrap();
            let path = directory.join(name);
            let target = fixture.root.0.join("terminal-target");
            match kind {
                "malformed" => std::fs::write(&path, b"{").unwrap(),
                "directory" => {
                    std::fs::create_dir(&path).unwrap();
                    std::fs::write(path.join("evidence"), b"preserve directory bytes").unwrap();
                }
                _ => {
                    if kind == "symlink" {
                        std::fs::write(&target, b"preserve linked bytes").unwrap();
                    }
                    std::os::unix::fs::symlink(&target, &path).unwrap();
                }
            }
            let before = path.symlink_metadata().unwrap();

            let saved = fixture.save();

            assert!(
                saved.is_err(),
                "{name}/{kind} must refuse conversion: {saved:?}"
            );
            let after = path.symlink_metadata().unwrap();
            assert_eq!(after.ino(), before.ino());
            assert_eq!(after.file_type(), before.file_type());
            match kind {
                "malformed" => assert_eq!(std::fs::read(&path).unwrap(), b"{"),
                "directory" => assert_eq!(
                    std::fs::read(path.join("evidence")).unwrap(),
                    b"preserve directory bytes"
                ),
                _ => {
                    assert_eq!(std::fs::read_link(&path).unwrap(), target);
                    if kind == "symlink" {
                        assert_eq!(std::fs::read(&target).unwrap(), b"preserve linked bytes");
                    } else {
                        assert_absent(&target);
                    }
                }
            }
            assert_eq!(std::fs::read(sidecar).unwrap(), bytes);
            fixture.assert_retained();
            assert!(fixture.queue.entries().unwrap().is_empty());
            let other = if name == MANIFEST_FILE {
                TERMINAL_MARKER
            } else {
                MANIFEST_FILE
            };
            assert_absent(&directory.join(other));
        }
    }
}

#[test]
fn failed_codec_probe_terminal_stat_error_is_not_absence() {
    let mut fixture = SaveFixture::new(30_000);
    let sidecar = fixture.persist_observation();
    let bytes = std::fs::read(&sidecar).unwrap();
    let directory = fixture.store.clip_dir(CLIP_ID);
    std::fs::create_dir_all(directory.parent().unwrap()).unwrap();
    std::fs::write(&directory, b"non-directory blocks terminal metadata").unwrap();

    assert!(fixture.save().is_err());

    assert_eq!(
        std::fs::read(directory).unwrap(),
        b"non-directory blocks terminal metadata"
    );
    assert_eq!(std::fs::read(sidecar).unwrap(), bytes);
    fixture.assert_retained();
    assert!(fixture.queue.entries().unwrap().is_empty());
}

#[test]
fn sidecar_persistence_failure_preserves_erroneous_and_partial_evidence() {
    for partial in [false, true] {
        let mut fixture = SaveFixture::new(30_000);
        std::fs::create_dir(fixture.sidecars.directory()).unwrap();
        let sidecar = fixture.sidecars.directory().join(format!("{CLIP_ID}.json"));
        let blocked = if partial {
            std::fs::write(&sidecar, b"{").unwrap();
            crate::clips::durable::temp_path(&sidecar).unwrap()
        } else {
            sidecar.clone()
        };
        std::fs::create_dir(&blocked).unwrap();
        std::fs::write(blocked.join("evidence"), b"preserve partial attribution").unwrap();

        assert!(matches!(fixture.save(), Err(PublishError::Io(_))));

        let source = fixture
            .sealed
            .path
            .as_ref()
            .expect("fixture reports its source");
        assert_eq!(std::fs::read(source).unwrap(), INVALID_MEDIA);
        assert_eq!(
            std::fs::read(blocked.join("evidence")).unwrap(),
            b"preserve partial attribution"
        );
        if partial {
            assert_eq!(std::fs::read(&sidecar).unwrap(), b"{");
        }
        assert_eq!(
            fixture.sidecars.pending("camera-real").unwrap().malformed,
            vec![sidecar]
        );
        assert!(fixture.queue.entries().unwrap().is_empty());
        fixture.assert_no_terminal();
    }
}

#[test]
fn failed_codec_probe_partial_publication_errors_remain_errors_with_attribution() {
    for boundary in ["queue", "marker", "cleanup"] {
        let mut fixture = SaveFixture::new(30_000);
        let sidecar = fixture.persist_observation();
        let bytes = std::fs::read(&sidecar).unwrap();
        let directory = fixture.store.clip_dir(CLIP_ID);
        std::fs::create_dir_all(&directory).unwrap();
        let blocked = match boundary {
            "queue" => {
                let lock = fixture
                    .queue
                    .directory()
                    .join(crate::delivery::LOCK_FILE_NAME);
                std::fs::remove_file(&lock).unwrap();
                lock
            }
            "marker" => directory.join(format!(".{TERMINAL_MARKER}.tmp")),
            _ => directory.join(MEDIA_FILE),
        };
        std::fs::create_dir(&blocked).unwrap();
        std::fs::write(blocked.join("evidence"), b"preserve publication obstacle").unwrap();

        let saved = fixture.save();

        if boundary == "queue" {
            assert!(matches!(saved, Err(PublishError::Queue(_))), "{saved:?}");
            std::fs::remove_dir_all(&blocked).unwrap();
        } else {
            assert!(matches!(saved, Err(PublishError::Io(_))), "{saved:?}");
            assert_eq!(
                std::fs::read(blocked.join("evidence")).unwrap(),
                b"preserve publication obstacle"
            );
        }
        let manifest = directory.join(MANIFEST_FILE);
        assert!(
            manifest.is_file(),
            "{boundary} failure occurs after manifest publication"
        );
        let manifest_bytes = std::fs::read(&manifest).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&manifest_bytes).unwrap();
        assert_eq!(value["reason_code"], FINALIZE_FAILED);
        let queued = fixture.queue.entries().unwrap();
        assert_eq!(queued.len(), usize::from(boundary != "queue"));
        let marker = directory.join(TERMINAL_MARKER);
        let marker_bytes = if boundary == "cleanup" {
            Some(std::fs::read(&marker).unwrap())
        } else {
            assert_absent(&marker);
            None
        };
        fixture.assert_retained();
        assert_eq!(std::fs::read(&sidecar).unwrap(), bytes);

        assert!(
            fixture.save().is_err(),
            "partial terminal cannot become a completed save"
        );

        assert_eq!(std::fs::read(manifest).unwrap(), manifest_bytes);
        if let Some(expected) = marker_bytes {
            assert_eq!(std::fs::read(marker).unwrap(), expected);
        } else {
            assert_absent(&marker);
        }
        assert_eq!(fixture.queue.entries().unwrap(), queued);
        assert_eq!(std::fs::read(sidecar).unwrap(), bytes);
        fixture.assert_retained();
    }
}
