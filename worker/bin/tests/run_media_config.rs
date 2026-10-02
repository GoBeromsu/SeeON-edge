//! Media assembly copies admitted flow and roster fields. It does not open a GPU.
//!
//! Non-default admitted geometry, cache, and reconnect must survive. Image
//! policy (mux timeout, live source, tracker size, queue, record slots,
//! preview cap, file URIs) is asserted because those fields are owned here,
//! against the installed SDK facts rather than against this helper.

use std::path::PathBuf;

use seeon_deepstream_native::{MEDIA_MAX_PREVIEW_BYTES, MEDIA_MAX_SOURCES};
use seeon_ml_worker::msg::{POSE_PER_CAMERA, RECORD_CAPACITY};
use seeon_ml_worker::relay::cameras::RuntimeCamera;
use seeon_ml_worker::run::FlowSettings;
use seeon_ml_worker::run::media_config::{
    ALLOW_FILE_URIS, MUX_BATCH_TIMEOUT_US, MUX_LIVE_SOURCE, MediaAssembly, MediaConfigError,
    TRACKER_HEIGHT, TRACKER_WIDTH, assemble,
};
use std::collections::BTreeMap;

const BOOT: &str = "018f6c6a-7b2e-7c3d-8e4f-1a2b3c4d5e6f";
const OTHER_BOOT: &str = "018f6c6a-7b2e-7c3d-8e4f-1a2b3c4d5e70";

fn flow(batch_size: u32) -> FlowSettings {
    FlowSettings {
        engine: PathBuf::from("/models/pose.engine"),
        identity_path: PathBuf::from("/models/pose.identity.json"),
        infer_config: PathBuf::from("/models/pose-infer.txt"),
        tracker_config: PathBuf::from("/models/tracker.yml"),
        tracker_library: PathBuf::from(
            "/opt/nvidia/deepstream/lib/libnvds_nvmultiobjecttracker.so",
        ),
        onnx: PathBuf::from("/models/pose.onnx"),
        parser_library: PathBuf::from("/models/libpose_parser.so"),
        record_dir: PathBuf::from("/var/lib/seeon/records"),
        record_cache_seconds: 17,
        frame_width: 1280,
        frame_height: 720,
        batch_size,
        rtsp_reconnect_interval_sec: 9,
        identity: BTreeMap::from([("batch_size".to_owned(), batch_size.to_string())]),
    }
}

fn camera(camera_id: &str, rtsp_url: &str) -> RuntimeCamera {
    RuntimeCamera {
        camera_id: camera_id.to_owned(),
        facility_id: "facility-7".to_owned(),
        rtsp_url: rtsp_url.to_owned(),
        fps: 12.5,
        frame_stride: 3,
        decode_backend: Some("nvdec".to_owned()),
        label: Some("hall".to_owned()),
        bed_zone_regions: Vec::new(),
        bed_zone_image_width: None,
        bed_zone_image_height: None,
    }
}

fn roster() -> Vec<RuntimeCamera> {
    vec![
        camera(
            "cam-a",
            "rtsp://viewer:s3cret@10.1.2.3:554/Streaming/Channels/101",
        ),
        camera("cam-b", "rtsp://10.9.8.7/live/bed-2"),
    ]
}

#[test]
fn empty_admitted_roster_is_idle_even_when_batch_is_nonzero() {
    let assembled = assemble(&flow(2), &[], BOOT).expect("empty roster");
    assert!(matches!(assembled, MediaAssembly::Idle));
}

#[test]
fn two_distinct_rtsp_sources_keep_admitted_fields_and_image_policy() {
    let admitted = flow(2);
    let cameras = roster();
    let MediaAssembly::Configured(config) =
        assemble(&admitted, &cameras, BOOT).expect("configured")
    else {
        panic!("nonempty roster must not be idle");
    };

    assert_eq!(config.sources.len(), 2);
    assert_eq!(config.infer_config_path, admitted.infer_config);
    assert_eq!(config.tracker_config_path, admitted.tracker_config);
    assert_eq!(config.tracker_library_path, admitted.tracker_library);
    assert_eq!(config.record_directory, admitted.record_dir);
    assert_eq!(config.record_cache_seconds, 17);
    assert_eq!(
        config.record_capacity,
        u32::try_from(RECORD_CAPACITY).unwrap()
    );
    assert_eq!(config.mux_width, 1280);
    assert_eq!(config.mux_height, 720);
    assert_eq!(config.mux_batch_timeout_us, MUX_BATCH_TIMEOUT_US);
    assert_eq!(MUX_BATCH_TIMEOUT_US, 33_000);
    assert!(!config.mux_live_source);
    assert_eq!(config.mux_live_source, MUX_LIVE_SOURCE);
    assert_eq!(config.tracker_width, TRACKER_WIDTH);
    assert_eq!(config.tracker_height, TRACKER_HEIGHT);
    assert_eq!(config.tracker_width, 960);
    assert_eq!(config.tracker_height, 544);
    assert_eq!(
        config.queue_max_buffers,
        u32::try_from(POSE_PER_CAMERA).unwrap()
    );
    assert_eq!(config.queue_max_buffers, 4);
    assert!(config.preview_enabled);
    assert_eq!(config.max_preview_bytes, MEDIA_MAX_PREVIEW_BYTES);
    assert_eq!(config.max_preview_bytes, 16 * 1024 * 1024);
    assert!(!config.allow_file_uris);
    assert_eq!(config.allow_file_uris, ALLOW_FILE_URIS);
    assert_eq!(config.rtsp_reconnect_interval_sec, 9);

    let tokens: Vec<u64> = config
        .sources
        .iter()
        .map(|source| source.binding.token)
        .collect();
    assert_eq!(tokens, vec![1, 2]);
    assert_eq!(
        tokens
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        2
    );
    for (index, source) in config.sources.iter().enumerate() {
        assert_eq!(source.source_id, u32::try_from(index).unwrap());
        assert_eq!(source.uri, cameras[index].rtsp_url);
        assert_eq!(source.record_prefix, format!("{BOOT}-{index}"));
        assert!(!source.record_prefix.contains("rtsp://"));
        assert!(!source.record_prefix.contains("s3cret"));
        assert!(!source.record_prefix.contains(&cameras[index].camera_id));
    }
    assert_ne!(config.sources[0].uri, config.sources[1].uri);
    assert_ne!(
        config.sources[0].record_prefix,
        config.sources[1].record_prefix
    );
}

#[test]
fn uppercase_rtsp_scheme_is_folded_without_touching_credentials_or_path() {
    let mut cameras = roster();
    cameras[0].rtsp_url = "RTSP://viewer:s3cret@10.1.2.3:554/Streaming/Channels/101".to_owned();
    cameras[1].rtsp_url = "Rtsp://User:PaSS@10.9.8.7/Live/Bed-2?token=AbC".to_owned();
    let MediaAssembly::Configured(config) = assemble(&flow(2), &cameras, BOOT).expect("configured")
    else {
        panic!("nonempty roster must not be idle");
    };
    assert_eq!(
        config.sources[0].uri,
        "rtsp://viewer:s3cret@10.1.2.3:554/Streaming/Channels/101"
    );
    assert_eq!(
        config.sources[1].uri,
        "rtsp://User:PaSS@10.9.8.7/Live/Bed-2?token=AbC"
    );
    assert!(config.sources[1].uri.contains("PaSS"));
    assert!(config.sources[1].uri.contains("Live/Bed-2"));
    assert!(!config.sources[0].uri.starts_with("RTSP://"));
}

#[test]
fn a_second_boot_changes_only_the_prefix() {
    let first = assemble(&flow(2), &roster(), BOOT).expect("first");
    let second = assemble(&flow(2), &roster(), OTHER_BOOT).expect("second");
    let (MediaAssembly::Configured(first), MediaAssembly::Configured(second)) = (first, second)
    else {
        panic!("configured");
    };
    assert_eq!(first.sources[0].binding, second.sources[0].binding);
    assert_ne!(
        first.sources[0].record_prefix,
        second.sources[0].record_prefix
    );
    assert_eq!(second.sources[1].record_prefix, format!("{OTHER_BOOT}-1"));
}

#[test]
fn roster_above_sixteen_and_batch_mismatch_are_refused_without_wrapping() {
    let cameras: Vec<RuntimeCamera> = (0..MEDIA_MAX_SOURCES + 1)
        .map(|index| {
            camera(
                &format!("cam-{index}"),
                &format!("rtsp://10.0.0.{index}/live"),
            )
        })
        .collect();
    assert_eq!(
        assemble(&flow(17), &cameras, BOOT).err(),
        Some(MediaConfigError::RosterLimit {
            cameras: 17,
            limit: MEDIA_MAX_SOURCES,
        })
    );

    let two = roster();
    assert_eq!(
        assemble(&flow(1), &two, BOOT).err(),
        Some(MediaConfigError::BatchMismatch {
            cameras: 2,
            batch_size: 1,
        })
    );
    assert_eq!(
        assemble(&flow(0), &two, BOOT).err(),
        Some(MediaConfigError::BatchMismatch {
            cameras: 2,
            batch_size: 0,
        })
    );
}

#[test]
fn invalid_boot_id_and_non_rtsp_source_are_typed_refusals() {
    let error = assemble(&flow(2), &roster(), "BOOT-not-a-uuid")
        .err()
        .expect("boot");
    assert_eq!(error, MediaConfigError::BootId);

    let mut cameras = roster();
    cameras[1].rtsp_url = "rtsp://user:hidden-pass@10.0.0.8/live".replace("rtsp", "file");
    let error = assemble(&flow(2), &cameras, BOOT).err().expect("uri");
    assert_eq!(error, MediaConfigError::Uri { index: 1 });
}

#[test]
fn admitted_reconnect_zero_is_kept_and_above_one_day_is_refused() {
    let mut admitted = flow(2);
    admitted.rtsp_reconnect_interval_sec = 0;
    let MediaAssembly::Configured(config) =
        assemble(&admitted, &roster(), BOOT).expect("disabled reconnect")
    else {
        panic!("configured");
    };
    assert_eq!(config.rtsp_reconnect_interval_sec, 0);

    admitted.rtsp_reconnect_interval_sec = 86_401;
    assert_eq!(
        assemble(&admitted, &roster(), BOOT).err(),
        Some(MediaConfigError::ReconnectInterval)
    );
}
