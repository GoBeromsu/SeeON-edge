use std::fs;
use std::path::{Path, PathBuf};

use super::files::{self, CommandError};
use super::sources::CapturedOnnx;
use super::{Captured, render_served, verify_parser};
use crate::cli::EngineBuildFlags;
use crate::seam::{IdSource, RandomIds};

const IMAGE: &str = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
const SOURCE: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const BED: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const FALL: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            ".engine-command-{}",
            RandomIds.uuid4().expect("id")
        ));
        fs::create_dir(&path).expect("scratch");
        Self(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.0) {
            eprintln!("owned command fixture cleanup failed: {error}");
            assert!(
                std::thread::panicking(),
                "owned fixture cleanup must succeed"
            );
        }
    }
}

fn flags(root: &Path, served: Option<&str>, force: bool) -> EngineBuildFlags {
    EngineBuildFlags {
        onnx: root.join("pose.onnx"),
        engine: root.join("out/live.engine"),
        identity: root.join("out/identity.json"),
        parser_lib: root.join("parser.so"),
        infer_config: root.join("infer.txt"),
        tracker_config: root.join("tracker.txt"),
        tracker_library: root.join("tracker.so"),
        stored_pose_engine: root.join("out/stored.engine"),
        bed_onnx: root.join("bed.onnx"),
        bed_engine: root.join("out/bed.engine"),
        fall_engine: root.join("out/fall.engine"),
        served_infer_config: served.map(|name| root.join(name)),
        image_digest: IMAGE.to_owned(),
        batch_size: 2,
        force,
    }
}

fn template(parser: &Path) -> String {
    format!(
        "[property]\nkeep=1\nmodel-engine-file=old.engine\nbatch-size=9\ncustom-lib-path={}\nparse-bbox-func-name=kept\n",
        parser.display()
    )
}

fn onnx(path: &Path, sha: &str) -> CapturedOnnx {
    CapturedOnnx {
        bytes: b"bytes".to_vec(),
        sha256: sha.to_owned(),
        source_path: path.to_path_buf(),
    }
}

fn captured(root: &Path, served: String, output: bool) -> Captured {
    Captured {
        pose: onnx(&root.join("pose.onnx"), SOURCE),
        bed: onnx(&root.join("bed.onnx"), BED),
        fall: onnx(&root.join("fall/model.onnx"), FALL),
        served,
        served_output: output,
        image: IMAGE.to_owned(),
        observer: SOURCE.to_owned(),
        input_digests: Vec::new(),
    }
}

#[test]
fn serving_removes_construction_input_and_rewrites_engine_and_batch() {
    let scratch = Scratch::new();
    let parser = scratch.0.join("parser.so");
    let rendered = render_served(&template(&parser), "/cache/live.engine", 2).expect("rendered");
    assert_eq!(rendered.matches("onnx-file=").count(), 0);
    assert_eq!(rendered.matches("model-engine-file=").count(), 1);
    assert_eq!(rendered.matches("batch-size=").count(), 1);
    assert!(rendered.contains("keep=1\n") && rendered.contains("parse-bbox-func-name=kept\n"));
    assert!(rendered.contains("model-engine-file=/cache/live.engine\n"));
    assert!(rendered.contains("batch-size=2\n"));
    verify_parser(&rendered, &parser).expect("parser");
    let build_template = template(&parser) + "onnx-file=/captured/input.onnx\n";
    assert_eq!(
        render_served(&build_template, "/cache/live.engine", 2).unwrap(),
        rendered
    );
    assert!(matches!(
        render_served(
            &(build_template + "onnx-file=/other.onnx\n"),
            "/b.engine",
            1
        ),
        Err(CommandError::Config)
    ));
}

#[test]
fn colliding_outputs_are_refused_and_explicit_template_output_is_allowed() {
    let scratch = Scratch::new();
    fs::create_dir_all(scratch.0.join("out")).expect("out");
    write_inputs(&scratch.0);
    let mut requested = flags(&scratch.0, None, false);
    requested.stored_pose_engine = requested.engine.clone();
    let sample = captured(&scratch.0, template(&requested.parser_lib), false);
    assert!(matches!(
        files::prepare(&requested, &sample),
        Err(CommandError::Collision)
    ));
    let explicit = flags(&scratch.0, Some("infer.txt"), false);
    assert!(files::prepare(&explicit, &sample).is_ok());
}

#[test]
fn none_served_keeps_template_and_verifies_engine_batch() {
    let scratch = Scratch::new();
    write_inputs(&scratch.0);
    let requested = flags(&scratch.0, None, false);
    let (text, output) = files::served_text(&requested).expect("readonly");
    assert!(!output);
    assert!(text.contains("model-engine-file=old.engine"));
    assert!(!text.contains("onnx-file="));
    let engine = scratch.0.join("out/live.engine");
    let rendered = render_served(&text, &engine.display().to_string(), 2).expect("render");
    fs::write(&requested.infer_config, &rendered).expect("rewrite");
    let (verified, still_input) = files::served_text(&requested).expect("verified template");
    assert!(!still_input);
    files::verify_existing_served(&verified, &engine, 2).expect("engine and batch");
    assert!(matches!(
        files::verify_existing_served(
            &(verified.clone() + " onnx-file = /would-build.onnx\n"),
            &engine,
            2
        ),
        Err(CommandError::Config)
    ));
    let sample = captured(&scratch.0, verified.clone(), false);
    let planned = files::prepare(&requested, &sample).expect("readonly plan");
    files::require_replaceable(&planned.layout, false).expect("outputs only");
    files::authorize(&planned.layout, false, &sample).expect("template is not an output");
    assert_eq!(
        fs::read_to_string(&requested.infer_config).unwrap(),
        verified
    );
}

#[test]
fn unknown_identity_requires_force_when_no_engines_exist() {
    let scratch = Scratch::new();
    write_inputs(&scratch.0);
    let requested = flags(&scratch.0, Some("served.txt"), false);
    let sample = captured(&scratch.0, "served".to_owned(), true);
    let planned = files::prepare(&requested, &sample).expect("plan");
    fs::write(&requested.identity, br#"{"schema_version":1}"#).expect("identity");
    assert!(matches!(
        files::authorize(&planned.layout, false, &sample),
        Err(CommandError::Target)
    ));
    files::authorize(&planned.layout, true, &sample).expect("force");
}

#[test]
fn absent_outputs_with_aliased_parents_and_hardlinked_sources_are_refused() {
    let scratch = Scratch::new();
    write_inputs(&scratch.0);
    std::os::unix::fs::symlink(scratch.0.join("out"), scratch.0.join("alias")).unwrap();
    let mut requested = flags(&scratch.0, Some("served.txt"), true);
    let sample = captured(&scratch.0, "served".to_owned(), true);
    requested.stored_pose_engine = scratch.0.join("alias/live.engine");
    assert!(matches!(
        files::prepare(&requested, &sample),
        Err(CommandError::Collision)
    ));
    requested.stored_pose_engine = scratch.0.join("out/stored.engine");
    fs::hard_link(&sample.fall.source_path, &requested.fall_engine).unwrap();
    assert!(matches!(
        files::prepare(&requested, &sample),
        Err(CommandError::Collision)
    ));
}

#[test]
fn changed_image_source_and_batch_are_misses_but_owned_outputs_can_be_rebuilt() {
    let scratch = Scratch::new();
    write_inputs(&scratch.0);
    let requested = flags(&scratch.0, Some("served.txt"), false);
    let mut sample = captured(&scratch.0, "served".to_owned(), true);
    let mut planned = files::prepare(&requested, &sample).unwrap();
    synthetic_cache(&mut planned.layout, &mut sample);
    let layout = &planned.layout;
    assert!(files::cache_candidate(layout, &sample, 2).is_some());
    files::authorize(layout, false, &sample).expect("recorded outputs");
    sample.image = format!("sha256:{}", "d".repeat(64));
    assert!(files::cache_candidate(layout, &sample, 2).is_none());
    files::authorize(layout, false, &sample).expect("changed requested image");
    sample.image = IMAGE.to_owned();
    sample.pose.sha256 = "d".repeat(64);
    assert!(files::cache_candidate(layout, &sample, 2).is_none());
    files::authorize(layout, false, &sample).expect("changed captured model");
    sample.pose.sha256 = SOURCE.to_owned();
    assert!(files::cache_candidate(layout, &sample, 3).is_none());
    files::authorize(layout, false, &sample).expect("changed requested batch");
    fs::write(&layout.final_fall, "foreign engine bytes").unwrap();
    assert!(matches!(
        files::authorize(layout, false, &sample),
        Err(CommandError::Target)
    ));
}

#[test]
fn flow_input_replacement_is_detected_before_publication() {
    let scratch = Scratch::new();
    write_inputs(&scratch.0);
    let requested = flags(&scratch.0, Some("served.txt"), false);
    let mut sample = captured(&scratch.0, "served".to_owned(), true);
    let mut planned = files::prepare(&requested, &sample).unwrap();
    synthetic_cache(&mut planned.layout, &mut sample);
    sample.input_digests = files::capture_inputs(&requested).unwrap();
    files::recheck_inputs(&planned.layout, &sample).expect("unchanged captures");
    fs::write(&requested.parser_lib, "changed parser").unwrap();
    assert!(matches!(
        files::recheck_inputs(&planned.layout, &sample),
        Err(CommandError::Io)
    ));
}

#[test]
fn partial_file_commit_never_publishes_the_new_identity() {
    let scratch = Scratch::new();
    write_inputs(&scratch.0);
    let requested = flags(&scratch.0, Some("served.txt"), false);
    let mut sample = captured(&scratch.0, "served".to_owned(), true);
    let mut planned = files::prepare(&requested, &sample).unwrap();
    synthetic_cache(&mut planned.layout, &mut sample);
    let layout = files::materialize(planned).unwrap();
    for path in [
        &layout.live,
        &layout.stored,
        &layout.bed,
        &layout.fall,
        &layout.served,
    ] {
        fs::write(path, "new staged bytes").unwrap();
    }
    let old_identity = fs::read(&layout.identity).unwrap();
    let staged_identity = scratch.0.join("new-identity.json");
    fs::write(&staged_identity, "new identity").unwrap();
    // A real filesystem refusal in the commit helper, not a crash or root-race proof.
    fs::remove_file(&layout.final_fall).unwrap();
    fs::create_dir(&layout.final_fall).unwrap();
    assert!(matches!(
        files::commit(&layout, &staged_identity, true),
        Err(CommandError::Path)
    ));
    assert_eq!(fs::read(&layout.final_live).unwrap(), b"new staged bytes");
    assert_eq!(fs::read(&layout.identity).unwrap(), old_identity);
    assert_eq!(fs::read(&staged_identity).unwrap(), b"new identity");
    assert!(files::cache_candidate(&layout, &sample, 2).is_none());
}

// Synthetic contract data only: these bytes are not engines or native build receipts.
fn synthetic_cache(layout: &mut super::Layout, sample: &mut Captured) {
    use crate::records::id::sha256_hex;
    use serde_json::json;
    layout.observer = layout.identity.parent().unwrap().join("observer.so");
    fs::write(&layout.observer, "synthetic observer").unwrap();
    sample.observer = sha256_hex(b"synthetic observer");
    fs::write(&layout.final_served, &sample.served).unwrap();
    let mut engines = serde_json::Map::new();
    for (role, path, source, input, dimensions) in [
        (
            "live_pose",
            &layout.final_live,
            SOURCE,
            "images",
            vec![1, 3, 640, 640],
        ),
        (
            "stored_pose",
            &layout.final_stored,
            SOURCE,
            "images",
            vec![1, 3, 640, 640],
        ),
        (
            "bed",
            &layout.final_bed,
            BED,
            "images",
            vec![1, 3, 1280, 1280],
        ),
        ("fall", &layout.final_fall, FALL, "window", vec![1, 30, 56]),
    ] {
        fs::write(path, role).unwrap();
        let mut entry = json!({
            "engine":path.file_name().unwrap().to_str().unwrap(),
            "engine_sha256":sha256_hex(role.as_bytes()), "onnx_sha256":source,
            "image_digest":IMAGE, "device":0, "device_name":"synthetic",
            "trt_version":1, "compute_capability":"1.0",
            "precision":"fp32", "tf32_enabled":false, "input":input,
            "dimensions":dimensions
        });
        if role == "live_pose" {
            entry.as_object_mut().unwrap().remove("dimensions");
            entry["precision"] = json!("fp16");
            entry["tf32_enabled"] = json!(true);
            entry["observer_library_sha256"] = json!(sample.observer);
            entry["min_dimensions"] = json!([1, 3, 640, 640]);
            entry["opt_dimensions"] = json!([2, 3, 640, 640]);
            entry["max_dimensions"] = json!([2, 3, 640, 640]);
        }
        engines.insert(role.to_owned(), entry);
    }
    let mut flow = serde_json::Map::new();
    for (key, path) in [
        ("parser_lib_sha256", &layout.parser_lib),
        ("infer_config_sha256", &layout.final_served),
        ("tracker_config_sha256", &layout.tracker_config),
        ("tracker_library_sha256", &layout.tracker_library),
    ] {
        flow.insert(key.to_owned(), json!(sha256_hex(&fs::read(path).unwrap())));
    }
    fs::write(
        &layout.identity,
        serde_json::to_vec(&json!({
            "schema_version":1, "batch_size":2, "engines":engines, "flow":flow
        }))
        .unwrap(),
    )
    .unwrap();
}

fn write_inputs(root: &Path) {
    fs::create_dir_all(root.join("out")).expect("out");
    fs::create_dir_all(root.join("fall")).expect("fall input");
    fs::write(root.join("fall/model.onnx"), "fall").expect("fall source");
    for name in [
        "pose.onnx",
        "bed.onnx",
        "parser.so",
        "tracker.txt",
        "tracker.so",
    ] {
        fs::write(root.join(name), name).expect(name);
    }
    fs::write(root.join("infer.txt"), template(&root.join("parser.so"))).expect("infer");
}
