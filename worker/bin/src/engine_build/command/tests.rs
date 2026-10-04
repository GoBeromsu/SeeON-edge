use std::fs;
use std::path::{Path, PathBuf};

use super::files::{self, CommandError};
use super::{AuxiliaryLayout, BuiltEngines, Captured, Layout, render_served, verify_parser};
use crate::cli::{AuxiliaryBuild, EngineBuildFlags};
use crate::run::model_sources::CapturedOnnx;
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
        auxiliary: AuxiliaryBuild::TensorRt {
            stored_pose_engine: root.join("out/stored.engine"),
            bed_engine: root.join("out/bed.engine"),
            fall_engine: root.join("out/fall.engine"),
        },
        bed_onnx: root.join("bed.onnx"),
        served_infer_config: served.map(|name| root.join(name)),
        image_digest: IMAGE.to_owned(),
        batch_size: 2,
        force,
    }
}

fn tensor_rt_paths(flags: &mut EngineBuildFlags) -> (&mut PathBuf, &mut PathBuf, &mut PathBuf) {
    match &mut flags.auxiliary {
        AuxiliaryBuild::TensorRt {
            stored_pose_engine,
            bed_engine,
            fall_engine,
        } => (stored_pose_engine, bed_engine, fall_engine),
        AuxiliaryBuild::OnnxRuntimeCpu => panic!("TensorRT fixture required"),
    }
}

fn observer_alias(root: &Path, observer: &Path, alias: &str) -> PathBuf {
    use std::os::unix::fs::symlink;
    match alias {
        "direct" => observer.to_path_buf(),
        "parent-symlink" => {
            let parent = root.join("observer-parent");
            symlink(observer.parent().unwrap(), &parent).unwrap();
            parent.join(observer.file_name().unwrap())
        }
        "symlink" | "hardlink" => {
            let path = root.join(format!("observer-{alias}.so"));
            if alias == "symlink" {
                symlink(observer, &path).unwrap();
            } else {
                fs::hard_link(observer, &path).unwrap();
            }
            path
        }
        _ => panic!("unrecognized observer alias"),
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
    let observer_path = root.join("observer.so");
    Captured {
        pose: onnx(&root.join("pose.onnx"), SOURCE),
        bed: onnx(&root.join("bed.onnx"), BED),
        fall: onnx(&root.join("fall/model.onnx"), FALL),
        served,
        served_output: output,
        image: IMAGE.to_owned(),
        observer: files::fingerprint_of(&observer_path).expect("observer"),
        observer_path,
        input_digests: Vec::new(),
    }
}

fn cpu_flags(root: &Path, served: Option<&str>, force: bool) -> EngineBuildFlags {
    let mut requested = flags(root, served, force);
    requested.auxiliary = AuxiliaryBuild::OnnxRuntimeCpu;
    requested
}

fn captured_cpu_sources(root: &Path, requested: &EngineBuildFlags) -> Captured {
    use crate::run::model_sources::capture_image_onnx;
    let (served, served_output) = files::served_text(requested).unwrap();
    let observer_path = root.join("observer.so");
    Captured {
        pose: capture_image_onnx(&requested.onnx).unwrap(),
        bed: capture_image_onnx(&requested.bed_onnx).unwrap(),
        // Hermetic admitted-member fixture; production fall selection remains
        // exclusively with model_sources::capture_fall.
        fall: capture_image_onnx(&root.join("fall/model.onnx")).unwrap(),
        served,
        served_output,
        image: requested.image_digest.clone(),
        observer: files::fingerprint_of(&observer_path).expect("observer"),
        observer_path,
        input_digests: files::capture_inputs(requested).unwrap(),
    }
}

// Synthetic file-contract receipt, never a hardware or execution observation.
fn synthetic_live_receipt(path: &Path, sample: &Captured) -> crate::engine_build::EngineReceipt {
    use crate::records::id::sha256_hex;
    fs::write(path, b"synthetic live engine").unwrap();
    crate::engine_build::EngineReceipt {
        document: serde_json::json!({
            "engine": path.file_name().unwrap().to_str().unwrap(),
            "engine_sha256": sha256_hex(b"synthetic live engine"),
            "onnx_sha256": sample.pose.sha256,
            "image_digest": sample.image,
            "device": 0, "device_name": "synthetic", "trt_version": 1,
            "compute_capability": "1.0", "precision": "fp16", "tf32_enabled": true,
            "input": "images", "observer_library_sha256": sample.observer,
            "min_dimensions": [1, 3, 640, 640],
            "opt_dimensions": [2, 3, 640, 640], "max_dimensions": [2, 3, 640, 640],
        }),
    }
}

fn staged_cpu_fixture(root: &Path) -> (EngineBuildFlags, Captured, Layout, BuiltEngines) {
    write_inputs(root);
    let requested = cpu_flags(root, Some("served.txt"), false);
    let sample = captured_cpu_sources(root, &requested);
    let planned = files::prepare(&requested, &sample).unwrap();
    let layout = files::materialize(planned).unwrap();
    let built = BuiltEngines::OnnxRuntimeCpu(synthetic_live_receipt(&layout.live, &sample));
    (requested, sample, layout, built)
}

fn cpu_cache_fixture(root: &Path) -> (EngineBuildFlags, Captured, Layout) {
    let (requested, sample, layout, built) = staged_cpu_fixture(root);
    files::require_replaceable(&layout, true).unwrap();
    files::authorize(&layout, false, &sample).unwrap();
    files::recheck_inputs(&layout, &sample).unwrap();
    let identity = super::publish_staged(&built, &layout, &requested, &sample).unwrap();
    files::commit(&layout, &identity, true).unwrap();
    (requested, sample, layout)
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
    let live = requested.engine.clone();
    *tensor_rt_paths(&mut requested).0 = live;
    let sample = captured(&scratch.0, template(&requested.parser_lib), false);
    assert!(matches!(
        files::prepare(&requested, &sample),
        Err(CommandError::Collision)
    ));
    let explicit = flags(&scratch.0, Some("infer.txt"), false);
    assert!(files::prepare(&explicit, &sample).is_ok());
}

#[test]
fn observer_aliases_are_refused_for_every_selected_output_even_with_force() {
    for cpu in [false, true] {
        let outputs: &[&str] = if cpu {
            &["live_pose", "served", "identity"]
        } else {
            &[
                "live_pose",
                "stored_pose",
                "bed",
                "fall",
                "served",
                "identity",
            ]
        };
        for force in [false, true] {
            for &output in outputs {
                for alias in ["direct", "parent-symlink", "symlink", "hardlink"] {
                    let scratch = Scratch::new();
                    write_inputs(&scratch.0);
                    let mut requested = if cpu {
                        cpu_flags(&scratch.0, Some("served.txt"), force)
                    } else {
                        flags(&scratch.0, Some("served.txt"), force)
                    };
                    let path = observer_alias(&scratch.0, &scratch.0.join("observer.so"), alias);
                    match output {
                        "live_pose" => requested.engine = path,
                        "stored_pose" => *tensor_rt_paths(&mut requested).0 = path,
                        "bed" => *tensor_rt_paths(&mut requested).1 = path,
                        "fall" => *tensor_rt_paths(&mut requested).2 = path,
                        "served" => requested.served_infer_config = Some(path),
                        "identity" => requested.identity = path,
                        _ => panic!("unrecognized fixture output"),
                    }
                    let (served, served_output) = files::served_text(&requested).unwrap();
                    let sample = captured(&scratch.0, served, served_output);
                    let original = fs::read(&sample.observer_path).unwrap();
                    assert!(
                        matches!(
                            files::prepare(&requested, &sample),
                            Err(CommandError::Collision)
                        ),
                        "cpu={cpu}, force={force}, output={output}, alias={alias}"
                    );
                    assert_eq!(fs::read(&sample.observer_path).unwrap(), original);
                }
            }
        }
    }
}

#[test]
fn infer_template_served_exception_never_admits_observer_aliases() {
    for cpu in [false, true] {
        for force in [false, true] {
            for explicit in [false, true] {
                let scratch = Scratch::new();
                write_inputs(&scratch.0);
                let mut requested = if cpu {
                    cpu_flags(&scratch.0, None, force)
                } else {
                    flags(&scratch.0, None, force)
                };
                requested.served_infer_config = explicit.then(|| requested.infer_config.clone());
                let (served, served_output) = files::served_text(&requested).unwrap();
                let sample = captured(&scratch.0, served, served_output);
                let original = fs::read(&sample.observer_path).unwrap();
                let planned = files::prepare(&requested, &sample).expect("distinct observer");
                assert_eq!(planned.layout.observer, sample.observer_path);
                files::recheck_inputs(&planned.layout, &sample).expect("captured observer");
                for alias in ["direct", "parent-symlink", "symlink", "hardlink"] {
                    let mut aliased = requested.clone();
                    aliased.infer_config = observer_alias(&scratch.0, &sample.observer_path, alias);
                    aliased.served_infer_config = explicit.then(|| aliased.infer_config.clone());
                    assert!(
                        matches!(
                            files::prepare(&aliased, &sample),
                            Err(CommandError::Collision)
                        ),
                        "cpu={cpu}, force={force}, explicit={explicit}, alias={alias}"
                    );
                    assert_eq!(fs::read(&sample.observer_path).unwrap(), original);
                }
            }
        }
    }
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
    *tensor_rt_paths(&mut requested).0 = scratch.0.join("alias/live.engine");
    assert!(matches!(
        files::prepare(&requested, &sample),
        Err(CommandError::Collision)
    ));
    *tensor_rt_paths(&mut requested).0 = scratch.0.join("out/stored.engine");
    fs::hard_link(&sample.fall.source_path, tensor_rt_paths(&mut requested).2).unwrap();
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
    let planned = files::prepare(&requested, &sample).unwrap();
    synthetic_cache(&planned.layout, &sample);
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
    let fall = layout
        .engines()
        .find(|(role, _, _)| *role == "fall")
        .unwrap()
        .2;
    fs::write(fall, "foreign engine bytes").unwrap();
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
    let planned = files::prepare(&requested, &sample).unwrap();
    synthetic_cache(&planned.layout, &sample);
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
    let sample = captured(&scratch.0, "served".to_owned(), true);
    let planned = files::prepare(&requested, &sample).unwrap();
    synthetic_cache(&planned.layout, &sample);
    let layout = files::materialize(planned).unwrap();
    for path in layout.staging() {
        fs::write(path, "new staged bytes").unwrap();
    }
    let old_identity = fs::read(&layout.identity).unwrap();
    let staged_identity = scratch.0.join("new-identity.json");
    fs::write(&staged_identity, "new identity").unwrap();
    // A real filesystem refusal in the commit helper, not a crash or root-race proof.
    let fall = layout
        .engines()
        .find(|(role, _, _)| *role == "fall")
        .unwrap()
        .2;
    fs::remove_file(fall).unwrap();
    fs::create_dir(fall).unwrap();
    assert!(matches!(
        files::commit(&layout, &staged_identity, true),
        Err(CommandError::Path)
    ));
    assert_eq!(fs::read(&layout.final_live).unwrap(), b"new staged bytes");
    assert_eq!(fs::read(&layout.identity).unwrap(), old_identity);
    assert_eq!(fs::read(&staged_identity).unwrap(), b"new identity");
    assert!(files::cache_candidate(&layout, &sample, 2).is_none());
}

#[test]
fn cpu_layout_neither_admits_nor_stages_unused_auxiliary_gpu_outputs() {
    let scratch = Scratch::new();
    write_inputs(&scratch.0);
    for name in ["stored.engine", "bed.engine", "fall.engine"] {
        fs::create_dir(scratch.0.join("out").join(name)).unwrap();
    }
    let requested = cpu_flags(&scratch.0, Some("served.txt"), false);
    let sample = captured_cpu_sources(&scratch.0, &requested);
    let planned = files::prepare(&requested, &sample).unwrap();
    assert_eq!(
        planned
            .layout
            .engines()
            .map(|(role, _, _)| role)
            .collect::<Vec<_>>(),
        ["live_pose"]
    );
    files::require_replaceable(&planned.layout, true).unwrap();
    files::authorize(&planned.layout, false, &sample).unwrap();
    let layout = files::materialize(planned).unwrap();
    let stages: Vec<_> = layout
        .staging()
        .map(|path| path.parent().unwrap())
        .collect();
    assert_eq!(stages.len(), 2);
    assert!(stages.iter().all(|path| path.is_dir()));
    assert!(
        stages[0]
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .starts_with(".engine-stage-live-")
    );
    assert!(
        stages[1]
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .starts_with(".engine-stage-served-")
    );
    for name in ["stored.engine", "bed.engine", "fall.engine"] {
        assert!(scratch.0.join("out").join(name).is_dir());
    }
}

#[test]
fn cpu_output_collisions_and_hardlinked_sources_are_refused_even_with_force() {
    let scratch = Scratch::new();
    write_inputs(&scratch.0);
    let base = cpu_flags(&scratch.0, Some("served.txt"), true);
    let sample = captured_cpu_sources(&scratch.0, &base);
    for source in [
        &base.onnx,
        &base.bed_onnx,
        &sample.fall.source_path,
        &base.parser_lib,
        &base.infer_config,
        &base.tracker_config,
        &base.tracker_library,
    ] {
        let mut requested = base.clone();
        requested.engine = source.clone();
        assert!(matches!(
            files::prepare(&requested, &sample),
            Err(CommandError::Collision)
        ));
    }
    let mut requested = base.clone();
    requested.identity = requested.engine.clone();
    assert!(matches!(
        files::prepare(&requested, &sample),
        Err(CommandError::Collision)
    ));
    requested = base.clone();
    requested.served_infer_config = Some(requested.engine.clone());
    assert!(matches!(
        files::prepare(&requested, &sample),
        Err(CommandError::Collision)
    ));
    requested = base.clone();
    fs::hard_link(&sample.fall.source_path, &requested.identity).unwrap();
    assert!(matches!(
        files::prepare(&requested, &sample),
        Err(CommandError::Collision)
    ));
    fs::remove_file(&requested.identity).unwrap();
    std::os::unix::fs::symlink(scratch.0.join("out"), scratch.0.join("alias")).unwrap();
    requested.identity = scratch.0.join("alias/live.engine");
    assert!(matches!(
        files::prepare(&requested, &sample),
        Err(CommandError::Collision)
    ));
    let explicit_template = cpu_flags(&scratch.0, Some("infer.txt"), false);
    assert!(files::prepare(&explicit_template, &sample).is_ok());
}

#[test]
fn cpu_identity_hashes_captured_bytes_after_sources_are_replaced_and_never_clobbers() {
    use crate::records::id::sha256_hex;
    use serde_json::{Value, json};
    let scratch = Scratch::new();
    let (requested, sample, layout, built) = staged_cpu_fixture(&scratch.0);
    for source in [&sample.pose, &sample.bed, &sample.fall] {
        fs::write(&source.source_path, b"replaced after capture").unwrap();
    }
    files::recheck_inputs(&layout, &sample).unwrap();
    let engines = super::engines_of(&built, &layout, &sample).unwrap();
    let crate::engine_build::EngineSet::OnnxRuntimeCpu { models, .. } = engines else {
        panic!("CPU models must not enter GPU receipts");
    };
    assert!(std::ptr::eq(
        models.stored_pose,
        sample.pose.bytes.as_slice()
    ));
    assert!(std::ptr::eq(models.bed, sample.bed.bytes.as_slice()));
    assert!(std::ptr::eq(models.fall, sample.fall.bytes.as_slice()));
    let destination = super::publish_staged(&built, &layout, &requested, &sample).unwrap();
    let bytes = fs::read(&destination).unwrap();
    let document: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(document["schema_version"], 2);
    assert_eq!(document["batch_size"], 2);
    assert_eq!(
        document["engines"]
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["live_pose"]
    );
    assert_eq!(document["engines"]["live_pose"]["image_digest"], IMAGE);
    assert_eq!(
        document["auxiliary"],
        json!({
            "runtime": "onnxruntime", "provider": "cpu", "models": {
                "stored_pose": {"onnx_sha256": sha256_hex(&sample.pose.bytes)},
                "bed": {"onnx_sha256": sha256_hex(&sample.bed.bytes)},
                "fall": {"onnx_sha256": sha256_hex(&sample.fall.bytes)},
            },
        })
    );
    for (key, path) in [
        ("parser_lib_sha256", &layout.parser_lib),
        ("infer_config_sha256", &layout.served),
        ("tracker_config_sha256", &layout.tracker_config),
        ("tracker_library_sha256", &layout.tracker_library),
    ] {
        assert_eq!(document["flow"][key], sha256_hex(&fs::read(path).unwrap()));
    }
    let repeat = crate::engine_build::publish_identity(crate::engine_build::IdentityRequest {
        engines,
        flow: crate::engine_build::FlowArtifacts {
            parser_lib: &requested.parser_lib,
            infer_config: &layout.served,
            tracker_config: &requested.tracker_config,
            tracker_library: &requested.tracker_library,
        },
        image_digest: &sample.image,
        batch_size: 2,
        destination: &destination,
    });
    assert!(matches!(
        repeat,
        Err(crate::engine_build::IdentityError::Io)
    ));
    assert_eq!(fs::read(&destination).unwrap(), bytes);
    assert!(matches!(
        super::publish_staged(&built, &layout, &requested, &sample),
        Err(CommandError::Io)
    ));
    assert_eq!(fs::read_to_string(&layout.served).unwrap(), sample.served);
    files::commit(&layout, &destination, true).unwrap();
    assert!(files::cache_candidate(&layout, &sample, 2).is_some());
    let fresh = captured_cpu_sources(&scratch.0, &requested);
    assert!(files::cache_candidate(&layout, &fresh, 2).is_none());
}

#[test]
fn cpu_cache_invalidates_each_recaptured_model_but_still_owns_the_live_engine() {
    let scratch = Scratch::new();
    let (requested, sample, layout) = cpu_cache_fixture(&scratch.0);
    assert!(files::cache_candidate(&layout, &sample, 2).is_some());
    for source in [&sample.pose, &sample.bed, &sample.fall] {
        fs::write(&source.source_path, b"changed model bytes").unwrap();
        assert!(
            files::cache_candidate(&layout, &sample, 2).is_some(),
            "captured bytes remain immutable"
        );
        let fresh = captured_cpu_sources(&scratch.0, &requested);
        assert!(files::cache_candidate(&layout, &fresh, 2).is_none());
        files::authorize(&layout, false, &fresh)
            .expect("changed model retains recorded GPU ownership");
        fs::write(&source.source_path, &source.bytes).unwrap();
    }
    assert!(files::cache_candidate(&layout, &sample, 2).is_some());
}

#[test]
fn cpu_cache_requires_exact_schema_and_provider_and_tensor_rt_cannot_reuse_it() {
    use serde_json::{Value, json};
    let scratch = Scratch::new();
    let (_, sample, layout) = cpu_cache_fixture(&scratch.0);
    let valid: Value = serde_json::from_slice(&fs::read(&layout.identity).unwrap()).unwrap();
    for (pointer, value) in [
        ("/schema_version", json!(1)),
        ("/schema_version", json!(3)),
        ("/auxiliary/runtime", json!("tensorrt")),
        ("/auxiliary/provider", json!("cuda")),
        ("/auxiliary/models/bed/onnx_sha256", json!("not-a-hash")),
    ] {
        let mut document = valid.clone();
        *document.pointer_mut(pointer).unwrap() = value;
        fs::write(&layout.identity, serde_json::to_vec(&document).unwrap()).unwrap();
        assert!(
            files::cache_candidate(&layout, &sample, 2).is_none(),
            "{pointer}"
        );
        assert!(matches!(
            files::authorize(&layout, false, &sample),
            Err(CommandError::Target)
        ));
        files::authorize(&layout, true, &sample).unwrap();
    }
    fs::write(&layout.identity, serde_json::to_vec(&valid).unwrap()).unwrap();
    let tensor_rt = flags(&scratch.0, Some("served.txt"), false);
    let planned = files::prepare(&tensor_rt, &sample).unwrap();
    assert!(files::cache_candidate(&planned.layout, &sample, 2).is_none());
    assert!(matches!(
        files::authorize(&planned.layout, false, &sample),
        Err(CommandError::Target)
    ));
    files::authorize(&planned.layout, true, &sample).unwrap();

    let other = Scratch::new();
    write_inputs(&other.0);
    let tensor_rt = flags(&other.0, Some("served.txt"), false);
    let tensor_sample = captured(&other.0, "served".to_owned(), true);
    let prior = files::prepare(&tensor_rt, &tensor_sample).unwrap();
    synthetic_cache(&prior.layout, &tensor_sample);
    let cpu = cpu_flags(&other.0, Some("served.txt"), false);
    let planned = files::prepare(&cpu, &tensor_sample).unwrap();
    assert!(files::cache_candidate(&planned.layout, &tensor_sample, 2).is_none());
    assert!(matches!(
        files::authorize(&planned.layout, false, &tensor_sample),
        Err(CommandError::Target)
    ));
    files::authorize(&planned.layout, true, &tensor_sample).unwrap();
}

#[test]
fn cpu_cache_preserves_image_batch_and_flow_fingerprint_checks() {
    let scratch = Scratch::new();
    let (_, mut sample, layout) = cpu_cache_fixture(&scratch.0);
    sample.image = format!("sha256:{}", "d".repeat(64));
    assert!(files::cache_candidate(&layout, &sample, 2).is_none());
    files::authorize(&layout, false, &sample).unwrap();
    sample.image = IMAGE.to_owned();
    assert!(files::cache_candidate(&layout, &sample, 3).is_none());
    files::authorize(&layout, false, &sample).unwrap();
    for path in [
        &layout.parser_lib,
        &layout.tracker_config,
        &layout.tracker_library,
        &layout.final_served,
        &layout.observer,
    ] {
        let original = fs::read(path).unwrap();
        fs::write(path, b"changed flow artifact").unwrap();
        assert!(files::cache_candidate(&layout, &sample, 2).is_none());
        if path != &layout.final_served {
            assert!(matches!(
                files::recheck_inputs(&layout, &sample),
                Err(CommandError::Io)
            ));
        }
        fs::write(path, original).unwrap();
    }
    assert!(files::cache_candidate(&layout, &sample, 2).is_some());
}

#[test]
fn cpu_foreign_or_nonregular_live_targets_are_refused_without_bypassing_force_safety() {
    let scratch = Scratch::new();
    let (_, sample, layout) = cpu_cache_fixture(&scratch.0);
    fs::write(&layout.final_live, b"foreign engine bytes").unwrap();
    assert!(files::cache_candidate(&layout, &sample, 2).is_none());
    assert!(matches!(
        files::authorize(&layout, false, &sample),
        Err(CommandError::Target)
    ));
    files::authorize(&layout, true, &sample).unwrap();
    fs::remove_file(&layout.final_live).unwrap();
    std::os::unix::fs::symlink(&sample.pose.source_path, &layout.final_live).unwrap();
    assert!(matches!(
        files::require_replaceable(&layout, true),
        Err(CommandError::Path)
    ));
    assert!(matches!(
        files::authorize(&layout, true, &sample),
        Err(CommandError::Path)
    ));
    fs::remove_file(&layout.final_live).unwrap();
    fs::create_dir(&layout.final_live).unwrap();
    assert!(matches!(
        files::require_replaceable(&layout, true),
        Err(CommandError::Path)
    ));
}

#[test]
fn cpu_partial_commit_keeps_old_identity_when_served_publication_refuses() {
    let scratch = Scratch::new();
    let (requested, sample, old_layout) = cpu_cache_fixture(&scratch.0);
    let old_identity = fs::read(&old_layout.identity).unwrap();
    let planned = files::prepare(&requested, &sample).unwrap();
    let layout = files::materialize(planned).unwrap();
    for path in layout.staging() {
        fs::write(path, b"new staged bytes").unwrap();
    }
    let identity = scratch.0.join("next-identity.json");
    fs::write(&identity, b"new identity").unwrap();
    fs::remove_file(&layout.final_served).unwrap();
    fs::create_dir(&layout.final_served).unwrap();
    assert!(matches!(
        files::commit(&layout, &identity, true),
        Err(CommandError::Path)
    ));
    assert_eq!(fs::read(&layout.final_live).unwrap(), b"new staged bytes");
    assert_eq!(fs::read(&layout.identity).unwrap(), old_identity);
    assert_eq!(fs::read(&identity).unwrap(), b"new identity");
    assert!(files::cache_candidate(&layout, &sample, 2).is_none());
}

#[test]
fn cpu_readonly_infer_config_is_verified_without_becoming_an_output() {
    let scratch = Scratch::new();
    write_inputs(&scratch.0);
    let requested = cpu_flags(&scratch.0, None, false);
    let rendered = render_served(
        &template(&requested.parser_lib),
        &requested.engine.display().to_string(),
        2,
    )
    .unwrap();
    fs::write(&requested.infer_config, &rendered).unwrap();
    let sample = captured_cpu_sources(&scratch.0, &requested);
    assert!(!sample.served_output);
    files::verify_existing_served(&sample.served, &requested.engine, 2).unwrap();
    let planned = files::prepare(&requested, &sample).unwrap();
    files::require_replaceable(&planned.layout, false).unwrap();
    files::authorize(&planned.layout, false, &sample).unwrap();
    let layout = files::materialize(planned).unwrap();
    let built = BuiltEngines::OnnxRuntimeCpu(synthetic_live_receipt(&layout.live, &sample));
    files::recheck_inputs(&layout, &sample).unwrap();
    let identity = super::publish_staged(&built, &layout, &requested, &sample).unwrap();
    assert!(!layout.served.exists());
    files::commit(&layout, &identity, false).unwrap();
    assert_eq!(
        fs::read_to_string(&requested.infer_config).unwrap(),
        rendered
    );
    assert!(files::cache_candidate(&layout, &sample, 2).is_some());
    fs::write(&requested.infer_config, b"replaced readonly config").unwrap();
    assert!(matches!(
        files::recheck_inputs(&layout, &sample),
        Err(CommandError::Io)
    ));
    assert!(files::cache_candidate(&layout, &sample, 2).is_none());
}

#[test]
fn engine_receipt_variant_must_match_the_provider_layout() {
    let scratch = Scratch::new();
    let (_, sample, cpu, built_cpu) = staged_cpu_fixture(&scratch.0);
    let tensor_rt = flags(&scratch.0, Some("served.txt"), false);
    let planned = files::prepare(&tensor_rt, &sample).unwrap();
    assert!(matches!(
        super::engines_of(&built_cpu, &planned.layout, &sample),
        Err(CommandError::Config)
    ));
    let built_tensor_rt = BuiltEngines::TensorRt(std::array::from_fn(|_| {
        crate::engine_build::EngineReceipt {
            document: serde_json::Value::Null,
        }
    }));
    assert!(matches!(
        super::engines_of(&built_tensor_rt, &cpu, &sample),
        Err(CommandError::Config)
    ));
}

// Synthetic contract data only: these bytes are not engines or native build receipts.
fn synthetic_cache(layout: &super::Layout, sample: &Captured) {
    use crate::records::id::sha256_hex;
    use serde_json::json;
    fs::write(&layout.final_served, &sample.served).unwrap();
    let mut engines = serde_json::Map::new();
    assert!(matches!(
        &layout.auxiliary,
        AuxiliaryLayout::TensorRt { .. }
    ));
    for (role, _, path) in layout.engines() {
        let (source, input, dimensions) = match role {
            "live_pose" | "stored_pose" => (SOURCE, "images", vec![1, 3, 640, 640]),
            "bed" => (BED, "images", vec![1, 3, 1280, 1280]),
            "fall" => (FALL, "window", vec![1, 30, 56]),
            _ => panic!("unrecognized fixture role"),
        };
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
    fs::write(root.join("observer.so"), b"synthetic observer").expect("observer");
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
