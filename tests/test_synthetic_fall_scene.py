"""Pure fixture contracts. Actual Blender rendering is a separate onsite gate."""

from __future__ import annotations

import builtins
import hashlib
import json
import math
import struct
import zlib
from itertools import pairwise, product
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests_support import synthetic_fall_scene as scene


def _glb(document):
    body = json.dumps(document).encode()
    body += b" " * (-len(body) % 4)
    return struct.pack("<4sIII4s", b"glTF", 2, 20 + len(body), len(body), b"JSON") + body


def _actor_document():
    # Minimal metadata for validation, not a renderable or admitted humanoid.
    return {
        "skins": [{"joints": [0]}],
        "meshes": [{}],
        "animations": [
            {
                "name": "declared-first-animation",
                "channels": [{"sampler": 0, "target": {"node": 0, "path": "rotation"}}],
                "samplers": [{"input": 0, "output": 1}],
            }
        ],
    }


def _bed_document():
    return {
        "asset": {"version": "2.0"},
        "buffers": [{"uri": scene.BED_MEMBERS[1][0], "byteLength": scene.BED_MEMBERS[1][1]}],
        "images": [{"uri": name} for name, *_ in scene.BED_MEMBERS[2:]],
    }


@pytest.fixture
def bed_closure(tmp_path, monkeypatch):
    # Synthetic bytes exercise the filesystem verifier with test-only digests.
    # Neither a real glTF import nor a substitute positive inference result.
    root = tmp_path / "bed"
    root.mkdir()
    (root / "textures").mkdir()
    members = []
    for name, size, _digest in scene.BED_MEMBERS:
        data = (
            json.dumps(_bed_document()).encode().ljust(size, b" ")
            if name == scene.BED_GLTF
            else b"\0" * size
        )
        assert len(data) == size
        (root / name).write_bytes(data)
        members.append((name, size, hashlib.sha256(data).hexdigest()))
    monkeypatch.setattr(scene, "BED_MEMBERS", tuple(members))
    return root


def _png():
    def chunk(kind, data):
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    header = struct.pack(">IIBBBBB", scene.WIDTH, scene.HEIGHT, 8, 2, 0, 0, 0)
    pixels = (b"\0" + b"\x80" * (scene.WIDTH * 3)) * scene.HEIGHT
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(pixels))
        + chunk(b"IEND", b"")
    )


@pytest.fixture
def scene_measurements():
    # Receipt serialization input only: no Blender mock, rendered actor, or vision result.
    return {
        "source": "pure-contract-test-not-render-evidence",
        "camera": scene.fit_camera([(0, 0, 0), (1, 2, 1.75)], 0.1),
        "actor_pose_sample": {"contract": scene.actor_pose_contract(), "bone_matrices": []},
        "bed_geometry": {"world_bounds_m": [[1.0, -0.4, 0.0], [3.2, 1.8, 2.0]]},
    }


def test_fall_phases_and_motion_have_exact_boundaries():
    assert [scene.phase_for(frame, "fall") for frame in (0, 149, 150, 179, 180, 359)] == [
        "upright",
        "upright",
        "falling",
        "falling",
        "lying",
        "lying",
    ]
    angles = [scene.fall_angle(frame, "fall") for frame in range(scene.FRAME_COUNT)]
    assert all(angle == 0 for angle in angles[:150])
    assert all(a > b for a, b in pairwise(angles[149:180]))
    assert angles[164] == pytest.approx(-math.pi / 4)
    assert all(angle == -math.pi / 2 for angle in angles[179:])
    assert all(scene.phase_for(frame, "normal") == "upright" for frame in range(360))
    assert all(scene.fall_angle(frame, "normal") == 0 for frame in range(360))


@pytest.mark.parametrize("frame", [-1, 360, True, 1.5])
def test_invalid_frame_is_rejected(frame):
    with pytest.raises(ValueError, match="frame"):
        scene.phase_for(frame, "fall")


@pytest.mark.parametrize("actors", [0, 2, 5, True])
def test_only_frozen_actor_counts_are_allowed(actors):
    with pytest.raises(ValueError, match="actors"):
        scene.actor_positions(actors)


def test_four_actor_positions_form_a_compact_row_without_changing_single_actor_layout():
    positions = scene.actor_positions(4)
    assert positions == ((-1.2, 0.0), (-0.4, 0.0), (0.4, 0.0), (1.2, 0.0))
    assert scene.actor_positions(1) == ((0.0, 0.0),)
    assert len(set(positions)) == 4
    assert [math.dist(a, b) for a, b in pairwise(positions)] == pytest.approx([0.8] * 3)
    assert scene.bed_placement(1) == {"center_xy_m": [2.1, 0.7], "heading_rad": 0.0}
    assert scene.bed_placement(4) == {"center_xy_m": [2.7, 0.7], "heading_rad": 0.0}


@pytest.mark.parametrize("seed", [-1, 2**32, True, 1.5])
def test_invalid_seed_is_rejected(seed):
    with pytest.raises(ValueError, match="seed"):
        scene.heading_for(seed)


def test_seed_selects_only_repeatable_heading_and_never_changes_timing():
    headings = [scene.heading_for(seed) for seed in range(8)]
    assert len(set(headings)) == 8
    assert headings[0] == pytest.approx(-math.pi / 32)
    assert headings[-1] == pytest.approx(3 * math.pi / 128)
    assert scene.heading_for(1) == scene.heading_for(9)
    assert scene.heading_for(2**32 - 1) == headings[7]
    for seed in (0, 1, 7, 2**32 - 1):
        normal = scene.motion_contract("normal", seed)
        fall = scene.motion_contract("fall", seed)
        assert normal["heading_rad"] == fall["heading_rad"] == scene.heading_for(seed)
        assert normal["upright_inclusive"] == [0, 359]
        assert normal["falling_inclusive"] == normal["lying_hold_inclusive"] == []
        assert normal["pitch_end_rad"] == 0
        assert normal["floor_posture"] is None
        assert fall["falling_inclusive"] == [150, 179]
        assert fall["lying_hold_inclusive"] == [180, 359]
        assert fall["floor_posture"] == "lying"
        assert fall["pitch_end_rad"] == scene.fall_angle(359, "fall")


def test_unknown_motion_cannot_be_labeled_upright():
    with pytest.raises(ValueError, match="motion"):
        scene.phase_for(0, "walk")
    with pytest.raises(ValueError, match="motion"):
        scene.motion_contract("walk", 1)


@pytest.mark.parametrize("seed", range(8))
def test_grounding_and_horizontal_sweep_include_between_frame_angles(seed):
    # Asymmetric mathematical vertices, not a surrogate humanoid or Blender mock.
    points = ((-0.4, -0.22, 0), (0.4, 0.15, 0.05), (0, 0.1, 1.75), (0.2, -0.18, 0.85))
    position, heading = (1.3, -0.2), scene.heading_for(seed)
    low, high = scene.swept_xy_bounds(points, position, heading)
    for step in range(901):
        angle = -step * math.pi / 1800
        moved = scene.grounded_points(points, position, angle, heading)
        assert min(point[2] for point in moved) == pytest.approx(scene.GROUND_CLEARANCE, abs=1e-12)
        for point in moved:
            assert all(low[i] - 1e-12 <= point[i] <= high[i] + 1e-12 for i in (0, 1))
        # Rigid placement must preserve pairwise distances, not flatten the mesh.
        assert math.dist(moved[0], moved[2]) == pytest.approx(math.dist(points[0], points[2]))


def _bed_enclosure(actors):
    # Conservative physical allowance, not measured GothicBed geometry or render evidence.
    half_width, half_length = scene.BED_WIDTH_RANGE[1] / 2, scene.BED_LENGTH / 2
    points = product(
        (-half_width, half_width), (-half_length, half_length), (0, scene.BED_HEIGHT_RANGE[1])
    )
    placement = scene.bed_placement(actors)
    x, y = placement["center_xy_m"]
    sine, cosine = math.sin(placement["heading_rad"]), math.cos(placement["heading_rad"])
    return [(x + px * cosine - py * sine, y + px * sine + py * cosine, z) for px, py, z in points]


@pytest.mark.parametrize("actors", [1, 4])
@pytest.mark.parametrize("dimensions", [(1.6, 2.4, 1.5), (2.4, 1.6, 1.5), (2.2, 2.2, 2.5)])
def test_bed_normalization_grounds_centers_and_preserves_shape(actors, dimensions):
    # Algebra only, never substituted for the imported asset during rendering.
    origin = (-3, 2, -0.7)
    points = list(product(*((origin[i], origin[i] + dimensions[i]) for i in range(3))))
    transform = scene.bed_normalization(points, actors)
    assert transform == scene.bed_normalization(points, actors)
    scale = transform["uniform_scale"]
    angle = transform["rotation_euler_rad"][2]
    x, y, z = transform["location_m"]
    moved = [
        (
            x + scale * (px * math.cos(angle) - py * math.sin(angle)),
            y + scale * (px * math.sin(angle) + py * math.cos(angle)),
            z + scale * pz,
        )
        for px, py, pz in points
    ]
    low, high = scene.geometry_bounds(moved)
    assert low[2] == pytest.approx(0, abs=1e-12)
    assert [(low[i] + high[i]) / 2 for i in (0, 1)] == pytest.approx(
        scene.bed_placement(actors)["center_xy_m"]
    )
    assert [high[i] - low[i] for i in range(3)] == pytest.approx(
        [min(dimensions[:2]) * scale, 2.2, dimensions[2] * scale]
    )
    assert transform["axis_alignment_rad"] == (math.pi / 2 if dimensions[0] > dimensions[1] else 0)
    for before_a, before_b, after_a, after_b in zip(
        points, points[1:], moved, moved[1:], strict=False
    ):
        assert math.dist(after_a, after_b) == pytest.approx(math.dist(before_a, before_b) * scale)


@pytest.mark.parametrize(
    "points",
    [
        [],
        [(0, 0, math.nan)],
        [(0, 0)],
        [(0, 0, 0), (0, 1, 1)],
        [(0, 0, 0), (5, 2, 1)],
        [(0, 0, 0), (0.2, 2.2, 1)],
        [(0, 0, 0), (1, 2.2, 0.2)],
        [(0, 0, 0), (1, 2.2, 3)],
    ],
)
def test_bed_normalization_rejects_unbounded_or_degenerate_geometry(points):
    with pytest.raises(ValueError, match="geometry|bounds"):
        scene.bed_normalization(points, 1)


def test_bed_normalization_has_a_vertex_budget():
    points = [(0, 0, 0), (1.6, 2.4, 1.5)] * (scene.BED_VERTEX_LIMIT // 2)
    with pytest.raises(ValueError, match="bounded"):
        scene.bed_normalization(points, 4)


def _project(camera, point):
    # Independent pinhole projection from the recorded camera transform.
    x, y, z = (point[i] - camera["location_m"][i] for i in range(3))
    pitch = camera["rotation_euler_rad"][0]
    depth = y * math.sin(pitch) - z * math.cos(pitch)
    up = y * math.cos(pitch) + z * math.sin(pitch)
    tangent_x = camera["sensor_width_mm"] / (2 * camera["lens_mm"])
    tangent_y = tangent_x * scene.HEIGHT / scene.WIDTH
    return 0.5 + x / (2 * depth * tangent_x), 0.5 + up / (2 * depth * tangent_y), depth


@pytest.mark.parametrize("seed", range(8))
def test_fixed_perspective_envelope_has_headroom_between_angular_samples(seed):
    # This wide mathematical box fits only the one-actor layout, not the compact row.
    # It is not rendered/detected geometry; four-box overlap is tested separately.
    vertices = tuple(product((-0.662, 0.662), (-0.182, 0.182), (0.0, 1.75)))
    positions, heading = scene.actor_positions(1), scene.heading_for(seed)
    bed_points = _bed_enclosure(1)
    envelope = scene.camera_envelope([vertices], positions, heading, bed_points)
    camera = envelope["camera"]
    assert envelope == scene.camera_envelope([vertices], positions, heading, bed_points)
    assert camera["projection"] == "PERSP"
    assert camera["continuous_motion_padding_m"] > 0
    checked = list(bed_points)
    for step in range(361):
        for position in positions:
            checked.extend(
                scene.grounded_points(vertices, position, -step * math.pi / 720, heading)
            )
    for point in checked:
        u, v, depth = _project(camera, point)
        assert scene.CAMERA_MARGIN <= u <= 1 - scene.CAMERA_MARGIN
        assert scene.CAMERA_MARGIN <= v <= 1 - scene.CAMERA_MARGIN
        assert camera["clip_m"][0] < depth < camera["clip_m"][1]


@pytest.mark.parametrize("seed", range(8))
def test_camera_limits_vertical_scale_bias_without_flattening_the_lying_axis(seed):
    # Pinhole geometry only: these axial segments are not detected body parts.
    # Keep the original wide enclosure; it cannot clear the compact four-actor row.
    vertices = tuple(product((-0.662, 0.662), (-0.182, 0.182), (0.0, 1.75)))
    positions, heading = scene.actor_positions(1), scene.heading_for(seed)
    camera = scene.camera_envelope([vertices], positions, heading, _bed_enclosure(1))["camera"]
    probes = (*vertices, (0, 0, 0), (0, 0, scene.ACTOR_HEIGHT))
    for position in positions:
        upright = [
            _project(camera, point)
            for point in scene.grounded_points(probes, position, 0, heading)[-2:]
        ]
        lying = [
            _project(camera, point)
            for point in scene.grounded_points(probes, position, -math.pi / 2, heading)[-2:]
        ]
        # Inverse depth is local magnification: head/foot bias stays below 15%.
        assert 1 < upright[0][2] / upright[1][2] < 1.15
        upright_span = upright[1][1] - upright[0][1]
        lying_span = lying[1][1] - lying[0][1]
        assert 0 < 0.5 * upright_span < lying_span < upright_span


def test_clearance_rejects_overlapping_people_or_bed_without_dropping_actors():
    actor = ((-0.4, -0.2), (0.4, 1.8))
    adjacent_bed = ((0.7, -0.2), (1.8, 2.0))
    scene.check_path_clearance([actor, adjacent_bed])
    with pytest.raises(ValueError, match="clearance"):
        scene.check_path_clearance([actor, actor, adjacent_bed])
    with pytest.raises(ValueError, match="clearance"):
        scene.check_path_clearance([actor, ((0.41, -0.2), (1.8, 2.0))])


@pytest.mark.parametrize(
    ("points", "padding"),
    [([], 0), ([(0, 0, 0)], -1), ([(0, 0, 0)], math.inf), ([(math.nan, 0, 0)], 0)],
)
def test_camera_rejects_invalid_bounds(points, padding):
    with pytest.raises(ValueError, match="camera"):
        scene.fit_camera(points, padding)


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"version https://git-lfs.github.com/spec/v1\n",
        struct.pack("<4sIII4s", b"glTF", 1, 20, 0, b"JSON"),
        struct.pack("<4sIII4s", b"nope", 2, 20, 0, b"JSON"),
        struct.pack("<4sIII4s", b"glTF", 2, 24, 0, b"JSON"),
    ],
)
def test_glb_refuses_bad_magic_version_length_and_pointer(data):
    with pytest.raises(ValueError):
        scene.validate_glb(data)


def test_glb_requires_skin_and_embedded_resources():
    with pytest.raises(ValueError, match="skinned"):
        scene.validate_glb(_glb({"meshes": [{}]}))
    for resource in ("buffers", "images"):
        for uri in (
            "https://invalid/model.bin",
            "../outside.bin",
            # Assembled at runtime so the repository privacy scanner does not flag it.
            "data" + ":application/octet-stream;base64,AA==",
        ):
            with pytest.raises(ValueError, match="URI"):
                scene.validate_glb(_glb({"skins": [{}], "meshes": [{}], resource: [{"uri": uri}]}))
    with pytest.raises(ValueError, match="chunk"):
        data = _glb({"skins": [{}], "meshes": [{}]}) + b"junk"
        data = data[:8] + struct.pack("<I", len(data)) + data[12:]
        scene.validate_glb(data)


def test_glb_selects_only_the_declared_first_animation():
    assert scene.validate_glb(_glb(_actor_document())) == {
        "index": 0,
        "name": "declared-first-animation",
        "channel_count": 1,
    }


@pytest.mark.parametrize("failure", ["missing", "multiple", "channels", "samplers", "foreign-node"])
def test_glb_requires_unambiguous_skin_animation(failure):
    document = _actor_document()
    if failure == "missing":
        document.pop("animations")
    elif failure == "multiple":
        document["animations"] *= 2
    elif failure in {"channels", "samplers"}:
        document["animations"][0][failure] = []
    else:
        document["animations"][0]["channels"][0]["target"]["node"] = 42
    with pytest.raises(ValueError, match="animation"):
        scene.validate_glb(_glb(document))


def _pose_curve(keyframes=((0.0, 0.0), (30.0, 1.0)), extrapolation="CONSTANT"):
    return SimpleNamespace(
        data_path='pose.bones["imported-joint"].rotation_quaternion',
        keyframe_points=[SimpleNamespace(co=key) for key in keyframes],
        extrapolation=extrapolation,
        mute=False,
        modifiers=[],
        sampled_points=[],
    )


@pytest.mark.parametrize(
    "key_times,extrapolation,endpoint_count",
    [
        ((0.0, 30.0), "CONSTANT", 0),
        ((-30.0, 0.0), "LINEAR", 0),
        ((-30.0, 30.0), "LINEAR", 0),
        ((1.2499985694885254, 60.0), "CONSTANT", 1),
        ((-60.0, -1.2499985694885254), "CONSTANT", 1),
        ((1.2499985694885254, 60.0), "LINEAR", None),
        ((-60.0, -1.2499985694885254), "LINEAR", None),
        ((1.2499985694885254,), "CONSTANT", 1),
        ((0.0,), "LINEAR", 0),
    ],
)
def test_action_sample_checks_each_curve_not_aggregate_range(
    key_times, extrapolation, endpoint_count
):
    # A broad channel covers zero; the other channel must satisfy its own boundary.
    action = SimpleNamespace(
        frame_range=(-60.0, 60.0),
        fcurves=[
            _pose_curve(((-60.0, 0.0), (60.0, 1.0))),
            _pose_curve(
                tuple((time, index) for index, time in enumerate(key_times)), extrapolation
            ),
        ],
    )
    if endpoint_count is None:
        with pytest.raises(ValueError, match="CONSTANT extrapolation"):
            scene._validate_action_sample(action)
    else:
        assert scene._validate_action_sample(action) == endpoint_count


@pytest.mark.parametrize(
    "keyframes",
    [
        (),
        ((math.nan, 0.0), (30.0, 1.0)),
        ((0.0, 0.0), (math.inf, 1.0)),
        ((0.0, -math.inf), (30.0, 1.0)),
        ((0.0, 0.0), (30.0, math.nan)),
        ((0.0, 0.0), (0.0, 1.0)),
        ((30.0, 0.0), (0.0, 1.0)),
    ],
)
def test_action_sample_rejects_empty_nonfinite_or_ambiguous_curve_keys(keyframes):
    action = SimpleNamespace(
        frame_range=(0.0, 30.0), fcurves=[_pose_curve(), _pose_curve(keyframes)]
    )
    with pytest.raises(ValueError, match="keyframe curves"):
        scene._validate_action_sample(action)


@pytest.mark.parametrize(
    "attribute,value",
    [("mute", True), ("modifiers", [object()]), ("sampled_points", [object()])],
)
def test_action_sample_rejects_unsupported_curves(attribute, value):
    curve = _pose_curve()
    setattr(curve, attribute, value)
    action = SimpleNamespace(frame_range=(0.0, 30.0), fcurves=[_pose_curve(), curve])
    with pytest.raises(ValueError, match="keyframe curves"):
        scene._validate_action_sample(action)


@pytest.mark.parametrize(
    "frame_range,has_curves",
    [
        ((math.nan, 30.0), True),
        ((0.0, math.inf), True),
        ((30.0, 0.0), True),
        ((0.0, 30.0), False),
    ],
)
def test_action_sample_rejects_invalid_or_empty_action(frame_range, has_curves):
    action = SimpleNamespace(frame_range=frame_range, fcurves=[_pose_curve()] if has_curves else [])
    with pytest.raises(ValueError, match="action range and keyed FCurves"):
        scene._validate_action_sample(action)


@pytest.mark.parametrize("first_key", [0.0, 1.2499985694885254])
@pytest.mark.parametrize(
    "failure",
    [
        None,
        "no-rig",
        "no-action",
        "ambiguous",
        "constant",
        "no-bone-curves",
        "nonconstant-extrapolation",
    ],
)
def test_pose_sampling_snapshots_before_clearing_and_restores_pose(monkeypatch, first_key, failure):
    # Adapter protocol/order only: this does not simulate Blender's skinning or inference.
    class Matrix(list):
        def copy(self):
            return Matrix([list(row) for row in self])

    identity = Matrix([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]])
    sampled = identity.copy()
    sampled[0][3] = 0.35
    bone = SimpleNamespace(name="imported-joint", matrix_basis=identity.copy())
    calls = []

    class Owner:
        def __init__(self, kind, data=None):
            self.type, self.data = kind, data
            self.animation_data = None
            self.matrix_basis = identity.copy()

        def animation_data_clear(self):
            calls.append("clear")
            self.animation_data = None
            self.matrix_basis = identity.copy()
            if self is rig:
                bone.matrix_basis = identity.copy()

    class Action:
        name = "imported-only-action"
        frame_range = (first_key, 60.0)
        fcurves = [
            _pose_curve(
                ((first_key, 0.0), (60.0, 0.0 if failure == "constant" else 1.0)),
            )
        ]

    action = Action()
    if failure == "no-bone-curves":
        action.fcurves[0].data_path = "location"
    elif failure == "nonconstant-extrapolation":
        action.fcurves[0].keyframe_points[0].co = (1.2499985694885254, 0.0)
        action.fcurves[0].extrapolation = "LINEAR"
    strip = SimpleNamespace(action=action, frame_start=1.0, frame_end=59.75)
    rig = Owner("ARMATURE", Owner("ARMATURE_DATA"))
    rig.pose = SimpleNamespace(bones=[bone])
    rig.animation_data = SimpleNamespace(
        action=action,
        drivers=[],
        nla_tracks=[SimpleNamespace(strips=[strip])],
        use_nla=True,
    )
    if failure == "no-action":
        rig.animation_data = None
    elif failure == "ambiguous":
        rig.animation_data.nla_tracks = [SimpleNamespace(strips=[SimpleNamespace(action=Action())])]
    mesh = Owner("MESH", Owner("MESH_DATA"))
    mesh.data.shape_keys = None
    imported = [mesh] if failure == "no-rig" else [rig, mesh]

    def frame_set(frame, *, subframe):
        calls.append(("sample", frame, subframe))
        assert rig.data.pose_position == "POSE"
        assert rig.animation_data.use_nla is False
        bone.matrix_basis = sampled.copy()
        rig.matrix_basis = sampled.copy()

    bpy = SimpleNamespace(
        context=SimpleNamespace(
            scene=SimpleNamespace(frame_set=frame_set),
            view_layer=SimpleNamespace(update=lambda: None),
        )
    )
    monkeypatch.setattr(
        scene,
        "_mesh_points",
        lambda *_args: [(0, 0, 0), (bone.matrix_basis[0][3], 0, 1)],
    )
    if failure is not None:
        with pytest.raises(ValueError, match="armature|animation"):
            scene._sample_actor_pose(bpy, imported, {"index": 0})
        assert calls == []
    else:
        record = scene._sample_actor_pose(bpy, imported, {"index": 0})
        assert calls == [("sample", 0, 0.0), "clear", "clear", "clear", "clear"]
        assert rig.data.pose_position == "POSE"
        assert rig.animation_data is None
        assert rig.matrix_basis == bone.matrix_basis == sampled
        assert record["bone_matrices"] == [{"bone": bone.name, "matrix_basis": sampled}]
        assert record["contract"] == scene.actor_pose_contract()
        assert record["action_frame_range"] == [first_key, 60.0]
        assert record["constant_endpoint_curve_count"] == (0 if first_key == 0 else 1)
        assert [key.co for key in action.fcurves[0].keyframe_points] == [
            (first_key, 0.0),
            (60.0, 1.0),
        ]
        assert (strip.frame_start, strip.frame_end) == (1.0, 59.75)
        assert record["sampled_vertex_count"] == 2


def test_bed_verifier_accepts_only_the_complete_test_pinned_closure(bed_closure):
    scene.verify_bed_asset(bed_closure)
    scene.validate_bed_gltf((bed_closure / scene.BED_GLTF).read_bytes())


@pytest.mark.parametrize("member", [name for name, *_ in scene.BED_MEMBERS])
@pytest.mark.parametrize("failure", ["missing", "size", "digest", "symlink", "directory"])
def test_bed_verifier_rejects_invalid_members(bed_closure, member, failure):
    path = bed_closure / member
    data = path.read_bytes()
    if failure == "size":
        path.write_bytes(data + b"x")
    elif failure == "digest":
        path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    else:
        path.unlink()
        if failure == "symlink":
            target = bed_closure.parent / "outside-member"
            target.write_bytes(data)
            path.symlink_to(target)
        elif failure == "directory":
            path.mkdir()
    with pytest.raises(ValueError, match="missing|regular|SHA-256"):
        scene.verify_bed_asset(bed_closure)


@pytest.mark.parametrize("directory", [".", "textures"])
def test_bed_verifier_rejects_symlink_directories(bed_closure, directory):
    path = bed_closure if directory == "." else bed_closure / directory
    target = bed_closure.parent / "outside-directory"
    path.rename(target)
    path.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="nonsymlink"):
        scene.verify_bed_asset(bed_closure)


@pytest.mark.parametrize("member", ["extra.gltf", "textures/extra.jpg", "nested", "flat-diff.jpg"])
def test_bed_verifier_rejects_unexpected_closure_members(bed_closure, member):
    (bed_closure / member).write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="unexpected"):
        scene.verify_bed_asset(bed_closure)


def test_bed_verifier_rejects_missing_root_and_wrong_root_type(tmp_path):
    root = tmp_path / "bed"
    with pytest.raises(ValueError, match="missing"):
        scene.verify_bed_asset(root)
    root.write_bytes(b"not a directory")
    with pytest.raises(ValueError, match="nonsymlink"):
        scene.verify_bed_asset(root)


@pytest.mark.parametrize("resource", ["buffers", "images"])
@pytest.mark.parametrize(
    "uri",
    [
        "https://invalid/asset.bin",
        "//invalid/asset.bin",
        "/asset.bin",
        "../GothicBed_01.bin",
        "textures/../../GothicBed_01.bin",
        "textures/../GothicBed_01_diff_1k.jpg",
        "textures\\GothicBed_01_diff_1k.jpg",
        "textures%2FGothicBed_01_diff_1k.jpg",
        "./GothicBed_01.bin",
        "GothicBed_01_diff_1k.jpg",
        "data" + ":image/jpeg;base64,AA==",  # runtime-assembled for the privacy scanner
        "file:///asset.bin",
        "",
        None,
    ],
)
def test_bed_gltf_rejects_nonliteral_or_external_uri_mapping(resource, uri):
    document = _bed_document()
    document[resource][0]["uri"] = uri
    with pytest.raises(ValueError, match="URI"):
        scene.validate_bed_gltf(json.dumps(document).encode())


@pytest.mark.parametrize(
    "failure", ["missing", "duplicate", "embedded", "extra-uri", "buffer-length", "wrong-version"]
)
def test_bed_gltf_rejects_inexact_resource_closure(failure):
    document = _bed_document()
    if failure == "missing":
        document["images"].pop()
    elif failure == "duplicate":
        document["images"][0] = document["images"][1]
    elif failure == "embedded":
        document["images"][0]["bufferView"] = 0
    elif failure == "extra-uri":
        document["extras"] = {"nested": [{"uri": "https://invalid/unexpected.jpg"}]}
    elif failure == "buffer-length":
        document["buffers"][0]["byteLength"] += 1
    else:
        document["asset"]["version"] = "1.0"
    with pytest.raises(ValueError, match="closure|URI|glTF"):
        scene.validate_bed_gltf(json.dumps(document).encode())


@pytest.mark.parametrize("invalid", [None, "foreign", "skinned"])
def test_editor_bone_shapes_are_removed_without_touching_actor_geometry(invalid):
    class Imported:
        def __init__(self, kind, modifiers=()):
            self.type = kind
            self.modifiers = list(modifiers)

    shape = Imported("MESH")
    actor = Imported("MESH", [SimpleNamespace(type="ARMATURE")])
    bone = SimpleNamespace(custom_shape=shape)
    rig = Imported("ARMATURE")
    rig.pose = SimpleNamespace(bones=[bone])
    imported = [rig, actor, shape]
    removed = []
    bpy = SimpleNamespace(
        data=SimpleNamespace(
            objects=SimpleNamespace(
                remove=lambda obj, *, do_unlink: removed.append((obj, do_unlink))
            )
        )
    )
    if invalid == "foreign":
        imported.remove(shape)
    elif invalid == "skinned":
        shape.modifiers.append(SimpleNamespace(type="ARMATURE"))
    if invalid is not None:
        with pytest.raises(ValueError, match="exclusively imported editor geometry"):
            scene._remove_bone_display_shapes(bpy, imported)
        assert bone.custom_shape is shape
        assert removed == []
    else:
        assert scene._remove_bone_display_shapes(bpy, imported) == [rig, actor]
        assert bone.custom_shape is None
        assert removed == [(shape, True)]
        assert actor.modifiers[0].type == "ARMATURE"


def test_compact_row_clears_frontal_bed_for_recorded_seed_one_continuous_bounds():
    # Parent-reported pinned CesiumMan seed-1 sweep, before this layout translation.
    # This is bounds arithmetic, not new render evidence or admission for other seeds.
    original_position = (-2.4, 0.0)
    original_bounds = (
        (-2.678314454034722, -0.5550720092277555),
        (-2.035554364155222, 1.746290635478406),
    )
    paths = [
        tuple(
            tuple(bound[axis] + position[axis] - original_position[axis] for axis in (0, 1))
            for bound in original_bounds
        )
        for position in scene.actor_positions(4)
    ]
    bed = scene.bed_placement(4)
    assert bed["heading_rad"] == 0.0
    x, y = bed["center_xy_m"]
    # Parent-reported normalized bed half-width, rounded to six decimal places.
    half_width, half_length = 0.805454, scene.BED_LENGTH / 2
    bed_bounds = ((x - half_width, y - half_length), (x + half_width, y + half_length))
    assert len(set(paths)) == 4
    gaps = [right[0][0] - left[1][0] for left, right in pairwise(paths)]
    assert gaps == pytest.approx([0.15724] * 3, abs=1e-6)
    bed_gap = bed_bounds[0][0] - paths[-1][1][0]
    assert bed_gap == pytest.approx(0.33010, abs=1e-6)
    assert min(*gaps, bed_gap) > scene.PATH_CLEARANCE
    scene.check_path_clearance([*paths, bed_bounds])
    # A maximum-width allowance is not the pinned bed: it must still fail clearance.
    wide_bed_bounds = [point[:2] for point in scene.geometry_bounds(_bed_enclosure(4))]
    with pytest.raises(ValueError, match="clearance"):
        scene.check_path_clearance([*paths, wide_bed_bounds])


@pytest.mark.parametrize("seed", range(8))
def test_compact_row_rejects_the_unchanged_wider_mathematical_enclosure(seed):
    # Keep the unsupported wide body intact: no shrinking to fit the declared row.
    vertices = tuple(product((-0.662, 0.662), (-0.182, 0.182), (0.0, 1.75)))
    with pytest.raises(ValueError, match="clearance"):
        scene.camera_envelope(
            [vertices] * 4, scene.actor_positions(4), scene.heading_for(seed), _bed_enclosure(4)
        )


def test_asset_hash_pin_cannot_be_overridden(tmp_path):
    path = tmp_path / "not-the-asset.glb"
    path.write_bytes(b"x" * scene.ASSET_BYTES)
    with pytest.raises(ValueError, match="declaration"):
        scene.verify_asset(path, "0" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        scene.verify_asset(path, scene.ASSET_SHA256)


def test_bad_asset_never_imports_bpy(tmp_path, monkeypatch):
    original = builtins.__import__
    attempted = []

    def guarded(name, *args, **kwargs):
        if name == "bpy":
            attempted.append(name)
            raise AssertionError("bpy imported before admission")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(ValueError, match="regular GLB"):
        scene.render(
            tmp_path / "missing",
            tmp_path / "new",
            bed_asset_root=tmp_path / "missing-bed",
            actors=1,
            motion="fall",
            seed=1,
            single_frame=0,
            declared_sha256=scene.ASSET_SHA256,
        )
    assert attempted == []
    assert not (tmp_path / "new").exists()


def test_unverified_bed_never_imports_bpy_or_creates_output(tmp_path, monkeypatch, bed_closure):
    original = builtins.__import__
    attempted = []

    def guarded(name, *args, **kwargs):
        if name == "bpy":
            attempted.append(name)
            raise AssertionError("bpy imported before bed verification")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(scene, "verify_asset", lambda *_args: {"index": 0})
    texture = bed_closure / scene.BED_MEMBERS[-1][0]
    data = texture.read_bytes()
    texture.write_bytes(b"x" + data[1:])
    with pytest.raises(ValueError, match="SHA-256"):
        scene.render(
            Path("unused"),
            tmp_path / "new",
            bed_asset_root=bed_closure,
            actors=1,
            motion="normal",
            seed=1,
            single_frame=0,
            declared_sha256=scene.ASSET_SHA256,
        )
    assert attempted == []
    assert not (tmp_path / "new").exists()


def test_cli_requires_explicit_bed_asset_root(capsys):
    with pytest.raises(SystemExit) as exc:
        scene.main(
            [
                "--asset",
                "unused.glb",
                "--sha256",
                scene.ASSET_SHA256,
                "--output-dir",
                "unused",
                "--actors",
                "1",
                "--motion",
                "fall",
            ]
        )
    assert exc.value.code == 2
    assert "--bed-asset-root" in capsys.readouterr().err


def test_existing_output_is_never_modified(tmp_path, monkeypatch):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep"
    marker.write_bytes(b"existing")
    monkeypatch.setattr(scene, "verify_asset", lambda *_args: None)
    monkeypatch.setattr(scene, "verify_bed_asset", lambda *_args: None)
    with pytest.raises(FileExistsError, match="overwrite"):
        scene.render(
            Path("unused"),
            output,
            bed_asset_root=Path("unused-bed"),
            actors=1,
            motion="fall",
            seed=1,
            single_frame=0,
            declared_sha256=scene.ASSET_SHA256,
        )
    assert marker.read_bytes() == b"existing"


def test_single_frame_receipt_is_nonqualifying_and_private_paths_absent(
    tmp_path, scene_measurements
):
    (tmp_path / "frame_0180.png").write_bytes(_png())
    receipt = scene.rendered_receipt(
        tmp_path,
        actors=1,
        motion="fall",
        seed=1,
        frames=(180,),
        blender_version=(4, 0, 2),
        scene_measurements=scene_measurements,
    )
    assert receipt["render_complete"] is False
    assert receipt["qualifying_full_corpus_member"] is False
    assert receipt["inference_qualification"] is False
    assert receipt["model_admission"] is False
    assert receipt["single_frame_smoke"] is True
    assert receipt["frames"][0]["phase"] == "lying"
    assert receipt["frames"][0]["pitch_rad"] == -math.pi / 2
    assert receipt["scene_measurements"] == scene_measurements
    assert receipt["motion_contract"] == scene.motion_contract("fall", 1)
    assert receipt["scene_contract"] == scene.scene_contract()
    assert receipt["assets"]["actor"]["sha256"] == scene.ASSET_SHA256
    assert receipt["assets"]["actor"]["bytes"] == scene.ASSET_BYTES
    assert receipt["assets"]["actor"]["license"] == "CC-BY-4.0"
    assert receipt["assets"]["bed"] == scene.bed_asset_contract()
    assert (
        receipt["generator_sha256"] == hashlib.sha256(Path(scene.__file__).read_bytes()).hexdigest()
    )
    assert receipt["frames"][0]["sha256"] == hashlib.sha256(_png()).hexdigest()
    assert receipt == scene.rendered_receipt(
        tmp_path,
        actors=1,
        motion="fall",
        seed=1,
        frames=(180,),
        blender_version=(4, 0, 2),
        scene_measurements=scene_measurements,
    )
    assert str(tmp_path) not in json.dumps(receipt)


def test_full_receipt_requires_every_frame_and_only_exact_renderer(tmp_path, scene_measurements):
    with pytest.raises(ValueError, match="missing"):
        scene.rendered_receipt(
            tmp_path,
            actors=4,
            motion="fall",
            seed=1,
            frames=tuple(range(360)),
            blender_version=(4, 0, 2),
            scene_measurements=scene_measurements,
        )
    with pytest.raises(ValueError, match="renderer"):
        scene.rendered_receipt(
            tmp_path,
            actors=1,
            motion="fall",
            seed=1,
            frames=(0,),
            blender_version=(4, 1, 0),
            scene_measurements=scene_measurements,
        )
    (tmp_path / "frame_0000.png").write_bytes(b"not png")
    with pytest.raises(ValueError, match="PNG"):
        scene.rendered_receipt(
            tmp_path,
            actors=1,
            motion="fall",
            seed=1,
            frames=(0,),
            blender_version=(4, 0, 2),
            scene_measurements=scene_measurements,
        )
    with pytest.raises(ValueError, match="unique"):
        scene.rendered_receipt(
            tmp_path,
            actors=1,
            motion="fall",
            seed=1,
            frames=(0, 0),
            blender_version=(4, 0, 2),
            scene_measurements=scene_measurements,
        )


def test_full_frame_set_never_implies_corpus_or_model_admission(tmp_path, scene_measurements):
    image = _png()
    for frame in range(360):
        (tmp_path / f"frame_{frame:04d}.png").write_bytes(image)
    receipt = scene.rendered_receipt(
        tmp_path,
        actors=4,
        motion="normal",
        seed=1,
        frames=tuple(range(360)),
        blender_version=(4, 0, 2),
        scene_measurements=scene_measurements,
    )
    assert receipt["render_complete"] is True
    assert receipt["qualifying_full_corpus_member"] is False
    assert receipt["inference_qualification"] is False
    assert receipt["model_admission"] is False
    assert [frame["frame"] for frame in receipt["frames"]] == list(range(360))
    assert all(
        frame["phase"] == "upright" and frame["pitch_rad"] == 0 for frame in receipt["frames"]
    )
    assert "unverified" in receipt["admission_status"]


def test_manifest_matches_executable_asset_and_renderer_contract():
    manifest = json.loads((Path(__file__).parent / "fixtures/synthetic-scene-v1.json").read_text())
    assert manifest["asset"]["digest"]["pinned_hex"] == scene.ASSET_SHA256
    assert manifest["asset"]["byte_length"] == scene.ASSET_BYTES
    assert manifest["asset"]["revision"] == scene.ASSET_REVISION
    assert manifest["bed_asset"] == scene.bed_asset_contract()
    assert manifest["render_contract"]["actor_pose"] == scene.actor_pose_contract()
    assert tuple(manifest["blender"]["provisioned"]) == scene.BLENDER_VERSION
    assert manifest["scene"]["frame_count"] == scene.FRAME_COUNT
    assert manifest["scene"]["width"] == scene.WIDTH
    assert manifest["scene"]["height"] == scene.HEIGHT
    assert manifest["scene"]["ingest_fps"] == scene.FPS
    assert manifest["render_contract"] == scene.scene_contract()
    for motion in manifest["scene"]["variants"]:
        assert manifest["motion_contracts"][motion] == scene.motion_contract(
            motion, manifest["scene"]["seed"]
        )
    assert manifest["admission"]["inference_qualification"] is False
    assert manifest["admission"]["model_admission"] is False
