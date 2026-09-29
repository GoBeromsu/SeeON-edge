"""Pinned, non-identifying Blender fixture. No downloading or inference.

Run with Blender 4.0.2: --background --python this_file -- --asset PATH
--sha256 PINNED_SHA256 --bed-asset-root DIRECTORY --output-dir NEW_DIRECTORY
--actors 1 --motion fall.
All output is render evidence only, including complete 360-frame sequences.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import struct
import sys
from itertools import pairwise
from pathlib import Path

ASSET_SHA256 = "b7001eaeea8254bd44773bcd247e78696d94169388fbb2a1800fc69434e777d9"
ASSET_BYTES = 438044
ASSET_REVISION = "03251428e295f20d8c4a65ddbbd7dafe4f251c6d"
BED_GLTF = "GothicBed_01_1k.gltf"
BED_MEMBERS = (
    (BED_GLTF, 2683, "67df189dd3a645936827f3cc123fb9e97ca9d2201d1e4090762bf477cab82fc0"),
    (
        "GothicBed_01.bin",
        658648,
        "2276e86480c49dfb63a4671a9637621f88727708cf83342156c157a1d89f747b",
    ),
    (
        "textures/GothicBed_01_diff_1k.jpg",
        116672,
        "f7e8611a731a87a42a35d92e59f78b848ac20475827b79cb8aa34df0dd5bcbd5",
    ),
    (
        "textures/GothicBed_01_nor_gl_1k.jpg",
        196568,
        "27533a33617689035cf9880803ed19af45550864442734ee9898697933b7319f",
    ),
    (
        "textures/GothicBed_01_arm_1k.jpg",
        180642,
        "049014a0bb54fd3da3b555d90966b63cad7e35feb5470b578524a7b1cec8506a",
    ),
)
BLENDER_VERSION = (4, 0, 2)
FRAME_COUNT = 360
WIDTH, HEIGHT, FPS = 640, 360, 30
ACTOR_HEIGHT = 1.75
GROUND_CLEARANCE = 0.025
PATH_CLEARANCE = 0.08
CAMERA_ELEVATION = math.radians(35)
CAMERA_LENS_MM = 70.0
CAMERA_SENSOR_MM = 36.0
CAMERA_MARGIN = 0.06
ENVELOPE_STEPS = 30
BED_LENGTH = 2.2
BED_WIDTH_RANGE = (0.8, 2.2)
BED_HEIGHT_RANGE = (0.35, 2.5)
BED_VERTEX_LIMIT = 200000
HUMAN_SURFACES = (
    ("shoes", 0.12, (0.045, 0.035, 0.03)),
    ("trousers", 0.86, (0.065, 0.09, 0.12)),
    ("shirt", 1.46, (0.16, 0.27, 0.42)),
    ("skin", 1.70, (0.68, 0.39, 0.23)),
    ("scalp", ACTOR_HEIGHT, (0.08, 0.045, 0.025)),
)


def actor_animation(document: dict) -> dict:
    """Only the pinned asset's single animation can be selected without name guesses."""
    animations = document.get("animations", [])
    if (
        not isinstance(animations, list)
        or len(animations) != 1
        or not isinstance(animations[0], dict)
    ):
        raise ValueError("actor requires exactly one real animation; select index 0")
    animation = animations[0]
    joints = {joint for skin in document.get("skins", []) for joint in skin.get("joints", [])}
    channels, samplers = animation.get("channels", []), animation.get("samplers", [])
    if not samplers or not any(
        channel.get("target", {}).get("node") in joints
        and channel["target"].get("path") in {"translation", "rotation", "scale"}
        and type(channel.get("sampler")) is int
        and 0 <= channel["sampler"] < len(samplers)
        for channel in channels
    ):
        raise ValueError("actor animation must drive its actual skin joints")
    return {"index": 0, "name": animation.get("name"), "channel_count": len(channels)}


def validate_glb(data: bytes) -> dict:
    """Refuse malformed/external-resource input before Blender can import it."""
    if len(data) < 20:
        raise ValueError("truncated GLB header")
    magic, version, size = struct.unpack_from("<4sII", data)
    if magic != b"glTF" or version != 2 or size != len(data):
        raise ValueError("invalid GLB magic, version, or declared size")
    length, kind = struct.unpack_from("<II", data, 12)
    if kind != 0x4E4F534A or length % 4 or 20 + length > len(data):
        raise ValueError("invalid GLB JSON chunk")
    document = json.loads(data[20 : 20 + length])
    if not isinstance(document, dict) or not document.get("skins") or not document.get("meshes"):
        raise ValueError("asset must contain a skinned humanoid mesh")
    for resource in (*document.get("buffers", []), *document.get("images", [])):
        if "uri" in resource:
            raise ValueError("external or URI resources are prohibited")
    offset = 20 + length
    while offset < len(data):
        if offset + 8 > len(data):
            raise ValueError("truncated GLB chunk")
        chunk_length, _kind = struct.unpack_from("<II", data, offset)
        if chunk_length % 4 or offset + 8 + chunk_length > len(data):
            raise ValueError("invalid GLB chunk length")
        offset += 8 + chunk_length
    return actor_animation(document)


def _read_pinned_file(path: Path, size: int, digest: str) -> bytes:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"missing regular asset member: {path.name}") from exc
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != size:
        raise ValueError(f"asset member is not a pinned regular file: {path.name}")
    with path.open("rb") as stream:
        data = stream.read(size + 1)
    if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"asset SHA-256 or byte length mismatch: {path.name}")
    return data


def verify_asset(path: Path, declared_sha256: str) -> dict:
    if declared_sha256 != ASSET_SHA256:
        raise ValueError("asset declaration differs from the pinned manifest")
    if path.is_symlink() or not path.is_file() or path.stat().st_size != ASSET_BYTES:
        raise ValueError("asset is not the pinned regular GLB")
    return validate_glb(_read_pinned_file(path, ASSET_BYTES, ASSET_SHA256))


def _verify_directory(path: Path, expected: set[str]) -> None:
    try:
        if not stat.S_ISDIR(path.lstat().st_mode):
            raise ValueError("bed closure requires nonsymlink directories")
        found = set()
        with os.scandir(path) as entries:
            for entry in entries:
                if entry.name not in expected:
                    raise ValueError("bed closure contains an unexpected member")
                found.add(entry.name)
    except FileNotFoundError as exc:
        raise ValueError("bed closure is missing a directory") from exc
    if found != expected:
        raise ValueError("bed closure is missing a member")


def validate_bed_gltf(data: bytes) -> None:
    """Exact literal URI membership, without normalization, rewriting or fetching."""
    document = json.loads(data)
    if (
        not isinstance(document, dict)
        or not isinstance(document.get("asset"), dict)
        or document["asset"].get("version") != "2.0"
    ):
        raise ValueError("bed must be glTF 2.0")
    buffers, images = document.get("buffers", []), document.get("images", [])
    if (
        not isinstance(buffers, list)
        or len(buffers) != 1
        or not isinstance(buffers[0], dict)
        or buffers[0].get("uri") != BED_MEMBERS[1][0]
        or buffers[0].get("byteLength") != BED_MEMBERS[1][1]
        or not isinstance(images, list)
        or len(images) != 3
        or any(not isinstance(image, dict) or "bufferView" in image for image in images)
    ):
        raise ValueError("bed buffer/image URI closure differs from the pinned members")
    expected_images = {name for name, _size, _digest in BED_MEMBERS[2:]}
    image_uris = [image.get("uri") for image in images]
    if any(not isinstance(uri, str) for uri in image_uris) or set(image_uris) != expected_images:
        raise ValueError("bed image URI closure differs from the pinned members")
    uris = []

    def collect(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "uri":
                    uris.append(child)
                else:
                    collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(document)
    expected = {name for name, _size, _digest in BED_MEMBERS[1:]}
    if (
        len(uris) != len(expected)
        or any(not isinstance(uri, str) for uri in uris)
        or set(uris) != expected
    ):
        raise ValueError("bed contains an external, unexpected or duplicate URI")


def verify_bed_asset(root: Path) -> None:
    _verify_directory(root, {BED_GLTF, "GothicBed_01.bin", "textures"})
    _verify_directory(root / "textures", {Path(name).name for name, *_ in BED_MEMBERS[2:]})
    gltf = b""
    for name, size, digest in BED_MEMBERS:
        data = _read_pinned_file(root / name, size, digest)
        if name == BED_GLTF:
            gltf = data
    validate_bed_gltf(gltf)


def bed_asset_contract() -> dict:
    return {
        "name": "Poly Haven Gothic Bed01",
        "artist": "Kirill Sannikov",
        "license": "CC0-1.0",
        "source": "https://polyhaven.com/a/GothicBed_01",
        "license_url": "https://polyhaven.com/license",
        "entrypoint": BED_GLTF,
        "members": [
            {"file": name, "bytes": size, "sha256": digest} for name, size, digest in BED_MEMBERS
        ],
        "uri_mapping": "exact local relative paths, including textures/; no rewriting",
        "materials": "retain imported PBR materials and all three pinned textures",
        "website_preview_used": False,
    }


def actor_pose_contract() -> dict:
    return {
        "animation_index": 0,
        "required_animation_count": 1,
        "sample_time_s": 0.0,
        "sample_frame": 0,
        "sample_subframe": 0.0,
        "pose_position": "POSE",
        "out_of_range_sampling": (
            "Blender CONSTANT endpoint extrapolation per FCurve; frame stays 0, keys unchanged"
        ),
        "freeze": "snapshot bone matrix_basis; clear animation data/NLA; restore sampled matrices",
        "geometry": "evaluated deformed skin, not undeformed rest vertices",
        "motion": "frozen imported sample with grounded rigid root rotation",
        "claims": "no walking, physics, prone/supine or model admission",
    }


def phase_for(frame: int, motion: str) -> str:
    if type(frame) is not int or not 0 <= frame < FRAME_COUNT:
        raise ValueError("frame must be an integer in [0, 359]")
    if motion not in {"normal", "fall"}:
        raise ValueError("unsupported motion")
    if motion == "normal" or frame < 150:
        return "upright"
    return "falling" if frame < 180 else "lying"


def fall_angle(frame: int, motion: str) -> float:
    phase = phase_for(frame, motion)
    if phase == "upright":
        return 0.0
    if phase == "lying":
        return -math.pi / 2
    progress = (frame - 149) / 30
    return -math.pi / 2 * progress * progress * (3 - 2 * progress)


def actor_positions(actors: int) -> tuple[tuple[float, float], ...]:
    if type(actors) is not int or actors not in {1, 4}:
        raise ValueError("actors must be 1 or 4")
    return ((0.0, 0.0),) if actors == 1 else ((-1.2, 0.0), (-0.4, 0.0), (0.4, 0.0), (1.2, 0.0))


def heading_for(seed: int) -> float:
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an unsigned 32-bit integer")
    return (seed % 8 - 4) * math.pi / 128


def bed_placement(actors: int) -> dict:
    actor_positions(actors)
    return {
        "center_xy_m": [2.1, 0.7] if actors == 1 else [2.7, 0.7],
        "heading_rad": 0.0,
    }


def geometry_bounds(points) -> list[list[float]]:
    if not points or not all(
        len(point) == 3 and all(math.isfinite(value) for value in point) for point in points
    ):
        raise ValueError("mesh requires nonempty finite 3D geometry")
    return [[reduce(point[i] for point in points) for i in range(3)] for reduce in (min, max)]


def bed_normalization(points, actors: int) -> dict:
    """Uniformly scale all geometry; align its longer horizontal extent to local Y."""
    low, high = geometry_bounds(points)
    dimensions = [high[i] - low[i] for i in range(3)]
    if len(points) >= BED_VERTEX_LIMIT or not all(0.1 < value < 5 for value in dimensions):
        raise ValueError("bed imported geometry exceeds the bounded physical input")
    scale = BED_LENGTH / max(dimensions[:2])
    width, height = min(dimensions[:2]) * scale, dimensions[2] * scale
    if not (
        BED_WIDTH_RANGE[0] <= width <= BED_WIDTH_RANGE[1]
        and BED_HEIGHT_RANGE[0] <= height <= BED_HEIGHT_RANGE[1]
    ):
        raise ValueError("bed normalized dimensions exceed the physical scene bounds")
    alignment = math.pi / 2 if dimensions[0] > dimensions[1] else 0.0
    placement = bed_placement(actors)
    angle = placement["heading_rad"] + alignment
    sine, cosine = math.sin(angle), math.cos(angle)
    origin = [(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, low[2]]
    x, y = placement["center_xy_m"]
    return {
        "imported_bounds": [low, high],
        "uniform_scale": scale,
        "axis_alignment_rad": alignment,
        "normalized_dimensions_m": [width, BED_LENGTH, height],
        "location_m": [
            x - scale * (origin[0] * cosine - origin[1] * sine),
            y - scale * (origin[0] * sine + origin[1] * cosine),
            -origin[2] * scale,
        ],
        "rotation_euler_rad": [0.0, 0.0, angle],
    }


def motion_contract(motion: str, seed: int) -> dict:
    phase_for(0, motion)
    return {
        "heading_rad": heading_for(seed),
        "rotation_order": "XYZ",
        "pitch_axis": "local-X",
        "pitch_end_rad": -math.pi / 2 if motion == "fall" else 0.0,
        "interpolation": "smoothstep((frame-149)/30)" if motion == "fall" else "constant",
        "upright_inclusive": [0, 149] if motion == "fall" else [0, 359],
        "falling_inclusive": [150, 179] if motion == "fall" else [],
        "lying_hold_inclusive": [180, 359] if motion == "fall" else [],
        "floor_posture": "lying" if motion == "fall" else None,
        "posture_note": "No prone/supine or model-admission claim",
        "pose": "first imported animation at time 0/frame 0, frozen POSE; rigid root motion",
        "grounding": "clearance minus minimum rotated evaluated posed-mesh vertex Z at each angle",
        "ground_clearance_m": GROUND_CLEARANCE,
    }


def scene_contract() -> dict:
    """Executable declarations shared with the manifest, not vision evidence."""
    return {
        "actor_height_m": ACTOR_HEIGHT,
        "actor_positions_xy_m": {
            str(count): [list(position) for position in actor_positions(count)] for count in (1, 4)
        },
        "independent_armatures": True,
        "actor_pose": actor_pose_contract(),
        "head_material": "synthetic skin and scalp on existing mesh; no face overlay",
        "human_surfaces": [
            {"name": name, "upper_height_m": height, "color": list(color)}
            for name, height, color in HUMAN_SURFACES
        ],
        "bed": {
            "asset": "Poly Haven Gothic Bed01",
            "placement": {str(count): bed_placement(count) for count in (1, 4)},
            "normalization": (
                "all evaluated mesh vertices; uniform scale; longer horizontal extent to local Y; "
                "center XY and ground minimum Z"
            ),
            "length_m": BED_LENGTH,
            "width_range_m": list(BED_WIDTH_RANGE),
            "height_range_m": list(BED_HEIGHT_RANGE),
            "imported_axis_extent_range_exclusive": [0.1, 5.0],
            "vertex_limit_exclusive": BED_VERTEX_LIMIT,
            "materials": "unchanged imported PBR materials/textures",
            "geometry": "own render of pinned glTF meshes; no procedural bed or 2D stamp",
        },
        "path_clearance_m": PATH_CLEARANCE,
        "camera": {
            "projection": "PERSP",
            "elevation_rad": CAMERA_ELEVATION,
            "lens_mm": CAMERA_LENS_MM,
            "sensor_width_mm": CAMERA_SENSOR_MM,
            "sensor_fit": "HORIZONTAL",
            "minimum_edge_margin": CAMERA_MARGIN,
            "envelope_steps": ENVELOPE_STEPS,
            "continuous_guard": "2 * maximum YZ radius * half angular sample spacing",
            "bounds": "evaluated posed actor meshes swept through full fall and all bed meshes",
            "fixed_for": "all frames and both motion variants of a count/seed layout",
            "excluded_from_bounds": ["floor", "wall", "lights"],
        },
    }


def ground_lift(points, angle: float) -> float:
    sine, cosine = math.sin(angle), math.cos(angle)
    return GROUND_CLEARANCE - min(y * sine + z * cosine for _x, y, z in points)


def grounded_points(points, position, angle: float, heading: float):
    sine, cosine = math.sin(angle), math.cos(angle)
    sh, ch = math.sin(heading), math.cos(heading)
    lift = ground_lift(points, angle)
    return tuple(
        (
            position[0] + x * ch - (y * cosine - z * sine) * sh,
            position[1] + x * sh + (y * cosine - z * sine) * ch,
            y * sine + z * cosine + lift,
        )
        for x, y, z in points
    )


def swept_xy_bounds(points, position, heading: float):
    """Exact horizontal bounds of the continuous negative-X quarter turn."""

    def arc_bounds(a, b):
        low, high = min(a, b), max(a, b)
        if a < 0 and b < 0:
            low = -math.hypot(a, b)
        if a > 0 and b > 0:
            high = math.hypot(a, b)
        return low, high

    sh, ch = math.sin(heading), math.cos(heading)
    xs, ys = [], []
    for x, y, z in points:
        xs.extend(position[0] + x * ch + value for value in arc_bounds(-y * sh, -z * sh))
        ys.extend(position[1] + x * sh + value for value in arc_bounds(y * ch, z * ch))
    return ((min(xs), min(ys)), (max(xs), max(ys)))


def fit_camera(points, motion_padding: float) -> dict:
    """Fit perspective constraints, including a continuous-motion error bound."""
    if not points or not math.isfinite(motion_padding) or motion_padding < 0:
        raise ValueError("camera requires points and nonnegative finite motion padding")
    sine, cosine = math.sin(CAMERA_ELEVATION), math.cos(CAMERA_ELEVATION)
    # Camera basis: right +X, up (0, sin(e), cos(e)), back (0, -cos(e), sin(e)).
    projected = [(x, y * sine + z * cosine, -y * cosine + z * sine) for x, y, z in points]
    if not all(math.isfinite(value) for point in projected for value in point):
        raise ValueError("camera bounds must be finite")
    center = tuple(
        (min(p[i] for p in projected) + max(p[i] for p in projected)) / 2 for i in range(3)
    )
    tangent_x = CAMERA_SENSOR_MM / (2 * CAMERA_LENS_MM)
    tangent_y = tangent_x * HEIGHT / WIDTH
    limit = 1 - 2 * CAMERA_MARGIN
    distance = max(
        z
        - center[2]
        + motion_padding
        + max(
            (abs(x - center[0]) + motion_padding) / (tangent_x * limit),
            (abs(y - center[1]) + motion_padding) / (tangent_y * limit),
            0.1,
        )
        for x, y, z in projected
    )
    back = center[2] + distance
    return {
        "projection": "PERSP",
        "lens_mm": CAMERA_LENS_MM,
        "sensor_width_mm": CAMERA_SENSOR_MM,
        "sensor_fit": "HORIZONTAL",
        "location_m": [
            center[0],
            center[1] * sine - back * cosine,
            center[1] * cosine + back * sine,
        ],
        "rotation_euler_rad": [math.pi / 2 - CAMERA_ELEVATION, 0.0, 0.0],
        "clip_m": [0.05, 100.0],
        "minimum_edge_margin": CAMERA_MARGIN,
        "continuous_motion_padding_m": motion_padding,
        "sampled_bounds_camera_basis_m": [
            [min(p[i] for p in projected) for i in range(3)],
            [max(p[i] for p in projected) for i in range(3)],
        ],
    }


def rendered_receipt(
    output: Path,
    *,
    actors: int,
    motion: str,
    seed: int,
    frames: tuple[int, ...],
    blender_version: tuple[int, ...],
    scene_measurements: dict,
) -> dict:
    if tuple(blender_version) != BLENDER_VERSION:
        raise ValueError("renderer is not pinned Blender 4.0.2")
    actor_positions(actors)
    contract = motion_contract(motion, seed)
    if len(set(frames)) != len(frames) or not frames:
        raise ValueError("render receipt requires unique rendered frames")
    images = []
    for frame in frames:
        phase = phase_for(frame, motion)
        name = f"frame_{frame:04d}.png"
        path = output / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("render is missing a regular frame image")
        data = path.read_bytes()
        if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError("rendered frame is not PNG")
        if struct.unpack_from(">II", data, 16) != (WIDTH, HEIGHT):
            raise ValueError("rendered frame dimensions differ from the contract")
        images.append(
            {
                "frame": frame,
                "phase": phase,
                "pitch_rad": fall_angle(frame, motion),
                "file": name,
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
            }
        )
    complete = frames == tuple(range(FRAME_COUNT))
    return {
        "schema": "synthetic-render-receipt.v1",
        "role": "render-evidence-only",
        "render_complete": complete,
        "qualifying_full_corpus_member": False,
        "inference_qualification": False,
        "model_admission": False,
        "admission_status": "unverified; render completeness is not model admission",
        "single_frame_smoke": len(frames) == 1,
        "assets": {
            "actor": {
                "file": "CesiumMan.glb",
                "bytes": ASSET_BYTES,
                "sha256": ASSET_SHA256,
                "revision": ASSET_REVISION,
                "license": "CC-BY-4.0",
                "attribution": "CesiumMan ©2017 Cesium, CC-BY-4.0",
                "source": (
                    "https://github.com/KhronosGroup/glTF-Sample-Assets/blob/"
                    f"{ASSET_REVISION}/Models/CesiumMan/glTF-Binary/CesiumMan.glb"
                ),
                "resources": "embedded GLB only; no URI resources",
            },
            "bed": bed_asset_contract(),
        },
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "blender_version": list(blender_version),
        "engine": "BLENDER_EEVEE",
        "width": WIDTH,
        "height": HEIGHT,
        "fps": FPS,
        "actors": actors,
        "motion": motion,
        "seed": seed,
        "scene_contract": scene_contract(),
        "motion_contract": contract,
        "scene_measurements": scene_measurements,
        "frames": images,
    }


def _material(bpy, name, color, *, fabric=False):
    material = bpy.data.materials.new(name)
    material.diffuse_color = (*color, 1.0)
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1.0)
    shader.inputs["Roughness"].default_value = 0.8
    if fabric:
        nodes, links = material.node_tree.nodes, material.node_tree.links
        noise = nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 160
        noise.inputs["Detail"].default_value = 2
        bump = nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = 0.18
        bump.inputs["Distance"].default_value = 0.003
        links.new(noise.outputs["Fac"], bump.inputs["Height"])
        links.new(bump.outputs["Normal"], shader.inputs["Normal"])
    return material


def _box(bpy, name, location, dimensions, material):
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    obj = bpy.context.object
    obj.name, obj.dimensions = name, dimensions
    obj.data.materials.append(material)
    return obj


def _mesh_points(bpy, meshes):
    graph = bpy.context.evaluated_depsgraph_get()
    points = []
    for obj in meshes:
        evaluated = obj.evaluated_get(graph)
        points.extend(evaluated.matrix_world @ vertex.co for vertex in evaluated.data.vertices)
    return points


def _pose_matrices(rig):
    return [
        {"bone": bone.name, "matrix_basis": [list(row) for row in bone.matrix_basis]}
        for bone in rig.pose.bones
    ]


def _validate_action_sample(action):
    """Validate each channel at frame zero; count constant endpoint samples."""
    start, end = action.frame_range
    if not (math.isfinite(start) and math.isfinite(end) and start <= end and action.fcurves):
        raise ValueError("imported animation requires a finite action range and keyed FCurves")
    constant_endpoints = 0
    for curve in action.fcurves:
        keys = [(float(key.co[0]), float(key.co[1])) for key in curve.keyframe_points]
        if (
            curve.mute
            or curve.modifiers
            or curve.sampled_points
            or not keys
            or not all(math.isfinite(value) for key in keys for value in key)
            or any(left[0] >= right[0] for left, right in pairwise(keys))
        ):
            raise ValueError(
                "imported animation requires unmodified finite ordered keyframe curves"
            )
        if not keys[0][0] <= 0 <= keys[-1][0]:
            if curve.extrapolation != "CONSTANT":
                raise ValueError(
                    "imported animation outside its FCurve key span at frame 0 "
                    "requires CONSTANT extrapolation"
                )
            constant_endpoints += 1
    return constant_endpoints


def _sample_actor_pose(bpy, imported, animation):
    """Select the only imported animation, evaluate at zero, then freeze the actual pose."""
    rigs = [obj for obj in imported if obj.type == "ARMATURE"]
    meshes = [obj for obj in imported if obj.type == "MESH"]
    if len(rigs) != 1 or not meshes or not rigs[0].pose.bones:
        raise ValueError("imported asset requires one animated skinned armature")
    rig = rigs[0]
    rig.data.pose_position = "POSE"
    owners = []
    for obj in imported:
        owners.append(obj)
        if obj.data is not None:
            owners.append(obj.data)
        if obj.type == "MESH" and obj.data.shape_keys is not None:
            owners.append(obj.data.shape_keys)
    constant_endpoint_curve_count = 0
    for owner in owners:
        data = owner.animation_data
        if data is None:
            continue
        if data.drivers:
            raise ValueError("imported sample must not depend on animation drivers")
        actions = {
            strip.action for track in data.nla_tracks for strip in track.strips if strip.action
        }
        if data.action is not None:
            actions.add(data.action)
        if len(actions) != 1:
            raise ValueError("imported animation owner requires one unambiguous action")
        action = next(iter(actions))
        constant_endpoint_curve_count += _validate_action_sample(action)
        data.use_nla = False
        data.action = action
        data.action_blend_type = "REPLACE"
        data.action_influence = 1.0
    if rig.animation_data is None or rig.animation_data.action is None:
        raise ValueError("imported armature has no real animation action")
    action = rig.animation_data.action
    curves = [curve for curve in action.fcurves if curve.data_path.startswith("pose.bones[")]
    start, end = action.frame_range
    if not any(
        len(curve.keyframe_points) >= 2
        and len({float(key.co[1]) for key in curve.keyframe_points}) > 1
        for curve in curves
    ):
        raise ValueError("imported armature requires changing bone animation keyframes")
    bpy.context.scene.frame_set(0, subframe=0.0)
    bpy.context.view_layer.update()
    matrices = {bone.name: bone.matrix_basis.copy() for bone in rig.pose.bones}
    transforms = [(obj, obj.matrix_basis.copy()) for obj in imported]
    sampled_points = [tuple(point) for point in _mesh_points(bpy, meshes)]
    record = {
        "contract": actor_pose_contract(),
        "animation": animation,
        "action": action.name,
        "action_frame_range": [float(start), float(end)],
        "constant_endpoint_curve_count": constant_endpoint_curve_count,
        "bone_matrices": _pose_matrices(rig),
        "sampled_bounds": geometry_bounds(sampled_points),
        "sampled_vertex_count": len(sampled_points),
    }
    if not all(
        math.isfinite(value) for matrix in matrices.values() for row in matrix for value in row
    ):
        raise ValueError("sampled bone matrices must be finite")
    for owner in owners:
        # animation_data_clear removes both the active action and its NLA tracks.
        owner.animation_data_clear()
    for obj, matrix in transforms:
        obj.matrix_basis = matrix
    for bone in rig.pose.bones:
        bone.matrix_basis = matrices[bone.name]
    bpy.context.view_layer.update()
    frozen_points = _mesh_points(bpy, meshes)
    if len(sampled_points) != len(frozen_points) or any(
        math.dist(before, after) > 1e-5
        for before, after in zip(sampled_points, frozen_points, strict=True)
    ):
        raise ValueError("clearing animation changed the evaluated sampled skin")
    return record


def _actor_geometry(bpy, controllers, pose_sample):
    """Inspect actual imported skin bindings and clone ownership before rendering."""
    posed_points, records, owned_data = [], [], set()
    for controller in controllers:
        descendants = list(controller.children_recursive)
        rigs = [obj for obj in descendants if obj.type == "ARMATURE"]
        meshes = [obj for obj in descendants if obj.type == "MESH"]
        if len(rigs) != 1 or not meshes:
            raise ValueError("each actor requires one independent armature and skinned meshes")
        rig = rigs[0]
        actual_pose = _pose_matrices(rig)
        expected_pose = pose_sample["bone_matrices"]
        if rig.data.pose_position != "POSE" or len(actual_pose) != len(expected_pose):
            raise ValueError("actor lost its sampled POSE")
        for actual, expected in zip(actual_pose, expected_pose, strict=True):
            if actual["bone"] != expected["bone"] or any(
                not math.isclose(a, b, abs_tol=1e-6)
                for row, expected_row in zip(
                    actual["matrix_basis"], expected["matrix_basis"], strict=True
                )
                for a, b in zip(row, expected_row, strict=True)
            ):
                raise ValueError("actor bone matrices differ from the imported sample")
        if any(
            obj.animation_data is not None
            or (obj.data is not None and obj.data.animation_data is not None)
            for obj in descendants
        ):
            raise ValueError("actor retained animation data after sampling")
        for obj in (rig, *meshes):
            pointer = obj.data.as_pointer()
            if pointer in owned_data:
                raise ValueError("actors share mesh or armature data")
            owned_data.add(pointer)
        for mesh in meshes:
            modifiers = list(mesh.modifiers)
            if (
                len(modifiers) != 1
                or modifiers[0].type != "ARMATURE"
                or modifiers[0].object != rig
                or not modifiers[0].use_vertex_groups
                or not modifiers[0].show_viewport
                or not modifiers[0].show_render
            ):
                raise ValueError("mesh is not bound exclusively to its own active armature")
            groups = {group.index for group in mesh.vertex_groups if group.name in rig.data.bones}
            if not mesh.data.polygons or any(
                not any(weight.group in groups and weight.weight > 0 for weight in vertex.groups)
                for vertex in mesh.data.vertices
            ):
                raise ValueError("humanoid mesh has missing skin weights or polygons")
        points = tuple(tuple(point) for point in _mesh_points(bpy, meshes))
        low, high = geometry_bounds(points)
        if abs(low[2]) > 1e-4 or abs(high[2] - ACTOR_HEIGHT) > 1e-4:
            raise ValueError("normalized posed geometry differs from the actor height contract")
        posed_points.append(points)
        records.append(
            {
                "controller": controller.name,
                "armature": rig.name,
                "bone_count": len(rig.data.bones),
                "pose_position": rig.data.pose_position,
                "posed_bounds_m": [low, high],
                "bone_matrices": actual_pose,
                "meshes": [
                    {
                        "name": mesh.name,
                        "vertices": len(mesh.data.vertices),
                        "material_polygons": {
                            material.name: sum(
                                polygon.material_index == index for polygon in mesh.data.polygons
                            )
                            for index, material in enumerate(mesh.data.materials)
                        },
                    }
                    for mesh in meshes
                ],
            }
        )
    return posed_points, records


def _dress_actor(bpy, meshes):
    # Preserve the asset's actual head/limb geometry, not a 2D face or person stamp.
    materials = [
        _material(bpy, name, color, fabric=name in {"shirt", "trousers"})
        for name, _height, color in HUMAN_SURFACES
    ]
    graph = bpy.context.evaluated_depsgraph_get()
    for mesh in meshes:
        evaluated = mesh.evaluated_get(graph)
        heights = [(evaluated.matrix_world @ vertex.co).z for vertex in evaluated.data.vertices]
        if len(heights) != len(mesh.data.vertices):
            raise ValueError("posed mesh topology changed before material assignment")
        mesh.data.materials.clear()
        for material in materials:
            mesh.data.materials.append(material)
        for polygon in mesh.data.polygons:
            height = sum(heights[index] for index in polygon.vertices) / len(polygon.vertices)
            polygon.material_index = next(
                (
                    index
                    for index, (_name, upper, _color) in enumerate(HUMAN_SURFACES)
                    if height < upper
                ),
                len(materials) - 1,
            )


def _place_actors(bpy, controllers, posed_points, positions, angle, heading):
    for controller, points, (x, y) in zip(controllers, posed_points, positions, strict=True):
        controller.location = (x, y, ground_lift(points, angle))
        controller.rotation_mode = "XYZ"
        controller.rotation_euler = (angle, 0, heading)
    bpy.context.view_layer.update()


def check_path_clearance(bounds):
    """Fail closed on overlapping full fall paths or a bed inside those paths."""
    for index, (low, high) in enumerate(bounds):
        for other_low, other_high in bounds[index + 1 :]:
            if not any(
                high[axis] + PATH_CLEARANCE <= other_low[axis]
                or other_high[axis] + PATH_CLEARANCE <= low[axis]
                for axis in (0, 1)
            ):
                raise ValueError("continuous fall paths or bed lack horizontal clearance")


def camera_envelope(posed_points, positions, heading, bed_points):
    """One camera, including unsampled angles; never fit walls or individual frames."""
    path_bounds = [
        swept_xy_bounds(points, position, heading)
        for points, position in zip(posed_points, positions, strict=True)
    ]
    bed_bounds = tuple(
        tuple(reduce(point[i] for point in bed_points) for i in range(2)) for reduce in (min, max)
    )
    check_path_clearance([*path_bounds, bed_bounds])
    points = list(bed_points)
    for step in range(ENVELOPE_STEPS + 1):
        for vertices, position in zip(posed_points, positions, strict=True):
            points.extend(
                grounded_points(vertices, position, -step * math.pi / (2 * ENVELOPE_STEPS), heading)
            )
    # Rotation moves a vertex by <= R*dθ. The min-Z grounding translation has
    # the same Lipschitz bound. Every angle is within half a step of a sample.
    radius = max(math.hypot(y, z) for vertices in posed_points for _x, y, z in vertices)
    fitted = fit_camera(points, radius * math.pi / (2 * ENVELOPE_STEPS))
    return {"camera": fitted, "actor_swept_xy_bounds_m": path_bounds, "bed_xy_bounds_m": bed_bounds}


def _frame_camera(bpy, camera, posed_points, positions, heading, bed_meshes):
    bed_points = [tuple(point) for point in _mesh_points(bpy, bed_meshes)]
    measurements = camera_envelope(posed_points, positions, heading, bed_points)
    fitted = measurements["camera"]
    camera.location = fitted["location_m"]
    camera.rotation_mode = "XYZ"
    camera.rotation_euler = fitted["rotation_euler_rad"]
    camera.data.type = fitted["projection"]
    camera.data.lens = fitted["lens_mm"]
    camera.data.sensor_fit = fitted["sensor_fit"]
    camera.data.sensor_width = fitted["sensor_width_mm"]
    camera.data.shift_x = camera.data.shift_y = 0
    camera.data.clip_start, camera.data.clip_end = fitted["clip_m"]
    bpy.context.view_layer.update()
    # Read back Blender's actual (float32) camera, not just the requested fit.
    fitted.update(
        {
            "location_m": list(camera.location),
            "rotation_euler_rad": list(camera.rotation_euler),
            "lens_mm": camera.data.lens,
            "sensor_width_mm": camera.data.sensor_width,
            "clip_m": [camera.data.clip_start, camera.data.clip_end],
        }
    )
    return measurements


def _bed(bpy, asset_root: Path, actors):
    scene = bpy.context.scene
    before_import = set(scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(asset_root / BED_GLTF))
    imported = [obj for obj in scene.objects if obj not in before_import]
    meshes = [obj for obj in imported if obj.type == "MESH"]
    if (
        not meshes
        or any(obj.type not in {"EMPTY", "MESH"} or obj.animation_data for obj in imported)
        or sum(len(mesh.data.vertices) for mesh in meshes) >= BED_VERTEX_LIMIT
    ):
        raise ValueError("bed requires bounded static imported mesh geometry")
    materials = {material for mesh in meshes for material in mesh.data.materials}
    if not materials or any(material is None or not material.use_nodes for material in materials):
        raise ValueError("bed is missing its imported PBR materials")
    images = {
        node.image
        for material in materials
        for node in material.node_tree.nodes
        if node.type == "TEX_IMAGE" and node.image is not None
    }
    if {Path(image.filepath).name for image in images} != {
        Path(name).name for name, *_ in BED_MEMBERS[2:]
    }:
        raise ValueError("bed is missing its pinned PBR textures")
    bpy.context.view_layer.update()
    transform = bed_normalization([tuple(point) for point in _mesh_points(bpy, meshes)], actors)
    root = bpy.data.objects.new("gothic-bed-placement", None)
    scene.collection.objects.link(root)
    for obj in imported:
        if obj.parent not in imported:
            matrix = obj.matrix_world.copy()
            obj.parent = root
            obj.matrix_world = matrix
    root.scale = (transform["uniform_scale"],) * 3
    root.rotation_mode = "XYZ"
    root.rotation_euler = transform["rotation_euler_rad"]
    root.location = transform["location_m"]
    bpy.context.view_layer.update()
    bounds = geometry_bounds([tuple(point) for point in _mesh_points(bpy, meshes)])
    dimensions = [bounds[1][axis] - bounds[0][axis] for axis in range(3)]
    expected = transform["normalized_dimensions_m"]
    center = bed_placement(actors)["center_xy_m"]
    if (
        abs(bounds[0][2]) > 1e-4
        or any(
            abs(actual - wanted) > 1e-4 for actual, wanted in zip(dimensions, expected, strict=True)
        )
        or any(abs((bounds[0][i] + bounds[1][i]) / 2 - center[i]) > 1e-4 for i in (0, 1))
    ):
        raise ValueError("bed evaluated geometry differs from its physical placement")
    return meshes, {
        "normalization": transform,
        "world_bounds_m": bounds,
        "world_dimensions_m": dimensions,
        "meshes": [
            {
                "name": mesh.name,
                "vertices": len(mesh.data.vertices),
                "polygons": len(mesh.data.polygons),
                "materials": [material.name for material in mesh.data.materials],
            }
            for mesh in meshes
        ],
        "textures": sorted({Path(image.filepath).name for image in images}),
        "materials_preserved": True,
    }


def _remove_bone_display_shapes(bpy, imported):
    """Exclude glTF importer's editor glyphs, not any skinned actor geometry."""
    bones = [bone for obj in imported if obj.type == "ARMATURE" for bone in obj.pose.bones]
    shapes = {bone.custom_shape for bone in bones if bone.custom_shape is not None}
    for shape in shapes:
        if shape not in imported or any(mod.type == "ARMATURE" for mod in shape.modifiers):
            raise ValueError("bone display shape is not exclusively imported editor geometry")
    geometry = [obj for obj in imported if obj not in shapes]
    for bone in bones:
        if bone.custom_shape in shapes:
            bone.custom_shape = None
    for shape in shapes:
        bpy.data.objects.remove(shape, do_unlink=True)
    return geometry


def _scene(bpy, asset: Path, bed_asset_root: Path, animation: dict, actors: int, seed: int):
    from mathutils import Vector

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = 32
    scene.eevee.use_gtao = True
    scene.eevee.gtao_distance = 3
    scene.render.resolution_x, scene.render.resolution_y = WIDTH, HEIGHT
    scene.render.resolution_percentage, scene.render.fps = 100, FPS
    scene.render.fps_base = 1.0
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1
    scene.render.use_border = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.8, 0.8, 0.8, 1)
    scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.5
    scene.frame_start, scene.frame_end = 0, FRAME_COUNT - 1
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    before_import = set(scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(asset))
    imported = [obj for obj in scene.objects if obj not in before_import]
    imported = _remove_bone_display_shapes(bpy, imported)
    meshes = [obj for obj in imported if obj.type == "MESH"]
    pose_sample = _sample_actor_pose(bpy, imported, animation)
    corners = _mesh_points(bpy, meshes)
    low = Vector(tuple(min(point[axis] for point in corners) for axis in range(3)))
    high = Vector(tuple(max(point[axis] for point in corners) for axis in range(3)))
    if high.z - low.z <= 0:
        raise ValueError("humanoid has no positive height")
    scale = ACTOR_HEIGHT / (high.z - low.z)
    origin = Vector(((low.x + high.x) / 2, (low.y + high.y) / 2, low.z))
    holder = bpy.data.objects.new("normalized-mesh", None)
    controller = bpy.data.objects.new("actor-motion", None)
    for obj in (holder, controller):
        scene.collection.objects.link(obj)
    holder.parent = controller
    for obj in imported:
        if obj.parent not in imported:
            matrix = obj.matrix_world.copy()
            obj.parent = holder
            obj.matrix_world = matrix
    holder.scale = (scale,) * 3
    holder.location = -origin * scale
    bpy.context.view_layer.update()
    _dress_actor(bpy, meshes)
    for image in list(bpy.data.images):
        bpy.data.images.remove(image)
    hierarchy = [controller, holder, *imported]
    controllers = [controller]
    for _index in range(1, actors):
        copies = {}
        for obj in hierarchy:
            duplicate = obj.copy()
            if obj.data is not None:
                duplicate.data = obj.data.copy()
            duplicate.animation_data_clear()
            scene.collection.objects.link(duplicate)
            copies[obj] = duplicate
        for original, duplicate in copies.items():
            duplicate.parent = copies.get(original.parent)
            if original.type == "ARMATURE":
                duplicate.data.pose_position = "POSE"
                for bone in duplicate.pose.bones:
                    bone.matrix_basis = original.pose.bones[bone.name].matrix_basis.copy()
            for modifier in duplicate.modifiers:
                if modifier.type == "ARMATURE":
                    if modifier.object not in copies:
                        raise ValueError("imported skin controller is outside the actor hierarchy")
                    modifier.object = copies[modifier.object]
        controllers.append(copies[controller])
    bpy.context.view_layer.update()
    posed_points, rig_records = _actor_geometry(bpy, controllers, pose_sample)
    # No patient imagery, face overlays or live camera input. Bed PBR is retained.
    floor = _material(bpy, "floor", (0.65, 0.62, 0.55))
    wall = _material(bpy, "wall", (0.76, 0.73, 0.68))
    _box(bpy, "floor", (0, 0, -0.05), (12, 10, 0.1), floor)
    _box(bpy, "wall", (0, 4.5, 1.5), (12, 0.12, 3), wall)
    bed_meshes, bed_geometry = _bed(bpy, bed_asset_root, actors)
    bpy.ops.object.light_add(type="AREA", location=(-3, -4, 7))
    light = bpy.context.object
    light.data.energy, light.data.shape, light.data.size = 1000, "DISK", 5
    light.rotation_euler = (
        (Vector((0, 1, 0.6)) - light.location).to_track_quat("-Z", "Y").to_euler()
    )
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    scene.camera = camera
    # Seed selects a declared, reproducible heading; it never changes timing.
    heading = heading_for(seed)
    measurements = _frame_camera(
        bpy, camera, posed_points, actor_positions(actors), heading, bed_meshes
    )
    measurements["actor_pose_sample"] = pose_sample
    measurements["actor_geometry"] = rig_records
    measurements["bed_geometry"] = bed_geometry
    measurements["bed_placement"] = bed_placement(actors)
    measurements["light"] = {
        "type": light.data.type,
        "location_m": list(light.location),
        "rotation_euler_rad": list(light.rotation_euler),
        "energy": light.data.energy,
        "size_m": light.data.size,
    }
    return scene, controllers, posed_points, heading, measurements


def render(
    asset: Path,
    output: Path,
    *,
    bed_asset_root: Path,
    actors: int,
    motion: str,
    seed: int,
    single_frame: int | None,
    declared_sha256: str,
) -> dict:
    animation = verify_asset(asset, declared_sha256)
    verify_bed_asset(bed_asset_root)
    positions = actor_positions(actors)
    frames = tuple(range(FRAME_COUNT)) if single_frame is None else (single_frame,)
    for frame in frames:
        phase_for(frame, motion)
    heading_for(seed)
    if output.exists() or output.is_symlink():
        raise FileExistsError("output must be a new directory; refusing overwrite")
    import bpy

    if tuple(bpy.app.version) != BLENDER_VERSION:
        raise ValueError("renderer must be Blender 4.0.2")
    output.mkdir(parents=False, exist_ok=False)
    scene, controllers, posed_points, heading, measurements = _scene(
        bpy, asset, bed_asset_root, animation, actors, seed
    )
    measurements["rendered_actor_roots"] = []
    for frame in frames:
        scene.frame_set(frame)
        angle = fall_angle(frame, motion)
        _place_actors(bpy, controllers, posed_points, positions, angle, heading)
        measurements["rendered_actor_roots"].append(
            {
                "frame": frame,
                "actors": [
                    {
                        "location_m": list(root.location),
                        "rotation_euler_rad": list(root.rotation_euler),
                    }
                    for root in controllers
                ],
            }
        )
        scene.render.filepath = str(output / f"frame_{frame:04d}.png")
        bpy.ops.render.render(write_still=True)
    receipt = rendered_receipt(
        output,
        actors=actors,
        motion=motion,
        seed=seed,
        frames=frames,
        blender_version=tuple(bpy.app.version),
        scene_measurements=measurements,
    )
    with (output / "render-receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--bed-asset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--actors", required=True, type=int, choices=(1, 4))
    parser.add_argument("--motion", required=True, choices=("normal", "fall"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--single-frame", type=int)
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    args = parser.parse_args(argv)
    receipt = render(
        args.asset,
        args.output_dir,
        bed_asset_root=args.bed_asset_root,
        actors=args.actors,
        motion=args.motion,
        seed=args.seed,
        single_frame=args.single_frame,
        declared_sha256=args.sha256,
    )
    print(
        json.dumps(
            {
                "render_complete": receipt["render_complete"],
                "frames": len(receipt["frames"]),
                "inference_qualification": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
