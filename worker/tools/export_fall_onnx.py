"""Export a verified pose+bbox56 proxy bundle's Torch weights to ONNX at publish time."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import torch
from torch import nn

from worker.adapters.model.errors import ModelLoadError
from worker.adapters.model.pose_bbox56_bundle_support import member_digest, read_json, verify_bundle


class _ProxyGru(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.encoder = nn.GRU(56, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        _, hidden = self.encoder(window)
        return self.classifier(hidden[-1])


def export_fall_onnx(bundle_dir: Path, *, force: bool = False) -> str:
    """Export ``model.onnx`` and return its pinned SHA-256 digest."""
    root = bundle_dir.expanduser().resolve()
    manifest = read_json(root / "bundle-manifest.json")
    verify_bundle(root, manifest)
    if not force:
        try:
            return member_digest(manifest, "model.onnx")
        except ModelLoadError:
            pass
    arch = read_json(root / "arch.json")
    if not isinstance(arch, dict):
        raise ModelLoadError("unsupported pose-bbox56 architecture")
    try:
        module = _ProxyGru(
            _positive_int(arch.get("hidden_size"), "hidden_size"),
            _positive_int(arch.get("num_layers"), "num_layers"),
            _dropout(arch.get("dropout")),
        ).eval()
        state = torch.load(root / "model.pt", map_location="cpu", weights_only=True)
        module.load_state_dict(state, strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ModelLoadError(f"cannot load pose-bbox56 model: {exc}") from exc
    with tempfile.NamedTemporaryFile(dir=root, suffix=".onnx", delete=False) as output:
        temporary = Path(output.name)
    try:
        torch.onnx.export(
            module,
            torch.zeros((1, 30, 56), dtype=torch.float32),
            temporary,
            input_names=["window"],
            opset_version=17,
            dynamo=False,
        )
        payload = temporary.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        destination = root / "model.onnx"
        differs = (
            destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() != digest
        )
        if differs and not force:
            raise ModelLoadError("model.onnx differs; pass --force to overwrite it")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    _write_manifest_with_onnx(root, manifest, digest)
    return digest


def _write_manifest_with_onnx(root: Path, manifest: object, digest: str) -> None:
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
        raise ModelLoadError("invalid bundle-manifest.json")
    if not all(isinstance(item, dict) for item in manifest["files"]):
        raise ModelLoadError("invalid bundle-manifest file entry")
    files = [
        item
        for item in manifest["files"]
        if isinstance(item, dict) and item.get("relative_path") != "model.onnx"
    ]
    payload = (root / "model.onnx").read_bytes()
    files.append({"relative_path": "model.onnx", "sha256": digest, "size": len(payload)})
    updated = {**manifest, "files": sorted(files, key=lambda item: str(item["relative_path"]))}
    (root / "bundle-manifest.json").write_text(
        json.dumps(updated, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ModelLoadError(f"invalid {name}")
    return value


def _dropout(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value < 1:
        raise ModelLoadError("invalid dropout")
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_dir", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(export_fall_onnx(args.bundle_dir, force=args.force))


if __name__ == "__main__":
    main()
