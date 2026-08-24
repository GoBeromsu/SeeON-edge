"""Materialize content digests into the image-owned DeepStream manifest."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(template_path: Path, output_path: Path) -> None:
    manifest: dict[str, Any] = json.loads(template_path.read_text(encoding="utf-8"))
    for plugin in manifest["plugins"]:
        plugin["sha256"] = _sha256(Path(plugin["path"]))
    manifest["native"]["sha256"] = _sha256(Path(manifest["native"]["path"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
