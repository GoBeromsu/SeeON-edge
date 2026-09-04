"""Stable camera identities and Flow source names."""

from __future__ import annotations

from dataclasses import dataclass

from worker.native.deepstream.metadata import SourceBinding


@dataclass(slots=True)
class _Source:
    uri: str
    generation: int
    epoch: int
    pad_index: int


class SourceTable:
    """Owns source lifecycle counters while retaining canonical camera IDs."""

    def __init__(self, *, worker_boot_id: str, child_instance_id: str, transform_id: str) -> None:
        self._worker_boot_id = worker_boot_id
        self._child_instance_id = child_instance_id
        self._transform_id = transform_id
        self._sources: dict[str, _Source] = {}
        self._generations: dict[str, int] = {}
        self._epochs: dict[str, int] = {}
        self._next_pad_index = 0

    def add(self, camera_id: str, uri: str) -> SourceBinding:
        if camera_id in self._sources:
            raise ValueError(f"source already exists: {camera_id}")
        generation = self._generations.get(camera_id, -1) + 1
        epoch = self._epochs.get(camera_id, -1) + 1
        self._generations[camera_id] = generation
        self._epochs[camera_id] = epoch
        self._sources[camera_id] = _Source(
            uri=uri,
            generation=generation,
            epoch=epoch,
            pad_index=self._next_pad_index,
        )
        self._next_pad_index += 1
        return self.binding(camera_id)

    def remove(self, camera_id: str) -> None:
        del self._sources[camera_id]

    def rebuild(self, camera_id: str) -> SourceBinding:
        source = self._sources[camera_id]
        source.generation += 1
        source.epoch += 1
        self._generations[camera_id] = source.generation
        self._epochs[camera_id] = source.epoch
        return self.binding(camera_id)

    def binding(self, camera_id: str) -> SourceBinding:
        source = self._sources[camera_id]
        return SourceBinding(
            worker_boot_id=self._worker_boot_id,
            child_instance_id=self._child_instance_id,
            camera_id=camera_id,
            source_generation=source.generation,
            stream_epoch=source.epoch,
            transform_id=self._transform_id,
        )

    def camera_id_for_pad(self, pad_index: int) -> str | None:
        """The camera on this mux pad, or ``None`` when the pad is unknown.

        Returning ``None`` rather than raising is deliberate: the only caller is
        the SDK probe callback, and an exception there aborts the whole
        pipeline process.
        """
        for camera_id, source in self._sources.items():
            if source.pad_index == pad_index:
                return camera_id
        return None

    def camera_id(self, pad_index: int) -> str:
        camera_id = self.camera_id_for_pad(pad_index)
        if camera_id is None:
            raise KeyError(f"unknown source pad index: {pad_index}")
        return camera_id

    def pad_index(self, camera_id: str) -> int:
        return self._sources[camera_id].pad_index

    def source_name(self, camera_id: str) -> str:
        return f"batch_capture-source-0_{self.pad_index(camera_id)}"

    def uri(self, camera_id: str) -> str:
        return self._sources[camera_id].uri

    def camera_ids(self) -> tuple[str, ...]:
        return tuple(self._sources)


__all__ = ["SourceTable"]
