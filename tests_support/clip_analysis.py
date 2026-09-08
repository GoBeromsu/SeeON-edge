"""Clip-analysis publication hook doubles."""

from __future__ import annotations

from dataclasses import dataclass, field

from worker.pipeline.output.evidence.clip_publication import ReadyClipPublication


def no_op_ready_hook(_publication: ReadyClipPublication) -> None:
    return


@dataclass
class RecordingReadyHook:
    publications: list[ReadyClipPublication] = field(default_factory=list)

    def __call__(self, publication: ReadyClipPublication) -> None:
        self.publications.append(publication)
