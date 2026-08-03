from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Final, Literal

from worker.interfaces.serving import ServingClient
from worker.pipeline.analytics import (
    Clock,
    ExtractorSpec,
    NamedExtractor,
    provision_extractors,
)

PRODUCTION_EXTRACTOR_NAMES: Final = ("pose", "person", "bed")

BoxSource = Literal["pose", "person"]


@dataclass(frozen=True, slots=True)
class SharedYoloExtractors:
    """Hold the process-shared YOLO extractors provisioned by runtime.

    ``person`` is ``None`` whenever ``box_source`` (issue #44) is "pose" --
    the person model is then never provisioned at all, not merely unused.
    """

    pose: NamedExtractor
    person: NamedExtractor | None
    bed: NamedExtractor

    @property
    def extractors(self) -> tuple[NamedExtractor, ...]:
        return tuple(
            extractor for extractor in (self.pose, self.person, self.bed) if extractor is not None
        )


def compose_yolo_extractors(
    serving_client: ServingClient,
    *,
    device: str,
    box_source: BoxSource = "pose",
    clock: Clock = perf_counter,
) -> SharedYoloExtractors:
    """Provision the production pose and bed runners, plus person only when
    ``box_source`` selects it as the authoritative box source (issue #44).

    When ``box_source`` is "pose" (the default), the person model is never
    provisioned -- there is no way to load it and then simply not schedule
    it, so a facility that never wants person boxes pays no extraction cost
    for them at all.
    """
    names = tuple(
        name
        for name in PRODUCTION_EXTRACTOR_NAMES
        if name != "person" or box_source == "person"
    )
    specs = tuple(
        ExtractorSpec(
            module_name=name,
            task=name,
            options=(("device", device),),
        )
        for name in names
    )
    extractors = provision_extractors(serving_client, specs, clock=clock)
    by_name = dict(zip(names, extractors, strict=True))
    return SharedYoloExtractors(
        pose=by_name["pose"],
        person=by_name.get("person"),
        bed=by_name["bed"],
    )


__all__ = [
    "PRODUCTION_EXTRACTOR_NAMES",
    "BoxSource",
    "SharedYoloExtractors",
    "compose_yolo_extractors",
]
