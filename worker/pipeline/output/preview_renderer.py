"""CPU-rendered overlays for the operator live preview."""

from __future__ import annotations

import io
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from worker.types.perception_frame import PersonBox
from worker.types.preview import FallPreviewState, OverlaySelection

DEFAULT_FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")


@dataclass(frozen=True, slots=True)
class BedZoneGeometry:
    polygons: tuple[tuple[tuple[int, int], ...], ...]
    image_width: int
    image_height: int


@dataclass(frozen=True, slots=True)
class PreviewTrack:
    box: PersonBox
    source_width: int
    source_height: int
    track_id: int | None = None


class PreviewRenderer:
    """Draw persisted geometry and CPU policy state on a clean SDK JPEG."""

    _NORMAL_COLOR = (34, 197, 94)
    _SUSPECTED_COLOR = (239, 68, 68)
    _UNKNOWN_COLOR = (59, 130, 246)
    _BED_COLOR = (245, 158, 11)

    def __init__(self, font_path: str | Path = DEFAULT_FONT_PATH) -> None:
        # Loading is deliberately eager: a bad edge image must fail during
        # composition rather than intermittently on an HTTP request thread.
        self._font = ImageFont.truetype(str(font_path), 18)

    def render(
        self,
        jpeg: bytes,
        selection: OverlaySelection,
        tracks: Iterable[PreviewTrack],
        bed_geometry: BedZoneGeometry | None,
        fall_states: Mapping[int, FallPreviewState],
    ) -> bytes:
        tracks = tuple(tracks)
        draw_people = selection.person and bool(tracks)
        draw_bed = selection.bed and self._valid_bed_geometry(bed_geometry)
        if not draw_people and not draw_bed:
            return jpeg

        try:
            image = Image.open(io.BytesIO(jpeg)).convert("RGB")
            image.load()
        except (OSError, ValueError) as error:
            raise RuntimeError("live-frame snapshot is not a decodable color JPEG") from error

        if draw_bed:
            assert bed_geometry is not None
            self._draw_bed(image, bed_geometry)
        if draw_people:
            self._draw_people(image, tracks, fall_states, bed_geometry)

        output = io.BytesIO()
        try:
            image.save(output, format="JPEG", quality=90)
        except OSError as error:
            raise RuntimeError("live-frame JPEG encoding failed") from error
        return output.getvalue()

    @staticmethod
    def _valid_bed_geometry(geometry: BedZoneGeometry | None) -> bool:
        return bool(
            geometry is not None
            and bool(geometry.polygons)
            and all(len(polygon) >= 3 for polygon in geometry.polygons)
            and geometry.image_width > 0
            and geometry.image_height > 0
        )

    def _draw_bed(self, image: Image.Image, geometry: BedZoneGeometry) -> None:
        width, height = image.size
        for bed_number, polygon in enumerate(geometry.polygons, start=1):
            points = tuple(
                (
                    self._scaled_coordinate(x, width, geometry.image_width),
                    self._scaled_coordinate(y, height, geometry.image_height),
                )
                for x, y in polygon
            )
            overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.polygon(points, fill=(*self._BED_COLOR, 64))
            image.paste(Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB"))
            draw = ImageDraw.Draw(image)
            draw.line((*points, points[0]), fill=self._BED_COLOR, width=3, joint="curve")
            self._draw_label(
                draw,
                points[0],
                f"침대{bed_number}",
                self._BED_COLOR,
                image.size,
            )

    def _draw_people(
        self,
        image: Image.Image,
        tracks: tuple[PreviewTrack, ...],
        fall_states: Mapping[int, FallPreviewState],
        bed_geometry: BedZoneGeometry | None,
    ) -> None:
        draw = ImageDraw.Draw(image)
        width, height = image.size
        beds = self._scaled_beds(bed_geometry, image.size)
        for track in tracks:
            if track.source_width <= 0 or track.source_height <= 0:
                continue
            box = track.box
            left = self._scaled_coordinate(box.x1, width, track.source_width)
            top = self._scaled_coordinate(box.y1, height, track.source_height)
            right = self._scaled_coordinate(box.x2, width, track.source_width)
            bottom = self._scaled_coordinate(box.y2, height, track.source_height)
            if right <= left or bottom <= top:
                continue
            state = fall_states.get(track.track_id) if track.track_id is not None else None
            color = self._status_color(state)
            draw.rectangle((left, top, right, bottom), outline=color, width=3)
            bed_number = self._bed_number_at(((left + right) // 2, bottom), beds)
            self._draw_label(
                draw,
                (left, top),
                self._person_label(box.confidence, state, bed_number),
                color,
                image.size,
            )

    @classmethod
    def _status_color(cls, state: FallPreviewState | None) -> tuple[int, int, int]:
        if state is None:
            return cls._UNKNOWN_COLOR
        return cls._SUSPECTED_COLOR if state.status == "suspected" else cls._NORMAL_COLOR

    @staticmethod
    def _person_label(
        confidence: float, state: FallPreviewState | None, bed_number: int | None
    ) -> str:
        """Operator-facing label: detection confidence, bed, and fall state.

        The NvDCF track counter is deliberately absent: it restarts per
        stream, climbs on every re-acquisition, and tells an operator nothing.
        The detector confidence does - a 25 % box on a bed rail reads as the
        weak guess it is.
        """
        parts = [f"사람 {confidence:.0%}"]
        if bed_number is not None:
            parts.append(f"침대{bed_number}")
        if state is not None:
            if state.status == "suspected":
                probability = "" if state.probability is None else f" {state.probability:.2f}"
                parts.append(f"낙상 의심{probability}")
            else:
                parts.append("정상")
        return " · ".join(parts)

    def _scaled_beds(
        self, geometry: BedZoneGeometry | None, image_size: tuple[int, int]
    ) -> tuple[tuple[tuple[int, int], ...], ...]:
        if not self._valid_bed_geometry(geometry):
            return ()
        assert geometry is not None
        width, height = image_size
        return tuple(
            tuple(
                (
                    self._scaled_coordinate(x, width, geometry.image_width),
                    self._scaled_coordinate(y, height, geometry.image_height),
                )
                for x, y in polygon
            )
            for polygon in geometry.polygons
        )

    @staticmethod
    def _bed_number_at(
        point: tuple[int, int], beds: tuple[tuple[tuple[int, int], ...], ...]
    ) -> int | None:
        """1-based index of the first saved bed polygon containing the point.

        The point is the bottom-centre of the person box, which is where a
        standing person touches the floor and where a lying person's box sits
        inside the bed. Ray casting; no shapely in the runtime image.
        """
        px, py = point
        for bed_number, polygon in enumerate(beds, start=1):
            inside = False
            count = len(polygon)
            for index in range(count):
                x1, y1 = polygon[index]
                x2, y2 = polygon[(index + 1) % count]
                if (y1 > py) != (y2 > py):
                    crossing = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
                    if px < crossing:
                        inside = not inside
            if inside:
                return bed_number
        return None

    def _draw_label(
        self,
        draw: ImageDraw.ImageDraw,
        anchor: tuple[int, int],
        text: str,
        color: tuple[int, int, int],
        image_size: tuple[int, int],
    ) -> None:
        left, anchor_top = anchor
        text_box = draw.textbbox((0, 0), text, font=self._font)
        label_width = text_box[2] - text_box[0] + 8
        label_height = text_box[3] - text_box[1] + 6
        left = min(max(left, 0), max(image_size[0] - label_width, 0))
        top = anchor_top - label_height
        if top < 0:
            top = min(max(anchor_top, 0), max(image_size[1] - label_height, 0))
        draw.rectangle((left, top, left + label_width, top + label_height), fill=color)
        draw.text((left + 4, top + 3 - text_box[1]), text, font=self._font, fill=(255, 255, 255))

    @staticmethod
    def _scaled_coordinate(value: int, output_size: int, source_size: int) -> int:
        return min(max(round(value * output_size / source_size), 0), output_size - 1)


__all__ = ["BedZoneGeometry", "PreviewRenderer", "PreviewTrack"]
