"""Closed camera mutation fields, preserving omitted fields separately from null."""

from pydantic import BaseModel, ConfigDict, Field

from backend.app.features.cameras.camera_values import CameraStatus


class CameraUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    label: str = Field(default="")
    rtsp_url: str = Field(default="")
    backend_camera_id: str | None = Field(default=None)
    mapping_pending: bool = Field(default=False)
    space_id: str | None = Field(default=None)
    decode_backend: str | None = Field(default=None)
    floor: int | None = Field(default=None)
    never_connected: bool = Field(default=True)
    last_probed_at: str | None = Field(default=None)
    last_ok_at: str | None = Field(default=None)
    edge_ref: str | None = Field(default=None)
    room_edge_ref: str | None = Field(default=None)
    status: CameraStatus = Field(default="unknown")


__all__ = ["CameraUpdate"]
