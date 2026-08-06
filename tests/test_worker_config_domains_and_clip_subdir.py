"""Worker-side plumbing for the two ml-api-local overrides added alongside the
detection-settings and clip-storage-location backend slices (see
``.omc/plans/redesign-api-contracts.md`` §4/§5):

* ``BackendWorkerConfigPayload.domains`` -- a per-domain enable/disable map
  (``backend/app/features/detection_settings``) that, once present at all,
  replaces the per-camera-``domains``-derived enabled set entirely rather than
  merging with it.
* ``BackendWorkerConfigPayload.clip_store_subdir`` -- a single relative
  subdirectory (``backend/app/features/clips/storage_location_store.py``)
  threaded into ``ClipRecordingConfig.store_subdir`` and, at the worker
  runtime, appended under ``configured_store_dir()``.

Both fields degrade fail-open per malformed entry (mirroring the existing
``detection_windows``/``cameras`` degradation covered by
``test_worker_config_pull_models.py``) rather than rejecting the whole pulled
payload.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from worker.pipeline.output.evidence.clip_config import CLIP_STORE_DIR_ENV, DEFAULT_CLIP_STORE_DIR
from worker.runtime.config.pull_models import BackendWorkerConfigPayload
from worker.runtime.config.worker_models import ClipRecordingConfig
from worker.runtime.worker import WorkerRuntime, _required_extractor_names


def _camera_payload(camera_id: str = "camera-1") -> dict[str, object]:
    return {
        "camera_id": camera_id,
        "facility_id": "facility-1",
        "rtsp_url": "rtsp://camera/stream",
    }


# --- resolved_domain_enabled: fail-open per-domain parsing -----------------


def test_resolved_domain_enabled_parses_a_well_formed_map() -> None:
    payload = BackendWorkerConfigPayload.model_validate(
        {
            "config_version": 1,
            "cameras": [_camera_payload()],
            "domains": {"fall": {"enabled": True}, "bed_exit": {"enabled": False}},
        }
    )
    assert payload.resolved_domain_enabled == {"fall": True, "bed_exit": False}


def test_resolved_domain_enabled_is_empty_when_domains_absent() -> None:
    payload = BackendWorkerConfigPayload.model_validate(
        {"config_version": 1, "cameras": [_camera_payload()]}
    )
    assert payload.resolved_domain_enabled == {}
    assert payload.domains is None


@pytest.mark.parametrize(
    "domains_value",
    [
        {"fall": "not-an-object"},
        {"fall": {"enabled": "not-a-bool"}},
        {"unknown-domain": {"enabled": True}},
        {"": {"enabled": True}},
    ],
)
def test_resolved_domain_enabled_drops_only_the_malformed_entry_and_logs(
    domains_value: dict[object, object],
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = BackendWorkerConfigPayload.model_validate(
        {
            "config_version": 1,
            "cameras": [_camera_payload()],
            "domains": {**domains_value, "bed_exit": {"enabled": True}},
        }
    )

    resolved = payload.resolved_domain_enabled

    assert resolved == {"bed_exit": True}
    assert capsys.readouterr().err


# --- resolved_clip_store_subdir: fail-open parsing --------------------------


def test_resolved_clip_store_subdir_accepts_a_relative_path() -> None:
    payload = BackendWorkerConfigPayload.model_validate(
        {
            "config_version": 1,
            "cameras": [_camera_payload()],
            "clip_store_subdir": "backup-drive/clips",
        }
    )
    assert payload.resolved_clip_store_subdir == "backup-drive/clips"


def test_resolved_clip_store_subdir_is_none_when_absent() -> None:
    payload = BackendWorkerConfigPayload.model_validate(
        {"config_version": 1, "cameras": [_camera_payload()]}
    )
    assert payload.resolved_clip_store_subdir is None


@pytest.mark.parametrize(
    "value",
    ["/etc/passwd", "../escape", "sub/../../escape", "", "   ", 123, {"path": "sub"}],
)
def test_resolved_clip_store_subdir_falls_open_to_none_on_malformed_value(
    value: object,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = BackendWorkerConfigPayload.model_validate(
        {
            "config_version": 1,
            "cameras": [_camera_payload()],
            "clip_store_subdir": value,
        }
    )

    assert payload.resolved_clip_store_subdir is None
    assert capsys.readouterr().err


# --- to_worker_config: merge behavior ---------------------------------------


def test_to_worker_config_with_no_domains_override_preserves_camera_domains_default() -> None:
    payload = BackendWorkerConfigPayload.model_validate(
        {
            "config_version": 1,
            "cameras": [{**_camera_payload(), "domains": ["fall"]}],
        }
    )

    config = payload.to_worker_config("http://ml-api:8000", "relay-secret")

    assert config.domains.enabled == ("fall",)
    assert config.clip.store_subdir is None


def test_to_worker_config_domains_override_replaces_camera_domains_entirely() -> None:
    """A local domains override wins outright, even over a per-camera
    ``domains`` list that would otherwise have enabled a different set."""
    payload = BackendWorkerConfigPayload.model_validate(
        {
            "config_version": 1,
            "cameras": [{**_camera_payload(), "domains": ["fall"]}],
            "domains": {"fall": {"enabled": False}, "bed_exit": {"enabled": True}},
        }
    )

    config = payload.to_worker_config("http://ml-api:8000", "relay-secret")

    assert config.domains.enabled == ("bed_exit",)


def test_to_worker_config_domains_override_can_represent_all_domains_off() -> None:
    """The empty-tuple-vs-None ambiguity this override was designed to avoid:
    an explicit "everything off" must stay an explicit empty tuple, not
    collapse into DomainsConfig's "nothing configured" ambient enable-all."""
    payload = BackendWorkerConfigPayload.model_validate(
        {
            "config_version": 1,
            "cameras": [{**_camera_payload(), "domains": ["fall", "bed_exit"]}],
            "domains": {"fall": {"enabled": False}, "bed_exit": {"enabled": False}},
        }
    )

    config = payload.to_worker_config("http://ml-api:8000", "relay-secret")

    assert config.domains.enabled == ()
    assert config.domains.enabled_domains == ()


# --- to_worker_config: issue #191 fail-open default ------------------------
#
# ``WorkerConfig.enabled_domains`` only falls open to the registry default
# (fall, bed_exit) when its ``domains`` field was never passed at all
# (pydantic's ``model_fields_set``). ``to_worker_config`` used to pass
# ``domains=`` unconditionally, so a relay pull that carried no domains
# signal whatsoever (no override, no per-camera domains, no detection
# windows -- the exact shape of a fresh install) permanently disarmed that
# fallback and booted with detection silently off.


def test_to_worker_config_with_no_domains_signal_falls_open_to_none() -> None:
    """No domains override, no per-camera domains, no detection windows:
    ``enabled_domains`` must be ``None`` (fail-open marker), not ``()``."""
    payload = BackendWorkerConfigPayload.model_validate(
        {"config_version": 1, "cameras": [_camera_payload()]}
    )

    config = payload.to_worker_config("http://ml-api:8000", "relay-secret")

    assert config.enabled_domains is None


def test_to_worker_config_with_explicit_empty_camera_domains_list_stays_off() -> None:
    """A camera that explicitly declares zero domains (``"domains": []``) is
    a genuine opt-out, distinct from a camera that never mentioned domains
    at all, and must resolve to empty -- not the registry default."""
    payload = BackendWorkerConfigPayload.model_validate(
        {
            "config_version": 1,
            "cameras": [{**_camera_payload(), "domains": []}],
        }
    )

    config = payload.to_worker_config("http://ml-api:8000", "relay-secret")

    assert config.enabled_domains == ()


def test_to_worker_config_with_specific_camera_domains_resolves_exactly_as_given() -> None:
    payload = BackendWorkerConfigPayload.model_validate(
        {
            "config_version": 1,
            "cameras": [{**_camera_payload(), "domains": ["fall"]}],
        }
    )

    config = payload.to_worker_config("http://ml-api:8000", "relay-secret")

    assert config.enabled_domains == ("fall",)


def test_no_domains_signal_still_schedules_the_registry_default_extractors() -> None:
    """The actual production consequence of issue #191: with no domains
    signal at all, the runtime must resolve a non-empty active-domain set
    and schedule the extractors those domains require, not silently
    schedule nothing (``Scheduler.task_intervals`` empty,
    ``CompositeExtractor.process`` calling no extractors)."""
    from types import SimpleNamespace

    payload = BackendWorkerConfigPayload.model_validate(
        {"config_version": 1, "cameras": [_camera_payload()]}
    )
    config = payload.to_worker_config("http://ml-api:8000", "relay-secret")
    runtime = SimpleNamespace(config=config)

    domain_names = WorkerRuntime._active_domain_names(runtime)
    required = _required_extractor_names(domain_names)

    assert domain_names == ("fall", "bed_exit")
    assert required != ()
    assert "pose" in required


def test_to_worker_config_clip_store_subdir_merges_into_a_local_clip_config() -> None:
    """The pulled subdir must land on top of a locally-sourced ``clip``
    config (e.g. clip.enabled from env, see #66/#68) rather than replacing it
    outright."""
    payload = BackendWorkerConfigPayload.model_validate(
        {
            "config_version": 1,
            "cameras": [_camera_payload()],
            "clip_store_subdir": "external-drive",
        }
    )
    local_clip = ClipRecordingConfig(enabled=True)

    config = payload.to_worker_config("http://ml-api:8000", "relay-secret", clip=local_clip)

    assert config.clip.enabled is True
    assert config.clip.store_subdir == "external-drive"


def test_to_worker_config_without_clip_store_subdir_leaves_local_clip_config_untouched() -> None:
    payload = BackendWorkerConfigPayload.model_validate(
        {"config_version": 1, "cameras": [_camera_payload()]}
    )
    local_clip = ClipRecordingConfig(enabled=True, store_subdir="already-set")

    config = payload.to_worker_config("http://ml-api:8000", "relay-secret", clip=local_clip)

    assert config.clip.store_subdir == "already-set"


# --- ClipRecordingConfig.store_subdir validator -----------------------------


def test_clip_recording_config_accepts_a_relative_store_subdir() -> None:
    config = ClipRecordingConfig(store_subdir="sub/dir")
    assert config.store_subdir == "sub/dir"


@pytest.mark.parametrize("value", ["/absolute", "sub/../escape", ".."])
def test_clip_recording_config_rejects_unsafe_store_subdir(value: str) -> None:
    with pytest.raises(ValidationError):
        ClipRecordingConfig(store_subdir=value)


# --- WorkerRuntime._resolved_clip_store_dir ---------------------------------


def _fake_runtime(store_subdir: str | None) -> WorkerRuntime:
    """A minimal stand-in with just enough shape for
    ``_resolved_clip_store_dir`` (an unbound instance method) to run: it only
    reads ``self.config.clip.store_subdir``."""
    from types import SimpleNamespace

    return SimpleNamespace(config=SimpleNamespace(clip=SimpleNamespace(store_subdir=store_subdir)))


def test_resolved_clip_store_dir_is_the_bare_root_when_no_subdir_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(CLIP_STORE_DIR_ENV, raising=False)
    runtime = _fake_runtime(None)

    resolved = WorkerRuntime._resolved_clip_store_dir(runtime)

    assert resolved == Path(DEFAULT_CLIP_STORE_DIR)


def test_resolved_clip_store_dir_appends_a_selected_subdir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(CLIP_STORE_DIR_ENV, str(tmp_path))
    runtime = _fake_runtime("backup/clips")

    resolved = WorkerRuntime._resolved_clip_store_dir(runtime)

    assert resolved == tmp_path / "backup" / "clips"


@pytest.mark.parametrize("unsafe_subdir", ["/etc/passwd", "../escape", "sub/../../escape"])
def test_resolved_clip_store_dir_defensively_rejects_an_unsafe_subdir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, unsafe_subdir: str
) -> None:
    """Belt-and-suspenders: pull_models.py already validates this before it
    reaches WorkerConfig, but a path feeding filesystem construction is
    re-checked here rather than trusted at a distance."""
    monkeypatch.setenv(CLIP_STORE_DIR_ENV, str(tmp_path))
    runtime = _fake_runtime(unsafe_subdir)

    resolved = WorkerRuntime._resolved_clip_store_dir(runtime)

    assert resolved == tmp_path
