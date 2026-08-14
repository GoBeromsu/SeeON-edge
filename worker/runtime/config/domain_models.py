from __future__ import annotations

import sys
from types import MappingProxyType
from typing import ClassVar, Final, TypeAlias
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from worker.domains.module_compiler import CompiledDetectionModuleRegistry
from worker.domains.registry import (
    DETECTION_MODULE_REGISTRY,
    DOMAIN_REGISTRY,
    EXTERNAL_DOMAIN_MODULE_IDS,
)
from worker.runtime.config.errors import ConfigValidationError

# Derived from the registry rather than hand-maintained, so this can never
# drift from the set of domains `worker.domains.registry` actually knows how
# to build -- see worker/domains/AGENTS.md and the module docstring below for
# how a new registry entry is supposed to reach here without an edit.
KNOWN_DOMAIN_NAMES: Final = frozenset(DOMAIN_REGISTRY)
_LEGACY_CONFIG_FIELDS: Final = MappingProxyType(
    {
        "fall": "fall",
        "bed_exit": "bed_exit",
    }
)
_LEGACY_WINDOW_FIELDS: Final = MappingProxyType(
    {
        "bed_exit": "night_window",
    }
)


class NightWindowConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    start: str
    end: str
    tz: str

    @field_validator("start", "end")
    @classmethod
    def _validate_time(cls, value: str) -> str:
        hour, separator, minute = value.partition(":")
        valid = (
            separator == ":"
            and hour.isdigit()
            and minute.isdigit()
            and len(hour) == 2
            and len(minute) == 2
            and 0 <= int(hour) <= 23
            and 0 <= int(minute) <= 59
        )
        if not valid:
            raise ConfigValidationError("night window time must use HH:MM")
        return value

    @field_validator("tz")
    @classmethod
    def _validate_timezone(cls, value: str) -> str:
        try:
            _ = ZoneInfo(value)
        except (ValueError, ZoneInfoNotFoundError) as error:
            raise ConfigValidationError(
                "night window tz must be a valid IANA timezone"
            ) from error
        return value

    @model_validator(mode="after")
    def _reject_degenerate_window(self) -> NightWindowConfig:
        # start == end would make DetectionWindow.contains's
        # `start <= t < end` range match nothing, permanently disabling the
        # domain -- reject it here rather than silently accepting an empty
        # window. Operators who want 24/7 detection omit the window entirely.
        if self.start == self.end:
            raise ConfigValidationError(
                "night window start and end must not be equal (an equal "
                "start/end window matches nothing); omit the window for "
                "24/7 detection instead"
            )
        return self


class FallDomainConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True


class BedExitDomainConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    # Deprecated alias for detection_windows["bed_exit"]; an explicit
    # DomainsConfig.detection_windows entry for "bed_exit" wins over this.
    night_window: NightWindowConfig | None = None


DomainConfig: TypeAlias = FallDomainConfig | BedExitDomainConfig


class DomainsConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    enabled: tuple[str, ...] | None = None
    # Canonical qualified selection. When present it is the complete active
    # module set and is checked against the injected compiled registry at boot.
    versions: dict[str, int] | None = None
    fall: FallDomainConfig | None = None
    bed_exit: BedExitDomainConfig | None = None
    # Per-domain detection windows, keyed by domain name. Deliberately
    # lenient (unlike ``enabled``): unknown domain names are accepted so this
    # stays forward-compatible with domains this worker build doesn't know
    # about yet.
    detection_windows: dict[str, NightWindowConfig | None] | None = None

    @field_validator("enabled")
    @classmethod
    def _validate_enabled(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        if value is None:
            return None
        duplicate = sorted(name for name in set(value) if value.count(name) > 1)
        if duplicate:
            raise ConfigValidationError(
                "domains.enabled contains duplicate domain: " + ", ".join(duplicate)
            )
        unknown = sorted(set(value) - KNOWN_DOMAIN_NAMES)
        if unknown:
            raise ConfigValidationError(
                "domains.enabled contains unknown domain: " + ", ".join(unknown)
            )
        return value

    @field_validator("versions")
    @classmethod
    def _validate_versions(cls, value: dict[str, int] | None) -> dict[str, int] | None:
        if value is None:
            return None
        if any(not module_id or version < 1 for module_id, version in value.items()):
            raise ConfigValidationError(
                "domains.versions requires non-empty ids and positive versions"
            )
        return value

    @model_validator(mode="after")
    def _reject_mixed_legacy_and_map(self) -> DomainsConfig:
        configured_names = KNOWN_DOMAIN_NAMES.intersection(self.model_fields_set)
        if self.versions is not None and (self.enabled is not None or configured_names):
            raise ConfigValidationError(
                "domains.versions cannot be combined with legacy domain selection"
            )
        if self.enabled is not None and configured_names:
            raise ConfigValidationError(
                "domains.enabled cannot be combined with per-domain config"
            )
        return self

    def selected_versions(
        self,
        registry: CompiledDetectionModuleRegistry = DETECTION_MODULE_REGISTRY,
    ) -> MappingProxyType[str, int]:
        if self.versions is not None:
            selected = dict(self.versions)
        else:
            selected = {
                EXTERNAL_DOMAIN_MODULE_IDS[name]: registry.latest_versions[
                    EXTERNAL_DOMAIN_MODULE_IDS[name]
                ]
                for name in self.resolved_active_names()
            }
        for module_id, version in selected.items():
            _ = registry.get(module_id, version)
        return MappingProxyType(selected)

    def domain_config(self, name: str) -> DomainConfig | None:
        field = _LEGACY_CONFIG_FIELDS.get(name)
        if field is None:
            raise ConfigValidationError(f"unknown domain: {name}")
        return getattr(self, field)

    def resolved_detection_window(self, name: str) -> NightWindowConfig | None:
        """Resolve domain ``name``'s detection window.

        An explicit ``detection_windows[name]`` entry (including an explicit
        ``None``, i.e. no window) always wins. Otherwise falls back to
        ``BedExitDomainConfig.night_window`` for "bed_exit" (the only
        pre-existing per-domain window alias); other domains have no legacy
        alias and default to no window (24/7).
        """
        if self.detection_windows is not None and name in self.detection_windows:
            return self.detection_windows[name]
        alias = _LEGACY_WINDOW_FIELDS.get(name)
        if alias is None:
            return None
        config = self.domain_config(name)
        return None if config is None else getattr(config, alias)

    @property
    def enabled_domains(self) -> tuple[str, ...] | None:
        if self.versions is not None:
            return tuple(self.versions)
        if self.enabled is not None:
            return self.enabled
        configured = tuple(
            name
            for name in _LEGACY_CONFIG_FIELDS
            if (config := self.domain_config(name)) is not None and config.enabled
        )
        configured_fields = KNOWN_DOMAIN_NAMES.intersection(self.model_fields_set)
        return configured if configured_fields else None

    def resolved_active_names(self) -> tuple[str, ...]:
        overrides = self.resolved_overrides()
        return tuple(
            name
            for name, registration in DOMAIN_REGISTRY.items()
            if overrides.get(name, registration.enabled)
        )

    def resolved_overrides(self) -> dict[str, bool]:
        """The per-domain enable/disable overlay on top of the registry.

        This is the single resolution path config feeds into
        ``DOMAIN_REGISTRY``: callers resolve one domain's active state as
        ``overrides.get(name, DOMAIN_REGISTRY[name].enabled)``, iterating
        ``DOMAIN_REGISTRY`` itself (never this map, and never this class) for
        the set of domains that exist -- see
        ``worker.runtime.config.worker_models.WorkerConfig.enabled_domains``.

        Returns an unconditional ``dict``, never ``None``: an empty map
        unambiguously means "no overrides, defer to the registry for every
        domain" now that config only *overlays* the registry instead of
        replacing it. A domain this config never mentions is simply absent
        from the map -- it is not forced off, which is what made "not
        configured" and "everything off" indistinguishable before this
        overlay existed.

        ``self.enabled`` (the legacy replace-list alias) takes priority when
        present, since ``_reject_mixed_legacy_and_map`` already forbids
        combining it with the per-domain map form below. It is still the
        live relay's wire shape for per-camera-declared domains (see
        ``pull_models.BackendWorkerConfigPayload.to_worker_config``), so it
        stays supported -- deprecated, not removed -- and is converted to an
        overlay by forcing every listed name on and every *other known*
        domain off, matching its old all-or-nothing replace semantics for
        anyone still sending it.
        """
        if self.enabled is not None:
            print(
                "domains.enabled (legacy replace-list) is deprecated; prefer "
                "per-domain domains.<name>.enabled overrides instead. "
                f"Treating {sorted(self.enabled)!r} as the complete active "
                "set and forcing every other known domain off.",
                file=sys.stderr,
            )
            listed = set(self.enabled)
            return {name: name in listed for name in KNOWN_DOMAIN_NAMES}
        overrides: dict[str, bool] = {}
        for name in _LEGACY_CONFIG_FIELDS:
            config = self.domain_config(name)
            if config is not None:
                overrides[name] = config.enabled
        return overrides


__all__ = [
    "KNOWN_DOMAIN_NAMES",
    "BedExitDomainConfig",
    "DomainConfig",
    "DomainsConfig",
    "FallDomainConfig",
    "NightWindowConfig",
]
