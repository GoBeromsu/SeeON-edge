"""Domain decision modules and their shared audit contracts."""

from __future__ import annotations

from worker.domains.base import (
    AuditContext,
    DomainAuditSnapshot,
    DomainDependencyError,
    DomainDetector,
)
from worker.domains.module_compiler import (
    CompiledDetectionModuleRegistry,
    DetectionModuleActivation,
    compile_detection_module_registry,
)
from worker.domains.module_definition import (
    CameraDetectionModule,
    CameraModuleContext,
    ComponentBinding,
    DetectionModuleActivationError,
    DetectionModuleCompilationError,
    DetectionModuleDefinition,
    PolicySchemaIdentity,
    ScheduleRule,
    SharedComponentIdentity,
)
from worker.domains.registry import (
    AVAILABLE_OBSERVATION_CHANNELS,
    DETECTION_MODULE_REGISTRY,
    DOMAIN_REGISTRY,
    EXTERNAL_DOMAIN_MODULE_IDS,
    BedExitDomainDependencies,
    DomainRegistration,
    FallDomainDependencies,
    enabled_domains,
    list_domains,
)

__all__ = [
    "AVAILABLE_OBSERVATION_CHANNELS",
    "DETECTION_MODULE_REGISTRY",
    "DOMAIN_REGISTRY",
    "EXTERNAL_DOMAIN_MODULE_IDS",
    "AuditContext",
    "CameraDetectionModule",
    "CameraModuleContext",
    "CompiledDetectionModuleRegistry",
    "ComponentBinding",
    "BedExitDomainDependencies",
    "DomainAuditSnapshot",
    "DomainDependencyError",
    "DetectionModuleActivation",
    "DetectionModuleActivationError",
    "DetectionModuleCompilationError",
    "DetectionModuleDefinition",
    "DomainDetector",
    "DomainRegistration",
    "FallDomainDependencies",
    "PolicySchemaIdentity",
    "ScheduleRule",
    "SharedComponentIdentity",
    "compile_detection_module_registry",
    "enabled_domains",
    "list_domains",
]
