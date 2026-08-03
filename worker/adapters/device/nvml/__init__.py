"""Real NVML GPU-telemetry adapter."""

from __future__ import annotations

from worker.adapters.device.nvml.probe import NvmlGpuStatus, NvmlImporter, probe_nvml_gpu_status

__all__ = ["NvmlGpuStatus", "NvmlImporter", "probe_nvml_gpu_status"]
