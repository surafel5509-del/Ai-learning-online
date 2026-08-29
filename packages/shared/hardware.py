"""Hardware detection and live monitoring — real values only.

Uses torch.cuda (when available) and psutil for CPU/memory. GPU metrics are
real NVML queries when a CUDA device exists; never faked.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class DeviceInfo:
    device: str          # 'cpu' | 'gpu'
    name: str
    cuda_available: bool
    cores: Optional[int] = None
    vram_total_mb: Optional[int] = None
    vram_used_mb: Optional[int] = None
    vram_free_mb: Optional[int] = None
    gpu_utilization_pct: Optional[float] = None
    cpu_count: Optional[int] = None
    memory_total_mb: Optional[float] = None
    memory_used_mb: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


def detect_device() -> DeviceInfo:
    """Detect compute device and base capabilities."""
    import torch
    try:
        import psutil
        cores = psutil.cpu_count(logical=True) or os.cpu_count() or 1
        mem = psutil.virtual_memory()
        mem_total = mem.total / (1024 * 1024)
        mem_used = mem.used / (1024 * 1024)
    except Exception:
        cores = os.cpu_count() or 1
        mem_total = mem_used = None

    if torch.cuda.is_available():
        idx = 0
        name = torch.cuda.get_device_name(idx)
        props = torch.cuda.get_device_properties(idx)
        vram_total = props.total_memory / (1024 * 1024)
        vram_used = torch.cuda.memory_allocated(idx) / (1024 * 1024)
        vram_free = vram_total - vram_used
        util = None
        try:
            util = float(torch.cuda.utilization(idx))
        except Exception:
            util = None
        return DeviceInfo(
            device="gpu", name=name, cuda_available=True, cores=cores,
            vram_total_mb=int(vram_total), vram_used_mb=int(vram_used),
            vram_free_mb=int(vram_free), gpu_utilization_pct=util,
            cpu_count=cores, memory_total_mb=mem_total, memory_used_mb=mem_used,
        )
    # CPU fallback
    import platform
    return DeviceInfo(
        device="cpu", name=platform.processor() or "CPU", cuda_available=False,
        cores=cores, cpu_count=cores,
        memory_total_mb=mem_total, memory_used_mb=mem_used,
    )


def live_stats() -> dict:
    """Snapshot of live CPU/GPU/memory utilization (real)."""
    out: dict = {}
    try:
        import psutil
        out["cpu_percent"] = psutil.cpu_percent(interval=None)
        out["cpu_count"] = psutil.cpu_count(logical=True)
        mem = psutil.virtual_memory()
        out["memory_total_mb"] = mem.total / (1024 * 1024)
        out["memory_used_mb"] = mem.used / (1024 * 1024)
        out["memory_percent"] = mem.percent
    except Exception:
        out["cpu_percent"] = None
    try:
        import torch
        if torch.cuda.is_available():
            idx = 0
            out["gpu_name"] = torch.cuda.get_device_name(idx)
            out["gpu_utilization_pct"] = float(torch.cuda.utilization(idx))
            out["vram_total_mb"] = torch.cuda.get_device_properties(idx).total_memory / (1024 * 1024)
            out["vram_used_mb"] = torch.cuda.memory_allocated(idx) / (1024 * 1024)
            out["vram_free_mb"] = out["vram_total_mb"] - out["vram_used_mb"]
    except Exception:
        pass
    return out


def resolve_device(choice: str) -> str:
    """Map 'auto'|'cpu'|'gpu' to an actual torch device string."""
    import torch
    if choice == "cpu":
        return "cpu"
    if choice == "gpu":
        if torch.cuda.is_available():
            return "cuda"
        raise RuntimeError("GPU requested but CUDA is not available on this host.")
    # auto
    return "cuda" if torch.cuda.is_available() else "cpu"
