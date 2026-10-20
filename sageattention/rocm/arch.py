"""
AMD GPU detection for the ROCm backend.

Everything here is cheap and import-safe on any torch build: nothing touches
the GPU until a function is called.
"""

import functools
import os

import torch

__all__ = ["is_rocm", "gcn_arch", "is_cdna3", "is_rdna3", "has_fp8", "fp8_dtype", "arch_family"]


def is_rocm() -> bool:
    """True when torch is a ROCm (HIP) build."""
    return getattr(torch.version, "hip", None) is not None


@functools.lru_cache(maxsize=None)
def gcn_arch(device_index: int = 0) -> str:
    """``gfx942``, ``gfx1101``, ... (feature suffixes such as ``:sramecc+`` stripped)."""
    override = os.environ.get("SAGEATTN_ROCM_ARCH")
    if override:
        return override
    name = torch.cuda.get_device_properties(device_index).gcnArchName
    return name.split(":")[0]


def arch_family(device_index: int = 0) -> str:
    """``cdna3`` (MI300), ``cdna4`` (MI350), ``cdna2``, ``rdna3`` (RX 7000), ``rdna4`` or ``other``."""
    a = gcn_arch(device_index)
    if a in ("gfx940", "gfx941", "gfx942"):
        return "cdna3"
    if a == "gfx950":
        return "cdna4"
    if a in ("gfx90a", "gfx908"):
        return "cdna2"
    if a.startswith("gfx11"):
        return "rdna3"
    if a.startswith("gfx12"):
        return "rdna4"
    return "other"


def is_cdna3(device_index: int = 0) -> bool:
    return arch_family(device_index) in ("cdna3", "cdna4")


def is_rdna3(device_index: int = 0) -> bool:
    return arch_family(device_index) == "rdna3"


def has_fp8(device_index: int = 0) -> bool:
    """Whether the matrix cores take FP8 operands (CDNA3/4, RDNA4). RDNA3 has none."""
    return arch_family(device_index) in ("cdna3", "cdna4", "rdna4")


def fp8_dtype(device_index: int = 0) -> torch.dtype:
    """
    The FP8 format the matrix cores use: ``float8_e4m3fnuz`` on gfx942 (bias 8,
    max 240, no inf/-0) and OCP ``float8_e4m3fn`` (max 448) on gfx950 / gfx12.
    """
    return torch.float8_e4m3fnuz if gcn_arch(device_index) in ("gfx940", "gfx941", "gfx942") else torch.float8_e4m3fn
