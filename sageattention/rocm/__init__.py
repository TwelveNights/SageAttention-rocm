"""
ROCm (AMD GPU) backend for SageAttention: Triton kernels with INT8 Q·K^T and
FP16 / FP8 P·V. See ``ROCM_NOTES.md`` at the repository root.
"""

from .api import QK_GRAN_TO_QBLK, default_config, sageattn_rocm, sageattn_varlen_rocm
from .arch import arch_family, fp8_dtype, gcn_arch, has_fp8, is_rocm

__all__ = [
    "sageattn_rocm",
    "sageattn_varlen_rocm",
    "default_config",
    "QK_GRAN_TO_QBLK",
    "is_rocm",
    "gcn_arch",
    "arch_family",
    "has_fp8",
    "fp8_dtype",
]
