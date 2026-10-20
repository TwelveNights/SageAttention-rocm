"""
ROCm implementations behind SageAttention's public API.

``sageattention.core`` routes every public entry point here when torch is a
ROCm build; the semantics (layouts, ``smooth_k``, ``return_lse``, GQA,
causal, masks, varlen) follow upstream. What differs is what runs:

- Q·K^T is always INT8 (``tl.dot`` on int8 -> MFMA/WMMA ``i32_*_i8``);
  upstream's ``qk_quant_gran`` ("per_block" / "per_warp" / "per_thread")
  maps to the Q-scale block height (128 / 32 / 16 rows here). K is
  quantized per 64 keys (or per kv tile if larger) after smoothing.
- P·V is FP16 with FP32 accumulation, or FP8 on GPUs whose matrix cores take
  FP8 (CDNA3 gfx942: e4m3fnuz) with per-channel V scales. Requests for FP8
  on RDNA3 (no FP8 hardware) run FP16 P·V instead. ``pv_accum_dtype`` only
  selects FP8 vs FP16 here: AMD matrix cores always accumulate in FP32, so
  upstream's FP16 / "fp32+fp16" accumulator variants have no equivalent
  (and are not needed for accuracy).
"""

import os
import warnings
from typing import Any, Optional

import torch

from . import arch as _arch
from .attn import sage_attn_fwd
from .quant import padded_head_dim, quant_qk_int8, quant_v_fp8, seq_mean

__all__ = ["sageattn_rocm", "sageattn_varlen_rocm", "default_config", "default_pv_dtype", "QK_GRAN_TO_QBLK"]

QK_GRAN_TO_QBLK = {"per_block": 128, "per_warp": 32, "per_thread": 16}

_warned: set[str] = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _warned:
        _warned.add(key)
        warnings.warn(msg, RuntimeWarning, stacklevel=3)


def _parse_env_config() -> dict | None:
    """``SAGEATTN_ROCM_CONFIG="BLOCK_M=128,BLOCK_N=64,num_warps=4,num_stages=2,waves_per_eu=2"`` (tuning/debug)."""
    s = os.environ.get("SAGEATTN_ROCM_CONFIG")
    if not s:
        return None
    cfg = {}
    for part in s.split(","):
        k, v = (x.strip() for x in part.split("="))
        try:
            cfg[k] = int(v)
        except ValueError:
            try:
                cfg[k] = float(v)
            except ValueError:
                cfg[k] = v
    return cfg


def default_config(head_dim: int, is_causal: bool, pv_fp8: bool, device_index: int = 0) -> dict:
    """Kernel tiling for this GPU family (tuned offline; see ROCM_NOTES.md)."""
    env = _parse_env_config()
    if env is not None:
        return env
    fam = _arch.arch_family(device_index)
    dp = padded_head_dim(head_dim)
    if fam in ("cdna3", "cdna4", "cdna2"):
        # MI300X sweeps (bench/rocm/bench_attn.py --sweep), B1 H24 N16384 D128 and B4 H24 N8192 D64
        base = dict(num_warps=4, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=32)
        if dp <= 64:
            return dict(BLOCK_M=128, BLOCK_N=128, schedule_hint="attention", **base)
        if dp <= 128:
            if pv_fp8:
                return dict(BLOCK_M=128, BLOCK_N=128, schedule_hint="attention", **base)
            return dict(BLOCK_M=128, BLOCK_N=64, **base)
        return dict(BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=1, waves_per_eu=1)
    # RDNA3 / RDNA4 / other: wave32. RX 7800 XT sweep, B1 H24 N4608 D128: 42 TFLOP/s (BLOCK_M=128, BLOCK_N=64,
    # 8 warps: 36-38); see native/ for the faster hand-written gfx11 kernel
    if dp <= 128:
        return dict(BLOCK_M=64, BLOCK_N=32, num_warps=4, num_stages=2)
    return dict(BLOCK_M=64, BLOCK_N=32, num_warps=4, num_stages=1)


def default_pv_dtype(device_index: int = 0) -> str:
    """P·V precision ``sageattn()`` uses on this GPU (``SAGEATTN_ROCM_PV=fp16|fp8`` overrides)."""
    env = os.environ.get("SAGEATTN_ROCM_PV")
    if env:
        return env
    return "fp16"


def _check_inputs(q, k, v):
    dtype = q.dtype
    assert q.is_cuda, "Input tensors must be on a GPU (cuda/hip device)."
    assert dtype in [torch.float16, torch.bfloat16], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim of qkv must be contiguous."
    if q.size(-1) > 256:
        raise ValueError(f"Unsupported head_dim: {q.size(-1)}")


def _want_fp8(pv_dtype: str, device_index: int) -> bool:
    if pv_dtype == "fp16":
        return False
    if pv_dtype != "fp8":
        raise ValueError(f"pv_dtype must be 'fp16' or 'fp8', got {pv_dtype!r}")
    if _arch.has_fp8(device_index):
        return True
    _warn_once(
        "fp8",
        f"SageAttention (ROCm): {_arch.gcn_arch(device_index)} has no FP8 matrix instructions; "
        "running P·V in FP16 instead.",
    )
    return False


def sageattn_rocm(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
    attn_mask: Optional[torch.Tensor] = None,
    smooth_k: bool = True,
    qk_quant_gran: str = "per_warp",
    pv_dtype: str = "fp16",
    config: Optional[dict] = None,
    _native: bool = True,
    **kwargs: Any,
):
    """
    INT8-QK SageAttention on AMD GPUs. Same arguments and return values as
    upstream's ``sageattn_qk_int8_pv_fp16_triton``; additionally
    ``qk_quant_gran`` (Q-scale granularity), ``pv_dtype`` ("fp16" or "fp8")
    and ``config`` (kernel tiling override).
    """
    _check_inputs(q, k, v)
    if tensor_layout not in ("HND", "NHD"):
        raise ValueError(f"tensor_layout {tensor_layout} not supported")
    if qk_quant_gran not in QK_GRAN_TO_QBLK:
        raise ValueError(f"qk_quant_gran must be one of {list(QK_GRAN_TO_QBLK)}")
    dtype = q.dtype
    dev = q.device.index if q.device.index is not None else torch.cuda.current_device()
    torch.cuda.set_device(v.device)  # upstream's workaround for multi-GPU / torch.compile

    head_dim = q.size(-1)
    seq_dim = 1 if tensor_layout == "NHD" else 2
    nh_dim = 2 if tensor_layout == "NHD" else 1
    if sm_scale is None:
        sm_scale = head_dim**-0.5

    if is_causal:
        assert attn_mask is None, "Mask should be None for causal attention."

    if q.numel() == 0:  # no queries (e.g. an empty batch): nothing to launch
        o = torch.empty_like(q)
        if return_lse:
            b, nq, hq = q.shape[0], q.shape[seq_dim], q.shape[nh_dim]
            return o, torch.empty((b, hq, nq), dtype=torch.float32, device=q.device)
        return o

    pv_fp8 = _want_fp8(pv_dtype, dev)
    if _native and config is None:
        # hand-written gfx11 kernel (native/sage_gfx11.hip); None -> not applicable, use Triton
        from .native.sage import try_sageattn

        res = try_sageattn(q, k, v, tensor_layout, is_causal, sm_scale, return_lse, smooth_k, qk_quant_gran, attn_mask, pv_fp8)
        if res is not None:
            return res

    if smooth_k:
        km = seq_mean(k, tensor_layout)  # [B, Hk, D] fp32
        if return_lse:
            nq, nk = q.size(nh_dim), k.size(nh_dim)
            km_b = km.repeat_interleave(nq // nk, dim=1) if nq != nk else km
            qh = q.transpose(1, 2) if tensor_layout == "NHD" else q
            lse_correction = torch.matmul(qh.float(), km_b.unsqueeze(-1)).squeeze(-1)  # [B, Hq, Nq]
    else:
        km = None

    cfg = dict(config) if config is not None else default_config(head_dim, is_causal, pv_fp8, dev)
    QBLK = QK_GRAN_TO_QBLK[qk_quant_gran]
    KBLK = max(64, cfg["BLOCK_N"])

    q_int8, q_scale, k_int8, k_scale = quant_qk_int8(q, k, km, sm_scale, BLKQ=QBLK, BLKK=KBLK, tensor_layout=tensor_layout)

    if pv_fp8:
        v_in, v_scale = quant_v_fp8(v, tensor_layout, _arch.fp8_dtype(dev))
    else:
        v_in = v if v.dtype == torch.float16 else v.to(torch.float16)
        v_scale = None

    if attn_mask is not None:
        assert attn_mask.dtype == torch.bool or attn_mask.dtype == q.dtype, "attn_mask must be of dtype bool or the same dtype as q."
        assert attn_mask.device == q.device, "All tensors must be on the same device."
        if tensor_layout == "HND":
            target_shape = (q.shape[0], q.shape[1], q.shape[2], k.shape[2])
        else:
            target_shape = (q.shape[0], q.shape[2], q.shape[1], k.shape[1])
        try:
            attn_mask = attn_mask.expand(target_shape)
        except Exception:
            raise AssertionError(f"attn_mask shape {attn_mask.shape} cannot be broadcast to {target_shape}")

    o = torch.empty_like(q)
    lse = sage_attn_fwd(
        q_int8,
        k_int8,
        v_in,
        q_scale,
        k_scale,
        o,
        QBLK=QBLK,
        KBLK=KBLK,
        config=cfg,
        tensor_layout=tensor_layout,
        is_causal=is_causal,
        attn_mask=attn_mask,
        v_scale=v_scale,
        return_lse=return_lse,
    )
    if return_lse:
        lse = lse / 1.44269504
        if smooth_k:
            lse = lse + lse_correction * sm_scale
        return o, lse
    return o


def sageattn_varlen_rocm(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    smooth_k: bool = True,
    qk_quant_gran: str = "per_warp",
    config: Optional[dict] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Upstream ``sageattn_varlen`` on AMD GPUs (INT8 QK, FP16 PV). ``q``/``k``/``v``: ``[total, H, D]``."""
    _check_inputs(q, k, v)
    assert cu_seqlens_q.is_contiguous() and cu_seqlens_k.is_contiguous(), "cu_seqlens_q and cu_seqlens_k must be contiguous."
    dev = q.device.index if q.device.index is not None else torch.cuda.current_device()
    torch.cuda.set_device(v.device)
    head_dim = q.size(-1)
    if sm_scale is None:
        sm_scale = head_dim**-0.5
    # like upstream: one mean over all sequences (any per-head constant shift of K is exact)
    km = k.mean(dim=0, keepdim=True, dtype=torch.float32) if smooth_k else None
    cfg = dict(config) if config is not None else default_config(head_dim, is_causal, False, dev)
    QBLK = QK_GRAN_TO_QBLK[qk_quant_gran]
    KBLK = max(64, cfg["BLOCK_N"])
    cu_q = cu_seqlens_q.to(torch.int32)
    cu_k = cu_seqlens_k.to(torch.int32)
    q_int8, q_scale, k_int8, k_scale, cqs, cks = quant_qk_int8(
        q, k, km, sm_scale, BLKQ=QBLK, BLKK=KBLK, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k
    )
    v_in = v if v.dtype == torch.float16 else v.to(torch.float16)
    o = torch.empty_like(q)
    sage_attn_fwd(
        q_int8,
        k_int8,
        v_in,
        q_scale,
        k_scale,
        o,
        QBLK=QBLK,
        KBLK=KBLK,
        config=cfg,
        is_causal=is_causal,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        cu_q_scale=cqs,
        cu_k_scale=cks,
        max_seqlen_q=max_seqlen_q,
    )
    return o
