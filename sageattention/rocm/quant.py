"""
Quantization kernels for the ROCm SageAttention backend (Triton).

- :func:`quant_qk_int8`: per-block symmetric INT8 quantization of Q and K,
  with K smoothing (``k - mean_seq(k)``) and the softmax scale (times
  ``log2(e)``, so the attention kernel can use ``exp2``) fused in. The
  outputs are written head-major and contiguous (``[B, H, N, Dp]``, or
  ``[H, total, Dp]`` for varlen), zero-padded to a power-of-two head dim
  ``Dp`` -- so callers never pad or transpose Q/K themselves.
- :func:`quant_v_fp8`: per-channel FP8 quantization of V (CDNA3: e4m3fnuz).

Rounding follows upstream's ``quant_per_block_int8_kernel``: ``x / scale``
rounded half away from zero, ``scale = max|x| / 127``.
"""

import torch
import triton
import triton.language as tl

__all__ = ["quant_qk_int8", "quant_v_fp8", "padded_head_dim"]


def padded_head_dim(d: int) -> int:
    """Power-of-two head dim the kernels compute with (>= 16)."""
    return max(16, triton.next_power_of_2(d))


@triton.jit
def _quant_int8_kernel(
    X,
    Mean,
    Out,
    Scale,
    CuSeq,
    CuScale,
    stride_xb,
    stride_xh,
    stride_xn,
    stride_mb,
    stride_mh,
    stride_ob,
    stride_oh,
    stride_on,
    stride_sb,
    stride_sh,
    L,
    scale_mul,
    D: tl.constexpr,
    DP: tl.constexpr,
    BLK: tl.constexpr,
    HAS_MEAN: tl.constexpr,
    VARLEN: tl.constexpr,
):
    blk = tl.program_id(0)
    h = tl.program_id(1).to(tl.int64)
    b = tl.program_id(2).to(tl.int64)

    if VARLEN:
        start = tl.load(CuSeq + b).to(tl.int64)
        L = (tl.load(CuSeq + b + 1) - start).to(tl.int32)
        if blk * BLK >= L:
            return
        sblk = tl.load(CuScale + b).to(tl.int64)
        x_base = X + start * stride_xn
        o_base = Out + start * stride_on
    else:
        sblk = 0
        x_base = X + b * stride_xb
        o_base = Out + b * stride_ob

    offs_n = blk * BLK + tl.arange(0, BLK)
    offs_d = tl.arange(0, DP)
    nmask = offs_n < L
    if D == DP:
        m = nmask[:, None]
    else:
        m = nmask[:, None] & (offs_d < D)[None, :]
    x = tl.load(x_base + h * stride_xh + offs_n[:, None] * stride_xn + offs_d[None, :], mask=m, other=0.0)
    x = x.to(tl.float32)
    if HAS_MEAN:
        mu = tl.load(Mean + b * stride_mb + h * stride_mh + offs_d, mask=offs_d < D, other=0.0).to(tl.float32)
        x = tl.where(m, x - mu[None, :], 0.0)
    x = x * scale_mul
    amax = tl.max(tl.max(tl.abs(x), axis=1), axis=0)
    scale = amax / 127.0
    inv = tl.where(amax > 0, 127.0 / amax, 0.0)
    xq = x * inv
    xq += 0.5 * tl.where(xq >= 0, 1.0, -1.0)
    xq = tl.minimum(tl.maximum(xq, -127.0), 127.0)
    tl.store(o_base + h * stride_oh + offs_n[:, None] * stride_on + offs_d[None, :], xq.to(tl.int8), mask=nmask[:, None])
    tl.store(Scale + b * stride_sb + h * stride_sh + sblk + blk, scale)


def _seq_strides(x: torch.Tensor, tensor_layout: str):
    """(b, h, n, stride_b, stride_h, stride_n) of a 4-D NHD/HND tensor."""
    if tensor_layout == "HND":
        b, h, n, _ = x.shape
        return b, h, n, x.stride(0), x.stride(1), x.stride(2)
    if tensor_layout == "NHD":
        b, n, h, _ = x.shape
        return b, h, n, x.stride(0), x.stride(2), x.stride(1)
    raise ValueError(f"Unknown tensor layout: {tensor_layout}")


def _quant_one(x, mean, sm_scale, blk, tensor_layout, cu_seqlens=None, max_seqlen=None):
    d = x.shape[-1]
    dp = padded_head_dim(d)
    assert x.stride(-1) == 1, "last dim must be contiguous"
    if cu_seqlens is None:
        b, h, n, sxb, sxh, sxn = _seq_strides(x, tensor_layout)
        out = torch.empty((b, h, n, dp), dtype=torch.int8, device=x.device)
        nblk = triton.cdiv(n, blk)
        scale = torch.empty((b, h, nblk), dtype=torch.float32, device=x.device)
        cu, cu_s = scale, scale  # unused
        grid = (nblk, h, b)
        sob, soh, son = out.stride(0), out.stride(1), out.stride(2)
        ssb, ssh = scale.stride(0), scale.stride(1)
        length = n
    else:
        total, h, _ = x.shape
        b = cu_seqlens.numel() - 1
        sxb, sxh, sxn = 0, x.stride(1), x.stride(0)
        out = torch.empty((h, total, dp), dtype=torch.int8, device=x.device)
        lens = cu_seqlens[1:] - cu_seqlens[:-1]
        cu_s = torch.nn.functional.pad(torch.cumsum((lens + blk - 1) // blk, 0), (1, 0)).to(torch.int32)
        # host sync for the scale-array size; upstream's varlen path does the same
        nblk_total = int(cu_s[-1].item())
        scale = torch.empty((h, nblk_total), dtype=torch.float32, device=x.device)
        cu = cu_seqlens
        grid = (triton.cdiv(max_seqlen, blk), h, b)
        sob, soh, son = 0, out.stride(0), out.stride(1)
        ssb, ssh = 0, scale.stride(0)
        length = 0
    if mean is not None:
        # varlen: one [1, H, D] mean shared by every sequence
        smb, smh = (mean.stride(0) if cu_seqlens is None else 0), mean.stride(1)
    else:
        smb, smh = 0, 0
    _quant_int8_kernel[grid](
        x,
        mean if mean is not None else x,
        out,
        scale,
        cu,
        cu_s,
        sxb,
        sxh,
        sxn,
        smb,
        smh,
        sob,
        soh,
        son,
        ssb,
        ssh,
        length,
        sm_scale,
        D=d,
        DP=dp,
        BLK=blk,
        HAS_MEAN=mean is not None,
        VARLEN=cu_seqlens is not None,
        num_warps=4 if blk * dp >= 64 * 128 else 2,
    )
    return out, scale, (cu_s if cu_seqlens is not None else None)


def quant_qk_int8(
    q: torch.Tensor,
    k: torch.Tensor,
    km: torch.Tensor | None,
    sm_scale: float,
    BLKQ: int = 128,
    BLKK: int = 64,
    tensor_layout: str = "HND",
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
):
    """
    Per-block INT8 quantization of Q (pre-multiplied by ``sm_scale * log2(e)``)
    and of K (minus ``km`` if given).

    Dense: ``q``/``k`` are 4-D in ``tensor_layout``; ``km`` is ``[B, Hk, D]``
    (any strides). Returns ``q_int8 [B, Hq, Nq, Dp]``, ``q_scale [B, Hq,
    ceil(Nq/BLKQ)]``, ``k_int8 [B, Hk, Nk, Dp]``, ``k_scale [B, Hk,
    ceil(Nk/BLKK)]`` (float32).

    Varlen (``cu_seqlens_*`` given): ``q``/``k`` are ``[total, H, D]``,
    ``km`` is ``[1, Hk, D]``. Returns ``q_int8 [Hq, total_q, Dp]``,
    ``q_scale [Hq, nblk_q]``, ``k_int8``, ``k_scale`` likewise, plus the
    per-sequence block offsets ``cu_q_scale``, ``cu_k_scale``.
    """
    LOG2E = 1.4426950408889634
    if cu_seqlens_q is None:
        qi, qs, _ = _quant_one(q, None, sm_scale * LOG2E, BLKQ, tensor_layout)
        ki, ks, _ = _quant_one(k, km, 1.0, BLKK, tensor_layout)
        return qi, qs, ki, ks
    qi, qs, cqs = _quant_one(q, None, sm_scale * LOG2E, BLKQ, "NHD", cu_seqlens_q, max_seqlen_q)
    ki, ks, cks = _quant_one(k, km, 1.0, BLKK, "NHD", cu_seqlens_k, max_seqlen_k)
    return qi, qs, ki, ks, cqs, cks


@triton.jit
def _quant_fp8_kernel(
    X,
    Scale,
    Out,
    stride_xb,
    stride_xh,
    stride_xn,
    stride_sb,
    stride_sh,
    stride_ob,
    stride_oh,
    stride_on,
    L,
    D: tl.constexpr,
    DP: tl.constexpr,
    BLK: tl.constexpr,
):
    blk = tl.program_id(0)
    h = tl.program_id(1).to(tl.int64)
    b = tl.program_id(2).to(tl.int64)
    offs_n = blk * BLK + tl.arange(0, BLK)
    offs_d = tl.arange(0, DP)
    m = (offs_n < L)[:, None] & (offs_d < D)[None, :]
    x = tl.load(X + b * stride_xb + h * stride_xh + offs_n[:, None] * stride_xn + offs_d[None, :], mask=m, other=0.0)
    s = tl.load(Scale + b * stride_sb + h * stride_sh + offs_d, mask=offs_d < D, other=1.0)
    y = x.to(tl.float32) / s[None, :]
    tl.store(
        Out + b * stride_ob + h * stride_oh + offs_n[:, None] * stride_on + offs_d[None, :],
        y.to(Out.dtype.element_ty),
        mask=(offs_n < L)[:, None],
    )


def quant_v_fp8(v: torch.Tensor, tensor_layout: str, fp8_dtype: torch.dtype, varlen: bool = False):
    """
    Per-channel (per batch, head and head-dim channel, over the sequence) FP8
    quantization of V. Returns ``v_fp8 [B, Hk, N, Dp]`` (``[1, Hk, total,
    Dp]`` for varlen, zero-padded) and ``v_scale [B, Hk, Dp]`` (float32;
    1 on the padding).
    """
    fmax = torch.finfo(fp8_dtype).max
    if varlen:
        v4 = v.unsqueeze(0)
        layout = "NHD"
    else:
        v4 = v
        layout = tensor_layout
    b, h, n, sxb, sxh, sxn = _seq_strides(v4, layout)
    d = v4.shape[-1]
    dp = padded_head_dim(d)
    seq_dim = 2 if layout == "HND" else 1
    amax = v4.abs().amax(dim=seq_dim).to(torch.float32)  # [B, H, D]
    scale = torch.ones((b, h, dp), dtype=torch.float32, device=v.device)
    scale[..., :d] = torch.clamp(amax, min=1e-8) / fmax
    out = torch.empty((b, h, n, dp), dtype=fp8_dtype, device=v.device)
    BLK = 64
    grid = (triton.cdiv(n, BLK), h, b)
    _quant_fp8_kernel[grid](
        v4,
        scale,
        out,
        sxb,
        sxh,
        sxn,
        scale.stride(0),
        scale.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        n,
        D=d,
        DP=dp,
        BLK=BLK,
        num_warps=4,
    )
    return out, scale
