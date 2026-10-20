"""
SageAttention forward kernel for AMD GPUs (Triton): INT8 Q·K^T, FP16 (or FP8
on CDNA3) P·V, FP32 online softmax in the exp2 domain.

Inputs come from :mod:`.quant`: ``q_int8``/``k_int8`` head-major with a
power-of-two head dim, Q pre-scaled by ``sm_scale * log2(e)``; per-block
scales ``q_scale`` (granularity ``QBLK`` rows) and ``k_scale`` (``KBLK``
rows, a multiple of the kernel's ``BLOCK_N``). V and the output are read and
written in the caller's layout (NHD or HND) through strides, so no transposes
or padding copies are made.

Per tile the softmax costs one int32 row max, one int->float conversion and
one FMA per score before ``exp2``: the dequantization scale ``q_scale[row] *
k_scale`` is positive, so ``max_j(s * qk_ij) = s * max_j(qk_ij)`` is taken on
the integer scores and folded into the exponent's FMA. Tiles that need no
masking (all but the causal diagonal and the ragged kv tail) skip all mask
arithmetic.
"""

import torch
import triton
import triton.language as tl

__all__ = ["sage_attn_fwd"]


@triton.jit
def _inner(
    acc,
    l_i,
    m_i,
    q,
    qs,
    k_base,
    v_base,
    ks_base,
    mask_base,
    lo,
    hi,
    offs_m,
    offs_n,
    offs_d,
    kv_len,
    stride_kn,
    stride_vn,
    stride_mn,
    D_V: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    KBLK: tl.constexpr,
    MASKED: tl.constexpr,
    CAUSAL: tl.constexpr,
    HAS_MASK: tl.constexpr,
    MASK_BOOL: tl.constexpr,
    P_SHIFT: tl.constexpr,
):
    for start_n in tl.range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        cols = start_n + offs_n
        if MASKED:
            kmask = cols < kv_len
            k = tl.load(k_base + cols[None, :] * stride_kn + offs_d[:, None], mask=kmask[None, :], other=0)
        else:
            k = tl.load(k_base + cols[None, :] * stride_kn + offs_d[:, None])
        ks = tl.load(ks_base + start_n // KBLK)
        s = qs * ks  # [BLOCK_M] dequantization scale, > 0
        qk = tl.dot(q, k)  # int32
        if MASKED:
            qkf = qk.to(tl.float32) * s[:, None]
            valid = kmask[None, :]
            if CAUSAL:
                valid = valid & (offs_m[:, None] >= cols[None, :])
            if HAS_MASK:
                mk = tl.load(
                    mask_base + cols[None, :] * stride_mn,
                    mask=kmask[None, :],
                    other=0,
                )
                if MASK_BOOL:
                    qkf = tl.where(mk != 0, qkf, -1.0e6)
                else:
                    # additive mask in natural-log units (torch SDPA semantics); scores are in log2 units
                    qkf = qkf + mk.to(tl.float32) * 1.4426950408889634
            qkf = tl.where(valid, qkf, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(qkf, 1) - P_SHIFT)
            # a row with nothing valid yet (only possible with an all-masked first tile) must not give inf - inf
            m_new = tl.where(m_new == float("-inf"), 0.0, m_new)
            p = tl.math.exp2(qkf - m_new[:, None])
        else:
            m_new = tl.maximum(m_i, tl.max(qk, 1).to(tl.float32) * s - P_SHIFT)
            p = tl.math.exp2(qk.to(tl.float32) * s[:, None] - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        if MASKED:
            if D_V == HEAD_DIM:
                v = tl.load(v_base + cols[:, None] * stride_vn + offs_d[None, :], mask=kmask[:, None], other=0.0)
            else:
                v = tl.load(
                    v_base + cols[:, None] * stride_vn + offs_d[None, :],
                    mask=kmask[:, None] & (offs_d < D_V)[None, :],
                    other=0.0,
                )
        else:
            if D_V == HEAD_DIM:
                v = tl.load(v_base + cols[:, None] * stride_vn + offs_d[None, :])
            else:
                v = tl.load(v_base + cols[:, None] * stride_vn + offs_d[None, :], mask=(offs_d < D_V)[None, :], other=0.0)
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_new
    return acc, l_i, m_i


@triton.jit
def _sage_fwd(
    Q,
    K,
    V,
    Qs,
    Ks,
    Vs,
    Out,
    Lse,
    Mask,
    CuQ,
    CuK,
    CuQs,
    CuKs,
    stride_qb,
    stride_qh,
    stride_qn,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_ob,
    stride_oh,
    stride_on,
    stride_qsb,
    stride_qsh,
    stride_ksb,
    stride_ksh,
    stride_vsb,
    stride_vsh,
    stride_lb,
    stride_lh,
    stride_mb,
    stride_mh,
    stride_mm,
    stride_mn,
    qo_len,
    kv_len,
    num_kv_groups,
    HEAD_DIM: tl.constexpr,
    D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QBLK: tl.constexpr,
    KBLK: tl.constexpr,
    CAUSAL: tl.constexpr,
    HAS_MASK: tl.constexpr,
    MASK_BOOL: tl.constexpr,
    PV_FP8: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    VARLEN: tl.constexpr,
    REVERSE_M: tl.constexpr,
    P_SHIFT: tl.constexpr,
):
    if REVERSE_M:
        start_m = tl.num_programs(0) - 1 - tl.program_id(0)
    else:
        start_m = tl.program_id(0)
    off_h = tl.program_id(1).to(tl.int64)
    off_b = tl.program_id(2).to(tl.int64)
    off_hk = off_h // num_kv_groups

    if VARLEN:
        q_start = tl.load(CuQ + off_b).to(tl.int64)
        qo_len = (tl.load(CuQ + off_b + 1) - q_start).to(tl.int32)
        if start_m * BLOCK_M >= qo_len:
            return
        k_start = tl.load(CuK + off_b).to(tl.int64)
        kv_len = (tl.load(CuK + off_b + 1) - k_start).to(tl.int32)
        qs_start = tl.load(CuQs + off_b).to(tl.int64)
        ks_start = tl.load(CuKs + off_b).to(tl.int64)
        q_base = Q + off_h * stride_qh + q_start * stride_qn
        k_base = K + off_hk * stride_kh + k_start * stride_kn
        v_base = V + off_hk * stride_vh + k_start * stride_vn
        o_base = Out + off_h * stride_oh + q_start * stride_on
        qs_base = Qs + off_h * stride_qsh + qs_start
        ks_base = Ks + off_hk * stride_ksh + ks_start
    else:
        q_base = Q + off_b * stride_qb + off_h * stride_qh
        k_base = K + off_b * stride_kb + off_hk * stride_kh
        v_base = V + off_b * stride_vb + off_hk * stride_vh
        o_base = Out + off_b * stride_ob + off_h * stride_oh
        qs_base = Qs + off_b * stride_qsb + off_h * stride_qsh
        ks_base = Ks + off_b * stride_ksb + off_hk * stride_ksh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    mrow = offs_m < qo_len

    q = tl.load(q_base + offs_m[:, None] * stride_qn + offs_d[None, :], mask=mrow[:, None], other=0)
    qs = tl.load(qs_base + offs_m // QBLK, mask=mrow, other=0.0)

    if HAS_MASK:
        mask_base = Mask + off_b * stride_mb + off_h * stride_mh + offs_m[:, None] * stride_mm
    else:
        mask_base = Mask

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    # FP8 P·V: P in [0, 1] is scaled by 2^P_SHIFT (m is tracked minus P_SHIFT) before the FP8 cast, keeping
    # small probabilities out of the FP8 subnormal range; l carries the same factor, so it cancels in acc / l

    if CAUSAL:
        hi = tl.minimum(kv_len, (start_m + 1) * BLOCK_M)
        full_hi = tl.minimum((start_m * BLOCK_M + 1) // BLOCK_N * BLOCK_N, kv_len // BLOCK_N * BLOCK_N)
    else:
        hi = kv_len
        full_hi = kv_len // BLOCK_N * BLOCK_N
    if HAS_MASK:
        full_hi = 0

    acc, l_i, m_i = _inner(
        acc, l_i, m_i, q, qs, k_base, v_base, ks_base, mask_base, 0, full_hi,
        offs_m, offs_n, offs_d, kv_len, stride_kn, stride_vn, stride_mn,
        D_V, HEAD_DIM, BLOCK_N, KBLK, False, CAUSAL, HAS_MASK, MASK_BOOL, P_SHIFT,
    )  # fmt: skip
    acc, l_i, m_i = _inner(
        acc, l_i, m_i, q, qs, k_base, v_base, ks_base, mask_base, full_hi, hi,
        offs_m, offs_n, offs_d, kv_len, stride_kn, stride_vn, stride_mn,
        D_V, HEAD_DIM, BLOCK_N, KBLK, True, CAUSAL, HAS_MASK, MASK_BOOL, P_SHIFT,
    )  # fmt: skip

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    o = acc / l_safe[:, None]
    if PV_FP8:
        if VARLEN:
            vs = tl.load(Vs + off_hk * stride_vsh + offs_d)
        else:
            vs = tl.load(Vs + off_b * stride_vsb + off_hk * stride_vsh + offs_d)
        o = o * vs[None, :]
    if D_V == HEAD_DIM:
        omask = mrow[:, None]
    else:
        omask = mrow[:, None] & (offs_d < D_V)[None, :]
    tl.store(o_base + offs_m[:, None] * stride_on + offs_d[None, :], o.to(Out.dtype.element_ty), mask=omask)

    if RETURN_LSE:
        # m_i is tracked minus P_SHIFT and l carries 2^P_SHIFT: the two cancel
        lse = m_i + tl.log2(l_safe)
        if VARLEN:
            lptr = Lse + off_h * stride_lh + q_start + offs_m
        else:
            lptr = Lse + off_b * stride_lb + off_h * stride_lh + offs_m
        tl.store(lptr, lse, mask=mrow)


def _strides_bhn(x: torch.Tensor, tensor_layout: str):
    if tensor_layout == "HND":
        return x.stride(0), x.stride(1), x.stride(2)
    return x.stride(0), x.stride(2), x.stride(1)


def sage_attn_fwd(
    q_int8: torch.Tensor,
    k_int8: torch.Tensor,
    v: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    out: torch.Tensor,
    *,
    QBLK: int,
    KBLK: int,
    config: dict,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    attn_mask: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    return_lse: bool = False,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    cu_q_scale: torch.Tensor | None = None,
    cu_k_scale: torch.Tensor | None = None,
    max_seqlen_q: int | None = None,
) -> torch.Tensor | None:
    """
    Launch the kernel. ``q_int8``/``k_int8``/``q_scale``/``k_scale`` as
    returned by :func:`.quant.quant_qk_int8`. ``v`` is fp16 in the caller's
    layout (``tensor_layout``; ``[total, Hk, D]`` for varlen), or, when
    ``v_scale`` is given, FP8 ``[B, Hk, N, Dp]`` from :func:`.quant.quant_v_fp8`.
    ``out`` (``q``'s logical shape and layout, real head dim) is written in
    place. ``config``: ``BLOCK_M``, ``BLOCK_N``, ``num_warps``, ``num_stages``
    and optional AMD knobs (``waves_per_eu``, ``matrix_instr_nonkdim``,
    ``kpack``). Returns the log2-domain LSE ``[B, Hq, Nq]`` (``[Hq, total]``
    varlen) when ``return_lse``.
    """
    varlen = cu_seqlens_q is not None
    BLOCK_M, BLOCK_N = config["BLOCK_M"], config["BLOCK_N"]
    assert KBLK % BLOCK_N == 0, "k-scale blocks must cover whole kv tiles"
    dp = q_int8.shape[-1]
    d_v = out.shape[-1]
    pv_fp8 = v_scale is not None

    if varlen:
        hq, total_q, _ = q_int8.shape
        hk = k_int8.shape[0]
        b = cu_seqlens_q.numel() - 1
        sq = (0, q_int8.stride(0), q_int8.stride(1))
        sk = (0, k_int8.stride(0), k_int8.stride(1))
        if pv_fp8:
            sv = (0, v.stride(1), v.stride(2))
        else:
            sv = (0, v.stride(1), v.stride(0))
        so = (0, out.stride(1), out.stride(0))
        sqs = (0, q_scale.stride(0))
        sks = (0, k_scale.stride(0))
        qo_len = kv_len = 0
        n_m = triton.cdiv(max_seqlen_q, BLOCK_M)
        lse = torch.empty((hq, total_q), dtype=torch.float32, device=q_int8.device) if return_lse else None
        sl = (0, lse.stride(0)) if return_lse else (0, 0)
    else:
        b, hq, qo_len, _ = q_int8.shape
        hk, kv_len = k_int8.shape[1], k_int8.shape[2]
        sq = q_int8.stride()[:3]
        sk = k_int8.stride()[:3]
        sv = v.stride()[:3] if pv_fp8 else _strides_bhn(v, tensor_layout)
        so = _strides_bhn(out, tensor_layout)
        sqs = q_scale.stride()[:2]
        sks = k_scale.stride()[:2]
        n_m = triton.cdiv(qo_len, BLOCK_M)
        lse = torch.empty((b, hq, qo_len), dtype=torch.float32, device=q_int8.device) if return_lse else None
        sl = lse.stride()[:2] if return_lse else (0, 0)
    svs = v_scale.stride()[:2] if pv_fp8 else (0, 0)

    if attn_mask is not None:
        sm = attn_mask.stride()
        mask_bool = attn_mask.dtype == torch.bool
        if mask_bool:
            attn_mask = attn_mask.view(torch.uint8)
    else:
        sm = (0, 0, 0, 0)
        mask_bool = False

    dummy = q_scale
    extra = {}
    for kname in ("waves_per_eu", "matrix_instr_nonkdim", "kpack", "schedule_hint"):
        if kname in config:
            extra[kname] = config[kname]
    grid = (n_m, hq, b)
    _sage_fwd[grid](
        q_int8,
        k_int8,
        v,
        q_scale,
        k_scale,
        v_scale if pv_fp8 else dummy,
        out,
        lse if return_lse else dummy,
        attn_mask if attn_mask is not None else dummy,
        cu_seqlens_q if varlen else dummy,
        cu_seqlens_k if varlen else dummy,
        cu_q_scale if varlen else dummy,
        cu_k_scale if varlen else dummy,
        *sq,
        *sk,
        *sv,
        *so,
        *sqs,
        *sks,
        *svs,
        *sl,
        *sm,
        qo_len,
        kv_len,
        hq // hk,
        HEAD_DIM=dp,
        D_V=d_v,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        QBLK=QBLK,
        KBLK=KBLK,
        CAUSAL=is_causal,
        HAS_MASK=attn_mask is not None,
        MASK_BOOL=mask_bool,
        PV_FP8=pv_fp8,
        RETURN_LSE=return_lse,
        VARLEN=varlen,
        REVERSE_M=is_causal,
        P_SHIFT=config.get("P_SHIFT", 7.0) if pv_fp8 else 0.0,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
        **extra,
    )
    return lse
