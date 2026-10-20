"""
Accuracy and API tests for the ROCm backend.

Methodology follows the SageAttention papers: cosine similarity and relative
L1 (sum|o - ref| / sum|ref|) of the output against FP32 attention
(``torch.nn.functional.scaled_dot_product_attention`` on fp32 copies), on
inputs with the outlier structure of real diffusion/LLM activations: K with a
large per-channel bias shared by all tokens (what ``smooth_k`` removes) and a
few heavy Q/K channels.

    pytest tests/rocm -q
"""

import math
import warnings

import pytest
import torch
import torch.nn.functional as F

import sageattention
from sageattention.rocm import api as rapi
from sageattention.rocm.arch import has_fp8, is_rocm

pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and is_rocm()), reason="needs a ROCm GPU")

FP8 = torch.cuda.is_available() and is_rocm() and has_fp8()


def realistic_qkv(B, Hq, Hk, Nq, Nk, D, dtype, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def r(*s):
        return torch.randn(*s, device="cuda", generator=g, dtype=torch.float32)

    q = r(B, Hq, Nq, D)
    k = r(B, Hk, Nk, D)
    v = r(B, Hk, Nk, D)
    # channel-wise K bias (shared by all tokens) and outlier channels
    k = k + 4.0 * r(B, Hk, 1, D)
    ch = torch.randperm(D, device="cuda", generator=g)[: max(1, D // 16)]
    q[..., ch] *= 6.0
    k[..., ch] *= 6.0
    # V with a per-channel offset (CogVideoX-like)
    v = v + 2.0 * r(B, Hk, 1, D)
    return q.to(dtype), k.to(dtype), v.to(dtype)


def ref_attn(q, k, v, is_causal=False, attn_mask=None, sm_scale=None):
    """FP32 reference in HND; GQA by repeating K/V heads."""
    hq, hk = q.shape[1], k.shape[1]
    if hq != hk:
        k = k.repeat_interleave(hq // hk, dim=1)
        v = v.repeat_interleave(hq // hk, dim=1)
    if attn_mask is not None and attn_mask.dtype != torch.bool:
        attn_mask = attn_mask.float()
    return F.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), is_causal=is_causal, attn_mask=attn_mask, scale=sm_scale
    )


def metrics(o, ref):
    o, ref = o.double().flatten(), ref.double().flatten()
    cos = F.cosine_similarity(o, ref, dim=0).item()
    l1 = ((o - ref).abs().sum() / ref.abs().sum()).item()
    rmse = torch.sqrt(((o - ref) ** 2).mean()).item()
    return cos, l1, rmse


# thresholds: INT8 QK + FP16 PV is ~1e-2 rel-L1 on these inputs; FP8 PV ~4e-2 (3-bit-mantissa P)
TOL = {"fp16": (0.9995, 0.03), "fp8": (0.998, 0.08)}


def to_layout(x, layout):
    return x.transpose(1, 2).contiguous() if layout == "NHD" else x


def from_layout(x, layout):
    return x.transpose(1, 2) if layout == "NHD" else x


PV = ["fp16"] + (["fp8"] if FP8 else [])


@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("layout", ["HND", "NHD"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("pv", PV)
def test_accuracy_main(D, causal, layout, dtype, pv):
    q, k, v = realistic_qkv(2, 6, 6, 1037, 1037, D, dtype)
    ref = ref_attn(q, k, v, is_causal=causal)
    o = rapi.sageattn_rocm(
        to_layout(q, layout), to_layout(k, layout), to_layout(v, layout), tensor_layout=layout, is_causal=causal, pv_dtype=pv
    )
    assert o.dtype == dtype and o.shape == to_layout(q, layout).shape
    cos, l1, _ = metrics(from_layout(o, layout), ref)
    assert cos > TOL[pv][0] and l1 < TOL[pv][1], (cos, l1)


@pytest.mark.parametrize("D", [32, 72, 96, 256])
def test_odd_head_dims(D):
    q, k, v = realistic_qkv(1, 4, 4, 515, 515, D, torch.float16)
    ref = ref_attn(q, k, v)
    o = rapi.sageattn_rocm(q, k, v)
    cos, l1, _ = metrics(o, ref)
    assert cos > 0.9995 and l1 < 0.03, (cos, l1)


@pytest.mark.parametrize("gran", ["per_block", "per_warp", "per_thread"])
def test_quant_granularity(gran):
    q, k, v = realistic_qkv(1, 4, 4, 2048, 2048, 128, torch.bfloat16)
    ref = ref_attn(q, k, v)
    cos, l1, _ = metrics(rapi.sageattn_rocm(q, k, v, qk_quant_gran=gran), ref)
    assert cos > 0.9995 and l1 < 0.03, (cos, l1)


def test_smooth_k_matters():
    """Without smoothing, the channel bias in K eats INT8 range; with it, error drops (upstream's claim)."""
    q, k, v = realistic_qkv(1, 4, 4, 2048, 2048, 128, torch.float16)
    k = k + 20.0 * torch.randn(1, 4, 1, 128, device="cuda", dtype=torch.float16)
    ref = ref_attn(q, k, v)
    _, l1_s, _ = metrics(rapi.sageattn_rocm(q, k, v, smooth_k=True), ref)
    _, l1_n, _ = metrics(rapi.sageattn_rocm(q, k, v, smooth_k=False), ref)
    assert l1_s < 0.03 and l1_s < l1_n * 0.7, (l1_s, l1_n)


@pytest.mark.parametrize("Nq,Nk", [(1, 333), (77, 4096), (4096, 77), (300, 1)])
def test_cross_attention_lengths(Nq, Nk):
    q, k, v = realistic_qkv(2, 4, 4, Nq, Nk, 128, torch.float16)
    ref = ref_attn(q, k, v)
    cos, l1, _ = metrics(rapi.sageattn_rocm(q, k, v), ref)
    assert cos > 0.9995 and l1 < 0.03, (cos, l1)


@pytest.mark.parametrize("causal", [False, True])
def test_gqa(causal):
    q, k, v = realistic_qkv(2, 8, 2, 1000, 1000, 128, torch.bfloat16)
    ref = ref_attn(q, k, v, is_causal=causal)
    cos, l1, _ = metrics(rapi.sageattn_rocm(q, k, v, is_causal=causal), ref)
    assert cos > 0.9995 and l1 < 0.03, (cos, l1)


@pytest.mark.parametrize("pv", PV)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("layout", ["HND", "NHD"])
@pytest.mark.parametrize("inputs", ["randn", "outliers"])
def test_return_lse(pv, causal, layout, inputs):
    N = 700
    if inputs == "randn":
        g = torch.Generator(device="cuda").manual_seed(0)
        q, k, v = (torch.randn(2, 4, N, 64, device="cuda", generator=g).half() for _ in range(3))
        k = k + 1.0  # a channel bias for smooth_k (and the LSE correction) to act on
    else:
        q, k, v = realistic_qkv(2, 4, 4, N, N, 64, torch.float16)
    sm = 64**-0.5
    s = torch.einsum("bhnd,bhmd->bhnm", q.float(), k.float()) * sm
    if causal:
        s = s.masked_fill(torch.ones(N, N, device="cuda", dtype=torch.bool).triu(1), float("-inf"))
    lse_ref = torch.logsumexp(s, dim=-1)
    # without smooth_k the K channel bias of the outlier inputs costs INT8 range (see test_smooth_k_matters)
    for smooth in (True, False) if inputs == "randn" else (True,):
        o, lse = rapi.sageattn_rocm(
            to_layout(q, layout), to_layout(k, layout), to_layout(v, layout),
            tensor_layout=layout, is_causal=causal, return_lse=True, smooth_k=smooth, pv_dtype=pv,
        )  # fmt: skip
        assert lse.shape == (2, 4, N) and lse.dtype == torch.float32
        err = (lse - lse_ref).abs()
        if inputs == "randn":
            assert err.max().item() < 0.05, (smooth, err.max().item())
        else:
            # logits reach ~100 here; INT8 Q/K error is proportional to their magnitude
            scale = s.masked_fill(s.isinf(), 0).abs().amax(-1)
            # (same error as a torch simulation of the per-block INT8 quantization: scratch/lse_diag.py)
            assert (err / scale).max().item() < 0.05 and (err / scale).mean().item() < 0.01, (smooth, err.max().item())
        cos, l1, _ = metrics(from_layout(o, layout), ref_attn(q, k, v, is_causal=causal))
        assert cos > TOL[pv][0] and l1 < TOL[pv][1]


@pytest.mark.parametrize("kind", ["bool", "float"])
def test_attn_mask(kind):
    B, H, N, D = 2, 4, 600, 128
    q, k, v = realistic_qkv(B, H, H, N, N, D, torch.float16)
    g = torch.Generator(device="cuda").manual_seed(1)
    if kind == "bool":
        mask = torch.rand(B, 1, N, N, device="cuda", generator=g) > 0.3
        mask[..., 0] = True  # no fully masked rows
    else:
        mask = (torch.randn(B, 1, N, N, device="cuda", generator=g) * 2).to(torch.float16)
    ref = ref_attn(q, k, v, attn_mask=mask)
    o = sageattention.sageattn_qk_int8_pv_fp16_triton(q, k, v, attn_mask=mask)
    cos, l1, _ = metrics(o, ref)
    assert cos > 0.9995 and l1 < 0.03, (cos, l1)


@pytest.mark.parametrize("causal", [False, True])
def test_varlen(causal):
    H, D = 4, 128
    lens = [700, 1, 129, 2048]
    tot = sum(lens)
    q, k, v = realistic_qkv(1, H, H, tot, tot, D, torch.float16)
    qn, kn, vn = (x[0].transpose(0, 1).contiguous() for x in (q, k, v))  # [total, H, D]
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), device="cuda", dtype=torch.int32)
    o = sageattention.sageattn_varlen(qn, kn, vn, cu, cu, max(lens), max(lens), is_causal=causal)
    outs, refs = [], []
    for i, L in enumerate(lens):
        s, e = cu[i].item(), cu[i + 1].item()
        refs.append(ref_attn(q[:, :, s:e], k[:, :, s:e], v[:, :, s:e], is_causal=causal)[0].transpose(0, 1))
        outs.append(o[s:e])
    cos, l1, _ = metrics(torch.cat(outs), torch.cat(refs))
    assert cos > 0.9995 and l1 < 0.03, (cos, l1)


def test_public_api_dispatch():
    q, k, v = realistic_qkv(1, 4, 4, 512, 512, 128, torch.bfloat16)
    ref = ref_attn(q, k, v)
    for fn in (
        sageattention.sageattn,
        sageattention.sageattn_qk_int8_pv_fp16_triton,
        sageattention.sageattn_qk_int8_pv_fp16_cuda,
        sageattention.sageattn_qk_int8_pv_fp8_cuda,
        sageattention.sageattn_qk_int8_pv_fp8_cuda_sm90,
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o = fn(q, k, v, tensor_layout="HND", is_causal=False)
        cos, l1, _ = metrics(o, ref)
        assert cos > 0.998 and l1 < 0.08, (fn.__name__, cos, l1)
    # ComfyUI-style call: NHD, keyword args
    qn, kn, vn = (x.transpose(1, 2).contiguous() for x in (q, k, v))
    o = sageattention.sageattn(qn, kn, vn, is_causal=False, tensor_layout="NHD")
    cos, l1, _ = metrics(o.transpose(1, 2), ref)
    assert cos > 0.9995


def test_noncontiguous_views():
    """Strided views (e.g. q/k/v split out of a fused QKV projection) are read in place."""
    B, N, H, D = 2, 777, 6, 128
    qkv = torch.randn(B, N, 3, H, D, device="cuda", dtype=torch.float16)
    q, k, v = qkv.unbind(2)  # NHD views, non-contiguous
    ref = ref_attn(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    o = sageattention.sageattn(q, k, v, tensor_layout="NHD")
    cos, l1, _ = metrics(o.transpose(1, 2), ref)
    assert cos > 0.9995 and l1 < 0.03


def test_fp8_fallback_on_rdna3():
    if FP8:
        pytest.skip("GPU has FP8")
    q, k, v = realistic_qkv(1, 2, 2, 256, 256, 64, torch.float16)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        o = rapi.sageattn_rocm(q, k, v, pv_dtype="fp8")
    cos, _, _ = metrics(o, ref_attn(q, k, v))
    assert cos > 0.9995


def test_sm_scale():
    q, k, v = realistic_qkv(1, 2, 2, 512, 512, 64, torch.float16)
    sm = 0.3 / math.sqrt(64)
    ref = ref_attn(q, k, v, sm_scale=sm)
    cos, l1, _ = metrics(rapi.sageattn_rocm(q, k, v, sm_scale=sm), ref)
    assert cos > 0.9995 and l1 < 0.03


# ---------------------------------------------------------------------------------------------------------------
# native gfx11 kernel (sageattention/rocm/native/sage_gfx11.hip)

GFX11 = torch.cuda.is_available() and is_rocm() and torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx11")
native = pytest.mark.skipif(not GFX11, reason="native kernel is gfx11-only")


@native
@pytest.mark.parametrize("D", [64, 128, 72, 96])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("layout", ["HND", "NHD"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("waves", [4, 8])
def test_native_accuracy(D, causal, layout, dtype, waves):
    from sageattention.rocm.native.sage import sageattn_native

    q, k, v = realistic_qkv(2, 6, 3, 1037, 1037, D, dtype)  # GQA 2:1, ragged length
    ref = ref_attn(q, k, v, is_causal=causal)
    o = sageattn_native(to_layout(q, layout), to_layout(k, layout), to_layout(v, layout), tensor_layout=layout,
                        is_causal=causal, waves=waves)  # fmt: skip
    o_t = rapi.sageattn_rocm(to_layout(q, layout), to_layout(k, layout), to_layout(v, layout), tensor_layout=layout,
                             is_causal=causal, _native=False)  # fmt: skip
    cos, l1, _ = metrics(from_layout(o, layout), ref)
    cos_t, l1_t, _ = metrics(from_layout(o_t, layout), ref)
    assert cos > 0.9995 and l1 < 0.03, (cos, l1)
    assert l1 < l1_t * 1.1 + 1e-3, (l1, l1_t)  # as accurate as the Triton kernel (same quantization)


@native
@pytest.mark.parametrize("Nq,Nk", [(1, 333), (77, 4096), (4096, 77), (300, 1)])
def test_native_lengths(Nq, Nk):
    from sageattention.rocm.native.sage import sageattn_native

    q, k, v = realistic_qkv(1, 4, 4, Nq, Nk, 128, torch.float16)
    cos, l1, _ = metrics(sageattn_native(q, k, v), ref_attn(q, k, v))
    assert cos > 0.9995 and l1 < 0.03, (cos, l1)


@native
def test_native_lse_and_dispatch():
    from sageattention.rocm.native import sage as ns

    q, k, v = realistic_qkv(2, 4, 4, 700, 700, 64, torch.float16)
    o_n, lse_n = ns.sageattn_native(q, k, v, return_lse=True)
    o_t, lse_t = rapi.sageattn_rocm(q, k, v, return_lse=True, _native=False)
    assert (lse_n - lse_t).abs().max().item() < 0.05
    # the public API uses the native kernel on gfx11
    o = sageattention.sageattn(q, k, v)
    assert ns.status().get(0) == "ok"
    assert torch.equal(o, ns.sageattn_native(q, k, v))
