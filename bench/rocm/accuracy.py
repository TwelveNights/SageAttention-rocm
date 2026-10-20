"""
Accuracy of the ROCm SageAttention kernels against FP32 attention, SageAttention-paper style
(cosine similarity, relative L1 = sum|o - ref| / sum|ref|, RMSE).

    python bench/rocm/accuracy.py                       # synthetic inputs with outlier structure
    python bench/rocm/accuracy.py --qkv runs/wan_qkv.pt # real Q/K/V captured from a model (wan_e2e.py --dump-qkv)

Rows: sage (fp16 P.V), sage-fp8 (CDNA3), sage-native (gfx11), and as references upstream's own Triton
SageAttention kernel (sageattention/triton, run unmodified on ROCm) and an FP16/BF16 SDPA.
"""

import argparse
import os
import sys
import warnings

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tests", "rocm"))

from sageattention.rocm import api as rapi  # noqa: E402
from sageattention.rocm.arch import gcn_arch, has_fp8  # noqa: E402


def metrics(o, ref):
    o, ref = o.double().flatten(), ref.double().flatten()
    cos = F.cosine_similarity(o, ref, dim=0).item()
    l1 = ((o - ref).abs().sum() / ref.abs().sum()).item()
    rmse = torch.sqrt(((o - ref) ** 2).mean()).item()
    return cos, l1, rmse


def upstream_triton(q, k, v, is_causal):
    """Upstream's sageattn_qk_int8_pv_fp16_triton path (its Triton kernels), HND."""
    from sageattention.triton.attn_qk_int8_per_block import forward as attn_false
    from sageattention.triton.attn_qk_int8_per_block_causal import forward as attn_true
    from sageattention.triton.quant_per_block import per_block_int8

    km = k.mean(dim=2, keepdim=True)
    sm = q.size(-1) ** -0.5
    qi, qs, ki, ks = per_block_int8(q, k, km=km, sm_scale=sm, tensor_layout="HND")
    vv = v.to(torch.float16)
    fwd = attn_true if is_causal else attn_false
    o, _ = fwd(qi, ki, vv, qs, ks, tensor_layout="HND", output_dtype=q.dtype)
    return o


def methods():
    m = {
        "sage (fp16 PV, Triton)": lambda q, k, v, c: rapi.sageattn_rocm(q, k, v, is_causal=c, _native=False),
    }
    if has_fp8():
        m["sage-fp8 (fp8 PV)"] = lambda q, k, v, c: rapi.sageattn_rocm(q, k, v, is_causal=c, pv_dtype="fp8")
    if gcn_arch().startswith("gfx11"):
        from sageattention.rocm.native.sage import sageattn_native

        m["sage-native (gfx11)"] = lambda q, k, v, c: sageattn_native(q, k, v, is_causal=c)
    # upstream's causal Triton kernel aborts the ROCm Triton 3.6 compiler ("operation destroyed but still has
    # uses", LLVM ERROR) -- only its non-causal kernel is used as a reference here
    m["upstream Triton SageAttn"] = lambda q, k, v, c: upstream_triton(q, k, v, c) if not c else None
    m["SDPA in input dtype"] = lambda q, k, v, c: F.scaled_dot_product_attention(q, k, v, is_causal=c)
    return m


def run_case(name, q, k, v, causal, rows):
    ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=causal)
    for mname, fn in methods().items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                o = fn(q, k, v, causal)
            if o is None:
                continue
            cos, l1, rmse = metrics(o, ref)
            rows.append((name, mname, cos, l1, rmse))
        except Exception as e:  # noqa: BLE001
            rows.append((name, mname, float("nan"), float("nan"), float("nan")))
            print(f"  {mname}: {type(e).__name__}: {str(e)[:120]}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qkv", default=None)
    p.add_argument("--dtype", default="bf16")
    args = p.parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    print(f"GPU {torch.cuda.get_device_name()} ({gcn_arch()})")
    rows = []
    if args.qkv:
        data = torch.load(args.qkv)
        for item in data:
            q, k, v = (item[x].cuda() for x in ("q", "k", "v"))  # NHD from diffusers -> HND
            q, k, v = (x.transpose(1, 2).contiguous() for x in (q, k, v))
            run_case(item["name"], q, k, v, False, rows)
    else:
        from test_sageattn_rocm import realistic_qkv

        for D in (64, 128):
            for causal in (False, True):
                q, k, v = realistic_qkv(2, 8, 8, 4096, 4096, D, dtype)
                run_case(f"D{D} {'causal' if causal else 'full'}", q, k, v, causal, rows)
    print(f"{'case':<28} {'method':<28} {'cos sim':>10} {'rel L1':>8} {'RMSE':>9}")
    for r in rows:
        print(f"{r[0]:<28} {r[1]:<28} {r[2]:>10.6f} {r[3]:>8.4f} {r[4]:>9.2e}")


if __name__ == "__main__":
    main()
