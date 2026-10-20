"""
Benchmark SageAttention (ROCm backend) against FP16 flash attention baselines.

    python bench/rocm/bench_attn.py --preset video          # Wan / HunyuanVideo shapes
    python bench/rocm/bench_attn.py --preset image
    python bench/rocm/bench_attn.py --shape 1,24,4608,128 --sweep   # tile sweep (kernel only)

Reports ms and TFLOP/s (4*B*H*Nq*Nk*D, halved for causal) per method:
  sage-fp16 / sage-fp8 : full sageattn call (quantization + attention)
  sage-*-kern          : attention kernel only, pre-quantized inputs
  triton-fa            : nunchaku-rocm's FP16 Triton flash attention (autotuned)
  sdpa                 : torch.nn.functional.scaled_dot_product_attention
  aiter                : aiter flash_attn_func (if importable)
"""

import argparse
import itertools
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))

from sageattention.rocm import api as sapi  # noqa: E402
from sageattention.rocm.arch import arch_family, fp8_dtype, gcn_arch  # noqa: E402
from sageattention.rocm.attn import sage_attn_fwd  # noqa: E402
from sageattention.rocm.quant import quant_qk_int8, quant_v_fp8  # noqa: E402

PRESETS = {
    # (B, H, N, D)
    "image": [
        (1, 24, 4608, 128),  # FLUX.1 1024^2 (4096 img + 512 txt)
        (1, 24, 4352, 128),  # FLUX.1 1024^2 with 256 txt
        (2, 24, 4096, 64),   # SD3-ish d=64
        (1, 24, 16896, 128),  # FLUX 2048^2
    ],
    "video": [
        (1, 12, 32760, 128),  # Wan2.1-1.3B 480p 81f
        (1, 40, 32760, 128),  # Wan2.1-14B 480p 81f
        (1, 24, 16384, 128),
        (1, 24, 45000, 128),  # HunyuanVideo 544p ~65f
        (1, 40, 75600, 128),  # Wan2.1-14B 720p 81f
    ],
    "video_small": [
        (1, 12, 16380, 128),  # Wan2.1-1.3B 480p 41f
        (1, 12, 32760, 128),  # Wan2.1-1.3B 480p 81f
        (1, 24, 16384, 128),
    ],
}


def timeit(fn, warmup=3, reps=10, rounds=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(rounds):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(reps):
            fn()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / reps)
    return best


def reps_for(flops):
    # aim for ~0.3 s per round
    est_ms = flops / 50e12 * 1e3
    return max(2, min(50, int(300 / max(est_ms, 1e-3))))


def make_inputs(B, H, N, D, dtype, layout):
    torch.manual_seed(0)
    q = torch.randn(B, H, N, D, device="cuda", dtype=dtype)
    k = torch.randn(B, H, N, D, device="cuda", dtype=dtype)
    v = torch.randn(B, H, N, D, device="cuda", dtype=dtype)
    if layout == "NHD":
        q, k, v = (x.transpose(1, 2).contiguous() for x in (q, k, v))
    return q, k, v


def run_shape(args, B, H, N, D):
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    causal = args.causal
    flops = 4 * B * H * N * N * D / (2 if causal else 1)
    reps = reps_for(flops)
    layout = args.layout
    q, k, v = make_inputs(B, H, N, D, dtype, layout)
    res = {}

    def tf(ms):
        return flops / ms * 1e-9

    methods = args.methods.split(",")
    fp8_ok = arch_family() in ("cdna3", "cdna4")
    if "sage-fp16" in methods:
        res["sage-fp16"] = timeit(lambda: sapi.sageattn_rocm(q, k, v, tensor_layout=layout, is_causal=causal), reps=reps)
    if "sage-fp8" in methods and fp8_ok:
        res["sage-fp8"] = timeit(
            lambda: sapi.sageattn_rocm(q, k, v, tensor_layout=layout, is_causal=causal, pv_dtype="fp8"), reps=reps
        )
    if "sage-kern" in methods:
        for pv in ["fp16"] + (["fp8"] if fp8_ok else []):
            cfg = sapi.default_config(D, causal, pv == "fp8")
            KBLK = max(64, cfg["BLOCK_N"])
            km = k.mean(dim=2 if layout == "HND" else 1, dtype=torch.float32)
            qi, qs, ki, ks = quant_qk_int8(q, k, km, D**-0.5, BLKQ=32, BLKK=KBLK, tensor_layout=layout)
            if pv == "fp8":
                vi, vs = quant_v_fp8(v, layout, fp8_dtype())
            else:
                vi, vs = v.to(torch.float16), None
            o = torch.empty_like(q)
            fn = lambda: sage_attn_fwd(  # noqa: E731
                qi, ki, vi, qs, ks, o, QBLK=32, KBLK=KBLK, config=cfg, tensor_layout=layout, is_causal=causal, v_scale=vs
            )
            res[f"sage-{pv}-kern"] = timeit(fn, reps=reps)
    if "triton-fa" in methods and not causal:
        from nunchaku_triton_attn import attention_fp16

        qh, kh, vh = (x.transpose(1, 2) if layout == "NHD" else x for x in (q, k, v))
        o = torch.empty(B, N, H * D, device="cuda", dtype=dtype)
        res["triton-fa"] = timeit(lambda: attention_fp16(qh, kh, vh, o, D**-0.5), reps=reps)
    if "sdpa" in methods and not args.no_sdpa:
        qh, kh, vh = (x.transpose(1, 2) if layout == "NHD" else x for x in (q, k, v))
        res["sdpa"] = timeit(lambda: F.scaled_dot_product_attention(qh, kh, vh, is_causal=causal), reps=reps)
    if "aiter" in methods:
        try:
            import aiter

            qn, kn, vn = (x.transpose(1, 2).contiguous() if layout == "HND" else x for x in (q, k, v))
            res["aiter"] = timeit(lambda: aiter.flash_attn_func(qn, kn, vn, causal=causal), reps=reps)
        except Exception as e:  # noqa: BLE001
            print(f"  aiter unavailable: {type(e).__name__}: {e}"[:200])
    line = f"B{B} H{H} N{N} D{D} {'causal' if causal else 'full'} {layout} {args.dtype}:"
    for name, ms in res.items():
        line += f"  {name} {ms:.3f}ms ({tf(ms):.0f})"
    print(line, flush=True)
    return res


def sweep(args, B, H, N, D):
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    layout = args.layout
    causal = args.causal
    q, k, v = make_inputs(B, H, N, D, dtype, layout)
    flops = 4 * B * H * N * N * D / (2 if causal else 1)
    reps = reps_for(flops)
    fam = arch_family()
    cdna = fam.startswith("cdna")
    pv_list = args.sweep_pv.split(",")
    if cdna:
        space = dict(
            BLOCK_M=[64, 128, 256],
            BLOCK_N=[32, 64, 128],
            num_warps=[2, 4, 8],
            num_stages=[1, 2],
            waves_per_eu=[0, 1, 2, 3],
            matrix_instr_nonkdim=[16, 32],
            kpack=[1, 2],
        )
    else:
        space = dict(BLOCK_M=[64, 128, 256], BLOCK_N=[32, 64, 128], num_warps=[2, 4, 8], num_stages=[1, 2])
    if args.space:
        for part in args.space.split(";"):
            kname, vals = part.split("=")
            space[kname] = [int(x) if x.lstrip("-").isdigit() else x for x in vals.split(",")]
    keys = list(space)
    km = k.mean(dim=2 if layout == "HND" else 1, dtype=torch.float32)
    results = []
    for pv in pv_list:
        if pv == "fp8":
            vi, vs = quant_v_fp8(v, layout, fp8_dtype())
        else:
            vi, vs = v.to(torch.float16), None
        for vals in itertools.product(*[space[x] for x in keys]):
            cfg = dict(zip(keys, vals))
            if cfg["BLOCK_M"] * cfg["BLOCK_N"] > 256 * 64 and D > 64 and cfg["num_warps"] < 4:
                continue
            KBLK = max(64, cfg["BLOCK_N"])
            qi, qs, ki, ks = quant_qk_int8(q, k, km, D**-0.5, BLKQ=32, BLKK=KBLK, tensor_layout=layout)
            o = torch.empty_like(q)
            fn = lambda: sage_attn_fwd(  # noqa: E731
                qi, ki, vi, qs, ks, o, QBLK=32, KBLK=KBLK, config=cfg, tensor_layout=layout, is_causal=causal, v_scale=vs
            )
            try:
                ms = timeit(fn, warmup=2, reps=max(2, reps // 3), rounds=2)
            except Exception as e:  # noqa: BLE001
                print(f"  {pv} {cfg}: FAIL {type(e).__name__}: {str(e)[:120]}", flush=True)
                continue
            tfl = flops / ms * 1e-9
            results.append((tfl, pv, cfg, ms))
            print(f"  {pv} {cfg}: {ms:.3f} ms {tfl:.0f} TFLOP/s", flush=True)
    results.sort(key=lambda r: -r[0])
    print("TOP:")
    for r in results[:12]:
        print(f"  {r[1]} {r[0]:.0f} TFLOP/s {r[3]:.3f} ms {r[2]}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="image")
    p.add_argument("--shape", default=None, help="B,H,N,D[;B,H,N,D...]")
    p.add_argument("--causal", action="store_true")
    p.add_argument("--layout", default="HND")
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--methods", default="sage-fp16,sage-fp8,sage-kern,triton-fa,sdpa,aiter")
    p.add_argument("--no-sdpa", action="store_true")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--sweep-pv", default="fp16")
    p.add_argument("--space", default=None, help='e.g. "BLOCK_M=128,256;num_warps=4"')
    args = p.parse_args()
    print(f"GPU {torch.cuda.get_device_name()} ({gcn_arch()}), torch {torch.__version__}", flush=True)
    shapes = (
        [tuple(int(x) for x in s.split(",")) for s in args.shape.split(";")] if args.shape else PRESETS[args.preset]
    )
    for B, H, N, D in shapes:
        if args.sweep:
            sweep(args, B, H, N, D)
        else:
            run_shape(args, B, H, N, D)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
