"""
Check that ComfyUI's `--use-sage-attention` path runs this package unchanged on an AMD GPU: imports
ComfyUI's attention module with that flag and compares `attention_sage` (ComfyUI's wrapper around
`sageattention.sageattn`) with `attention_pytorch` on FLUX- and Wan-like shapes, in both of ComfyUI's
call layouts (B, N, H*D and pre-split B, H, N, D).

    python example/rocm/comfyui_check.py --comfy /path/to/ComfyUI
"""

import argparse
import sys
import time

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--comfy", required=True)
    args = p.parse_args()
    sys.argv = [sys.argv[0], "--use-sage-attention"]
    sys.path.insert(0, args.comfy)
    import comfy.options  # noqa: E402

    comfy.options.enable_args_parsing()  # as ComfyUI's main.py does, so --use-sage-attention is seen
    import comfy.ldm.modules.attention as A  # noqa: E402
    from comfy import model_management  # noqa: E402

    print("sage enabled:", model_management.sage_attention_enabled(), "available:", A.SAGE_ATTENTION_IS_AVAILABLE,
          "supports mask:", A.SAGE_ATTENTION_SUPPORTS_MASK, "sage3 available:", A.SAGE_ATTENTION3_IS_AVAILABLE)  # fmt: skip
    print("optimized_attention ->", A.optimized_attention.__name__)
    torch.manual_seed(0)
    for b, n, h, d in [(1, 4608, 24, 128), (1, 14040, 12, 128), (2, 4096, 10, 64)]:
        q, k, v = (torch.randn(b, n, h * d, device="cuda", dtype=torch.bfloat16) for _ in range(3))
        k = k + 2.0 * torch.randn(1, 1, h * d, device="cuda", dtype=torch.bfloat16)
        ref = A.attention_pytorch(q, k, v, h)
        for fn in (A.attention_sage, A.optimized_attention) + ((A.attention3_sage,) if A.SAGE_ATTENTION3_IS_AVAILABLE else ()):
            out = fn(q, k, v, h)
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(5):
                fn(q, k, v, h)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t) / 5 * 1e3
            cos = torch.nn.functional.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0).item()
            l1 = ((out.float() - ref.float()).abs().sum() / ref.float().abs().sum()).item()
            print(f"  B{b} N{n} H{h} D{d} {fn.__name__:<20} cos {cos:.6f} relL1 {l1:.4f} {ms:.2f} ms  shape {tuple(out.shape)}")
        # pre-split layout (skip_reshape=True -> HND)
        qs, ks, vs = (x.view(b, n, h, d).transpose(1, 2) for x in (q, k, v))
        out = A.attention_sage(qs, ks, vs, h, skip_reshape=True)
        cos = torch.nn.functional.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0).item()
        print(f"  B{b} N{n} H{h} D{d} attention_sage(skip_reshape) cos {cos:.6f} shape {tuple(out.shape)}")
        t = time.perf_counter()
        for _ in range(5):
            A.attention_pytorch(q, k, v, h)
        torch.cuda.synchronize()
        print(f"  B{b} N{n} H{h} D{d} attention_pytorch {(time.perf_counter() - t) / 5 * 1e3:.2f} ms")


if __name__ == "__main__":
    main()
