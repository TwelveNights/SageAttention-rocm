"""
Validate the ROCm backend on the local AMD GPU: test suite, accuracy table and (optionally) kernel benchmarks.

    python scripts/rocm/validate.py                 # tests + accuracy (a few minutes)
    python scripts/rocm/validate.py --bench         # + the kernel benchmarks reported in the README
    python scripts/rocm/validate.py --bench-only    # benchmarks only (run after a validate pass so
                                                    # Triton / hipRTC compile caches are warm)

Works the same on Linux and Windows; run it from the repository root with the Python environment
that has the ROCm build of PyTorch and Triton (triton-windows on Windows). Each step runs in its own
process; the script exits non-zero if any step fails. Useful environment variables (all optional):

  SAGEATTN_CACHE_DIR    compiled native gfx11 kernel cache (default ~/.cache/sageattention or
                        %LOCALAPPDATA%\\sageattention\\cache)
  TRITON_CACHE_DIR      Triton's kernel cache
  AMD_COMGR_CACHE_DIR   hipRTC/comgr compiler cache (ROCm writes one per user by default)
  SAGEATTN_NATIVE=0     test / benchmark the Triton kernel only on gfx11

Benchmark timings are only meaningful on an otherwise idle GPU.
"""

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", action="store_true", help="also run the kernel benchmarks")
    p.add_argument("--bench-only", action="store_true", help="only run the kernel benchmarks")
    p.add_argument("--methods", default=None, help="bench_attn.py --methods (default: per-arch selection)")
    args = p.parse_args()

    import torch

    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
        raise SystemExit("needs a ROCm build of PyTorch with a visible AMD GPU")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    print(f"GPU: {torch.cuda.get_device_name(0)} ({arch}), torch {torch.__version__}, HIP {torch.version.hip}")
    methods = args.methods or ("sage-fp16,sage-triton,triton-fa,sdpa" if arch.startswith("gfx11") else "sage-fp16,sage-fp8,triton-fa,sdpa")

    steps = []
    if not args.bench_only:
        steps += [
            ["-m", "pytest", "tests/rocm", "-q", "-p", "no:cacheprovider"],
            ["bench/rocm/accuracy.py"],
        ]
    if args.bench or args.bench_only:
        steps += [
            ["bench/rocm/bench_attn.py", "--preset", "image", "--methods", methods],
            ["bench/rocm/bench_attn.py", "--preset", "video_small", "--methods", methods],
            ["bench/rocm/bench_attn.py", "--shape", "1,24,4608,128;1,12,16380,128", "--causal",
             "--methods", methods.replace(",sdpa", "")],  # fmt: skip
        ]

    t0, failed = time.time(), []
    for step in steps:
        print(">>>", " ".join(step), flush=True)
        rc = subprocess.run([sys.executable, "-u", "-W", "ignore", *step], cwd=ROOT).returncode
        print(f"[{time.time() - t0:.0f} s] exit {rc}", flush=True)
        if rc:
            failed.append((step[0] if step[0] != "-m" else step[1], rc))
    print("FAILED: " + ", ".join(f"{s} (exit {rc})" for s, rc in failed) if failed else "all steps passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
