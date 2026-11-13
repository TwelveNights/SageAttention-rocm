# SageAttention for AMD GPUs (ROCm)

This is a ROCm port of [SageAttention](https://github.com/thu-ml/SageAttention) (thu-ml), the 8-bit
"quantize Q·K^T to INT8" attention that speeds up diffusion and video models as a drop-in replacement
for FlashAttention / SDPA. The `rocm` branch adds an AMD backend under `sageattention/rocm/` and leaves
the public API unchanged, so anything that already calls `sageattention.sageattn` -- ComfyUI's
`--use-sage-attention`, diffusers' `set_attention_backend("sage")`, your own code -- runs on Radeon and
Instinct GPUs without modification. The original CUDA code paths are untouched; the upstream README is
kept as [README_upstream.md](README_upstream.md).

| GPU | Q·K^T | P·V | Kernel |
| --- | --- | --- | --- |
| RDNA3: Radeon RX 7000 / PRO W7000 (gfx11; tested gfx1100, gfx1101) | INT8 (WMMA) | FP16 | hand-written HIP kernel, compiled at first use with hipRTC (no compiler toolchain needed); Triton fallback |
| CDNA3: Instinct MI300X (gfx942) | INT8 (MFMA) | FP16, or FP8 (e4m3fnuz) opt-in | Triton |
| other AMD GPUs supported by Triton (gfx90a, gfx950, RDNA4, ...) | INT8 | FP16 | Triton (untested) |

Install is pure Python: nothing is compiled at install time, and it works on **Linux and native
Windows** (AMD's ROCm PyTorch wheels for Windows plus `triton-windows`).

Validated on:

| GPU | OS | Software |
| --- | --- | --- |
| Radeon RX 7800 XT 16 GB (gfx1101) | Windows 11 | ROCm 7.2.1 PyTorch wheels (torch 2.9.1), triton-windows 3.6.0.post26 |
| Radeon PRO W7900 48 GB (gfx1100) | Ubuntu 24.04 | ROCm 7.2.2 driver, torch 2.10.0+rocm7.0, Triton 3.6 |
| Instinct MI300X (gfx942) | Linux | ROCm 7.2, torch 2.10.0+rocm7.0, Triton 3.6 |

Headline numbers (details and baselines in [Results](#results)):

- **ComfyUI, Wan2.1 T2V 1.3B** (832x480, 33 frames, 20 steps): 2.15x faster sampling than ComfyUI's default attention on the RX 7800 XT
  (Windows; 7.33 -> 3.40 s/step) and 1.95x on the Radeon PRO W7900 (Linux; 5.16 -> 2.65 s/step), with
  `--use-sage-attention` and no other change.
- **Kernel**: on RDNA3 the native kernel runs at 53-56 TFLOP/s on the RX 7800 XT, 1.2-1.4x nunchaku-rocm's
  FP16 Triton flash attention and 3.5x PyTorch SDPA on Windows (FLUX 1024² shape, SDPA with
  `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`; 12x without); on MI300X it is on par with
  or ahead of aiter's bf16 FlashAttention (CK/assembly) and slower than aiter's FP8 kernel.
- **diffusers, Wan2.1 T2V 1.3B**: 1.21x (33 frames) to 1.25x (81 frames) end to end on the RX 7800 XT
  against the Triton flash attention, 1.51x against SDPA on MI300X.
- **Accuracy**: output cosine similarity >= 0.9999 against FP32 attention on synthetic DiT-like inputs and
  >= 0.99993 on real Wan2.1 activations, as upstream.

## Contents

- [Install](#install): [Linux](#linux) · [Windows](#windows) · [check the install](#check-the-install)
- [ComfyUI](#comfyui)
- [Python and diffusers](#python-and-diffusers)
- [Results](#results)
- [Validation and benchmark scripts](#validation-and-benchmark-scripts)
- [Limitations](#limitations)
- [Configuration](#configuration)
- [Credits and license](#credits-and-license)

## Install

Requirements: a ROCm build of PyTorch (>= 2.6; tested 2.9.1 and 2.10.0) and Triton (Linux: the `triton`
that comes with ROCm PyTorch; Windows: `triton-windows`). Install SageAttention **into the same Python
environment** that runs your model or ComfyUI, and never let pip replace your ROCm PyTorch -- install with
`--no-deps` (SageAttention needs nothing beyond torch and triton).

The package is the usual `sageattention` 2.2.0 (same name and API as upstream). `setup.py` detects the
ROCm build of PyTorch and skips the CUDA extensions; set `SAGEATTN_ROCM=1` to force that if the detection
cannot import torch (e.g. on a machine that also has an NVIDIA CUDA toolkit).

### Linux

```bash
# inside the venv / conda env that has ROCm PyTorch (e.g. from https://download.pytorch.org/whl/rocm7.0)
python -c "import torch, triton; print(torch.__version__, torch.version.hip, triton.__version__)"

git clone -b rocm https://github.com/TwelveNights/SageAttention-rocm.git
pip install --no-deps --no-build-isolation ./SageAttention-rocm
# with uv:  uv pip install --python /path/to/venv/bin/python --no-deps ./SageAttention-rocm
```

### Windows

Native Windows works with AMD's ROCm PyTorch wheels (ROCm 7.x for Windows, RX 7000 / RX 9000 series)
and [`triton-windows`](https://github.com/woct0rdho/triton-windows). Use 3.6 or newer (tested:
3.6.0.post26).

**ComfyUI Windows portable for AMD** (`ComfyUI_windows_portable_amd`, ships `python_embeded` with ROCm
PyTorch), from the folder that contains `python_embeded`:

```bat
python_embeded\python.exe -c "import torch; print(torch.__version__, torch.version.hip)"
python_embeded\python.exe -m pip install "triton-windows>=3.6"
git clone -b rocm https://github.com/TwelveNights/SageAttention-rocm.git
python_embeded\python.exe -m pip install --no-deps --no-build-isolation .\SageAttention-rocm
```

If `--no-build-isolation` complains about a missing `setuptools`, drop the flag and run `set SAGEATTN_ROCM=1`
first (the isolated build cannot import torch to detect ROCm).

**Your own venv** (what the RX 7800 XT numbers below were measured with), e.g. with AMD's wheels:

```powershell
# torch / torchvision / rocm_sdk from AMD's index for your GPU and ROCm version, then:
pip install "triton-windows>=3.6"
pip install --no-deps --no-build-isolation .\SageAttention-rocm
```

On RDNA3 the native kernel is compiled on first use with hipRTC (`hiprtc*.dll` from the ROCm runtime that
PyTorch loaded -- no Visual Studio, no HIP SDK) in a few seconds and cached (see
[Configuration](#configuration)). Without hipRTC it falls back to the Triton kernel.

### Check the install

```bash
python -c "import torch; from sageattention import sageattn; q = torch.randn(1, 24, 4608, 128, device='cuda', dtype=torch.float16); o = sageattn(q, q, q); r = torch.nn.functional.scaled_dot_product_attention(q, q, q); print('cos', torch.nn.functional.cosine_similarity(o.flatten().float(), r.flatten().float(), dim=0).item())"
```

prints a cosine close to 1 (the first call on RDNA3 includes the native kernel's one-time compile). For the
full test suite and benchmarks see [Validation and benchmark scripts](#validation-and-benchmark-scripts).

## ComfyUI

ComfyUI already supports SageAttention through `sageattention.sageattn`; this port just has to be importable
from ComfyUI's Python.

1. Install it into ComfyUI's Python environment as above (portable: `python_embeded\python.exe -m pip ...`;
   a venv / conda install: that environment's `pip`). If you set up ComfyUI yourself, install ComfyUI's own
   requirements without letting them replace ROCm PyTorch -- some of them (`torchsde`, `kornia`, `spandrel`)
   depend on `torch` and pip may otherwise pull a CUDA build:

   ```bash
   python -c "import torch, torchvision; print(f'torch=={torch.__version__}\ntorchvision=={torchvision.__version__}')" > constraints.txt
   pip install -r ComfyUI/requirements.txt -c constraints.txt
   ```

2. Launch ComfyUI with the flag:

   ```bash
   python main.py --use-sage-attention
   # Windows portable: add --use-sage-attention to the python_embeded\python.exe ... ComfyUI\main.py line
   # of the .bat launcher you use
   ```

3. Confirm it is active. The startup log must say

   ```
   Using sage attention
   ```

   (instead of `Using pytorch attention`). If `sageattention` cannot be imported, ComfyUI logs
   `To use the --use-sage-attention feature, the sageattention package must be installed first.` and keeps
   PyTorch attention. If a call into SageAttention fails at runtime, ComfyUI logs
   `Error running sage attention: ..., using pytorch attention instead.` -- so check the log after the first
   generation too. `example/rocm/comfyui_check.py` checks the patched function directly: it imports ComfyUI
   with the flag, asserts that `optimized_attention` is `attention_sage`, and compares it with
   `attention_pytorch` on FLUX / Wan shapes:

   ```bash
   python example/rocm/comfyui_check.py --comfy /path/to/ComfyUI
   # sage enabled: True available: True supports mask: False sage3 available: False
   # optimized_attention -> attention_sage
   #   B1 N4608 H24 D128 attention_sage  cos 0.999781 ...
   ```

There is nothing to change in workflows: every model that goes through ComfyUI's `optimized_attention`
(Wan, HunyuanVideo, LTX-Video, FLUX, SD3, Qwen-Image, ...) uses SageAttention for its unmasked attention. Calls
with an attention mask (mostly text encoders) stay on PyTorch attention, as with upstream SageAttention on
NVIDIA. ComfyUI passes `smooth_k=False`, so its accuracy is slightly below calling `sageattn()` directly
(cos 0.9998 instead of 0.9999 in the check above); images are not visibly affected.

**Example workflow.** [`example/rocm/comfyui/wan2.1_t2v_1.3B.json`](example/rocm/comfyui/wan2.1_t2v_1.3B.json)
(API format: load it in the ComfyUI web UI or POST it to `/prompt`) is the stock Wan2.1 text-to-video graph
with ComfyUI's repackaged files from
[Comfy-Org/Wan_2.1_ComfyUI_repackaged](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged):
`split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors` -> `models/diffusion_models/`,
`split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors` -> `models/text_encoders/`,
`split_files/vae/wan_2.1_vae.safetensors` -> `models/vae/`.

**Measured** with [`example/rocm/comfyui_ab.py`](example/rocm/comfyui_ab.py), which starts ComfyUI once per
attention mode, runs the workflow three times with the same seeds in every mode and compares the frames:

```bash
python example/rocm/comfyui_ab.py --comfy /path/to/ComfyUI --workflow example/rocm/comfyui/wan2.1_t2v_1.3B.json \
    --modes default,sage,kitchen --runs 3 --out runs/comfy_ab
```

| GPU (OS) | attention | sampler s/step | speedup | time per video (incl. VAE decode) | peak VRAM | PSNR vs default, runs 1-3 |
| --- | --- | --- | --- | --- | --- | --- |
| RX 7800 XT (Windows 11) | default (`Using pytorch attention`) | 7.33 | 1.00x | 195.6 s | 13.1 GiB | reference |
| RX 7800 XT (Windows 11) | default + `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` | 7.42 | 0.99x | 208.7 s | 13.1 GiB | same frames as default |
| RX 7800 XT (Windows 11) | **`--use-sage-attention`** | **3.40** | **2.15x** | **122.5 s** | 13.1 GiB | 29.3 / 28.8 / 26.9 dB |
| RX 7800 XT (Windows 11) | ComfyUI's INT8 "comfy kitchen attention" (Model Attention Backend node) | 3.55 | 2.06x | 123.9 s | 13.1 GiB | 23.9 / 23.5 / 20.8 dB |
| W7900 (Ubuntu 24.04) | default (`Using pytorch attention`) | 5.16 | 1.00x | 105.2 s | 16.3 GiB | reference |
| W7900 (Ubuntu 24.04) | **`--use-sage-attention`** | **2.65** | **1.95x** | **57.6 s** | 16.3 GiB | 26.4 / 31.5 / 25.7 dB |
| W7900 (Ubuntu 24.04) | ComfyUI's INT8 "comfy kitchen attention" | 2.75 | 1.88x | 59.4 s | 16.2 GiB | 23.7 / 22.5 / 20.8 dB |

ComfyUI @ b1a247c, Wan2.1 T2V 1.3B fp16, 832x480, 33 frames, 20 steps, uni_pc, CFG 6, seeds 42-44; s/step is
the median over runs 2-3 (run 1 includes loading and kernel compilation); "time per video" is ComfyUI's
"Prompt executed in" (prompt encoding is cached after run 1, the VAE decode -- about 50 s on the 7800 XT --
is included); peak VRAM is device memory in use, polled. RX 7800 XT: torch 2.9.1 + ROCm 7.2.1 wheels,
triton-windows 3.6.0.post26; W7900: torch 2.10.0+rocm7.0, Triton 3.6. Same seeds give the same scene,
composition and motion in all modes; SageAttention's 26-31 dB to the default output is in the same
range as the spread between *exact* attention implementations in the diffusers comparison below
(27.6-32.6 dB), while ComfyUI's own INT8 attention, about as fast here, lands 4-6 dB further away. The
default is PyTorch SDPA through ComfyUI's backend priority list (flash first); enabling PyTorch's experimental
AOTriton kernels on top changes nothing in ComfyUI (row 2, separate session). Against a well-tuned FP16 flash
attention (nunchaku-rocm's Triton kernel) the gain is smaller: 1.2-1.25x end to end (diffusers results below).
A repeat session on the 7800 XT gave 3.49 and 3.76 s/step for SageAttention, plus one run at 11.3 s/step that
was disturbed by other activity on the machine (not included above). On MI300X (Linux, torch 2.10.0+rocm7.0) the
same workflow runs at 0.519 -> 0.391 s/step (1.33x; 12.9 -> 10.6 s per video), PSNR vs default
28.2 / 31.3 / 26.5 dB; ComfyUI's INT8 attention is not available there (the node falls back to
PyTorch).

**With nunchaku-rocm's nodes.** [nunchaku-rocm](https://github.com/TwelveNights/nunchaku-rocm)'s
*Nunchaku DiT Loader (ROCm)* models (4-bit FLUX.1 / FLUX.2 [klein]) compute attention with nunchaku's own
fused Triton kernels, not through ComfyUI's `optimized_attention`, so `--use-sage-attention` does not change
them; it is safe to use the flag in the same ComfyUI (it then applies to every other model in your
workflows, e.g. a Wan video model next to a Nunchaku image model).

**SageAttention3** (`--use-sage-attention3`) needs NVIDIA Blackwell FP4 hardware. Do not install
`sageattention3_blackwell` on AMD: it would only fall back to this INT8 kernel.

## Python and diffusers

```python
from sageattention import sageattn

# q, k, v: (batch, heads, seq, head_dim) fp16/bf16 on the AMD GPU ("HND"), or tensor_layout="NHD"
out = sageattn(q, k, v, tensor_layout="HND", is_causal=False)
```

All of upstream's entry points keep their signatures and dispatch to the AMD backend on a ROCm PyTorch:
`sageattn`, `sageattn_qk_int8_pv_fp16_triton` (also takes `attn_mask`), `sageattn_qk_int8_pv_fp16_cuda`,
`sageattn_qk_int8_pv_fp8_cuda`, `sageattn_qk_int8_pv_fp8_cuda_sm90`, `sageattn_varlen`. The `*_cuda` names are
historical; on AMD they select Q-scale granularity (`per_warp` = 32 rows, `per_thread` = 16) and FP16 vs FP8
P·V. Supported: NHD/HND with arbitrary strides, fp16/bf16, head dims up to 256 (33-128 on the native gfx11
kernel), causal, GQA/MQA, masks (Triton kernel), `return_lse`, varlen, `qo_len != kv_len`, `torch.compile`
(the Triton path is traced; the native kernel is a graph break). See [ROCM_NOTES.md](ROCM_NOTES.md) for
the kernel-level API and design notes.

**diffusers** (recent versions; checked with 0.40 and 0.41) has a SageAttention backend that calls `sageattention.sageattn`:

```python
pipe = WanPipeline.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", torch_dtype=torch.bfloat16).to("cuda")
pipe.transformer.set_attention_backend("sage")
```

(or `with attention_backend("sage"):` from `diffusers.models.attention_dispatch` for a block of code).
Checked on the W7900 and RX 7800 XT: the backend reaches `sageattention.rocm`, and the output matches the
default backend (cos 0.9999). `example/rocm/wan_e2e.py` runs Wan2.1 end to end with each attention
implementation (`--attn sage-diffusers` uses this backend) and compares the videos.

## Results

All numbers below are reproducible with the scripts in this repository (commands in each section).

### Kernel speed

TFLOP/s = 4·B·H·N²·D / time (non-causal), bf16 inputs, HND, "sage" includes quantization; best of 3
(`bench/rocm/bench_attn.py`). Baselines: the FP16 Triton flash attention from nunchaku-rocm (`bench/rocm/nunchaku_triton_attn.py`, vendored for
benchmarking), aiter's CK/assembly FlashAttention on MI300X, and PyTorch SDPA.

RX 7800 XT (gfx1101, Windows 11, ROCm 7.2.1, torch 2.9.1, triton-windows 3.6; "native" = the hipRTC kernel,
which `sageattn()` uses on gfx11):

| shape (B, H, N, D) | sage native | sage Triton | Triton FA fp16 (nunchaku) | torch SDPA |
| --- | --- | --- | --- | --- |
| FLUX 1024² (1, 24, 4608, 128) | 4.93 ms (53) | 5.91 ms (44) | 6.55 ms (40) | 59.2 ms (4) |
| SD3-like (2, 24, 4096, 64) | 4.23 ms (49) | 5.09 ms (41) | 5.31 ms (39) | |
| FLUX 2048² (1, 24, 16896, 128) | 63.1 ms (56) | 75.0 ms (47) | 81.6 ms (43) | |
| Wan 1.3B 480p 41f (1, 12, 16380, 128) | 30.0 ms (55) | 36.6 ms (45) | 44.4 ms (37) | |
| Wan 1.3B 480p 81f (1, 12, 32760, 128) | 118.7 ms (56) | 144.6 ms (46) | 166.8 ms (40) | |
| same, NHD layout | 118.9 ms (55) | 146.6 ms (45) | | |
| (1, 24, 16384, 128) | 59.8 ms (55) | 70.5 ms (47) | 76.7 ms (43) | |
| causal (1, 24, 4608, 128) | 2.76 ms (47) | 3.29 ms (40) | | |
| causal (1, 12, 16380, 128) | 16.3 ms (50) | 18.7 ms (44) | | |

(TFLOP/s in parentheses.) The SDPA column is PyTorch's default on this setup; with
`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` (enables its AOTriton flash kernels, marked experimental on RDNA3
in this PyTorch build) SDPA takes 17.0 ms (15 TFLOP/s) at FLUX 1024² instead of 55-59 ms. On the W7900 (Linux,
torch 2.10) the variable makes no difference. RDNA3's WMMA runs INT8 at the *same* rate as FP16 and has no FP8, so INT8 Q·K^T
buys no matrix throughput there; the speedup comes from the hand-written kernel, which reaches ~75% of the
7800 XT's 74.6 TFLOP/s FP16 peak (boost clock) at video shapes and ~71% at FLUX 1024².

Radeon PRO W7900 (gfx1100, Linux, ROCm 7.2.2 driver, torch 2.10 ROCm 7.0): native 66-87 TFLOP/s, Triton sage
59-70, Triton FA 53-64, torch SDPA 22-23 on the same shapes (e.g. Wan 1.3B 81f: 84-90 ms native, 98-107 ms
Triton sage, 114-123 ms Triton FA, 292 ms SDPA; two sessions).

MI300X (gfx942, ROCm 7.2, torch 2.10, Triton 3.6; aiter 0.1.7 `flash_attn_func`; shared VM, ±10%):

| shape (B, H, N, D) | sage FP16 P·V | sage FP8 P·V | aiter bf16 | aiter FP8 (per-tensor) | Triton FA fp16 (nunchaku) | torch SDPA |
| --- | --- | --- | --- | --- | --- | --- |
| FLUX 1024² (1, 24, 4608, 128) | 0.57 ms (459) | 0.51 ms (516) | 0.58 ms (451) | 0.42 ms (621) | 0.84 ms (311) | 0.85 ms (307) |
| SD3-like (2, 24, 4096, 64) | 0.57 ms (361) | 0.55 ms (375) | 0.66 ms (313) | -- | 0.76 ms (271) | 0.72 ms (288) |
| FLUX 2048² (1, 24, 16896, 128) | 6.67 ms (526) | 5.36 ms (654) | 7.30 ms (481) | 5.34 ms (658) | 10.85 ms (323) | 11.38 ms (308) |
| Wan 1.3B 480p (1, 12, 32760, 128) | 12.6 ms (524) | 10.4 ms (636) | 14.2 ms (465) | 10.7 ms (615) | 19.3 ms (342) | 22.3 ms (296) |
| Wan 14B 480p (1, 40, 32760, 128) | 38.7 ms (569) | 31.4 ms (700) | 40.7 ms (540) | 34.3 ms (640) | 58.2 ms (377) | 68.5 ms (321) |
| HunyuanVideo-like (1, 24, 45000, 128) | 43.7 ms (569) | 35.8 ms (694) | 45.8 ms (544) | 37.2 ms (668) | 65.6 ms (379) | 77.6 ms (321) |
| Wan 14B 720p (1, 40, 75600, 128) | 196 ms (598) | 161 ms (729) | 215 ms (545) | 179 ms (653) | 302 ms (388) | 349 ms (335) |

"aiter FP8" is a speed reference only (per-tensor FP8 Q/K/V cast outside the timed region, no scaling).
Upstream's own Triton SageAttention, where it compiles on ROCm Triton 3.6 (head dim 64 only): 0.71 ms (289)
on the SD3-like shape. Summary: the fastest of everything measured against the Triton FA and SDPA on all
three GPUs; on MI300X the default FP16 P·V path is on par with aiter bf16 at FLUX 1024² and ~10% ahead at
long video shapes, but slower than aiter's per-tensor FP8 kernel; sage FP8 P·V beats aiter FP8 only at
>= ~32K tokens.

### Accuracy

`bench/rocm/accuracy.py`: output vs FP32 attention, cosine similarity and relative L1 as in the SageAttention
papers. Synthetic inputs have a large per-channel K bias and a few 6x outlier Q/K channels, like real DiT
activations (B2 H8 N4096, bf16):

| head dim, mask | method | MI300X cos sim | MI300X rel-L1 | RX 7800 XT cos sim | RX 7800 XT rel-L1 |
| --- | --- | --- | --- | --- | --- |
| 64, full | sage, FP16 P·V | 0.999913 | 0.0097 | 0.999920 | 0.0095 |
| 64, full | sage, FP8 P·V (gfx942) | 0.999691 | 0.0215 | -- | -- |
| 64, full | upstream's own Triton SageAttention, run unmodified | 0.999899 | 0.0104 | 0.999899 | 0.0106 |
| 64, full | SDPA in bf16 (for scale) | 0.999999 | 0.0014 | 0.999999 | 0.0014 |
| 128, full | sage, FP16 P·V | 0.999898 | 0.0107 | 0.999901 | 0.0106 |
| 128, full | sage, FP8 P·V | 0.999652 | 0.0232 | -- | -- |
| 128, causal | sage, FP16 P·V | 0.999902 | 0.0103 | 0.999905 | 0.0101 |
| 128, causal | sage, FP8 P·V | 0.999643 | 0.0235 | -- | -- |

On the 7800 XT the native and Triton kernels give the same metrics to 6 digits (same quantization). The
default `sageattn()` granularity is `per_warp` (32-row Q scales), slightly finer than upstream's per-block
Triton path. Real activations -- self-attention Q/K/V captured from Wan2.1-T2V-1.3B (14040 tokens, 12 heads,
d=128) at 5 (step, block) points of a 20-step run (`wan_e2e.py --dump-qkv`, `accuracy.py --qkv`), MI300X:

| method | cos sim (min..max) | rel-L1 (min..max) |
| --- | --- | --- |
| sage, FP16 P·V | 0.99993 .. 0.99998 | 0.0037 .. 0.0100 |
| sage, FP8 P·V | 0.99974 .. 0.99998 | 0.0062 .. 0.0201 |
| SDPA in bf16 | 0.999998 .. 0.999999 | 0.0011 .. 0.0017 |

FP8 P·V keeps P in e4m3 (3 mantissa bits), as upstream's FP8 kernels do; it has ~2x the FP16 path's error, so
`sageattn()` uses FP16 P·V by default on gfx942 and FP8 is opt-in.

### End to end, diffusers (Wan2.1-T2V-1.3B)

`example/rocm/wan_e2e.py`: 480x832, 33 frames, 20 steps, CFG 5, same seed; prompts encoded once with UMT5-XXL;
initial noise from a CPU generator so runs on different GPUs start from identical latents. "PSNR vs
reference" compares the decoded 33-frame video with the MI300X bf16-SDPA run.

| GPU | attention | denoise (20 steps) | per step | speedup | PSNR vs reference | latent cos |
| --- | --- | --- | --- | --- | --- | --- |
| RX 7800 XT | Triton FA fp16 (nunchaku) | 85.9 s | 4.20 s | 1.00x | 29.81 dB | 0.9975 |
| RX 7800 XT | **SageAttention (native)** | **70.9 s** | **3.52 s** | **1.21x** | 26.80 dB | 0.9943 |
| W7900 | torch SDPA | 103.6 s | 5.19 s | 0.62x | 28.94 dB | 0.9964 |
| W7900 | Triton FA fp16 | 64.2 s | 3.22 s | 1.00x | 32.63 dB | 0.9989 |
| W7900 | **SageAttention (native)** | **55.3 s** | **2.78 s** | **1.16x** | 29.83 dB | 0.9971 |
| MI300X | torch SDPA (bf16) | 12.1 s | 0.55 s | 1.00x | (reference) | 1 |
| MI300X | Triton FA fp16 | 11.6 s | 0.58 s | 1.04x | 30.25 dB | 0.9976 |
| MI300X | aiter FA (bf16) | 8.3 s | 0.41 s | 1.46x | 27.58 dB | 0.9943 |
| MI300X | **SageAttention, FP16 P·V** | **8.0 s** | **0.40 s** | **1.51x** | 28.69 dB | 0.9964 |
| MI300X | SageAttention, FP8 P·V | 9.7 s | 0.48 s | 1.26x | 25.94 dB | 0.9925 |

Diffusion sampling amplifies any rounding difference, so the useful yardstick is the spread between *exact*
attention implementations: they land 27.6-32.6 dB from the reference (SDPA itself on the W7900: 28.9 dB);
SageAttention lands at 26.8-29.8 dB, and the frames show the same scene, composition and motion. At 81 frames
(32760 tokens, 10 steps) the 7800 XT goes from 154.4 s to 123.7 s (1.25x). Peak VRAM while denoising is
unchanged (3.9 GiB at 33 frames, 4.8 GiB at 81). The ComfyUI results are in the [ComfyUI](#comfyui) section.

### 3D generation (image to 3D model)

Same settings with PyTorch attention vs SageAttention, warm runs; the outputs match (same mesh and texture, rendered
side by side).

| Model | GPU | PyTorch attention | SageAttention | speedup |
| --- | --- | --- | --- | --- |
| Pixal3D in ComfyUI (`--use-sage-attention`), full image -> textured GLB | RX 7800 XT, Windows | 387 s | 215 s | 1.8x |
| Hunyuan3D-2.1 PBR texturing ([Hunyuan3D-2.1-rocm](https://github.com/TwelveNights/Hunyuan3D-2.1-rocm)) | W7900 | 64 s | 46 s | 1.4x |
| Hunyuan3D-2.1 shape, FlashVDM decoder | W7900 | 54 s | 42 s | 1.3x |
| TripoSG ([TripoSG-rocm](https://github.com/TwelveNights/TripoSG-rocm)), 50 steps | W7900 | 37 s | 30 s | 1.2x |

TripoSG and the Hunyuan3D-2.1 painter in ComfyUI: [ComfyUI-3D-rocm](https://github.com/TwelveNights/ComfyUI-3D-rocm) (uses SageAttention when installed).

In ComfyUI's TRELLIS.2 / Pixal3D the two high-resolution sampler stages (sparse attention over tens of thousands of
tokens) go from 161 s to 68 s and from 88 s to 37 s.

## Validation and benchmark scripts

| script | what it does |
| --- | --- |
| `scripts/rocm/validate.py [--bench]` | test suite + accuracy table (+ the kernel benchmarks above) on the local GPU; Linux and Windows |
| `tests/rocm/` (`pytest tests/rocm`) | 116 tests: NHD/HND, fp16/bf16, head dims 32-256, causal, GQA, masks, LSE, varlen, ragged lengths, native vs Triton, `torch.compile` (skipped on Windows, which has no host compiler for Inductor). 115 pass + 1 skip on the RX 7800 XT, 116 on the W7900; on MI300X the 70 gfx11-only tests skip |
| `bench/rocm/accuracy.py [--qkv file]` | accuracy table (synthetic or captured Q/K/V) |
| `bench/rocm/bench_attn.py --preset image\|video\|video_small` | kernel speed vs baselines (`--methods`, `--shape B,H,N,D`, `--causal`, `--layout NHD`) |
| `example/rocm/wan_e2e.py` | diffusers Wan2.1 end to end with each attention implementation, video comparison |
| `example/rocm/comfyui_check.py --comfy DIR` | ComfyUI's `--use-sage-attention` path at function level |
| `example/rocm/comfyui_ab.py --comfy DIR --workflow WF` | ComfyUI workflow A/B: default, default + `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`, `--use-sage-attention`, ComfyUI's INT8 "comfy kitchen" attention; timings, VRAM, PSNR |

Timings are only meaningful on an idle GPU; run `validate.py` once before `validate.py --bench-only` so the
Triton and hipRTC compile caches are warm.

## Limitations

- SageAttention3 (FP4 microscaling, `sageattention3_blackwell/`) needs NVIDIA Blackwell FP4 tensor cores and
  is not ported; on ROCm `sageattn3_blackwell()` falls back to `sageattn` with a warning.
- INT4 Q·K^T (SageAttention2's 4-bit variant; RDNA3 has 2x-rate INT4 WMMA) is not implemented: it costs
  accuracy and is not in upstream's current API.
- Causal attention at short lengths is not tuned on MI300X (B4 H32 N4096: 1.67 ms vs 1.27 ms for aiter bf16);
  only the non-causal path was tuned for DiTs. FP8 P·V (MI300X) is opt-in because of its accuracy.
- Masked attention runs on the Triton kernel (not the native gfx11 one); ComfyUI does not send masks to
  SageAttention at all (see [ComfyUI](#comfyui)).
- Tested on gfx942, gfx1100 and gfx1101 only. RDNA4 (gfx12), RDNA2, gfx90a and gfx950 run the generic Triton
  path, untested. Windows support depends on `triton-windows` and AMD's ROCm wheels for Windows.
- The first call per shape class compiles Triton kernels (and, on RDNA3, the native kernel once per GPU
  arch); later runs load them from the caches.

## Configuration

| variable | effect |
| --- | --- |
| `SAGEATTN_NATIVE=0` / `1` | RDNA3: disable the native hipRTC kernel (Triton only) / require it (raise instead of falling back) |
| `SAGEATTN_CACHE_DIR` | where compiled native kernels are cached (default `~/.cache/sageattention`, `%LOCALAPPDATA%\sageattention\cache` on Windows) |
| `SAGEATTN_ROCM_PV=fp8` | MI300X: FP8 P·V in `sageattn()` (default FP16) |
| `SAGEATTN_ROCM=1` | install time: force the pure-Python ROCm install |
| `AMD_COMGR_CACHE_DIR` | ROCm's own hipRTC compiler cache (ROCm default: a per-user directory) |
| `TRITON_CACHE_DIR` | Triton's kernel cache |

## Credits and license

- SageAttention, SageAttention2/2++ and SageAttention3 are by Jintao Zhang, Jia Wei, Haofeng Huang, Pengle
  Zhang, Jun Zhu, Jianfei Chen and co-authors (Tsinghua University, thu-ml); please cite their papers (see
  [README_upstream.md](README_upstream.md#citation)) if you use this work. This port re-implements their
  quantization scheme and kernels for AMD GPUs; the algorithms are theirs.
- The native-kernel infrastructure (`sageattention/rocm/native/hip.py`, `hiprtc.py`, `build.py`,
  `hip_prelude.h`) is adapted from
  [nunchaku-rocm](https://github.com/TwelveNights/nunchaku-rocm), and the benchmark baseline
  `bench/rocm/nunchaku_triton_attn.py` is its Triton flash attention, vendored unchanged except for one
  inlined helper (Apache-2.0).
- License: Apache-2.0, same as upstream ([LICENSE](LICENSE)). The ROCm backend is new code under
  `sageattention/rocm/`, `bench/rocm/`, `example/rocm/`, `tests/rocm/` and `scripts/rocm/`, plus small
  dispatch changes in `sageattention/core.py`, `setup.py` and `sageattention3_blackwell/`.
- Not affiliated with or endorsed by the SageAttention authors or AMD.
