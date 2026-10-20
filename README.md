# SageAttention
<!-- We are continuously updating more features. You could **Star** and **Watch** our repository to stay updated.

--- -->
This repository provides the official implementation of SageAttention, SageAttention2, and SageAttention2++, which achieve surprising speedup on most GPUs without lossing accuracy across all models in a plug-and-play way.

**SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration**  
Jintao Zhang, Jia Wei, Haofeng Huang, Pengle Zhang, Jun Zhu, Jianfei Chen  
Paper: https://arxiv.org/abs/2410.02367

**SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization**  
Jintao Zhang, Haofeng Huang, Pengle Zhang, Jia Wei, Jun Zhu, Jianfei Chen  
Paper: https://arxiv.org/abs/2411.10958

**SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training**  
Jintao Zhang, Jia Wei, Haoxu Wang, Pengle Zhang, Xiaoming Xu, Haofeng Huang, Kai Jiang, Jianfei Chen, Jun Zhu  
Paper: https://arxiv.org/abs/2505.11594


![Local Image](./assets/2.png)
*Note: [SageAttention2++](https://arxiv.org/pdf/2505.21136) achieves higher speed while maintaining the same accuracy performance.*

## Current Features
<!-- This is a beta release of SageAttention2. We welcome any feedback on accuracy, performance issues, bugs, feature requests, or suggestions. Please feel free to open an issue or launch a pull request! -->

+ Optmized kernels for **Ampere, Ada and Hopper GPUs.**
+ INT8 quantization and smoothing for $QK^\top$ with support for varying granularities.
+ FP8 quantization for $PV$, and FP16 accumulator for FP8/FP16 $PV$.
+ Two-level accumulation strategy for $PV$ to improve accuracy in FP8 MMA and WGMMA.
+ Support `torch.compile` with non-cudagraphs mode and distributed inference.


## Project Updates
- [2025-09-27]: 🎉 [SageAttention3](https://arxiv.org/abs/2505.11594) is accepted by NeurIPS 2025 as a **Spotlight** paper! 
- [2025-09-27]: The code of [SageAttention3](https://arxiv.org/abs/2505.11594) is released in this repository at  [sageattention3_blackwell](./sageattention3_blackwell/). We would still greatly appreciate it if you could take a moment to fill out the Form in [Huggingface](https://huggingface.co/jt-zhang/SageAttention3). Please note that since SageAttention2 is more accurate, we still recommend using SageAttention2 for precision-sensitive applications.
- [2025-07-01]: The code of [SageAttention2++](https://arxiv.org/pdf/2505.21136) is released in this repository. We would still greatly appreciate it if you could take a moment to fill out the Form in [Huggingface](https://huggingface.co/jt-zhang/SageAttention2_plus). Thank you very much!

![Local Image](./assets/5090_sageattn2++.png)

![Local Image](./assets/4090_sageattn2++.png)

- [2025-06-19]: [Sparse SageAttention1 API](https://github.com/jt-zhang/Sparse_SageAttention_API) and [Sparse SageAttention2 API](https://github.com/thu-ml/SpargeAttn) can compute attention with any block sparse pattern very fast.
- [2025-05-02]: 🎉SageAttention2 and [SpargeAttn](https://github.com/thu-ml/SpargeAttn) are accepted by ICML 2025! 
- [2025-02-25]: 🔥 We release [SpargeAttn](https://github.com/thu-ml/SpargeAttn), a sparse attention based on SageAttention2, which could acclerate any model without training.
- [2025-02-15]: 🔥 The compilation code is updated to support RTX5090! On RTX5090, SageAttention reaches 560T, 2.7x faster than FlashAttention2!
- [2025-01-28]: 🔥⚡SageAttention is now available on Hopper GPUs (H100, H800, H20)! It matches the speed of FlashAttention3-FP8 but offers **much better accuracy!**

| **FlashAttention2** | **FlashAttention3** | **FlashAttention3-FP8** | **SageAttention** |
|----------------------|----------------------|----------------------|----------------------|
| ![FlashAttention2](assets/cogvideox1.5_fa2_example.gif) | ![FlashAttention3](assets/cogvideox1.5_fa3_example.gif)  | ![FlashAttention3-FP8](assets/cogvideox1.5_fa3fp8_example.gif) | ![SageAttention](assets/cogvideox1.5_sage_example.gif) |
| **25'34''** | **17'32''** | **12'14''** | **12'07''** |

*Results for [CogVideoX1.5-5B](https://huggingface.co/THUDM/CogVideoX1.5-5B) on NVIDIA H20 GPU*

![Local Image](./assets/H100_hd128.png)

![Local Image](./assets/H20_hd128.png)

- [2025-01-24]: 🎉SageAttention is accepted by ICLR 2025! 
- [2024-12-20]: 🔥Update the [SageAttention2 Paper](https://arxiv.org/abs/2411.10958).   

- [2024-12-20]: 🔥Release SageAttention 2.0.1 Beta! In this version, we introduce a new feature: per-thread quantization, which offers finer granularity while maintaining hardware efficiency.
- [2024-11-21]: 🔥SageAttention 2.0.0 beta is released! Now SageAttention has measured speedup on L20, L40, A100, A800, and A6000, RTX3090 and RTX4090.
- [2024-11-12]: Support for `sageattn_varlen` is available now.
- [2024-11-11]: Support for different sequence lengths between `q` and `k,v`,  `(batch_size, head_num, seq_len, head_dim)` or `(batch_size, seq_len, head_num, head_dim)` input shapes, and `group-query attention` is available now.


## AMD GPUs (ROCm)

> This section documents the ROCm port on the `rocm` branch of this fork (not upstream). Everything below is
> new code under `sageattention/rocm/`; the CUDA paths are untouched.

On a ROCm build of PyTorch every public entry point -- `sageattn`, `sageattn_qk_int8_pv_fp16_triton`,
`sageattn_qk_int8_pv_fp16_cuda`, `sageattn_qk_int8_pv_fp8_cuda`, `sageattn_qk_int8_pv_fp8_cuda_sm90`,
`sageattn_varlen` -- keeps its signature and dispatches to the AMD backend, so callers such as ComfyUI
(`--use-sage-attention`) and diffusers (`transformer.set_attention_backend("sage")`) work unchanged.

**What runs where**

| GPU | Q·K^T | P·V | Kernel |
| --- | --- | --- | --- |
| CDNA3 (MI300X, gfx942) | INT8 (MFMA i8) | FP16, or FP8 e4m3**fnuz** with `pv_dtype="fp8"` / the `*_fp8_*` entry points | Triton (`sageattention/rocm/attn.py`) |
| RDNA3 (RX 7000 / W7000, gfx11) | INT8 (WMMA iu8) | FP16 (RDNA3 has no FP8) | hand-written HIP kernel compiled at runtime with hipRTC (`sageattention/rocm/native/sage_gfx11.hip`), Triton fallback |
| other AMD GPUs | INT8 | FP16 | Triton |

Quantization follows upstream: per-block INT8 Q (pre-scaled by `sm_scale·log2 e`) and K (after `smooth_k`,
i.e. subtracting the per-channel mean of K over the sequence), FP32 online softmax in the exp2 domain.
`qk_quant_gran` maps to the Q-scale block height (`per_block` 128 rows, `per_warp` 32, `per_thread` 16 --
there are no NVIDIA warps / mma thread layouts to mirror). AMD matrix cores always accumulate in FP32, so
`pv_accum_dtype` only selects FP8 vs FP16 P·V. Supported: NHD/HND (any strides, e.g. views of a fused QKV),
fp16/bf16, head dims up to 256 (33-128 on the native gfx11 kernel), causal, GQA/MQA, `attn_mask` (bool or
additive; Triton kernel), `return_lse`, varlen, `qo_len != kv_len`, `torch.compile` (Triton path traced, native
kernel graph-breaks). Tested on gfx942, gfx1100 and gfx1101; gfx90a / gfx950 / gfx12 use the same Triton code
but are untested.

**SageAttention3** (`sageattention3_blackwell/`, microscaling FP4) needs Blackwell's block-scaled FP4 tensor
cores; no AMD GPU before CDNA4 has an equivalent, so it is not ported. On ROCm its `setup.py` installs the
Python package only and `sageattn3_blackwell()` falls back to `sageattention.sageattn` (INT8) with a one-time
warning.

**Install** (pure Python + Triton, nothing is compiled at install time):

```bash
pip install --no-build-isolation .     # detects ROCm torch (or set SAGEATTN_ROCM=1) and skips the CUDA extensions
```

The gfx11 kernel is compiled on first use with hipRTC from the ROCm runtime torch already loaded (no hipcc,
no MSVC on Windows) in a few seconds, self-tested against the Triton kernel, and cached under
`$SAGEATTN_CACHE_DIR` (default `~/.cache/sageattention` / `%LOCALAPPDATA%\sageattention\cache`).
`SAGEATTN_NATIVE=0` disables it, `=1` forces it (and raises instead of falling back). `SAGEATTN_ROCM_PV=fp8`
makes `sageattn()` use FP8 P·V on gfx942 (default FP16, see accuracy below).

**Accuracy** (`bench/rocm/accuracy.py`; output vs FP32 SDPA, cosine similarity and relative L1 as in the
SageAttention papers; synthetic inputs have a large per-channel K bias and a few 6x outlier Q/K channels, like
real DiT activations). `pytest tests/rocm` checks the same metrics over NHD/HND, fp16/bf16, head dims
32-256, causal, GQA, masks, LSE, varlen and ragged lengths (115 pass on the RX 7800 XT (+1 skipped: torch.compile
needs a host compiler on Windows), 70 + 70 gfx11-only skips on MI300X, 116 on the W7900).

Synthetic (B2 H8 N4096, bf16):

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

On the 7800 XT the native gfx11 kernel and the Triton kernel give the same metrics to 6 digits (same
quantization). The default `sageattn()` granularity here is `per_warp` (32-row Q scales), which is why it is a
little more accurate than upstream's per-block Triton path. Upstream's Triton kernels themselves only compile on
ROCm Triton 3.6 for head dim 64 (head dim 128 and the causal kernel crash the AMD pipeliner).

Real activations -- self-attention Q/K/V captured from Wan2.1-T2V-1.3B (14040 tokens, 12 heads, d=128) at 5
(step, block) points of a 20-step run (`wan_e2e.py --dump-qkv`, `accuracy.py --qkv`), MI300X:

| method | cos sim (min..max) | rel-L1 (min..max) |
| --- | --- | --- |
| sage, FP16 P·V | 0.99993 .. 0.99998 | 0.0037 .. 0.0100 |
| sage, FP8 P·V | 0.99974 .. 0.99998 | 0.0062 .. 0.0201 |
| SDPA in bf16 | 0.999998 .. 0.999999 | 0.0011 .. 0.0017 |

FP8 P·V keeps P in e4m3 (3 mantissa bits), as upstream's FP8 kernels do; it is ~2x the FP16 path's error, so
`sageattn()` defaults to FP16 P·V on gfx942 and FP8 is opt-in (`SAGEATTN_ROCM_PV=fp8` or the `*_fp8_*` APIs).

**Kernel speed** (TFLOP/s = 4·B·H·N²·D / time, non-causal, bf16 in, HND; "full" includes quantization;
best of 3 × N runs; the MI300X is a shared VM, numbers ±10%):

MI300X (gfx942, ROCm 7.2, torch 2.10, Triton 3.6; aiter 0.1.7 `flash_attn_func` = CK/asm FlashAttention v3):

| shape (B, H, N, D) | sage FP16 P·V | sage FP8 P·V | aiter bf16 | aiter FP8 (per-tensor) | Triton FA fp16 (nunchaku) | torch SDPA |
| --- | --- | --- | --- | --- | --- | --- |
| FLUX 1024² (1, 24, 4608, 128) | 0.57 ms (459) | 0.51 ms (516) | 0.58 ms (451) | 0.42 ms (621) | 0.84 ms (311) | 0.85 ms (307) |
| SD3-like (2, 24, 4096, 64) | 0.57 ms (361) | 0.55 ms (375) | 0.66 ms (313) | -- | 0.76 ms (271) | 0.72 ms (288) |
| FLUX 2048² (1, 24, 16896, 128) | 6.67 ms (526) | 5.36 ms (654) | 7.30 ms (481) | 5.34 ms (658) | 10.85 ms (323) | 11.38 ms (308) |
| Wan 1.3B 480p (1, 12, 32760, 128) | 12.6 ms (524) | 10.4 ms (636) | 14.2 ms (465) | 10.7 ms (615) | 19.3 ms (342) | 22.3 ms (296) |
| Wan 14B 480p (1, 40, 32760, 128) | 38.7 ms (569) | 31.4 ms (700) | 40.7 ms (540) | 34.3 ms (640) | 58.2 ms (377) | 68.5 ms (321) |
| HunyuanVideo-like (1, 24, 45000, 128) | 43.7 ms (569) | 35.8 ms (694) | 45.8 ms (544) | 37.2 ms (668) | 65.6 ms (379) | 77.6 ms (321) |
| Wan 14B 720p (1, 40, 75600, 128) | 196 ms (598) | 161 ms (729) | 215 ms (545) | 179 ms (653) | 302 ms (388) | 349 ms (335) |

(TFLOP/s in parentheses. "aiter FP8" is a speed reference only: per-tensor FP8 Q/K/V cast outside the timed
region, no scaling. NHD inputs run at the same speed.) Upstream's own Triton SageAttention, where it compiles
(D=64): 0.71 ms (289) on the SD3-like shape.

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

Radeon PRO W7900 (gfx1100, Linux, ROCm 7.2.2 driver / torch 2.10 ROCm 7.0): native 66-87 TFLOP/s, Triton
sage 59-70, Triton FA 53-64, torch SDPA 22-23 on the same shapes (e.g. Wan 1.3B 81f: 84-90 ms native,
98-107 ms Triton sage, 114-123 ms Triton FA, 292 ms SDPA; two sessions).

RDNA3's WMMA runs INT8 at the *same* rate as FP16 (no FP8 at all), so INT8 Q·K^T buys no matrix throughput
there; the gfx11 speedup comes from the hand-written kernel (~75% of the 7800 XT's 74.6 TFLOP/s FP16 boost-clock peak
at large video shapes, 56 TFLOP/s; ~71% at FLUX 1024²).

**End to end** (Wan2.1-T2V-1.3B, diffusers, 480×832, 33 frames, 20 steps, CFG 5, same seed):

Prompts encoded once with UMT5-XXL; initial noise from a CPU generator, so runs on different GPUs start from
identical latents. "PSNR vs reference" compares the decoded 33-frame video with the MI300X bf16-SDPA run.

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
attention implementations: they land 27.6-32.6 dB from the reference (SDPA itself on the W7900: 28.9 dB), and
SageAttention lands at 26.8-29.8 dB (the same kernel on two RDNA3 boards differs by 27.3 dB from itself); the
frames show the same scene, composition and motion (`frame016.png` of each run). At 81 frames (32760 tokens, 10 steps) the 7800 XT goes from 154.4 s to 123.7 s (1.25x). Peak VRAM while
denoising is unchanged (3.9 GiB at 33 frames, 4.8 GiB at 81). MI300X timings are on a shared VM (±10%).

ComfyUI: `example/rocm/comfyui_check.py` imports ComfyUI with `--use-sage-attention` and checks that
`optimized_attention` resolves to `attention_sage` and runs this package unchanged (MI300X: 0.65 vs 0.98 ms
FLUX 1024², 2.4 vs 4.7 ms Wan 480p 33f against `attention_pytorch`; note ComfyUI calls `sageattn` with
`smooth_k=False`).

**Summary.** Fastest of everything measured vs nunchaku's Triton FA and torch SDPA on all three GPUs; on MI300X
the default (FP16 P·V) path is competitive with aiter bf16 (within noise at FLUX 1024², ~10% ahead at long video
shapes) but slower than aiter's per-tensor FP8 kernel, and sage FP8 P·V beats aiter FP8 only at >= ~32K tokens.

**Limitations / not done.** Causal attention at short lengths is slower than aiter on MI300X (B4 H32 N4096:
1.67 ms vs 1.27 ms for aiter bf16; at H32 N16384 4.86 vs 4.69 ms), since only the non-causal path was tuned for DiTs.
On RDNA3, INT4 Q·K^T (SageAttention2's 4-bit variant, `v_wmma_i32_16x16x16_iu4` at 2x the INT8 rate) could
shorten the Q·K^T half further but costs accuracy and is not in upstream's current API, so it is not
implemented. FP8 P·V is opt-in (see accuracy). SageAttention3 is not portable (above).

See [`ROCM_NOTES.md`](./ROCM_NOTES.md) for kernel entry points and design notes, `bench/rocm/` and
`example/rocm/wan_e2e.py` to reproduce.

## Installation
### Base environment
+ `python>=3.9`   , `torch>=2.3.0`  , `triton>=3.0.0` 
- `CUDA`:
  + `>=12.8` for Blackwell or SageAttention2++
  + `>=12.4` for fp8 support on Ada
  + `>=12.3` for fp8 support on Hopper
  + `>=12.0` for Ampere
+ `flash-attn` for benchmarking

### Install Package

For SageAttention V1 in Triton (slower than SageAttention V2/V2++/V3), refer to [SageAttention-1](https://github.com/thu-ml/SageAttention/tree/sageattention-1) branch and install using pip: `pip install sageattention==1.0.6`

To use SageAttention 2.2.0 (containing SageAttention2++), you can install using pip:
```
pip install sageattention==2.2.0 --no-build-isolation
```

**Or** you can compile from source:
```
git clone https://github.com/thu-ml/SageAttention.git
cd SageAttention 
export EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=32 # Optional
python setup.py install
```

To benchmark the speed against FlashAttention3, please compile FlashAttention3 from source:
```
git clone https://github.com/Dao-AILab/flash-attention.git --recursive
git checkout b7d29fb3b79f0b78b1c369a52aaa6628dabfb0d7 # 2.7.2 release
cd hopper
python setup.py install
```

## How to Use
```python
from sageattention import sageattn
attn_output = sageattn(q, k, v, tensor_layout="HND", is_causal=False)
```
+ `q, k, v` are **FP16/BF16** dtype with the shape `(batch_size, head_num, seq_len, head_dim)` using default `tensor_layout="HND"`. For shape `(batch_size, seq_len, head_num, head_dim)`, set `tensor_layout="NHD"`. 
+ `is_causal` determines the use of a causal mask.

### Available APIs:
+ `sageattn`: Automatically selects the optimal kernel based on the GPU to achieve a good performance-accuracy trade-off.
+ `sageattn_qk_int8_pv_fp16_triton`: INT8 quantization for $QK^\top$ and FP16 for $PV$ using Triton backend.
+ `sageattn_qk_int8_pv_fp16_cuda`: INT8 quantization for $QK^\top$ and FP16 for $PV$ using CUDA backend.
+ `sageattn_qk_int8_pv_fp8_cuda`: INT8 quantization for $QK^\top$ and FP8 for $PV$ using CUDA backend. (Note that setting `pv_accum_dtype=fp32+fp16` corresponds to SageAttention2++.)
+ `sageattn_qk_int8_pv_fp8_cuda_sm90`: INT8 quantization for $QK^\top$ and FP8 for $PV$ using CUDA backend, specifically optimized for Hopper GPUs.
+ `sageattn_varlen`: INT8 quantization for $QK^\top$ and FP16 for $PV$ using Triton backend. Support for varying sequence lengths within the same batch.

For optimal speed and accuracy performance on custom devices and models, we strongly recommend referring to the [this file](./sageattention/core.py) for detailed guidance.

> **Note:**
Support for different sequence lengths between `q` and `k,v` and `group-query attention` is available.


### Plug-and-play Example

We can replace `scaled_dot_product_attention` easily. 
We will take [CogvideoX](https://huggingface.co/zai-org/CogVideoX-2b) as an example:

Add the following codes and run
```diff
import torch.nn.functional as F

+ from sageattention import sageattn
+ F.scaled_dot_product_attention = sageattn

```

Specifically,

```bash
cd example
python cogvideox_infer.py --model cogvideox-2b --compile --attention_type sage
```

**You can get a lossless video in** `./example/videos/<model>/<attention_type>/` **faster than by using** `--attention_type sdpa`. More examples and guidance can be found under the `example/` directory.

> **Note:** Not all models works with `F.scaled_dot_product_attention = sageattn`. Technically, you should replace the original Attention by modifying the `Attention Class` of the target model. For image and video models, we suggest only replacing the attention in DiT (see `example/modify_mochi.py` for detail).

### Kernel Benchmarking
We provide a benchmarking script to compare the speed of different kernels including SageAttention, FlashAttention2 and FlashAttention3. Please refer to the `benchmark/` directory for more details.
 
## Performance
### Speed of Kernels

`8+8` means the kernel with INT8 quantization for $QK^\top$ and FP8 quantization for $PV$. `8+16` uses FP16 with FP16 accumulator for $PV$.

![Local Image](./assets/5090_sageattn2++.png)

![Local Image](./assets/4090_sageattn2++.png)

![Local Image](./assets/4090_hd128.png)

![Local Image](./assets/L20_hd128.png)

![Local Image](./assets/H100_hd128.png)

![Local Image](./assets/H20_hd128.png)

![Local Image](./assets/A100_hd128.png)

![Local Image](./assets/3090_hd128.png)

> **Note:** The TOPS results refer only to the Attention Kernel, excluding the quantization and smoothing.

### End-to-end Performance
#### **End-to-End Accuracy:**

![Local Image](./assets/22.png)

![Local Image](./assets/23.png)

![Local Image](./assets/24.png)

![Local Image](./assets/25.png)

#### **End-to-End Speedup:**

![Local Image](./assets/26.png)
*Note: SageAttention2++ achieves higher speed.*

## Citation
**If you use this code or find our work valuable, please cite:**
```
@inproceedings{zhang2025sageattention,
  title={SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration}, 
  author={Zhang, Jintao and Wei, Jia and Zhang, Pengle and Zhu, Jun and Chen, Jianfei},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
@inproceedings{zhang2024sageattention2,
  title={Sageattention2: Efficient attention with thorough outlier smoothing and per-thread int4 quantization},
  author={Zhang, Jintao and Huang, Haofeng and Zhang, Pengle and Wei, Jia and Zhu, Jun and Chen, Jianfei},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2025}
}
@article{zhang2025sageattention2++,
  title={Sageattention2++: A more efficient implementation of sageattention2},
  author={Zhang, Jintao and Xu, Xiaoming and Wei, Jia and Huang, Haofeng and Zhang, Pengle and Xiang, Chendong and Zhu, Jun and Chen, Jianfei},
  journal={arXiv preprint arXiv:2505.21136},
  year={2025}
}
@article{zhang2025sageattention3,
  title={SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training},
  author={Zhang, Jintao and Wei, Jia and Zhang, Pengle and Xu, Xiaoming and Huang, Haofeng and Wang, Haoxu and Jiang, Kai and Zhu, Jun and Chen, Jianfei},
  journal={arXiv preprint arXiv:2505.11594},
  year={2025}
}
```
