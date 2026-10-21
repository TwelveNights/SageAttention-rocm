# SageAttention on ROCm -- notes for kernel reuse

Status: **working and validated** on MI300X (gfx942), RX 7800 XT (gfx1101) and Radeon PRO W7900 (gfx1100).
`pytest tests/rocm`: 116 passed on the W7900, 70 + 70 gfx11-only skips on MI300X, 115 on the 7800 XT (the
torch.compile test is skipped on Windows). Install, ComfyUI usage, numbers, accuracy and end-to-end results: [README.md](README.md).
Pure Python + Triton, plus one HIP kernel for RDNA3 compiled at runtime with hipRTC (no build step, no
hipcc/MSVC). Tests: `pytest tests/rocm`; everything at once: `python scripts/rocm/validate.py`.

## Entry points

Public API (unchanged names; on a ROCm torch they dispatch to `sageattention/rocm/`):

- `sageattention.sageattn(q, k, v, tensor_layout="HND"|"NHD", is_causal=False, sm_scale=None, return_lse=False)`
- `sageattn_qk_int8_pv_fp16_triton(..., attn_mask=None, smooth_k=True)` -- per-block Q scales (128 rows), FP16 PV
- `sageattn_qk_int8_pv_fp16_cuda(..., qk_quant_gran="per_warp"|"per_thread")` -- same kernels, finer Q scales
- `sageattn_qk_int8_pv_fp8_cuda(...)`, `sageattn_qk_int8_pv_fp8_cuda_sm90(...)` -- FP8 PV on gfx942, FP16 PV on gfx11
- `sageattn_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, is_causal=False)`

One-stop ROCm function: `sageattention.rocm.sageattn_rocm(q, k, v, tensor_layout, is_causal, sm_scale,
return_lse, attn_mask, smooth_k, qk_quant_gran="per_warp", pv_dtype="fp16"|"fp8", config=None)`. On gfx11 it
runs the native kernel when applicable (no mask / varlen / fp8), else Triton; `_native=False` forces Triton.

### Building blocks (for SpargeAttn / Sparse-VideoGen / Radial Attention ports)

- `sageattention.rocm.quant.seq_mean(k, layout)` -> fp32 `[B,Hk,D]` (deterministic; fast for HND too).
- `sageattention.rocm.quant.quant_qk_int8(q, k, km, sm_scale, BLKQ, BLKK, tensor_layout, nq_pad=None, nk_pad=None)` ->
  `q_int8 [B,Hq,Nq,Dp]`, `q_scale [B,Hq,ceil(Nq/BLKQ)]` (fp32), `k_int8 [B,Hk,Nk,Dp]`, `k_scale [B,Hk,ceil(Nk/BLKK)]`.
  Q is pre-multiplied by `sm_scale*log2(e)` (scores come out in the exp2 domain), K has `km` subtracted.
  Outputs are head-major contiguous, head dim zero-padded to a power of two `Dp`; `n*_pad` zero-pads rows.
  Varlen: pass `cu_seqlens_*`, `max_seqlen_*` (`[total,H,D]` in, `[H,total,Dp]` out + per-sequence block offsets).
- `sageattention.rocm.quant.quant_v_fp8(v, tensor_layout, fp8_dtype)` -> `v_fp8 [B,Hk,N,Dp]`, `v_scale [B,Hk,Dp]`.
- `sageattention.rocm.quant.transpose_v_fp16(v, layout, n_pad)` -> fp16 `[B,Hk,Dp,n_pad]`, keys permuted per
  16-block as `[0,2,..,14,1,3,..,15]` (the native kernel's P.V operand order).
- `sageattention.rocm.attn.sage_attn_fwd(q_int8, k_int8, v, q_scale, k_scale, out, QBLK=, KBLK=, config=,
  tensor_layout=, is_causal=, attn_mask=, v_scale=, return_lse=, cu_seqlens_*...)` -- the Triton kernel
  `_sage_fwd`. `KBLK` must be a multiple of `config["BLOCK_N"]`; `QBLK` is free (Q scales are loaded per row).
  V and `out` are read/written through strides in the caller's layout. Returns the log2-domain LSE if asked.
  **Block-sparse hook:** the KV loop is `_inner(acc, l_i, m_i, ..., lo, hi, ..., MASKED, ...)`, a plain
  `@triton.jit` function called twice (unmasked tiles, then the masked causal-diagonal / ragged tail). A sparse
  variant only needs to iterate over a list of kv-tile indices instead of `range(lo, hi, BLOCK_N)`; everything
  else (scales, online softmax, FP8 P scaling) carries over. `default_config(head_dim, is_causal, pv_fp8)` gives
  the tuned tiles per arch.
- `sageattention.rocm.native.sage.sageattn_native(q, k, v, tensor_layout, is_causal, sm_scale, return_lse,
  smooth_k, qk_quant_gran, waves=None)` -- the gfx11 HIP kernel (`native/sage_gfx11.hip`, kernels
  `sage_d{64,128}_tk4_w{4,8}_{bf16,fp16}_{0,1}`), dense only, causal supported. A block-sparse version would
  skip stages in its main loop (each stage = 64 keys; the global->LDS prefetch of the next stage would need
  the next *active* stage index instead of `it + 1`).

## Hardware notes

- gfx942 (MI300X): int8 and fp8 MFMA both run at 2x the fp16 rate. FP8 is `float8_e4m3fnuz` (max 240),
  not OCP e4m3fn. P is scaled by 2^7 before the FP8 cast (free: folded into the exp2 offset); l carries the
  same factor. Best tiles: `BLOCK_M=128, BLOCK_N=64` (fp16 PV) / `BLOCK_N=128` + `schedule_hint="attention"`
  (fp8 PV), `num_warps=4, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=32`.
- gfx11 (RDNA3): WMMA iu8 runs at the *same* rate as fp16 (int4 is 2x); there is no FP8. So INT8 QK gives no
  matrix-core speedup there; the gain comes from the native kernel (S^T = K Q^T / O^T = V^T P^T layout trick:
  softmax state is lane-local, one v_permlanex16 exchange per value; magic-number int->float; lazy rescaling;
  pkrtz P conversion with bias compensation). Triton reaches ~42 TFLOP/s on the 7800 XT, the native kernel ~56.
- SageAttention3 (Blackwell FP4 microscaling, `sageattention3_blackwell/`) is CUDA/sm100+-only; on ROCm
  `sageattn3_blackwell()` falls back to `sageattention.sageattn`.

## Numbers

See [README.md](README.md#results).
