# SageAttention on ROCm -- notes for kernel reuse

Status: **working** (INT8 Q·K^T + FP16 P·V on gfx942 and gfx11; FP8 P·V on gfx942).
Pure Python + Triton, no build step.

## Entry points

Public API (unchanged names; on a ROCm torch they dispatch to `sageattention/rocm/`):

- `sageattention.sageattn(q, k, v, tensor_layout="HND"|"NHD", is_causal=False, sm_scale=None, return_lse=False)`
- `sageattn_qk_int8_pv_fp16_triton(..., attn_mask=None, smooth_k=True)` -- per-block Q scales (128 rows), FP16 PV
- `sageattn_qk_int8_pv_fp16_cuda(..., qk_quant_gran="per_warp"|"per_thread")` -- same kernel, finer Q scales
- `sageattn_qk_int8_pv_fp8_cuda(...)`, `sageattn_qk_int8_pv_fp8_cuda_sm90(...)` -- FP8 PV on gfx942, FP16 PV on gfx11
- `sageattn_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, is_causal=False)`

Lower level (for SpargeAttn / Sparse-VideoGen / Radial Attention ports):

- `sageattention.rocm.sageattn_rocm(q, k, v, tensor_layout, is_causal, sm_scale, return_lse, attn_mask, smooth_k, qk_quant_gran, pv_dtype="fp16"|"fp8", config=None)`
- `sageattention.rocm.quant.quant_qk_int8(q, k, km, sm_scale, BLKQ, BLKK, tensor_layout)` ->
  `q_int8 [B,Hq,Nq,Dp]`, `q_scale [B,Hq,ceil(Nq/BLKQ)]` (fp32), `k_int8 [B,Hk,Nk,Dp]`, `k_scale [B,Hk,ceil(Nk/BLKK)]`.
  Q is pre-multiplied by `sm_scale*log2(e)` (scores come out in the exp2 domain), K has `km` (fp32 `[B,Hk,D]`)
  subtracted. Outputs are head-major contiguous, head dim zero-padded to a power of two `Dp`.
- `sageattention.rocm.quant.quant_v_fp8(v, tensor_layout, fp8_dtype)` -> `v_fp8 [B,Hk,N,Dp]`, `v_scale [B,Hk,Dp]`.
- `sageattention.rocm.attn.sage_attn_fwd(q_int8, k_int8, v, q_scale, k_scale, out, QBLK=, KBLK=, config=, tensor_layout=, is_causal=, attn_mask=, v_scale=, return_lse=, cu_seqlens_*...)`
  -- the Triton kernel `_sage_fwd`. `KBLK` must be a multiple of `config["BLOCK_N"]`; `QBLK` is free
  (Q scales are loaded per row). V and `out` are read/written through strides in the caller's layout.
  The inner loop `_inner(...)` is a plain `@triton.jit` function taking `lo, hi` kv bounds and
  `MASKED` -- a block-sparse variant only needs to iterate over a list of kv tiles instead of `range(lo, hi)`.

## Hardware notes

- gfx942 (MI300X): int8 and fp8 MFMA both run at 2x the fp16 rate. FP8 is `float8_e4m3fnuz` (max 240),
  not OCP e4m3fn. P is scaled by 2^7 before the FP8 cast (free: folded into the exp2 offset).
- gfx11 (RDNA3): WMMA iu8 runs at the *same* rate as fp16 (int4 is 2x); there is no FP8.
- SageAttention3 (Blackwell FP4 microscaling, `sageattention3_blackwell/`) is CUDA/sm100+-only; not portable.

## Numbers

See README "ROCm" section (kept up to date).
