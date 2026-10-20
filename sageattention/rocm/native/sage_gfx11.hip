// SageAttention forward for RDNA3 (gfx11, wave32): INT8 Q.K^T on v_wmma_i32_16x16x16_iu8, FP16 P.V on
// v_wmma_f32_16x16x16_f16, FP32 online softmax in the exp2 domain. Compiled at runtime with hipRTC (see build.py).
//
// Same maths as _sage_fwd in ../attn.py (the Triton kernel): Q and K arrive as per-block INT8 with fp32 scales
// (Q pre-multiplied by sm_scale * log2 e, K smoothed), so score(q, k) = qs[q] * ks[k] * (Q_q . K_k) and
// O = softmax2(scores) V.
//
// Layout trick. Each wave owns 16 queries and computes the TRANSPOSED products
//     S^T = K Q^T   (A = K tile from LDS, B = this wave's Q, held in registers for the whole kernel)
//     O^T = V^T P^T (A = V^T tile from LDS, B = P^T built in registers)
// With the WMMA C/D layout (lane (r16, hl), element i <-> row 2i + hl, column r16), lane (r16, hl) then holds
// query r16's scores for keys 2i + hl, and query r16's output for head-dim channels 2i + hl: the softmax
// statistics (max, sum, rescale factor) of a query live in its two lanes r16 and r16 + 16 and never need a
// cross-lane broadcast; only the row max and the P operand are exchanged between the two half-waves, with
// v_permlanex16. The P^T operand of a 16-key tile is laid out as [even keys | odd keys]: the even keys come from
// the hl = 0 lane, the odd ones from the hl = 1 lane. V^T is pre-permuted to the same key order in global memory
// (host side, quant pass), so the A operand is a plain 32-byte LDS read.
//
// int -> float without v_cvt_f32_i32: the QK WMMA chain starts from C = the int32 bit pattern of 1.5 * 2^23;
// |Q.K| <= 127^2 * 128 < 2^22, so the result's bits read as the float 1.5 * 2^23 + Q.K exactly, and the score
// minus the running max is one FMA: fma(bits, s, -(1.5 * 2^23 * s + m)). The rounding of that constant (relative
// 2^-24 of 1.5 * 2^23 * s, i.e. <= ~1 unit of the integer dot product times s) is far below the INT8
// quantization error of the dot product itself (hundreds of units).
//
// Inputs (prepared by native/sage.py):
//   q  [B, Hq, Nq_pad, D] int8, k [B, Hk, Nk_pad, D] int8  (rows past Nq / Nk are zero)
//   vt [B, Hk, D, Nk_pad] fp16, key order permuted within each 16-key block (zeros past Nk)
//   qs [B, Hq, Nq_pad / QBLK] fp32, ks [B, Hk, Nk_pad / KBLK] fp32 (KBLK a multiple of the kv tile)
//   o  output (bf16 or fp16) with element strides (NHD or HND), lse [B, Hq, Nq] fp32 (log2 domain) or null
#include "hip_prelude.h"

struct AttnArgs {
  const signed char* q;
  const signed char* k;
  const u16* vt;
  const float* qs;
  const float* ks;
  u16* o;
  float* lse;
  long long sqb, sqh, skb, skh, svb, svh, sob, soh, son, sqsb, sqsh, sksb, sksh, slb, slh;
  int nq, nk, nk_pad, groups, qblk_shift, kblk, causal, dv;  // dv: real head dim of V / O (<= D)
};

#define MAGIC_I 0x4B400000
#define MAGIC_F 12582912.0f

// Lazy rescaling (as FlashAttention-4): the running max m only moves when a tile's max exceeds it by more than
// LAZY_TAU (log2 units), so p = 2^(score - m) stays <= 2^LAZY_TAU (exact maths; fp16 P has room to 65504) and the
// O rescale -- D/2 multiplies per lane per tile -- is skipped (wave-uniform branch) unless some lane moved its max.
#ifndef SAGE_LAZY_TAU
#define SAGE_LAZY_TAU 8.0f
#endif
// P -> fp16 with v_cvt_pkrtz_f16_f32 (round toward zero, one instruction per pair instead of two conversions and
// a pack). Its mean relative truncation, 2^-11 * E[1 / (1 + mantissa)] = 3.5e-4 for log-uniform mantissas, is
// compensated by computing p * (1 + 3.5e-4) (a constant added to the exponent, free): the row sum l is taken from
// the same scaled p, so the factor cancels in O = sum(p v) / l and only the rounding direction changes.
#ifndef SAGE_PKRTZ
#define SAGE_PKRTZ 1
#endif
#define RTZ_LOG2_BIAS 5.08e-4f

static __device__ __forceinline__ int f2i(float x) { return __builtin_bit_cast(int, x); }

// the value of lane ^ 16 (the other half-wave, same r16)
static __device__ __forceinline__ int xhalf(int x) {
  return (int)__builtin_amdgcn_permlanex16((unsigned)x, (unsigned)x, 0x76543210u, 0xfedcba98u, false, false);
}
static __device__ __forceinline__ float xhalf_f(float x) { return i2f(xhalf(f2i(x))); }

static __device__ __forceinline__ unsigned pack_h2(float a, float b) {
#if SAGE_PKRTZ
  return __builtin_bit_cast(unsigned, __builtin_amdgcn_cvt_pkrtz(a, b));
#else
  return (unsigned)__builtin_bit_cast(u16, (_Float16)a) | ((unsigned)__builtin_bit_cast(u16, (_Float16)b) << 16);
#endif
}

struct OutBF16 {
  static __device__ __forceinline__ u16 cvt(float f) { return f32_to_bf16(f); }
};
struct OutFP16 {
  static __device__ __forceinline__ u16 cvt(float f) { return __builtin_bit_cast(u16, (_Float16)f); }
};

template <int D, int TK, int WAVES, class TO, bool LSE>
static __device__ __forceinline__ void sage_fwd(const AttnArgs& a) {
  constexpr int DC = D / 16;        // 16-wide head-dim chunks
  constexpr int BN = 16 * TK;       // keys per main-loop stage
  constexpr int KP = D + 16;        // LDS bytes per K row (16 bytes of padding against bank conflicts)
  constexpr int VP = BN * 2 + 16;   // LDS bytes per V^T row
  constexpr int V_OFF = BN * KP;
  constexpr int BUF = V_OFF + D * VP;
  constexpr int NT = WAVES * 32;
  constexpr int K_CH = BN * D / 16;  // 16-byte chunks per stage
  constexpr int V_CH = D * BN * 2 / 16;
  constexpr int VCPR = BN * 2 / 16;  // chunks per V^T row
  constexpr int K_PER = (K_CH + NT - 1) / NT, V_PER = (V_CH + NT - 1) / NT;
  constexpr int PV_GROUP = 4;       // head-dim tiles per scheduling group of the P.V WMMAs
  __shared__ __attribute__((aligned(16))) unsigned char smem[2 * BUF];

  const int tid = tid_x();
  const int lane = tid & 31;
  const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
  const int r16 = lane & 15, hl = lane >> 4;
  const int h = __builtin_amdgcn_workgroup_id_y(), b = __builtin_amdgcn_workgroup_id_z();
  const int hk = h / a.groups;
  const int q0 = (int)bid_x() * (16 * WAVES) + wave * 16;
  const int qi = q0 + r16;  // this lane's query

  // Q, the B operand of every QK WMMA: lane r16 holds query qi's D int8 values
  v4i qf[DC];
  {
    const signed char* qrow = a.q + b * a.sqb + h * a.sqh + (long long)qi * D;
#pragma unroll
    for (int c = 0; c < DC; c++) qf[c] = *(const v4i*)(qrow + 16 * c);
  }
  const float qsc = a.qs[b * a.sqsb + h * a.sqsh + (qi >> a.qblk_shift)];

  const unsigned char* kg = (const unsigned char*)(a.k + b * a.skb + hk * a.skh);
  const unsigned char* vg = (const unsigned char*)(a.vt + b * a.svb + hk * a.svh);
  const float* ksp = a.ks + b * a.sksb + hk * a.sksh;

  // per-thread copy assignments (byte offsets): stage tiles are global -> registers -> LDS
  int k_src[K_PER], k_dst[K_PER], v_src[V_PER], v_dst[V_PER];
#pragma unroll
  for (int q = 0; q < K_PER; q++) {
    int c = tid + q * NT;
    c = c < K_CH ? c : tid;  // spare lanes repeat their first chunk
    k_src[q] = c * 16;       // a K tile is BN * D contiguous bytes
    k_dst[q] = (c / DC) * KP + (c % DC) * 16;
  }
#pragma unroll
  for (int q = 0; q < V_PER; q++) {
    int c = tid + q * NT;
    c = c < V_CH ? c : tid;
    const int d = c / VCPR, part = c % VCPR;
    v_src[q] = d * a.nk_pad * 2 + part * 16;
    v_dst[q] = V_OFF + d * VP + part * 16;
  }
  struct Regs {
    v4i k[K_PER];
    v4i v[V_PER];
  };
  auto gload = [&](int it, Regs& r) {
    const unsigned char* kt = kg + (long long)it * BN * D;
    const unsigned char* vt = vg + (long long)it * BN * 2;
#pragma unroll
    for (int q = 0; q < K_PER; q++) r.k[q] = *(const v4i*)(kt + k_src[q]);
#pragma unroll
    for (int q = 0; q < V_PER; q++) r.v[q] = *(const v4i*)(vt + v_src[q]);
  };
  auto sstore = [&](int buf, const Regs& r) {
    unsigned char* base = smem + buf * BUF;
#pragma unroll
    for (int q = 0; q < K_PER; q++) *(v4i*)(base + k_dst[q]) = r.k[q];
#pragma unroll
    for (int q = 0; q < V_PER; q++) *(v4i*)(base + v_dst[q]) = r.v[q];
  };

  v8f oacc[DC];  // O^T tiles: lane (r16, hl), element i = O[qi, 16 dt + 2 i + hl]
#pragma unroll
  for (int dt = 0; dt < DC; dt++) oacc[dt] = (v8f){};
  float m_i = -__builtin_inff(), l_i = 0.0f;  // l_i: this lane's half of the keys

  v8i magic = {MAGIC_I, MAGIC_I, MAGIC_I, MAGIC_I, MAGIC_I, MAGIC_I, MAGIC_I, MAGIC_I};
  // opaque to the optimizer: clang 22 (ROCm 7.2) miscompiles per-element extracts of a WMMA result whose C
  // operand is a constant splat (see nunchaku's w4a4_gfx11.hip)
  __asm__ volatile("" : "+v"(magic));

  auto compute = [&](int buf, int n0, bool tail) {
    const unsigned char* base = smem + buf * BUF;
    // S^T = K Q^T for TK 16-key tiles, two tiles at a time: two independent WMMA chains interleave (a chain's
    // WMMAs depend on each other), and the sched_barrier keeps the scheduler from hoisting every tile's LDS reads
    // at once (register pressure)
    v8i sacc[TK];
#pragma unroll
    for (int kp = 0; kp < TK; kp += 2) {
      sacc[kp] = magic;
      sacc[kp + 1] = magic;
#pragma unroll
      for (int c = 0; c < DC; c++)
#pragma unroll
        for (int kk = 0; kk < 2; kk++) {
          const v4i ka = *(const v4i*)(base + ((kp + kk) * 16 + r16) * KP + c * 16);
          sacc[kp + kk] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, ka, true, qf[c], sacc[kp + kk], false);
        }
      __builtin_amdgcn_sched_barrier(0);
    }
    // keys this lane holds: n0 + 16 kt + 2 i + hl. Invalid ones (past nk, or after the query when causal) get
    // INT_MIN for the max and p = 0.
    bool valid[TK][8];
#pragma unroll
    for (int kt = 0; kt < TK; kt++)
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const int key = n0 + kt * 16 + 2 * i + hl;
        valid[kt][i] = !tail || (key < a.nk && (!a.causal || key <= qi));
      }
    int mx = -2147483647 - 1;
#pragma unroll
    for (int kt = 0; kt < TK; kt++)
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const int v = valid[kt][i] ? sacc[kt][i] : (-2147483647 - 1);
        mx = v > mx ? v : mx;
      }
    {
      const int o = xhalf(mx);
      mx = o > mx ? o : mx;
    }
    const float s = qsc * ksp[n0 / a.kblk];  // dequantization scale of every score of this lane, > 0
    // a row with no valid key in this tile (causal: queries before the tile) keeps its max
    const float mt = mx == (-2147483647 - 1) ? -__builtin_inff() : (float)(mx - MAGIC_I) * s;
    const float m_new = mt > m_i + SAGE_LAZY_TAU ? mt : m_i;  // (m_i = -inf: takes any finite mt)
    const float m_use = m_new == -__builtin_inff() ? 0.0f : m_new;
    const float alpha = __builtin_amdgcn_exp2f(m_i - m_use);  // exactly 1 when m did not move
#if SAGE_PKRTZ
    const float cst = RTZ_LOG2_BIAS - __builtin_fmaf(MAGIC_F, s, m_use);
#else
    const float cst = -__builtin_fmaf(MAGIC_F, s, m_use);
#endif
    float rs = 0.0f;
    unsigned pw[TK][4];
#pragma unroll
    for (int kt = 0; kt < TK; kt++)
#pragma unroll
      for (int j = 0; j < 4; j++) {
#ifdef SAGE_DBG_NOEXP  // perf experiments only (wrong results)
        float p0 = __builtin_fmaf(i2f(sacc[kt][2 * j]), s, cst);
        float p1 = __builtin_fmaf(i2f(sacc[kt][2 * j + 1]), s, cst);
#else
        float p0 = __builtin_amdgcn_exp2f(__builtin_fmaf(i2f(sacc[kt][2 * j]), s, cst));
        float p1 = __builtin_amdgcn_exp2f(__builtin_fmaf(i2f(sacc[kt][2 * j + 1]), s, cst));
#endif
        if (tail) {
          p0 = valid[kt][2 * j] ? p0 : 0.0f;
          p1 = valid[kt][2 * j + 1] ? p1 : 0.0f;
        }
        rs += p0 + p1;
        pw[kt][j] = pack_h2(p0, p1);
      }
    l_i = __builtin_fmaf(l_i, alpha, rs);
    m_i = m_new;
    if (__builtin_amdgcn_ballot_w32(alpha != 1.0f)) {
#pragma unroll
      for (int dt = 0; dt < DC; dt++)
#pragma unroll
        for (int i = 0; i < 8; i++) oacc[dt][i] *= alpha;
    }
    // O^T += V^T P^T, one 16-key tile at a time: its P^T operand (replacing the scores in registers), then the
    // head-dim tiles two at a time (two independent accumulator chains in flight). The sched_barriers bound how
    // many V^T operands the scheduler hoists (register pressure).
#pragma unroll
    for (int kt = 0; kt < TK; kt++) {
      v8i bop;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const int own = (int)pw[kt][j], other = xhalf(own);
        bop[j] = hl ? other : own;      // even keys: from the hl = 0 lane
        bop[4 + j] = hl ? own : other;  // odd keys: from the hl = 1 lane
      }
      const v16h pb = __builtin_bit_cast(v16h, bop);
#pragma unroll
      for (int dp = 0; dp < DC; dp += PV_GROUP) {
#pragma unroll
        for (int dd = 0; dd < PV_GROUP; dd++) {
#ifdef SAGE_DBG_NOPV  // perf experiments only (wrong results)
          if (dd == 0 && dp == 0) oacc[0][0] += __builtin_bit_cast(float, bop[0]);
#else
          const v16h va = *(const v16h*)(base + V_OFF + ((dp + dd) * 16 + r16) * VP + kt * 32);
          oacc[dp + dd] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(va, pb, oacc[dp + dd]);
#endif
        }
        __builtin_amdgcn_sched_barrier(0);
      }
    }
  };

  // kv range: causal rows of this workgroup only need keys < its last query + 1
  int nk_eff = a.nk;
  if (a.causal) {
    const int q_end = (int)bid_x() * (16 * WAVES) + 16 * WAVES;
    nk_eff = q_end < nk_eff ? q_end : nk_eff;
  }
  const int NIT = (nk_eff + BN - 1) / BN;
  // causal: tiles that start after the workgroup's first query need masking
  int n_full = NIT - 1;
  if (a.causal) {
    const int qwg0 = (int)bid_x() * (16 * WAVES);
    const int f = (qwg0 + 1) / BN;  // tiles [0, f) end at or before key qwg0
    n_full = f < n_full ? f : n_full;
  }

  Regs r0;
  gload(0, r0);
  sstore(0, r0);
  syncthreads();
  // Software pipeline: the next stage's global loads go out before this stage's compute and are written to LDS
  // after it (see w4a4_gfx11.hip); the sched_barriers pin that order, which needs branch-free loop bodies -- hence
  // one loop for unmasked tiles and one for masked ones (causal diagonal, and always the last tile).
  auto stage = [&](int it, bool masked) {
    Regs rn;
    gload(it + 1 < NIT ? it + 1 : it, rn);  // the last stage re-loads itself into the idle buffer (branch-free)
    __builtin_amdgcn_sched_barrier(0);
    compute(it & 1, it * BN, masked);
    __builtin_amdgcn_sched_barrier(0);
    sstore((it + 1) & 1, rn);
    syncthreads();
  };
  int it = 0;
  for (; it < n_full; it++) stage(it, false);
  for (; it < NIT; it++) stage(it, true);

  // epilogue: combine the two half-waves' sums, normalize, store
  const float l_tot = l_i + xhalf_f(l_i);
  const float inv = l_tot > 0.0f ? 1.0f / l_tot : 0.0f;
  if (qi < a.nq) {
    u16* orow = a.o + b * a.sob + h * a.soh + (long long)qi * a.son;
    if (a.dv == D) {
#pragma unroll
      for (int dt = 0; dt < DC; dt++)
#pragma unroll
        for (int i = 0; i < 8; i++) orow[dt * 16 + 2 * i + hl] = TO::cvt(oacc[dt][i] * inv);
    } else {
#pragma unroll
      for (int dt = 0; dt < DC; dt++)
#pragma unroll
        for (int i = 0; i < 8; i++) {
          const int d = dt * 16 + 2 * i + hl;
          if (d < a.dv) orow[d] = TO::cvt(oacc[dt][i] * inv);
        }
    }
#if SAGE_PKRTZ
    if (LSE && hl == 0) a.lse[b * a.slb + h * a.slh + qi] = m_i + __builtin_log2f(l_tot) - RTZ_LOG2_BIAS;
#else
    if (LSE && hl == 0) a.lse[b * a.slb + h * a.slh + qi] = m_i + __builtin_log2f(l_tot);
#endif
  }
}

// Instantiations: sage_d<D>_tk<TK>_w<WAVES>_<bf16|fp16>_<lse 0|1>; the host table (sage.py: VARIANTS) must match.
#define SAGE_KERNEL(D, TK, WAVES, DT, TO, LSE)                                                       \
  EXTC __global__ void __launch_bounds__(WAVES * 32) sage_d##D##_tk##TK##_w##WAVES##_##DT##_##LSE(AttnArgs a) { \
    sage_fwd<D, TK, WAVES, TO, LSE>(a);                                                             \
  }
#define SAGE_VARIANTS(D, TK, WAVES)               \
  SAGE_KERNEL(D, TK, WAVES, bf16, OutBF16, 0)     \
  SAGE_KERNEL(D, TK, WAVES, bf16, OutBF16, 1)     \
  SAGE_KERNEL(D, TK, WAVES, fp16, OutFP16, 0)     \
  SAGE_KERNEL(D, TK, WAVES, fp16, OutFP16, 1)

SAGE_VARIANTS(128, 4, 8)
SAGE_VARIANTS(128, 4, 4)
SAGE_VARIANTS(64, 4, 8)
SAGE_VARIANTS(64, 4, 4)
#ifdef SAGE_EXTRA_VARIANTS  // tuning experiments
SAGE_VARIANTS(128, 2, 8)
SAGE_VARIANTS(128, 2, 4)
SAGE_VARIANTS(64, 8, 8)
SAGE_VARIANTS(64, 8, 4)
#endif
