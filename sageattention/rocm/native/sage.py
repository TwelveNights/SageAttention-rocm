"""
Host side of the native RDNA3 SageAttention kernel (``sage_gfx11.hip``).

:func:`sageattention.rocm.api.sageattn_rocm` calls :func:`try_sageattn` first;
it runs the native kernel and returns the output, or returns ``None`` to let
the Triton kernel run:

- ``SAGEATTN_NATIVE=0`` disables the native kernel;
- unset (or ``auto``) uses it on gfx11 GPUs only;
- ``SAGEATTN_NATIVE=1`` tries it on any AMD GPU and *raises* when it cannot be
  built, loaded or launched (tests / debugging).

Calls it does not support (:func:`unsupported_reason`: attention masks,
varlen, FP8 P.V, head dims other than 64/128 after padding, ...) always go to
Triton. The first supported call on a device compiles the kernels with hipRTC
(or loads them from the cache, see :mod:`.build`) and checks every variant
against the Triton kernel; a failure there, or a launch error later, disables
the native path on that device for the rest of the process with one warning.
"""

import ctypes
import os
import threading
import warnings

import torch

from . import build

__all__ = ["VARIANTS", "mode", "status", "sageattn_native", "try_sageattn", "unsupported_reason"]

_SOURCE = "sage_gfx11.hip"
_HEADERS = ["hip_prelude.h"]
_OPTIONS = ["-O3", "-std=c++17"] + os.environ.get("SAGEATTN_NATIVE_OPTS", "").split()  # extra -D... for experiments

# (D, TK, WAVES): head dim, 16-key tiles per stage, waves (x 16 queries) per workgroup. Must match SAGE_VARIANTS in
# sage_gfx11.hip.
VARIANTS = [(128, 4, 8), (128, 4, 4), (64, 4, 8), (64, 4, 4)]
KBLK = 64  # K-scale block (a multiple of the 64-key stage)


def _tk() -> int:
    """16-key tiles per stage; SAGEATTN_NATIVE_TK (with -DSAGE_EXTRA_VARIANTS) is for tuning experiments."""
    return int(os.environ.get("SAGEATTN_NATIVE_TK", "4"))


def _kblk() -> int:
    return max(KBLK, 16 * _tk())


class _AttnArgs(ctypes.Structure):
    """Mirror of ``struct AttnArgs`` in sage_gfx11.hip."""

    _fields_ = (
        [(n, ctypes.c_void_p) for n in ("q", "k", "vt", "qs", "ks", "o", "lse")]
        + [
            (n, ctypes.c_longlong)
            for n in ("sqb", "sqh", "skb", "skh", "svb", "svh", "sob", "soh", "son")
            + ("sqsb", "sqsh", "sksb", "sksh", "slb", "slh")
        ]
        + [(n, ctypes.c_int) for n in ("nq", "nk", "nk_pad", "groups", "qblk_shift", "kblk", "causal", "dv")]
    )


def mode() -> str:
    """``"off"``, ``"auto"`` or ``"force"``, from ``SAGEATTN_NATIVE`` (read on every call)."""
    v = os.environ.get("SAGEATTN_NATIVE", "").strip().lower()
    if v in ("0", "off", "false", "no"):
        return "off"
    if v in ("1", "on", "true", "yes", "force"):
        return "force"
    return "auto"


def _device_arch(device: int) -> str:
    if torch.version.hip is None:
        return ""
    return getattr(torch.cuda.get_device_properties(device), "gcnArchName", "").split(":")[0]


def pick_waves(b: int, h: int, nq: int) -> int:
    """8 waves (128 queries) per workgroup, unless that leaves too few workgroups for the GPU."""
    n_wg8 = b * h * ((nq + 127) // 128)
    return 8 if n_wg8 >= 120 else 4


def unsupported_reason(q, k, v, tensor_layout, attn_mask, pv_fp8, varlen=False) -> str | None:
    if varlen:
        return "varlen"
    if attn_mask is not None:
        return "attn_mask"
    if pv_fp8:
        return "fp8 P.V"
    if q.dtype not in (torch.float16, torch.bfloat16):
        return f"dtype {q.dtype}"
    d = q.shape[-1]
    if not 32 < d <= 128:  # computed with the head dim padded to 64 or 128
        return f"head_dim {d}"
    if tensor_layout not in ("HND", "NHD"):
        return f"layout {tensor_layout}"
    if q.shape[-1] != k.shape[-1] or v.shape[-1] != q.shape[-1]:
        return "head dims differ"
    return None


class _Kernels:
    def __init__(self, device: int, arch: str):
        from .hip import Module

        self.device = device
        code, self.from_cache = build.code_object(_SOURCE, _HEADERS, arch, _OPTIONS, "the native SageAttention kernels")
        try:
            self.module = Module(code, device)
        except Exception:
            if not self.from_cache:
                raise
            code, self.from_cache = build.code_object(
                _SOURCE, _HEADERS, arch, _OPTIONS, "the native SageAttention kernels", refresh=True
            )
            self.module = Module(code, device)
        if "DBG" not in " ".join(_OPTIONS):  # SAGEATTN_NATIVE_OPTS=-DSAGE_DBG_*: perf experiments, wrong results
            self._self_test()

    def launch(self, q_int8, k_int8, vt, q_scale, k_scale, out, lse, *, nq, nk, groups, qblk, causal, waves,
               tensor_layout):  # fmt: skip
        b, hq, nq_pad, dp = q_int8.shape
        d = 128 if dp > 64 else 64
        assert dp == d
        so = (out.stride(0), out.stride(1), out.stride(2)) if tensor_layout == "HND" else (
            out.stride(0), out.stride(2), out.stride(1))  # fmt: skip
        a = _AttnArgs(
            q_int8.data_ptr(),
            k_int8.data_ptr(),
            vt.data_ptr(),
            q_scale.data_ptr(),
            k_scale.data_ptr(),
            out.data_ptr(),
            lse.data_ptr() if lse is not None else 0,
            q_int8.stride(0),
            q_int8.stride(1),
            k_int8.stride(0),
            k_int8.stride(1),
            vt.stride(0),
            vt.stride(1),
            *so,
            q_scale.stride(0),
            q_scale.stride(1),
            k_scale.stride(0),
            k_scale.stride(1),
            lse.stride(0) if lse is not None else 0,
            lse.stride(1) if lse is not None else 0,
            nq,
            nk,
            vt.shape[-1],
            groups,
            qblk.bit_length() - 1,
            _kblk(),
            1 if causal else 0,
            out.shape[-1],
        )
        dt = "bf16" if out.dtype == torch.bfloat16 else "fp16"
        name = f"sage_d{d}_tk{_tk()}_w{waves}_{dt}_{1 if lse is not None else 0}"
        grid = ((nq + 16 * waves - 1) // (16 * waves), hq, b)
        self.module.launch(name, grid, waves * 32, (a,))

    def _self_test(self) -> None:
        from ..api import sageattn_rocm

        g = torch.Generator(device=f"cuda:{self.device}").manual_seed(0)
        with torch.cuda.device(self.device):
            for d, _, waves in VARIANTS:
                for dtype in (torch.bfloat16, torch.float16):
                    for causal, lse in ((False, False), (True, True)):
                        q, k, v = (
                            torch.randn(1, 2, 300, d, device="cuda", generator=g, dtype=torch.float32).to(dtype)
                            for _ in range(3)
                        )
                        ref = sageattn_rocm(q, k, v, is_causal=causal, return_lse=lse, _native=False)
                        got = _run(self, q, k, v, "HND", causal, d**-0.5, lse, True, "per_warp", waves=waves)
                        ref_o, got_o = (ref[0], got[0]) if lse else (ref, got)
                        cos = torch.nn.functional.cosine_similarity(
                            ref_o.float().flatten(), got_o.float().flatten(), dim=0
                        ).item()
                        if not cos > 0.9995:
                            raise RuntimeError(
                                f"native SageAttention self-test failed (d={d}, waves={waves}, {dtype}, "
                                f"causal={causal}): cosine vs Triton {cos:.6f}"
                            )
                        if lse and (ref[1] - got[1]).abs().max().item() > 0.05:
                            raise RuntimeError("native SageAttention self-test failed: LSE mismatch")


_lock = threading.Lock()
_state: dict[int, "_Kernels | BaseException"] = {}


def _kernels(device: int, force: bool) -> "_Kernels | None":
    k = _state.get(device)
    if k is None:
        with _lock:
            k = _state.get(device)
            if k is None:
                try:
                    k = _Kernels(device, _device_arch(device))
                except Exception as e:  # noqa: BLE001
                    k = e
                _state[device] = k
                if isinstance(k, BaseException) and not force:
                    _warn(device, k)
    if isinstance(k, BaseException):
        if force:
            raise k
        return None
    return k


def _warn(device: int, err: BaseException) -> None:
    warnings.warn(
        f"SageAttention: native gfx11 kernel disabled on device {device} ({type(err).__name__}: {err}); "
        "using the Triton kernel",
        RuntimeWarning,
        stacklevel=4,
    )


def status() -> dict[int, str]:
    return {d: ("ok" if isinstance(k, _Kernels) else f"failed: {k}") for d, k in _state.items()}


def _run(kern, q, k, v, tensor_layout, is_causal, sm_scale, return_lse, smooth_k, qk_quant_gran, waves=None):
    from ..api import QK_GRAN_TO_QBLK
    from ..quant import quant_qk_int8, seq_mean, transpose_v_fp16

    seq_dim = 1 if tensor_layout == "NHD" else 2
    nh_dim = 2 if tensor_layout == "NHD" else 1
    b, nq, hq = q.shape[0], q.shape[seq_dim], q.shape[nh_dim]
    nk, hk = k.shape[seq_dim], k.shape[nh_dim]
    if is_causal and nq != nk:
        raise ValueError("causal attention needs qo_len == kv_len")
    if waves is None:
        waves = pick_waves(b, hq, nq)
    qblk = QK_GRAN_TO_QBLK[qk_quant_gran]
    nq_pad = -(-nq // 128) * 128
    nk_pad = -(-nk // 128) * 128
    km = seq_mean(k, tensor_layout) if smooth_k else None
    q_int8, q_scale, k_int8, k_scale = quant_qk_int8(
        q, k, km, sm_scale, BLKQ=qblk, BLKK=_kblk(), tensor_layout=tensor_layout, nq_pad=nq_pad, nk_pad=nk_pad
    )
    vt = transpose_v_fp16(v, tensor_layout, nk_pad)
    out = torch.empty_like(q)
    lse = torch.empty((b, hq, nq), dtype=torch.float32, device=q.device) if return_lse else None
    kern.launch(q_int8, k_int8, vt, q_scale, k_scale, out, lse, nq=nq, nk=nk, groups=hq // hk, qblk=qblk,
                causal=is_causal, waves=waves, tensor_layout=tensor_layout)  # fmt: skip
    if return_lse:
        lse = lse / 1.44269504
        if smooth_k:
            km_b = km.repeat_interleave(hq // hk, dim=1) if hq != hk else km
            qh = q.transpose(1, 2) if tensor_layout == "NHD" else q
            lse = lse + torch.matmul(qh.float(), km_b.unsqueeze(-1)).squeeze(-1) * sm_scale
        return out, lse
    return out


def sageattn_native(q, k, v, tensor_layout="HND", is_causal=False, sm_scale=None, return_lse=False, smooth_k=True,
                    qk_quant_gran="per_warp", waves=None):  # fmt: skip
    """Run the native kernel unconditionally (raises if it is unavailable)."""
    device = q.device.index if q.device.index is not None else torch.cuda.current_device()
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5
    return _run(_kernels(device, force=True), q, k, v, tensor_layout, is_causal, sm_scale, return_lse, smooth_k,
                qk_quant_gran, waves)  # fmt: skip


def try_sageattn(q, k, v, tensor_layout, is_causal, sm_scale, return_lse, smooth_k, qk_quant_gran, attn_mask,
                 pv_fp8):  # fmt: skip
    """The native result, or ``None`` when the Triton kernel should run instead."""
    m = mode()
    if m == "off":
        return None
    device = q.device.index if q.device.index is not None else torch.cuda.current_device()
    if m == "auto" and not _device_arch(device).startswith("gfx11"):
        return None
    if unsupported_reason(q, k, v, tensor_layout, attn_mask, pv_fp8) is not None:
        return None
    seq_dim = 1 if tensor_layout == "NHD" else 2
    if is_causal and q.shape[seq_dim] != k.shape[seq_dim]:
        return None
    kern = _kernels(device, force=(m == "force"))
    if kern is None:
        return None
    try:
        return _run(kern, q, k, v, tensor_layout, is_causal, sm_scale, return_lse, smooth_k, qk_quant_gran)
    except Exception as e:  # noqa: BLE001
        if m == "force":
            raise
        _state[device] = e
        _warn(device, e)
        return None
