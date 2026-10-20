"""
Hand-written HIP kernels for GPUs where Triton leaves performance behind.

Currently one: SageAttention forward for RDNA3 (gfx11), ``sage_gfx11.hip``
(see :mod:`.sage`), compiled at runtime with hipRTC -- no host compiler needed.
``hip.py``, ``hiprtc.py``, ``build.py`` and ``hip_prelude.h`` are adapted from
nunchaku-rocm's native kernel infrastructure.
"""
