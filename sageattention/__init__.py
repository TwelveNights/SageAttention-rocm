import os as _os

if _os.name == "nt":
    from ._triton_windows import require_triton as _require_triton

    _require_triton()

from .core import sageattn, sageattn_varlen
from .core import sageattn_qk_int8_pv_fp16_triton
from .core import sageattn_qk_int8_pv_fp16_cuda 
from .core import sageattn_qk_int8_pv_fp8_cuda
from .core import sageattn_qk_int8_pv_fp8_cuda_sm90