"""
Windows + ROCm PyTorch: SageAttention's kernels need Triton, from triton-windows in the release series that matches the
installed PyTorch. It cannot be a dependency of this package (pip cannot condition on the installed PyTorch), so a
missing Triton is reported with the exact install command before anything imports it.
"""

import importlib.util
import re
import sys

# triton-windows release series for each PyTorch minor version: the Triton version PyTorch's own Linux wheels pin,
# except that before 2.12 it is 3.6, the first triton-windows release with the AMD backend (3.6.0.post26).
# torch.compile needs the matching series; newer PyTorch gets the newest series.
TRITON_FOR_TORCH = [
    ((2, 11), ">=3.6.0.post26,<3.7"),
    ((2, 12), ">=3.7.0,<3.7.1"),
    ((2, 13), ">=3.7.1,<3.8"),
    ((2, 14), ">=3.8,<3.9"),
    ((2, 15), ">=3.9,<3.10"),
]


def triton_windows_requirement(torch_version: str) -> str:
    """The triton-windows requirement matching a PyTorch version such as ``2.9.1+rocm7.2.1``."""
    m = re.match(r"(\d+)\.(\d+)", torch_version)
    minor = (int(m.group(1)), int(m.group(2))) if m else (99, 0)
    for upto, spec in TRITON_FOR_TORCH:
        if minor <= upto:
            return "triton-windows" + spec
    return "triton-windows>=3.9"


def require_triton() -> None:
    if importlib.util.find_spec("triton") is not None:
        return
    import torch

    if not torch.version.hip:
        return
    req = triton_windows_requirement(torch.__version__)
    raise ImportError(
        f"SageAttention on Windows needs Triton matching PyTorch {torch.__version__}; install it once with:\n"
        f'  "{sys.executable}" -s -m pip install "{req}"'
    )
