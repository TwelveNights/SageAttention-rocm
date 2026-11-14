"""
Windows: what Triton needs to compile its small HIP launcher, besides ``triton-windows`` itself.

- The ROCm SDK devel package matching the installed ROCm wheels (HIP headers). It comes from repo.radeon.com, so
  pip cannot install the matching version as a dependency of this package; we point Triton at it when it is
  there and otherwise say exactly what to run.
- Python's ``include`` / ``libs`` folders, which embedded Pythons (ComfyUI's portable build) leave out. These are
  plain files; they are downloaded once from the triton-windows project.
"""

import importlib.metadata as metadata
import importlib.util
import os
import re
import sys
import sysconfig
import tempfile
import urllib.request
import warnings
import zipfile

ROCM_DEVEL_URL = "https://repo.radeon.com/rocm/windows/rocm-rel-{v}/rocm_sdk_devel-{v}-py3-none-win_amd64.whl"
PYTHON_INCLUDE_LIBS = {
    (3, 10): "python_3.10.11_include_libs.zip",
    (3, 11): "python_3.11.9_include_libs.zip",
    (3, 12): "python_3.12.7_include_libs.zip",
    (3, 13): "python_3.13.2_include_libs.zip",
}
INCLUDE_LIBS_URL = "https://github.com/woct0rdho/triton-windows/releases/download/v3.0.0-windows.post1/{}"

# set by prepare(): what the user has to install before the Triton kernels can compile, or None
MISSING: str | None = None


def _version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def _use_rocm_devel() -> bool:
    """Point Triton at the ROCm SDK devel package (HIP headers, clang). False if it is not installed."""
    if os.environ.get("ROCM_HOME") or os.environ.get("HIP_PATH"):
        return True
    if importlib.util.find_spec("rocm_sdk") is None:
        return False
    try:
        from rocm_sdk import _devel

        root = str(_devel.get_devel_root())
    except Exception:  # noqa: BLE001 - the devel package is missing
        return False
    os.environ["ROCM_HOME"] = root
    os.environ.setdefault("HIP_PATH", root)
    os.environ["PATH"] = os.pathsep.join(
        [os.path.join(root, "bin"), os.path.join(root, "lib", "llvm", "bin"), os.environ.get("PATH", "")]
    )
    # Triton may be imported already, and its AMD driver reads the ROCm location only at import
    driver = sys.modules.get("triton.backends.amd.driver")
    include = os.path.join(root, "include")
    if driver is not None and include not in getattr(driver, "include_dirs", [include]):
        driver.include_dirs.append(include)
    return True


def _add_python_include_libs() -> None:
    base = os.path.dirname(sysconfig.get_paths()["include"])
    lib = f"python{sys.version_info.major}{sys.version_info.minor}.lib"
    if os.path.exists(os.path.join(base, "include", "Python.h")) and os.path.exists(os.path.join(base, "libs", lib)):
        return
    name = PYTHON_INCLUDE_LIBS.get(sys.version_info[:2])
    if name is None:
        return
    warnings.warn(f"SageAttention: adding Python's include/ and libs/ (needed by Triton) to {base}, once", stacklevel=2)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, name)
            urllib.request.urlretrieve(INCLUDE_LIBS_URL.format(name), path)
            with zipfile.ZipFile(path) as z:
                z.extractall(base)
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"SageAttention: could not add Python's include/libs ({e}); Triton may fail to compile",
                      stacklevel=2)


def prepare() -> None:
    """Called once at import on Windows with a ROCm PyTorch."""
    global MISSING
    _add_python_include_libs()
    if _use_rocm_devel():
        return
    rocm = _version("rocm-sdk-core")
    command = f'"{sys.executable}" -s -m pip install'
    if rocm and re.fullmatch(r"\d+\.\d+\.\d+", rocm):
        command += f" {ROCM_DEVEL_URL.format(v=rocm)}"
    else:
        command += f" rocm-sdk-devel=={rocm} (from the index your ROCm PyTorch came from)"
    MISSING = ("SageAttention on Windows needs the ROCm SDK devel package that matches your ROCm wheels, once "
               f"(close ComfyUI / Python first):\n  {command}")
