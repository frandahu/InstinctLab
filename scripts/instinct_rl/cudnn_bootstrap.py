"""Prefer the cuDNN wheel required by PyTorch before Isaac Sim starts."""

import ctypes
import os
import re
import sys
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path


def preload_torch_cudnn() -> None:
    """Avoid mixing Isaac Sim's cuDNN with PyTorch's separately bundled version."""
    if sys.platform != "linux":
        return

    try:
        torch_dist = distribution("torch")
        cudnn_dist = distribution("nvidia-cudnn-cu12")
    except PackageNotFoundError:
        return

    required_version = None
    for requirement in torch_dist.requires or ():
        match = re.match(r"nvidia-cudnn-cu12\s*==\s*(\d+(?:\.\d+){2,3})", requirement, re.IGNORECASE)
        if match:
            required_version = match.group(1)
            break
    if required_version != cudnn_dist.version:
        return

    # A normal PyTorch wheel already loads cuDNN from its own site-packages.
    if Path(torch_dist.locate_file("")).resolve() == Path(cudnn_dist.locate_file("")).resolve():
        return

    cudnn_lib = Path(cudnn_dist.locate_file("nvidia/cudnn/lib/libcudnn.so.9")).resolve()
    if not cudnn_lib.is_file():
        raise RuntimeError(f"PyTorch's required cuDNN library is missing: {cudnn_lib}")

    # cuDNN 9 loads sub-libraries later. The loader must see this directory at
    # process startup; changing LD_LIBRARY_PATH in the running process is too late.
    lib_dir = str(cudnn_lib.parent)
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_dir not in ld_library_path.split(":"):
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = f"{lib_dir}:{ld_library_path}" if ld_library_path else lib_dir
        os.execve(sys.executable, [sys.executable, *sys.argv], env)

    lib = ctypes.CDLL(str(cudnn_lib), mode=ctypes.RTLD_GLOBAL)
    lib.cudnnGetVersion.restype = ctypes.c_size_t
    major, minor, patch = (int(part) for part in required_version.split(".")[:3])
    expected_version = major * 10000 + minor * 100 + patch
    actual_version = lib.cudnnGetVersion()
    if actual_version != expected_version:
        raise RuntimeError(f"Expected cuDNN {expected_version} from {cudnn_lib}, got {actual_version}")
    print(f"[INFO] Preloaded PyTorch-compatible cuDNN {required_version} from {cudnn_lib}", flush=True)
