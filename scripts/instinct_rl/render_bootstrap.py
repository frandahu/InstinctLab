"""Select NumPy 1.x before Isaac Sim 4.5/5.x loads its compiled RGB interfaces.

Only the evaluation process is affected. Do not downgrade a shared environment
while a training process is using it. Prefer Isaac Sim's bundled NumPy; if it is
absent, cache a single compatible wheel under the repository's ignored outputs.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import re
import subprocess
import sys
import tempfile
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path


def _package_version(package):
    path = package / "version.py"
    if not path.is_file():
        return None
    match = re.search(r"^(?:version|__version__)\s*=\s*['\"]([^'\"]+)['\"]", path.read_text(encoding="utf-8"), re.M)
    return match.group(1) if match else None


def _compatible_package(package):
    version = _package_version(package)
    return version is not None and version.startswith("1.26.") and any(
        (package / "core" / ("_multiarray_umath" + suffix)).is_file() for suffix in EXTENSION_SUFFIXES
    )


def _bundled_numpy_candidates():
    roots = [Path("/isaac-sim"), Path(__file__).resolve().parents[2].parent / "isaaclab" / "_isaac_sim"]
    for name in ("ISAACSIM_PATH", "ISAAC_PATH", "CARB_APP_PATH", "EXP_PATH"):
        if os.environ.get(name):
            path = Path(os.environ[name]).resolve()
            roots.extend((path, *list(path.parents)[:3]))
    roots.extend(list(Path(sys.executable).resolve().parents)[:4])
    spec = importlib.util.find_spec("isaacsim")
    if spec is not None and spec.origin:
        roots.append(Path(spec.origin).parent)
    seen = set()
    for root in roots:
        root = root.resolve()
        if root in seen:
            continue
        seen.add(root)
        for folder in ("exts", "extscache"):
            for extension in ("omni.kit.pip_archive*", "omni.isaac.ml_archive*"):
                yield from sorted(root.glob(f"{folder}/{extension}/pip_prebundle/numpy"))


def _cached_numpy_package():
    cache_root = Path(__file__).resolve().parents[2] / "outputs" / "stair_eval" / "runtime_deps"
    tag = f"numpy126-py{sys.version_info.major}{sys.version_info.minor}-{sys.platform}-{platform.machine()}"
    target = cache_root / tag
    package = target / "numpy"
    if _compatible_package(package):
        return package
    if target.exists():
        raise RuntimeError(f"Incomplete NumPy evaluation cache: {target}. Rename it and retry.")
    cache_root.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Caching evaluation-only NumPy 1.26.4 in {target}", flush=True)
    with tempfile.TemporaryDirectory(prefix="numpy126-", dir=str(cache_root)) as temporary:
        staging = Path(temporary) / "packages"
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
            "--only-binary=:all:", "--no-deps", "--target", str(staging), "numpy==1.26.4",
        ], check=True)
        if not _compatible_package(staging / "numpy"):
            raise RuntimeError("Cached NumPy wheel does not match this Python interpreter")
        try:
            staging.rename(target)
        except FileExistsError:
            # Another evaluator may have completed the same cache concurrently.
            if not _compatible_package(package):
                raise
    return package


def _load_numpy_package(package):
    """Load only NumPy, without adding the entire Isaac Sim pip archive to sys.path."""
    spec = importlib.util.spec_from_file_location(
        "numpy", package / "__init__.py", submodule_search_locations=[str(package)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["numpy"] = module
    spec.loader.exec_module(module)
    return module


def preload_isaac_numpy():
    """Run before AppLauncher/PyTorch; replacing an already-loaded NumPy is unsafe."""
    existing = sys.modules.get("numpy")
    if existing is not None:
        if int(existing.__version__.split(".")[0]) >= 2:
            raise RuntimeError("NumPy 2.x was imported before render bootstrap; start eval_stairs.py in a fresh Python process")
        module, source = existing, "existing_environment"
    else:
        spec = importlib.util.find_spec("numpy")
        if spec is None or not spec.origin:
            raise RuntimeError("NumPy is missing from the evaluation Python environment")
        default = Path(spec.origin).parent
        version = _package_version(default)
        if version is None:
            raise RuntimeError(f"Cannot identify NumPy version before Isaac Sim startup: {default}")
        if int(version.split(".")[0]) < 2:
            package, source = default, "existing_environment"
        else:
            package = next((path for path in _bundled_numpy_candidates() if _compatible_package(path)), None)
            source = "isaac_sim_bundle"
            if package is None:
                package, source = _cached_numpy_package(), "evaluation_cache"
            print(f"[INFO] Replacing NumPy {version} for this evaluation process only", flush=True)
        module = _load_numpy_package(package)
    if int(module.__version__.split(".")[0]) >= 2:
        raise RuntimeError("Isaac Sim 4.5/5.x RGB rendering requires NumPy <2")
    record = {"version": module.__version__, "path": str(module.__file__), "source": source}
    print(f"[INFO] Evaluation NumPy {record['version']} from {record['path']}", flush=True)
    return record
