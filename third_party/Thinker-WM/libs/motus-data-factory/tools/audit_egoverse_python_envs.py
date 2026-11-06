#!/usr/bin/env python3
"""Inventory existing Python environments for EgoVerse without modifying them."""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROBE = r"""
import importlib
import json
import sys

result = {
    "executable": sys.executable,
    "python": ".".join(str(value) for value in sys.version_info[:3]),
}
for name in ("zarr", "torch", "numpy", "PIL", "omegaconf"):
    try:
        module = importlib.import_module(name)
        result[name] = str(getattr(module, "__version__", "present"))
    except Exception as error:
        result[name] = "ERROR:" + type(error).__name__ + ":" + str(error)
print(json.dumps(result, sort_keys=True))
"""


def _conda_environments() -> list[str]:
    executables = {
        value
        for value in (
            shutil.which("conda"),
            "/home/fengodchen/miniconda3/bin/conda",
            "/opt/conda/bin/conda",
        )
        if value and Path(value).is_file()
    }
    environments = []
    for executable in sorted(executables):
        try:
            completed = subprocess.run(
                [executable, "env", "list", "--json"],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            environments.extend(json.loads(completed.stdout).get("envs", []))
        except Exception as error:
            print(
                f"CONDA_LIST_ERROR executable={executable} error={error}",
                flush=True,
            )
    return environments


def _candidates() -> list[str]:
    candidates = {sys.executable}
    for name in ("python3.13", "python3.12", "python3.11", "python3", "python"):
        value = shutil.which(name)
        if value:
            candidates.add(value)
    for environment in _conda_environments():
        candidates.add(str(Path(environment) / "bin" / "python"))
    patterns = (
        "/home/*/miniconda*/bin/python",
        "/home/*/miniconda*/envs/*/bin/python",
        "/home/*/miniforge*/bin/python",
        "/home/*/miniforge*/envs/*/bin/python",
        "/opt/conda/bin/python",
        "/opt/conda/envs/*/bin/python",
        "/root/miniconda*/bin/python",
        "/root/miniconda*/envs/*/bin/python",
        "/usr/bin/python3*",
        "/usr/local/bin/python3*",
    )
    for pattern in patterns:
        candidates.update(glob.glob(pattern))
    return sorted(
        value
        for value in candidates
        if os.path.isfile(value) and os.access(value, os.X_OK)
    )


def main() -> int:
    compatible = []
    candidates = _candidates()
    print(f"candidate_count={len(candidates)}", flush=True)
    for executable in candidates:
        try:
            completed = subprocess.run(
                [executable, "-c", PROBE],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            result = json.loads(completed.stdout.strip().splitlines()[-1])
            result["candidate"] = executable
            version = tuple(
                int(value) for value in result["python"].split(".")[:2]
            )
            dependencies = ("torch", "numpy", "PIL", "omegaconf")
            result["egoverse_compatible"] = (
                version >= (3, 11)
                and result.get("zarr") == "3.1.5"
                and all(
                    not str(result.get(name, "")).startswith("ERROR:")
                    for name in dependencies
                )
            )
            if result["egoverse_compatible"]:
                compatible.append(executable)
            print(
                "PYTHON_ENV " + json.dumps(result, sort_keys=True),
                flush=True,
            )
        except Exception as error:
            print(
                f"PYTHON_ENV_ERROR candidate={executable} error={error}",
                flush=True,
            )
    print(
        "compatible_environments=" + json.dumps(compatible),
        flush=True,
    )
    print("EGOVERSE_PYTHON_ENV_AUDIT_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
