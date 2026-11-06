#!/usr/bin/env python3
"""Read-only inventory of cached packages usable for EgoVerse."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


RELEVANT = re.compile(
    r"(python-3\.(11|12|13)|torch|pytorch|zarr|numcodecs|numpy|"
    r"pillow|omegaconf|donfig|google.crc32c|packaging|typing.extensions)",
    re.IGNORECASE,
)


def _print_conda_cache(root: Path) -> None:
    print(f"===== CONDA CACHE {root} =====", flush=True)
    if not root.is_dir():
        print("NOT_FOUND", flush=True)
        return
    matches = sorted(
        path.name for path in root.iterdir() if RELEVANT.search(path.name)
    )
    for value in matches:
        print(value, flush=True)
    print(f"relevant_entries={len(matches)}", flush=True)


def _print_pip_cache(python: Path) -> None:
    print(f"===== PIP CACHE {python} =====", flush=True)
    if not python.is_file():
        print("NOT_FOUND", flush=True)
        return
    for arguments in (("cache", "dir"), ("cache", "list")):
        try:
            result = subprocess.run(
                [str(python), "-m", "pip", *arguments],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            for line in result.stdout.splitlines():
                if arguments[-1] == "dir" or RELEVANT.search(line):
                    print(line, flush=True)
        except Exception as error:
            print(f"ERROR arguments={arguments} error={error}", flush=True)


def _print_cached_archives(root: Path) -> None:
    print(f"===== ARCHIVES {root} =====", flush=True)
    if not root.is_dir():
        print("NOT_FOUND", flush=True)
        return
    matches = []
    for directory, _, filenames in os.walk(root):
        relative_depth = len(Path(directory).relative_to(root).parts)
        if relative_depth >= 8:
            continue
        for filename in filenames:
            if (
                filename.endswith((".whl", ".conda", ".tar.bz2"))
                and RELEVANT.search(filename)
            ):
                path = Path(directory) / filename
                matches.append((str(path), path.stat().st_size))
    for path, size in sorted(matches):
        print(f"{path} {size}", flush=True)
    print(f"relevant_archives={len(matches)}", flush=True)


def main() -> int:
    _print_conda_cache(Path("/home/fengodchen/miniconda3/pkgs"))
    for python in (
        Path("/home/fengodchen/miniconda3/bin/python"),
        Path("/home/fengodchen/miniconda3/envs/motus/bin/python"),
    ):
        _print_pip_cache(python)
    for root in (
        Path("/home/fengodchen/.cache/pip"),
        Path("/home/fengodchen/.cache/uv"),
        Path("/root/.cache/pip"),
        Path("/root/.cache/uv"),
    ):
        _print_cached_archives(root)
    print("EGOVERSE_OFFLINE_CACHE_AUDIT_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
