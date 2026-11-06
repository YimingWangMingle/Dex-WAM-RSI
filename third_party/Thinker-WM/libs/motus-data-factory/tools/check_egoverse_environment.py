#!/usr/bin/env python3
"""Fail-closed environment gate for native EgoVerse Zarr v3 access."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--egoverse-root", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = Path(args.egoverse_root)
    required_roots = (
        root / "processed_v3" / "aria",
        root / "processed_v3" / "eva",
        root / "processed_v3" / "scale",
        root / "processed_v3" / "mecka" / "freeform",
        root / "processed_v3" / "mecka" / "flagship",
    )
    print(f"python_executable={sys.executable}", flush=True)
    print(f"python_version={sys.version}", flush=True)
    if sys.version_info < (3, 11):
        raise RuntimeError("EgoVerse requires Python >=3.11")

    import zarr

    print(f"zarr_version={zarr.__version__}", flush=True)
    if str(zarr.__version__) != "3.1.5":
        raise RuntimeError(
            f"EgoVerse requires zarr==3.1.5, found {zarr.__version__}"
        )

    import motus_data
    from motus_data.egoverse import EGOVERSE_PROFILES

    print(f"motus_data_path={motus_data.__file__}", flush=True)
    print(f"egoverse_root={root}", flush=True)
    if not root.is_dir():
        raise FileNotFoundError(root)
    for required in required_roots:
        if not required.is_dir():
            raise FileNotFoundError(required)
        print(f"readable_source_root={required}", flush=True)
    print(
        "egoverse_profiles=" + ",".join(sorted(EGOVERSE_PROFILES)),
        flush=True,
    )
    print("EGOVERSE_ENVIRONMENT_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
