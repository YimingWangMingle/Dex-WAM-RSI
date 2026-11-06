#!/usr/bin/env python3
"""Build robotwin_3d Episode Catalog + SQLite offset index."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from thinker_wm.data_prepare.robotwin_3d import build_robotwin_3d_catalog_and_index  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/data/xiongwei.zhao/robotwin_3d")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: tmp/catalogs/robotwin_3d (V13 live) or "
        "tmp/catalogs/robotwin_3d_seen100 with --all-seen.",
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--all-seen",
        action="store_true",
        help="Write every official seen paraphrase (≈100/episode). "
        "Refuses to overwrite the live V13 catalog.",
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = str(
            _REPO
            / (
                "tmp/catalogs/robotwin_3d_seen100"
                if args.all_seen
                else "tmp/catalogs/robotwin_3d"
            )
        )
    meta = build_robotwin_3d_catalog_and_index(
        data_root=args.data_root,
        output_dir=args.output_dir,
        max_episodes=args.max_episodes,
        force=args.force,
        all_seen=args.all_seen,
    )
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
