#!/usr/bin/env python3
"""Filter robotwin_3d catalog to one task, then build index + all-split ranges."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from thinker_wm.data_prepare import ensure_motus_data_importable  # noqa: E402

ensure_motus_data_importable()

from motus_data.catalog.sqlite_index import build_catalog_offset_index  # noqa: E402
from thinker_wm.data_prepare import robotwin_3d as _robotwin_3d  # noqa: E402,F401


def _task_of(rec: dict) -> str:
    meta = rec.get("meta") or {}
    name = meta.get("task_name")
    if name:
        return str(name)
    for item in rec.get("texts") or []:
        nested = (item or {}).get("meta") or {}
        if nested.get("task_name"):
            return str(nested["task_name"])
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        action="append",
        dest="tasks",
        default=None,
        help="Task name; repeat for a multi-task subset. Default: turn_switch",
    )
    parser.add_argument(
        "--src-catalog",
        default=str(
            _REPO / "tmp/catalogs/robotwin_3d/robotwin_3d.episode_catalog.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: tmp/catalogs/robotwin_3d_<task>/",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--num-video-frames", type=int, default=48)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=None,
        help="Build train + holdout ranges instead of a single all-split file. "
        "Split seed matches the parent catalog.",
    )
    args = parser.parse_args()

    tasks = [str(t).strip() for t in (args.tasks or ["turn_switch"]) if str(t).strip()]
    want = set(tasks)
    slug = "_".join(tasks)
    src = Path(args.src_catalog)
    out_dir = Path(args.output_dir) if args.output_dir else (
        _REPO / "tmp/catalogs" / f"robotwin_3d_{slug}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog = out_dir / f"robotwin_3d_{slug}.episode_catalog.jsonl"
    meta_path = out_dir / f"robotwin_3d_{slug}.episode_catalog.meta.json"
    splits = (
        ["all"] if args.holdout_fraction is None else ["train", "holdout"]
    )

    if catalog.exists() and not args.force:
        raise FileExistsError(f"{catalog} exists; pass --force")

    n = 0
    per_task: dict[str, int] = {t: 0 for t in tasks}
    digest = hashlib.sha256()
    tmp = catalog.with_suffix(catalog.suffix + ".tmp")
    with src.open("r", encoding="utf-8") as fin, tmp.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            name = _task_of(rec)
            if name not in want:
                continue
            raw = json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n"
            fout.write(raw)
            digest.update(raw.encode("utf-8"))
            n += 1
            per_task[name] = per_task.get(name, 0) + 1
    if n == 0:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"no episodes for tasks={tasks}")
    tmp.replace(catalog)

    metadata = {
        "format": "motus_episode_catalog_v2",
        "catalog_id": f"robotwin_3d_{slug}_20260909",
        "source": "robotwin_3d",
        "child": "robotwin_3d",
        "episodes": n,
        "sha256": digest.hexdigest(),
        "meta": {
            "task_names": tasks,
            "episodes_per_task": per_task,
            "parent_catalog": str(src),
            "data_version": "flexpi_robotwin_3d_v1",
        },
    }
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    index_info = build_catalog_offset_index(catalog, force=True)
    print(json.dumps(
        {"episodes": n, "per_task": per_task, "catalog": str(catalog), **index_info},
        indent=2,
    ))

    from thinker_wm.data_prepare.sampling_ranges import build_sampling_ranges  # noqa: E402
    from motus_data.releases import get_episode_dataset_release  # noqa: E402
    from motus_data.repository import EpisodeRepository  # noqa: E402

    reader_options = get_episode_dataset_release("robotwin_3d").reader_options
    repository = EpisodeRepository(
        catalog,
        index_path=index_info["index_path"],
        verify_sha256=False,
        reader_options=reader_options,
    )
    try:
        for split in splits:
            ranges = out_dir / f"robotwin_3d_{slug}.{split}.ranges.sqlite"
            stats = build_sampling_ranges(
                repository,
                ranges,
                dataset="robotwin_3d",
                num_video_frames=int(args.num_video_frames),
                action_fps_ratio=1,
                layout="consecutive",
                workers=int(args.workers),
                split=split,
                holdout_fraction=args.holdout_fraction,
                force=True,
                reader_options=reader_options,
            )
            print(json.dumps({"ranges": str(ranges), **{k: stats[k] for k in (
                "episodes", "intervals", "valid_starts", "split"
            ) if k in stats}}, indent=2))
    finally:
        repository.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
