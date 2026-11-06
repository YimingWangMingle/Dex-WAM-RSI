#!/usr/bin/env python3
"""Index RoboColiseum, export RSI training manifests or seal ICL candidates."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thinker_wm.data_prepare.robocoliseum_core import SCHEMA, digest, load_catalog, read_json, scan
from data.robocoliseum import experience_candidate, training_manifest


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    commands = ap.add_subparsers(dest="command", required=True)
    index = commands.add_parser("index")
    index.add_argument("--root", required=True)
    index.add_argument("--out-dir", required=True)
    index.add_argument("--split-map", help="JSON object episode_id -> split; complete coverage required")
    index.add_argument("--validate-rows", action="store_true")
    index.add_argument("--validate-videos", action="store_true")
    index.add_argument("--hash-media", action="store_true", help="hash every video (large I/O)")
    index.add_argument("--workers", type=int, default=1, help="bounded task readers, 1..8")
    for name in ("training", "experience"):
        sub = commands.add_parser(name)
        sub.add_argument("--catalog", required=True)
        sub.add_argument("--output", required=True)
        sub.add_argument("--task", help="exact task directory name")
        sub.add_argument("--split", choices=["train", "development"], default="train")
        if name == "training":
            sub.add_argument("--action-offset", type=int, choices=[0, 1], required=True)
            sub.add_argument("--parent-checkpoint", required=True)
            sub.add_argument("--data-version", required=True)
            sub.add_argument("--allow-reviewed-demonstrations", action="store_true")
        else:
            sub.add_argument("--package-id", required=True)
            sub.add_argument("--max-episodes", type=int, default=5)
    args = ap.parse_args()
    if args.command == "index":
        out = Path(args.out_dir)
        if out.exists():
            raise FileExistsError("index output must be a new directory")
        records, errors = scan(args.root, split_map=read_json(args.split_map) if args.split_map else None,
                               validate_rows=args.validate_rows, validate_videos=args.validate_videos,
                               hash_media=args.hash_media, workers=args.workers)
        out.mkdir(parents=True)
        with (out / "episodes.jsonl").open("x", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        report = dict(schema=SCHEMA, source_root=str(Path(args.root).resolve()),
                      episodes=len(records), frames=sum(r["length"] for r in records),
                      tasks=dict(Counter(r["task"] for r in records)),
                      splits=dict(Counter(r["split"] for r in records)),
                      records_sha256=digest(records), rejected=errors,
                      native_success_available=False,
                      explicit_split_map=bool(args.split_map), scene_independence_verified=False,
                      validation=dict(rows=args.validate_rows,
                                      video_headers=args.validate_videos or args.hash_media,
                                      video_hashes=args.hash_media))
        write_new(out / "report.json", report)
        print(json.dumps({k: v for k, v in report.items() if k not in ("tasks", "rejected")}, indent=2))
        print(f"rejected={len(errors)}; details: {out / 'report.json'}")
        return 1 if errors or not records else 0
    records = load_catalog(args.catalog)
    records = [r for r in records if r["split"] == args.split and (not args.task or r["task"] == args.task)]
    if not records:
        raise ValueError("no episodes match selection")
    if args.command == "training":
        result = training_manifest(records, action_offset=args.action_offset,
                                   parent_checkpoint=args.parent_checkpoint, data_version=args.data_version,
                                   allow_reviewed_demonstrations=args.allow_reviewed_demonstrations)
        write_new(args.output, result)
        print(f"training rows={len(result['rows'])}; {args.output}")
    else:
        if args.max_episodes < 1:
            raise ValueError("max-episodes must be positive")
        result = experience_candidate(records[:args.max_episodes], args.output, package_id=args.package_id)
        print(json.dumps(result.identity(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
