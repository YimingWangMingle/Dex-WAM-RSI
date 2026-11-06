#!/usr/bin/env python3
"""Build a RoboDojo episode catalog (v2 records) and a scene-level split.

  python scripts/build_robodojo_catalog.py --root /path/to/robodojo --field-map cfg.yaml \
      --robot <robot name> --out-dir tmp/catalogs/robodojo --val-fraction 0.1 --split-seed <n> \
      [--exclude-scenes-file eval_scenes.txt] [--task-from parent_dir] [--max-episodes N]
      [--tasks stack_bowls insert_tubes]   # with --task-from root_subdir: <root>/<task>/...

Writes ``robodojo.episode_catalog.jsonl`` (+ ``.meta.json``) and
``robodojo.split.json``. Scenes listed in --exclude-scenes-file (formal test
scenes) never enter train or val. When motus_data is importable the records
and metadata are validated against its v2 contract and an offset index is
built; otherwise the catalog is written unvalidated and the script says so.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from thinker_wm.data_prepare.robodojo_core import (  # noqa: E402
    SOURCE_DATASET, FieldMap, RoboDojoHDF5, episode_record, split_by_scene,
)


def _validators():
    try:
        from thinker_wm.data_prepare._bootstrap import ensure_motus_data_importable
        ensure_motus_data_importable()
        from motus_data.catalog.sqlite_index import build_catalog_offset_index
        from motus_data.index_contracts import (
            MOTUS_EPISODE_CATALOG_METADATA_V2, validate_catalog_metadata_v2,
            validate_episode_catalog_binding, validate_episode_record_v2)
        return dict(fmt=MOTUS_EPISODE_CATALOG_METADATA_V2, record=validate_episode_record_v2,
                    meta=validate_catalog_metadata_v2, bind=validate_episode_catalog_binding,
                    index=build_catalog_offset_index)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] motus_data unavailable ({exc}); catalog NOT validated", file=sys.stderr)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--field-map", required=True)
    ap.add_argument("--robot", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--split-seed", type=int, required=True)
    ap.add_argument("--exclude-scenes-file")
    ap.add_argument("--task-from", choices=["parent_dir", "root_subdir", "none"], default="parent_dir")
    ap.add_argument("--tasks", nargs="*", help="keep only these task names")
    ap.add_argument("--catalog-id", default="robodojo_v1")
    ap.add_argument("--max-episodes", type=int)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    fm = FieldMap.load(a.field_map)
    io = RoboDojoHDF5(fm)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    catalog = out / "robodojo.episode_catalog.jsonl"
    meta_path = out / "robodojo.episode_catalog.meta.json"
    split_path = out / "robodojo.split.json"
    if catalog.exists() and not a.force:
        raise SystemExit(f"{catalog} exists; pass --force")
    files = sorted([*Path(a.root).rglob("*.hdf5"), *Path(a.root).rglob("*.h5")])
    task_of = {"parent_dir": lambda p: p.parent.name,
               "root_subdir": lambda p: p.relative_to(a.root).parts[0],
               "none": lambda p: None}[a.task_from]
    if a.tasks:
        files = [p for p in files if task_of(p) in set(a.tasks)]
    if a.max_episodes:
        files = files[: a.max_episodes]
    v = _validators()
    records, errors = [], []
    for p in files:
        task = task_of(p)
        try:
            rec = episode_record(p, fm, io, robot=a.robot, task_name=task)
            if v:
                v["record"](rec)
            records.append(rec)
        except Exception as exc:  # noqa: BLE001
            errors.append({"file": str(p), "error": str(exc)})
    if not records:
        raise SystemExit(f"no valid episodes ({len(errors)} errors): {errors[:3]}")
    digest = hashlib.sha256()
    tmp = catalog.with_name(f".{catalog.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as h:
        for rec in records:
            line = json.dumps(rec, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
            h.write(line)
            digest.update(line.encode())
    metadata = {"format": v["fmt"] if v else "motus_episode_catalog_v2", "catalog_id": a.catalog_id,
                "source": SOURCE_DATASET, "child": SOURCE_DATASET, "episodes": len(records),
                "sha256": digest.hexdigest(),
                "meta": {"data_root": str(Path(a.root).resolve()), "field_map": str(a.field_map),
                         "field_map_digest": fm.digest, "control_hz": fm.control_hz,
                         "skipped": len(errors)}}
    if v:
        v["meta"](metadata)
        for rec in records:
            v["bind"](rec, metadata)
    os.replace(tmp, catalog)
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    excl = []
    if a.exclude_scenes_file:
        excl = [s.strip() for s in Path(a.exclude_scenes_file).read_text().splitlines() if s.strip()]
    split = split_by_scene(records, val_fraction=a.val_fraction, seed=a.split_seed, exclude_scenes=excl)
    split["meta"] = {"split_seed": a.split_seed, "val_fraction": a.val_fraction,
                     "excluded_scenes": excl, "catalog_sha256": metadata["sha256"]}
    split_path.write_text(json.dumps(split, indent=2) + "\n")
    if errors:
        (out / "robodojo.catalog_errors.json").write_text(json.dumps(errors, indent=2))
    if v:
        v["index"](catalog, force=True)
    print(f"{len(records)} episodes ({len(errors)} skipped) -> {catalog}; "
          f"train {len(split['train'])} / val {len(split['val'])} / excluded {len(split['excluded'])}")


if __name__ == "__main__":
    main()
