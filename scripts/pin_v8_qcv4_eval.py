#!/usr/bin/env python3
"""Pin V8 16-GPU periodic_eval clips and refuse any QC-v4 DROP.

Writes scripts/family_holdout_qcv4_16gpu.json. Pass --prepare to decode
the official cache under periodic_eval.cache_dir before training starts.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from omegaconf import OmegaConf

from thinker_wm.models.depth_reader import load_depth_selections
from thinker_wm.periodic_eval.config import (
    PeriodicEvalConfig,
    assign_sources,
    build_source_spec,
    parse_periodic_eval_config,
    resolve_source_weights,
    spec_key,
)


def _episode_ids_for_assignment(config, assignment, *, batch_size: int, seed: int) -> list[str]:
    from thinker_wm.data_prepare.factory import _indexed_sources
    from motus_data.indexed_pipeline import open_dataset
    from motus_data.releases import get_episode_dataset_release
    from motus_data.sample_pool import resolve_sample_pool

    use_eval = str(assignment.split) != "train"
    sources = _indexed_sources(config, eval=use_eval)
    source = next((s for s in sources if s["name"] == assignment.source_name), None)
    if source is None:
        raise SystemExit(f"source {assignment.source_name!r} missing")
    dataset_name = str(source["dataset"])
    release = get_episode_dataset_release(dataset_name, source.get("release"))
    repository = open_dataset(
        dataset_name,
        release=release,
        catalog_root=source.get("catalog_root"),
        catalog_path=source.get("catalog_path"),
        index_path=source.get("index_path"),
        readers=source.get("readers"),
    )
    try:
        pool = resolve_sample_pool(source["sample_pool"], repository, seed=seed)
        groups = pool.episode_groups
        order = sorted(groups)
        if not order:
            raise SystemExit(f"{assignment.source_name} {assignment.split} pool is empty")
        random.Random(f"{seed}|{assignment.source_name}").shuffle(order)
        ids: list[str] = []
        shard = int(assignment.shard_ordinal)
        for i in range(int(batch_size)):
            ids.append(order[(shard * batch_size + i) % len(order)])
        return ids
    finally:
        close = getattr(repository, "close", None)
        if callable(close):
            close()


def _roots_from_config(config) -> list[str]:
    return [str(p) for p in OmegaConf.to_object(config.depth.roots)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(_REPO / "third_party/Thinker-WM/configs/depth_ablation_treatment_v8.yaml"))
    parser.add_argument("--world-size", type=int, default=16)
    parser.add_argument("--out", default=str(_REPO / "scripts/family_holdout_qcv4_16gpu.json"))
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    pe_cfg: PeriodicEvalConfig = parse_periodic_eval_config(config)
    weights = resolve_source_weights(pe_cfg, config)
    assignments = assign_sources(
        weights,
        args.world_size,
        pe_cfg.train_pool_fraction,
        holdout_family_counts=pe_cfg.holdout_family_counts,
    )
    keep_index = load_depth_selections(_roots_from_config(config), prod_keep_only=True)
    # Reuse opened pools: 4 holdout sources + a few train shards.
    _eid_cache: dict[tuple[str, str, int, int], list[str]] = {}

    rows = []
    drops = []
    for rank, asg in enumerate(assignments):
        default_bs = max(1, int(config.training.batch_size) // 4)
        batch_size = int(asg.batch_size) if asg.batch_size is not None else default_bs
        spec = build_source_spec(config, pe_cfg, asg, batch_size=batch_size)
        cache_key = (asg.source_name, asg.split, int(asg.shard_ordinal), batch_size)
        if cache_key not in _eid_cache:
            _eid_cache[cache_key] = _episode_ids_for_assignment(
                config, asg, batch_size=batch_size, seed=int(pe_cfg.seed)
            )
        eids = _eid_cache[cache_key]
        for i, eid in enumerate(eids):
            kept = eid in keep_index
            rec = {
                "rank": rank,
                "sample_index": i,
                "source": asg.source_name,
                "split": asg.split,
                "shard": int(asg.shard_ordinal),
                "episode_id": eid,
                "prod_keep": kept,
                "spec_key": spec_key(spec),
            }
            rows.append(rec)
            if not kept:
                drops.append(rec)
            mark = "KEEP" if kept else "DROP"
            print(f"rank{rank} {asg.split:7s} {asg.source_name:36s} {eid} {mark}")

    if drops:
        print(f"\nREFUSE: {len(drops)} DROP clips in the 16-GPU eval plan", file=sys.stderr)
        for rec in drops:
            print(f"  {rec}", file=sys.stderr)
        return 2

    out = Path(args.out)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    holdout = [r for r in rows if r["split"] == "holdout"]
    train = [r for r in rows if r["split"] == "train"]
    print(
        f"\nOK {len(rows)} clips all KEEP "
        f"(holdout {len(holdout)}, train-eval {len(train)}) -> {out}"
    )

    if not args.prepare:
        return 0

    from thinker_wm.periodic_eval import prepare as prepare_mod

    cache_root = Path(pe_cfg.cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    support = cache_root / "_prepare"
    support.mkdir(parents=True, exist_ok=True)
    config_dump = support / "v8_config_dump.yaml"
    OmegaConf.save(config, config_dump)

    seen = set()
    for rank, asg in enumerate(assignments):
        default_bs = max(1, int(config.training.batch_size) // 4)
        batch_size = int(asg.batch_size) if asg.batch_size is not None else default_bs
        spec = build_source_spec(config, pe_cfg, asg, batch_size=batch_size)
        key = spec_key(spec)
        if key in seen:
            continue
        seen.add(key)
        spec_path = support / f"{key}.spec.json"
        spec_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False, default=str))
        out_dir = cache_root / key
        print(f"prepare rank{rank} {asg.split} {asg.source_name} shard={asg.shard_ordinal} -> {key}")
        rc = prepare_mod.run(config_dump, spec_path, out_dir)
        if rc != 0:
            print(f"prepare failed rc={rc} key={key}", file=sys.stderr)
            return rc
    print(f"prepared {len(seen)} cache dirs under {cache_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
