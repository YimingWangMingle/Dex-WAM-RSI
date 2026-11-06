#!/usr/bin/env python3
"""EgoScale FFS metric depth dump, aligned to training RGB (480x640 pad).

uint16 mm, left eye only, episode timeline. 4-way shard + resume.

  python scripts/dump_egoscale_ffs_depth.py --make-jobs
  CUDA_VISIBLE_DEVICES=0 python scripts/dump_egoscale_ffs_depth.py --shard 0/4
  python scripts/dump_egoscale_ffs_depth.py --status
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
import shutil
import sqlite3
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoke_egoscale_aligned import (  # noqa: E402
    DEPTH_CLIP_M,
    FFS_ROOT,
    NEAR_M,
    TRAIN_H,
    TRAIN_W,
    WEIGHTS,
    colorize,
    decode_frame,
    depth_from_pair,
    edge_alignment,
    homography_selfcheck,
    rectify_setup,
    train_transform_rgb,
    zaxis_denominator,
)

OUT = Path("/data/xiongwei.zhao/egoscale_ffs_depth/v1")
INFER_WH = (640, 360)
SEED = 20260823
QC_N = 13
# Clip mode: dump only the frames training actually decodes. Production config
# (stage2_20260818_v1) samples k=1 windows of WIN+1 video frames at native
# stride RATIO. We store CLIPS_PER_EP such windows per episode plus a companion
# depth_selections.jsonl so the depth branch samples exactly these windows.
CLIPS_PER_EP = 3
WIN = 48
RATIO = 6
MIN_VALID = 0.70
MIN_EDGE = 0.70
MAX_BREATH = 0.05
MED_LO, MED_HI = 0.15, 8.0
CATALOG_ROOT = Path("/data/jiyuan.wang/EpisodeCatalogs")
SOURCES = {
    "egoscale_first500h_shard00": {
        "catalog": CATALOG_ROOT / "_20260816/egoscale_first500h_shards_v1/egoscale_first500h.shard00_of10.20260816_shards_v1.episode_catalog.jsonl",
        "ranges": Path("/data/chen.liao/thinkerwm-pretrain/tmp/sampling_ranges/segmented_20260818_v1/egoscale_first500h_shard00.train.ranges.sqlite"),
        "holdout_ranges": Path("/data/chen.liao/thinkerwm-pretrain/tmp/sampling_ranges/segmented_20260818_v1/egoscale_first500h_shard00.holdout.ranges.sqlite"),
        "quota": 500,
    },
    "egoscale_first500h_remaining90pct": {
        "catalog": CATALOG_ROOT / "_20260816/egoscale_first500h_shards_v1/egoscale_first500h.remaining90pct.20260816_shards_v1.episode_catalog.jsonl",
        "ranges": Path("/data/chen.liao/thinkerwm-pretrain/tmp/sampling_ranges/segmented_20260818_v1/egoscale_first500h_remaining90pct.train.ranges.sqlite"),
        "holdout_ranges": Path("/data/chen.liao/thinkerwm-pretrain/tmp/sampling_ranges/segmented_20260818_v1/egoscale_first500h_remaining90pct.holdout.ranges.sqlite"),
        "quota": 4500,
    },
    "egoscale_second1700h_batch01": {
        "catalog": CATALOG_ROOT / "_20260820/egoscale_second1700h_batch01_full_v1/egoscale_second1700h_batch01.full.20260820_full_v1.episode_catalog.jsonl",
        "ranges": Path("/data/chen.liao/thinkerwm-pretrain/tmp/sampling_ranges/segmented_20260818_v1/egoscale_second1700h_batch01.train.ranges.sqlite"),
        "holdout_ranges": Path("/data/chen.liao/thinkerwm-pretrain/tmp/sampling_ranges/segmented_20260818_v1/egoscale_second1700h_batch01.holdout.ranges.sqlite"),
        "quota": 10200,
    },
    "egoscale_second1700h_batch02": {
        "catalog": CATALOG_ROOT / "_20260821/egoscale_second1700h_batch02_full_v1/egoscale_second1700h_batch02.full.20260821_full_v1.episode_catalog.jsonl",
        "ranges": Path("/data/chen.liao/thinkerwm-pretrain/tmp/sampling_ranges/segmented_20260818_v1/egoscale_second1700h_batch02.train.ranges.sqlite"),
        "holdout_ranges": Path("/data/chen.liao/thinkerwm-pretrain/tmp/sampling_ranges/segmented_20260818_v1/egoscale_second1700h_batch02.holdout.ranges.sqlite"),
        "quota": 4800,
    },
}


def train_ids(ranges_path: Path) -> set[str]:
    con = sqlite3.connect(f"file:{ranges_path}?mode=ro", uri=True)
    ids = {r[0] for r in con.execute("SELECT episode_id FROM episode_ranges")}
    con.close()
    return ids


def rec_to_job(rec: dict, source: str) -> dict:
    views = rec["video"]["meta"]["view_paths"]
    return {
        "episode_id": rec["episode_id"],
        "source": source,
        "group": rec["meta"]["source_group_key"],
        "left_mp4": views["base_0"],
        "right_mp4": views["base_1"],
        "calib": rec["meta"]["calibration"]["intrinsics"],
        "n_src": int(rec["video"]["count"]),
        "fps": float(rec["video"]["meta"].get("fps", 30.0)),
    }


def sample_source(source: str, n: int, rng: random.Random) -> list[dict]:
    spec = SOURCES[source]
    keep = train_ids(spec["ranges"])
    by_group: dict[str, list] = {}
    with open(spec["catalog"]) as f:
        for line in f:
            rec = json.loads(line)
            if rec["episode_id"] not in keep:
                continue
            if int(rec["video"]["count"]) < 289:
                continue
            job = rec_to_job(rec, source)
            by_group.setdefault(job["group"], []).append(job)
    groups = list(by_group)
    rng.shuffle(groups)
    picked, used = [], set()
    for g in groups:
        rng.shuffle(by_group[g])
        job = by_group[g][0]
        picked.append(job)
        used.add(job["episode_id"])
        if len(picked) >= n:
            return picked
    leftovers = [j for g in groups for j in by_group[g] if j["episode_id"] not in used]
    rng.shuffle(leftovers)
    picked.extend(leftovers[: n - len(picked)])
    return picked[:n]


def holdout_jobs() -> list[dict]:
    """All official holdout episodes with video.count >= 289. No quota."""
    jobs: list[dict] = []
    for source, spec in SOURCES.items():
        keep = train_ids(spec["holdout_ranges"])
        n_ok = 0
        with open(spec["catalog"]) as f:
            for line in f:
                rec = json.loads(line)
                if rec["episode_id"] not in keep:
                    continue
                if int(rec["video"]["count"]) < 289:
                    continue
                jobs.append(rec_to_job(rec, source))
                n_ok += 1
        print(f"holdout {source} {n_ok}", flush=True)
    jobs.sort(key=lambda j: (j["source"], j["episode_id"]))
    return jobs


def make_jobs() -> list[dict]:
    rng = random.Random(SEED)
    jobs = []
    for source, spec in SOURCES.items():
        part = sample_source(source, spec["quota"], rng)
        print(f"sampled {source} {len(part)}/{spec['quota']}", flush=True)
        jobs.extend(part)
    rng.shuffle(jobs)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "jobs.json").write_text(json.dumps(jobs, separators=(",", ":")))
    verify = []
    seen_src = {}
    for job in jobs:
        seen_src.setdefault(job["source"], []).append(job)
    for source, part in seen_src.items():
        verify.extend(part[:2])
    (OUT / "verify_jobs.json").write_text(json.dumps(verify, indent=2))
    print(f"wrote {len(jobs)} jobs -> {OUT / 'jobs.json'}; verify={len(verify)}", flush=True)
    return jobs


def ep_dir(job: dict) -> Path:
    return OUT / job["source"] / job["episode_id"].replace(":", "_")


def load_ranges_map(ranges_path: Path) -> dict[str, tuple[list, int]]:
    con = sqlite3.connect(f"file:{ranges_path}?mode=ro", uri=True)
    m: dict[str, tuple[list, int]] = {}
    for eid, intervals, elen in con.execute(
        "SELECT episode_id, intervals, episode_len FROM episode_ranges"
    ):
        m[eid] = (json.loads(intervals), int(elen))
    con.close()
    return m


def plan_clips(episode_id: str, intervals: list, episode_len: int, n: int = CLIPS_PER_EP) -> list[dict]:
    """Choose n window starts spread over the valid-start domain and expand each
    into the exact native video_frame_indices training would decode (k=1)."""
    dom = [(int(lo), int(hi)) for lo, hi in intervals if int(hi) >= int(lo)]
    total = sum(hi - lo + 1 for lo, hi in dom)
    if total <= 0:
        return []
    n = min(n, total)
    limit = episode_len - 1
    starts: list[int] = []
    for k in range(n):
        q = int((k + 0.5) * total / n)
        acc = 0
        for lo, hi in dom:
            w = hi - lo + 1
            if q < acc + w:
                starts.append(lo + (q - acc))
                break
            acc += w
    clips = []
    for s in sorted(set(starts)):
        idx = [min(s + RATIO * j, limit) for j in range(WIN + 1)]
        clips.append({"start": int(s), "video_frame_indices": idx})
    return clips


def append_selection(row: dict) -> None:
    path = OUT / "depth_selections.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)


def append_progress(row: dict) -> None:
    path = OUT / "progress.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)


def write_depth(path: Path, arr: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import zarr
        from numcodecs import Blosc

        if path.exists():
            shutil.rmtree(path)
        z = zarr.open(
            str(path),
            mode="w",
            shape=arr.shape,
            chunks=(min(32, arr.shape[0]), arr.shape[1], arr.shape[2]),
            dtype="u2",
            compressor=Blosc(cname="zstd", clevel=3, shuffle=1),
        )
        z[:] = arr
        return "zarr_zstd"
    except Exception as exc:
        npz = path.with_suffix(".npz")
        np.savez_compressed(npz, depth=arr)
        return f"npz:{type(exc).__name__}"


def qc_indices(n: int) -> list[int]:
    return [int(round(i * (n - 1) / (QC_N - 1))) for i in range(QC_N)]


def qc_episode(depth: np.ndarray, rgb0: np.ndarray, qc_idx: list[int], pad_box, selfcheck: float) -> dict:
    y0, nh, x0, nw = pad_box
    region = np.zeros((TRAIN_H, TRAIN_W), bool)
    region[y0:y0 + nh, x0:x0 + nw] = True
    qc = depth[np.asarray(qc_idx, np.int64)]
    valid = qc > 0
    valid_in = float(valid[:, region].mean()) if valid.size else 0.0
    pad_zero = bool((qc[:, ~region] == 0).all()) if qc.size else False
    meds = []
    edges = []
    for t, idx in enumerate(qc_idx):
        mm = qc[t]
        m = mm[region] > 0
        if m.sum() < 100:
            continue
        meds.append(float(np.median(mm[region][m]) / 1000.0))
        if t == 0:
            edges.append(edge_alignment(rgb0, mm, pad_box).get("precision_2px", float("nan")))
    meds_a = np.asarray(meds, np.float32)
    if len(meds_a) >= 2:
        a = meds_a / np.median(meds_a)
        breath = float(np.median(np.abs(np.log(np.clip(a, 1e-6, None)))))
    else:
        breath = float("nan")
    med = float(np.median(meds_a)) if len(meds_a) else float("nan")
    edge = float(edges[0]) if edges else float("nan")
    # NOTE: breath/edge over full-episode QC frames conflate scene change with
    # instability, so they are diagnostic only. Temporal stability is validated
    # separately on consecutive frames. Hard gate keeps geometry + range checks.
    keep = (
        valid_in >= MIN_VALID
        and pad_zero
        and (not np.isfinite(med) or MED_LO <= med <= MED_HI)
        and (not np.isfinite(selfcheck) or selfcheck < 1.0)
    )
    return {
        "valid_fraction_in_region": valid_in,
        "pad_zero_ok": pad_zero,
        "edge_precision_2px_diag": edge,
        "scene_depth_var_diag": breath,
        "depth_median_m": med,
        "homography_selfcheck_mad": selfcheck,
        "prod_keep": bool(keep),
        "reason": "" if keep else "gate_fail",
    }


_setup_cache: dict[str, tuple[dict, np.ndarray]] = {}


def get_setup(calib: str, infer_wh: tuple[int, int] | None = None) -> tuple[dict, np.ndarray]:
    key = f"{calib}|{infer_wh}"
    hit = _setup_cache.get(key)
    if hit is not None:
        return hit
    setup = rectify_setup(Path(calib), size=infer_wh)
    denom = zaxis_denominator(setup)
    denom = np.where(np.abs(denom) < 1e-6, 1.0, denom)
    _setup_cache[key] = (setup, denom)
    return setup, denom


def load_ffs(valid_iters: int):
    import torch

    sys.path.insert(0, str(FFS_ROOT))
    from core.utils.utils import InputPadder

    if not WEIGHTS.is_file():
        raise FileNotFoundError(WEIGHTS)
    model = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    model.args.valid_iters = valid_iters
    model.cuda().eval()
    return model, InputPadder


def dump_one(job: dict, model, InputPadder, valid_iters: int, infer_wh: tuple[int, int],
             ranges_map: dict[str, tuple[list, int]]) -> dict:
    dest = ep_dir(job)
    dest.mkdir(parents=True, exist_ok=True)
    meta_path = dest / "metadata.json"
    eid = job["episode_id"]
    rr = ranges_map.get(eid)
    if rr is None:
        return {"episode_id": eid, "source": job["source"], "status": "skip_no_ranges", "complete": False}
    intervals, elen = rr
    clips = plan_clips(eid, intervals, elen)
    if not clips:
        return {"episode_id": eid, "source": job["source"], "status": "skip_no_clip", "complete": False}
    needed = sorted({i for c in clips for i in c["video_frame_indices"]})
    if meta_path.is_file():
        old = json.loads(meta_path.read_text())
        if old.get("complete") and set(needed).issubset(set(old.get("frame_index", []))):
            return {**old, "episode_id": eid, "status": "skip_done"}
    n = int(job["n_src"])
    setup, denom = get_setup(job["calib"], infer_wh)
    t0 = time.time()
    cap_l = cv2.VideoCapture(job["left_mp4"])
    cap_r = cv2.VideoCapture(job["right_mp4"])
    need_set = set(needed)
    max_idx = needed[-1]
    depth = np.zeros((len(needed), TRAIN_H, TRAIN_W), np.uint16)
    row_of = {idx: r for r, idx in enumerate(needed)}
    pad_box = None
    t_dec = t_inf = 0.0
    rgb0 = None
    selfcheck = float("nan")
    got = 0
    import torch

    try:
        with torch.no_grad():
            for i in range(max_idx + 1):
                t = time.time()
                if i in need_set:
                    ok_l, bgr_l = cap_l.read()
                    ok_r, bgr_r = cap_r.read()
                else:
                    ok_l = cap_l.grab()
                    ok_r = cap_r.grab()
                    bgr_l = bgr_r = None
                t_dec += time.time() - t
                if not ok_l or not ok_r:
                    break
                if i not in need_set:
                    continue
                left = cv2.cvtColor(bgr_l, cv2.COLOR_BGR2RGB)
                right = cv2.cvtColor(bgr_r, cv2.COLOR_BGR2RGB)
                if rgb0 is None:
                    if left.shape[:2] != (1080, 1920):
                        raise RuntimeError(f"unexpected rgb shape {left.shape}")
                    selfcheck = homography_selfcheck(left, setup)
                    rgb0 = train_transform_rgb(left)
                mm, pad_box, sec = depth_from_pair(
                    left, right, setup, denom, model, valid_iters, InputPadder, infer_wh
                )
                depth[row_of[i]] = mm
                t_inf += sec
                got += 1
    finally:
        cap_l.release()
        cap_r.release()
    if got != len(needed):
        raise RuntimeError(f"decoded {got}/{len(needed)} needed frames (max_idx={max_idx}, n_src={n})")
    qc_idx = [row_of[needed[int(round(i * (len(needed) - 1) / (QC_N - 1)))]] for i in range(QC_N)]
    qc = qc_episode(depth, rgb0, qc_idx, pad_box, selfcheck)
    kind = write_depth(dest / "depth.zarr", depth)
    viz = np.concatenate([rgb0, colorize(depth[0].astype(np.float32) / 1000.0)], 1)
    cv2.imwrite(str(dest / "viz.png"), cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
    row = {
        "episode_id": eid,
        "source": job["source"],
        "group": job["group"],
        "status": "ok",
        "complete": True,
        "teacher": "Fast-FoundationStereo",
        "unit": "uint16_mm",
        "layout": "clips_native_index",
        "shape": list(depth.shape),
        "rgb_hw": [TRAIN_H, TRAIN_W],
        "infer_wh": list(infer_wh),
        "camera": "base_0_unrectified",
        "pad_box_yhxw": list(pad_box),
        "n_src": n,
        "episode_len": elen,
        "clips_per_episode": CLIPS_PER_EP,
        "win": WIN,
        "stride": RATIO,
        "clip_starts": [c["start"] for c in clips],
        "frame_index": needed,
        "codec": kind,
        "qc": qc,
        "prod_keep": qc["prod_keep"],
        "sec_decode": t_dec,
        "sec_infer": t_inf,
        "sec_per_frame": t_inf / max(got, 1),
        "sec_total": time.time() - t0,
    }
    meta_path.write_text(json.dumps(row, indent=2))
    append_selection({
        "episode_id": eid,
        "source": job["source"],
        "prod_keep": qc["prod_keep"],
        "clips": [c["video_frame_indices"] for c in clips],
    })
    return row


def is_complete(job: dict) -> bool:
    path = ep_dir(job) / "metadata.json"
    if not path.is_file():
        return False
    try:
        m = json.loads(path.read_text())
        # Only clip-layout dumps count as complete; legacy full-timeline dumps
        # are reprocessed (overwritten) into clip format.
        return bool(m.get("complete")) and m.get("layout") == "clips_native_index"
    except Exception:
        return False


def grouped_shard_jobs(jobs: list[dict], world: int) -> list[list[dict]]:
    """Assign whole session groups to one GPU, like AgiBot tar groups.

    EgoScale has no tar to scan; the win is keeping stereo maps/calib hot
    and avoiding split-brain cache across ranks.
    """
    pending = [j for j in jobs if not is_complete(j)]
    groups: dict[str, list[dict]] = {}
    for job in pending:
        groups.setdefault(job["group"], []).append(job)
    bins: list[list[dict]] = [[] for _ in range(world)]
    loads = [0] * world
    for group in sorted(groups.values(), key=lambda g: sum(int(j["n_src"]) for j in g), reverse=True):
        target = min(range(world), key=lambda i: loads[i])
        bins[target].extend(group)
        loads[target] += sum(int(j["n_src"]) for j in group)
    print(
        "session_grouped_plan "
        + json.dumps({"pending": len(pending), "groups": len(groups), "shard_jobs": [len(b) for b in bins], "shard_frames": loads}),
        flush=True,
    )
    return bins


def run_shard(jobs: list[dict], shard: int, world: int, valid_iters: int, infer_wh: tuple[int, int]) -> None:
    mine = grouped_shard_jobs(jobs, world)[shard]
    print(f"shard {shard}/{world} pending={len(mine)} infer_wh={infer_wh} "
          f"clips={CLIPS_PER_EP} win={WIN} stride={RATIO} "
          f"ranges={'holdout_ranges' if 'holdout' in Path(OUT).as_posix() else 'ranges'} out={OUT}",
          flush=True)
    ranges_map: dict[str, tuple[list, int]] = {}
    key = "holdout_ranges" if "holdout" in Path(OUT).as_posix() else "ranges"
    for src in {j["source"] for j in mine}:
        ranges_map.update(load_ranges_map(SOURCES[src][key]))
    model, InputPadder = load_ffs(valid_iters)
    done = 0
    for job in mine:
        eid = job["episode_id"]
        try:
            row = dump_one(job, model, InputPadder, valid_iters, infer_wh, ranges_map)
        except Exception as exc:
            row = {
                "episode_id": eid,
                "source": job["source"],
                "status": "fail",
                "complete": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"FAIL {eid} {row['error']}", flush=True)
        append_progress(row)
        done += 1
        print(
            f"[{done}/{len(mine)}] {eid} {row.get('status')} keep={row.get('prod_keep')} "
            f"T={len(row.get('frame_index', []))} {row.get('sec_total', 0):.0f}s "
            f"valid={row.get('qc', {}).get('valid_fraction_in_region')} "
            f"edge={row.get('qc', {}).get('edge_precision_2px_diag')}",
            flush=True,
        )


def status() -> None:
    rows = []
    for p in OUT.glob("*/*/metadata.json"):
        try:
            rows.append(json.loads(p.read_text()))
        except Exception:
            continue
    ok = [r for r in rows if r.get("status") == "ok"]
    keep = [r for r in ok if r.get("prod_keep")]
    fail = [r for r in rows if r.get("status") == "fail"]
    pf = [r["sec_per_frame"] for r in ok if r.get("sec_per_frame")]
    jobs_n = 0
    if (OUT / "jobs.json").is_file():
        jobs_n = len(json.loads((OUT / "jobs.json").read_text()))
    remain = max(jobs_n - len(rows), 0)
    sec_pf = float(np.median(pf)) if pf else 0.042
    print(
        json.dumps(
            {
                "jobs": jobs_n,
                "unique_done": len(rows),
                "ok": len(ok),
                "prod_keep": len(keep),
                "fail": len(fail),
                "keep_rate_ok": (len(keep) / len(ok)) if ok else None,
                "sec_per_frame_med": sec_pf,
                "remain": remain,
                "eta_4gpu_h": remain * 1290 * sec_pf / 3600 / 4,
                "out": str(OUT),
            },
            indent=2,
        )
    )


def verify_jobs(valid_iters: int, infer_wh: tuple[int, int]) -> int:
    """Multi-session alignment QC: 3 frames per verify episode, no full dump."""
    path = OUT / "verify_jobs.json"
    jobs = json.loads(path.read_text())
    model, InputPadder = load_ffs(valid_iters)
    out_dir = OUT / "_verify"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for job in jobs:
        setup, denom = get_setup(job["calib"], infer_wh)
        n = int(job["n_src"])
        idxs = [min(n - 1, max(0, int(round(i * (n - 1) / 2)))) for i in range(3)]
        root = Path(job["calib"]).parents[1]
        dest = out_dir / job["episode_id"].replace(":", "_")
        dest.mkdir(parents=True, exist_ok=True)
        frame_rows = []
        rgb0 = None
        pad_box = None
        selfcheck = float("nan")
        t0 = time.time()
        for k, idx in enumerate(idxs):
            left = decode_frame(Path(job["left_mp4"]), idx)
            right = decode_frame(Path(job["right_mp4"]), idx)
            if k == 0:
                selfcheck = homography_selfcheck(left, setup)
                rgb0 = train_transform_rgb(left)
            mm, pad_box, sec = depth_from_pair(
                left, right, setup, denom, model, valid_iters, InputPadder, infer_wh
            )
            valid = mm > 0
            y0, nh, x0, nw = pad_box
            region = np.zeros((TRAIN_H, TRAIN_W), bool)
            region[y0:y0 + nh, x0:x0 + nw] = True
            ea = edge_alignment(rgb0 if k == 0 else train_transform_rgb(left), mm, pad_box)
            frame_rows.append({
                "frame": idx,
                "sec": sec,
                "valid_fraction_in_region": float(valid[region].mean()),
                "pad_zero_ok": bool((mm[~region] == 0).all()),
                "depth_p02_p50_p98_m": np.percentile(mm[valid] / 1000.0, [2, 50, 98]).tolist()
                if valid.any() else [float("nan")] * 3,
                "edge_alignment": ea,
            })
            if k == 0:
                rgb = train_transform_rgb(left)
                viz = np.concatenate([rgb, colorize(mm.astype(np.float32) / 1000.0)], 1)
                cv2.imwrite(str(dest / "viz.png"), cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
        edges = [r["edge_alignment"].get("precision_2px", float("nan")) for r in frame_rows]
        valids = [r["valid_fraction_in_region"] for r in frame_rows]
        row = {
            "episode_id": job["episode_id"],
            "source": job["source"],
            "group": job["group"],
            "root": str(root),
            "homography_selfcheck_mad": selfcheck,
            "edge_precision_2px_mean": float(np.nanmean(edges)),
            "valid_fraction_mean": float(np.mean(valids)),
            "pad_zero_ok": all(r["pad_zero_ok"] for r in frame_rows),
            "sec_total": time.time() - t0,
            "frames": frame_rows,
        }
        rows.append(row)
        print(
            f"VERIFY {job['source']} {job['episode_id']} "
            f"edge={row['edge_precision_2px_mean']:.3f} valid={row['valid_fraction_mean']:.3f} "
            f"H={selfcheck:.5f} pad={row['pad_zero_ok']}",
            flush=True,
        )
    summary = {"infer_wh": list(infer_wh), "n": len(rows), "episodes": rows}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"n": len(rows), "out": str(out_dir / "summary.json")}, indent=2))
    return 0


def main() -> int:
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--make-jobs", action="store_true")
    parser.add_argument("--make-holdout-jobs", action="store_true")
    parser.add_argument("--shard", default="")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--valid-iters", type=int, default=8)
    parser.add_argument("--infer-wh", default="640x360")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    OUT = Path(args.out)
    train_v1 = Path("/data/xiongwei.zhao/egoscale_ffs_depth/v1")
    if args.make_holdout_jobs and OUT.resolve() == train_v1.resolve():
        raise SystemExit("refuse: --make-holdout-jobs would write into train dump v1; pass --out <new-dir>")
    if args.make_holdout_jobs:
        OUT.mkdir(parents=True, exist_ok=True)
        jobs = holdout_jobs()
        (OUT / "jobs.json").write_text(json.dumps(jobs, separators=(",", ":")))
        (OUT / "SPLIT.json").write_text(
            json.dumps(
                {
                    "kind": "official_holdout_never_in_stage2_train",
                    "n_jobs": len(jobs),
                    "do_not_overwrite": [
                        "/data/xiongwei.zhao/egoscale_ffs_depth/v1",
                        "/data/xiongwei.zhao/agibot_da3_depth/v1",
                        "/data/xiongwei.zhao/agibot_da3_depth_fulltimeline/v1",
                    ],
                    "ranges_field": "holdout_ranges",
                },
                indent=2,
            )
            + "\n"
        )
        print(f"wrote {OUT / 'jobs.json'} n={len(jobs)} out={OUT}", flush=True)
        return 0
    if args.make_jobs:
        make_jobs()
        return 0
    if args.status:
        status()
        return 0
    infer_wh = tuple(int(x) for x in args.infer_wh.lower().split("x"))
    if args.verify:
        return verify_jobs(args.valid_iters, infer_wh)
    if not args.shard:
        raise SystemExit("need --make-jobs, --status, --verify, or --shard i/n")
    shard, world = (int(x) for x in args.shard.split("/"))
    jobs = json.loads((OUT / "jobs.json").read_text())
    run_shard(jobs, shard, world, args.valid_iters, infer_wh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
