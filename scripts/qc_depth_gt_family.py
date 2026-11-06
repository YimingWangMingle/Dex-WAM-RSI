#!/usr/bin/env python3
"""Re-QC Agi DA3 / Ego FFS depth sidecars. Drop whole clips, no downweight.

Agi (v4): drop if ANY of
  - background median |Δz| > 2.5 cm
  - dump-time DA3-vs-sensor AbsRel (13-frame, in metadata.qc) > 0.10
  - near-motion DA3-vs-sensor AbsRel > 0.25 when enough valid sensor
    pixels exist (optional; NaN skips this gate — no tar reload by default)

Sensor is a referee only, never pasted as GT. Holes do not vote.

Ego (v4): drop if ANY quality fail (no longer wait for jitter):
  flying>1.5% OR hole_bg>5% OR close_valid<0.80 OR valid<0.90
  OR jitter_bg > 5 cm.

Holes are mm==0 only. z>=4 m is not a hole. No pixel downweight.

Writes a report dir. Does not rewrite selections unless --apply is passed
(then backups each ``depth_selections*.jsonl`` to ``*.pre_qc_v4.jsonl``).

  python scripts/qc_depth_gt_family.py
  python scripts/qc_depth_gt_family.py --workers 8 --apply
  python scripts/qc_depth_gt_family.py --reuse <old_qc_clips.jsonl> --apply
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter

ROOTS = [
    ("agi", Path("/data/xiongwei.zhao/agibot_da3_depth/v1")),
    ("agi", Path("/data/xiongwei.zhao/agibot_da3_depth_fulltimeline/v1")),
    ("ego", Path("/data/xiongwei.zhao/egoscale_ffs_depth/v1")),
    ("ego", Path("/data/xiongwei.zhao/egoscale_ffs_depth/holdout_v1")),
]
OUT = Path("/data/xiongwei.zhao/thinkerwam-depth/outputs/qc_depth_gt_v4")

Z_NEAR = 0.15
Z_FAR = 4.0
CLOSE_M = 0.6
JITTER_AGI = 0.025
JITTER_EGO = 0.05
HOLE_BG = 0.03
HOLE_MOT = 0.15
HOLE_BG_EGO = 0.05
FLYING = 0.02
FLYING_EGO = 0.015
VALID_MIN = 0.90
CLOSE_MIN = 0.80
ABSREL_DUMP = 0.10
ABSREL_NEAR_MOT = 0.25
NEAR_REF_M = 1.0
ABSREL_ALL = 0.15
ABSREL_MOT = 0.30
MOTION_DZ = 0.10
MOTION_RESID = 0.05
BLUR_KS = 41


def _fam(source: str, tag: str) -> str:
    s = (source or "").lower()
    if tag == "agi" or "agibot" in s:
        return "agi"
    return "ego"


def load_mm(ep_dir: Path, native: list[int]) -> np.ndarray | None:
    meta_path = ep_dir / "metadata.json"
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    frame_index = meta.get("frame_index")
    if isinstance(frame_index, list) and frame_index:
        native_to_row = {int(n): r for r, n in enumerate(frame_index)}
        if any(i not in native_to_row for i in native):
            return None
        rows = np.asarray([native_to_row[i] for i in native], np.int64)
    else:
        shape = meta.get("shape") or []
        n = int(shape[0]) if shape else 0
        if n <= 0 or any(i < 0 or i >= n for i in native):
            return None
        rows = np.asarray(native, np.int64)

    zarr_path = ep_dir / "depth.zarr"
    npz_path = ep_dir / "depth.npz"
    if zarr_path.exists():
        try:
            import zarr

            arr = zarr.open(str(zarr_path), mode="r")
            return np.asarray(arr.oindex[rows])
        except Exception:
            pass
    if npz_path.is_file():
        with np.load(npz_path) as npz:
            return np.asarray(npz["depth"][rows])
    return None


def motion_mask(z: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Object-ish motion: |Δz| residual after large box, no RGB."""
    t, h, w = z.shape
    out = np.zeros((t, h, w), dtype=bool)
    if t < 2:
        return out
    pair = valid[1:] & valid[:-1]
    dz = np.abs(z[1:] - z[:-1])
    moving = pair & (dz > MOTION_DZ)
    pad = BLUR_KS // 2
    for i in range(dz.shape[0]):
        lf = uniform_filter(dz[i], size=BLUR_KS, mode="nearest")
        moving[i] = moving[i] & ((dz[i] - lf) > MOTION_RESID)
    out[1:] |= moving
    out[:-1] |= moving
    return out


def flying_frac(z: np.ndarray, valid: np.ndarray) -> float:
    if not valid.any():
        return float("nan")
    step = max(1, z.shape[0] // 8)
    z = z[::step]
    valid = valid[::step]
    med = uniform_filter(z, size=(1, 5, 5), mode="nearest")
    rel = np.abs(z - med) / np.clip(z, 1e-3, None)
    fly = valid & (rel > 0.25) & (np.abs(z - med) > 0.15)
    return float(fly[valid].mean())


def interior_mask(mm: np.ndarray) -> np.ndarray:
    """Drop letterbox / pad: rows/cols that are almost all zero."""
    row_ok = (mm > 0).mean(axis=(0, 2)) > 0.05
    col_ok = (mm > 0).mean(axis=(0, 1)) > 0.05
    return row_ok[:, None] & col_ok[None, :]


def clip_metrics(mm: np.ndarray) -> dict:
    interior = interior_mask(mm)
    z = mm.astype(np.float32) / 1000.0
    valid = interior & (mm > 0) & np.isfinite(z) & (z > Z_NEAR) & (z < Z_FAR)
    hole = interior & (mm == 0)
    mot = motion_mask(z, valid)
    bg = valid & ~mot
    hole_bg = float(hole[~mot & interior].mean()) if (interior & ~mot).any() else float("nan")
    hole_mot = float(hole[mot].mean()) if mot.any() else float("nan")

    jitter_bg = float("nan")
    if z.shape[0] >= 2:
        pair = bg[1:] & bg[:-1]
        if pair.any():
            jitter_bg = float(np.median(np.abs(z[1:] - z[:-1])[pair]))

    near = interior & (z < CLOSE_M)
    close_valid = float(valid[near].mean()) if near.any() else float("nan")
    return {
        "n_frames": int(z.shape[0]),
        "valid_frac": float(valid[:, interior].mean()) if interior.any() else float("nan"),
        "motion_frac": float(mot[:, interior].mean()) if interior.any() else float("nan"),
        "hole_bg": hole_bg,
        "hole_motion": hole_mot,
        "jitter_bg_m": jitter_bg,
        "close_valid_frac": close_valid,
        "close_frac": float(near.mean()) if interior.any() else float("nan"),
        "flying_frac": flying_frac(z, valid),
        "absrel_all": float("nan"),
        "absrel_motion": float("nan"),
        "absrel_dump": float("nan"),
        "absrel_near_motion": float("nan"),
        "sensor_near_overlap": float("nan"),
        "depth_p50_m": float(np.median(z[valid])) if valid.any() else float("nan"),
    }


def decide(fam: str, m: dict) -> tuple[bool, str]:
    reasons = []
    jitter = m["jitter_bg_m"]
    if fam == "agi":
        if np.isfinite(jitter) and jitter > JITTER_AGI:
            reasons.append(f"jitter_bg={jitter:.3f}>{JITTER_AGI}")
        dump = m.get("absrel_dump", float("nan"))
        if np.isfinite(dump) and dump > ABSREL_DUMP:
            reasons.append(f"absrel_dump={dump:.3f}>{ABSREL_DUMP}")
        near = m.get("absrel_near_motion", float("nan"))
        ov = m.get("sensor_near_overlap", float("nan"))
        if np.isfinite(near) and np.isfinite(ov) and ov >= 0.05 and near > ABSREL_NEAR_MOT:
            reasons.append(f"absrel_near_mot={near:.3f}>{ABSREL_NEAR_MOT}")
        if not reasons:
            return False, ""
        return True, "+".join(reasons)

    if np.isfinite(jitter) and jitter > JITTER_EGO:
        reasons.append(f"jitter_bg={jitter:.3f}>{JITTER_EGO}")
    if np.isfinite(m["hole_bg"]) and m["hole_bg"] > HOLE_BG_EGO:
        reasons.append(f"hole_bg={m['hole_bg']:.3f}")
    if np.isfinite(m["hole_motion"]) and m["hole_motion"] > HOLE_MOT:
        reasons.append(f"hole_motion={m['hole_motion']:.3f}")
    if np.isfinite(m["flying_frac"]) and m["flying_frac"] > FLYING_EGO:
        reasons.append(f"flying={m['flying_frac']:.3f}")
    if np.isfinite(m["valid_frac"]) and m["valid_frac"] < VALID_MIN:
        reasons.append(f"valid={m['valid_frac']:.3f}")
    if np.isfinite(m["close_valid_frac"]) and m["close_valid_frac"] < CLOSE_MIN:
        reasons.append(f"close_valid={m['close_valid_frac']:.3f}")
    if not reasons:
        return False, ""
    return True, "+".join(reasons)


def _jobs_for_root(fam_tag: str, root: Path) -> list[dict]:
    jobs = []
    paths = [root / "depth_selections.jsonl", *sorted(root.glob("depth_selections.shard*.jsonl"))]
    seen = set()
    for path in paths:
        if not path.is_file():
            continue
        for line in path.open():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            eid = row.get("episode_id")
            src = row.get("source")
            clips = row.get("clips")
            if not eid or not src or not isinstance(clips, list):
                continue
            key = str(eid)
            if key in seen:
                continue
            seen.add(key)
            if not bool(row.get("prod_keep", False)):
                continue
            ep_dir = root / str(src) / key.replace(":", "_")
            jobs.append(
                {
                    "fam": _fam(str(src), fam_tag),
                    "root": str(root),
                    "source": str(src),
                    "episode_id": key,
                    "ep_dir": str(ep_dir),
                    "clips": [[int(i) for i in c] for c in clips],
                }
            )
    return jobs


def _eval_job(job: dict) -> dict:
    ep_dir = Path(job["ep_dir"])
    clip_rows = []
    keep_clips = []
    any_drop = False
    reasons = []
    union = sorted({int(i) for clip in job["clips"] for i in clip})
    packed = load_mm(ep_dir, union) if union else None
    native_to_row = {n: r for r, n in enumerate(union)}
    dump_absrel = _dump_absrel(ep_dir)
    for clip in job["clips"]:
        if packed is None:
            rec = {"ok": False, "drop": True, "reason": "unreadable", "n": len(clip)}
            clip_rows.append(rec)
            any_drop = True
            reasons.append("unreadable")
            continue
        rows = [native_to_row[int(i)] for i in clip]
        mm = packed[rows]
        m = clip_metrics(mm)
        m["absrel_dump"] = dump_absrel
        drop, reason = decide(job["fam"], m)
        rec = {"ok": True, "drop": drop, "reason": reason, **m}
        clip_rows.append(rec)
        if drop:
            any_drop = True
            reasons.append(reason)
        else:
            keep_clips.append(clip)
    return {
        "episode_id": job["episode_id"],
        "source": job["source"],
        "fam": job["fam"],
        "root": job["root"],
        "n_clips": len(job["clips"]),
        "n_keep_clips": len(keep_clips),
        "drop_episode": bool(len(keep_clips) == 0),
        "had_clip_drop": any_drop,
        "reasons": reasons,
        "keep_clips": keep_clips,
        "clips": clip_rows,
    }


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and math.isfinite(float(x))]
    return float(sum(xs) / len(xs)) if xs else float("nan")


def summarize(rows: list[dict]) -> dict:
    out: dict = {"n_episodes": len(rows)}
    by = defaultdict(list)
    for r in rows:
        by[r["fam"]].append(r)
        by["all"].append(r)
    for tag, items in by.items():
        clip_m = []
        for r in items:
            for c in r["clips"]:
                if c.get("ok"):
                    clip_m.append(c)
        dropped = [r for r in items if r["drop_episode"]]
        out[tag] = {
            "n_episodes": len(items),
            "n_drop_episodes": len(dropped),
            "drop_episode_frac": len(dropped) / max(len(items), 1),
            "n_clips": sum(r["n_clips"] for r in items),
            "n_keep_clips": sum(r["n_keep_clips"] for r in items),
            "jitter_bg_m": _mean([c["jitter_bg_m"] for c in clip_m]),
            "hole_bg": _mean([c["hole_bg"] for c in clip_m]),
            "hole_motion": _mean([c["hole_motion"] for c in clip_m]),
            "flying_frac": _mean([c["flying_frac"] for c in clip_m]),
            "valid_frac": _mean([c["valid_frac"] for c in clip_m]),
            "close_valid_frac": _mean([c.get("close_valid_frac") for c in clip_m]),
        }
    return out


def _dump_absrel(ep_dir: Path) -> float:
    meta_path = ep_dir / "metadata.json"
    if not meta_path.is_file():
        return float("nan")
    try:
        qc = (json.loads(meta_path.read_text()).get("qc") or {})
        raw = qc.get("absrel_med")
        if isinstance(raw, (int, float)) and math.isfinite(float(raw)):
            return float(raw)
    except (OSError, json.JSONDecodeError, TypeError):
        return float("nan")
    return float("nan")


def _selection_files(root: Path) -> list[Path]:
    files = [root / "depth_selections.jsonl", *sorted(root.glob("depth_selections.shard*.jsonl"))]
    return [p for p in files if p.is_file()]


def _backup_name(src: Path) -> Path:
    return src.with_name(src.name.replace(".jsonl", ".pre_qc_v4.jsonl"))


def rescore_cached(job: dict, old: dict) -> dict:
    """Reuse clip metrics from a prior scan; re-run v4 decide() + dump AbsRel."""
    old_clips = old.get("clips") or []
    if len(old_clips) != len(job["clips"]):
        return _eval_job(job)
    dump_absrel = _dump_absrel(Path(job["ep_dir"]))
    clip_rows = []
    keep_clips = []
    reasons = []
    any_drop = False
    for clip, oc in zip(job["clips"], old_clips):
        if not oc.get("ok"):
            rec = {
                "ok": False,
                "drop": True,
                "reason": oc.get("reason") or "unreadable",
                "n": len(clip),
            }
            clip_rows.append(rec)
            any_drop = True
            reasons.append(str(rec["reason"]))
            continue
        m = {
            "n_frames": oc.get("n_frames"),
            "valid_frac": oc.get("valid_frac"),
            "motion_frac": oc.get("motion_frac"),
            "hole_bg": oc.get("hole_bg"),
            "hole_motion": oc.get("hole_motion"),
            "jitter_bg_m": oc.get("jitter_bg_m"),
            "close_valid_frac": oc.get("close_valid_frac"),
            "close_frac": oc.get("close_frac"),
            "flying_frac": oc.get("flying_frac"),
            "absrel_all": oc.get("absrel_all", float("nan")),
            "absrel_motion": oc.get("absrel_motion", float("nan")),
            "absrel_dump": dump_absrel,
            "absrel_near_motion": oc.get("absrel_near_motion", float("nan")),
            "sensor_near_overlap": oc.get("sensor_near_overlap", float("nan")),
            "depth_p50_m": oc.get("depth_p50_m"),
        }
        drop, reason = decide(job["fam"], m)
        rec = {"ok": True, "drop": drop, "reason": reason, **m}
        clip_rows.append(rec)
        if drop:
            any_drop = True
            reasons.append(reason)
        else:
            keep_clips.append(clip)
    return {
        "episode_id": job["episode_id"],
        "source": job["source"],
        "fam": job["fam"],
        "root": job["root"],
        "n_clips": len(job["clips"]),
        "n_keep_clips": len(keep_clips),
        "drop_episode": bool(len(keep_clips) == 0),
        "had_clip_drop": any_drop,
        "reasons": reasons,
        "keep_clips": keep_clips,
        "clips": clip_rows,
        "rescored": True,
    }


def apply_selections(rows: list[dict]) -> None:
    by_root: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        by_root[r["root"]][r["episode_id"]] = r
    for root_s, updates in by_root.items():
        root = Path(root_s)
        for src in _selection_files(root):
            bak = _backup_name(src)
            if not bak.exists():
                bak.write_bytes(src.read_bytes())
            out_lines = []
            n_hit = 0
            for line in src.open():
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                eid = str(row.get("episode_id") or "")
                u = updates.get(eid)
                if u is not None:
                    n_hit += 1
                    row["prod_keep_pre_qc_v4"] = bool(row.get("prod_keep", False))
                    row["qc_v4_drop"] = bool(u["drop_episode"])
                    row["qc_v4_reasons"] = u["reasons"]
                    if u["drop_episode"]:
                        row["prod_keep"] = False
                    elif u["had_clip_drop"]:
                        row["clips"] = u["keep_clips"]
                        row["prod_keep"] = True
                out_lines.append(json.dumps(row, ensure_ascii=False))
            tmp = src.with_name(src.name + ".qc_v4.tmp")
            tmp.write_text("\n".join(out_lines) + "\n")
            os.replace(tmp, src)
            print(f"applied {src} hits={n_hit} (backup {bak})", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="debug: first N jobs")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument(
        "--reuse",
        type=Path,
        default=None,
        help="prior qc_clips.jsonl: reuse metrics, re-run v4 decide + dump AbsRel",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cached: dict[tuple[str, str], dict] = {}
    if args.reuse is not None:
        with args.reuse.open() as fp:
            for line in fp:
                if not line.strip():
                    continue
                rec = json.loads(line)
                cached[(str(rec["root"]), str(rec["episode_id"]))] = rec
        print(f"reuse {len(cached)} episodes from {args.reuse}", flush=True)

    jobs = []
    for tag, root in ROOTS:
        if not (root / "depth_selections.jsonl").is_file() and not list(
            root.glob("depth_selections.shard*.jsonl")
        ):
            print("skip missing", root, flush=True)
            continue
        part = _jobs_for_root(tag, root)
        print(f"index {root} keep_eps={len(part)}", flush=True)
        jobs.extend(part)
    if args.limit:
        jobs = jobs[: int(args.limit)]

    rows = []
    need_eval = []
    for job in jobs:
        old = cached.get((job["root"], job["episode_id"]))
        if old is not None:
            rows.append(rescore_cached(job, old))
        else:
            need_eval.append(job)
    print(
        f"rescored={len(rows)} eval={len(need_eval)} workers={args.workers}",
        flush=True,
    )

    done = 0
    if need_eval:
        with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
            futs = [pool.submit(_eval_job, j) for j in need_eval]
            for fut in as_completed(futs):
                rows.append(fut.result())
                done += 1
                if done % 200 == 0 or done == len(need_eval):
                    print(f"  {done}/{len(need_eval)}", flush=True)

    rows.sort(key=lambda r: (r["fam"], r["source"], r["episode_id"]))
    jsonl = args.out / "qc_clips.jsonl"
    with jsonl.open("w") as fp:
        for r in rows:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = summarize(rows)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)

    if args.apply:
        apply_selections(rows)
    else:
        print("dry-run: not rewriting depth_selections.jsonl (pass --apply)", flush=True)
    print("wrote", jsonl, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
