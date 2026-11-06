#!/usr/bin/env python3
"""8-GPU bake-off: DA3 vs VGGT vs AgiBot-Beta sensor on the same 13 frames.

Stage1 QC protocol: 49 RGB, score every 4th frame (13), per-frame median scale.

  python scripts/compare_da3_vggt.py --make-jobs
  CUDA_VISIBLE_DEVICES=0 python scripts/compare_da3_vggt.py --shard 0/8
  python scripts/compare_da3_vggt.py --reduce
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoke_da3_depth import (  # noqa: E402
    DA3_EXT,
    DA3_LOCAL,
    T4W_ROOT,
    _stub,
    decode_first_n,
    extract_tar_member,
)

VGGT_ROOT = "/code1/zxw_data/pretrain/third_party/vggt"
VGGT_CKPT = "/code1/zxw_data/checkpoint/VGGT/model.safetensors"
CATALOG = (
    "/data/jiyuan.wang/EpisodeCatalogs/_20260805/agibot_beta_effector_profiles_v1/"
    "agibot_beta_parallel_gripper.20260805_full_v1.episode_catalog.jsonl"
)
OUT = Path("/data/xiongwei.zhao/agibot_da3_depth/da3_vs_vggt")
N_RGB = 49
STRIDE = 4
H, W = 480, 640
NEAR, FAR = 0.1, 10.0


def vggt_preprocess_crop(frames: torch.Tensor, target_side: int = 518) -> torch.Tensor:
    t, _, h, w = frames.shape
    new_w = target_side
    new_h = round(h * (new_w / w) / 14) * 14
    x = F.interpolate(frames, size=(new_h, new_w), mode="bicubic", antialias=True, align_corners=False)
    x = x.clamp_(0.0, 1.0)
    if new_h > target_side:
        top = (new_h - target_side) // 2
        x = x[:, :, top:top + target_side, :]
    return x


def rec_to_job(rec: dict) -> dict:
    st = rec["video"]["storage"]
    return {
        "episode_id": rec["episode_id"],
        "task_id": rec["meta"]["task_id"],
        "tar_path": st["path"],
        "member": st.get("member", ""),
        "offset": int(st["offset"]),
        "size": int(st["size"]),
        "n_src": int(rec["video"]["count"]),
    }


def make_jobs(n: int) -> list[dict]:
    seen = set()
    jobs = []
    with open(CATALOG) as f:
        for line in f:
            rec = json.loads(line)
            if rec["video"]["count"] < N_RGB:
                continue
            tid = rec["meta"]["task_id"]
            if tid in seen:
                continue
            seen.add(tid)
            jobs.append(rec_to_job(rec))
            if len(jobs) == n:
                break
    if len(jobs) < n:
        raise RuntimeError(f"only {len(jobs)} unique tasks")
    return jobs


def load_sensor(tar_path: str, ep: str, indices: list[int]) -> np.ndarray:
    needed = {f"{ep}/depth/head_depth_{i:06d}.png": i for i in indices}
    got: dict[int, np.ndarray] = {}
    with tarfile.open(tar_path, mode="r|") as tf:
        for m in tf:
            if m.name not in needed:
                continue
            raw = tf.extractfile(m).read()
            d = np.asarray(Image.open(io.BytesIO(raw)))
            if d.dtype != np.uint16:
                raise RuntimeError(f"{m.name} dtype={d.dtype}")
            got[needed[m.name]] = d
            if len(got) == len(indices):
                break
    missing = [i for i in indices if i not in got]
    if missing:
        raise FileNotFoundError(f"{ep} missing sensor {missing[:6]}")
    stack = np.stack([got[i] for i in indices], 0).astype(np.float32) / 1000.0
    if stack.shape[1:] != (H, W):
        stack = np.stack(
            [cv2.resize(stack[t], (W, H), interpolation=cv2.INTER_NEAREST) for t in range(len(indices))],
            0,
        )
    return stack


def resize_depth(d: np.ndarray) -> np.ndarray:
    if d.shape == (H, W):
        return d.astype(np.float32)
    return cv2.resize(d.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)


def score(pred: np.ndarray, sen: np.ndarray) -> dict:
    """pred/sen [T,H,W] meters. Per-frame median scale (Stage1) + one global scale."""
    t = pred.shape[0]
    m_s = np.isfinite(sen) & (sen > NEAR) & (sen < FAR)
    absrels, deltas, scales = [], [], []
    aligned = np.zeros_like(pred)
    for i in range(t):
        p = resize_depth(pred[i])
        g = sen[i]
        m = m_s[i] & np.isfinite(p) & (p > 1e-4)
        if m.sum() < 100:
            continue
        s = float(np.median(g[m]) / max(float(np.median(p[m])), 1e-6))
        pa = p * s
        aligned[i] = pa
        rel = np.abs(pa[m] - g[m]) / np.clip(g[m], 1e-6, None)
        ratio = np.maximum(pa[m] / g[m], g[m] / pa[m])
        absrels.append(float(np.median(rel)))
        deltas.append(float((ratio < 1.25).mean()))
        scales.append(s)
    both = m_s & np.isfinite(pred) & (pred > 1e-4)
    if both.any():
        a_g = float(np.median(pred[both] / np.clip(sen[both], 1e-6, None)))
        pa = pred / max(a_g, 1e-6)
        rel_g = np.abs(pa[both] - sen[both]) / np.clip(sen[both], 1e-6, None)
        absrel_global = float(np.median(rel_g))
    else:
        a_g, absrel_global = float("nan"), float("nan")
    sc = np.asarray(scales, np.float32)
    breath = float(np.median(np.abs(np.log(sc / np.median(sc))))) if len(sc) else float("nan")
    return {
        "absrel_perframe_med": float(np.median(absrels)) if absrels else float("nan"),
        "delta125_mean": float(np.mean(deltas)) if deltas else float("nan"),
        "scale_breath_med": breath,
        "n_ok": len(absrels),
        "sensor_valid": float(m_s.mean()),
        "alpha_pred_over_sensor_global": a_g,
        "absrel_global_scale": absrel_global,
        "per_frame_absrel": absrels,
        "per_frame_scale": scales,
    }, aligned


def colorize(d: np.ndarray, lo: float, hi: float) -> np.ndarray:
    x = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    return cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[:, :, ::-1]


def load_da3(device):
    _stub()
    sys.path[:0] = [T4W_ROOT, DA3_EXT]
    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained(DA3_LOCAL, local_files_only=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device)


def load_vggt(device):
    sys.path.insert(0, VGGT_ROOT)
    from vggt.models.vggt import VGGT
    from safetensors.torch import load_file

    model = VGGT()
    missing, unexpected = model.load_state_dict(load_file(VGGT_CKPT), strict=False)
    if missing:
        print(f"vggt missing {len(missing)}", flush=True)
    if unexpected:
        print(f"vggt unexpected {len(unexpected)}", flush=True)
    return model.to(device).eval()


def run_da3(model, frames: list, device) -> np.ndarray:
    out = []
    with torch.no_grad():
        for rgb in frames:
            pred = model.inference([rgb], export_feat_layers=[])
            d = np.asarray(pred.depth, np.float32)
            if d.ndim == 3:
                d = d[0]
            out.append(resize_depth(d))
    return np.stack(out, 0)


def run_vggt(model, frames: list, device) -> np.ndarray:
    x = torch.from_numpy(np.stack(frames, 0)).permute(0, 3, 1, 2).float() / 255.0
    pp = vggt_preprocess_crop(x, 518).to(device)
    imgs = pp.unsqueeze(0)
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            tokens, patch_start = model.aggregator(imgs)
        with torch.cuda.amp.autocast(enabled=False):
            depth, _ = model.depth_head(tokens, images=imgs, patch_start_idx=patch_start)
    d = depth[0, ..., 0].float().cpu().numpy()
    return np.stack([resize_depth(d[t]) for t in range(d.shape[0])], 0)


def run_shard(jobs: list[dict], shard: int, world: int) -> None:
    mine = [j for i, j in enumerate(jobs) if i % world == shard]
    device = torch.device("cuda:0")
    print(f"shard {shard}/{world} n={len(mine)}", flush=True)
    da3 = load_da3(device)
    print("DA3 ready", flush=True)
    vggt = load_vggt(device)
    print("VGGT ready", flush=True)
    idxs = list(range(0, N_RGB, STRIDE))
    for job in mine:
        eid = job["episode_id"]
        ep = eid.split(":")[-1]
        dest = OUT / "eps" / eid.replace(":", "_")
        dest.mkdir(parents=True, exist_ok=True)
        raw = str(dest / "_tmp.mp4")
        print(f"== {eid} task={job['task_id']}", flush=True)
        t0 = time.time()
        extract_tar_member(job["tar_path"], job["offset"], job["size"], raw)
        frames49 = decode_first_n(raw, N_RGB)
        try:
            os.remove(raw)
        except OSError:
            pass
        frames13 = [frames49[i] for i in idxs]
        sen = load_sensor(job["tar_path"], ep, idxs)
        t_io = time.time() - t0
        t1 = time.time()
        d_da3 = run_da3(da3, frames13, device)
        t_da3 = time.time() - t1
        t2 = time.time()
        d_vggt = run_vggt(vggt, frames13, device)
        t_vggt = time.time() - t2
        m_da3, a_da3 = score(d_da3, sen)
        m_vggt, a_vggt = score(d_vggt, sen)
        row = {
            "episode_id": eid,
            "task_id": job["task_id"],
            "n_src": job["n_src"],
            "sec_io": t_io,
            "sec_da3": t_da3,
            "sec_vggt": t_vggt,
            "da3": {k: v for k, v in m_da3.items() if k not in ("per_frame_absrel", "per_frame_scale")},
            "vggt": {k: v for k, v in m_vggt.items() if k not in ("per_frame_absrel", "per_frame_scale")},
            "da3_per_frame_absrel": m_da3["per_frame_absrel"],
            "vggt_per_frame_absrel": m_vggt["per_frame_absrel"],
        }
        (dest / "metrics.json").write_text(json.dumps(row, indent=2))
        lo = min(np.percentile(a_da3[a_da3 > 0], 2) if (a_da3 > 0).any() else 0.3,
                 np.percentile(sen[sen > NEAR], 2))
        hi = max(np.percentile(a_da3[a_da3 > 0], 98) if (a_da3 > 0).any() else 2.0,
                 np.percentile(sen[sen > NEAR], 98))
        rgb = frames13[0]
        if rgb.shape[:2] != (H, W):
            rgb = cv2.resize(rgb, (W, H))
        strip = np.concatenate(
            [rgb, colorize(a_da3[0], lo, hi), colorize(a_vggt[0], lo, hi), colorize(sen[0], lo, hi)],
            axis=1,
        )
        cv2.putText(strip, "RGB | DA3 | VGGT | sensor", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(str(dest / "compare.png"), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
        print(
            f"  da3 absrel={m_da3['absrel_perframe_med']:.3f} d125={m_da3['delta125_mean']:.3f} "
            f"breath={m_da3['scale_breath_med']:.3f} {t_da3:.1f}s | "
            f"vggt absrel={m_vggt['absrel_perframe_med']:.3f} d125={m_vggt['delta125_mean']:.3f} "
            f"breath={m_vggt['scale_breath_med']:.3f} {t_vggt:.1f}s",
            flush=True,
        )


def reduce() -> None:
    rows = []
    for p in sorted((OUT / "eps").glob("*/metrics.json")):
        rows.append(json.loads(p.read_text()))
    if not rows:
        raise SystemExit("no metrics yet")

    def med(key_a, key_b):
        xs = [r[key_a][key_b] for r in rows if np.isfinite(r[key_a][key_b])]
        return float(np.median(xs)) if xs else float("nan")

    summary = {
        "n": len(rows),
        "da3_absrel_med": med("da3", "absrel_perframe_med"),
        "vggt_absrel_med": med("vggt", "absrel_perframe_med"),
        "da3_delta125_med": med("da3", "delta125_mean"),
        "vggt_delta125_med": med("vggt", "delta125_mean"),
        "da3_breath_med": med("da3", "scale_breath_med"),
        "vggt_breath_med": med("vggt", "scale_breath_med"),
        "da3_absrel_global_med": med("da3", "absrel_global_scale"),
        "vggt_absrel_global_med": med("vggt", "absrel_global_scale"),
        "rows": rows,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    print(f"\n{'eid':<32} {'task':>6} {'DA3':>8} {'VGGT':>8} {'win'}", flush=True)
    da3_w = vggt_w = 0
    for r in rows:
        a, b = r["da3"]["absrel_perframe_med"], r["vggt"]["absrel_perframe_med"]
        win = "DA3" if a < b else "VGGT"
        if a < b:
            da3_w += 1
        else:
            vggt_w += 1
        print(f"{r['episode_id']:<32} {r['task_id']:>6} {a:8.3f} {b:8.3f} {win}", flush=True)
    print(f"wins  DA3 {da3_w}  VGGT {vggt_w}  (lower AbsRel)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-jobs", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--shard", default=None, help="i/N")
    ap.add_argument("--reduce", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs_path = OUT / "jobs.json"
    if args.make_jobs:
        jobs = make_jobs(args.n_jobs)
        jobs_path.write_text(json.dumps(jobs, indent=2))
        print(f"wrote {jobs_path} n={len(jobs)}")
        for j in jobs:
            print(f"  {j['episode_id']} task={j['task_id']} n={j['n_src']}")
        return 0
    if args.reduce:
        reduce()
        return 0
    if not args.shard:
        raise SystemExit("need --make-jobs / --shard i/N / --reduce")
    i, n = args.shard.split("/")
    jobs = json.loads(jobs_path.read_text())
    run_shard(jobs, int(i), int(n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
