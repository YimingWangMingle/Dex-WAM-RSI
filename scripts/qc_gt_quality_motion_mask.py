#!/usr/bin/env python3
"""DA3 vs sensor AbsRel inside vs outside the training 3-gate motion mask.

Smoke clip first (on-disk DA3 + one known tar). Optional second clip 730
only if sensor frames arrive within 2 min.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from thinker_wm.models.depth_motion import depth_motion_mask  # noqa: E402

SMOKE = Path("/data/xiongwei.zhao/agibot_da3_depth/smoke")
OUT = Path("/data/xiongwei.zhao/thinkerwam-depth/outputs/gt_quality_motion_mask")
TAR_SMOKE = "/data/agibot_beta/observations/327/648642-685046.tar"
EP_SMOKE = "648642"
EID_SMOKE = "agibot_beta:327:648642"
N = 49
H, W = 480, 640
NEAR, FAR = 0.1, 10.0
SENSOR_CACHE = SMOKE / "sensor_uint16_mm.npy"
CATALOG_PG = (
    "/data/jiyuan.wang/EpisodeCatalogs/_20260805/agibot_beta_effector_profiles_v1/"
    "agibot_beta_parallel_gripper.20260805_full_v1.episode_catalog.jsonl"
)
PYTHON = "/home/fengodchen/miniconda3/envs/motus/bin/python3.10"


def load_sensor_early(tar_path: str, ep: str, indices: list[int]) -> np.ndarray:
    needed = {f"{ep}/depth/head_depth_{i:06d}.png": i for i in indices}
    got: dict[int, np.ndarray] = {}
    t0 = time.time()
    with tarfile.open(tar_path, mode="r|") as tf:
        for m in tf:
            if m.name not in needed:
                continue
            d = np.asarray(Image.open(io.BytesIO(tf.extractfile(m).read())))
            if d.dtype != np.uint16:
                raise RuntimeError(f"{m.name} dtype={d.dtype}")
            got[needed[m.name]] = d
            if len(got) == len(needed):
                break
    print(f"  sensor tar {tar_path} got {len(got)}/{len(needed)} in {time.time()-t0:.1f}s", flush=True)
    missing = [i for i in indices if i not in got]
    if missing:
        raise FileNotFoundError(f"missing sensor frames {missing[:8]}")
    stack = np.stack([got[i] for i in indices], 0)
    if stack.shape[1:] != (H, W):
        out = np.empty((stack.shape[0], H, W), np.uint16)
        for i in range(stack.shape[0]):
            out[i] = cv2.resize(stack[i], (W, H), interpolation=cv2.INTER_NEAREST)
        stack = out
    return stack


def load_sensor_timeout(tar_path: str, ep: str, indices: list[int], timeout_s: float) -> np.ndarray | None:
    """Stream r| but abort if no complete set within timeout_s."""
    needed = {f"{ep}/depth/head_depth_{i:06d}.png": i for i in indices}
    got: dict[int, np.ndarray] = {}
    t0 = time.time()
    last_log = t0
    try:
        with tarfile.open(tar_path, mode="r|") as tf:
            for m in tf:
                now = time.time()
                if now - t0 > timeout_s:
                    print(
                        f"  SKIP {ep}: tar scan {now-t0:.0f}s with {len(got)}/{len(needed)} frames",
                        flush=True,
                    )
                    return None
                if now - last_log > 20:
                    print(f"  ... scanning {ep} {now-t0:.0f}s got {len(got)}/{len(needed)}", flush=True)
                    last_log = now
                if m.name not in needed:
                    continue
                d = np.asarray(Image.open(io.BytesIO(tf.extractfile(m).read())))
                if d.dtype != np.uint16:
                    return None
                got[needed[m.name]] = d
                if len(got) == len(needed):
                    break
    except Exception as exc:
        print(f"  SKIP {ep}: sensor load failed: {exc}", flush=True)
        return None
    if len(got) != len(needed):
        print(f"  SKIP {ep}: incomplete sensor {len(got)}/{len(needed)}", flush=True)
        return None
    stack = np.stack([got[i] for i in indices], 0)
    if stack.shape[1:] != (H, W):
        out = np.empty((stack.shape[0], H, W), np.uint16)
        for i in range(stack.shape[0]):
            out[i] = cv2.resize(stack[i], (W, H), interpolation=cv2.INTER_NEAREST)
        stack = out
    print(f"  sensor {ep} {len(indices)} frames in {time.time()-t0:.1f}s", flush=True)
    return stack


def catalog_video(eid: str, catalog: str) -> dict | None:
    with open(catalog) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("episode_id") == eid:
                st = rec["video"]["storage"]
                return {
                    "tar_path": st["path"],
                    "offset": int(st["offset"]),
                    "size": int(st["size"]),
                }
    return None


def extract_tar_member(path: str, offset: int, size: int, dest: str) -> None:
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    with open(path, "rb") as src, open(dest, "wb") as dst:
        src.seek(offset)
        left = size
        while left > 0:
            chunk = src.read(min(8 * 1024 * 1024, left))
            if not chunk:
                raise IOError(f"short read {path}@{offset}")
            dst.write(chunk)
            left -= len(chunk)


def decode_first_n(mp4: str, n: int) -> np.ndarray:
    import imageio_ffmpeg

    with tempfile.TemporaryDirectory(prefix="gtq_ff_") as td:
        pattern = os.path.join(td, "frame_%06d.png")
        cmd = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-i", mp4, "-vf", f"select=lt(n\\,{n})", "-vsync", "0",
            "-frames:v", str(n), pattern,
        ]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {p.stderr[-500:]}")
        paths = sorted(Path(td).glob("frame_*.png"))
        if len(paths) < n:
            raise RuntimeError(f"got {len(paths)} frames, want {n}")
        frames = []
        for fp in paths[:n]:
            bgr = cv2.imread(str(fp))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rgb.shape[:2] != (H, W):
                rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
            frames.append(rgb)
        return np.stack(frames, 0)


def try_smoke_rgb() -> np.ndarray | None:
    clip = catalog_video(EID_SMOKE, CATALOG_PG)
    if clip is None:
        print("  RGB: episode not in parallel-gripper catalog", flush=True)
        return None
    raw = str(SMOKE / "_tmp_head.mp4")
    try:
        t0 = time.time()
        extract_tar_member(clip["tar_path"], clip["offset"], clip["size"], raw)
        frames = decode_first_n(raw, N)
        print(f"  RGB catalog extract+decode {time.time()-t0:.1f}s shape={frames.shape}", flush=True)
        return frames
    except Exception as exc:
        print(f"  RGB extract failed ({exc}); depth-only mask", flush=True)
        return None
    finally:
        try:
            os.remove(raw)
        except OSError:
            pass


def score_clip(
    *,
    eid: str,
    da3_mm: np.ndarray,
    sensor_mm: np.ndarray,
    rgb_u8: np.ndarray | None,
    mask_note: str,
) -> dict:
    da3 = da3_mm.astype(np.float32) / 1000.0
    sen = sensor_mm.astype(np.float32) / 1000.0
    m_s = np.isfinite(sen) & (sen > NEAR) & (sen < FAR) & (sensor_mm > 0)
    m_d = np.isfinite(da3) & (da3 > 0) & (da3_mm > 0)
    both = m_s & m_d

    depth_t = torch.from_numpy(da3).unsqueeze(0)
    valid_t = torch.from_numpy(m_d).unsqueeze(0)
    rgb_t = None
    if rgb_u8 is not None:
        rgb_t = torch.from_numpy(rgb_u8.astype(np.float32) / 255.0).permute(0, 3, 1, 2).unsqueeze(0)
    mask = depth_motion_mask(depth_t, valid_t, rgb=rgb_t)[0].numpy()

    n_both = int(both.sum())
    if n_both < 1000:
        raise RuntimeError(f"{eid}: too few overlap pixels: {n_both}")
    alpha = float(np.median(da3[both] / np.clip(sen[both], 1e-6, None)))
    da3_al = da3 / alpha
    absrel = np.abs(da3_al - sen) / np.clip(sen, 1e-6, None)
    ratio = np.maximum(da3_al / np.clip(sen, 1e-6, None), sen / np.clip(da3_al, 1e-6, None))

    inm = both & mask
    outm = both & ~mask
    dt = np.abs(da3[1:] - da3[:-1])
    mdt = m_d[1:] & m_d[:-1]
    jm = mask[1:] & mask[:-1] & mdt
    jo = (~mask[1:]) & (~mask[:-1]) & mdt

    rec = {
        "eid": eid,
        "n_frames": int(da3.shape[0]),
        "mask_source": mask_note,
        "mask_frac": float((mask & m_d).sum() / max(int(m_d.sum()), 1)),
        "sensor_valid_all": float(m_s.mean()),
        "sensor_valid_in_mask": float(m_s[mask].mean()) if mask.any() else float("nan"),
        "overlap_n": n_both,
        "n_in": int(inm.sum()),
        "n_out": int(outm.sum()),
        "scale_alpha_da3_over_sensor": alpha,
        "absrel_med_in": float(np.median(absrel[inm])) if inm.any() else float("nan"),
        "absrel_med_out": float(np.median(absrel[outm])) if outm.any() else float("nan"),
        "absrel_mean_in": float(np.mean(absrel[inm])) if inm.any() else float("nan"),
        "absrel_mean_out": float(np.mean(absrel[outm])) if outm.any() else float("nan"),
        "delta_1.25_in": float(np.mean(ratio[inm] < 1.25)) if inm.any() else float("nan"),
        "delta_1.25_out": float(np.mean(ratio[outm] < 1.25)) if outm.any() else float("nan"),
        "da3_temporal_med_in_mm": float(np.median(dt[jm]) * 1000.0) if jm.any() else float("nan"),
        "da3_temporal_med_out_mm": float(np.median(dt[jo]) * 1000.0) if jo.any() else float("nan"),
        "da3_temporal_med_all_mm": float(np.median(dt[mdt]) * 1000.0) if mdt.any() else float("nan"),
    }
    rec["absrel_med_ratio_in_over_out"] = rec["absrel_med_in"] / max(rec["absrel_med_out"], 1e-9)
    return rec


def measure_smoke() -> dict:
    print("=== smoke agibot_beta:327:648642 ===", flush=True)
    da3_mm = np.load(SMOKE / "depth_uint16_mm.npy")
    assert da3_mm.shape == (N, H, W), da3_mm.shape
    if SENSOR_CACHE.is_file():
        sensor_mm = np.load(SENSOR_CACHE)
        print(f"  sensor cache {SENSOR_CACHE} {sensor_mm.shape}", flush=True)
    else:
        sensor_mm = load_sensor_early(TAR_SMOKE, EP_SMOKE, list(range(N)))
        np.save(SENSOR_CACHE, sensor_mm)
        print(f"  cached sensor → {SENSOR_CACHE}", flush=True)
    rgb = try_smoke_rgb()
    if rgb is None:
        note = "depth_only_3gate_minus_rgb (|Δz|>0.10 + blur residual 0.05 ks=41; RGB missing on smoke disk)"
        print("  MASK: depth-only (no RGB gate)", flush=True)
    else:
        note = "training_3gate_rgb_plus_depth (|Δz|>0.10 + blur residual 0.05 ks=41 + RGB luma>0.05)"
        print("  MASK: full 3-gate with RGB from catalog video frames 0..48", flush=True)
    return score_clip(eid=EID_SMOKE, da3_mm=da3_mm, sensor_mm=sensor_mm, rgb_u8=rgb, mask_note=note)


def try_clip_730() -> dict | None:
    """Optional: holdout shard 1 window of 730:877738. Skip if tar scan >2 min."""
    print("=== optional agibot_beta:730:877738 ===", flush=True)
    tar = "/data/agibot_beta/observations/730/869933-878054.tar"
    if not Path(tar).is_file():
        print("  SKIP 730: tar missing", flush=True)
        return None
    try:
        sys.path.insert(0, str(REPO / "third_party/Thinker-WM/train"))
        sys.path.insert(0, str(REPO / "scripts"))
        from thinker_wm.data_prepare import create_diverse_eval_batch
        from thinker_wm.data_prepare.depth_attach import DepthAttacher
        from eval_official_holdout_clips import apply_pool
        from eval_video_ckpt import load_config
        from thinker_wm.models.depth_codec import decode_to_depth_m
        from omegaconf import OmegaConf

        cfg = load_config(str(REPO / "third_party/Thinker-WM/configs/depth_ablation_treatment_v7.yaml"))
        apply_pool(cfg, "official_holdout")
        t0 = time.time()
        batch = create_diverse_eval_batch(
            cfg,
            "agibot_beta_dexterous_hand",
            batch_size=1,
            shard_ordinal=1,
            seed=int(cfg.periodic_eval.seed),
            use_eval_pool=True,
            episode_id="agibot_beta:730:877738",
        )
        print(f"  create_diverse_eval_batch {time.time()-t0:.1f}s", flush=True)
        attacher = DepthAttacher(
            OmegaConf.to_object(cfg.depth.roots),
            z_near=float(cfg.depth.z_near),
            z_far=float(cfg.depth.z_far),
            mode=str(cfg.depth.mode),
            invalid_to=str(getattr(cfg.depth, "invalid_to", "far")),
            prod_keep_only=bool(getattr(cfg.depth, "prod_keep_only", True)),
        )
        batch = dict(attacher.attach(batch))
        if not bool(batch["depth_present"][0]):
            print("  SKIP 730: depth sidecar missing", flush=True)
            return None
        gt01 = torch.cat([batch["depth_first"].unsqueeze(1), batch["depth_frames"]], 1)[0].float()
        rgb = torch.cat([batch["first_frame"].unsqueeze(1), batch["video_frames"]], 1)[0].float()
        da3_m, da3_valid = decode_to_depth_m(
            gt01,
            z_near=float(cfg.depth.z_near),
            z_far=float(cfg.depth.z_far),
            mode=str(cfg.depth.mode),
            in_range="01",
        )
        idx = [int(x) for x in list(batch["frame_indices"][0])]
        print(f"  frame_indices n={len(idx)} lo={min(idx)} hi={max(idx)}", flush=True)
        sen_mm = load_sensor_timeout(tar, "877738", idx, timeout_s=120.0)
        if sen_mm is None:
            return None
        da3_mm = np.clip(np.rint(da3_m.numpy() * 1000.0), 0, 65535).astype(np.uint16)
        da3_mm[~da3_valid.numpy()] = 0
        rgb_u8 = (rgb.clamp(0, 1).permute(0, 2, 3, 1).numpy() * 255.0).astype(np.uint8)
        return score_clip(
            eid="agibot_beta:730:877738",
            da3_mm=da3_mm,
            sensor_mm=sen_mm,
            rgb_u8=rgb_u8,
            mask_note="training_3gate_from_eval_batch (RGB+DA3 codec log 0.15-4.0)",
        )
    except Exception as exc:
        print(f"  SKIP 730: {type(exc).__name__}: {exc}", flush=True)
        return None


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    clips = [measure_smoke()]
    extra = try_clip_730()
    if extra is not None:
        clips.append(extra)

    report = {
        "note": (
            "DA3 vs AgiBot sensor, after per-clip median scale alpha on overlap. "
            "Sensor gate 0.1-10m and uint16>0. Motion mask is training 3-gate "
            "(no min_area, no pose)."
        ),
        "clips": clips,
    }
    out_path = OUT / "gt_quality.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)

    c0 = clips[0]
    print("\n----- interpretation -----", flush=True)
    print(
        f"Smoke {c0['eid']}: mask_frac={c0['mask_frac']:.4f} ({c0['mask_source']}). "
        f"AbsRel med in={c0['absrel_med_in']:.3f} out={c0['absrel_med_out']:.3f} "
        f"({c0['absrel_med_ratio_in_over_out']:.2f}x). "
        f"AbsRel mean in={c0['absrel_mean_in']:.3f} out={c0['absrel_mean_out']:.3f}. "
        f"delta<1.25 in={c0['delta_1.25_in']:.3f} out={c0['delta_1.25_out']:.3f}. "
        f"DA3 |Δz| med in={c0['da3_temporal_med_in_mm']:.1f}mm out={c0['da3_temporal_med_out_mm']:.1f}mm "
        f"(100mm gate). sensor_valid all={c0['sensor_valid_all']:.3f} "
        f"in_mask={c0['sensor_valid_in_mask']:.3f}. "
        f"n_in={c0['n_in']} n_out={c0['n_out']}.",
        flush=True,
    )
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
