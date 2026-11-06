#!/usr/bin/env python3
"""FFS metric depth on Dema head stereo, left-eye output only. Single GPU.

Pairs left/right mp4s under upto_0626_dema_tele_v30. No official calib in the
tree, so pairs are treated as already-rectified and converted with documented
assumed fx/baseline. Results go to <root>/depth/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/data/chen.liao/Downstream_dema/upto_0626_dema_tele_v30")
LEFT_DIR = ROOT / "videos/observation.images.sensor_camera_stereo_left_image_raw/chunk-000"
RIGHT_DIR = ROOT / "videos/observation.images.sensor_camera_stereo_right_image_raw/chunk-000"
OUT = ROOT / "depth"
FFS_ROOT = Path("/code1/zxw_data/pretrain/third_party/Fast-FoundationStereo")
WEIGHTS = FFS_ROOT / "weights/23-36-37/model_best_bp2_serialize.pth"

NATIVE_W, NATIVE_H = 1920, 1536
INFER_W, INFER_H = 640, 512
# No factory calib on disk. These are scale placeholders so Z = fx*B/disp is finite.
# Re-scale later if real K/T arrives. fx ~ 0.8*width at native, baseline 6 cm.
ASSUMED_FX_NATIVE = 1536.0
ASSUMED_BASELINE_M = 0.06
DEPTH_CLIP_M = 8.0
NEAR_M = 0.10


def colorize(depth_m: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth_m) & (depth_m > 0)
    vis = np.zeros((*depth_m.shape, 3), np.uint8)
    if not valid.any():
        return vis
    lo, hi = np.percentile(depth_m[valid], [2, 98])
    value = np.clip((depth_m - lo) / max(hi - lo, 1e-6), 0, 1)
    vis = cv2.applyColorMap((value * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    vis[~valid] = 0
    return vis


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


def run_ffs(model, left_rgb, right_rgb, valid_iters, InputPadder):
    import torch

    image0 = torch.from_numpy(left_rgb).cuda().float()[None].permute(0, 3, 1, 2)
    image1 = torch.from_numpy(right_rgb).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(image0.shape, divis_by=32, force_square=False)
    image0, image1 = padder.pad(image0, image1)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        disp = model.forward(
            image0, image1, iters=valid_iters, test_mode=True, optimize_build_volume="pytorch1"
        )
    disp = padder.unpad(disp.float()).cpu().numpy().reshape(left_rgb.shape[:2]).clip(0, None)
    return disp


def pair_files() -> list[tuple[Path, Path, str]]:
    lefts = {p.name: p for p in LEFT_DIR.glob("file-*.mp4")}
    rights = {p.name: p for p in RIGHT_DIR.glob("file-*.mp4")}
    names = sorted(set(lefts) & set(rights))
    return [(lefts[n], rights[n], n.replace(".mp4", "")) for n in names]


def dump_file(left_p: Path, right_p: Path, stem: str, model, InputPadder, valid_iters: int, fx: float, baseline: float) -> dict:
    dest_npy = OUT / "left" / "chunk-000" / f"{stem}_uint16_mm.npy"
    dest_vis = OUT / "left" / "chunk-000" / f"{stem}_vis.mp4"
    dest_meta = OUT / "left" / "chunk-000" / f"{stem}.json"
    dest_npy.parent.mkdir(parents=True, exist_ok=True)
    if dest_npy.is_file() and dest_vis.is_file() and dest_meta.is_file():
        meta = json.loads(dest_meta.read_text())
        meta["skipped"] = True
        return meta

    cap_l = cv2.VideoCapture(str(left_p))
    cap_r = cv2.VideoCapture(str(right_p))
    n_l = int(cap_l.get(cv2.CAP_PROP_FRAME_COUNT))
    n_r = int(cap_r.get(cv2.CAP_PROP_FRAME_COUNT))
    n = min(n_l, n_r)
    if n <= 0:
        cap_l.release()
        cap_r.release()
        raise RuntimeError(f"empty pair {stem}: left={n_l} right={n_r}")

    writer = cv2.VideoWriter(
        str(dest_vis),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (INFER_W * 2, INFER_H),
    )
    depths = []
    t0 = time.time()
    infer_s = 0.0
    valid_fracs = []
    for i in range(n):
        ok_l, bgr_l = cap_l.read()
        ok_r, bgr_r = cap_r.read()
        if not (ok_l and ok_r):
            n = i
            break
        left = cv2.resize(cv2.cvtColor(bgr_l, cv2.COLOR_BGR2RGB), (INFER_W, INFER_H), interpolation=cv2.INTER_LINEAR)
        right = cv2.resize(cv2.cvtColor(bgr_r, cv2.COLOR_BGR2RGB), (INFER_W, INFER_H), interpolation=cv2.INTER_LINEAR)
        t = time.time()
        disp = run_ffs(model, left, right, valid_iters, InputPadder)
        infer_s += time.time() - t
        depth_m = np.divide(fx * baseline, disp, out=np.zeros_like(disp), where=disp > 0)
        valid = np.isfinite(depth_m) & (depth_m > NEAR_M) & (depth_m < DEPTH_CLIP_M)
        depth_m[~valid] = 0.0
        valid_fracs.append(float(valid.mean()))
        depths.append(np.clip(np.rint(depth_m * 1000.0), 0, 65535).astype(np.uint16))
        panel = np.concatenate([cv2.cvtColor(left, cv2.COLOR_RGB2BGR), colorize(depth_m)], axis=1)
        writer.write(panel)
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            fps = (i + 1) / max(elapsed, 1e-6)
            print(
                f"{stem} {i+1}/{n} {fps:.2f} f/s remaining={(n-i-1)/max(fps,1e-6)/60:.1f} min "
                f"valid={valid_fracs[-1]:.2f}",
                flush=True,
            )
    cap_l.release()
    cap_r.release()
    writer.release()
    stack = np.stack(depths, 0) if depths else np.zeros((0, INFER_H, INFER_W), np.uint16)
    np.save(dest_npy, stack)
    meta = {
        "stem": stem,
        "left": str(left_p),
        "right": str(right_p),
        "frames": int(stack.shape[0]),
        "shape": list(stack.shape),
        "infer_wh": [INFER_W, INFER_H],
        "fx": fx,
        "baseline_m": baseline,
        "eye": "left",
        "sec_total": time.time() - t0,
        "sec_infer": infer_s,
        "valid_frac_mean": float(np.mean(valid_fracs)) if valid_fracs else 0.0,
        "npy": str(dest_npy),
        "vis": str(dest_vis),
        "skipped": False,
    }
    dest_meta.write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ("stem", "frames", "sec_total", "valid_frac_mean")}, indent=2), flush=True)
    return meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-iters", type=int, default=8)
    parser.add_argument("--max-files", type=int, default=0, help="0 = all paired files")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    fx = ASSUMED_FX_NATIVE * (INFER_W / NATIVE_W)
    pairs = pair_files()
    if args.max_files > 0:
        pairs = pairs[: args.max_files]
    manifest = {
        "out": str(OUT),
        "eye": "left",
        "note": "left-eye FFS only; treated as pre-rectified; assumed fx/baseline",
        "assumed_fx_native": ASSUMED_FX_NATIVE,
        "assumed_fx_infer": fx,
        "assumed_baseline_m": ASSUMED_BASELINE_M,
        "infer_wh": [INFER_W, INFER_H],
        "pairs": len(pairs),
        "missing_right_only_left": sorted(
            p.name for p in LEFT_DIR.glob("file-*.mp4") if not (RIGHT_DIR / p.name).is_file()
        ),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)

    model, InputPadder = load_ffs(args.valid_iters)
    print("FFS loaded", flush=True)
    rows = []
    for left_p, right_p, stem in pairs:
        rows.append(dump_file(left_p, right_p, stem, model, InputPadder, args.valid_iters, fx, ASSUMED_BASELINE_M))
        (OUT / "progress.json").write_text(json.dumps(rows, indent=2))
    print(f"done files={len(rows)} out={OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
