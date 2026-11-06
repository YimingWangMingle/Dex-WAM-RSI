#!/usr/bin/env python3
"""EgoScale FFS depth aligned to the training RGB grid + pixel-correspondence QC.

Pipeline mirrors the production intent:
  base_0/base_1 -> stereo rectify (undistorted_intrinsics) -> FFS disparity
    -> Z_rect meters -> homography warp back to un-rectified base_0 (+Z-axis fix)
    -> SAME transform as egoscale reader (1/3 nearest resize + top/bottom pad 60)
    -> 480x640 uint16 mm (pad/invalid = 0), left eye only.

Focus: verify that saved depth is pixel-aligned to the training RGB and passes QC.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

DEFAULT_ROOT = Path("/data/EGOSCALE-SECOND-1700H-batch_02/session_f7Wc1b81tl1ez6_00010_task_0001_lerobot_v3")
FFS_ROOT = Path("/code1/zxw_data/pretrain/third_party/Fast-FoundationStereo")
WEIGHTS = FFS_ROOT / "weights/23-36-37/model_best_bp2_serialize.pth"
TRAIN_H, TRAIN_W = 480, 640
NATIVE_H, NATIVE_W = 1080, 1920
DEPTH_CLIP_M = 20.0
NEAR_M = 0.1


def decode_frame(path: Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"failed to decode frame {index}: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def decode_range(path: Path, start: int, count: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(count):
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) != count:
        raise RuntimeError(f"decoded {len(frames)}/{count} from {path} at {start}")
    return frames


def scale_intrinsics(k: np.ndarray, out_wh: tuple[int, int]) -> np.ndarray:
    sx = out_wh[0] / NATIVE_W
    sy = out_wh[1] / NATIVE_H
    s = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], np.float64)
    return s @ k


def rectify_setup(intrinsics_path: Path, size: tuple[int, int] | None = None) -> dict:
    out_w, out_h = size if size is not None else (NATIVE_W, NATIVE_H)
    intr = json.loads(Path(intrinsics_path).read_text())
    k_left = np.asarray(intr["cameras"]["left"]["K"], np.float64)
    k_right = np.asarray(intr["cameras"]["right"]["K"], np.float64)
    if (out_w, out_h) != (NATIVE_W, NATIVE_H):
        k_left = scale_intrinsics(k_left, (out_w, out_h))
        k_right = scale_intrinsics(k_right, (out_w, out_h))
    stereo = intr["stereo"]
    d0 = np.zeros(5, np.float64)
    r = np.asarray(stereo["R_left_to_right"], np.float64)
    t = np.asarray(stereo["T_left_to_right_m"], np.float64).reshape(3, 1)
    r1, r2, p1, p2, _, _, _ = cv2.stereoRectify(
        k_left, d0, k_right, d0, (out_w, out_h), r, t, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )
    map1_l, map2_l = cv2.initUndistortRectifyMap(k_left, d0, r1, p1, (out_w, out_h), cv2.CV_32FC1)
    map1_r, map2_r = cv2.initUndistortRectifyMap(k_right, d0, r2, p2, (out_w, out_h), cv2.CV_32FC1)
    p1k = p1[:3, :3]
    # H maps base_0 (un-rectified left) pixel -> rectified-left pixel: H = P1k @ R1 @ Kleft^-1
    h_b2r = p1k @ r1 @ np.linalg.inv(k_left)
    return {
        "size": (out_w, out_h),
        "k_left": k_left,
        "r1": r1,
        "p1k": p1k,
        "baseline_m": abs(float(p2[0, 3] / p2[0, 0])),
        "fx_rect": float(p1k[0, 0]),
        "maps": (map1_l, map2_l, map1_r, map2_r),
        "h_b2r": h_b2r,
    }


def zaxis_denominator(setup: dict) -> np.ndarray:
    """Per-base_0-pixel factor s.t. Z_base0 = Z_rect / denom (rotation of optical axis)."""
    out_w, out_h = setup.get("size", (NATIVE_W, NATIVE_H))
    kinv = np.linalg.inv(setup["k_left"])
    ys, xs = np.mgrid[0:out_h, 0:out_w].astype(np.float64)
    ones = np.ones_like(xs)
    pix = np.stack([xs, ys, ones], 0).reshape(3, -1)
    rays = kinv @ pix  # (3, N), z-component == 1
    denom = (setup["r1"][2, :] @ rays).reshape(out_h, out_w)
    return denom


def train_transform_depth(depth_native: np.ndarray) -> np.ndarray:
    scale = min(TRAIN_H / NATIVE_H, TRAIN_W / NATIVE_W)
    new_h, new_w = int(NATIVE_H * scale), int(NATIVE_W * scale)
    resized = cv2.resize(depth_native, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    out = np.zeros((TRAIN_H, TRAIN_W), depth_native.dtype)
    y0, x0 = (TRAIN_H - new_h) // 2, (TRAIN_W - new_w) // 2
    out[y0:y0 + new_h, x0:x0 + new_w] = resized
    return out, (y0, new_h, x0, new_w)


def train_transform_rgb(rgb_native: np.ndarray) -> np.ndarray:
    scale = min(TRAIN_H / NATIVE_H, TRAIN_W / NATIVE_W)
    new_h, new_w = int(NATIVE_H * scale), int(NATIVE_W * scale)
    resized = cv2.resize(rgb_native, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    out = np.zeros((TRAIN_H, TRAIN_W, 3), np.uint8)
    y0, x0 = (TRAIN_H - new_h) // 2, (TRAIN_W - new_w) // 2
    out[y0:y0 + new_h, x0:x0 + new_w] = resized
    return out


def colorize(depth_m: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth_m) & (depth_m > 0)
    if not valid.any():
        return np.zeros((*depth_m.shape, 3), np.uint8)
    lo, hi = np.percentile(depth_m[valid], [2, 98])
    value = np.clip((depth_m - lo) / max(hi - lo, 1e-6), 0, 1)
    vis = cv2.applyColorMap((value * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[:, :, ::-1]
    vis[~valid] = 0
    return vis


def edge_alignment(rgb: np.ndarray, depth_mm: np.ndarray, pad_box) -> dict:
    """Precision of depth edges landing on RGB edges within the valid (non-pad) rows."""
    y0, nh, x0, nw = pad_box
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    rgb_edges = cv2.Canny(gray, 60, 160) > 0
    d = depth_mm.astype(np.float32)
    valid = d > 0
    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    thr = np.percentile(grad[valid], 95) if valid.any() else 0.0
    depth_edges = (grad > max(thr, 1.0)) & valid
    region = np.zeros_like(valid)
    region[y0:y0 + nh, x0:x0 + nw] = True
    depth_edges &= region
    rgb_edges &= region
    if depth_edges.sum() < 50:
        return {"depth_edge_px": int(depth_edges.sum()), "precision_2px": float("nan")}
    rgb_edges_dil = cv2.dilate(rgb_edges.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    hit = (depth_edges & rgb_edges_dil).sum()
    return {
        "depth_edge_px": int(depth_edges.sum()),
        "precision_2px": float(hit / max(int(depth_edges.sum()), 1)),
    }


def run_ffs(model, left_rect, right_rect, valid_iters, InputPadder):
    import torch

    image0 = torch.from_numpy(left_rect).cuda().float()[None].permute(0, 3, 1, 2)
    image1 = torch.from_numpy(right_rect).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(image0.shape, divis_by=32, force_square=False)
    image0, image1 = padder.pad(image0, image1)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        disp = model.forward(image0, image1, iters=valid_iters, test_mode=True, optimize_build_volume="pytorch1")
    torch.cuda.synchronize()
    sec = time.time() - t0
    disp = padder.unpad(disp.float()).cpu().numpy().reshape(left_rect.shape[:2]).clip(0, None)
    return disp, sec


def _resize_to_setup(img: np.ndarray, out_wh: tuple[int, int]) -> np.ndarray:
    out_w, out_h = out_wh
    if img.shape[1] == out_w and img.shape[0] == out_h:
        return img
    return cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)


def depth_from_pair(left, right, setup, denom, model, valid_iters, InputPadder, infer_wh=None):
    out_wh = setup.get("size") or infer_wh or (NATIVE_W, NATIVE_H)
    left = _resize_to_setup(left, out_wh)
    right = _resize_to_setup(right, out_wh)
    map1_l, map2_l, map1_r, map2_r = setup["maps"]
    left_rect = cv2.remap(left, map1_l, map2_l, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right, map1_r, map2_r, cv2.INTER_LINEAR)
    disp, sec = run_ffs(model, left_rect, right_rect, valid_iters, InputPadder)
    z_rect = np.divide(
        setup["fx_rect"] * setup["baseline_m"],
        disp,
        out=np.zeros_like(disp),
        where=disp > 0,
    )
    z_base0 = cv2.warpPerspective(
        z_rect,
        setup["h_b2r"],
        out_wh,
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
    )
    z_base0 = z_base0 / denom
    valid = np.isfinite(z_base0) & (z_base0 > NEAR_M) & (z_base0 < DEPTH_CLIP_M)
    z_base0[~valid] = 0.0
    depth_mm_native = np.clip(np.rint(z_base0 * 1000.0), 0, 65535).astype(np.uint16)
    depth_mm_train, pad_box = train_transform_depth(depth_mm_native)
    return depth_mm_train, pad_box, sec


def homography_selfcheck(left: np.ndarray, setup: dict) -> float:
    out_wh = setup.get("size", (NATIVE_W, NATIVE_H))
    left = _resize_to_setup(left, out_wh)
    map1_l, map2_l, _, _ = setup["maps"]
    left_rect_remap = cv2.remap(left, map1_l, map2_l, cv2.INTER_LINEAR)
    left_rect_h = cv2.warpPerspective(left, setup["h_b2r"], out_wh, flags=cv2.INTER_LINEAR)
    mask = (left_rect_remap.sum(2) > 0) & (left_rect_h.sum(2) > 0)
    if not mask.any():
        return float("nan")
    return float(np.abs(left_rect_remap[mask].astype(np.float32) - left_rect_h[mask].astype(np.float32)).mean())


def temporal_qc(results: list[dict]) -> dict:
    med = np.asarray([r["depth_p02_p50_p98_m"][1] for r in results], np.float32)
    secs = np.asarray([r["sec"] for r in results], np.float32)
    warm = secs[1:] if len(secs) > 1 else secs
    delta = np.diff(med) if len(med) > 1 else np.zeros(0, np.float32)
    return {
        "n": len(results),
        "depth_median_m": {
            "mean": float(med.mean()) if len(med) else float("nan"),
            "std": float(med.std()) if len(med) else float("nan"),
            "cv_pct": float(med.std() / med.mean() * 100) if len(med) and med.mean() > 0 else float("nan"),
            "min": float(med.min()) if len(med) else float("nan"),
            "max": float(med.max()) if len(med) else float("nan"),
        },
        "consecutive_median_delta_m": {
            "median_abs": float(np.median(np.abs(delta))) if len(delta) else float("nan"),
            "p95_abs": float(np.quantile(np.abs(delta), 0.95)) if len(delta) else float("nan"),
            "max_abs": float(np.max(np.abs(delta))) if len(delta) else float("nan"),
            "max_relative_pct": float(np.max(np.abs(delta) / np.clip(med[:-1], 1e-6, None)) * 100)
            if len(delta) else float("nan"),
        },
        "warm_sec_per_frame": {
            "median": float(np.median(warm)) if len(warm) else float("nan"),
            "p90": float(np.quantile(warm, 0.9)) if len(warm) else float("nan"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--frames", default="300,600,900")
    parser.add_argument("--out", type=Path, default=Path("/data/xiongwei.zhao/egoscale_aligned_smoke"))
    parser.add_argument("--valid-iters", type=int, default=8)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--infer-wh", default="", help="FFS size WxH after rectify, e.g. 640x360; empty=native")
    args = parser.parse_args()
    if not WEIGHTS.is_file():
        raise FileNotFoundError(WEIGHTS)
    args.out.mkdir(parents=True, exist_ok=True)
    infer_wh = None
    if args.infer_wh.strip():
        infer_w, infer_h = (int(x) for x in args.infer_wh.lower().split("x"))
        infer_wh = (infer_w, infer_h)
    frame_indices = [int(x) for x in args.frames.split(",") if x.strip()]
    left_mp4 = args.root / "videos/base_0_camera/rgb/image/chunk-000/episode_000000.mp4"
    right_mp4 = args.root / "videos/base_1_camera/rgb/image/chunk-000/episode_000000.mp4"
    calib = args.root / "calibration/undistorted_intrinsics.json"

    import torch

    sys.path.insert(0, str(FFS_ROOT))
    from core.utils.utils import InputPadder

    setup = rectify_setup(calib, size=infer_wh)
    denom = zaxis_denominator(setup)
    denom = np.where(np.abs(denom) < 1e-6, 1.0, denom)

    consecutive = len(frame_indices) > 1 and frame_indices == list(range(frame_indices[0], frame_indices[-1] + 1))
    if consecutive:
        lefts = decode_range(left_mp4, frame_indices[0], len(frame_indices))
        rights = decode_range(right_mp4, frame_indices[0], len(frame_indices))
    else:
        lefts = [decode_frame(left_mp4, i) for i in frame_indices]
        rights = [decode_frame(right_mp4, i) for i in frame_indices]

    selfcheck_mad = homography_selfcheck(lefts[0], setup)

    model = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    model.args.valid_iters = args.valid_iters
    model.cuda().eval()

    results = []
    panels = []
    writer = None
    pad_box = None
    for frame, left, right in zip(frame_indices, lefts, rights):
        depth_mm_train, pad_box, sec = depth_from_pair(
            left, right, setup, denom, model, args.valid_iters, InputPadder, infer_wh
        )
        rgb_train = train_transform_rgb(left)
        depth_m_train = depth_mm_train.astype(np.float32) / 1000.0
        valid_train = depth_mm_train > 0
        y0, nh, x0, nw = pad_box
        region = np.zeros((TRAIN_H, TRAIN_W), bool)
        region[y0:y0 + nh, x0:x0 + nw] = True
        pad_zero_ok = bool((depth_mm_train[~region] == 0).all())
        ea = edge_alignment(rgb_train, depth_mm_train, pad_box)
        depth_color = colorize(depth_m_train)
        overlay = rgb_train.copy()
        d = depth_mm_train.astype(np.float32)
        gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)
        de = (grad > max(np.percentile(grad[valid_train], 95) if valid_train.any() else 1.0, 1.0)) & valid_train
        overlay[de] = (255, 0, 0)
        panel = np.concatenate([rgb_train, depth_color, overlay], axis=1)
        if args.summary_only:
            if writer is None:
                writer = cv2.VideoWriter(
                    str(args.out / "aligned_rgb_depth.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    args.video_fps,
                    (panel.shape[1], panel.shape[0]),
                )
            writer.write(cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        else:
            panels.append(panel)
            cv2.imwrite(str(args.out / f"frame_{frame:06d}_panel.png"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        results.append({
            "frame": frame,
            "sec": sec,
            "valid_fraction_in_region": float(valid_train[region].mean()),
            "pad_zero_ok": pad_zero_ok,
            "depth_p02_p50_p98_m": np.percentile(depth_m_train[valid_train], [2, 50, 98]).tolist()
            if valid_train.any() else [float("nan")] * 3,
            "edge_alignment": ea,
        })
    if writer is not None:
        writer.release()
    if panels:
        strip = np.concatenate(panels, axis=0)
        cv2.imwrite(str(args.out / "aligned_panels.png"), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
    summary = {
        "session": args.root.name,
        "train_hw": [TRAIN_H, TRAIN_W],
        "pad_box_yhxw": list(pad_box) if pad_box is not None else None,
        "baseline_m": setup["baseline_m"],
        "fx_rect": setup["fx_rect"],
        "infer_wh": list(infer_wh) if infer_wh else [NATIVE_W, NATIVE_H],
        "homography_selfcheck_mad_intensity": selfcheck_mad,
        "zaxis_denom_min_max": [float(denom.min()), float(denom.max())],
        "temporal": temporal_qc(results),
        "frames": results,
        "panel_layout": "left=train_rgb | mid=depth_color | right=rgb+depth_edges(red)",
        "out": str(args.out),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
