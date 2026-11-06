#!/usr/bin/env python3
"""Rectify one EgoScale stereo pair and run Fast-FoundationStereo metric depth."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/data/EGOSCALE-SECOND-1700H-batch_02/session_f7Wc1b81tl1ez6_00010_task_0001_lerobot_v3")
FFS_ROOT = Path("/code1/zxw_data/pretrain/third_party/Fast-FoundationStereo")
WEIGHTS = FFS_ROOT / "weights/23-36-37/model_best_bp2_serialize.pth"


def decode_frame(path: Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"failed to decode frame {index}: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def rectify(left: np.ndarray, right: np.ndarray, calibration: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Rectify delivered undistorted images using their zero-distortion intrinsics."""
    intr = json.loads((ROOT / "calibration/undistorted_intrinsics.json").read_text())
    left_cam, right_cam = intr["cameras"]["left"], intr["cameras"]["right"]
    stereo = intr["stereo"]
    k_left = np.asarray(left_cam["K"], np.float64)
    k_right = np.asarray(right_cam["K"], np.float64)
    d0 = np.zeros(5, np.float64)
    r = np.asarray(stereo["R_left_to_right"], np.float64)
    t = np.asarray(stereo["T_left_to_right_m"], np.float64).reshape(3, 1)
    h, w = left.shape[:2]
    r1, r2, p1, p2, _, _, _ = cv2.stereoRectify(
        k_left, d0, k_right, d0, (w, h), r, t, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )
    map1_l, map2_l = cv2.initUndistortRectifyMap(k_left, d0, r1, p1, (w, h), cv2.CV_32FC1)
    map1_r, map2_r = cv2.initUndistortRectifyMap(k_right, d0, r2, p2, (w, h), cv2.CV_32FC1)
    left_rect = cv2.remap(left, map1_l, map2_l, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right, map1_r, map2_r, cv2.INTER_LINEAR)
    baseline_m = abs(float(p2[0, 3] / p2[0, 0]))
    return left_rect, right_rect, p1[:3, :3], baseline_m


def colorize(depth_m: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth_m) & (depth_m > 0)
    if not valid.any():
        return np.zeros((*depth_m.shape, 3), np.uint8)
    lo, hi = np.percentile(depth_m[valid], [2, 98])
    value = np.clip((depth_m - lo) / max(hi - lo, 1e-6), 0, 1)
    return cv2.applyColorMap((value * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[:, :, ::-1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", type=int, default=600)
    parser.add_argument("--frames", default="", help="comma-separated frame indices; overrides --frame")
    parser.add_argument("--out", type=Path, default=Path("/data/xiongwei.zhao/egoscale_ffs_smoke"))
    parser.add_argument("--valid-iters", type=int, default=8)
    parser.add_argument("--summary-only", action="store_true", help="write only aggregate metrics and video")
    parser.add_argument("--video-fps", type=float, default=10.0)
    args = parser.parse_args()
    if not WEIGHTS.is_file():
        raise FileNotFoundError(WEIGHTS)
    args.out.mkdir(parents=True, exist_ok=True)
    frame_indices = [int(x) for x in args.frames.split(",") if x.strip()] or [args.frame]

    import torch

    sys.path.insert(0, str(FFS_ROOT))
    from core.utils.utils import InputPadder

    model = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    model.args.valid_iters = args.valid_iters
    model.cuda().eval()
    results = []
    panels = []
    writer = None
    for frame in frame_indices:
        left = decode_frame(ROOT / "videos/base_0_camera/rgb/image/chunk-000/episode_000000.mp4", frame)
        right = decode_frame(ROOT / "videos/base_1_camera/rgb/image/chunk-000/episode_000000.mp4", frame)
        left, right, k, baseline_m = rectify(left, right, {})
        frame_out = args.out / f"frame_{frame:06d}"
        if not args.summary_only:
            frame_out.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(frame_out / "left_rect.png"), cv2.cvtColor(left, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(frame_out / "right_rect.png"), cv2.cvtColor(right, cv2.COLOR_RGB2BGR))
        image0 = torch.from_numpy(left).cuda().float()[None].permute(0, 3, 1, 2)
        image1 = torch.from_numpy(right).cuda().float()[None].permute(0, 3, 1, 2)
        padder = InputPadder(image0.shape, divis_by=32, force_square=False)
        image0, image1 = padder.pad(image0, image1)
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            disparity = model.forward(
                image0, image1, iters=args.valid_iters, test_mode=True, optimize_build_volume="pytorch1"
            )
        torch.cuda.synchronize()
        sec = time.time() - start
        disparity = padder.unpad(disparity.float()).cpu().numpy().reshape(left.shape[:2]).clip(0, None)
        depth_m = np.divide(k[0, 0] * baseline_m, disparity, out=np.zeros_like(disparity), where=disparity > 0)
        valid = np.isfinite(depth_m) & (depth_m > 0) & (depth_m < 20)
        depth_m[~valid] = 0
        depth_vis = colorize(depth_m)
        panel = np.concatenate([left, depth_vis], axis=1)
        if args.summary_only:
            if writer is None:
                writer = cv2.VideoWriter(
                    str(args.out / "timeline_rgb_depth.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    args.video_fps,
                    (panel.shape[1], panel.shape[0]),
                )
            writer.write(cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        else:
            np.save(frame_out / "depth_m.npy", depth_m)
            cv2.imwrite(str(frame_out / "depth_vis.png"), cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(frame_out / "comparison.png"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
            panels.append(panel)
        results.append(
            {
                "frame": frame,
                "sec": sec,
                "valid_fraction_0_20m": float(valid.mean()),
                "depth_p02_p50_p98_m": np.percentile(depth_m[valid], [2, 50, 98]).tolist(),
            }
        )
    if writer is not None:
        writer.release()
    if panels:
        strip = np.concatenate(panels, axis=0)
        cv2.imwrite(str(args.out / "timeline_rgb_depth.png"), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
    summary = {
        "frames": results,
        "shape": [1080, 1920],
        "baseline_m": baseline_m,
        "fx_rect": float(k[0, 0]),
        "out": str(args.out),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
