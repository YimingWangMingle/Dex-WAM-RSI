#!/usr/bin/env python3
"""Visualize 3 RGB + 3 FFV1 depth views from Flex-π robotwin_3d."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

CAMS = (
    ("cam_high", "head"),
    ("cam_left_wrist", "left wrist"),
    ("cam_right_wrist", "right wrist"),
)
RGB_KEY = "observation.images.{cam}"
DEPTH_KEY = "observation.depth_ffv1.{cam}"


def _ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def extract_frame(path: Path, frame_idx: int, pix_fmt: str, h: int = 240, w: int = 320) -> np.ndarray:
    bpp = 3 if pix_fmt == "rgb24" else 2
    raw = subprocess.check_output(
        [
            _ffmpeg(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-vf",
            f"select=eq(n\\,{frame_idx})",
            "-vframes",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            pix_fmt,
            "-",
        ]
    )
    expect = h * w * bpp
    if len(raw) != expect:
        raise RuntimeError(f"{path} frame {frame_idx}: got {len(raw)} bytes, want {expect}")
    if pix_fmt == "rgb24":
        return np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
    return np.frombuffer(raw, dtype=np.uint16).reshape(h, w)


def episode_paths(root: Path, ep_i: int) -> dict[str, Path]:
    ch = f"chunk-{ep_i // 1000:03d}"
    ep = f"episode_{ep_i:06d}"
    out = {"chunk": ch, "ep": ep}
    for cam, _ in CAMS:
        out[f"rgb_{cam}"] = root / "videos" / ch / RGB_KEY.format(cam=cam) / f"{ep}.mp4"
        out[f"dep_{cam}"] = root / "videos" / ch / DEPTH_KEY.format(cam=cam) / f"{ep}.mkv"
    return out


def task_of(tm: dict, ep_i: int) -> str:
    for name, t in tm["tasks"].items():
        if t["episode_start"] <= ep_i <= t["episode_end"]:
            return name
    return "unknown"


def colorize_mm(mm: np.ndarray, vmin: float, vmax: float) -> tuple[np.ndarray, dict]:
    valid = mm > 0
    stats = {
        "valid": float(valid.mean()),
        "median_mm": float(np.median(mm[valid])) if valid.any() else 0.0,
        "p05_mm": float(np.percentile(mm[valid], 5)) if valid.any() else 0.0,
        "p95_mm": float(np.percentile(mm[valid], 95)) if valid.any() else 0.0,
    }
    return mm.astype(np.float32), stats


def draw_sheet(
    rgb: dict[str, np.ndarray],
    depth: dict[str, np.ndarray],
    title: str,
    out: Path,
    vmin: float,
    vmax: float,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.4), dpi=140)
    fig.suptitle(title, fontsize=12, y=0.98)
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("#111111")
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    for c, (cam, label) in enumerate(CAMS):
        ax = axes[0, c]
        ax.imshow(rgb[cam])
        ax.set_title(f"RGB  {label}", fontsize=10)
        ax.set_axis_off()
        mm, st = colorize_mm(depth[cam], vmin, vmax)
        masked = np.ma.masked_where(mm <= 0, mm)
        axd = axes[1, c]
        im = axd.imshow(masked, cmap=cmap, norm=norm)
        axd.set_title(
            f"Depth  {label}  med={st['median_mm']:.0f}mm  valid={st['valid']:.0%}",
            fontsize=9,
        )
        axd.set_axis_off()
        div = make_axes_locatable(axd)
        cax = div.append_axes("right", size="4%", pad=0.04)
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("mm", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def draw_overview(
    samples: list[dict],
    out: Path,
    vmin: float,
    vmax: float,
) -> None:
    n = len(samples)
    fig, axes = plt.subplots(n, 6, figsize=(18.5, 2.55 * n), dpi=120)
    if n == 1:
        axes = np.expand_dims(axes, 0)
    fig.suptitle(
        f"robotwin_3d mid-frame  |  RGB then depth (turbo {vmin:.0f}–{vmax:.0f} mm)",
        fontsize=12,
        y=0.995,
    )
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("#111111")
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    col_titles = [f"RGB {lab}" for _, lab in CAMS] + [f"D {lab}" for _, lab in CAMS]
    for r, s in enumerate(samples):
        for c, (cam, _) in enumerate(CAMS):
            ax = axes[r, c]
            ax.imshow(s["rgb"][cam])
            if r == 0:
                ax.set_title(col_titles[c], fontsize=8)
            ax.set_axis_off()
        for c, (cam, _) in enumerate(CAMS):
            ax = axes[r, 3 + c]
            mm = s["depth"][cam].astype(np.float32)
            masked = np.ma.masked_where(mm <= 0, mm)
            ax.imshow(masked, cmap=cmap, norm=norm)
            if r == 0:
                ax.set_title(col_titles[3 + c], fontsize=8)
            ax.set_axis_off()
        axes[r, 0].set_ylabel(s["label"], fontsize=8, rotation=0, ha="right", va="center", labelpad=70)
    fig.tight_layout(rect=(0.06, 0, 1, 0.97))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/data/xiongwei.zhao/robotwin_3d")
    p.add_argument(
        "--out",
        default="/code1/zxw_data/pretrain/ThinkerWAM-Depth-Action-RGB-RoboTwin/tmp/inspect_robotwin_3d",
    )
    p.add_argument("--episodes", type=int, nargs="*", default=[0, 550, 1650, 9999, 13749, 21450, 27499])
    p.add_argument("--frac", type=float, default=0.5, help="frame = frac * (T-1)")
    p.add_argument("--vmin", type=float, default=50.0)
    p.add_argument("--vmax", type=float, default=1400.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tm = json.loads((root / "task_episode_map.json").read_text())
    Lmap = {json.loads(l)["episode_index"]: json.loads(l)["length"] for l in open(root / "meta/episodes.jsonl")}

    samples = []
    for ep_i in args.episodes:
        T = Lmap[ep_i]
        fi = int(round(args.frac * (T - 1)))
        paths = episode_paths(root, ep_i)
        task = task_of(tm, ep_i)
        rgb, depth = {}, {}
        for cam, _ in CAMS:
            rgb[cam] = extract_frame(paths[f"rgb_{cam}"], fi, "rgb24")
            depth[cam] = extract_frame(paths[f"dep_{cam}"], fi, "gray16le")
        title = f"{paths['ep']}  {paths['chunk']}  {task}  frame={fi}/{T}  ({args.frac:.0%} of episode)"
        sheet = out_dir / f"{paths['ep']}_{task}_f{fi:04d}.png"
        draw_sheet(rgb, depth, title, sheet, args.vmin, args.vmax)
        print(f"wrote {sheet}")
        samples.append(
            {
                "label": f"{task}\n{paths['ep']}\nf{fi}/{T}",
                "rgb": rgb,
                "depth": depth,
            }
        )

    overview = out_dir / "sheet_rgb_depth_overview.png"
    draw_overview(samples, overview, args.vmin, args.vmax)
    print(f"wrote {overview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
