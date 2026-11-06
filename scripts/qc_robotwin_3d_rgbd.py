#!/usr/bin/env python3
"""QC Flex-π robotwin_3d RGB + FFV1 depth pairing before writing a loader.

Checks, per episode × camera:
  - six files present (3 RGB mp4 + 3 depth mkv)
  - RGB / depth frame count and (H, W) match
  - RGB not blank / saturated
  - depth uint16-mm: valid ratio, median in tabletop range, not constant

Does not require meta/ or parquet. Incomplete extract is OK: missing files
are counted as incomplete, not as QC fail.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np


def _ffmpeg_exe() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError(
            "ffmpeg not found; install ffmpeg or imageio-ffmpeg"
        ) from exc

CAMS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
RGB_KEY = "observation.images.{cam}"
DEPTH_KEY = "observation.depth_ffv1.{cam}"

# RoboTwin tabletop GT depth, millimetres.
DEPTH_MEDIAN_MM_LO = 80.0
DEPTH_MEDIAN_MM_HI = 4000.0
DEPTH_VALID_MIN = {"cam_high": 0.40, "cam_left_wrist": 0.15, "cam_right_wrist": 0.15}
RGB_MEAN_LO = 8.0
RGB_MEAN_HI = 247.0


def _ffmpeg() -> str:
    return _ffmpeg_exe()


def probe_video(path: Path) -> dict:
    """Parse `ffmpeg -i` banner. imageio-ffmpeg ships ffmpeg only, not ffprobe."""
    proc = subprocess.run(
        [_ffmpeg(), "-hide_banner", "-i", str(path)],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    banner = proc.stderr
    w = h = n = None
    codec = pix_fmt = None
    fps = None
    duration_s = None
    for line in banner.splitlines():
        if "Duration:" in line:
            # Duration: 00:00:02.78,
            part = line.split("Duration:")[1].split(",")[0].strip()
            hh, mm, ss = part.split(":")
            duration_s = float(hh) * 3600 + float(mm) * 60 + float(ss)
        if "Video:" not in line:
            continue
        codec = line.split("Video:")[1].split()[0].strip(" ,")
        if "gray16le" in line:
            pix_fmt = "gray16le"
        elif "yuv420p" in line:
            pix_fmt = "yuv420p"
        if "x" in line:
            for tok in line.replace(",", " ").split():
                if "x" in tok and tok[0].isdigit():
                    a, b = tok.split("x")
                    if a.isdigit() and b.isdigit():
                        w, h = int(a), int(b)
                        break
        if "fps" in line:
            for tok, nxt in zip(line.split(), line.split()[1:]):
                if nxt == "fps":
                    try:
                        fps = float(tok.strip(","))
                    except ValueError:
                        pass
    if w and h and duration_s is not None and fps:
        n = int(round(duration_s * fps))
    if w is None or h is None:
        raise RuntimeError(f"cannot probe {path}\n{banner[-400:]}")
    return {"w": w, "h": h, "n": n, "codec": codec, "pix_fmt": pix_fmt, "fps": fps}


def _decode_raw(path: Path, pix_fmt: str, bpp: int, h: int, w: int) -> np.ndarray:
    raw = subprocess.check_output(
        [
            _ffmpeg(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            pix_fmt,
            "-",
        ]
    )
    bpf = h * w * bpp
    if len(raw) % bpf != 0:
        raise RuntimeError(f"byte length {len(raw)} not divisible by {bpf}: {path}")
    n = len(raw) // bpf
    if pix_fmt == "rgb24":
        return np.frombuffer(raw, dtype=np.uint8).reshape(n, h, w, 3)
    if pix_fmt == "gray16le":
        return np.frombuffer(raw, dtype=np.uint16).reshape(n, h, w)
    raise ValueError(pix_fmt)


def rgb_sample_stats(path: Path, probe: dict) -> dict:
    frames = _decode_raw(path, "rgb24", 3, probe["h"], probe["w"])
    return {
        "n": int(frames.shape[0]),
        "h": int(frames.shape[1]),
        "w": int(frames.shape[2]),
        "mean": float(frames.mean()),
        "std": float(frames.std()),
    }


def depth_decode_full(path: Path, h: int, w: int) -> np.ndarray:
    return _decode_raw(path, "gray16le", 2, h, w)


def depth_stats(mm: np.ndarray) -> dict:
    valid = mm > 0
    ratio = float(valid.mean())
    if valid.any():
        vals = mm[valid].astype(np.float64)
        median = float(np.median(vals))
        p05 = float(np.percentile(vals, 5))
        p95 = float(np.percentile(vals, 95))
        std = float(vals.std())
    else:
        median = p05 = p95 = std = 0.0
    return {
        "n": int(mm.shape[0]),
        "h": int(mm.shape[1]),
        "w": int(mm.shape[2]),
        "valid_ratio": ratio,
        "median_mm": median,
        "p05_mm": p05,
        "p95_mm": p95,
        "std_mm": std,
        "zero_ratio": float((mm == 0).mean()),
    }


def decide_cam(cam: str, rgb: dict, dep: dict, t_match: bool, hw_match: bool) -> list[str]:
    fails = []
    if not t_match:
        fails.append(f"T_mismatch rgb={rgb['n']} depth={dep['n']}")
    if not hw_match:
        fails.append(f"HW_mismatch rgb={rgb['h']}x{rgb['w']} depth={dep['h']}x{dep['w']}")
    if rgb["mean"] < RGB_MEAN_LO or rgb["mean"] > RGB_MEAN_HI:
        fails.append(f"rgb_mean={rgb['mean']:.1f}")
    if rgb["std"] < 2.0:
        fails.append(f"rgb_flat std={rgb['std']:.2f}")
    if dep["valid_ratio"] < DEPTH_VALID_MIN[cam]:
        fails.append(f"depth_valid={dep['valid_ratio']:.3f}")
    if not (DEPTH_MEDIAN_MM_LO <= dep["median_mm"] <= DEPTH_MEDIAN_MM_HI):
        fails.append(f"depth_median_mm={dep['median_mm']:.1f}")
    if dep["std_mm"] < 5.0:
        fails.append(f"depth_flat std={dep['std_mm']:.2f}")
    return fails


def episode_paths(root: Path, chunk: str, ep: str) -> dict[str, Path]:
    out = {}
    for cam in CAMS:
        out[f"rgb_{cam}"] = root / "videos" / chunk / RGB_KEY.format(cam=cam) / f"{ep}.mp4"
        out[f"dep_{cam}"] = root / "videos" / chunk / DEPTH_KEY.format(cam=cam) / f"{ep}.mkv"
    return out


def pairing_status(paths: dict[str, Path]) -> str:
    have = {k: p.is_file() for k, p in paths.items()}
    if all(have.values()):
        return "complete"
    if not any(have.values()):
        return "absent"
    return "incomplete"


def qc_episode(paths: dict[str, Path]) -> dict:
    cams = {}
    fails = []
    for cam in CAMS:
        rgb_p, dep_p = paths[f"rgb_{cam}"], paths[f"dep_{cam}"]
        rgb_probe = probe_video(rgb_p)
        dep_probe = probe_video(dep_p)
        rgb = rgb_sample_stats(rgb_p, rgb_probe)
        mm = depth_decode_full(dep_p, dep_probe["h"], dep_probe["w"])
        dep = depth_stats(mm)
        t_match = int(rgb["n"]) == int(dep["n"])
        hw_match = rgb["h"] == dep["h"] and rgb["w"] == dep["w"]
        cam_fails = decide_cam(cam, rgb, dep, t_match, hw_match)
        cams[cam] = {
            "rgb": rgb,
            "depth": dep,
            "rgb_codec": rgb_probe.get("codec"),
            "depth_codec": dep_probe.get("codec"),
            "depth_pix_fmt": dep_probe.get("pix_fmt"),
            "fails": cam_fails,
        }
        fails.extend(f"{cam}:{f}" for f in cam_fails)
    return {
        "status": "fail" if fails else "pass",
        "fails": fails,
        "cams": cams,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/data/xiongwei.zhao/robotwin_3d")
    p.add_argument("--chunk", default="chunk-000")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=64, help="exclusive episode index")
    p.add_argument("--max-qc", type=int, default=16, help="decode this many complete episodes")
    p.add_argument(
        "--out",
        default="/code1/zxw_data/pretrain/ThinkerWAM-Depth-Action-RGB-RoboTwin/tmp/qc_robotwin_3d",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {"complete": 0, "incomplete": 0, "absent": 0}
    qc_n = {"pass": 0, "fail": 0}
    records = []

    print(f"scan {root}/videos/{args.chunk} episode_{args.start:06d}..{args.end - 1:06d}")
    complete_ids = []
    for i in range(args.start, args.end):
        ep = f"episode_{i:06d}"
        paths = episode_paths(root, args.chunk, ep)
        pair = pairing_status(paths)
        counts[pair] += 1
        rec = {"episode": ep, "chunk": args.chunk, "pair": pair}
        if pair == "complete":
            complete_ids.append(ep)
            missing = []
        else:
            missing = [k for k, p in paths.items() if not p.is_file()]
        rec["missing"] = missing
        records.append(rec)

    print(
        f"pair  complete={counts['complete']}  incomplete={counts['incomplete']}  "
        f"absent={counts['absent']}"
    )

    for ep in complete_ids[: args.max_qc]:
        paths = episode_paths(root, args.chunk, ep)
        try:
            qc = qc_episode(paths)
        except Exception as exc:
            qc = {"status": "fail", "fails": [f"decode:{exc}"], "cams": {}}
        qc_n[qc["status"]] = qc_n.get(qc["status"], 0) + 1
        rec = next(r for r in records if r["episode"] == ep)
        rec.update(qc)
        cam_line = []
        for cam, info in qc.get("cams", {}).items():
            d = info.get("depth", {})
            r = info.get("rgb", {})
            cam_line.append(
                f"{cam}: T={r.get('n')}/{d.get('n')} "
                f"{r.get('h')}x{r.get('w')} valid={d.get('valid_ratio', 0):.2f} "
                f"med={d.get('median_mm', 0):.0f}mm"
            )
        print(f"{qc['status']:4} {ep}  {' | '.join(cam_line)}")
        if qc["fails"]:
            print(f"     {qc['fails']}")

    report = {
        "root": str(root),
        "chunk": args.chunk,
        "range": [args.start, args.end],
        "pair": counts,
        "qc": qc_n,
        "records": records,
    }
    out_json = out_dir / f"{args.chunk}_e{args.start:06d}_{args.end:06d}.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"qc decode  pass={qc_n['pass']} fail={qc_n['fail']}")
    print(f"wrote {out_json}")
    return 0 if qc_n["fail"] == 0 and counts["complete"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
