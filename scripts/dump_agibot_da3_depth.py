#!/usr/bin/env python3
"""AgiBot-Beta DA3 metric depth dump (clip mode).

uint16 mm, 480x640. New episodes dump 3 training windows (49 frames at
native stride 6). Already-complete full-timeline dumps are skipped.

  python scripts/dump_agibot_da3_depth.py --make-jobs
  python scripts/dump_agibot_da3_depth.py --append-jobs
  CUDA_VISIBLE_DEVICES=0 python scripts/dump_agibot_da3_depth.py --shard 0/16
  python scripts/dump_agibot_da3_depth.py --status
  python scripts/dump_agibot_da3_depth.py --merge-selections
"""
from __future__ import annotations

import argparse
import fcntl
import io
import json
import logging
import os
import queue
import random
import re
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoke_da3_depth import (  # noqa: E402
    DA3_EXT,
    DA3_LOCAL,
    T4W_ROOT,
    _stub,
    decode_first_n,
    extract_tar_member,
    to_uint16_mm,
)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

OUT = Path("/data/xiongwei.zhao/agibot_da3_depth/v1")
H, W = 480, 640
NEAR, FAR = 0.1, 10.0
QC_N = 13
SEED = 20260823
MIN_LEN = 260
CLIPS_PER_EP = 3
WIN = 48
RATIO = 6
QUOTA = {
    "agibot_beta_parallel_gripper": 6300,
    "agibot_beta_dexterous_hand": 700,
}
# Second-round add-on: ~double the AgiBot clip pool. 9:1 gripper:hand.
APPEND_QUOTA = {
    "agibot_beta_parallel_gripper": 7200,
    "agibot_beta_dexterous_hand": 800,
}
_PROGRESS_SHARD: int | None = None
SOURCES = {
    "agibot_beta_parallel_gripper": {
        "catalog": (
            "/data/jiyuan.wang/EpisodeCatalogs/_20260805/agibot_beta_effector_profiles_v1/"
            "agibot_beta_parallel_gripper.20260805_full_v1.episode_catalog.jsonl"
        ),
        "ranges": (
            "/data/chen.liao/thinkerwm-pretrain/tmp/sampling_ranges/segmented_20260818_v1/"
            "agibot_beta_parallel_gripper.train.ranges.sqlite"
        ),
        "holdout_ranges": (
            "/data/chen.liao/thinkerwm-pretrain/tmp/sampling_ranges/segmented_20260818_v1/"
            "agibot_beta_parallel_gripper.holdout.ranges.sqlite"
        ),
    },
    "agibot_beta_dexterous_hand": {
        "catalog": (
            "/data/jiyuan.wang/EpisodeCatalogs/_20260805/agibot_beta_effector_profiles_v1/"
            "agibot_beta_dexterous_hand.20260805_full_v1.episode_catalog.jsonl"
        ),
        "ranges": (
            "/data/chen.liao/thinkerwm-pretrain/tmp/sampling_ranges/segmented_20260818_v1/"
            "agibot_beta_dexterous_hand.train.ranges.sqlite"
        ),
        "holdout_ranges": (
            "/data/chen.liao/thinkerwm-pretrain/tmp/sampling_ranges/segmented_20260818_v1/"
            "agibot_beta_dexterous_hand.holdout.ranges.sqlite"
        ),
    },
}

# Smoke ranges mix a few holdout ids; never production train, but drop them
# from a "never-seen" dump so the set stays clean.
SMOKE_RANGES = [
    Path("/data/chen.liao/thinkerwm-pretrain/tmp/sampling_ranges/segmented_20260818_v1_smoke/agibot_beta_parallel_gripper.ranges.sqlite"),
    Path("/data/chen.liao/thinkerwm-pretrain/tmp/sampling_ranges/segmented_20260818_v1_smoke/agibot_beta_dexterous_hand.ranges.sqlite"),
]


def train_ids(ranges_path: str) -> set[str]:
    con = sqlite3.connect(f"file:{ranges_path}?mode=ro", uri=True)
    ids = {r[0] for r in con.execute("SELECT episode_id FROM episode_ranges")}
    con.close()
    return ids


def rec_to_job(rec: dict, source: str) -> dict:
    st = rec["video"]["storage"]
    return {
        "episode_id": rec["episode_id"],
        "source": source,
        "task_id": rec["meta"]["task_id"],
        "tar_path": st["path"],
        "member": st.get("member", ""),
        "offset": int(st["offset"]),
        "size": int(st["size"]),
        "n_src": int(rec["video"]["count"]),
        "fps": float(rec["video"].get("fps", 30.0)),
    }


def sample_source(
    source: str, n: int, rng: random.Random, exclude: set[str] | None = None,
) -> list[dict]:
    spec = SOURCES[source]
    keep = train_ids(spec["ranges"])
    exclude = exclude or set()
    by_task: dict[int, list] = {}
    with open(spec["catalog"]) as f:
        for line in f:
            rec = json.loads(line)
            if rec["episode_id"] not in keep:
                continue
            if rec["episode_id"] in exclude:
                continue
            if rec["video"]["count"] < MIN_LEN:
                continue
            by_task.setdefault(rec["meta"]["task_id"], []).append(rec_to_job(rec, source))
    tasks = list(by_task)
    rng.shuffle(tasks)
    picked, used = [], set()
    for tid in tasks:
        rng.shuffle(by_task[tid])
        job = by_task[tid][0]
        picked.append(job)
        used.add(job["episode_id"])
        if len(picked) >= n:
            return picked
    leftovers = [j for tid in tasks for j in by_task[tid] if j["episode_id"] not in used]
    rng.shuffle(leftovers)
    picked.extend(leftovers[: n - len(picked)])
    return picked[:n]


def holdout_jobs(exclude: set[str] | None = None) -> list[dict]:
    """All official holdout episodes that pass MIN_LEN. No train quota sampling."""
    exclude = exclude or set()
    jobs: list[dict] = []
    for source, spec in SOURCES.items():
        keep = train_ids(spec["holdout_ranges"]) - exclude
        with open(spec["catalog"]) as f:
            for line in f:
                rec = json.loads(line)
                if rec["episode_id"] not in keep:
                    continue
                if rec["video"]["count"] < MIN_LEN:
                    continue
                jobs.append(rec_to_job(rec, source))
    jobs.sort(key=lambda j: (j["source"], j["task_id"], j["episode_id"]))
    return jobs


def ep_dir(job: dict) -> Path:
    return OUT / job["source"] / job["episode_id"].replace(":", "_")


def load_ranges_map(ranges_path: str) -> dict[str, tuple[list, int]]:
    con = sqlite3.connect(f"file:{ranges_path}?mode=ro", uri=True)
    m: dict[str, tuple[list, int]] = {}
    for eid, intervals, elen in con.execute(
        "SELECT episode_id, intervals, episode_len FROM episode_ranges"
    ):
        m[eid] = (json.loads(intervals), int(elen))
    con.close()
    return m


def plan_clips(intervals: list, episode_len: int, n: int = CLIPS_PER_EP) -> list[dict]:
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


def clip_qc_indices(needed: list[int]) -> list[int]:
    if not needed:
        return []
    if len(needed) <= QC_N:
        return list(needed)
    return [needed[int(round(i * (len(needed) - 1) / (QC_N - 1)))] for i in range(QC_N)]


def attach_clips(job: dict, ranges_map: dict[str, tuple[list, int]]) -> bool:
    rr = ranges_map.get(job["episode_id"])
    if rr is None:
        return False
    intervals, elen = rr
    clips = plan_clips(intervals, elen)
    if not clips:
        return False
    needed = sorted({i for c in clips for i in c["video_frame_indices"]})
    job["_clips"] = clips
    job["_needed"] = needed
    job["clip_qc_idx"] = clip_qc_indices(needed)
    job["_episode_len"] = elen
    return True


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)


def append_selection(row: dict) -> None:
    # Per-shard files avoid NFS flock races across two pods.
    if _PROGRESS_SHARD is not None:
        _append_jsonl(OUT / f"depth_selections.shard{_PROGRESS_SHARD}.jsonl", row)
        return
    _append_jsonl(OUT / "depth_selections.jsonl", row)


def append_progress(row: dict) -> None:
    if _PROGRESS_SHARD is not None:
        _append_jsonl(OUT / f"progress.shard{_PROGRESS_SHARD}.jsonl", row)
        return
    _append_jsonl(OUT / "progress.jsonl", row)


def _tar_octal(buf: bytes) -> int:
    raw = buf.split(b"\0", 1)[0].strip()
    return int(raw or b"0", 8)


def read_tar_members_from(
    tar_path: str,
    start_data_off: int,
    names: set[str],
    name_prefix: str,
) -> dict[str, bytes]:
    """Walk one episode from its video data offset; do not scan the whole 45GB tar."""
    if not names:
        return {}
    fd = os.open(tar_path, os.O_RDONLY)
    got: dict[str, bytes] = {}
    pos = max(int(start_data_off) - 512, 0)
    pending_long: str | None = None
    zeros = 0
    try:
        while len(got) < len(names):
            hdr = os.pread(fd, 512, pos)
            if len(hdr) < 512 or hdr == b"\x00" * 512:
                zeros += 1
                if zeros >= 2 or len(hdr) < 512:
                    break
                pos += 512
                continue
            zeros = 0
            size = _tar_octal(hdr[124:136])
            data_off = pos + 512
            next_pos = data_off + ((size + 511) // 512) * 512
            typ = chr(hdr[156]) if hdr[156] else "0"
            name = hdr[0:100].split(b"\0", 1)[0].decode("utf-8", "replace")
            ustar = hdr[345:500].split(b"\0", 1)[0].decode("utf-8", "replace")
            if ustar:
                name = f"{ustar}/{name}" if name else ustar
            if pending_long is not None:
                name = pending_long
                pending_long = None
            if typ == "L":
                pending_long = os.pread(fd, size, data_off).split(b"\0", 1)[0].decode("utf-8", "replace")
                pos = next_pos
                continue
            if name_prefix and not name.startswith(name_prefix):
                break
            if name in names:
                got[name] = os.pread(fd, size, data_off)
            pos = next_pos
    finally:
        os.close(fd)
    return got


def _pngs_to_stack(raw_by_idx: dict[int, bytes], indices: list[int], ep: str) -> np.ndarray:
    missing = [i for i in indices if i not in raw_by_idx]
    if missing:
        raise FileNotFoundError(f"{ep} missing sensor {missing[:6]}")
    frames = []
    for i in indices:
        d = np.asarray(Image.open(io.BytesIO(raw_by_idx[i])))
        if d.dtype != np.uint16:
            raise RuntimeError(f"{ep} depth {i} dtype={d.dtype}")
        frames.append(d)
    stack = np.stack(frames, 0)
    if stack.shape[1:] != (H, W):
        stack = np.stack(
            [cv2.resize(stack[t], (W, H), interpolation=cv2.INTER_NEAREST) for t in range(len(indices))],
            0,
        )
    return stack


def _read_tar_member_at(fd: int, header_off: int, expected_name: str) -> bytes | None:
    """Return file bytes if header_off is a real ustar header for expected_name."""
    hdr = os.pread(fd, 512, header_off)
    if len(hdr) < 512:
        return None
    name = hdr[0:100].split(b"\0", 1)[0].decode("utf-8", "replace")
    ustar = hdr[345:500].split(b"\0", 1)[0].decode("utf-8", "replace")
    if ustar:
        name = f"{ustar}/{name}" if name else ustar
    if name != expected_name:
        return None
    try:
        size = _tar_octal(hdr[124:136])
    except ValueError:
        return None
    if size <= 0 or size > 20_000_000:
        return None
    return os.pread(fd, size, header_off + 512)


def read_tar_members_scan(
    tar_path: str,
    names: set[str],
    around: int,
    back: int = 2_000_000_000,
    fwd: int = 32_000_000,
) -> dict[str, bytes]:
    """Find 512-aligned headers by exact name. Depth is often packed before video."""
    if not names:
        return {}
    want = {n.encode(): n for n in names}
    found: dict[str, bytes] = {}
    size = os.path.getsize(tar_path)
    start = max(0, int(around) - back)
    end = min(size, int(around) + fwd)
    fd = os.open(tar_path, os.O_RDONLY)
    chunk = 8 * 1024 * 1024
    try:
        pos = start
        while pos < end and len(found) < len(want):
            buf = os.pread(fd, chunk + 256, pos)
            for raw, name in list(want.items()):
                if name in found:
                    continue
                j = 0
                while True:
                    k = buf.find(raw, j)
                    if k < 0:
                        break
                    off = pos + k
                    if off % 512 == 0:
                        blob = _read_tar_member_at(fd, off, name)
                        if blob is not None:
                            found[name] = blob
                            break
                    j = k + 1
            pos += chunk
    finally:
        os.close(fd)
    return found


def load_sensor_indices(
    tar_path: str, ep: str, indices: list[int], video_offset: int,
) -> np.ndarray:
    needed = {f"{ep}/depth/head_depth_{i:06d}.png": i for i in indices}
    blobs = read_tar_members_from(tar_path, video_offset, set(needed), f"{ep}/")
    missing = set(needed) - set(blobs)
    if missing:
        extra = read_tar_members_scan(tar_path, missing, video_offset)
        if extra:
            print(
                f"  sensor_backfill {ep} n={len(extra)}/{len(missing)} "
                f"(depth packed before video)",
                flush=True,
            )
        blobs.update(extra)
    raw_by_idx = {needed[name]: blob for name, blob in blobs.items()}
    return _pngs_to_stack(raw_by_idx, indices, ep)


def qc_indices(job: dict) -> list[int]:
    n = int(job["n_src"])
    return [int(round(i * (n - 1) / (QC_N - 1))) for i in range(QC_N)]


def load_sensor_group(jobs: list[dict]) -> dict[str, np.ndarray]:
    """Read QC depth PNGs by seeking from each episode's video offset."""
    if not jobs:
        return {}
    tar_path = jobs[0]["tar_path"]
    if any(j["tar_path"] != tar_path for j in jobs):
        raise ValueError("sensor group spans multiple tars")
    out: dict[str, np.ndarray] = {}
    for job in jobs:
        eid = job["episode_id"]
        ep = eid.split(":")[-1]
        indices = job.get("clip_qc_idx") or qc_indices(job)
        try:
            out[eid] = load_sensor_indices(tar_path, ep, indices, int(job["offset"]))
        except FileNotFoundError as exc:
            print(f"  warn sensor {eid} {exc}", flush=True)
    return out


def qc_sensor_and_da3(da3_mm: np.ndarray, sensor_mm: np.ndarray) -> dict:
    da3 = da3_mm.astype(np.float32) / 1000.0
    sen = sensor_mm.astype(np.float32) / 1000.0
    m_s = np.isfinite(sen) & (sen > NEAR) & (sen < FAR) & (sensor_mm > 0)
    m_d = da3_mm > 0
    both = m_s & m_d
    sensor_valid = float(m_s.mean())
    if both.sum() < 500:
        return {
            "sensor_valid": sensor_valid,
            "absrel_med": float("nan"),
            "scale_alpha": float("nan"),
            "breath": float("nan"),
            "prod_keep": False,
            "reason": "overlap_too_small",
        }
    ratio = da3[both] / np.clip(sen[both], 1e-6, None)
    alpha = float(np.median(ratio))
    da3_al = da3 / max(alpha, 1e-6)
    absrel = float(np.median(np.abs(da3_al[both] - sen[both]) / np.clip(sen[both], 1e-6, None)))
    alphas = []
    for t in range(da3.shape[0]):
        b = both[t]
        if b.sum() < 100:
            continue
        alphas.append(float(np.median(da3[t][b] / np.clip(sen[t][b], 1e-6, None))))
    a = np.asarray(alphas, np.float32)
    breath = float(np.median(np.abs(np.log(a / np.median(a))))) if len(a) else float("nan")
    keep = sensor_valid >= 0.40 and absrel <= 0.15 and (not np.isfinite(breath) or breath <= 0.05)
    return {
        "sensor_valid": sensor_valid,
        "absrel_med": absrel,
        "scale_alpha": alpha,
        "breath": breath,
        "prod_keep": bool(keep),
        "reason": "" if keep else "gate_fail",
    }


def write_depth(path: Path, arr: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import zarr
        from numcodecs import Blosc

        if path.exists():
            import shutil
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


def colorize_mm(mm: np.ndarray) -> np.ndarray:
    valid = mm > 0
    if not valid.any():
        return np.zeros((*mm.shape, 3), np.uint8)
    lo, hi = np.percentile(mm[valid], [2, 98])
    x = np.clip((mm.astype(np.float32) - lo) / max(hi - lo, 1.0), 0, 1)
    return cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[:, :, ::-1]


def probe_hw(mp4: str) -> tuple[int, int]:
    import imageio_ffmpeg

    p = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", mp4],
        capture_output=True, text=True,
    )
    m = re.search(r"(\d{2,5})x(\d{2,5})", p.stderr)
    if not m:
        raise RuntimeError(f"no video size in ffmpeg probe: {p.stderr[-300:]}")
    return int(m.group(1)), int(m.group(2))


def _kill_proc(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.kill()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.wait()


def iter_rgb(
    mp4: str,
    n_hint: int,
    stop: threading.Event | None = None,
    proc_holder: dict | None = None,
):
    import imageio_ffmpeg

    w, h = probe_hw(mp4)
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        exe, "-hide_banner", "-loglevel", "error", "-nostdin",
        "-i", mp4, "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc_holder is not None:
        proc_holder["proc"] = proc
    nbytes = w * h * 3
    n = 0
    try:
        while n_hint <= 0 or n < n_hint:
            if stop is not None and stop.is_set():
                break
            buf = proc.stdout.read(nbytes)
            if not buf or len(buf) < nbytes:
                break
            yield np.frombuffer(buf, dtype=np.uint8).copy().reshape(h, w, 3)
            n += 1
    finally:
        if proc.stdout:
            proc.stdout.close()
        if proc.stderr:
            proc.stderr.close()
        _kill_proc(proc)
    if n_hint > 0 and n != n_hint:
        print(f"  warn decode {n}/{n_hint} {w}x{h}", flush=True)


def prefetch_rgb(mp4: str, n_hint: int, maxsize: int = 16):
    q: queue.Queue = queue.Queue(maxsize=maxsize)
    stop = threading.Event()
    holder: dict = {}

    def _run() -> None:
        try:
            for rgb in iter_rgb(mp4, n_hint, stop, holder):
                if stop.is_set():
                    break
                q.put(rgb)
        finally:
            q.put(None)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    try:
        while True:
            item = q.get()
            if item is None:
                break
            yield item
    finally:
        stop.set()
        if holder.get("proc") is not None:
            _kill_proc(holder["proc"])
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break
        t.join(timeout=8)


def load_da3(device):
    _stub()
    sys.path[:0] = [T4W_ROOT, DA3_EXT]
    logging.getLogger("depth_anything_3").setLevel(logging.ERROR)
    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained(DA3_LOCAL, local_files_only=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device)


def infer_one(model, rgb, device):
    """Same DA3 Giant geometry as inference(), without the 8-way preprocess pool."""
    import torch

    img_t, _, ixt, _ = model.input_processor._process_one(
        rgb, None, None, process_res=504, process_res_method="upper_bound_resize",
    )
    imgs = img_t.to(device, non_blocking=True)[None, None].float()
    raw = model.forward(imgs, None, None, [])
    # Same extract as OutputProcessor._extract_depth: (B,N,1,H,W) -> (N,H,W)
    d = raw["depth"].squeeze(0).squeeze(-1).float().cpu().numpy()
    if d.ndim == 3:
        d = d[0]
    is_metric = int(getattr(raw, "is_metric", 0) or 0)
    k = None
    ixt_t = raw.get("intrinsics") if hasattr(raw, "get") else None
    if ixt_t is not None:
        arr = ixt_t.squeeze(0).detach().float().cpu().numpy()
        k = np.asarray(arr, np.float32).reshape(-1, 3, 3)[0].tolist()
    elif ixt is not None:
        k = np.asarray(ixt, np.float32).reshape(3, 3).tolist()
    return d, is_metric, k


def dump_one(
    job: dict,
    model,
    device,
    sensor_mm: np.ndarray | None = None,
    sec_sensor: float | None = None,
) -> dict:
    dest = ep_dir(job)
    dest.mkdir(parents=True, exist_ok=True)
    meta_path = dest / "metadata.json"
    if meta_path.is_file():
        old = json.loads(meta_path.read_text())
        if old.get("complete"):
            return {**old, "episode_id": job["episode_id"], "status": "skip_done"}
    eid = job["episode_id"]
    ep = eid.split(":")[-1]
    n = int(job["n_src"])
    clips = job.get("_clips")
    needed = job.get("_needed")
    if not clips or not needed:
        return {"episode_id": eid, "source": job["source"], "status": "skip_no_clip", "complete": False}
    qc_idx = job.get("clip_qc_idx") or clip_qc_indices(needed)
    t0 = time.time()
    if sensor_mm is None:
        try:
            sensor_mm = load_sensor_indices(job["tar_path"], ep, qc_idx, int(job["offset"]))
        except FileNotFoundError as exc:
            row = {
                "episode_id": eid,
                "source": job["source"],
                "status": "skip_sensor",
                "complete": True,
                "prod_keep": False,
                "qc": {"sensor_valid": 0.0, "reason": str(exc)},
            }
            meta_path.write_text(json.dumps(row, indent=2))
            return row
    sen_m = sensor_mm.astype(np.float32) / 1000.0
    sensor_valid = float(((sen_m > NEAR) & (sen_m < FAR) & (sensor_mm > 0)).mean())
    t_sen = time.time() - t0 if sec_sensor is None else sec_sensor
    if sensor_valid < 0.40:
        row = {
            "episode_id": eid,
            "source": job["source"],
            "status": "skip_sensor",
            "complete": True,
            "prod_keep": False,
            "qc": {"sensor_valid": sensor_valid, "reason": "sensor_valid<0.40"},
            "sec_sensor": t_sen,
        }
        meta_path.write_text(json.dumps(row, indent=2))
        return row

    raw = dest / "_tmp.mp4"
    extract_tar_member(job["tar_path"], job["offset"], job["size"], str(raw))
    t_ext = time.time() - t0 - t_sen
    row_of = {idx: r for r, idx in enumerate(needed)}
    need_set = set(needed)
    depth = np.zeros((len(needed), H, W), np.uint16)
    rgb0 = None
    is_metric = None
    k0 = None
    t_inf = 0.0
    got = 0
    decoded = 0
    import torch

    with torch.no_grad():
        for i, rgb in enumerate(prefetch_rgb(str(raw), n)):
            decoded = i + 1
            if i not in need_set:
                continue
            if rgb0 is None:
                rgb0 = rgb
            t = time.time()
            d, metric, k = infer_one(model, rgb, device)
            torch.cuda.synchronize()
            t_inf += time.time() - t
            if is_metric is None:
                is_metric = metric
                k0 = k
            depth[row_of[i]] = to_uint16_mm(d, H, W)
            got += 1
            if got == len(needed):
                break
    try:
        raw.unlink()
    except OSError:
        pass
    if got != len(needed):
        raise RuntimeError(f"inferred {got}/{len(needed)} needed frames (decoded={decoded}, n_src={n})")
    da3_qc = depth[np.asarray([row_of[i] for i in qc_idx], np.int64)]
    qc = qc_sensor_and_da3(da3_qc, sensor_mm)
    kind = write_depth(dest / "depth.zarr", depth)
    if rgb0 is not None:
        if rgb0.shape[:2] != (H, W):
            rgb0 = cv2.resize(rgb0, (W, H))
        viz = np.concatenate([rgb0, colorize_mm(depth[0]), colorize_mm(sensor_mm[0])], 1)
        cv2.imwrite(str(dest / "viz.png"), cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
    row = {
        "episode_id": eid,
        "source": job["source"],
        "task_id": job["task_id"],
        "status": "ok",
        "complete": True,
        "teacher": "DA3NESTED-GIANT-LARGE-1.1",
        "unit": "uint16_mm",
        "layout": "clips_native_index",
        "shape": list(depth.shape),
        "rgb_hw": [H, W],
        "n_src": n,
        "episode_len": job.get("_episode_len", n),
        "clips_per_episode": CLIPS_PER_EP,
        "win": WIN,
        "stride": RATIO,
        "clip_starts": [c["start"] for c in clips],
        "frame_index": needed,
        "n_got": got,
        "is_metric": is_metric,
        "intrinsics": k0,
        "codec": kind,
        "qc": qc,
        "prod_keep": qc["prod_keep"],
        "sec_sensor": t_sen,
        "sec_extract": t_ext,
        "sec_infer": t_inf,
        "sec_per_frame": t_inf / max(got, 1),
        "sec_total": time.time() - t0 + t_sen,
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
    meta_path = ep_dir(job) / "metadata.json"
    if not meta_path.is_file():
        return False
    try:
        return bool(json.loads(meta_path.read_text()).get("complete"))
    except Exception:
        return False


def grouped_shard_jobs(jobs: list[dict], world: int) -> list[list[list[dict]]]:
    """Greedily assign whole tar groups by source-frame load.

    Assignment uses the full job list, not ``is_complete``. Filtering
    incompletes here would make the 16-way plan depend on who finished
    first: a late-starting pod would get a different shard map and could
    race the other pod on the same episode. Skip completed episodes inside
    ``dump_one`` instead.
    """
    groups: dict[str, list[dict]] = {}
    for job in jobs:
        groups.setdefault(job["tar_path"], []).append(job)
    bins: list[list[list[dict]]] = [[] for _ in range(world)]
    loads = [0] * world
    for group in sorted(groups.values(), key=lambda g: sum(int(j["n_src"]) for j in g), reverse=True):
        target = min(range(world), key=lambda i: loads[i])
        bins[target].append(group)
        loads[target] += sum(int(j["n_src"]) for j in group)
    print(
        "tar_grouped_plan "
        + json.dumps(
            {
                "jobs": sum(len(g) for g in groups.values()),
                "tar_groups": len(groups),
                "shard_jobs": [sum(len(g) for g in b) for b in bins],
                "shard_frames": loads,
            }
        ),
        flush=True,
    )
    return bins


def _ranges_field() -> str:
    """Holdout dumps must plan clips from holdout sqlite, not train."""
    return "holdout_ranges" if "holdout" in Path(OUT).as_posix() else "ranges"


def run_shard(jobs: list[dict], shard: int, world: int) -> None:
    global _PROGRESS_SHARD
    _PROGRESS_SHARD = shard
    ranges_map: dict[str, tuple[list, int]] = {}
    key = _ranges_field()
    for src in {j["source"] for j in jobs}:
        ranges_map.update(load_ranges_map(SOURCES[src][key]))
    mine = grouped_shard_jobs(jobs, world)[shard]
    for group in mine:
        for job in group:
            attach_clips(job, ranges_map)
    import torch

    device = torch.device("cuda:0")
    torch.set_num_threads(1)
    n_jobs = sum(len(group) for group in mine)
    print(
        f"shard {shard}/{world} tar_groups={len(mine)} jobs={n_jobs} "
        f"clips={CLIPS_PER_EP} win={WIN} stride={RATIO} ranges={_ranges_field()} out={OUT}",
        flush=True,
    )
    model = load_da3(device)
    print(f"DA3 loaded vram={torch.cuda.memory_allocated()/1e9:.2f}G", flush=True)
    done = 0
    for group in mine:
        try:
            t_sensor = time.time()
            sensors = load_sensor_group(group)
            sensor_share = (time.time() - t_sensor) / len(group)
            print(
                f"sensor_group n={len(group)} {time.time() - t_sensor:.1f}s "
                f"{group[0]['tar_path']}",
                flush=True,
            )
        except Exception as exc:
            print(
                f"GROUP_SENSOR_FAIL tar={group[0]['tar_path']} "
                f"{type(exc).__name__}: {exc}; falling back per episode",
                flush=True,
            )
            sensors = {}
            sensor_share = None
        for job in group:
            eid = job["episode_id"]
            try:
                row = dump_one(job, model, device, sensors.get(eid), sensor_share)
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
                f"[{done}/{n_jobs}] {eid} {row.get('status')} keep={row.get('prod_keep')} "
                f"T={row.get('n_got')} {row.get('sec_total', 0):.0f}s "
                f"absrel={row.get('qc', {}).get('absrel_med', float('nan'))}",
                flush=True,
            )


def status() -> None:
    rows = []
    for p in OUT.glob("*/*/metadata.json"):
        try:
            rows.append(json.loads(p.read_text()))
        except Exception:
            continue
    n = len(rows)
    ok = [r for r in rows if r.get("status") == "ok"]
    keep = [r for r in ok if r.get("prod_keep")]
    skip_s = [r for r in rows if r.get("status") == "skip_sensor"]
    fail = [r for r in rows if r.get("status") == "fail"]
    done = [r for r in rows if r.get("status") == "skip_done"]
    pf = [r["sec_per_frame"] for r in ok if r.get("sec_per_frame")]
    jobs_n = 0
    jp = OUT / "jobs.json"
    if jp.is_file():
        jobs_n = len(json.loads(jp.read_text()))
    remain = max(jobs_n - n, 0)
    sec_pf = float(np.median(pf)) if pf else 0.221
    # leftover frames unknown; use mean 1799
    eta_h = remain * 1799 * sec_pf / 3600 / 8
    print(
        json.dumps(
            {
                "jobs": jobs_n,
                "unique_done": n,
                "ok": len(ok),
                "prod_keep": len(keep),
                "skip_sensor": len(skip_s),
                "fail": len(fail),
                "already": len(done),
                "keep_rate_ok": (len(keep) / len(ok)) if ok else None,
                "sec_per_frame_med": sec_pf,
                "remain": remain,
                "eta_8gpu_h": eta_h,
                "out": str(OUT),
            },
            indent=2,
        )
    )
    if ok:
        print("last ok:", ok[-1]["episode_id"], "absrel", ok[-1].get("qc", {}).get("absrel_med"))


def verify_fast(n_frames: int = 8) -> int:
    """Old inference() vs infer_one on one already-dumped episode."""
    import torch

    jobs = json.loads((OUT / "jobs.json").read_text())
    job = None
    for j in jobs:
        meta = ep_dir(j) / "metadata.json"
        if meta.is_file() and json.loads(meta.read_text()).get("status") == "ok":
            job = j
            break
    if job is None:
        raise SystemExit("no completed ok episode to verify")
    dest = ep_dir(job)
    raw = dest / "_verify.mp4"
    extract_tar_member(job["tar_path"], job["offset"], job["size"], str(raw))
    frames = decode_first_n(str(raw), n_frames)
    raw.unlink(missing_ok=True)
    device = torch.device("cuda:0")
    model = load_da3(device)
    diffs = []
    t_old = t_new = 0.0
    with torch.no_grad():
        for i, rgb in enumerate(frames):
            t0 = time.time()
            pred = model.inference([rgb], export_feat_layers=[])
            torch.cuda.synchronize()
            dt_old = time.time() - t0
            t_old += dt_old
            d_old = np.asarray(pred.depth, np.float32)
            if d_old.ndim == 3:
                d_old = d_old[0]
            mm_old = to_uint16_mm(d_old, H, W)
            t1 = time.time()
            d_new, metric, _ = infer_one(model, rgb, device)
            torch.cuda.synchronize()
            dt_new = time.time() - t1
            t_new += dt_new
            mm_new = to_uint16_mm(d_new, H, W)
            both = (mm_old > 0) & (mm_new > 0)
            abs_mm = float(np.median(np.abs(mm_old[both].astype(np.int32) - mm_new[both].astype(np.int32))))
            diffs.append(abs_mm)
            print(
                f"  frame {i} old={dt_old:.3f}s new={dt_new:.3f}s "
                f"diff_med_mm={abs_mm:.2f} is_metric={metric}",
                flush=True,
            )
    print(
        json.dumps(
            {
                "episode_id": job["episode_id"],
                "n": n_frames,
                "diff_med_mm": float(np.median(diffs)),
                "diff_max_med_mm": float(max(diffs)),
                "sec_old": t_old,
                "sec_new": t_new,
                "sec_old_per": t_old / n_frames,
                "sec_new_per": t_new / n_frames,
            },
            indent=2,
        )
    )
    if float(np.median(diffs)) > 2.0:
        raise SystemExit("fast path depth drifted >2mm vs official inference")
    return 0


def verify_grouped_sensor() -> int:
    """Prove grouped tar reading returns byte-identical sensor arrays."""
    jobs = json.loads((OUT / "jobs.json").read_text())
    by_tar: dict[str, list[dict]] = {}
    for job in jobs:
        by_tar.setdefault(job["tar_path"], []).append(job)
    candidates = [group for group in by_tar.values() if len(group) >= 2]
    if not candidates:
        raise SystemExit("no tar contains multiple jobs")
    # Prefer a small group: verification does not need a large union scan.
    group = min(candidates, key=len)[:2]
    old = {
        job["episode_id"]: load_sensor_indices(
            job["tar_path"], job["episode_id"].split(":")[-1], qc_indices(job), int(job["offset"])
        )
        for job in group
    }
    grouped = load_sensor_group(group)
    report = []
    for job in group:
        eid = job["episode_id"]
        a, b = old[eid], grouped[eid]
        report.append(
            {
                "episode_id": eid,
                "shape": list(a.shape),
                "equal": bool(np.array_equal(a, b)),
                "max_abs_mm": int(np.abs(a.astype(np.int32) - b.astype(np.int32)).max()),
            }
        )
    print(json.dumps({"tar": group[0]["tar_path"], "episodes": report}, indent=2))
    if not all(item["equal"] for item in report):
        raise SystemExit("grouped sensor differs from per-episode reader")
    return 0


def used_episode_ids() -> set[str]:
    ids: set[str] = set()
    for name in ("jobs.json", "jobs_round1.json"):
        p = OUT / name
        if not p.is_file():
            continue
        try:
            for job in json.loads(p.read_text()):
                eid = job.get("episode_id")
                if eid:
                    ids.add(eid)
        except Exception:
            continue
    for p in (OUT / "depth_selections.jsonl",):
        if not p.is_file():
            continue
        for line in p.open():
            if not line.strip():
                continue
            try:
                eid = json.loads(line).get("episode_id")
            except Exception:
                continue
            if eid:
                ids.add(eid)
    return ids


def append_new_jobs() -> int:
    jobs_path = OUT / "jobs.json"
    if jobs_path.is_file() and not (OUT / "jobs_round1.json").is_file():
        jobs_path.replace(OUT / "jobs_round1.json")
        print(f"backed up {jobs_path} -> {OUT / 'jobs_round1.json'}", flush=True)
    used = used_episode_ids()
    print(f"exclude already-used episode_ids={len(used)}", flush=True)
    rng = random.Random(SEED + 1)
    jobs: list[dict] = []
    for src, n in APPEND_QUOTA.items():
        part = sample_source(src, n, rng, exclude=used)
        print(f"{src}: appended {len(part)} (asked {n})", flush=True)
        jobs.extend(part)
        used.update(j["episode_id"] for j in part)
    rng.shuffle(jobs)
    jobs_path.write_text(json.dumps(jobs))
    (OUT / "jobs_round2.json").write_text(json.dumps(jobs))
    print(f"wrote {jobs_path} n={len(jobs)} (round2 only; old jobs in jobs_round1.json)", flush=True)
    return 0


def merge_selections() -> int:
    """Rebuild canonical depth_selections.jsonl from shard files + existing rows."""
    dest = OUT / "depth_selections.jsonl"
    seen: dict[str, dict] = {}
    paths = [dest, *sorted(OUT.glob("depth_selections.shard*.jsonl"))]
    for path in paths:
        if not path.is_file():
            continue
        for line in path.open():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            eid = row.get("episode_id")
            if not eid:
                continue
            seen[eid] = row
    tmp = dest.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f:
        for row in seen.values():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(dest)
    print(f"merged selections n={len(seen)} -> {dest}", flush=True)
    return 0


def main() -> int:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-jobs", action="store_true")
    ap.add_argument(
        "--append-jobs",
        action="store_true",
        help="Sample a second-round quota excluding already-dumped episode ids. "
        "Backs up jobs.json to jobs_round1.json and writes only the new jobs.",
    )
    ap.add_argument(
        "--merge-selections",
        action="store_true",
        help="Merge depth_selections.shard*.jsonl into depth_selections.jsonl.",
    )
    ap.add_argument(
        "--make-holdout-jobs",
        action="store_true",
        help="Write ALL official holdout jobs (minus smoke-range ids) to --out. "
        "Does not touch the train dump at v1.",
    )
    ap.add_argument("--shard", default=None)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--verify-fast", action="store_true")
    ap.add_argument("--verify-grouped-sensor", action="store_true")
    ap.add_argument(
        "--out",
        type=Path,
        default=OUT,
        help="Dump root. Default is the train sidecar v1; holdout must use a NEW dir.",
    )
    args = ap.parse_args()
    OUT = Path(args.out)
    if args.make_holdout_jobs and OUT.resolve() == Path("/data/xiongwei.zhao/agibot_da3_depth/v1").resolve():
        raise SystemExit("refuse: --make-holdout-jobs would write into train dump v1; pass --out <new-dir>")
    OUT.mkdir(parents=True, exist_ok=True)
    jobs_path = OUT / "jobs.json"
    if args.make_holdout_jobs:
        smoke = set()
        for p in SMOKE_RANGES:
            if p.is_file():
                smoke |= train_ids(str(p))
        hold = set()
        for spec in SOURCES.values():
            hold |= train_ids(spec["holdout_ranges"])
        exclude = smoke & hold
        jobs = holdout_jobs(exclude=exclude)
        jobs_path.write_text(json.dumps(jobs))
        (OUT / "SPLIT.json").write_text(
            json.dumps(
                {
                    "kind": "official_holdout_never_in_stage2_train",
                    "n_jobs": len(jobs),
                    "excluded_smoke_holdout_ids": sorted(exclude),
                    "do_not_overwrite": [
                        "/data/xiongwei.zhao/agibot_da3_depth/v1",
                        "/data/xiongwei.zhao/agibot_da3_depth_fulltimeline/v1",
                        "/data/xiongwei.zhao/egoscale_ffs_depth/v1",
                    ],
                    "ranges_field": "holdout_ranges",
                },
                indent=2,
            )
            + "\n"
        )
        print(f"wrote {jobs_path} n={len(jobs)} excluded_smoke={len(exclude)} out={OUT}")
        return 0
    if args.make_jobs:
        rng = random.Random(SEED)
        jobs = []
        for src, n in QUOTA.items():
            part = sample_source(src, n, rng)
            print(f"{src}: {len(part)}", flush=True)
            jobs.extend(part)
        rng.shuffle(jobs)
        jobs_path.write_text(json.dumps(jobs))
        print(f"wrote {jobs_path} n={len(jobs)}")
        return 0
    if args.append_jobs:
        return append_new_jobs()
    if args.merge_selections:
        return merge_selections()
    if args.status:
        status()
        return 0
    if args.verify_fast:
        return verify_fast()
    if args.verify_grouped_sensor:
        return verify_grouped_sensor()
    if not args.shard:
        raise SystemExit("need --make-jobs / --shard i/N / --status")
    i, n = args.shard.split("/")
    jobs = json.loads(jobs_path.read_text())
    run_shard(jobs, int(i), int(n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
