#!/usr/bin/env python3
"""Fetch a lightweight micro dataset (<= 1GB) from the H200 machine for local debugging.

Reads a Motus manifest on the H200 (via the jump host used in
H-Server-ssh/ssh-to-h200.sh), picks a small number of train records
(evenly spaced for task diversity), downloads the referenced files with
rsync while preserving the remote directory layout, and writes a local
manifest with rewritten absolute paths.

Example:
    python scripts/fetch_h200_micro_dataset.py --num-records 32 --output-dir micro_data
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_MANIFEST = (
    "/data/jiyuan.wang/Manifests/_0713/holdout_1pct_seed20260713/manifests/"
    "robotwin_aloha_manifest.holdout1pct_seed20260713.motus_manifest.jsonl"
)
DEFAULT_JUMP_HOST = "chen.liao@10.10.94.35"
DEFAULT_SSH_USER_HOST = "fengodchen@localhost"
DEFAULT_SSH_PORT = "10023"

# Remote helper executed on the H200. Reads the manifest, selects train
# records (evenly spaced for task diversity, or via byte-offset probing for
# huge manifests that cannot be scanned), collects referenced files (tar
# members are extracted in place to a temp dir using the recorded offsets),
# applies the byte cap and prints a JSON summary to stdout.
REMOTE_HELPER = r"""
import json
import os
import subprocess
import sys

manifest_path = sys.argv[1]
num_records = int(sys.argv[2])
max_bytes = int(sys.argv[3])
seek_samples = int(sys.argv[4])
extract_dir = sys.argv[5]
hdf5_slim = sys.argv[6] == "1"
transcode_h264 = sys.argv[7] == "1"
ffmpeg_bin = sys.argv[8]

ROBOTMIND_JPEG_BACKENDS = {"robotmind_hdf5_jpeg", "robotmind_tianyi_hdf5_jpeg", "hdf5_jpeg"}

def slim_robotmind_record(record, extract_dir, tag):
    # Build a per-record slim trajectory.hdf5 holding only the 49 referenced
    # video frames (rewritten to indices 0..N) plus the full state/action key
    # arrays (small; absolute action frame_indices stay valid). Returns
    # (rewritten_record, [(path, size)]) or None on failure.
    import h5py
    import numpy as np

    out = json.loads(json.dumps(record))
    video = out.get("video") or {}
    state = out.get("state") or {}
    action = out.get("action") or {}
    src_uri = video.get("uri") or video.get("path")
    frame_indices = [int(i) for i in (video.get("frame_indices") or [])]
    if not src_uri or not frame_indices:
        return None
    video_key = str(video.get("key") or "camera_observations/color_images/camera_top")
    slim_path = os.path.join(extract_dir, "%s_trajectory_slim.hdf5" % tag)
    try:
        with h5py.File(src_uri, "r") as fsrc, h5py.File(slim_path, "w") as fdst:
            src_ds = fsrc[video_key]
            total = int(src_ds.shape[0])
            safe = [max(0, min(total - 1, i)) for i in frame_indices]
            parent = video_key.rsplit("/", 1)[0]
            if parent:
                fdst.require_group(parent)
            base = h5py.check_vlen_dtype(src_ds.dtype)
            if base is not None:
                out_ds = fdst.create_dataset(
                    video_key, (len(safe),), dtype=h5py.vlen_dtype(np.dtype(base))
                )
                for j, i in enumerate(safe):
                    out_ds[j] = np.asarray(src_ds[i], dtype=base)
            else:
                fdst.create_dataset(
                    video_key, data=np.stack([np.asarray(src_ds[i]) for i in safe])
                )
            copied = set()
            for keys in (state.get("keys") or {}, action.get("keys") or {}):
                for hkey in keys.values():
                    if hkey in copied:
                        continue
                    copied.add(hkey)
                    parent = hkey.rsplit("/", 1)[0]
                    grp = fdst.require_group(parent) if parent else fdst
                    grp.create_dataset(hkey.rsplit("/", 1)[-1], data=np.asarray(fsrc[hkey]))
    except (OSError, KeyError):
        if os.path.exists(slim_path):
            os.remove(slim_path)
        return None
    video["frame_indices"] = list(range(len(safe)))
    for item in (video, state, action):
        if "uri" in item:
            item["uri"] = slim_path
        elif "path" in item:
            item["path"] = slim_path
    return out, [(slim_path, os.path.getsize(slim_path))]

def parse_record(line):
    try:
        record = json.loads(line)
    except (TypeError, ValueError):
        return None
    if record.get("schema_version") != "motus_manifest_v1":
        return None
    if record.get("split") != "train":
        return None
    return record

def iter_records_full(path):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = parse_record(line)
            if record is not None:
                yield record

def iter_records_seek(path, k):
    # Probe k evenly spaced byte offsets; take the first complete record
    # after each offset. Avoids scanning a huge manifest end to end.
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        for i in range(k):
            f.seek(size * i // k)
            f.readline()  # skip partial line
            record = parse_record(f.readline())
            if record is not None:
                yield record

if seek_samples > 0:
    candidates = list(iter_records_seek(manifest_path, seek_samples))
    num_train_records = -1  # unknown without a full scan
else:
    candidates = list(iter_records_full(manifest_path))
    num_train_records = len(candidates)

if not candidates:
    print(json.dumps({"error": "no train records found"}))
    sys.exit(1)

if len(candidates) <= num_records:
    ordered = candidates
else:
    # Oversample evenly spaced candidates so records that do not fit the byte
    # cap (or are missing files) can be skipped in favor of later ones.
    probe = min(len(candidates), num_records * 8)
    step = len(candidates) / probe
    ordered = [candidates[int(i * step)] for i in range(probe)]

if extract_dir:
    os.makedirs(extract_dir, exist_ok=True)
extract_cache = {}
missing = []

def prepare_record(record, tag):
    # Returns (rewritten_record, [(path, size), ...]) or None if a referenced
    # file is missing. Tar members are extracted to extract_dir and the item
    # is rewritten to its non-tar storage backend. In hdf5_slim mode RobotMind
    # records get a per-record slim trajectory.hdf5 instead of the full file.
    if hdf5_slim:
        video_backend = ((record.get("video") or {}).get("storage_backend")) or ""
        if video_backend in ROBOTMIND_JPEG_BACKENDS:
            return slim_robotmind_record(record, extract_dir, tag)
    out = json.loads(json.dumps(record))
    files = []
    for key in ("video", "state", "action"):
        item = out.get(key) or {}
        uri = item.get("uri") or item.get("path")
        member = item.get("member")
        if not uri:
            continue
        if member and item.get("tar_offset_data") is not None:
            offset = int(item["tar_offset_data"])
            size = int(item["tar_size"])
            cache_key = (uri, member)
            extracted = extract_cache.get(cache_key)
            if extracted is None:
                extracted = os.path.join(
                    extract_dir, "%03d_%s" % (len(extract_cache), os.path.basename(member))
                )
                try:
                    with open(uri, "rb") as tf:
                        tf.seek(offset)
                        data = tf.read(size)
                except OSError:
                    missing.append("%s::%s" % (uri, member))
                    return None
                if len(data) != size:
                    missing.append("%s::%s (short read)" % (uri, member))
                    return None
                with open(extracted, "wb") as of:
                    of.write(data)
                if transcode_h264 and key == "video":
                    # agibot_beta videos are AV1; the local debug box cannot
                    # decode them, so transcode to H.264 on the H200.
                    h264_path = extracted + ".h264.mp4"
                    try:
                        subprocess.run(
                            [ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
                             "-i", extracted, "-c:v", "libx264", "-preset", "veryfast",
                             "-crf", "18", "-pix_fmt", "yuv420p", "-an", h264_path],
                            check=True, capture_output=True, text=True,
                        )
                    except (OSError, subprocess.CalledProcessError) as exc:
                        missing.append("%s::%s (transcode failed: %s)" % (uri, member, exc))
                        return None
                    os.remove(extracted)
                    extracted = h264_path
                extract_cache[cache_key] = extracted
            files.append((extracted, os.path.getsize(extracted)))
            if "uri" in item:
                item["uri"] = extracted
            else:
                item["path"] = extracted
            backend = item.get("storage_backend") or ""
            if backend == "file_tar":
                item["storage_backend"] = "file"
            elif backend.endswith("_tar"):
                item["storage_backend"] = backend[: -len("_tar")]
            for drop in ("member", "tar_offset_data", "tar_size"):
                item.pop(drop, None)
        else:
            try:
                size = os.path.getsize(uri)
            except OSError:
                missing.append(uri)
                return None
            files.append((uri, size))
    return out, files

kept_records = []
kept_files = []
seen = set()
total_bytes = 0
for record in ordered:
    if len(kept_records) >= num_records:
        break
    prepared = prepare_record(record, "%03d" % len(kept_records))
    if prepared is None:
        continue
    rewritten, files = prepared
    new_bytes = sum(size for path, size in files if path not in seen)
    if total_bytes + new_bytes > max_bytes:
        continue
    total_bytes += new_bytes
    for path, _ in files:
        if path not in seen:
            seen.add(path)
            kept_files.append(path)
    kept_records.append(rewritten)

print(json.dumps({
    "num_train_records": num_train_records,
    "num_selected": len(ordered),
    "num_usable": len(kept_records),
    "total_bytes": total_bytes,
    "files": kept_files,
    "missing": missing,
    "records": kept_records,
    "extract_dir": extract_dir,
}))
"""


def run_ssh(args: argparse.Namespace, remote_command: str, input_text: str = None) -> str:
    cmd = [
        "ssh", "-o", "BatchMode=yes",
        "-J", args.jump_host,
        "-p", args.ssh_port,
        args.ssh_user_host,
        remote_command,
    ]
    result = subprocess.run(
        cmd, input=input_text, capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ssh command failed: {' '.join(cmd)}\n{result.stderr.strip()}")
    return result.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help="Manifest path on the H200")
    parser.add_argument("--num-records", type=int, default=32, help="Number of train records to sample")
    parser.add_argument("--output-dir", default="micro_data", help="Local output directory")
    parser.add_argument("--max-bytes", type=int, default=1024 ** 3, help="Cumulative download size cap")
    parser.add_argument("--jump-host", default=DEFAULT_JUMP_HOST, help="SSH jump host")
    parser.add_argument("--ssh-user-host", default=DEFAULT_SSH_USER_HOST, help="SSH user@host of the H200")
    parser.add_argument("--ssh-port", default=DEFAULT_SSH_PORT, help="SSH port of the H200")
    parser.add_argument("--dataset-name", default="robotwin_aloha", help="Subdirectory name under output-dir")
    parser.add_argument(
        "--seek-samples",
        type=int,
        default=0,
        help="If >0, probe this many evenly spaced byte offsets instead of "
        "scanning the whole manifest (for huge manifests, e.g. agibot_beta)",
    )
    parser.add_argument(
        "--hdf5-slim",
        action="store_true",
        help="For RobotMind hdf5_jpeg records: build per-record slim "
        "trajectory.hdf5 (only the referenced frames + state/action keys) "
        "instead of downloading full episode files",
    )
    parser.add_argument(
        "--remote-python",
        default="python3",
        help="Python interpreter on the H200 (needs h5py when --hdf5-slim is on)",
    )
    parser.add_argument(
        "--transcode-video-h264",
        action="store_true",
        help="Transcode extracted tar-member videos to H.264 on the H200 "
        "(agibot_beta videos are AV1, which the local debug box cannot decode)",
    )
    parser.add_argument(
        "--remote-ffmpeg",
        default="ffmpeg",
        help="ffmpeg binary on the H200 (used with --transcode-video-h264)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    data_dir = output_dir / args.dataset_name
    data_dir.mkdir(parents=True, exist_ok=True)

    remote_extract_dir = f"/tmp/motus_micro_extract_{args.dataset_name}"

    print(f"[fetch] selecting {args.num_records} records from {args.manifest} on H200 ...")
    run_ssh(args, f"rm -rf {remote_extract_dir} && mkdir -p {remote_extract_dir}")
    stdout = run_ssh(
        args,
        f"{args.remote_python} - {args.manifest} {args.num_records} {args.max_bytes} "
        f"{args.seek_samples} {remote_extract_dir} {1 if args.hdf5_slim else 0} "
        f"{1 if args.transcode_video_h264 else 0} {args.remote_ffmpeg}",
        input_text=REMOTE_HELPER,
    )
    summary = json.loads(stdout)
    if "error" in summary:
        raise RuntimeError(summary["error"])

    files = summary["files"]
    records = summary["records"]
    print(
        f"[fetch] train_records={summary['num_train_records']} selected={summary['num_selected']} "
        f"usable={summary['num_usable']} unique_files={len(files)} "
        f"total={summary['total_bytes'] / 1e6:.1f} MB missing={len(summary['missing'])}"
    )
    for path in summary["missing"]:
        print(f"[fetch] WARNING missing remote file: {path}")
    if not records:
        raise RuntimeError("no usable records after applying the byte cap")

    # rsync the referenced files, preserving the remote layout relative to "/".
    files_from = data_dir / ".files_from.txt"
    files_from.write_text("\n".join(path.lstrip("/") for path in files) + "\n")
    rsync_cmd = [
        "rsync", "-avz",
        "-e", f"ssh -o BatchMode=yes -J {args.jump_host} -p {args.ssh_port}",
        "--files-from", str(files_from),
        f"{args.ssh_user_host}:/",
        str(data_dir) + "/",
    ]
    print(f"[fetch] downloading {len(files)} files with rsync ...")
    result = subprocess.run(rsync_cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"rsync failed:\n{result.stderr.strip()}")
    files_from.unlink()
    run_ssh(args, f"rm -rf {remote_extract_dir}")

    # Write the local manifest with rewritten absolute paths.
    local_manifest = output_dir / f"{args.dataset_name}_manifest.micro.jsonl"
    with local_manifest.open("w") as f:
        for record in records:
            for key in ("video", "state", "action"):
                item = record.get(key) or {}
                uri = item.get("uri") or item.get("path")
                if uri:
                    local_path = data_dir / uri.lstrip("/")
                    if "uri" in item:
                        item["uri"] = str(local_path)
                    else:
                        item["path"] = str(local_path)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[fetch] wrote local manifest: {local_manifest} ({len(records)} records)")
    print(f"[fetch] data directory: {data_dir}")
    print("[fetch] done")


if __name__ == "__main__":
    sys.exit(main())
