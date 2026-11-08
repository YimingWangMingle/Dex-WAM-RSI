"""Read the public robot state, recorded commands and camera paths in DexJoCo rand_full."""
from itertools import product
import json
from pathlib import Path

import numpy as np

from ..runtime.contracts import DUAL, SINGLE


def discover_episodes(root: Path, tasks) -> dict[str, list[tuple[str, Path]]]:
    """Find all 100 recordings per requested task in stable episode-ID order."""
    root = Path(root).expanduser().resolve(strict=True)
    tasks = sorted(tasks)
    if not tasks:
        raise ValueError("supply at least one DexJoCo task")
    candidates = (root, root / "raw", root / "dexjoco_raw_datasets_rand_full")
    matches = [p for p in candidates if any((p / t).is_dir() for t in tasks)]
    if len(matches) != 1:
        raise ValueError(f"expected one rand_full task directory under {root}, found {matches}")
    task_root = matches[0]
    if task_root.name == "dexjoco_raw_datasets":
        raise ValueError(f"{task_root} is the upstream rand_obj layout; supply the rand_full directory")
    result = {}
    for task in tasks:
        directory = task_root / task
        if not directory.is_dir():
            raise ValueError(f"missing DexJoCo task directory: {directory}")
        episodes = sorted(
            (p.name, p) for p in directory.iterdir()
            if not p.name.startswith(".") and p.is_dir()
            and ((p / "replay.zarr").is_dir() or (p / "videos").is_dir())
        )
        if len(episodes) != 100:
            raise ValueError(f"{directory}: expected 100 training episodes, found {len(episodes)}")
        result[task] = episodes
    return result


def _check_array_chunks(store: Path, name: str) -> None:
    """Reject missing chunks before the Zarr reader substitutes their fill value."""
    directory = store / name
    metadata = json.loads((directory / ".zarray").read_text(encoding="utf-8"))
    shape, chunks = metadata.get("shape"), metadata.get("chunks")
    if (metadata.get("zarr_format") != 2 or not isinstance(shape, list)
            or not isinstance(chunks, list) or not shape or len(chunks) != len(shape)
            or any(type(n) is not int or n < 1 for n in shape + chunks)):
        raise ValueError(f"{directory}: invalid Zarr v2 array dimensions")
    separator = metadata.get("dimension_separator", ".")
    if separator not in (".", "/"):
        raise ValueError(f"{directory}: unsupported Zarr chunk separator {separator!r}")
    chunk_axes = [range((size + chunk - 1) // chunk) for size, chunk in zip(shape, chunks)]
    for index in product(*chunk_axes):
        file = directory / separator.join(map(str, index))
        if not file.is_file():
            raise ValueError(f"missing recording chunk: {file}")


def load_public_episode(path: Path, task: str) -> dict:
    """Read one episode with source timestamps and sign-canonicalized WXYZ quaternions."""
    import zarr

    path = Path(path)
    store = path / "replay.zarr"
    for name in ("data/state", "data/action_rotvec", "data/timestamp", "meta/episode_ends"):
        _check_array_chunks(store, name)
    group = zarr.open_group(str(store), mode="r")
    dual = task.startswith("bimanual_")
    state_dim, action_dim = (46, 44) if dual else (23, 22)
    raw_state = group["data/state"]
    if raw_state.ndim != 3 or raw_state.shape[1] != 1 or raw_state.shape[2] < state_dim:
        raise ValueError(f"{path}: invalid raw state layout {raw_state.shape}")
    state = np.array(raw_state[:, 0, :state_dim], dtype=float, copy=True)
    actions = np.asarray(group["data/action_rotvec"][:], dtype=float)
    timestamps = np.asarray(group["data/timestamp"][:], dtype=float)
    n = len(state)
    if n < 2 or actions.shape != (n, action_dim) or timestamps.shape != (n,):
        raise ValueError(f"{path}: incompatible state, action or timestamp dimensions")
    if not all(np.isfinite(a).all() for a in (state, actions, timestamps)):
        raise ValueError(f"{path}: nonfinite public state, action or timestamp")
    if not np.allclose(np.diff(timestamps), 1 / 30, rtol=.01, atol=1e-4):
        raise ValueError(f"{path}: timestamps must advance at 30 Hz")
    if np.asarray(group["meta/episode_ends"][:]).tolist() != [n]:
        raise ValueError(f"{path}: expected one complete episode of {n} steps")
    for start in ((3, 10) if dual else (3,)):
        quat = state[:, start:start + 4]
        norm = np.linalg.norm(quat, axis=1)
        if not np.allclose(norm, 1, atol=.02):
            raise ValueError(f"{path}: public state quaternion must have unit norm")
        quat /= norm[:, None]
        first = np.argmax(np.abs(quat) > 1e-8, axis=1)
        quat *= np.sign(quat[np.arange(n), first])[:, None]
    views = ({"wrist_left": "wrist_left", "wrist_right": "wrist_right"} if dual
             else {"base": "random_camera", "wrist": "wrist"})
    cameras = {view: path / "videos" / (source + ".mp4") for view, source in views.items()}
    for video in cameras.values():
        if not video.is_file():
            raise ValueError(f"missing source camera video: {video}")
    return dict(state=state, actions=actions, timestamps=timestamps, cameras=cameras,
                action_contract=DUAL if dual else SINGLE, task=task, episode_id=path.name,
                source_split="training", native_success=None)
