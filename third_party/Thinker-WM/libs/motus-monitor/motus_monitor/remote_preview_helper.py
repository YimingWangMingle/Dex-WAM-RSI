"""Motus 训练集预览远端 helper（集群侧长驻进程 + 本机一次性 CLI）。

本文件是**自包含**的远端预览服务：由本机 monitor server 经 SSH 传输到集群
临时目录（``/tmp/motus_preview_helper_<hash>.py``）并以 ``--serve`` 启动，
通过 stdin/stdout 帧协议为本机 ``/api/dataset/*`` 端点提供按需解码能力
（数据在集群存储上，解码必须发生在集群侧）。也可以在本机直接运行（开发/
冒烟），或用一次性 CLI 把预览产物 dump 到目录。

协议（``--serve``，行帧 + 二进制）：

- 启动后立即输出一行握手 JSON：``{"hello": "motus_preview_v1", "pid": N}``；
  重的 import（torch/motus_data）与 catalog 发现推迟到第一个请求。
- 之后每行一个 JSON 请求 ``{"op": ..., ...}``；响应为一行 JSON 头
  ``{"ok": true, "json_bytes": J, "blob_bytes": B}`` 后紧跟 J 字节 JSON
  载荷与 B 字节二进制 blob；错误为 ``{"ok": false, "error": ...}``。
- **stdin EOF 立即退出**——本机 server 异常死亡时 ssh 通道断开即触发，
  这是 helper 不变成孤儿进程的核心保证；另有 ``--idle-timeout`` 兜底。

ops：list_datasets / list_episodes（支持 prefix 组内分页）/ episode_tree
（机器人→任务 两级分组树）/ episode_info / video_frames /
video_mp4 / action_state / selection_preview。selection 即
motus_sample_selection_v1（episode_id + 四组索引），训练切片只需给
``video_frame_indices`` + ``action_video_fps_ratio``，state/action 索引由
helper 重建：consecutive 布局按 (start*ratio, (start+N)*ratio]；stride
布局（视频跳帧、ratio>1）按 state=[start]、action=[start+1..start+N*ratio]
并对 action_count-1 clamp（复现末帧填充）。text_indices 缺省为全集。
"""

from __future__ import annotations

import argparse
import io
import json
import os
import select
import sys
import tempfile
import time
from pathlib import Path

PROTOCOL = "motus_preview_v1"

MAX_FRAMES_PER_REQUEST = 128
MAX_CLIP_FRAMES = 240
MAX_SELECTION_FRAMES = 64
MAX_TEXT_ITEMS = 200
JPEG_QUALITY = 85
DEFAULT_MAX_POINTS = 4096


# ---------------------------------------------------------------------------
# JSON / 编码小工具
# ---------------------------------------------------------------------------
def _dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def _err(message: str) -> dict:
    return {"ok": False, "error": str(message)}


def _int_list(values, *, name: str, maximum: int) -> list:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be a list")
    if len(values) > maximum:
        raise ValueError(f"{name} length {len(values)} exceeds limit {maximum}")
    return [int(v) for v in values]


def _encode_jpegs(frames, max_dim: int | None) -> tuple[list, bytes, list]:
    """torch uint8 [n,H,W,3] -> (sizes, concat_jpeg_bytes, [(w,h), ...])."""
    from PIL import Image

    sizes, chunks, shapes = [], [], []
    for i in range(frames.shape[0]):
        img = Image.fromarray(frames[i].numpy())
        if max_dim and max(img.size) > int(max_dim):
            img.thumbnail((int(max_dim), int(max_dim)), Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=JPEG_QUALITY)
        data = buf.getvalue()
        sizes.append(len(data))
        chunks.append(data)
        shapes.append(list(img.size))  # (w, h)
    return sizes, b"".join(chunks), shapes


def _encode_mp4(frames_iterable, fps: int, max_dim: int | None, progress=None, total: int | None = None) -> bytes:
    """把 uint8 HWC 帧序列编成 h264 MP4，返回文件字节。

    progress(frac, stage) 非空时每帧回报（解码在 frames_iterable 内部发生，
    这里的逐帧进度真实反映 ffmpeg 解码进度）。
    """
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    try:
        writer = None
        try:
            writer = imageio.get_writer(
                tmp.name,
                fps=int(fps),
                codec="libx264",
                macro_block_size=1,
                output_params=["-crf", "23", "-preset", "veryfast", "-pix_fmt", "yuv420p"],
            )
        except Exception:
            if writer is not None:
                writer.close()
            writer = imageio.get_writer(tmp.name, fps=int(fps), macro_block_size=1)
        for i, frame in enumerate(frames_iterable):
            arr = frame.numpy() if hasattr(frame, "numpy") else np.asarray(frame)
            if max_dim and max(arr.shape[1], arr.shape[0]) > int(max_dim):
                img = Image.fromarray(arr)
                img.thumbnail((int(max_dim), int(max_dim)), Image.BILINEAR)
                arr = np.asarray(img)
            writer.append_data(arr)
            if progress and total:
                progress(min(0.99, (i + 1) / total), "decode_video")
        writer.close()
        return Path(tmp.name).read_bytes()
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def _f4_bytes(tensor) -> bytes:
    import numpy as np

    return np.ascontiguousarray(tensor.numpy(), dtype="<f4").tobytes()


def _text_items_payload(items) -> list:
    out = []
    for item in items:
        out.append(
            {
                "index": int(item.index),
                "kind": str(item.kind),
                "text": str(item.text),
                "frame_range": (
                    [int(item.frame_range[0]), int(item.frame_range[1])]
                    if item.frame_range
                    else None
                ),
            }
        )
    return out


# ---------------------------------------------------------------------------
# 预览上下文：catalog 发现 + 仓库缓存（全部懒加载，首个请求时才 import 重库）
# ---------------------------------------------------------------------------
class PreviewContext:
    def __init__(
        self,
        repo_root: str,
        catalog_root: str | None,
        explicit: list,
        motus_data_dir: str | None = None,
    ):
        if repo_root:
            root = Path(repo_root).resolve()
            for path in (root, root / "libs" / "motus-data-factory"):
                if path.is_dir() and str(path) not in sys.path:
                    sys.path.insert(0, str(path))
        # 本机上传到集群的 motus_data 副本优先于仓库检出：版本与本地严格一致，
        # 不受集群 submodule 同步状态影响（最后 insert 的排在最前，保证胜出）。
        if motus_data_dir:
            p = Path(motus_data_dir).resolve()
            if p.is_dir() and str(p) not in sys.path:
                sys.path.insert(0, str(p))
        # 本机无 ffmpeg 且未配置 MOTUS_FFMPEG_BIN 时回退 imageio_ffmpeg 自带
        # 二进制（与 data_prepare/_bootstrap.py 同一约定）。
        if not os.environ.get("MOTUS_FFMPEG_BIN"):
            import shutil

            if shutil.which("ffmpeg") is None and not os.path.exists(
                "/code1/7TB/v2/bin/ffmpeg-static/bin/ffmpeg"
            ):
                try:
                    import imageio_ffmpeg

                    os.environ["MOTUS_FFMPEG_BIN"] = imageio_ffmpeg.get_ffmpeg_exe()
                except ImportError:
                    pass
        from motus_data.catalog.sqlite_index import default_catalog_index_path
        from motus_data.repository import EpisodeRepository

        self._EpisodeRepository = EpisodeRepository
        self._default_index_path = default_catalog_index_path
        self.catalog_root = catalog_root
        self.catalogs = self._discover(explicit)
        self._repos: dict = {}
        self._trees: dict = {}  # dataset -> op_episode_tree payload（catalog 不可变）

    # -- catalog 发现 -------------------------------------------------------
    def _discover(self, explicit: list) -> dict:
        """dataset 名 -> catalog jsonl 路径。

        顺序：motus_data releases（规范短名）→ catalog_root 递归扫描（补充
        未注册 catalog，如本机 micro）→ 显式 name:path 覆盖。
        """
        found: dict = {}
        notes: list = []
        try:
            from motus_data.releases import (
                episode_dataset_releases,
                get_episode_dataset_release,
            )

            datasets = sorted({rid.split(":")[0] for rid in episode_dataset_releases()})
            for dataset in datasets:
                try:
                    release = get_episode_dataset_release(dataset)
                    catalog_path, _ = release.resolve_paths(self.catalog_root)
                except Exception as exc:
                    notes.append(f"{dataset}: resolve failed: {exc}")
                    continue
                if catalog_path.is_file():
                    found[dataset] = catalog_path
                else:
                    notes.append(f"{dataset}: missing {catalog_path}")
        except Exception as exc:
            notes.append(f"releases unavailable: {exc}")

        if self.catalog_root:
            root = Path(self.catalog_root)
            if root.is_dir():
                taken = {p.resolve() for p in found.values()}
                for path in sorted(root.rglob("*.episode_catalog.jsonl")):
                    if path.resolve() in taken:
                        continue
                    stem = path.name[: -len(".episode_catalog.jsonl")]
                    short = stem.split(".")[0]
                    if stem not in found:
                        found[stem] = path
                    if short and short not in found:
                        found[short] = path

        for pair in explicit or []:
            name, sep, path = str(pair).partition(":")
            if not sep or not name:
                raise ValueError(f"--catalog expects name:path, got {pair!r}")
            found[name] = Path(path)

        if not found:
            raise RuntimeError(
                "no episode catalogs found "
                f"(catalog_root={self.catalog_root!r}, explicit={explicit!r}); "
                f"notes: {'; '.join(notes[:6])}"
            )
        return found

    # -- 仓库 ---------------------------------------------------------------
    def repo(self, dataset: str):
        dataset = str(dataset)
        if dataset not in self.catalogs:
            raise KeyError(
                f"unknown dataset {dataset!r}; available={sorted(self.catalogs)}"
            )
        repo = self._repos.get(dataset)
        if repo is None:
            catalog = self.catalogs[dataset]
            index = self._default_index_path(catalog)
            repo = self._EpisodeRepository(
                catalog,
                index_path=index if index.is_file() else None,
                verify_sha256=False,
            )
            self._repos[dataset] = repo
        return repo

    def close(self) -> None:
        for repo in self._repos.values():
            try:
                repo.close()
            except Exception:
                pass
        self._repos.clear()


# ---------------------------------------------------------------------------
# op 实现：每个 handler 返回 (payload_dict, blob_bytes)
# ---------------------------------------------------------------------------
def op_list_datasets(ctx: PreviewContext, _req: dict, progress=None) -> tuple[dict, bytes]:
    datasets = []
    for name in sorted(ctx.catalogs):
        catalog = ctx.catalogs[name]
        entry = {"name": name, "catalog_path": str(catalog)}
        meta_path = catalog.with_suffix(".meta.json")
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            entry["episodes"] = int(meta.get("record_count", meta.get("episodes", 0)))
            entry["source"] = str(meta.get("source", meta.get("source_dataset", "")))
            entry["catalog_id"] = str(meta.get("catalog_id", ""))
        except Exception:
            entry["episodes"] = None
        datasets.append(entry)
    return {"datasets": datasets}, b""


def _episode_page_sqlite(
    index_path: Path,
    offset: int,
    limit: int,
    q: str | None,
    prefix: str | None = None,
):
    import sqlite3

    clauses: list = []
    args: list = []
    if prefix:
        # 前缀范围等价于 startswith，但走 episode_id 的 UNIQUE 索引
        upper = prefix[:-1] + chr(ord(prefix[-1]) + 1)
        clauses.append("(episode_id >= ? AND episode_id < ?)")
        args += [prefix, upper]
    if q:
        clauses.append("(episode_id LIKE ?)")
        args.append(f"%{q}%")
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    uri = f"{index_path.resolve().as_uri()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    try:
        total = connection.execute(
            f"SELECT COUNT(*) FROM episodes{where}", args
        ).fetchone()[0]
        rows = connection.execute(
            f"SELECT episode_id FROM episodes{where} "
            "ORDER BY ordinal LIMIT ? OFFSET ?",
            (*args, limit, offset),
        ).fetchall()
        return int(total), [str(row[0]) for row in rows]
    finally:
        connection.close()


def op_list_episodes(ctx: PreviewContext, req: dict, progress=None) -> tuple[dict, bytes]:
    dataset = str(req["dataset"])
    offset = max(0, int(req.get("offset", 0)))
    limit = max(1, min(500, int(req.get("limit", 50))))
    q = req.get("q") or None
    prefix = req.get("prefix") or None
    repo = ctx.repo(dataset)
    if repo.index_path is not None:
        total, ids = _episode_page_sqlite(repo.index_path, offset, limit, q, prefix)
    else:
        from itertools import islice

        all_ids = repo.iter_episode_ids()
        if prefix:
            all_ids = (eid for eid in all_ids if eid.startswith(prefix))
        if q:
            all_ids = (eid for eid in all_ids if q in eid)
        ids = list(islice(all_ids, offset, offset + limit))
        # 带过滤时总数不划算，前端按 -1 处理
        total = -1 if (q or prefix) else len(repo)
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "episode_ids": ids,
        "prefix": prefix,
    }, b""


def _episode_group(episode_id: str) -> tuple[str, str, str]:
    """episode_id → (robot, task, group_prefix)（monitor 展示用层级解析规则）。

    - ``robotwin:<robot>:<split>:<task>:<ep>`` → robot=seg1, task=seg3（前缀含 split 段）
    - ``robotmind:<robot>:<task>:<ep>``       → robot=seg1, task=seg2
    - ``agibot/agibot_beta:<task_id>:<ep>``   → robot=seg0, task=seg1
    其余格式回退 robot=seg0、task 为空（prefix 为空串，代表整库未分组）。
    """
    parts = str(episode_id).split(":")
    family = parts[0] if parts else ""
    if family == "robotwin" and len(parts) >= 5:
        return parts[1], parts[3], ":".join(parts[:4]) + ":"
    if family == "robotmind" and len(parts) >= 4:
        return parts[1], parts[2], ":".join(parts[:3]) + ":"
    if len(parts) >= 3:
        return parts[0], parts[1], ":".join(parts[:2]) + ":"
    return family, "", ""


def op_episode_tree(ctx: PreviewContext, req: dict, progress=None) -> tuple[dict, bytes]:
    """机器人→任务 两级分组树（子id 由 list_episodes?prefix= 分页拉取）。

    全量扫描 episode_id 列按 (robot, task) 聚合计数，保持 ordinal 首次出现
    顺序；结果按 dataset 缓存在 helper 进程内（catalog 为不可变 release）。
    task 段为纯数字时（agibot 系）读每组首条记录的 ``meta.task_name`` 作 label。
    """
    dataset = str(req["dataset"])
    cached = ctx._trees.get(dataset)
    if cached is not None:
        return cached, b""
    repo = ctx.repo(dataset)
    if repo.index_path is not None:
        import sqlite3

        uri = f"{repo.index_path.resolve().as_uri()}?mode=ro&immutable=1"
        connection = sqlite3.connect(uri, uri=True)
        try:
            ids = [
                str(row[0])
                for row in connection.execute(
                    "SELECT episode_id FROM episodes ORDER BY ordinal"
                )
            ]
        finally:
            connection.close()
    else:
        ids = list(repo.iter_episode_ids())
    groups: dict = {}
    order: list = []
    for eid in ids:
        robot, task, prefix = _episode_group(eid)
        key = (robot, task)
        group = groups.get(key)
        if group is None:
            group = {
                "robot": robot,
                "task": task,
                "prefix": prefix,
                "count": 0,
                "first": eid,
            }
            groups[key] = group
            order.append(group)
        group["count"] += 1
    for group in order:
        if group["task"] and group["task"].isdigit():
            try:
                meta = repo.get_episode(group["first"]).info.metadata or {}
                task_name = str(meta.get("task_name") or "")
            except Exception:
                task_name = ""
            if task_name:
                group["label"] = f"{group['task']} · {task_name}"
    robots: list = []
    by_robot: dict = {}
    for group in order:
        node = by_robot.get(group["robot"])
        if node is None:
            node = {"name": group["robot"], "count": 0, "tasks": []}
            by_robot[group["robot"]] = node
            robots.append(node)
        node["count"] += group["count"]
        node["tasks"].append(
            {
                "name": group["task"],
                "label": group.get("label") or group["task"] or "（未分组）",
                "count": group["count"],
                "prefix": group["prefix"],
            }
        )
    payload = {"dataset": dataset, "total": len(ids), "robots": robots}
    ctx._trees[dataset] = payload
    return payload, b""


def _expand_key_names(names: list, field_dims) -> list:
    """按每 key 字段宽度把 per-key 名称展开成逐维名称（宽 1 直接用名）。"""
    out: list = []
    for name, width in zip(names, field_dims):
        width = int(width)
        if width <= 1:
            out.append(str(name))
        else:
            out.extend(f"{name}[{j}]" for j in range(width))
    return out


def _dim_names(info, modality: str) -> list:
    """从 catalog 元数据提取逐维语义名（无标注 → []，前端回退 a{N}/dim_{N}）。

    已知来源：modality schema meta 的 ``feature_names``（agibot_beta/libero，
    逐维）、``key_order_names``（robotmind，per-key；有 ``field_dimensions``
    时展开为逐维），episode 级 meta 的 ``state_action_keys_order``（robotmind，
    同上）。长度与实际维度对不上时由前端按长度校验丢弃。
    """
    try:
        schema = (info.modality_schemas or {}).get(modality)
        schema_meta = dict(schema.metadata) if schema is not None else {}
        episode_meta = dict(info.metadata or {})
    except Exception:
        return []
    feature_names = schema_meta.get("feature_names")
    if (
        isinstance(feature_names, (list, tuple))
        and feature_names
        and all(isinstance(n, str) and n for n in feature_names)
    ):
        return [str(n) for n in feature_names]
    key_names = schema_meta.get("key_order_names") or episode_meta.get(
        "state_action_keys_order"
    )
    if (
        isinstance(key_names, (list, tuple))
        and key_names
        and all(isinstance(n, str) and n for n in key_names)
    ):
        names = [str(n) for n in key_names]
        field_dims = schema_meta.get("field_dimensions") or episode_meta.get(
            "field_dimensions"
        )
        if isinstance(field_dims, (list, tuple)) and len(field_dims) == len(names):
            try:
                return _expand_key_names(names, field_dims)
            except Exception:
                pass
        return names
    return []


def op_episode_info(ctx: PreviewContext, req: dict, progress=None) -> tuple[dict, bytes]:
    repo = ctx.repo(str(req["dataset"]))
    ep = repo.get_episode(str(req["episode_id"]))
    info = ep.info
    texts = _text_items_payload(ep.read_text(range(min(info.text_count, MAX_TEXT_ITEMS))))
    schemas = {
        name: {
            "count": int(schema.count),
            "dtype": str(schema.dtype),
            "item_shape": [None if s is None else int(s) for s in schema.item_shape],
        }
        for name, schema in info.modality_schemas.items()
    }
    ratio_hint = None
    if info.video_count > 0 and info.action_count > 0:
        ratio_hint = round(info.action_count / info.video_count, 4)
    payload = {
        "episode_id": info.episode_id,
        "catalog_id": info.catalog_id,
        "source_dataset": info.source_dataset,
        "source_child": info.source_child,
        "robot_name": info.robot_name,
        "video_count": info.video_count,
        "state_count": info.state_count,
        "action_count": info.action_count,
        "text_count": info.text_count,
        "ratio_hint": ratio_hint,
        "texts": texts,
        "metadata": info.metadata,
        "modality_schemas": schemas,
        "action_names": _dim_names(info, "action"),
        "state_names": _dim_names(info, "state"),
    }
    return payload, b""


def _frame_indices_from_req(req: dict, video_count: int) -> list:
    if req.get("indices") is not None:
        indices = _int_list(req["indices"], name="indices", maximum=MAX_FRAMES_PER_REQUEST)
    else:
        start = max(0, int(req.get("start", 0)))
        count = max(1, min(MAX_FRAMES_PER_REQUEST, int(req.get("count", 24))))
        stride = max(1, int(req.get("stride", 1)))
        indices = [start + i * stride for i in range(count)]
    return [i for i in indices if 0 <= i < video_count]


def op_video_frames(ctx: PreviewContext, req: dict, progress=None) -> tuple[dict, bytes]:
    repo = ctx.repo(str(req["dataset"]))
    ep = repo.get_episode(str(req["episode_id"]))
    indices = _frame_indices_from_req(req, ep.info.video_count)
    if not indices:
        raise ValueError("no valid frame indices in range")
    max_dim = req.get("max_dim")
    # 大请求分块解码并回报进度（ffmpeg 单趟解码不可中断，分块让前端看到推进）
    if progress and len(indices) > 16:
        import torch

        parts = []
        chunk = 16
        for pos in range(0, len(indices), chunk):
            part = indices[pos : pos + chunk]
            parts.append(ep.read_video(part, output_format="rgb_uint8_hwc", video_size=None))
            progress(min(0.99, (pos + len(part)) / len(indices)), "decode_frames")
        frames = torch.cat(parts)
    else:
        frames = ep.read_video(indices, output_format="rgb_uint8_hwc", video_size=None)
    sizes, blob, shapes = _encode_jpegs(frames, max_dim)
    return {
        "indices": indices,
        "sizes": sizes,
        "shapes": shapes,
        "encoding": "jpeg",
        "quality": JPEG_QUALITY,
    }, blob


def _resolve_fps(ep, fallback: float) -> float:
    """解析源视频真实 fps：catalog meta.fps → 探测视频容器（file/tar 源）→ fallback。

    openvid/egoverse 在 catalog meta 里带 fps；robotwin/agibot 等文件型源探测
    mp4 容器；robotmind（HDF5 内 JPEG 帧）无容器 fps，回退请求值。探测失败静默
    回退，不影响预览。访问 ep 的规范化 record/reader（monitor 自持代码，与
    motus_data 内部结构绑定）。
    """
    try:
        meta = getattr(ep.info, "metadata", None) or {}
        fps = meta.get("fps")
        if isinstance(fps, (int, float)) and float(fps) > 0:
            return float(fps)
    except Exception:
        pass
    record = getattr(ep, "_record", None)
    reader = getattr(ep, "_reader", None)
    if record is None:
        return float(fallback)
    try:
        locator = (record.get("video") or {}).get("locator") or {}
        backend = locator.get("storage_backend")
        import decord

        if backend == "file":
            return float(
                decord.VideoReader(str(locator["uri"]), num_threads=1).get_avg_fps()
            )
        if backend == "tar_member" and reader is not None and hasattr(
            reader, "_temporary_video"
        ):
            with reader._temporary_video(record) as path:
                return float(decord.VideoReader(path, num_threads=1).get_avg_fps())
    except Exception:
        pass
    return float(fallback)


def op_video_mp4(ctx: PreviewContext, req: dict, progress=None) -> tuple[dict, bytes]:
    repo = ctx.repo(str(req["dataset"]))
    ep = repo.get_episode(str(req["episode_id"]))
    video_count = ep.info.video_count
    # 真实源 fps（meta → 容器探测 → 请求值兜底），保证预览按真实速度播放
    req_fps = float(int(req.get("fps", 12)))
    fps = max(
        1, min(60, int(round(_resolve_fps(ep, req_fps if req_fps > 0 else 12.0))))
    )
    seconds = req.get("seconds")
    if seconds:
        # 按真实时间取窗：count = seconds × fps，帧连续不跳
        count = max(1, int(float(seconds) * fps))
    else:
        count = int(req.get("count", 96))
    count = max(1, min(MAX_CLIP_FRAMES, count))
    stride = max(1, int(req.get("stride", 1)))
    start = max(0, min(int(req.get("start", 0)), video_count - 1))
    max_dim = req.get("max_dim")
    stop = min(video_count, start + count * stride)
    indices = list(range(start, stop, stride))[:MAX_CLIP_FRAMES]
    if not indices:
        raise ValueError("empty clip range")

    def _frames():
        chunk = 24
        for pos in range(0, len(indices), chunk):
            part = indices[pos : pos + chunk]
            tensor = ep.read_video(part, output_format="rgb_uint8_hwc", video_size=None)
            for i in range(tensor.shape[0]):
                yield tensor[i]

    blob = _encode_mp4(_frames(), fps, max_dim, progress=progress, total=len(indices))
    return {
        "indices": indices,
        "fps": fps,
        "frames": len(indices),
        "encoding": "h264/mp4",
    }, blob


def _downsample(n: int, max_points: int) -> list:
    if n <= 0:
        return []
    stride = max(1, (n + max_points - 1) // max_points)
    return list(range(0, n, stride))


def op_action_state(ctx: PreviewContext, req: dict, progress=None) -> tuple[dict, bytes]:
    repo = ctx.repo(str(req["dataset"]))
    ep = repo.get_episode(str(req["episode_id"]))
    max_points = max(16, min(65536, int(req.get("max_points", DEFAULT_MAX_POINTS))))
    a_start = max(0, int(req.get("start", 0)))
    a_count = req.get("count")
    data = ep.materialize(modalities=("state", "action"))
    action = data.action
    state = data.state

    payload: dict = {"dtype": "<f4"}
    blob_parts: list = []
    if action is not None and action.shape[0] > 0:
        a_total = int(action.shape[0])
        a_stop = a_total if a_count is None else min(a_total, a_start + int(a_count))
        window = list(range(a_start, max(a_start, a_stop)))
        picked = _downsample(len(window), max_points)
        indices = [window[i] for i in picked]
        tensor = action[indices] if indices else action[:0]
        payload.update(
            action_count=a_total,
            action_indices=indices,
            d_action=int(action.shape[1]) if action.dim() > 1 else 1,
            action_names=_dim_names(ep.info, "action"),
        )
        blob_parts.append(_f4_bytes(tensor.reshape(len(indices), -1)))
    else:
        payload.update(action_count=0, action_indices=[], d_action=0)
    if state is not None and state.shape[0] > 0:
        s_total = int(state.shape[0])
        s_indices = _downsample(s_total, max_points)
        tensor = state[s_indices]
        payload.update(
            state_count=s_total,
            state_indices=s_indices,
            d_state=int(state.shape[1]) if state.dim() > 1 else 1,
        )
        blob_parts.append(_f4_bytes(tensor.reshape(len(s_indices), -1)))
    else:
        payload.update(state_count=0, state_indices=[], d_state=0)
    payload["episode_id"] = ep.info.episode_id
    return payload, b"".join(blob_parts)


def op_selection_preview(ctx: PreviewContext, req: dict, progress=None) -> tuple[dict, bytes]:
    repo = ctx.repo(str(req["dataset"]))
    if req.get("selection") is not None:
        selection = dict(req["selection"])
        ep = repo.get_episode(str(selection["episode_id"]))
        info = ep.info
        video_indices = _int_list(
            selection["video_frame_indices"],
            name="video_frame_indices",
            maximum=MAX_SELECTION_FRAMES,
        )
    else:
        ep = repo.get_episode(str(req["episode_id"]))
        info = ep.info
        video_indices = _int_list(
            req["video_frame_indices"],
            name="video_frame_indices",
            maximum=MAX_SELECTION_FRAMES,
        )
        start, last = video_indices[0], video_indices[-1]
        ratio = req.get("action_video_fps_ratio")
        ratio = int(ratio) if ratio else None
        # stride 布局（旧时代 1:6 语义）：视频每 ratio 帧抽 1 帧、动作连续
        # N*ratio 帧、短 episode 末帧填充。判定：ratio>1 且总跨度满 stride
        # 或首个间隔 == ratio（尾部被 clamp 的填充窗口）。
        is_stride = bool(
            ratio
            and ratio > 1
            and len(video_indices) > 1
            and (
                (last - start) == (len(video_indices) - 1) * ratio
                or (video_indices[1] - video_indices[0]) == ratio
            )
        )
        state_indices = req.get("state_indices")
        action_indices = req.get("action_indices")
        if state_indices is None:
            if is_stride:
                state_indices = [start]
            else:
                state_indices = [start * ratio] if ratio else []
        if action_indices is None:
            if is_stride:
                max_action = max(0, int(info.action_count) - 1)
                n_action = (len(video_indices) - 1) * ratio
                action_indices = [
                    min(start + k, max_action) for k in range(1, n_action + 1)
                ]
            else:
                action_indices = (
                    list(range(start * ratio + 1, last * ratio + 1)) if ratio else []
                )
        text_indices = req.get("text_indices")
        selection = {
            "episode_id": info.episode_id,
            "video_frame_indices": video_indices,
            "state_indices": [int(v) for v in state_indices],
            "action_indices": [int(v) for v in action_indices],
            "text_indices": (
                [int(v) for v in text_indices]
                if text_indices is not None
                else list(range(info.text_count))
            ),
        }
    # 与 read_selection 相同的校验，然后分块解码视频以回报进度（重解码是
    # 切片预览的主要耗时，agibot 等 ffmpeg 源尤为明显）
    from motus_data.index_contracts import validate_sample_selection_v1

    validate_sample_selection_v1(selection)
    video_indices = list(selection["video_frame_indices"])
    import torch

    parts = []
    chunk = 12
    for pos in range(0, len(video_indices), chunk):
        part = video_indices[pos : pos + chunk]
        parts.append(ep.read_video(part, output_format="rgb_uint8_hwc", video_size=None))
        if progress:
            progress(min(0.98, (pos + len(part)) / len(video_indices)), "decode_frames")
    frames = torch.cat(parts) if len(parts) > 1 else parts[0]
    state = ep.read_state(selection["state_indices"])
    action = ep.read_action(selection["action_indices"])
    texts = ep.read_text(selection["text_indices"])
    max_dim = req.get("max_dim")
    sizes, jpeg_blob, shapes = _encode_jpegs(frames, max_dim)
    action = action.reshape(action.shape[0], -1) if action.numel() else action
    state = state.reshape(state.shape[0], -1) if state.numel() else state
    payload = {
        "episode_id": info.episode_id,
        "robot_name": info.robot_name,
        "selection": {
            "episode_id": info.episode_id,
            "video_frame_indices": video_indices,
            "state_indices": list(selection["state_indices"]),
            "action_indices": list(selection["action_indices"]),
            "text_indices": list(selection["text_indices"]),
        },
        "jpeg_sizes": sizes,
        "shapes": shapes,
        "d_action": int(action.shape[1]) if action.dim() > 1 and action.numel() else 0,
        "n_action": int(action.shape[0]),
        "d_state": int(state.shape[1]) if state.dim() > 1 and state.numel() else 0,
        "n_state": int(state.shape[0]),
        "action_names": _dim_names(info, "action"),
        "texts": _text_items_payload(texts),
        "dtype": "<f4",
    }
    blob = jpeg_blob + _f4_bytes(action) + _f4_bytes(state)
    return payload, blob


def op_catalog_index(ctx: PreviewContext, req: dict, progress=None) -> tuple[dict, bytes]:
    """覆盖率统计用的 catalog 索引：每 episode 的 video.count（+ 首个 fps）。

    只抽取 episode_id/video.count 两个字段，响应体 gzip 压缩进 blob
    （payload 只放摘要），避免大 catalog 全量 JSON 走未压缩通道。
    """
    import gzip

    dataset = str(req["dataset"])
    if dataset not in ctx.catalogs:
        raise KeyError(
            f"unknown dataset {dataset!r}; available={sorted(ctx.catalogs)}"
        )
    catalog = ctx.catalogs[dataset]
    stat = catalog.stat()
    episodes: list = []
    fps = None
    with catalog.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            episode_id = record.get("episode_id")
            if episode_id is None:
                continue
            count = (record.get("video") or {}).get("count")
            episodes.append(
                {
                    "episode_id": str(episode_id),
                    "frames": int(count) if isinstance(count, (int, float)) else None,
                }
            )
            if fps is None:
                for container in (record.get("meta"), (record.get("video") or {}).get("meta")):
                    if isinstance(container, dict):
                        value = container.get("fps")
                        if isinstance(value, (int, float)) and value > 0:
                            fps = float(value)
                            break
    body = _dumps({"episodes": episodes, "fps": fps}).encode("utf-8")
    payload = {
        "dataset": dataset,
        "catalog_path": str(catalog),
        "size": int(stat.st_size),
        "mtime": float(stat.st_mtime),
        "episodes": len(episodes),
        "compressed": "gzip",
    }
    return payload, gzip.compress(body)


OPS = {
    "list_datasets": op_list_datasets,
    "list_episodes": op_list_episodes,
    "episode_tree": op_episode_tree,
    "episode_info": op_episode_info,
    "catalog_index": op_catalog_index,
    "video_frames": op_video_frames,
    "video_mp4": op_video_mp4,
    "action_state": op_action_state,
    "selection_preview": op_selection_preview,
}


# ---------------------------------------------------------------------------
# --serve 主循环
# ---------------------------------------------------------------------------
def _write_response(stdout, payload: dict, blob: bytes) -> None:
    body = _dumps(payload).encode("utf-8")
    head = _dumps({"ok": True, "json_bytes": len(body), "blob_bytes": len(blob)}).encode(
        "utf-8"
    )
    stdout.write(head + b"\n")
    stdout.write(body)
    stdout.write(blob)
    stdout.flush()


def _write_error(stdout, message: str) -> None:
    stdout.write(_dumps(_err(message)).encode("utf-8") + b"\n")
    stdout.flush()


def _progress_writer(stdout):
    """生成进度回报回调：op 内部在解码分块边界调用，写 interim 进度行。

    进度行 ``{"progress": frac, "stage": ...}`` 在最终响应头之前发出；
    客户端读到带 ok 字段的头行才算响应到达（见 dataset_preview.py）。
    """

    def report(frac: float, stage: str = "decode") -> None:
        stdout.write(
            _dumps({"progress": round(float(frac), 4), "stage": str(stage)}).encode(
                "utf-8"
            )
            + b"\n"
        )
        stdout.flush()

    return report


def serve(args: argparse.Namespace) -> int:
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    stdout.write(
        _dumps({"hello": PROTOCOL, "pid": os.getpid()}).encode("utf-8") + b"\n"
    )
    stdout.flush()

    ctx: PreviewContext | None = None
    idle_timeout = float(args.idle_timeout)
    last_request = time.time()
    try:
        while True:
            if idle_timeout > 0:
                remaining = last_request + idle_timeout - time.time()
                if remaining <= 0:
                    break
                ready, _, _ = select.select([stdin.fileno()], [], [], remaining)
                if not ready:
                    break  # idle 超时兜底退出
            line = stdin.readline()
            if not line:
                break  # EOF：本机 server 已退出/通道断开
            last_request = time.time()
            try:
                req = json.loads(line.decode("utf-8"))
                op = str(req.get("op"))
                handler = OPS.get(op)
                if handler is None:
                    raise ValueError(f"unknown op {op!r}; known={sorted(OPS)}")
                if ctx is None:
                    ctx = PreviewContext(
                        args.repo_root,
                        args.catalog_root,
                        args.catalog or [],
                        motus_data_dir=args.motus_data_dir,
                    )
                payload, blob = handler(ctx, req, _progress_writer(stdout))
                _write_response(stdout, payload, blob)
            except Exception as exc:  # 单个请求失败不影响长驻进程
                _write_error(stdout, f"{type(exc).__name__}: {exc}")
    finally:
        if ctx is not None:
            ctx.close()
    return 0


# ---------------------------------------------------------------------------
# 一次性 CLI（冒烟 / 调试）
# ---------------------------------------------------------------------------
def _filmstrip(frames, max_dim: int, cols: int = 6):
    from PIL import Image

    images = []
    for i in range(frames.shape[0]):
        img = Image.fromarray(frames[i].numpy())
        if max_dim and max(img.size) > max_dim:
            img.thumbnail((max_dim, max_dim), Image.BILINEAR)
        images.append(img)
    if not images:
        raise ValueError("no frames for filmstrip")
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (w * cols, h * rows), (16, 16, 16))
    for pos, img in enumerate(images):
        sheet.paste(img, ((pos % cols) * w, (pos // cols) * h))
    return sheet


def _selection_from_window(episode_id: str, args: argparse.Namespace, text_count: int) -> dict:
    start = int(args.window_start)
    n = int(args.num_video_frames)
    ratio = int(args.ratio)
    return {
        "episode_id": episode_id,
        "video_frame_indices": list(range(start, start + n + 1)),
        "state_indices": [start * ratio],
        "action_indices": list(range(start * ratio + 1, (start + n) * ratio + 1)),
        "text_indices": list(range(text_count)),
    }


def cmd_dump(args: argparse.Namespace) -> int:
    import numpy as np

    ctx = PreviewContext(
        args.repo_root,
        args.catalog_root,
        args.catalog or [],
        motus_data_dir=args.motus_data_dir,
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = str(args.dataset)
    repo = ctx.repo(dataset)
    if args.episode_id:
        episode_id = str(args.episode_id)
    else:
        episode_id = next(repo.iter_episode_ids())
    ep = repo.get_episode(episode_id)

    info_payload, _ = op_episode_info(
        ctx, {"dataset": dataset, "episode_id": episode_id}
    )
    (out_dir / "info.json").write_text(
        json.dumps(info_payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    video_count = ep.info.video_count
    strip_indices = _downsample(video_count, 24)
    frames = ep.read_video(strip_indices, output_format="rgb_uint8_hwc", video_size=None)
    _filmstrip(frames, max_dim=320).save(out_dir / "filmstrip.jpg", quality=88)

    clip_count = min(96, video_count)
    clip_stride = max(1, video_count // clip_count)
    clip_blob = op_video_mp4(
        ctx,
        {
            "dataset": dataset,
            "episode_id": episode_id,
            "start": 0,
            "count": 96,
            "stride": clip_stride,
            "fps": 12,
        },
    )[1]
    (out_dir / "clip.mp4").write_bytes(clip_blob)

    data = ep.materialize(modalities=("state", "action"))
    np.savez(
        out_dir / "action_state.npz",
        action=data.action.numpy() if data.action is not None else np.zeros((0, 0)),
        state=data.state.numpy() if data.state is not None else np.zeros((0, 0)),
    )

    if args.selection:
        selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    elif args.window_start is not None:
        selection = _selection_from_window(episode_id, args, ep.info.text_count)
    else:
        selection = None
    if selection is not None:
        payload, blob = op_selection_preview(
            ctx, {"dataset": dataset, "selection": selection}
        )
        (out_dir / "selection.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        pos = 0
        frames_dir = out_dir / "selection_frames"
        frames_dir.mkdir(exist_ok=True)
        for i, size in enumerate(payload["jpeg_sizes"]):
            (frames_dir / f"{i:03d}.jpg").write_bytes(blob[pos : pos + size])
            pos += size
    print(f"dumped episode {episode_id} -> {out_dir}")
    ctx.close()
    return 0


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        default="",
        help="仓库根目录（提供 motus_data 的 libs/motus-data-factory）；--serve 必填",
    )
    parser.add_argument(
        "--catalog-root",
        default=None,
        help="Episode Catalog 根目录（release 相对路径解析 + 递归扫描补充）",
    )
    parser.add_argument(
        "--catalog",
        action="append",
        default=None,
        help="显式 catalog 映射 name:path（可多次），覆盖发现结果",
    )
    parser.add_argument(
        "--motus-data-dir",
        default=None,
        help="本机上传到集群的 motus_data 包副本目录（sys.path 优先于 repo-root 检出）",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="长驻服务模式（stdin/stdout 帧协议，EOF 退出）",
    )
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=8 * 3600,
        help="--serve 无请求自退秒数（默认 8h；0=不限）",
    )
    # 一次性 dump
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--episode-id", default=None, help="缺省取第一个 episode")
    parser.add_argument("--out", default="/tmp/motus_preview_dump")
    parser.add_argument("--selection", default=None, help="selection JSON 文件")
    parser.add_argument("--window-start", type=int, default=None)
    parser.add_argument("--num-video-frames", type=int, default=48)
    parser.add_argument("--ratio", type=int, default=1)
    return parser


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.serve:
        return serve(args)
    if args.dataset:
        return cmd_dump(args)
    print("error: need --serve or --dataset (one-shot dump)", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
