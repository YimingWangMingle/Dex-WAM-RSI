"""Monitor web server（Flask 版）。

Serves the static SPA from ``monitor/web/`` plus the REST/SSE API:

    GET  /api/health
    GET  /api/runs
    GET  /api/runs/{run_id}
    GET  /api/runs/{run_id}/metrics?stream=metrics|system|events|eval&offset=<o>&tail=<n>
    GET  /api/runs/{run_id}/history?stream=metrics|system&max_points=<n>
    GET  /api/runs/{run_id}/slices?source=&offset=&limit=
    GET  /api/runs/{run_id}/file?path=<rel>  run 数据目录内的文件字节（eval 媒体等）
    GET  /api/tree
    GET  /api/annotations                 list all
    GET  /api/annotations?ckpt=<key>      single annotation
    POST /api/annotations                 create/replace one (LLM-friendly)
    GET  /api/sync/status
    GET  /api/auth/whoami                 登录状态 {auth_enabled, user}
    GET  /api/data_root                   当前用户的查看根/启动目录/同步写入目录
    POST /api/data_root {path}            切换当前用户的查看根（持久化于其 workspace）
    GET  /api/browse?path=<abs>[&remote=<名>] 只读目录浏览（辅助选择查看根；remote 经 SSH 浏览远端；只列子目录，不读文件）
    GET  /api/known_roots                 后台扫描发现的监控数据目录（roots 本机 + remote_roots 远端）
    GET  /api/remotes                     同步远端列表（全局 + 当前用户自加）
    POST /api/remotes {path, remote?}     把远端目录加为当前用户的同步远端（拉进其 workspace，持久化 workspace.json）
    GET  /api/events                      Server-Sent Events change feed
    GET  /api/llm/help                    machine-readable API documentation
    GET  /api/dataset/status              preview backend state (ssh/local/disabled)
    GET  /api/dataset/datasets            episode catalogs visible to the helper
    GET  /api/dataset/episodes?dataset=&offset=&limit=&q=&prefix=
    GET  /api/dataset/episode_tree?dataset=  robot→task hierarchy (counts, prefixes)
    GET  /api/dataset/episode?dataset=&id=        episode info + texts
    GET  /api/dataset/action?dataset=&id=&max_points=&start=&count=
    GET  /api/dataset/frames?dataset=&id=&indices=|start&count&stride&max_dim=
    GET  /api/dataset/clip?dataset=&id=&start=&count=&stride=&fps=&max_dim=
    GET  /api/dataset/selection?dataset=&episode_id=&start=&n=&ratio=  (or selection=<json>, or frames=<json>)

安全模型：

- 强制登录（``create_app(..., user_store=...)``，cli 恒传入）：除 ``/login``、
  ``/register`` 与 ``/static/*`` 外全部端点要求登录（Flask session）；首个注册
  账户自动批准，其后注册写入 pending，由仅本机的管理后台审批。public 模式额外
  按 ``HostAllowlist`` 校验 ``Host`` 头防 DNS rebinding，由 cli 以 HTTPS（内部
  CA 证书）承载。
- 每用户 workspace：查看根与网页添加的同步远端都按登录用户隔离持久化在
  ``<启动目录>/workspaces/<user>/``（互不影响）；无会话的兼容入口
  （``make_server``/``serve_forever``，无 user_store）维持旧的进程全局查看根行为。
- 目录浏览（``/api/browse``，本机与远端）只读文件树：只列子目录名与
  ``runs/`` 标记，不提供任何文件内容的读取/下载；run 文件字节仍仅限
  ``/api/runs/{id}/file`` 的 ``runs/<run_id>`` 沙箱内。
- 不主动发 ``Access-Control-Allow-Origin``（SPA 同源，不需要 CORS）；需要跨域时
  由 ``cors_origin`` 显式配置单一来源。
- dataset 重渲染端点（frames/clip/selection）有并发闸门（大小=heavy workers 数），
  超限返回 429，避免打满集群解码池。
"""

from __future__ import annotations

import bisect
import gzip
import json
import logging
import mimetypes
import os
import queue
import re
import threading
import time
import urllib.parse
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, Response, redirect, request, send_file, session

from .paths import DATA_DIR_NAME, repo_root
from .registry import Registry
from .store import ChunkedJsonlReader, atomic_write_json, read_json

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent / "web"
STREAM_NAMES = ("metrics", "system", "events", "eval")

_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9._-]+$")

# 走集群重解码的 dataset 子端点（并发闸门保护）
_HEAVY_DATASET_SUBS = ("frames", "clip", "selection")


def _valid_run_id(run_id: str) -> bool:
    return bool(_SAFE_RUN_ID.match(run_id or "")) and run_id not in (".", "..")


def _safe_username(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name or ""))[:64] or "user"


class Workspace:
    """每登录用户一个 workspace：``<启动目录>/workspaces/<user>/``。

    - ``workspace.json``：``{"view_root": str, "added_remotes": [...]}``
      （查看根与网页添加的同步远端都按用户隔离，互不影响）；
    - ``data/``：该用户网页添加的同步远端的本地拉取目录（加远端后其查看根
      自动切到这里，同步到位后 run 即出现）。
    """

    def __init__(self, startup_root: object, username: str):
        self.username = str(username)
        self.dir = Path(str(startup_root)) / "workspaces" / _safe_username(username)
        self.data_dir = self.dir / "data"
        self.path = self.dir / "workspace.json"

    def _load(self) -> Dict[str, Any]:
        doc = read_json(self.path, default=None)
        return doc if isinstance(doc, dict) else {}

    def _save(self, doc: Dict[str, Any]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.path, doc)

    def view_root(self, default: Path) -> Path:
        raw = str(self._load().get("view_root") or "")
        if raw:
            candidate = Path(raw)
            if candidate.is_dir():
                return candidate
        return Path(default)

    def set_view_root(self, root: object, default: Path) -> None:
        doc = self._load()
        if Path(str(root)) == Path(default):
            doc.pop("view_root", None)
        else:
            doc["view_root"] = str(Path(str(root)))
        self._save(doc)

    def added_remotes(self) -> List[Dict[str, Any]]:
        items = self._load().get("added_remotes")
        if not isinstance(items, list):
            return []
        return [dict(i) for i in items if isinstance(i, dict)]

    def add_remote(self, item: Dict[str, Any]) -> None:
        doc = self._load()
        items = doc.get("added_remotes")
        if not isinstance(items, list):
            items = []
        items.append(dict(item))
        doc["added_remotes"] = items
        self._save(doc)


class ViewContext:
    """一次请求的有效视图：按当前登录用户 workspace 解析的数据根 + 对应 Registry。"""

    __slots__ = ("state", "root", "registry")

    def __init__(self, state: "ServerState", root: Path):
        self.state = state
        self.root = root
        self.registry = state.registry_for(root)


class ServerState:
    """Shared state across request-handler threads."""

    def __init__(self, data_root: object, sync_engine: Any = None, dataset_backend: Any = None):
        self.data_root = Path(str(data_root))
        self.startup_data_root = self.data_root
        self.registry = Registry(self.data_root)
        # 无会话兼容模式（make_server/serve_forever，无 user_store）的运行时查看根
        # 仍持久化在启动目录的 .view_root.json；登录用户的默认查看根也是它，
        # 各用户自己的切换持久化在各自 workspace（见 Workspace）。
        view = read_json(self.startup_data_root / ".view_root.json", default=None)
        if isinstance(view, dict):
            candidate = Path(str(view.get("view_root") or ""))
            if candidate.is_dir() and candidate != self.data_root:
                self.data_root = candidate
                self.registry = Registry(candidate)
        self._registries: Dict[str, Registry] = {str(self.data_root): self.registry}
        self._reg_lock = threading.Lock()
        self._root_lock = threading.Lock()
        self.sync_engine = sync_engine
        self.sync_engines: List[Any] = [sync_engine] if sync_engine is not None else []
        self.dataset_backend = dataset_backend
        self.version = 0
        self.history_cache: Dict[str, Any] = {}  # root|run_id|... -> (end_offset, payload)
        self.slices_cache: Dict[str, Any] = {}  # root|run_id|... -> (fingerprint, payload)
        self.catalog_cache: Dict[str, Any] = {}  # catalog path -> (fingerprint, payload)
        self._subscribers: List["queue.Queue"] = []
        self._sub_lock = threading.Lock()
        self._stop = threading.Event()
        workers = getattr(dataset_backend, "workers", 2) if dataset_backend is not None else 2
        self.heavy_gate = threading.BoundedSemaphore(max(1, int(workers)))

    def registry_for(self, root: object) -> Registry:
        """按数据根取（并缓存）Registry：同一路径多用户共享一个实例。"""
        key = str(root)
        with self._reg_lock:
            registry = self._registries.get(key)
            if registry is None:
                registry = Registry(key)
                self._registries[key] = registry
            return registry

    def workspace(self, username: str) -> Workspace:
        return Workspace(self.startup_data_root, username)

    def request_root(self) -> Path:
        """当前请求的有效数据根：登录用户取其 workspace 的 view_root，否则全局默认根。"""
        user = session.get("user") if request else None
        if user:
            return self.workspace(user).view_root(self.data_root)
        return self.data_root

    def view(self) -> ViewContext:
        return ViewContext(self, self.request_root())

    def set_data_root(self, path: Path) -> None:
        """切换**全局默认**查看根（仅无会话兼容模式用；登录用户走 workspace）。"""
        with self._root_lock:
            self.data_root = path
            with self._reg_lock:
                self.registry = self._registries.get(str(path)) or Registry(path)
                self._registries[str(path)] = self.registry
        self._broadcast({"type": "update", "version": self.version, "ts": time.time()})
        marker = self.startup_data_root / ".view_root.json"
        try:
            if path == self.startup_data_root:
                marker.unlink(missing_ok=True)
            else:
                atomic_write_json(marker, {"view_root": str(path)})
        except OSError as exc:
            logger.warning("monitor: cannot persist view root to %s: %s", marker, exc)

    # -- change notification (drives SSE) -----------------------------------
    def subscribe(self) -> "queue.Queue":
        q: "queue.Queue" = queue.Queue(maxsize=64)
        with self._sub_lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: "queue.Queue") -> None:
        with self._sub_lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def _broadcast(self, payload: Dict[str, Any]) -> None:
        with self._sub_lock:
            subscribers = list(self._subscribers)
        for q in subscribers:
            try:
                q.put_nowait(payload)
            except queue.Full:
                pass

    def _data_signature(self) -> int:
        if self.sync_engines:
            return sum(int(engine.data_version) for engine in self.sync_engines)
        # Local mode: cheap fs signature over the runs directory.
        total = 0
        runs_dir = self.data_root / "runs"
        if runs_dir.is_dir():
            for path in runs_dir.rglob("*"):
                if path.is_file():
                    try:
                        total += path.stat().st_size
                    except OSError:
                        continue
        return total

    def notifier_loop(self, interval: float = 1.5) -> None:
        while not self._stop.is_set():
            try:
                signature = self._data_signature()
                if signature != self.version:
                    self.version = signature
                    self._broadcast(
                        {"type": "update", "version": self.version, "ts": time.time()}
                    )
            except Exception as exc:
                logger.debug("monitor notifier error: %s", exc)
            self._stop.wait(interval)

    def start_notifier(self) -> threading.Thread:
        thread = threading.Thread(
            target=self.notifier_loop, name="motus-monitor-notifier", daemon=True
        )
        thread.start()
        return thread


# ---------------------------------------------------------------------------
# 已知监控数据目录后台扫描（切换查看根的快捷列表）
# ---------------------------------------------------------------------------
KNOWN_ROOTS_CACHE = ".known_roots.json"
KNOWN_ROOTS_FIRST_DELAY = 30.0   # 启动 30s 后首扫
KNOWN_ROOTS_INTERVAL = 600.0     # 之后每 10 分钟一轮
KNOWN_ROOTS_MAX_DIRS = 200       # 单轮最多查看的目录数
KNOWN_ROOTS_MAX_SECONDS = 5.0    # 单轮耗时上限（超时本轮提前结束）


def _probe_data_root(path: Path) -> Optional[Dict[str, Any]]:
    """path 若含 runs/ 子目录则返回 {path, run_count, latest_ts}，否则 None。"""
    try:
        runs = path / "runs"
        if not runs.is_dir():
            return None
        count = 0
        latest = 0.0
        with os.scandir(str(runs)) as it:
            for entry in it:
                try:
                    if not entry.is_dir():
                        continue
                    count += 1
                    mtime = entry.stat().st_mtime
                    if mtime > latest:
                        latest = mtime
                except OSError:
                    continue
        return {"path": str(path), "run_count": count, "latest_ts": latest or None}
    except OSError:
        return None


class KnownRootsScanner:
    """低负载后台浅扫常见父目录，发现含 runs/ 的监控数据目录。

    候选父目录：启动 data_root 及其父目录、默认 ``<仓库上一级>/motus_monitor_data``、
    ``~/.local/share/motus_monitor_data``、各 sync remote 的 local_dir 及其父目录。
    结果原子写 ``<startup_data_root>/.known_roots.json``（启动时先读缓存立即可用）。
    单线程、限量限耗时、所有异常静默——扫描失败绝不能影响服务。
    """

    def __init__(self, startup_data_root: object, sync_engines: Any = ()):
        self.startup_data_root = Path(str(startup_data_root))
        self.cache_path = self.startup_data_root / KNOWN_ROOTS_CACHE
        self._sync_engines = list(sync_engines or [])
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        cached = read_json(self.cache_path, default=None)
        if not isinstance(cached, dict):
            cached = {}
        roots = cached.get("roots")
        remote_roots = cached.get("remote_roots")
        self._snapshot: Dict[str, Any] = {
            "roots": roots if isinstance(roots, list) else [],
            "remote_roots": remote_roots if isinstance(remote_roots, list) else [],
            "scanned_at": cached.get("scanned_at"),
        }

    def start(self) -> threading.Thread:
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._loop, name="motus-monitor-known-roots", daemon=True
            )
            self._thread.start()
        return self._thread

    def stop(self) -> None:
        self._stop.set()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "roots": list(self._snapshot["roots"]),
                "remote_roots": list(self._snapshot.get("remote_roots") or []),
                "scanned_at": self._snapshot["scanned_at"],
            }

    def _loop(self) -> None:
        if self._stop.wait(KNOWN_ROOTS_FIRST_DELAY):
            return
        while not self._stop.is_set():
            try:
                self.scan_once()
            except Exception:
                pass
            self._stop.wait(KNOWN_ROOTS_INTERVAL)

    def _candidates(self) -> List[Path]:
        candidates: List[Path] = []

        def add(p: Any) -> None:
            try:
                path = Path(str(p)).expanduser()
            except Exception:
                return
            if path not in candidates:
                candidates.append(path)

        add(self.startup_data_root)
        add(self.startup_data_root.parent)
        add(repo_root().parent / DATA_DIR_NAME)
        add(Path.home() / ".local" / "share" / DATA_DIR_NAME)
        for engine in self._sync_engines:
            local_dir = getattr(engine, "local_dir", None)
            if local_dir:
                add(local_dir)
                add(Path(str(local_dir)).parent)
        return [p for p in candidates if p.is_dir()]

    def scan_once(self) -> None:
        deadline = time.monotonic() + KNOWN_ROOTS_MAX_SECONDS
        found: Dict[str, Dict[str, Any]] = {}
        examined = 0
        for candidate in self._candidates():
            if examined >= KNOWN_ROOTS_MAX_DIRS or time.monotonic() > deadline:
                break
            examined = self._walk(candidate, 0, found, examined, deadline)
        roots = sorted(found.values(), key=lambda r: -(r.get("latest_ts") or 0))
        snapshot = {
            "roots": roots,
            "remote_roots": self._scan_remote_roots(),
            "scanned_at": time.time(),
        }
        with self._lock:
            self._snapshot = snapshot
        try:
            atomic_write_json(self.cache_path, snapshot)
        except OSError:
            pass

    def _scan_remote_roots(self) -> List[Dict[str, Any]]:
        """经 SSH 浅扫各 ssh 远端（候选=远端数据根及其父目录），找含 runs/ 的目录。

        先按 (target, remote_dir) 去重（不同用户的引擎可能指向同一远端目录），
        每个唯一远端一次一次性 ssh 调用（远端脚本自带目录数/耗时上限）；全部异常
        静默——远端不可达只让本轮 remote_roots 为空，不影响本地扫描结果。
        """
        from .sync import SshBackend, SSH_SCAN_ROOTS_SCRIPT, ssh_run_json

        out: List[Dict[str, Any]] = []
        seen_paths = set()
        backends = []
        seen_backends = set()
        for engine in self._sync_engines:
            backend = getattr(engine, "backend", None)
            if not isinstance(backend, SshBackend):
                continue
            key = (backend.target, backend.remote_dir)
            if key in seen_backends:
                continue
            seen_backends.add(key)
            backends.append((engine, backend))
        for engine, backend in backends:
            base = backend.remote_dir.rstrip("/")
            parent = base.rsplit("/", 1)[0] or "/"
            candidates = [base] if parent == base else [base, parent]
            try:
                rows = ssh_run_json(
                    backend.target, backend.ssh_args,
                    SSH_SCAN_ROOTS_SCRIPT, candidates, timeout=30,
                )
            except Exception as exc:
                logger.debug("known-roots remote scan failed [%s]: %s", engine.name, exc)
                continue
            name = engine.name or "remote"
            for row in rows or []:
                if not isinstance(row, dict) or not row.get("path"):
                    continue
                key = (backend.target, str(row["path"]))
                if key in seen_paths:
                    continue
                seen_paths.add(key)
                out.append(
                    {
                        "path": str(row["path"]),
                        "run_count": int(row.get("run_count") or 0),
                        "latest_ts": row.get("latest_ts"),
                        "remote": name,
                        "remote_target": backend.target,
                    }
                )
        return out

    def _walk(
        self,
        path: Path,
        depth: int,
        found: Dict[str, Dict[str, Any]],
        examined: int,
        deadline: float,
    ) -> int:
        if examined >= KNOWN_ROOTS_MAX_DIRS or time.monotonic() > deadline:
            return examined
        examined += 1
        info = _probe_data_root(path)
        if info is not None:
            found[info["path"]] = info
            return examined  # 数据根本身不再下钻
        if depth >= 2:
            return examined
        try:
            with os.scandir(str(path)) as it:
                children = [
                    entry for entry in it
                    if not entry.name.startswith(".") and entry.is_dir()
                ]
        except OSError:
            return examined
        for child in children:
            if examined >= KNOWN_ROOTS_MAX_DIRS or time.monotonic() > deadline:
                break
            examined = self._walk(
                Path(child.path), depth + 1, found, examined, deadline
            )
        return examined


# 同一启动目录可能建多个主站 app（如测试/嵌入场景），扫描线程按启动目录共享一个实例
_KNOWN_ROOTS_SCANNERS: Dict[str, KnownRootsScanner] = {}
_KNOWN_ROOTS_LOCK = threading.Lock()


def _shared_known_roots_scanner(
    startup_data_root: object, sync_engines: Any
) -> KnownRootsScanner:
    key = str(startup_data_root)
    with _KNOWN_ROOTS_LOCK:
        scanner = _KNOWN_ROOTS_SCANNERS.get(key)
        if scanner is None:
            scanner = KnownRootsScanner(startup_data_root, sync_engines)
            _KNOWN_ROOTS_SCANNERS[key] = scanner
            scanner.start()
    return scanner


# ---------------------------------------------------------------------------
# 响应小工具
# ---------------------------------------------------------------------------
def _json(payload: Any, status: int = 200) -> Response:
    body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
    return Response(
        body,
        status=status,
        content_type="application/json; charset=utf-8",
        headers={"Cache-Control": "no-store"},
    )


def _error(status: int, message: str) -> Response:
    return _json({"error": message}, status=status)


def _bytes(body: bytes, content_type: str) -> Response:
    return Response(
        body, content_type=content_type, headers={"Cache-Control": "no-store"}
    )


def _packed(payload: Any, blob: bytes, content_type: str = "application/octet-stream") -> Response:
    """4 字节大端 JSON 长度 + JSON 头 + blob 的单体二进制响应。"""
    body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
    return _bytes(len(body).to_bytes(4, "big") + body + blob, content_type)


def _file_bytes(path: Path) -> Response:
    try:
        body = path.read_bytes()
    except OSError:
        return _error(404, f"not found: {path.name}")
    content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    if path.suffix == ".js":
        content_type = "text/javascript"
    return _bytes(body, content_type)


def static_file(name: str) -> Response:
    """/static/<name>：resolve 后必须仍落在 web/ 内（relative_to 防逃逸）。"""
    base = WEB_DIR.resolve()
    target = (base / name).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        return _error(404, "not found")
    if not target.is_file():
        return _error(404, "not found")
    return _file_bytes(target)


# ---------------------------------------------------------------------------
# dataset 预览相关
# ---------------------------------------------------------------------------
def _q1(name: str, default: Optional[str] = None) -> Optional[str]:
    value = request.args.get(name)
    return value if value is not None else default


def _qint(name: str, default: int) -> int:
    try:
        return int(request.args.get(name, default))
    except (TypeError, ValueError):
        return default


def _stream_dataset(state: ServerState, backend: Any, req: Dict[str, Any], *, packed: bool, content_type: str) -> Response:
    """stream=1 的重渲染端点：响应体 = ``PRG <frac> <stage>\\n`` 行（0..N 条）
    + ``LEN <n>\\n`` + n 字节成品（packed 或裸 blob）；出错为 ``ERR <json>\\n``。
    前端 fetch 流式读取并实时更新进度条。并发闸门在 worker 线程内持有整个解码期。"""
    from .dataset_preview import DatasetPreviewError

    updates: "queue.Queue" = queue.Queue(maxsize=256)
    done = object()

    def worker() -> None:
        acquired = state.heavy_gate.acquire(timeout=30)
        if not acquired:
            updates.put(("ERR", "dataset decode workers busy; retry later"))
            updates.put(done)
            return
        try:
            def progress(frac: float, stage: str) -> None:
                updates.put(f"PRG {frac:.3f} {stage}\n".encode("utf-8"))

            payload, blob = backend.cached_request(progress=progress, _timeout=600, **req)
            if packed:
                head = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
                body = len(head).to_bytes(4, "big") + head + blob
            else:
                body = blob
            updates.put(("LEN", body))
        except (DatasetPreviewError, KeyError, ValueError) as exc:
            updates.put(("ERR", str(exc)))
        except Exception as exc:  # worker 线程兜底：任何异常都必须变成 ERR 行
            updates.put(("ERR", f"{type(exc).__name__}: {exc}"))
        finally:
            state.heavy_gate.release()
            updates.put(done)

    threading.Thread(target=worker, name="motus-dataset-stream", daemon=True).start()

    def gen():
        while True:
            item = updates.get()
            if item is done:
                break
            if isinstance(item, tuple):
                if item[0] == "LEN":
                    body = item[1]
                    yield f"LEN {len(body)}\n".encode("utf-8")
                    yield body
                else:
                    yield b"ERR " + json.dumps(
                        {"error": item[1]}, ensure_ascii=False
                    ).encode("utf-8") + b"\n"
            else:
                yield item

    return Response(gen(), content_type=content_type)


def _handle_dataset(state: ServerState, sub: str) -> Response:
    from .dataset_preview import DatasetPreviewError

    backend = state.dataset_backend
    if backend is None:
        return _error(
            400,
            "dataset preview not configured; start serve with --remote (ssh) "
            "or --dataset-local",
        )
    dataset = _q1("dataset")
    episode_id = _q1("id") or _q1("episode_id")

    # 重渲染端点并发闸门（stream 路径在 worker 线程内自持闸门）
    gated = sub in _HEAVY_DATASET_SUBS
    acquired = False
    if gated and _q1("stream") != "1":
        acquired = state.heavy_gate.acquire(timeout=30)
        if not acquired:
            return _error(429, "dataset decode workers busy; retry later")
    try:
        if sub == "datasets":
            payload, _ = backend.request(op="list_datasets")
            return _json(payload)
        if sub == "episodes":
            if not dataset:
                return _error(400, "need dataset=")
            payload, _ = backend.request(
                op="list_episodes",
                dataset=dataset,
                offset=max(0, _qint("offset", 0)),
                limit=max(1, min(500, _qint("limit", 50))),
                q=_q1("q"),
                prefix=_q1("prefix"),
            )
            return _json(payload)
        if sub == "episode_tree":
            if not dataset:
                return _error(400, "need dataset=")
            payload, _ = backend.request(op="episode_tree", dataset=dataset)
            return _json(payload)
        if sub == "episode":
            if not dataset or not episode_id:
                return _error(400, "need dataset= and id=")
            payload, _ = backend.request(
                op="episode_info", dataset=dataset, episode_id=episode_id
            )
            return _json(payload)
        if sub == "action":
            if not dataset or not episode_id:
                return _error(400, "need dataset= and id=")
            req: Dict[str, Any] = {
                "op": "action_state",
                "dataset": dataset,
                "episode_id": episode_id,
                "max_points": _qint("max_points", 4096),
                "start": _qint("start", 0),
            }
            if _q1("count") is not None:
                req["count"] = _qint("count", 0)
            payload, blob = backend.request(**req)
            return _packed(payload, blob)
        if sub == "frames":
            if not dataset or not episode_id:
                return _error(400, "need dataset= and id=")
            req = {
                "op": "video_frames",
                "dataset": dataset,
                "episode_id": episode_id,
                "max_dim": _qint("max_dim", 0) or None,
            }
            indices = _q1("indices")
            if indices:
                req["indices"] = [int(v) for v in indices.split(",") if v.strip()]
            else:
                req["start"] = _qint("start", 0)
                req["count"] = _qint("count", 24)
                req["stride"] = _qint("stride", 1)
            if _q1("stream") == "1":
                return _stream_dataset(state, backend, req, packed=True,
                                       content_type="application/octet-stream")
            payload, blob = backend.cached_request(_timeout=600, **req)
            return _packed(payload, blob)
        if sub == "clip":
            if not dataset or not episode_id:
                return _error(400, "need dataset= and id=")
            req = {
                "op": "video_mp4",
                "dataset": dataset,
                "episode_id": episode_id,
                "start": _qint("start", 0),
                "count": _qint("count", 96),
                "stride": _qint("stride", 1),
                "fps": _qint("fps", 12),
                "max_dim": _qint("max_dim", 0) or None,
            }
            raw_seconds = _q1("seconds")
            if raw_seconds is not None:
                try:
                    req["seconds"] = float(raw_seconds)
                except (TypeError, ValueError):
                    pass
            if _q1("stream") == "1":
                return _stream_dataset(state, backend, req, packed=False,
                                       content_type="video/mp4")
            payload, blob = backend.cached_request(_timeout=600, **req)
            if not blob:
                return _error(502, f"empty clip: {payload}")
            return _bytes(blob, "video/mp4")
        if sub == "selection":
            if not dataset:
                return _error(400, "need dataset=")
            raw_selection = _q1("selection")
            if raw_selection:
                try:
                    selection = json.loads(raw_selection)
                except ValueError:
                    return _error(400, "invalid selection JSON")
                req = {"op": "selection_preview", "dataset": dataset,
                       "selection": selection}
            else:
                if not episode_id:
                    return _error(400, "need episode_id=")
                start = _qint("start", 0)
                n = max(1, min(63, _qint("n", 48)))
                # frames=<json list>：训练切片的真实视频索引（stride 布局为
                # 跳帧+末帧填充）；缺省回退连续 range(start, start+n+1)
                raw_frames = _q1("frames")
                if raw_frames:
                    try:
                        video_indices = [int(v) for v in json.loads(raw_frames)]
                    except (TypeError, ValueError):
                        return _error(400, "invalid frames JSON")
                    if not video_indices:
                        return _error(400, "frames must be non-empty")
                else:
                    video_indices = list(range(start, start + n + 1))
                req = {
                    "op": "selection_preview",
                    "dataset": dataset,
                    "episode_id": episode_id,
                    "video_frame_indices": video_indices,
                    "action_video_fps_ratio": _qint("ratio", 1),
                    "text_indices": None,
                }
            max_dim = _qint("max_dim", 0)
            if max_dim:
                req["max_dim"] = max_dim
            if _q1("stream") == "1":
                return _stream_dataset(state, backend, req, packed=True,
                                       content_type="application/octet-stream")
            payload, blob = backend.cached_request(_timeout=600, **req)
            return _packed(payload, blob)
        return _error(404, f"unknown dataset endpoint: {sub}")
    except DatasetPreviewError as exc:
        return _error(502, str(exc))
    except (KeyError, ValueError) as exc:
        return _error(400, str(exc))
    finally:
        if acquired:
            state.heavy_gate.release()


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
def create_app(
    data_root: object,
    sync_engine: Any = None,
    dataset_backend: Any = None,
    *,
    sync_engines: Optional[List[Any]] = None,
    user_store: Any = None,
    host_allowlist: Any = None,
    cors_origin: Optional[str] = None,
    secure_cookies: bool = False,
) -> Flask:
    """构建主站 Flask app。

    ``user_store`` 非空即强制登录：除 /login、/register、/static/* 外全部
    要求会话；``host_allowlist``（``HostAllowlist`` 或 host 列表）非空时对
    Host 头做白名单校验（防 DNS rebinding），列表运行时可由管理后台追加。
    ``secure_cookies`` 只在 HTTPS 承载（public 模式）时打开——HTTP 的 local
    模式打开会导致浏览器拒存 session cookie（非 localhost 来源直接登不上）。
    """
    # static_folder=None：禁用 Flask 内置 /static/ 路由（用下方带逃逸防护的自定义路由）
    app = Flask("motus_monitor", static_folder=None)
    state = ServerState(
        data_root,
        sync_engine=sync_engine,
        dataset_backend=dataset_backend,
    )
    if sync_engines:
        state.sync_engines = list(sync_engines)
        state.sync_engine = sync_engines[0] if sync_engines else None
    state.start_notifier()
    state.known_roots_scanner = _shared_known_roots_scanner(
        state.startup_data_root, state.sync_engines
    )
    app.state = state  # type: ignore[attr-defined]
    app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024

    if user_store is not None:
        from .auth import load_or_create_secret_key

        app.secret_key = load_or_create_secret_key(state.startup_data_root)
        app.config.update(
            SESSION_COOKIE_HTTPONLY=True,
            SESSION_COOKIE_SAMESITE="Lax",
            SESSION_COOKIE_SECURE=bool(secure_cookies),
        )

    allowlist = None
    if host_allowlist is not None:
        from .auth import HostAllowlist

        if isinstance(host_allowlist, HostAllowlist):
            allowlist = host_allowlist
        else:
            allowlist = HostAllowlist(state.startup_data_root, host_allowlist)

    @app.before_request
    def _security_gate():
        # Host 白名单（public 模式；防 DNS rebinding 驱动的浏览器攻击）
        if allowlist is not None:
            host = (request.host or "").split(":", 1)[0].lower()
            if host not in allowlist and host not in ("localhost", "127.0.0.1", "::1"):
                return _error(403, f"host not allowed: {host}")
        if user_store is None:
            return None
        path = request.path
        if path in ("/login", "/logout", "/register") or path.startswith("/static/"):
            return None
        if session.get("user"):
            return None
        if path.startswith("/api/"):
            return _error(401, "authentication required")
        nxt = request.full_path if request.query_string else request.path
        return redirect(f"/login?next={urllib.parse.quote(nxt, safe='')}")

    @app.after_request
    def _cors(resp: Response) -> Response:
        if cors_origin:
            resp.headers["Access-Control-Allow-Origin"] = cors_origin
        return resp

    # ------------------------------------------------------------ 登录/登出
    if user_store is not None:
        from .auth import LoginRateLimiter

        limiter = LoginRateLimiter()

        @app.route("/login", methods=["GET", "POST"])
        def login():
            if request.method == "GET":
                return _file_bytes(WEB_DIR / "login.html")
            client = request.headers.get("X-Forwarded-For", request.remote_addr or "?")
            if not limiter.allow(f"{client}"):
                return _error(429, "too many login attempts; retry later")
            body = request.get_json(silent=True) or request.form
            username = str(body.get("username") or "")
            password = str(body.get("password") or "")
            verdict = user_store.check(username, password)
            if verdict == "pending":
                return _error(403, "账户已注册，正在等待管理员审批")
            if verdict != "ok":
                return _error(401, "用户名或密码错误")
            session.clear()
            session["user"] = username
            nxt = str(body.get("next") or "/")
            if not nxt.startswith("/") or nxt.startswith("//"):
                nxt = "/"
            if request.get_json(silent=True) is not None:
                return _json({"ok": True, "next": nxt})
            return redirect(nxt)

        @app.route("/register", methods=["GET", "POST"])
        def register():
            if request.method == "GET":
                return _file_bytes(WEB_DIR / "register.html")
            client = request.headers.get("X-Forwarded-For", request.remote_addr or "?")
            if not limiter.allow(f"reg:{client}"):
                return _error(429, "too many attempts; retry later")
            body = request.get_json(silent=True) or request.form
            username = str(body.get("username") or "")
            password = str(body.get("password") or "")
            try:
                status = user_store.register(username, password)
            except ValueError as exc:
                return _error(400, str(exc))
            # 首个账户自动 approved（可直接登录）；其后 pending：等管理后台审批
            return _json({"ok": True, "pending": status == "pending"})

        @app.route("/logout")
        def logout():
            session.clear()
            return redirect("/login")

    # ------------------------------------------------------------ 静态与页面
    @app.get("/")
    @app.get("/index.html")
    def index():
        return _file_bytes(WEB_DIR / "index.html")

    @app.get("/static/<path:name>")
    def static_assets(name: str):
        return static_file(name)

    # ------------------------------------------------------------ 基础 API
    @app.get("/api/health")
    def api_health():
        return _json({"ok": True, "ts": time.time()})

    @app.get("/api/auth/whoami")
    def api_whoami():
        """登录状态：无 user_store（兼容入口）auth_enabled=false；否则取 session。"""
        if user_store is None:
            return _json({"auth_enabled": False, "user": None,
                          "view_root": str(state.data_root)})
        return _json({"auth_enabled": True, "user": session.get("user"),
                      "view_root": str(state.request_root())})

    @app.get("/api/runs")
    def api_runs():
        return _json({"runs": state.view().registry.list_runs()})

    @app.get("/api/runs/<run_id>")
    @app.get("/api/runs/<run_id>/<tail>")
    def api_run(run_id: str, tail: Optional[str] = None):
        if not _valid_run_id(run_id):
            return _error(400, f"invalid run_id: {run_id!r}")
        if tail == "metrics":
            return _handle_metrics(state.view(), run_id)
        if tail == "history":
            return _handle_history(state.view(), run_id)
        if tail == "slices":
            return _handle_slices(state.view(), run_id)
        if tail == "file":
            return _handle_run_file(state.view(), run_id)
        if tail is None:
            detail = state.view().registry.get_run(run_id)
            if detail is None:
                return _error(404, f"unknown run: {run_id}")
            return _json(detail)
        return _error(404, "unknown endpoint")

    @app.get("/api/tree")
    def api_tree():
        return _json(state.view().registry.build_tree())

    @app.route("/api/annotations", methods=["GET", "POST"])
    def api_annotations():
        if request.method == "POST":
            body = request.get_json(silent=True)
            if not isinstance(body, dict):
                return _error(400, "expected JSON object body")
            return _handle_post_annotation(state.view(), body)
        registry = state.view().registry
        key = request.args.get("ckpt")
        if key:
            doc = registry.get_annotation(key)
            if doc is None:
                return _error(404, f"no annotation for {key}")
            return _json(doc)
        return _json({"annotations": registry.list_annotations()})

    def _visible_engines() -> List[Any]:
        """当前请求可见的同步引擎：全局引擎 + 当前登录用户自己添加的引擎。"""
        user = session.get("user")
        return [
            e
            for e in state.sync_engines
            if getattr(e, "owner", None) in (None, user)
        ]

    @app.get("/api/sync/status")
    def api_sync_status():
        engines = _visible_engines()
        if not engines:
            return _json({"backend": None, "data_version": None,
                          "data_root": str(state.request_root()), "remotes": []})
        remotes = []
        for engine in engines:
            item = dict(engine.status)
            item["data_version"] = engine.data_version
            item["local_dir"] = str(engine.local_dir)
            item["scope"] = "user" if getattr(engine, "owner", None) else "global"
            remotes.append(item)
        status = dict(remotes[0])
        status["remotes"] = remotes
        status["data_root"] = str(state.request_root())
        return _json(status)

    # ------------------------------------------------------------ 查看根
    @app.get("/api/data_root")
    def api_data_root():
        return _json({
            "data_root": str(state.request_root()),
            "startup_data_root": str(state.startup_data_root),
            "sync_local_dirs": [str(e.local_dir) for e in _visible_engines()],
        })

    @app.post("/api/data_root")
    def api_set_data_root():
        body = request.get_json(silent=True) or {}
        raw = str(body.get("path") or "").strip()
        if not raw:
            return _error(400, "need path")
        path = Path(raw).expanduser()
        if not path.is_absolute():
            return _error(400, "path must be absolute")
        try:
            path = path.resolve()
        except OSError as exc:
            return _error(400, f"cannot resolve path: {exc}")
        if not path.is_dir():
            return _error(400, f"not a directory: {path}")
        user = session.get("user")
        if user:
            # 登录用户：只切换自己 workspace 的查看根（互不影响）
            state.workspace(user).set_view_root(path, state.data_root)
            return _json({"ok": True, "data_root": str(state.request_root())})
        # 无会话兼容模式（无 user_store）：维持旧的进程全局切换
        state.set_data_root(path)
        return _json({"ok": True, "data_root": str(state.data_root)})

    def _ssh_engines() -> List[Any]:
        from .sync import SshBackend

        return [
            e
            for e in _visible_engines()
            if isinstance(getattr(e, "backend", None), SshBackend)
        ]

    def _find_ssh_engine(name: Optional[str]) -> Any:
        engines = _ssh_engines()
        if not name:
            return engines[0] if engines else None
        for engine in engines:
            if (engine.name or "") == name:
                return engine
        return None

    @app.get("/api/browse")
    def api_browse():
        """只读目录浏览（网页版 ls/tree，辅助选择查看根；只列子目录）。

        缺省浏览服务器本机文件系统；带 ``remote=<远端名>``（或 ``remote=1``
        取第一个 ssh 远端）时经一次性 SSH 浏览**远端（如 H-Server）**的目录。
        """
        remote_name = request.args.get("remote")
        if remote_name is not None:
            from .sync import SSH_BROWSE_SCRIPT, ssh_run_json

            engine = _find_ssh_engine(None if remote_name in ("", "1") else remote_name)
            if engine is None:
                return _error(400, "no ssh remote configured")
            backend = engine.backend
            raw = request.args.get("path") or backend.remote_dir
            if not raw.startswith("/"):
                return _error(400, "path must be absolute")
            try:
                result = ssh_run_json(
                    backend.target, backend.ssh_args,
                    SSH_BROWSE_SCRIPT, [raw], timeout=30,
                )
            except Exception as exc:
                return _error(502, f"remote browse failed: {exc}")
            if result.get("error"):
                return _error(400, f"remote: {result['error']}")
            dirs = [
                {
                    "name": e["name"],
                    "hidden": str(e["name"]).startswith("."),
                    "has_runs_dir": bool(e.get("has_runs")),
                }
                for e in result.get("entries") or []
            ]
            return _json(
                {
                    "path": result.get("path") or raw,
                    "parent": result.get("parent"),
                    "looks_like_data_root": bool(result.get("looks_like_data_root")),
                    "dirs": dirs,
                    "remote": engine.name or "remote",
                    "remote_target": backend.target,
                }
            )
        raw = request.args.get("path") or str(state.request_root())
        path = Path(raw).expanduser()
        if not path.is_absolute():
            return _error(400, "path must be absolute")
        try:
            path = path.resolve()
        except OSError as exc:
            return _error(400, f"cannot resolve path: {exc}")
        if not path.is_dir():
            return _error(400, f"not a directory: {path}")
        dirs = []
        try:
            # 只读文件树：仅列子目录名（含 runs/ 标记），不触碰任何文件内容
            for child in sorted(path.iterdir(), key=lambda p: p.name):
                try:
                    if not child.is_dir():
                        continue
                except OSError:
                    continue
                dirs.append({
                    "name": child.name,
                    "hidden": child.name.startswith("."),
                    "has_runs_dir": (child / "runs").is_dir(),
                })
        except OSError as exc:
            return _error(400, f"cannot list directory: {exc}")
        return _json({
            "path": str(path),
            "parent": str(path.parent) if path.parent != path else None,
            "looks_like_data_root": (path / "runs").is_dir(),
            "dirs": dirs,
        })

    @app.get("/api/known_roots")
    def api_known_roots():
        """后台扫描发现的监控数据目录（root-picker 快捷列表；含 remote_roots 远端项）。"""
        return _json(state.known_roots_scanner.snapshot())

    # ------------------------------------------------------------ 远端（H-Server）管理
    ADDED_REMOTES_CACHE = ".added_remotes.json"

    def _added_remotes_path() -> Path:
        return state.startup_data_root / ADDED_REMOTES_CACHE

    def _added_remote_names() -> set:
        data = read_json(_added_remotes_path(), default=None)
        names = set()
        if isinstance(data, dict):
            for item in data.get("remotes") or []:
                if isinstance(item, dict) and item.get("name"):
                    names.add(str(item["name"]))
        return names

    @app.get("/api/remotes")
    def api_remotes():
        """同步远端列表（root-picker 远端模式用）：全局 + 当前用户自加。"""
        added = _added_remote_names()
        remotes = []
        for engine in _visible_engines():
            backend = getattr(engine, "backend", None)
            owner = getattr(engine, "owner", None)
            remotes.append(
                {
                    "name": engine.name or "remote",
                    "backend": getattr(backend, "description", None),
                    "ssh": type(backend).__name__ == "SshBackend",
                    "remote_dir": getattr(backend, "remote_dir", None),
                    "scope": "user" if owner else "global",
                    "added_via_web": bool(owner) or (engine.name or "") in added,
                }
            )
        return _json({"remotes": remotes})

    @app.post("/api/remotes")
    def api_add_remote():
        """网页端把远端（H-Server）某个目录加为**当前用户**的同步远端。

        参考已有 ssh 远端复用其 target/ssh_args，只换路径；登录用户的新 engine
        拉取进其 workspace 的 data/ 目录并把其查看根切到那里（同步到位后 run
        自动出现），持久化在该用户 ``workspace.json``（serve 重启后自动恢复，
        各用户互不影响）。无会话兼容模式维持旧行为：拉进启动数据根并持久化
        ``.added_remotes.json``。
        """
        from .sync import SyncDaemon, SyncEngine, SshBackend

        body = request.get_json(silent=True) or {}
        path = str(body.get("path") or "").strip()
        if not path.startswith("/"):
            return _error(400, "path must be absolute (远端绝对路径)")
        path = path.rstrip("/")
        reference = _find_ssh_engine(body.get("remote") or None)
        if reference is None:
            return _error(400, "no ssh remote configured (先配一个 --remote 作为跳板)")
        ref_backend = reference.backend
        user = session.get("user")
        # 同用户范围内：已有 engine 同步同 target+path 时不重复添加
        for engine in _ssh_engines():
            backend = engine.backend
            if backend.target == ref_backend.target and backend.remote_dir == path:
                return _json(
                    {
                        "ok": True,
                        "name": engine.name or "remote",
                        "note": "该目录已在同步列表中，run 会随同步自动出现。",
                        "already": True,
                    }
                )
        # 名称：用户给或按路径末段派生；全局唯一
        existing = {e.name or "" for e in state.sync_engines} | _added_remote_names()
        base_name = re.sub(r"[^A-Za-z0-9._-]+", "-", str(body.get("name") or "")).strip("-")
        if not base_name:
            slug = re.sub(r"[^A-Za-z0-9._-]+", "-", path).strip("-")[-40:] or "root"
            base_name = f"web-{slug}"
        if user:
            base_name = f"{_safe_username(user)}--{base_name}"
        name, i = base_name, 2
        while name in existing:
            name = f"{base_name}-{i}"
            i += 1
        interval = float(body.get("interval") or 30.0)
        backend = SshBackend(ref_backend.target, path, ssh_args=ref_backend.ssh_args)
        if user:
            workspace = state.workspace(user)
            local_dir = workspace.data_dir
        else:
            workspace = None
            local_dir = state.startup_data_root
        engine = SyncEngine(local_dir, backend=backend, name=name)
        engine.owner = user  # None=全局引擎；网页添加的归当前用户
        daemon = SyncDaemon(engine, interval=interval)
        daemon.start()
        state.sync_engines.append(engine)
        if workspace is not None:
            workspace.add_remote(
                {
                    "name": name,
                    "owner": user,
                    "remote": f"{ref_backend.target}:{path}",
                    "ssh_args": ref_backend.ssh_args,
                    "interval": interval,
                    "added_at": time.time(),
                }
            )
            # 加完即可看到 run：把该用户的查看根切到其 workspace 数据目录
            workspace.set_view_root(workspace.data_dir, state.data_root)
        else:
            data = read_json(_added_remotes_path(), default=None)
            if not isinstance(data, dict):
                data = {}
            items = data.get("remotes")
            if not isinstance(items, list):
                items = []
            items.append(
                {
                    "name": name,
                    "remote": f"{ref_backend.target}:{path}",
                    "ssh_args": ref_backend.ssh_args,
                    "interval": interval,
                    "added_at": time.time(),
                }
            )
            data["remotes"] = items
            try:
                atomic_write_json(_added_remotes_path(), data)
            except OSError as exc:
                logger.warning("persist added remote failed (ignored): %s", exc)
        logger.info(
            "web added sync remote [%s]%s: %s",
            name, f" (user {user})" if user else "", backend.description,
        )
        note = "已加入同步：数据随周期拉取进本机镜像后，run 自动出现在列表。"
        if user:
            note = "已加入你的 workspace 同步，查看根已切换；数据拉取到位后 run 自动出现。"
        return _json(
            {
                "ok": True,
                "name": name,
                "backend": backend.description,
                "note": note,
            }
        )

    @app.get("/api/llm/help")
    def api_llm_help():
        return _json(llm_help_document())

    @app.get("/api/events")
    def api_events():
        q = state.subscribe()

        def gen():
            try:
                yield ": connected\n\n"
                while True:
                    try:
                        payload = q.get(timeout=15)
                        data = json.dumps(payload, ensure_ascii=False, default=str)
                        yield f"data: {data}\n\n"
                    except queue.Empty:
                        yield ": ping\n\n"
            finally:
                state.unsubscribe(q)

        return Response(
            gen(),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/dataset/status")
    def api_dataset_status():
        backend = state.dataset_backend
        if backend is None:
            return _json({"enabled": False,
                          "hint": "serve with --remote (ssh) or --dataset-local"})
        return _json(backend.status())

    @app.get("/api/dataset/<sub>")
    def api_dataset(sub: str):
        return _handle_dataset(state, sub)

    return app


# ---------------------------------------------------------------------------
# handlers（供路由调用）
# ---------------------------------------------------------------------------
def _handle_metrics(ctx: ViewContext, run_id: str) -> Response:
    stream = request.args.get("stream") or "metrics"
    if stream not in STREAM_NAMES:
        return _error(400, f"stream must be one of {STREAM_NAMES}")
    offset = request.args.get("offset")
    base = ctx.root / "runs" / run_id / stream
    extra: Dict[str, Any] = {}
    if stream == "system" and base.is_dir():
        # System samples are written per node (runs/<id>/system/<node>/).
        nodes = sorted(p.name for p in base.iterdir() if p.is_dir())
        extra["nodes"] = nodes
        if nodes:
            node = request.args.get("node")
            if node is None:
                node = nodes[0]
            if node not in nodes:
                return _error(404, f"unknown node: {node}")
            base = base / node
            extra["node"] = node
    reader = ChunkedJsonlReader(base)
    if request.args.get("tail"):
        try:
            tail_bytes = int(request.args.get("tail_bytes") or "65536")
        except ValueError:
            tail_bytes = 65536
        payload = reader.read_tail(max_bytes=tail_bytes)
    else:
        payload = reader.read_incremental(offset)
    payload.update(extra)
    return _json(payload)


def _handle_history(ctx: ViewContext, run_id: str) -> Response:
    """Full history of a run's metrics (or system) stream, decimated.

    Unlike the byte-incremental /metrics endpoint (tail + follow), this
    one scans the whole stream so the UI can show/compare the complete
    past of long runs. The result is cached in memory keyed by the stream
    end offset, so repeated loads are cheap and a running run invalidates
    automatically as it grows. Metrics records are resume-deduped by step
    (last write wins); system records keep arrival order.
    """
    try:
        max_points = int(request.args.get("max_points") or "3000")
    except ValueError:
        max_points = 3000
    max_points = max(100, min(20000, max_points))

    stream = request.args.get("stream") or "metrics"
    if stream not in ("metrics", "system"):
        return _error(400, "stream must be metrics or system")

    extra: Dict[str, Any] = {}
    if stream == "system":
        base = ctx.root / "runs" / run_id / "system"
        if not base.is_dir():
            return _error(404, f"unknown run: {run_id}")
        nodes = sorted(p.name for p in base.iterdir() if p.is_dir())
        extra["nodes"] = nodes
        if not nodes:
            return _json({"run_id": run_id, "stream": stream, "total_records": 0,
                          "returned": 0, "decimated": False, "records": [], **extra})
        node = request.args.get("node") or nodes[0]
        if node not in nodes:
            return _error(404, f"unknown node: {node}")
        base = base / node
        cache_key = f"{ctx.root}|{run_id}|system|{node}"
    else:
        base = ctx.root / "runs" / run_id / "metrics"
        if not base.is_dir():
            return _error(404, f"unknown run: {run_id}")
        cache_key = f"{ctx.root}|{run_id}|metrics"

    reader = ChunkedJsonlReader(base)
    end = reader.end_offset()
    cached = ctx.state.history_cache.get(cache_key)
    if cached is not None and cached[0] == end:
        return _json(cached[1])

    if stream == "metrics":
        # 全量扫描；同一 step 后写胜出（resume 重跑的语义，与前端 ingestSteps 一致）
        by_step: Dict[float, Dict[str, Any]] = {}
        for record in reader.iter_records():
            step = record.get("step")
            if isinstance(step, (int, float)):
                by_step[step] = record
        ordered = [by_step[s] for s in sorted(by_step)]
    else:
        ordered = [r for r in reader.iter_records() if isinstance(r.get("ts"), (int, float))]

    total = len(ordered)
    if total <= max_points:
        records = ordered
    else:
        stride = (total - 1) / (max_points - 1)
        picked = {min(total - 1, round(i * stride)) for i in range(max_points)}
        picked.add(total - 1)
        records = [ordered[i] for i in sorted(picked)]
    payload = {
        "run_id": run_id,
        "stream": stream,
        "total_records": total,
        "returned": len(records),
        "decimated": total > max_points,
        "records": records,
        **extra,
    }
    ctx.state.history_cache[cache_key] = (end, payload)
    return _json(payload)


def _handle_post_annotation(ctx: ViewContext, body: Dict[str, Any]) -> Response:
    ckpt = body.get("ckpt") or {}
    run_id = ckpt.get("run_id") or body.get("run_id")
    step = ckpt.get("step", body.get("step"))
    if not run_id or step is None:
        return _error(400, "need ckpt.run_id and ckpt.step")
    try:
        step = int(step)
    except (TypeError, ValueError):
        return _error(400, "ckpt.step must be an integer")
    try:
        doc = ctx.registry.put_annotation(
            run_id=str(run_id),
            step=step,
            benchmarks=body.get("benchmarks") or {},
            notes=str(body.get("notes", "")),
            updated_by=str(body.get("updated_by", "api")),
            extra=body.get("extra"),
        )
    except ValueError as exc:
        return _error(400, str(exc))
    ctx.state._broadcast({"type": "annotation", "key": doc["key"]})
    return _json({"ok": True, "annotation": doc})


def _run_ckpt_steps(ctx: ViewContext, run_id: str) -> List[int]:
    return sorted(
        {
            int(c["step"])
            for c in (ctx.registry.get_run(run_id) or {}).get("checkpoints") or []
            if isinstance(c, dict) and isinstance(c.get("step"), (int, float))
        }
    )


def _ckpt_segments(
    slices: List[Dict[str, Any]],
    ckpt_steps: List[int],
    run_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """按 ckpt 段落聚合每个数据源的切片数。

    段落语义：ckpt@S 覆盖 (prev_ckpt_step, S]，末尾追加 (last_ckpt, +inf)
    的“最新”段（ckpt_step=None）。run_id 仅谱系（lineage）模式下标注。
    """
    seg_counts: List[Dict[str, int]] = [dict() for _ in range(len(ckpt_steps) + 1)]
    seg_totals = [0] * (len(ckpt_steps) + 1)
    for item in slices:
        idx = bisect.bisect_left(ckpt_steps, item["step"])
        seg = seg_counts[idx]
        seg[item["source"]] = seg.get(item["source"], 0) + 1
        seg_totals[idx] += 1
    rows = []
    for i, counts in enumerate(seg_counts):
        if not seg_totals[i]:
            continue
        is_tail = i == len(ckpt_steps)
        row = {
            "ckpt_step": None if is_tail else ckpt_steps[i],
            "step_min": None if i == 0 else ckpt_steps[i - 1] + 1,
            "step_max": None if is_tail else ckpt_steps[i],
            "total": seg_totals[i],
            "by_source": {name: counts[name] for name in sorted(counts)},
        }
        if run_id is not None:
            row["run_id"] = run_id
        rows.append(row)
    return rows


def _scan_metrics_slices(ctx: ViewContext, run_id: str) -> Any:
    """旧格式切片扫描：metrics 流 step_record 里的 episode_ids/frame_indices。

    全扫聚合（同一 step 后写胜出），按流尾 offset + ckpt 段做内存缓存。
    返回 (slices, by_ckpt)，slices 按 step 倒序。
    """
    base = ctx.root / "runs" / run_id / "metrics"
    ckpt_steps = _run_ckpt_steps(ctx, run_id)
    reader = ChunkedJsonlReader(base)
    end = reader.end_offset()
    cache_key = f"{ctx.root}|{run_id}"
    cached = ctx.state.slices_cache.get(cache_key)
    if cached is not None and cached[0] == (end, tuple(ckpt_steps)):
        return cached[1]
    by_step: Dict[float, Dict[str, Any]] = {}
    for record in reader.iter_records():
        step = record.get("step")
        if not isinstance(step, (int, float)):
            continue
        if record.get("episode_ids"):
            by_step[step] = record
    slices = []
    for step in sorted(by_step):
        record = by_step[step]
        episode_ids = record.get("episode_ids") or []
        frame_indices = record.get("frame_indices") or []
        ratios = record.get("action_video_fps_ratios") or []
        source = str(
            record.get("mix_source_name")
            or record.get("source_dataset")
            or "unknown"
        )
        for i, episode_id in enumerate(episode_ids):
            frames = frame_indices[i] if i < len(frame_indices) else []
            ratio = ratios[i] if i < len(ratios) else None
            slices.append(
                {
                    "step": int(step),
                    "source": source,
                    "episode_id": str(episode_id),
                    "window_start": int(frames[0]) if frames else None,
                    "num_video_frames": (len(frames) - 1) if frames else None,
                    "ratio": int(ratio) if ratio else None,
                    # 训练时的真实视频索引（stride 布局为跳帧+末帧填充，
                    # 前端点击切片时原样回传 /api/dataset/selection）
                    "frame_indices": [int(f) for f in frames],
                    "sample_idx": i,
                }
            )
    by_ckpt = _ckpt_segments(slices, ckpt_steps)
    # 展示按 step 倒序（最新在前），统计不受影响
    slices.sort(key=lambda item: item["step"], reverse=True)
    ctx.state.slices_cache[cache_key] = ((end, tuple(ckpt_steps)), (slices, by_ckpt))
    return slices, by_ckpt


def _handle_slices(ctx: ViewContext, run_id: str) -> Response:
    """训练切片聚合入口。

    有 ``runs/<run_id>/slices/`` 目录（全 rank 切片记录，训练侧
    ``log_slice_record`` 写入）走 step→node→gpu 新结构；旧 run（切片只在
    rank0 metrics 流里）完全走原逻辑原响应结构（向后兼容）。
    """
    run_dir = ctx.root / "runs" / run_id
    if (run_dir / "slices").is_dir():
        return _handle_slices_ranked(ctx, run_id)
    return _handle_slices_legacy(ctx, run_id)


def _handle_slices_legacy(ctx: ViewContext, run_id: str) -> Response:
    """聚合 run 的训练切片（episode_id + 窗口起点），供点击后按需解码。

    切片元数据由训练侧写进 step_record（episode_ids / frame_indices /
    action_video_fps_ratios），随 metrics 流同步；本端点全扫聚合（同一
    step 后写胜出），按流尾 offset 做内存缓存。媒体不在此产生——前端点
    击切片后走 /api/dataset/selection 远端解码。
    """
    base = ctx.root / "runs" / run_id / "metrics"
    if not base.is_dir():
        return _error(404, f"unknown run: {run_id}")
    source_filter = _q1("source")
    offset = max(0, _qint("offset", 0))
    limit = max(1, min(2000, _qint("limit", 200)))
    step_min = _qint("step_min", None)
    step_max = _qint("step_max", None)

    slices, by_ckpt = _scan_metrics_slices(ctx, run_id)

    # step 范围过滤（ckpt 段 / 精确 step 由前端以 step_min/step_max 表达）
    ranged = [
        s
        for s in slices
        if (step_min is None or s["step"] >= step_min)
        and (step_max is None or s["step"] <= step_max)
    ]
    if _q1("steps_only") == "1":
        # 仅返回范围内的 step 列表（前端 ckpt 段的 step 选择器用）
        steps = sorted({s["step"] for s in ranged}, reverse=True)
        return _json(
            {"run_id": run_id, "steps": steps, "total_slices": len(ranged)}
        )

    # 统计与 top 复用随 step 范围联动（不受 source 过滤影响）
    source_counts: Dict[str, int] = {}
    source_episodes: Dict[str, set] = {}
    episode_counter: Counter = Counter()
    for s in ranged:
        source_counts[s["source"]] = source_counts.get(s["source"], 0) + 1
        source_episodes.setdefault(s["source"], set()).add(s["episode_id"])
        episode_counter[(s["source"], s["episode_id"])] += 1
    by_source = {
        name: {
            "count": source_counts[name],
            "unique_episodes": len(source_episodes.get(name, ())),
        }
        for name in sorted(source_counts)
    }
    top_episodes = [
        {"source": source, "episode_id": episode_id, "count": count}
        for (source, episode_id), count in episode_counter.most_common(20)
    ]

    filtered = (
        [s for s in ranged if s["source"] == source_filter]
        if source_filter
        else ranged
    )
    page = filtered[offset : offset + limit]
    return _json(
        {
            "run_id": run_id,
            "total_slices": len(filtered),
            "offset": offset,
            "limit": limit,
            "returned": len(page),
            "slices": page,
            "by_source": by_source,
            "by_ckpt": by_ckpt,
            "top_episodes": top_episodes,
            "source_filter": source_filter,
            "step_min": step_min,
            "step_max": step_max,
        }
    )


# ---------------------------------------------------------------------------
# 全 rank 训练切片（runs/<run_id>/slices/rank%04d.jsonl）
# ---------------------------------------------------------------------------
def _ranked_run_slices(ctx: ViewContext, run_id: str) -> Optional[List[Dict[str, Any]]]:
    """读取 slices/ 目录下全部 rank 文件，展开为带 node/gpu/rank 的切片列表。

    同一 (rank, step) 后写胜出（append-only，resume 重跑同 step 会重复写）。
    按各文件 (name, size) 元组做内存缓存；无 slices 目录返回 None。
    """
    slices_dir = ctx.root / "runs" / run_id / "slices"
    if not slices_dir.is_dir():
        return None
    files = []
    for path in sorted(slices_dir.glob("rank*.jsonl")):
        try:
            files.append((path, path.stat().st_size))
        except OSError:
            continue
    fingerprint = tuple((p.name, size) for p, size in files)
    cache_key = f"{ctx.root}|ranked:{run_id}"
    cached = ctx.state.slices_cache.get(cache_key)
    if cached is not None and cached[0] == fingerprint:
        return cached[1]
    by_rank_step: Dict[Any, Dict[str, Any]] = {}
    for path, _size in files:
        try:
            with path.open("r", encoding="utf-8") as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue
                    step = record.get("step")
                    rank = record.get("rank")
                    if not isinstance(step, (int, float)) or not isinstance(rank, (int, float)):
                        continue
                    by_rank_step[(int(rank), int(step))] = record
        except OSError:
            continue
    slices: List[Dict[str, Any]] = []
    for (rank, step) in sorted(by_rank_step):
        record = by_rank_step[(rank, step)]
        node = str(record.get("node") or "node?")
        gpu = record.get("gpu")
        for i, sample in enumerate(record.get("samples") or []):
            if not isinstance(sample, dict) or sample.get("episode_id") is None:
                continue
            frames = sample.get("frame_indices") or []
            ratio = sample.get("ratio")
            window_start = sample.get("window_start")
            if window_start is None and frames:
                window_start = frames[0]
            slices.append(
                {
                    "step": int(step),
                    "run_id": run_id,
                    "node": node,
                    "gpu": int(gpu) if isinstance(gpu, (int, float)) else None,
                    "rank": rank,
                    "source": str(sample.get("source") or "unknown"),
                    "episode_id": str(sample.get("episode_id")),
                    "window_start": int(window_start)
                    if isinstance(window_start, (int, float))
                    else None,
                    "num_video_frames": (len(frames) - 1) if frames else None,
                    "ratio": int(ratio) if isinstance(ratio, (int, float)) and ratio else None,
                    "frame_indices": [int(f) for f in frames],
                    "sample_idx": i,
                }
            )
    ctx.state.slices_cache[cache_key] = (fingerprint, slices)
    return slices


def _any_run_slices(ctx: ViewContext, run_id: str) -> List[Dict[str, Any]]:
    """run 的切片列表：优先全 rank slices/，旧 run 回退 metrics 流扫描。"""
    ranked = _ranked_run_slices(ctx, run_id)
    if ranked is not None:
        return ranked
    base = ctx.root / "runs" / run_id / "metrics"
    if base.is_dir():
        slices, _ = _scan_metrics_slices(ctx, run_id)
        return slices
    return []


def _lineage_run_ids(ctx: ViewContext, run_id: str, max_depth: int = 32) -> List[str]:
    """沿 run.json parent.run_id 链向上收集 [本 run, 父, 祖父, ...]（环保护+深度上限）。"""
    chain = [run_id]
    seen = {run_id}
    current = run_id
    while len(chain) < max_depth:
        doc = ctx.registry.get_run(current) or {}
        parent = doc.get("parent") or {}
        parent_run = parent.get("run_id")
        if not parent_run or not isinstance(parent_run, str) or parent_run in seen:
            break
        chain.append(parent_run)
        seen.add(parent_run)
        current = parent_run
    return chain


def _slice_window(item: Dict[str, Any]) -> Optional[Any]:
    """切片窗口区间 [frame_indices[0], frame_indices[-1]]；缺索引时由 window_start+num 回退。"""
    frames = item.get("frame_indices") or []
    if len(frames) >= 2:
        return (int(frames[0]), int(frames[-1]))
    start = item.get("window_start")
    num = item.get("num_video_frames")
    if isinstance(start, (int, float)) and isinstance(num, (int, float)) and num > 0:
        return (int(start), int(start) + int(num))
    return None


def _intervals_union_length(intervals: List[Any]) -> int:
    """[a, b] 区间并集长度（b - a 求和，非重叠部分）。"""
    spans = sorted((int(a), int(b)) for a, b in intervals if int(b) >= int(a))
    total = 0
    cur = None
    for a, b in spans:
        if cur is None:
            cur = [a, b]
        elif a <= cur[1]:
            cur[1] = max(cur[1], b)
        else:
            total += cur[1] - cur[0]
            cur = [a, b]
    if cur is not None:
        total += cur[1] - cur[0]
    return total


def _catalog_index(state: ServerState) -> Dict[str, Path]:
    """dataset 名 -> episode catalog jsonl 路径（供覆盖率统计读 video.count）。

    来源：dataset_backend 的 local 模式（--dataset-local，默认扫
    micro_data/catalogs）的 catalog_root + 显式 name:path；另加仓库内
    micro_data/catalogs 兜底（dev/冒烟）。ssh 远端 catalog 本端不可读，
    找不到的源由调用方记入 missing_catalogs。
    """
    paths: Dict[str, Path] = {}
    roots: List[Path] = []
    pairs: List[str] = []
    backend = state.dataset_backend
    if backend is not None and getattr(backend, "mode", None) == "local":
        if getattr(backend, "catalog_root", None):
            roots.append(Path(str(backend.catalog_root)))
        pairs = list(getattr(backend, "catalogs", None) or [])
    repo_catalogs = Path(__file__).resolve().parent.parent / "micro_data" / "catalogs"
    if repo_catalogs.is_dir():
        roots.append(repo_catalogs)
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.episode_catalog.jsonl")):
            stem = path.name[: -len(".episode_catalog.jsonl")]
            short = stem.split(".")[0]
            paths.setdefault(stem, path)
            if short:
                paths.setdefault(short, path)
    for pair in pairs:
        name, sep, path = str(pair).partition(":")
        if sep and name:
            paths[name] = Path(path)
    return paths


def _catalog_line_fps(line: Dict[str, Any]) -> Optional[float]:
    """catalog 行里的视频 fps（部分源有：episode meta.fps / video.meta.fps）。"""
    for container in (line.get("meta"), (line.get("video") or {}).get("meta")):
        if isinstance(container, dict):
            fps = container.get("fps")
            if isinstance(fps, (int, float)) and fps > 0:
                return float(fps)
    return None


def _load_catalog(state: ServerState, path: Path) -> Optional[Dict[str, Any]]:
    """读 catalog 全集：每 episode 的 video.count（总帧数）+ 源级 fps。按文件指纹缓存。"""
    try:
        fingerprint = (str(path), path.stat().st_size, int(path.stat().st_mtime))
    except OSError:
        return None
    cached = state.catalog_cache.get(fingerprint[0])
    if cached is not None and cached[0] == fingerprint:
        return cached[1]
    episodes = []
    fps = None
    try:
        with path.open("r", encoding="utf-8") as fp:
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
                    fps = _catalog_line_fps(record)
    except OSError:
        return None
    payload = {"episodes": episodes, "fps": fps}
    state.catalog_cache[fingerprint[0]] = (fingerprint, payload)
    return payload


def _configured_source_names(run_doc: Optional[Dict[str, Any]]) -> Optional[List[str]]:
    """run.json 配置快照里的数据源集合（dataset.sources[].name，缺省回退 .dataset）。"""
    config = (run_doc or {}).get("config") or {}
    dataset = config.get("dataset") or {}
    sources = dataset.get("sources")
    if not isinstance(sources, list) or not sources:
        return None
    names = []
    for entry in sources:
        if isinstance(entry, dict):
            name = entry.get("name") or entry.get("dataset")
            if name:
                names.append(str(name))
    return names or None


_COVERAGE_EPISODE_LIMIT = 500
# 覆盖率 catalog 缓存：命中 TTL 内直接用（catalog 基本不可变，重建最多延迟 1h 感知）；
# 拉取失败短 TTL 防止每个 slices 请求都打 helper。
_CATALOG_CACHE_TTL = 3600.0
_CATALOG_FAIL_TTL = 60.0


def _coverage_cache_path(state: ServerState, source: str) -> Path:
    # 覆盖率 catalog 与查看根无关（源 catalog 内容共享），统一缓存在启动目录
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", source)
    return state.startup_data_root / ".cache" / "dataset" / "coverage" / f"{safe}.json"


def _catalog_for_source(
    state: ServerState, source: str, local_paths: Dict[str, Path]
) -> Optional[Dict[str, Any]]:
    """单源 catalog（每 episode video.count + fps），供覆盖率统计。

    解析顺序：内存/磁盘缓存（TTL）→ dataset helper 的 catalog_index op
    （ssh 远端或本机 local 模式均可，覆盖集群上的生产 catalog）→ 服务器本机
    catalog 文件兜底（_catalog_index，含仓库 micro_data/catalogs）。
    """
    now = time.time()
    cache_key = f"coverage:{source}"
    cached = state.catalog_cache.get(cache_key)
    if cached is not None and cached[0] > now:
        return cached[1]

    disk_path = _coverage_cache_path(state, source)
    disk = read_json(disk_path, default=None)
    if isinstance(disk, dict) and float(disk.get("fetched_at") or 0) > now - _CATALOG_CACHE_TTL:
        episodes = disk.get("episodes")
        if isinstance(episodes, list):
            payload = {"episodes": episodes, "fps": disk.get("fps")}
            state.catalog_cache[cache_key] = (now + _CATALOG_CACHE_TTL, payload)
            return payload

    backend = state.dataset_backend
    if backend is not None:
        try:
            head, blob = backend.request(
                _timeout=120.0, op="catalog_index", dataset=source
            )
            body = gzip.decompress(blob) if head.get("compressed") == "gzip" else blob
            data = json.loads(body.decode("utf-8"))
            episodes = data.get("episodes")
            if isinstance(episodes, list):
                payload = {"episodes": episodes, "fps": data.get("fps")}
                try:
                    disk_path.parent.mkdir(parents=True, exist_ok=True)
                    atomic_write_json(
                        disk_path,
                        {
                            "fetched_at": now,
                            "catalog_path": head.get("catalog_path"),
                            "size": head.get("size"),
                            "mtime": head.get("mtime"),
                            "episodes": episodes,
                            "fps": data.get("fps"),
                        },
                    )
                except OSError:
                    pass
                state.catalog_cache[cache_key] = (now + _CATALOG_CACHE_TTL, payload)
                return payload
        except Exception as exc:
            logger.debug("coverage catalog via helper failed for %s: %s", source, exc)

    path = local_paths.get(source)
    payload = _load_catalog(state, path) if path else None
    state.catalog_cache[cache_key] = (
        now + (_CATALOG_CACHE_TTL if payload is not None else _CATALOG_FAIL_TTL),
        payload,
    )
    return payload


def _compute_slice_stats(
    ctx: ViewContext,
    run_id: str,
    slices: List[Dict[str, Any]],
    chain: List[str],
) -> Dict[str, Any]:
    """切片统计：每源帧数口径（累计/并集/重叠）+ 覆盖率（catalog 全集分母）。

    帧数口径（已定）：窗口区间=[frame_indices[0], frame_indices[-1]]；
    累计=区间求和（含重叠）、并集=非重叠、差=重叠；catalog 行有 fps 的源
    附秒。覆盖率分母=该 run 训练配置的数据源集合（run.json 配置快照），
    拿不到配置时退化为切片中出现过的源（denominator=observed）。
    """
    per_source: Dict[str, Dict[str, Any]] = {}
    for item in slices:
        entry = per_source.setdefault(
            item["source"],
            {"count": 0, "episodes": set(), "intervals": {}, "fed_frames": 0, "frame_sets": {}},
        )
        entry["count"] += 1
        entry["episodes"].add(item["episode_id"])
        fed = item.get("frame_indices") or []
        entry["fed_frames"] += len(fed)
        if fed:
            entry["frame_sets"].setdefault(item["episode_id"], set()).update(
                int(f) for f in fed
            )
        window = _slice_window(item)
        if window is not None:
            entry["intervals"].setdefault(item["episode_id"], []).append(window)

    catalogs = _catalog_index(ctx.state)

    def catalog_for(source: str) -> Optional[Dict[str, Any]]:
        return _catalog_for_source(ctx.state, source, catalogs)

    by_source: Dict[str, Any] = {}
    union_by_source_episode: Dict[str, Dict[str, int]] = {}
    for name in sorted(per_source):
        entry = per_source[name]
        episode_union: Dict[str, int] = {}
        frames_total = 0
        frames_union = 0
        for episode_id, intervals in entry["intervals"].items():
            frames_total += sum(b - a for a, b in intervals)
            union = _intervals_union_length(intervals)
            episode_union[episode_id] = union
            frames_union += union
        frames_fed = int(entry["fed_frames"])
        frames_unique = sum(len(s) for s in entry["frame_sets"].values())
        stats = {
            "count": entry["count"],
            "unique_episodes": len(entry["episodes"]),
            # 实际喂入口径（推荐展示）：喂入=切片实际帧数求和（含重复），
            # 去重=同集内不重复源帧数，重复=两者差
            "frames_fed": frames_fed,
            "frames_unique": frames_unique,
            "frames_repeat": frames_fed - frames_unique,
            # 时间轴跨度口径（tooltip）：窗口 [首帧,末帧] 区间的求和/并集/重叠
            "frames_total": frames_total,
            "frames_union": frames_union,
            "frames_overlap": frames_total - frames_union,
        }
        catalog = catalog_for(name)
        fps = (catalog or {}).get("fps")
        if fps:
            stats["fps"] = fps
            stats["seconds_fed"] = round(frames_fed / fps, 2)
            stats["seconds_unique"] = round(frames_unique / fps, 2)
            stats["seconds_total"] = round(frames_total / fps, 2)
            stats["seconds_union"] = round(frames_union / fps, 2)
            stats["seconds_overlap"] = round((frames_total - frames_union) / fps, 2)
        by_source[name] = stats
        union_by_source_episode[name] = episode_union

    # 覆盖率：分母=run 配置的数据源集合（谱系模式取链上各 run 配置的并集），
    # 经本地 catalog 机制读每源全部 episode 的 video.count。
    configured: List[str] = []
    for rid in chain:
        for name in _configured_source_names(ctx.registry.get_run(rid)) or []:
            if name not in configured:
                configured.append(name)
    denominator = "config" if configured else "observed"
    coverage_sources = configured or sorted(per_source)
    coverage: Dict[str, Any] = {"denominator": denominator, "sources": {}}
    missing: List[str] = []
    for name in coverage_sources:
        catalog = catalog_for(name)
        if catalog is None:
            missing.append(name)
            continue
        fed = union_by_source_episode.get(name, {})
        episodes_out = []
        episodes_fed = 0
        frames_covered = 0
        frames_total = 0
        fed_rows = []
        unfed_rows = []
        for episode in catalog["episodes"]:
            total_frames = episode["frames"] or 0
            frames_total += total_frames
            covered = min(fed.get(episode["episode_id"], 0), total_frames) if total_frames else 0
            pct = round(covered / total_frames * 100, 2) if total_frames else 0.0
            row = {
                "episode_id": episode["episode_id"],
                "covered_frames": covered,
                "total_frames": total_frames,
                "pct": pct,
            }
            if covered > 0:
                episodes_fed += 1
                frames_covered += covered
                fed_rows.append(row)
            else:
                unfed_rows.append(row)
        # 已喂的在前（覆盖率降序），未喂的按 catalog 顺序列后；超长截断
        fed_rows.sort(key=lambda r: r["pct"], reverse=True)
        episodes_out = fed_rows + unfed_rows
        truncated = len(episodes_out) > _COVERAGE_EPISODE_LIMIT
        episodes_out = episodes_out[:_COVERAGE_EPISODE_LIMIT]
        coverage["sources"][name] = {
            "episodes_fed": episodes_fed,
            "episodes_total": len(catalog["episodes"]),
            "frames_covered": frames_covered,
            "frames_total": frames_total,
            "coverage_pct": round(frames_covered / frames_total * 100, 2) if frames_total else 0.0,
            "episodes": episodes_out,
            "episodes_truncated": truncated,
        }
    if missing:
        coverage["missing_catalogs"] = missing
    extra_observed = [n for n in sorted(per_source) if n not in coverage_sources]
    if extra_observed:
        coverage["extra_observed"] = extra_observed

    stats: Dict[str, Any] = {
        "by_source": by_source,
        "lineage": {"enabled": len(chain) > 1, "runs": chain},
    }
    if coverage["sources"]:
        stats["coverage"] = coverage
    else:
        stats["coverage"] = None
        stats["coverage_note"] = (
            "无可用 episode catalog（dataset helper 与服务器本地均未找到对应源），"
            "覆盖率统计不可用"
        )
    return stats


def _handle_slices_ranked(ctx: ViewContext, run_id: str) -> Response:
    """全 rank 切片聚合：分页单位为 step，响应 step→node→gpu 树 + 统计。

    lineage=1 时沿 run.json parent 链聚合本 run + 全部祖先 run 的切片再算
    统计/覆盖率；by_ckpt 分段保持按 run 语义（每段标注 run_id）。
    """
    run_dir = ctx.root / "runs" / run_id
    if not run_dir.is_dir():
        return _error(404, f"unknown run: {run_id}")
    source_filter = _q1("source")
    offset = max(0, _qint("offset", 0))
    limit = max(1, min(2000, _qint("limit", 100)))
    step_min = _qint("step_min", None)
    step_max = _qint("step_max", None)
    lineage = _q1("lineage") == "1"

    chain = _lineage_run_ids(ctx, run_id) if lineage else [run_id]
    per_run: Dict[str, List[Dict[str, Any]]] = {}
    all_slices: List[Dict[str, Any]] = []
    for rid in chain:
        run_slices = _any_run_slices(ctx, rid)
        per_run[rid] = run_slices
        all_slices.extend(run_slices)

    # by_ckpt：按 run 分段（每段标注 run_id；本 run 语义与旧逻辑一致）
    by_ckpt: List[Dict[str, Any]] = []
    for rid in chain:
        by_ckpt.extend(
            _ckpt_segments(per_run[rid], _run_ckpt_steps(ctx, rid), run_id=rid)
        )

    stats = _compute_slice_stats(ctx, run_id, all_slices, chain)

    # step 范围过滤（ckpt 段 / 精确 step 由前端以 step_min/step_max 表达）
    ranged = [
        s
        for s in all_slices
        if (step_min is None or s["step"] >= step_min)
        and (step_max is None or s["step"] <= step_max)
    ]
    if _q1("steps_only") == "1":
        # 仅返回范围内的 step 列表（前端 ckpt 段的 step 选择器用）
        steps = sorted({s["step"] for s in ranged}, reverse=True)
        return _json(
            {"run_id": run_id, "ranked": True, "steps": steps, "total_slices": len(ranged)}
        )

    # 统计与 top 复用随 step 范围联动（不受 source 过滤影响；与旧逻辑一致）
    source_counts: Dict[str, int] = {}
    source_episodes: Dict[str, set] = {}
    episode_counter: Counter = Counter()
    for s in ranged:
        source_counts[s["source"]] = source_counts.get(s["source"], 0) + 1
        source_episodes.setdefault(s["source"], set()).add(s["episode_id"])
        episode_counter[(s["source"], s["episode_id"])] += 1
    by_source = {
        name: {
            "count": source_counts[name],
            "unique_episodes": len(source_episodes.get(name, ())),
        }
        for name in sorted(source_counts)
    }
    top_episodes = [
        {"source": source, "episode_id": episode_id, "count": count}
        for (source, episode_id), count in episode_counter.most_common(20)
    ]

    filtered = (
        [s for s in ranged if s["source"] == source_filter]
        if source_filter
        else ranged
    )
    # 按 step 聚合成 step→node→gpu 树（分页单位=step；step 内无匹配切片的
    # node/gpu 分支省略）
    steps_map: Dict[int, Dict[str, Any]] = {}
    for item in filtered:
        step_entry = steps_map.setdefault(item["step"], {"total": 0, "nodes": {}})
        node_map = step_entry["nodes"].setdefault(item["node"], {})
        gpu_key = str(item["gpu"]) if item["gpu"] is not None else "?"
        node_map.setdefault(gpu_key, []).append(item)
        step_entry["total"] += 1
    ordered_steps = sorted(steps_map, reverse=True)
    total_steps = len(ordered_steps)
    page_steps = ordered_steps[offset : offset + limit]
    steps_out = []
    for step in page_steps:
        entry = steps_map[step]
        for gpu_list in entry["nodes"].values():
            for slices_list in gpu_list.values():
                slices_list.sort(key=lambda s: (s.get("rank") or 0, s.get("sample_idx") or 0))
        steps_out.append({"step": step, "total": entry["total"], "nodes": entry["nodes"]})
    return _json(
        {
            "run_id": run_id,
            "ranked": True,
            "lineage": lineage,
            "total_steps": total_steps,
            "total_slices": len(filtered),
            "offset": offset,
            "limit": limit,
            "returned": len(steps_out),
            "steps": steps_out,
            "stats": stats,
            "by_source": by_source,
            "by_ckpt": by_ckpt,
            "top_episodes": top_episodes,
            "source_filter": source_filter,
            "step_min": step_min,
            "step_max": step_max,
        }
    )


def _handle_run_file(ctx: ViewContext, run_id: str) -> Response:
    """返回 run 数据目录内的文件字节（periodic eval 的首帧/对比视频/曲线图等）。

    安全约束与 /static/<name> 相同：resolve 后必须仍落在 runs/<run_id>/
    之内（relative_to 兜底 ".."/绝对路径/软链逃逸），否则一律 404。
    """
    rel = request.args.get("path")
    if not rel:
        return _error(400, "need path=")
    base = (ctx.root / "runs" / run_id).resolve()
    target = (base / rel).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        return _error(404, "not found")
    if not target.is_file():
        return _error(404, "not found")
    content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
    # send_file(conditional=True)：流式发送 + Range/206——eval 对比视频常几百 MB，
    # 整读进内存会让首字节延迟过长，Chrome 下载器易报“网络错误”；206 也是
    # 浏览器视频/下载管理器的正常路径。download_name 让浏览器“另存为/下载”
    # 用真实文件名（否则 URL 末段是 /file，保存下来全叫 "file"）。
    resp = send_file(target, mimetype=content_type, conditional=True,
                     download_name=target.name)
    resp.headers["Cache-Control"] = "no-store"
    return resp


def llm_help_document() -> Dict[str, Any]:
    """Machine-readable API description so an LLM agent can write benchmarks."""
    return {
        "name": "motus-monitor",
        "purpose": "Write/read benchmark metrics for training checkpoints.",
        "endpoints": [
            {
                "method": "GET",
                "path": "/api/tree",
                "description": "Run/checkpoint forest. Each checkpoint has a stable "
                "`key` (format `<run_id>__step_<N>`), its step, path, and whether "
                "annotations exist.",
            },
            {
                "method": "GET",
                "path": "/api/runs",
                "description": "List training runs with state, last_step, max_steps.",
            },
            {
                "method": "GET",
                "path": "/api/annotations?ckpt=<key>",
                "description": "Read the annotation document of one checkpoint.",
            },
            {
                "method": "POST",
                "path": "/api/annotations",
                "description": "Create or REPLACE the annotation of one checkpoint. "
                "The document is synced back to the cluster automatically. "
                "Public (HTTPS) deployments require a logged-in session.",
                "body_schema": {
                    "ckpt": {"run_id": "string (required)", "step": "integer (required)"},
                    "benchmarks": {
                        "<benchmark_name>": {
                            "<metric_name>": "number|string",
                            "note": "optional string",
                        }
                    },
                    "notes": "optional free-text string",
                    "updated_by": "string, e.g. 'llm:gpt-x' or a user name",
                },
                "example": {
                    "curl": (
                        "curl -X POST http://localhost:8377/api/annotations "
                        "-H 'Content-Type: application/json' -d '{"
                        '"ckpt": {"run_id": "<run_id>", "step": 1000}, '
                        '"benchmarks": {"simpler_env": {"success_rate": 0.42}}, '
                        '"updated_by": "llm"}\''
                    )
                },
            },
        ],
        "notes": [
            "Benchmarks are free-form: any benchmark name and metric keys are accepted.",
            "POST replaces the whole annotation for that checkpoint: merge "
            "client-side when adding one benchmark to an existing document.",
            "Checkpoint keys are listed by GET /api/tree; run ids by GET /api/runs.",
        ],
    }


# ---------------------------------------------------------------------------
# 运行入口
# ---------------------------------------------------------------------------
def make_wsgi_server(app: Flask, host: str, port: int, ssl_context: Any = None):
    """threaded werkzeug WSGI server（可放线程里 serve_forever）。"""
    import ssl

    import werkzeug.serving as _serving
    from werkzeug.serving import WSGIRequestHandler
    from werkzeug.serving import make_server as _make_server

    # HTTPS 下 SSE（/api/events 等流式端点）客户端断开时，先 BrokenPipe（本身
    # 会被静默），随后 werkzeug execute() 的 finally 排空读对死掉的 TLS 连接抛
    # ssl.SSLError: [SSL: UNEXPECTED_EOF_WHILE_READING]——OpenSSL 3 下对端无
    # close_notify 时抛的是普通 SSLError，不在 werkzeug 默认白名单（只含
    # ssl.SSLEOFError）里，于是每次断连刷一条 "Error on request" 完整 traceback。
    # 连接本来已死，按掉线静默处理即可。该名单是 werkzeug.serving 的模块级变量
    # （run_wsgi 运行时读取），handler 子类无法覆盖，只能就地扩充。
    _dropped = getattr(_serving, "connection_dropped_errors", None)
    if _dropped is not None and ssl.SSLError not in _dropped:
        _serving.connection_dropped_errors = _dropped + (ssl.SSLError,)

    class _QuietRequestHandler(WSGIRequestHandler):
        """请求行访问日志降到 DEBUG：werkzeug 默认每请求一条 INFO，
        前端轮询会把终端刷屏（旧版 http.server 的 log_message 同样是 DEBUG）。"""

        def log(self, type: str, message: str, *args: Any) -> None:
            super().log("debug" if type == "info" else type, message, *args)

    return _make_server(host, int(port), app, threaded=True, ssl_context=ssl_context,
                        request_handler=_QuietRequestHandler)


def make_server(
    data_root: object,
    port: int = 8377,
    host: str = "127.0.0.1",
    sync_engine: Any = None,
    dataset_backend: Any = None,
):
    """兼容入口（冒烟测试等）：构建本机模式 app 并包装成 werkzeug server。

    返回对象的 ``server_address`` / ``serve_forever()`` / ``shutdown()`` /
    ``server_close()`` 与旧 stdlib ThreadingHTTPServer 用法一致。
    """
    app = create_app(
        data_root, sync_engine=sync_engine, dataset_backend=dataset_backend
    )
    return make_wsgi_server(app, host, port)


def serve_forever(
    data_root: object,
    port: int = 8377,
    host: str = "127.0.0.1",
    sync_engine: Any = None,
    dataset_backend: Any = None,
) -> None:
    """本机/开发模式入口（与历史行为一致：HTTP、无认证、单应用）。

    退出（含 SIGTERM）时确保 dataset helper 被回收。helper 的兜底链：
    close() 关 stdin → helper EOF 自退；server 被 kill -9 时
    PR_SET_PDEATHSIG 杀 ssh → 远端 EOF 自退（见 dataset_preview.py）。
    """
    import signal

    app = create_app(
        data_root, sync_engine=sync_engine, dataset_backend=dataset_backend
    )
    httpd = make_wsgi_server(app, host, port)
    logger.info("monitor web server: http://%s:%d (data: %s)", host, port, data_root)

    def _on_sigterm(_signum: int, _frame: Any) -> None:
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    previous_sigterm = signal.signal(signal.SIGTERM, _on_sigterm)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        signal.signal(signal.SIGTERM, previous_sigterm)
        if dataset_backend is not None:
            dataset_backend.close()


def run_servers(
    servers: List[Dict[str, Any]],
    dataset_backend: Any = None,
    daemons: Optional[List[Any]] = None,
) -> None:
    """多应用入口（public HTTPS 主站 + 本机管理后台）。

    ``servers`` 项：{"name", "app", "host", "port", "ssl_context"}。
    每个应用一个 werkzeug 线程；主线程等待 Ctrl-C / SIGTERM 后统一关停，
    并确保 dataset helper 被回收。
    """
    import signal

    httpds = []
    for spec in servers:
        httpd = make_wsgi_server(
            spec["app"], spec["host"], spec["port"], ssl_context=spec.get("ssl_context")
        )
        thread = threading.Thread(
            target=httpd.serve_forever,
            name=f"motus-monitor-{spec['name']}",
            daemon=True,
        )
        thread.start()
        httpds.append(httpd)
        scheme = "https" if spec.get("ssl_context") else "http"
        logger.info(
            "monitor %s server: %s://%s:%d", spec["name"], scheme, spec["host"], spec["port"]
        )

    stop = threading.Event()

    def _on_sigterm(_signum: int, _frame: Any) -> None:
        stop.set()

    previous_sigterm = signal.signal(signal.SIGTERM, _on_sigterm)
    try:
        while not stop.wait(1.0):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        for httpd in httpds:
            httpd.shutdown()
        for httpd in httpds:
            httpd.server_close()
        if dataset_backend is not None:
            dataset_backend.close()
        for daemon in daemons or []:
            daemon.stop()
