"""本机 dataset 预览后端：托管预览 helper 子进程池的完整生命周期。

``remote_preview_helper.py`` 是集群侧的按需解码服务（数据在集群存储上）。
本模块负责：

- **传输**：ssh 模式启动时把 helper 源码 ``cat`` 到集群
  ``/tmp/motus_preview_helper_<sha1>.py``（按内容哈希命名，每次启动必传，
  版本天然一致；不依赖集群仓库检出里有这份代码）。同时把本地
  ``libs/motus-data-factory/motus_data`` 包打包上传到
  ``/tmp/motus_preview_motus_data_<sha1>/``，helper 的 ``sys.path`` 优先该
  副本——motus_data 版本也与本地严格一致，不再依赖集群 submodule 手动同步。
- **进程池**：1 个 control 进程（list/info/action 等轻请求）+ ``workers``
  个 heavy 进程（frames/mp4/selection 重渲染），各自独立 stdin/stdout
  帧协议与请求锁——重解码不再阻塞浏览等其他操作。
- **进度透传**：helper 在解码分块边界写 interim 进度行，``request`` 的
  ``progress`` 回调逐条上报（server 端转成 HTTP 流式进度）。
- **自动终止（三层）**：
    1. 正常关闭：``close()`` 关 stdin → helper 读到 EOF 自退；仍存活则按
       握手拿到的远端 pid ``ssh kill``，最后 terminate 本地 ssh。
    2. 本机 server 异常死亡（含 kill -9）：Popen ``preexec_fn`` 设置
       ``PR_SET_PDEATHSIG``，内核立刻杀掉 ssh 客户端 → TCP 断 → 远端
       sshd 关通道 → helper stdin EOF → 自退。
    3. helper 侧 ``--idle-timeout``（默认 8h）无请求自退，最终兜底。
- **缓存**：重渲染请求（frames/mp4/selection）按请求哈希落到
  ``<data_root>/.cache/dataset/``（不在 sync 子目录内，不会被同步）。

``local`` 模式（开发/冒烟）不起 ssh，直接以本机 python 跑 helper，
其余协议与缓存完全相同。
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import select
import shlex
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

HELPER_PATH = Path(__file__).resolve().parent / "remote_preview_helper.py"
REPO_ROOT = Path(__file__).resolve().parent.parent
TMP_PREFIX = "/tmp/motus_preview_helper_"
MOTUS_DATA_TMP_PREFIX = "/tmp/motus_preview_motus_data_"
MOTUS_DATA_PKG = REPO_ROOT / "libs" / "motus-data-factory" / "motus_data"
PROTOCOL = "motus_preview_v1"

# 看门狗：周期性 ping control worker，失败即整池重建（指数退避）
SUPERVISOR_INTERVAL = 20.0
SUPERVISOR_MAX_BACKOFF = 120.0
SUPERVISOR_PING_TIMEOUT = 60.0  # 首次 ping 含 torch 等重库懒加载，给足超时

DEFAULT_REMOTE_PYTHON = "/home/fengodchen/miniconda3/envs/motus311/bin/python"
DEFAULT_REMOTE_REPO = "/code1/llm_team/chen.liao/ThinkerWM/thinkerwm-pretrain"
DEFAULT_CATALOG_ROOT = "/data/jiyuan.wang/EpisodeCatalogs"

# 这些 op 渲染成本高：走 heavy 进程 + 磁盘缓存；其余走 control 进程
CACHED_OPS = {"video_frames", "video_mp4", "selection_preview"}
LIGHT_OPS = {"list_datasets", "list_episodes", "episode_tree", "episode_info", "action_state", "catalog_index"}


class DatasetPreviewError(RuntimeError):
    """Raised when the preview helper cannot serve a request."""


def _set_pdeathsig() -> None:
    """Popen preexec_fn：父进程死亡时内核直接 SIGKILL 本子进程（Linux）。"""
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, 9)  # PR_SET_PDEATHSIG=1, SIGKILL=9
    except Exception:
        pass


def ssh_target_from_remote(remote: str) -> Optional[str]:
    """从 --remote 规格里提取 ssh target（与 sync.parse_remote 同规则）。"""
    remote = (remote or "").strip()
    if not remote or remote.startswith("local://"):
        return None
    if remote.startswith("ssh://"):
        return remote[len("ssh://"):].partition("/")[0] or None
    if ":" in remote:
        return remote.partition(":")[0] or None
    return None  # bare path = 本地目录


class _FramedReader:
    """在 pipe fd 上做带超时的行/定长读取（自有缓冲，不混用 BufferedReader）。"""

    def __init__(self, fd: int):
        self.fd = fd
        self.buf = bytearray()

    def _fill(self, timeout: float) -> None:
        ready, _, _ = select.select([self.fd], [], [], timeout)
        if not ready:
            raise TimeoutError(f"helper response timed out after {timeout}s")
        chunk = os.read(self.fd, 1 << 20)
        if not chunk:
            raise EOFError("helper stdout closed")
        self.buf += chunk

    def readline(self, timeout: float) -> bytes:
        while True:
            pos = self.buf.find(b"\n")
            if pos >= 0:
                line = bytes(self.buf[:pos])
                del self.buf[: pos + 1]
                return line
            self._fill(timeout)

    def read_exact(self, size: int, timeout: float) -> bytes:
        while len(self.buf) < size:
            self._fill(timeout)
        out = bytes(self.buf[:size])
        del self.buf[:size]
        return out


class DatasetPreviewBackend:
    """管理 helper 进程池（ssh 远端或本机），按 op 轻重分流请求。"""

    def __init__(
        self,
        *,
        mode: str,  # "ssh" | "local"
        target: Optional[str] = None,
        ssh_args: str = "",
        remote_python: str = DEFAULT_REMOTE_PYTHON,
        remote_repo: str = DEFAULT_REMOTE_REPO,
        catalog_root: Optional[str] = DEFAULT_CATALOG_ROOT,
        catalogs: Optional[list] = None,  # ["name:path", ...]（local 模式常用）
        workers: int = 2,  # heavy 渲染进程数（另加 1 个 control 进程）
        cache_dir: Optional[object] = None,
        request_timeout: float = 300.0,
        idle_timeout: float = 8 * 3600,
    ):
        if mode not in ("ssh", "local"):
            raise ValueError(f"mode must be ssh|local, got {mode!r}")
        if mode == "ssh" and not target:
            raise ValueError("ssh mode requires target")
        self.mode = mode
        self.target = target
        self.ssh_args = ssh_args or ""
        self.remote_python = remote_python
        self.remote_repo = remote_repo
        self.catalog_root = catalog_root
        self.catalogs = list(catalogs or [])
        self.workers = max(1, int(workers))
        self.cache_dir = Path(str(cache_dir)) if cache_dir else None
        self.request_timeout = float(request_timeout)
        self.idle_timeout = float(idle_timeout)

        self._pool: list = []  # [control, heavy0, heavy1, ...]
        self._tmp_path: Optional[str] = None
        self._motus_data_dir: Optional[str] = None  # 集群上上传的 motus_data 副本目录
        self._lifecycle_lock = threading.Lock()
        self._dispatch_lock = threading.Lock()
        self._rr = 0
        self._started = False
        self._closed = False
        self._error: Optional[str] = None
        # 看门狗状态（供 /api/dataset/status 与管理后台展示）
        self._sup_stop = threading.Event()
        self._sup_thread: Optional[threading.Thread] = None
        self._sup: Dict[str, Any] = {
            "last_ok_ts": None,
            "last_error": None,
            "restarts": 0,
        }

    # ------------------------------------------------------------------ ssh
    def _ssh_base(self) -> list:
        return (
            [
                "ssh",
                "-o", "BatchMode=yes",
                "-o", "ConnectTimeout=10",
                "-o", "Compression=yes",
            ]
            + shlex.split(self.ssh_args)
            + [str(self.target)]
        )

    def _ssh_run(self, remote_cmd: str, *, input_bytes: bytes = b"", timeout: float = 60) -> bytes:
        proc = subprocess.run(
            self._ssh_base() + [remote_cmd],
            input=input_bytes,
            capture_output=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace")[-400:]
            raise DatasetPreviewError(f"ssh exited {proc.returncode}: {stderr.strip()}")
        return proc.stdout

    def _sweep_stale_helpers(self) -> None:
        """清掉上次异常遗留的自己的 helper（正则形式不会匹配到 pkill 自身）。"""
        try:
            self._ssh_run(
                "pkill -f 'motus_preview_helper_[a-z0-9]*[.]py --serve' 2>/dev/null; true",
                timeout=30,
            )
        except Exception as exc:
            logger.debug("stale helper sweep failed (ignored): %s", exc)

    def _transfer_locked(self) -> str:
        source = HELPER_PATH.read_bytes()
        digest = hashlib.sha1(source).hexdigest()[:12]
        tmp = f"{TMP_PREFIX}{digest}.py"
        out = self._ssh_run(
            f"cat > {tmp} && echo TRANSFER_OK && wc -c < {tmp}",
            input_bytes=source,
            timeout=60,
        )
        if b"TRANSFER_OK" not in out or str(len(source)).encode() not in out:
            raise DatasetPreviewError(f"helper transfer failed: {out[-200:]!r}")
        return tmp

    # -------------------------------------------------------- motus_data 传输
    def _motus_data_files(self) -> list:
        """本地 motus_data 包内需上传的文件（排除 __pycache__/字节码）。"""
        pkg = MOTUS_DATA_PKG
        if not pkg.is_dir():
            raise DatasetPreviewError(f"local motus_data package missing: {pkg}")
        files = []
        for path in sorted(pkg.rglob("*")):
            if not path.is_file():
                continue
            if "__pycache__" in path.parts or path.suffix in (".pyc", ".pyo"):
                continue
            files.append(path)
        return files

    def _motus_data_digest(self) -> str:
        """motus_data 包内容哈希（文件名 + 字节），保证改库即换目录。"""
        digest = hashlib.sha1()
        for path in self._motus_data_files():
            digest.update(str(path.relative_to(MOTUS_DATA_PKG)).encode("utf-8"))
            digest.update(path.read_bytes())
        return digest.hexdigest()[:12]

    def _motus_data_tar_bytes(self) -> bytes:
        """打包本地 motus_data 包为 tar.gz（结构 <tar>/motus_data/...）。"""
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for path in self._motus_data_files():
                tar.add(
                    path,
                    arcname=f"{MOTUS_DATA_PKG.name}/{path.relative_to(MOTUS_DATA_PKG)}",
                )
        return buf.getvalue()

    def _transfer_motus_data_locked(self) -> Optional[str]:
        """把本地 motus_data 包按内容哈希上传到集群 /tmp，helper 优先 import 该副本。

        幂等：远端已有同哈希副本（校验 motus_data/__init__.py 存在）则跳过。
        本地缺包时返回 None 回退到集群仓库检出（旧行为），不阻塞预览。
        """
        if not MOTUS_DATA_PKG.is_dir():
            logger.warning(
                "local motus_data package missing at %s; helper will use cluster "
                "repo checkout",
                MOTUS_DATA_PKG,
            )
            return None
        digest = self._motus_data_digest()
        remote_dir = f"{MOTUS_DATA_TMP_PREFIX}{digest}"
        probe = self._ssh_run(
            f"test -f {remote_dir}/motus_data/__init__.py && echo READY || echo MISSING",
            timeout=30,
        )
        if b"READY" in probe:
            return remote_dir
        out = self._ssh_run(
            f"mkdir -p {remote_dir} && tar xzf - -C {remote_dir} && "
            f"test -f {remote_dir}/motus_data/__init__.py && echo UPLOAD_OK",
            input_bytes=self._motus_data_tar_bytes(),
            timeout=180,
        )
        if b"UPLOAD_OK" not in out:
            raise DatasetPreviewError(f"motus_data transfer failed: {out[-200:]!r}")
        return remote_dir

    # -------------------------------------------------------------- 进程管理
    def _spawn_locked(self, kind: str) -> dict:
        """拉起一个 helper 进程并完成握手（调用方持 _lifecycle_lock）。"""
        if self.mode == "local":
            argv = [
                sys.executable,
                "-u",
                str(HELPER_PATH),
                "--serve",
                "--repo-root",
                str(REPO_ROOT),
                "--idle-timeout",
                str(self.idle_timeout),
            ]
            if self.catalog_root:
                argv += ["--catalog-root", str(self.catalog_root)]
            for pair in self.catalogs:
                argv += ["--catalog", str(pair)]
            self._tmp_path = str(HELPER_PATH)
            proc = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                preexec_fn=_set_pdeathsig,
            )
        else:
            remote_argv = [
                self.remote_python,
                "-u",
                self._tmp_path,
                "--serve",
                "--repo-root",
                self.remote_repo,
                "--idle-timeout",
                str(self.idle_timeout),
            ]
            if self.catalog_root:
                remote_argv += ["--catalog-root", self.catalog_root]
            for pair in self.catalogs:
                remote_argv += ["--catalog", str(pair)]
            if self._motus_data_dir:
                remote_argv += ["--motus-data-dir", self._motus_data_dir]
            remote_cmd = (
                " ".join(shlex.quote(arg) for arg in remote_argv)
                + f" 2>>{self._tmp_path}.{kind}.log"
            )
            proc = subprocess.Popen(
                self._ssh_base() + [remote_cmd],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                preexec_fn=_set_pdeathsig,
            )
        worker = {
            "kind": kind,
            "proc": proc,
            "reader": _FramedReader(proc.stdout.fileno()),
            "pid": None,
            "lock": threading.Lock(),
        }
        try:
            line = worker["reader"].readline(timeout=60)
            hello = json.loads(line.decode("utf-8"))
        except Exception as exc:
            self._teardown_worker(worker)
            raise DatasetPreviewError(
                f"helper handshake failed: {exc}"
                + ("" if self.mode == "local" else f" (see {self._tmp_path}.{kind}.log on remote)")
            ) from exc
        if hello.get("hello") != PROTOCOL:
            self._teardown_worker(worker)
            raise DatasetPreviewError(f"unexpected helper handshake: {hello!r}")
        worker["pid"] = int(hello.get("pid") or 0) or None
        return worker

    def _launch_locked(self) -> None:
        if self.mode == "ssh":
            self._sweep_stale_helpers()
            self._tmp_path = self._transfer_locked()
            self._motus_data_dir = self._transfer_motus_data_locked()
        self._pool = [self._spawn_locked("control")]
        try:
            for i in range(self.workers):
                self._pool.append(self._spawn_locked(f"heavy{i}"))
        except Exception:
            for worker in self._pool:
                self._teardown_worker(worker)
            self._pool = []
            raise
        self._error = None
        logger.info(
            "dataset preview helper pool up: mode=%s procs=%s path=%s",
            self.mode,
            [(w["kind"], w["pid"]) for w in self._pool],
            self._tmp_path,
        )

    def start(self) -> None:
        """幂等启动；失败只记录 error（server 照常提供其余功能）。"""
        with self._lifecycle_lock:
            if self._started:
                return
            self._started = True
            if self._closed:
                return
            try:
                self._launch_locked()
            except Exception as exc:
                self._error = f"{type(exc).__name__}: {exc}"
                logger.warning("dataset preview backend unavailable: %s", self._error)
        self._ensure_supervisor()

    # -------------------------------------------------------------- 看门狗
    def _ensure_supervisor(self) -> None:
        if self._sup_thread is not None:
            return
        self._sup_thread = threading.Thread(
            target=self._supervisor_loop,
            name="motus-preview-supervisor",
            daemon=True,
        )
        self._sup_thread.start()

    def _ping_once(self, timeout: float = SUPERVISOR_PING_TIMEOUT) -> None:
        """对 control worker 发轻量请求探活；任何失败都抛 DatasetPreviewError。"""
        with self._lifecycle_lock:
            pool = list(self._pool)
        if not pool:
            raise DatasetPreviewError(self._error or "helper pool is empty")
        worker = pool[0]
        proc = worker.get("proc")
        if proc is None or proc.poll() is not None:
            raise DatasetPreviewError("control helper process is not running")
        with worker["lock"]:
            self._roundtrip(worker, {"op": "list_datasets"}, None, timeout)

    def _supervisor_loop(self, interval: float = SUPERVISOR_INTERVAL) -> None:
        backoff = interval
        while not self._sup_stop.wait(backoff):
            if self._closed or not self._started:
                continue
            try:
                self._ping_once()
            except DatasetPreviewError as exc:
                self._sup["last_error"] = str(exc)
                logger.warning("dataset helper ping failed (%s); restarting pool", exc)
                try:
                    self.restart()
                except DatasetPreviewError:
                    backoff = min(SUPERVISOR_MAX_BACKOFF, max(interval, backoff * 2))
                else:
                    backoff = interval
            except Exception as exc:  # 兜底：看门狗自身不能死
                self._sup["last_error"] = f"{type(exc).__name__}: {exc}"
                backoff = min(SUPERVISOR_MAX_BACKOFF, max(interval, backoff * 2))
            else:
                self._sup["last_ok_ts"] = time.time()
                self._sup["last_error"] = None
                backoff = interval

    def restart(self) -> None:
        """拆掉并重建整个 helper 池（ssh 模式含 helper/motus_data 重传与 stale 清理）。

        供看门狗与管理后台手动重启使用；幂等、与请求路径并发安全。
        """
        with self._lifecycle_lock:
            if self._closed:
                return
            try:
                for worker in self._pool:
                    self._teardown_worker(worker)
                self._pool = []
                if self.mode == "ssh":
                    self._sweep_stale_helpers()
                    self._tmp_path = self._transfer_locked()
                    self._motus_data_dir = self._transfer_motus_data_locked()
                self._pool = [self._spawn_locked("control")]
                for i in range(self.workers):
                    self._pool.append(self._spawn_locked(f"heavy{i}"))
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                self._sup["last_error"] = message
                self._error = message
                raise DatasetPreviewError(message) from exc
            self._error = None
            self._sup["restarts"] += 1
            self._sup["last_ok_ts"] = time.time()
            self._sup["last_error"] = None
            logger.info("dataset preview helper pool restarted (total %d)", self._sup["restarts"])

    def _teardown_worker(self, worker: dict) -> None:
        proc = worker.get("proc")
        worker["proc"] = None
        worker["reader"] = None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()  # EOF → helper 自退
        except OSError:
            pass
        try:
            proc.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            pass
        if self.mode == "ssh" and worker.get("pid"):
            try:
                self._ssh_run(f"kill {worker['pid']} 2>/dev/null; true", timeout=15)
            except Exception:
                pass
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def close(self) -> None:
        """幂等关闭（atexit / server finally / 信号路径都会调）。"""
        self._sup_stop.set()
        with self._lifecycle_lock:
            self._closed = True
            pool, self._pool = self._pool, []
            for worker in pool:
                self._teardown_worker(worker)

    # ------------------------------------------------------------------ 请求
    def _pick_worker(self, op: str) -> dict:
        if not self._pool:
            raise DatasetPreviewError(self._error or "helper is not running")
        if op in LIGHT_OPS or len(self._pool) == 1:
            return self._pool[0]
        with self._dispatch_lock:
            self._rr = (self._rr + 1) % (len(self._pool) - 1)
            return self._pool[1 + self._rr]

    def _roundtrip(
        self,
        worker: dict,
        req: Dict[str, Any],
        progress: Optional[Callable],
        timeout: float,
    ) -> Tuple[Dict[str, Any], bytes]:
        proc, reader = worker["proc"], worker["reader"]
        line = json.dumps(req, ensure_ascii=False, default=str).encode("utf-8") + b"\n"
        proc.stdin.write(line)
        proc.stdin.flush()
        while True:
            head = json.loads(reader.readline(timeout).decode("utf-8"))
            if "ok" in head:
                break
            # interim 进度行：有回调则透传，无回调直接丢弃，继续等最终响应
            if "progress" in head:
                if progress:
                    try:
                        progress(float(head["progress"]), str(head.get("stage", "decode")))
                    except Exception:
                        pass
                continue
            raise DatasetPreviewError(f"bad helper response: {head!r}")
        if not head.get("ok"):
            raise DatasetPreviewError(str(head.get("error", "unknown helper error")))
        payload = json.loads(
            reader.read_exact(int(head["json_bytes"]), timeout).decode("utf-8")
        )
        blob = reader.read_exact(int(head["blob_bytes"]), timeout)
        return payload, blob

    def _request(
        self,
        req: Dict[str, Any],
        progress: Optional[Callable],
        timeout: float,
    ) -> Tuple[Dict[str, Any], bytes]:
        if self._closed:
            raise DatasetPreviewError("backend is closed")
        if not self._started:
            self.start()
        worker = self._pick_worker(str(req.get("op")))
        with worker["lock"]:
            for attempt in (0, 1):
                try:
                    return self._roundtrip(worker, req, progress, timeout)
                except (BrokenPipeError, EOFError, OSError) as exc:
                    if self._closed:
                        raise DatasetPreviewError("backend is closed") from exc
                    if attempt == 1:
                        raise DatasetPreviewError(f"helper request failed twice: {exc}") from exc
                    logger.warning(
                        "dataset helper %s died (%s); relaunching", worker["kind"], exc
                    )
                    with self._lifecycle_lock:
                        if not self._closed:
                            self._teardown_worker(worker)
                            worker.update(self._spawn_locked(worker["kind"]))
        raise DatasetPreviewError("unreachable")

    def request(
        self,
        progress: Optional[Callable] = None,
        _timeout: Optional[float] = None,
        **req: Any,
    ) -> Tuple[Dict[str, Any], bytes]:
        return self._request(
            req, progress, float(_timeout) if _timeout else self.request_timeout
        )

    def cached_request(
        self,
        progress: Optional[Callable] = None,
        _timeout: Optional[float] = None,
        **req: Any,
    ) -> Tuple[Dict[str, Any], bytes]:
        """重渲染 op 走磁盘缓存；其余等同 request()。"""
        if self.cache_dir is None or req.get("op") not in CACHED_OPS:
            return self.request(progress=progress, _timeout=_timeout, **req)
        key = hashlib.sha1(
            json.dumps(req, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        path = self.cache_dir / f"{key}.bin"
        try:
            raw = path.read_bytes()
            json_len = int.from_bytes(raw[:4], "big")
            payload = json.loads(raw[4 : 4 + json_len].decode("utf-8"))
            return payload, raw[4 + json_len :]
        except (OSError, ValueError, KeyError):
            pass
        payload, blob = self.request(progress=progress, _timeout=_timeout, **req)
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        raw = len(body).to_bytes(4, "big") + body + blob
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
            tmp.write_bytes(raw)
            os.replace(tmp, path)
        except OSError as exc:
            logger.debug("dataset cache write failed (ignored): %s", exc)
        return payload, blob

    # ------------------------------------------------------------------ 状态
    def status(self) -> Dict[str, Any]:
        with self._lifecycle_lock:
            workers = [
                {
                    "kind": w["kind"],
                    "pid": w.get("pid"),
                    "running": bool(w.get("proc")) and w["proc"].poll() is None,
                }
                for w in self._pool
            ]
            return {
                "enabled": True,
                "mode": self.mode,
                "target": self.target,
                "running": any(w["running"] for w in workers),
                "workers": workers,
                "helper_pid": workers[0]["pid"] if workers else None,
                "helper_path": self._tmp_path,
                "catalog_root": self.catalog_root,
                "cache_dir": str(self.cache_dir) if self.cache_dir else None,
                "error": self._error,
                "supervisor": dict(self._sup),
            }
