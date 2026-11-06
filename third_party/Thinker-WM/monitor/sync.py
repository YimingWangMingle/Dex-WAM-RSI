"""Incremental bidirectional sync between the cluster and the local machine.

Direction of travel:

* ``runs/``        cluster -> local (training data; byte-level increments)
* ``annotations/`` local -> cluster (benchmark notes written in the web UI)

The transport is plain SSH; per-poll cost is one fast round trip even through
jump hosts because connection reuse is inherited from the user's ssh config
(``ControlMaster``/``ControlPersist`` in ``~/.ssh/config``). The remote side
only needs ``python3`` (every training container has it): a small helper
script is passed base64-encoded on the command line and reads a one-line JSON
command from stdin.

Increments are byte-exact thanks to the append-only chunked JSONL layout: the
local side remembers each file's synced size and only asks for the tail that
grew since the last poll. Small rewritten JSON files are simply re-fetched in
full (their size changes).

``local://<path>`` remotes use the same engine against another local
directory -- used by the smoke tests and same-machine setups.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import shlex
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .store import atomic_write_json, read_json

logger = logging.getLogger(__name__)

PULL_SUBDIR = "runs"
PUSH_SUBDIR = "annotations"
STATE_DIRNAME = ".sync"
STATE_FILENAME = "state.json"

# 单次 ssh fetch 的批量上限：块必须小到一次调用（受 ssh 超时约束）传得完。
FETCH_CHUNK_FILES = 128
FETCH_CHUNK_BYTES = 64 * 1024 * 1024

# ---------------------------------------------------------------------------
# Remote helper (executed ON THE CLUSTER via `python3 -c <launcher> <b64>`).
# Protocol: one JSON command line on stdin, framed output on stdout.
#   {"op": "list",  "dir": root, "sub": "runs"}
#       -> stdout lines: ["relative/path", 12345]   (JSON per line)
#   {"op": "fetch", "dir": root, "files": [{"path": p, "offset": n}, ...]}
#       -> for each: header "#<nbytes> <path>\n" then <nbytes> raw bytes
#   {"op": "push",  "dir": root, "files": [{"path": p, "size": n}, ...]}
#       stdin: command line, then the raw bytes of every file concatenated
#       -> stdout: "OK\n"
# ---------------------------------------------------------------------------
_REMOTE_HELPER = r'''
import json, os, sys

def safe_path(root, rel):
    root_real = os.path.realpath(root)
    full = os.path.realpath(os.path.join(root_real, rel))
    if full != root_real and not full.startswith(root_real + os.sep):
        raise ValueError("unsafe path: " + rel)
    return full

def main():
    stdin = sys.stdin.buffer
    cmd = json.loads(stdin.readline().decode("utf-8"))
    root = cmd["dir"]
    op = cmd["op"]
    if op == "list":
        base = safe_path(root, cmd.get("sub", "") or ".")
        out = sys.stdout
        if os.path.isdir(base):
            for dirpath, dirnames, filenames in os.walk(base):
                dirnames[:] = [d for d in dirnames if d != ".sync"]
                for name in filenames:
                    if name.endswith(".tmp"):
                        continue
                    full = os.path.join(dirpath, name)
                    try:
                        st = os.stat(full)
                    except OSError:
                        continue
                    rel = os.path.relpath(full, root)
                    out.write(json.dumps([rel, st.st_size, st.st_mtime_ns]) + "\n")
        out.flush()
    elif op == "fetch":
        w = sys.stdout.buffer
        for item in cmd["files"]:
            rel = item["path"]
            offset = max(0, int(item.get("offset", 0)))
            try:
                with open(safe_path(root, rel), "rb") as fp:
                    fp.seek(offset)
                    data = fp.read()
            except OSError:
                data = b""
            w.write(("#%d %s\n" % (len(data), rel)).encode("utf-8"))
            w.write(data)
        w.flush()
    elif op == "push":
        for item in cmd["files"]:
            rel = item["path"]
            size = int(item["size"])
            data = b""
            while len(data) < size:
                chunk = stdin.read(size - len(data))
                if not chunk:
                    break
                data += chunk
            data = data[:size]
            full = safe_path(root, rel)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            tmp = full + ".tmp"
            with open(tmp, "wb") as fp:
                fp.write(data)
            os.replace(tmp, full)
        sys.stdout.write("OK\n")
        sys.stdout.flush()

main()
'''
_HELPER_B64 = base64.b64encode(_REMOTE_HELPER.encode("utf-8")).decode("ascii")
_HELPER_LAUNCHER = "import sys,base64;exec(base64.b64decode(sys.argv[1]).decode())"


class SyncError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
class LocalBackend:
    """Same protocol against another local directory (tests / same machine)."""

    def __init__(self, remote_dir: object):
        self.remote_dir = Path(str(remote_dir))
        self.description = f"local://{self.remote_dir}"

    def list_files(self, sub: str) -> Dict[str, Tuple[int, int]]:
        """rel_path -> (size_bytes, mtime_ns)."""
        base = self.remote_dir / sub
        result: Dict[str, Tuple[int, int]] = {}
        if not base.is_dir():
            return result
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d != STATE_DIRNAME]
            for name in filenames:
                if name.endswith(".tmp"):
                    continue
                full = Path(dirpath) / name
                try:
                    stat = full.stat()
                    result[str(full.relative_to(self.remote_dir))] = (
                        int(stat.st_size),
                        int(stat.st_mtime_ns),
                    )
                except OSError:
                    continue
        return result

    def fetch(self, files: List[Tuple[str, int]]) -> Dict[str, bytes]:
        out: Dict[str, bytes] = {}
        for rel, offset in files:
            try:
                with (self.remote_dir / rel).open("rb") as fp:
                    fp.seek(max(0, int(offset)))
                    out[rel] = fp.read()
            except OSError:
                out[rel] = b""
        return out

    def push(self, files: List[Tuple[str, bytes]]) -> None:
        for rel, data in files:
            full = self.remote_dir / rel
            full.parent.mkdir(parents=True, exist_ok=True)
            tmp = full.with_name(full.name + ".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, full)


class SshBackend:
    """Backend speaking to the remote helper over SSH.

    Connection reuse is INHERITED from the user's ~/.ssh/config (e.g. a
    ``Host *`` stanza with ``ControlMaster auto`` / ``ControlPersist``): we
    deliberately do not force our own ControlPath, because that would require
    a fresh authentication -- which fails on targets that only accept
    interactive (already-established) master connections. Extra options
    (ProxyJump, ports, ...) go through ``ssh_args``.
    """

    def __init__(self, target: str, remote_dir: str, ssh_args: str = "", timeout: int = 180):
        if not target:
            raise SyncError("ssh target is required (e.g. fengodchen@localhost)")
        self.target = target
        self.remote_dir = remote_dir.rstrip("/")
        self.ssh_args = ssh_args or ""
        self.timeout = int(timeout)
        self.description = f"ssh://{target}/{self.remote_dir}"
        # ssh joins remote argv with spaces and hands it to the REMOTE SHELL,
        # so build one properly quoted command string ourselves (the launcher
        # contains ';' and the base64 payload must survive word splitting).
        remote_argv = ["python3", "-u", "-c", _HELPER_LAUNCHER, _HELPER_B64]
        remote_cmd = " ".join(shlex.quote(arg) for arg in remote_argv)
        self.base_cmd = (
            [
                "ssh",
                "-o", "BatchMode=yes",
                "-o", "ConnectTimeout=10",
                "-o", "Compression=yes",
            ]
            + shlex.split(ssh_args or "")
            + [target, remote_cmd]
        )

    def _run(self, command: Dict[str, Any], payload: bytes = b"") -> bytes:
        stdin = json.dumps(command).encode("utf-8") + b"\n" + payload
        try:
            proc = subprocess.run(
                self.base_cmd,
                input=stdin,
                capture_output=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise SyncError(f"ssh timed out after {self.timeout}s") from exc
        except OSError as exc:
            raise SyncError(f"ssh failed to start: {exc}") from exc
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace")[-400:]
            raise SyncError(f"ssh exited {proc.returncode}: {stderr.strip()}")
        return proc.stdout

    def list_files(self, sub: str) -> Dict[str, Tuple[int, int]]:
        """rel_path -> (size_bytes, mtime_ns)."""
        raw = self._run({"op": "list", "dir": self.remote_dir, "sub": sub})
        result: Dict[str, Tuple[int, int]] = {}
        for line in raw.decode("utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line.startswith("["):
                continue
            try:
                rel, size, mtime_ns = json.loads(line)
                result[str(rel)] = (int(size), int(mtime_ns))
            except (ValueError, TypeError):
                continue
        return result

    def fetch(self, files: List[Tuple[str, int]]) -> Dict[str, bytes]:
        if not files:
            return {}
        command = {
            "op": "fetch",
            "dir": self.remote_dir,
            "files": [{"path": rel, "offset": int(offset)} for rel, offset in files],
        }
        raw = self._run(command)
        out: Dict[str, bytes] = {}
        pos = 0
        for rel, _ in files:
            # header: "#<nbytes> <path>\n"
            end = raw.find(b"\n", pos)
            if end < 0 or not raw[pos:pos + 1] == b"#":
                raise SyncError(f"corrupt fetch stream near {rel!r}")
            header = raw[pos + 1:end].decode("utf-8", errors="replace")
            size_str, _, header_path = header.partition(" ")
            try:
                size = int(size_str)
            except ValueError:
                raise SyncError(f"corrupt fetch header: {header!r}")
            pos = end + 1
            data = raw[pos:pos + size]
            if len(data) != size:
                raise SyncError(f"truncated fetch payload for {rel!r}")
            out[header_path or rel] = data
            pos += size
        return out

    def push(self, files: List[Tuple[str, bytes]]) -> None:
        if not files:
            return
        command = {
            "op": "push",
            "dir": self.remote_dir,
            "files": [{"path": rel, "size": len(data)} for rel, data in files],
        }
        payload = b"".join(data for _, data in files)
        raw = self._run(command, payload)
        if b"OK" not in raw[-16:]:
            raise SyncError(f"unexpected push reply: {raw[-80:]!r}")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
def parse_remote(remote: str, ssh_args: str = ""):
    """Build a backend from a remote spec.

    Accepted forms:
      ``local:///abs/path``                     -> LocalBackend
      ``<target>:<remote_dir>``                 -> SshBackend (scp-like)
      ``ssh://<target>/<remote_dir>``           -> SshBackend
    """
    remote = (remote or "").strip()
    if not remote:
        return None
    if remote.startswith("local://"):
        return LocalBackend(remote[len("local://"):])
    if remote.startswith("ssh://"):
        rest = remote[len("ssh://"):]
        target, _, path = rest.partition("/")
        return SshBackend(target, "/" + path, ssh_args=ssh_args)
    if ":" in remote:
        target, _, path = remote.partition(":")
        return SshBackend(target, path, ssh_args=ssh_args)
    # Bare path: treat as local directory.
    return LocalBackend(remote)


class SyncEngine:
    """One-shot-capable sync engine; also drives :class:`SyncDaemon`.

    ``name`` 用于多远端部署：命名实例的状态文件为 ``.sync/<name>.json``
    （未命名保持旧版 ``.sync/state.json``），多个 engine 可把不同远端合并
    拉取进同一个 ``local_dir/runs/``。
    """

    def __init__(self, local_dir: object, backend=None, debounce_push: bool = True,
                 name: Optional[str] = None):
        self.local_dir = Path(str(local_dir))
        self.local_dir.mkdir(parents=True, exist_ok=True)
        (self.local_dir / PULL_SUBDIR).mkdir(exist_ok=True)
        (self.local_dir / PUSH_SUBDIR).mkdir(exist_ok=True)
        self.backend = backend
        self.name = name or ""
        if name:
            safe = re.sub(r"[^A-Za-z0-9._-]+", "-", str(name))[:60] or "remote"
            self.state_path = self.local_dir / STATE_DIRNAME / f"{safe}.json"
        else:
            self.state_path = self.local_dir / STATE_DIRNAME / STATE_FILENAME
        self._state: Dict[str, Any] = self._load_state()
        self.data_version = 0  # bumped whenever new bytes land locally
        self.status: Dict[str, Any] = {
            "name": self.name,
            "backend": getattr(backend, "description", None),
            "cycles": 0,
            "last_success_ts": None,
            "last_error": None,
            "last_error_ts": None,
            "bytes_pulled_total": 0,
            "bytes_pushed_total": 0,
            "last_pull_files": 0,
            "last_pull_bytes": 0,
            "last_push_files": 0,
            # 本周期拉取进度（前端进度条；周期间保留上周期的最终值）
            "phase": "idle",  # idle | listing | pulling
            "pull_pending_files": 0,
            "pull_pending_bytes": 0,
            "pull_done_files": 0,
            "pull_done_bytes": 0,
            "pull_current_file": None,
        }
        self._lock = threading.Lock()

    def _load_state(self) -> Dict[str, Any]:
        """读取同步状态；命名 engine 首次运行时接管旧版单远端状态文件。

        多远端改造前状态固定在 ``.sync/state.json``。升级后命名 engine
        （如 remote0）若还没有自己的状态文件，会直接接管这份旧状态并
        将旧文件改名退役——否则远端全部历史数据都会被视为"新文件"，
        首次全量拉取在一次 ssh 调用里传不完，超时后状态又从未落盘，
        每个周期都从零重试，同步永久卡死。
        """
        state = read_json(self.state_path, default=None)
        if state is None and self.name:
            legacy_path = self.local_dir / STATE_DIRNAME / STATE_FILENAME
            if legacy_path != self.state_path:
                state = read_json(legacy_path, default=None)
                if state is not None:
                    atomic_write_json(self.state_path, state)
                    try:
                        legacy_path.rename(legacy_path.with_name(
                            f"{legacy_path.name}.claimed-by-{self.name}"))
                    except OSError:
                        logger.warning("sync [%s]: failed to retire legacy state %s",
                                       self.name, legacy_path)
                    logger.info("sync [%s]: adopted legacy state %s",
                                self.name, legacy_path)
        state = state or {}
        state.setdefault("files", {})
        state.setdefault("pushed", {})
        return state

    # ------------------------------------------------------------------ pull
    def _local_size(self, rel: str) -> int:
        try:
            return int((self.local_dir / rel).stat().st_size)
        except OSError:
            return 0

    def _pull_runs(self) -> Tuple[int, int]:
        assert self.backend is not None
        status = self.status
        status["phase"] = "listing"
        try:
            remote_files = self.backend.list_files(PULL_SUBDIR)
            synced: Dict[str, Any] = self._state["files"]
            to_fetch: List[Tuple[str, int]] = []
            new_state: Dict[str, Any] = {}
            for rel, meta in remote_files.items():
                size, mtime_ns = int(meta[0]), int(meta[1])
                prev = synced.get(rel)
                if isinstance(prev, dict):
                    prev_size = int(prev.get("size", 0))
                    prev_mtime = int(prev.get("mtime_ns", 0))
                elif prev is not None:  # legacy int state
                    prev_size, prev_mtime = int(prev), 0
                else:
                    prev_size, prev_mtime = 0, 0
                # 自愈：state 只信到本地文件的真实大小——state 脏（历史 bug
                # 把未拉取的文件记成远端完整大小、或本地文件被删/截断）时
                # 自动补拉，无需人工重置 state。
                effective_size = min(prev_size, self._local_size(rel))
                # state 先继承本地已同步位置；只有实际拉取成功后才更新为
                # 新位置（见下方 chunk 循环）。绝不能在这里直接记远端
                # size——否则 fetch 中途失败时，未拉取的文件会被误认为已
                # 同步，之后 size 不变就永远补不回来。
                entry = {"size": effective_size, "mtime_ns": prev_mtime}
                if rel.endswith(".jsonl"):
                    # Append-only stream chunks: fetch just the grown tail. A
                    # smaller size means the file was rotated/truncated remotely,
                    # so restart it from zero.
                    if size > effective_size:
                        to_fetch.append((rel, effective_size))
                    elif size < effective_size:
                        to_fetch.append((rel, 0))
                    new_state[rel] = entry
                else:
                    # Whole-file rewritten docs (run.json/status.json/...): any
                    # size OR mtime change means the content changed -- refetch
                    # the entire (small) file.
                    if size != effective_size or mtime_ns != prev_mtime:
                        to_fetch.append((rel, 0))
                    new_state[rel] = entry
            if not to_fetch:
                self._state["files"] = new_state
                return 0, 0

            status.update(
                phase="pulling",
                pull_pending_files=len(to_fetch),
                pull_pending_bytes=sum(
                    max(0, int(remote_files[rel][0]) - offset) for rel, offset in to_fetch
                ),
                pull_done_files=0,
                pull_done_bytes=0,
                pull_current_file=None,
            )
            pulled_files = 0
            pulled_bytes = 0
            for chunk in self._fetch_chunks(to_fetch, remote_files):
                blobs = self.backend.fetch(chunk)
                for rel, offset in chunk:
                    data = blobs.get(rel)
                    if data is None:
                        continue
                    status["pull_current_file"] = rel
                    local_path = self.local_dir / rel
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    mode = "r+b" if offset > 0 and local_path.exists() else "wb"
                    with local_path.open(mode) as fp:
                        if offset > 0:
                            fp.seek(offset)
                        fp.write(data)
                        fp.truncate()
                    new_state[rel] = {
                        "size": offset + len(data),
                        "mtime_ns": int(remote_files[rel][1]),
                    }
                    pulled_files += 1
                    pulled_bytes += len(data)
                    status["pull_done_files"] += 1
                    status["pull_done_bytes"] += len(data)
                # 每块落盘一次状态：大 backlog（如首次全量同步）中途超时时，
                # 下一周期从断点继续，而不是反复从零重拉。
                self._state["files"] = new_state
                atomic_write_json(self.state_path, self._state)
            status["pull_current_file"] = None
            return pulled_files, pulled_bytes
        finally:
            status["phase"] = "idle"

    @staticmethod
    def _fetch_chunks(
        to_fetch: List[Tuple[str, int]],
        remote_files: Dict[str, Any],
    ) -> List[List[Tuple[str, int]]]:
        """把待拉清单切成单次 ssh 调用传得完的小块（按条数与字节数双上限）。"""
        chunks: List[List[Tuple[str, int]]] = []
        chunk: List[Tuple[str, int]] = []
        chunk_bytes = 0
        for rel, offset in to_fetch:
            est = max(0, int(remote_files[rel][0]) - offset)
            if chunk and (len(chunk) >= FETCH_CHUNK_FILES
                          or chunk_bytes + est > FETCH_CHUNK_BYTES):
                chunks.append(chunk)
                chunk, chunk_bytes = [], 0
            chunk.append((rel, offset))
            chunk_bytes += est
        if chunk:
            chunks.append(chunk)
        return chunks

    # ------------------------------------------------------------------ push
    def _pending_push(self) -> List[Tuple[str, bytes, List[int]]]:
        pushed: Dict[str, List[int]] = self._state["pushed"]
        result: List[Tuple[str, bytes, List[int]]] = []
        local_annotations = self.local_dir / PUSH_SUBDIR
        if local_annotations.is_dir():
            for path in sorted(local_annotations.glob("*.json")):
                rel = f"{PUSH_SUBDIR}/{path.name}"
                try:
                    stat = path.stat()
                except OSError:
                    continue
                marker = [int(stat.st_size), int(stat.st_mtime_ns)]
                if pushed.get(rel) != marker:
                    result.append((rel, path.read_bytes(), marker))
        # Remote-deleted tracking is intentionally not implemented: local is
        # the authoritative writer for annotations.
        return result

    def sync_once(self) -> Dict[str, Any]:
        """Run one pull+push cycle; safe to call concurrently (serialized)."""
        with self._lock:
            self.status["cycles"] += 1
            summary: Dict[str, Any] = {"ok": True}
            if self.backend is None:
                summary["ok"] = True
                summary["note"] = "no remote configured (local mode)"
                return summary
            try:
                files, nbytes = self._pull_runs()
                push_items = self._pending_push()
                if push_items:
                    self.backend.push([(rel, data) for rel, data, _ in push_items])
                    pushed: Dict[str, List[int]] = self._state["pushed"]
                    for rel, _, marker in push_items:
                        pushed[rel] = marker
                self.status["bytes_pulled_total"] += nbytes
                self.status["bytes_pushed_total"] += sum(len(d) for _, d, _ in push_items)
                self.status["last_pull_files"] = files
                self.status["last_pull_bytes"] = nbytes
                self.status["last_push_files"] = len(push_items)
                self.status["last_success_ts"] = time.time()
                self.status["last_error"] = None
                if files or push_items:
                    self.data_version += 1
                atomic_write_json(self.state_path, self._state)
                summary.update(
                    pull_files=files,
                    pull_bytes=nbytes,
                    push_files=len(push_items),
                )
            except SyncError as exc:
                self.status["last_error"] = str(exc)
                self.status["last_error_ts"] = time.time()
                summary.update(ok=False, error=str(exc))
            return summary


class SyncDaemon(threading.Thread):
    """Background poller: sync every ``interval`` seconds with error backoff."""

    def __init__(self, engine: SyncEngine, interval: float = 2.0, max_backoff: float = 30.0):
        super().__init__(name="motus-monitor-sync", daemon=True)
        self.engine = engine
        self.interval = max(0.5, float(interval))
        self.max_backoff = float(max_backoff)
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        backoff = self.interval
        while not self._stop_event.is_set():
            summary = self.engine.sync_once()
            if summary.get("ok"):
                backoff = self.interval
            else:
                logger.warning("monitor sync failed: %s", summary.get("error"))
                backoff = min(self.max_backoff, max(self.interval, backoff * 2))
            self._stop_event.wait(backoff)


def build_engine(
    local_dir: object,
    remote: Optional[str] = None,
    ssh_args: str = "",
    name: Optional[str] = None,
) -> SyncEngine:
    backend = parse_remote(remote, ssh_args=ssh_args) if remote else None
    return SyncEngine(local_dir, backend=backend, name=name)


# ---------------------------------------------------------------------------
# 一次性 SSH 调用（web 端"浏览 H-Server 目录 / 扫描远端数据根"用）
#
# 与 sync helper 不同：不走长驻帧协议，每次调用起一个
# ``ssh <args> <target> python3 -c <script> -- <argv...>`` 短进程，stdout
# 最后一行是 JSON 结果。远端只需 python3；BatchMode 免密（复用用户
# ~/.ssh/config 的 ControlMaster 持久连接）。
# ---------------------------------------------------------------------------
def ssh_run_json(
    target: str, ssh_args: str, script: str, argv: List[str], timeout: int = 30
) -> Any:
    remote_argv = ["python3", "-c", script, "--"] + [str(a) for a in argv]
    remote_cmd = " ".join(shlex.quote(a) for a in remote_argv)
    cmd = (
        [
            "ssh",
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=10",
            "-o", "Compression=yes",
        ]
        + shlex.split(ssh_args or "")
        + [target, remote_cmd]
    )
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise SyncError(f"ssh timed out after {timeout}s") from exc
    except OSError as exc:
        raise SyncError(f"ssh failed to start: {exc}") from exc
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace")[-300:]
        raise SyncError(f"ssh exited {proc.returncode}: {stderr.strip()}")
    out = proc.stdout.decode("utf-8", errors="replace").strip()
    if not out:
        raise SyncError("empty reply from remote")
    try:
        return json.loads(out.splitlines()[-1])
    except ValueError as exc:
        raise SyncError(f"bad reply from remote: {out[-200:]!r}") from exc


# 浏览远端目录（只列子目录；entries 上限 500，全部异常进 error 字段）
SSH_BROWSE_SCRIPT = r"""
import json, os, sys
args = [a for a in sys.argv[1:] if a != "--"]
path = args[0] if args else "/"
out = {"path": path, "parent": os.path.dirname(path.rstrip("/")) or "/", "entries": []}
try:
    entries = []
    with os.scandir(path) as it:
        for e in it:
            try:
                if not e.is_dir(follow_symlinks=True):
                    continue
                entries.append({"name": e.name,
                                "has_runs": os.path.isdir(os.path.join(e.path, "runs"))})
            except OSError:
                continue
            if len(entries) >= 500:
                break
    entries.sort(key=lambda x: x["name"])
    out["entries"] = entries
    out["looks_like_data_root"] = os.path.isdir(os.path.join(path, "runs"))
except OSError as exc:
    out["error"] = str(exc)
print(json.dumps(out))
"""

# 扫描远端候选目录（深度≤2 找含 runs/ 的目录；目录数/耗时双上限，异常静默）
SSH_SCAN_ROOTS_SCRIPT = r"""
import json, os, sys, time
deadline = time.time() + 20.0
results, seen = [], set()
def probe(d):
    if d in seen or len(results) >= 200 or time.time() > deadline:
        return
    seen.add(d)
    runs = os.path.join(d, "runs")
    if not os.path.isdir(runs):
        return
    count, latest = 0, 0.0
    try:
        with os.scandir(runs) as it:
            for e in it:
                if e.is_dir(follow_symlinks=True):
                    count += 1
                    try:
                        latest = max(latest, e.stat(follow_symlinks=True).st_mtime)
                    except OSError:
                        pass
    except OSError:
        pass
    results.append({"path": d, "run_count": count, "latest_ts": latest})

for cand in [a for a in sys.argv[1:] if a != "--"]:
    if time.time() > deadline or len(results) >= 200:
        break
    probe(cand)
    try:
        with os.scandir(cand) as it:
            for e in it:
                if time.time() > deadline:
                    break
                try:
                    if e.is_dir(follow_symlinks=True):
                        probe(e.path)
                except OSError:
                    continue
    except OSError:
        continue
results.sort(key=lambda r: r.get("latest_ts") or 0, reverse=True)
print(json.dumps(results))
"""
