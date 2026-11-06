"""账户与登录（monitor 独立服务的认证层；仅 server 侧使用，依赖 Flask/werkzeug）。

- 用户存储：``<data_root>/users.json``（0600），密码以 werkzeug scrypt 哈希保存。
  记录带 ``status``（``approved``/``pending``）：主站 ``/register`` 自助注册时，
  **首个账户直接 approved**（避免新部署无可用账户的死锁），其后注册写入
  pending，管理后台审批（approve）后才能登录；旧记录无该字段按 approved 兼容。
- 会话：Flask signed-cookie session；secret key 持久化于 ``<data_root>/.secret_key``
  （0600，首次自动生成）。cookie 打 ``Secure``/``HttpOnly``/``SameSite=Lax``。
- 登录失败限流：每 IP 每分钟最多 10 次尝试（内存计数，防在线爆破）。
- 账户的增删与审批只发生在仅本机绑定的管理后台（admin.py），主站只读 + 注册。
- Host 白名单：``HostAllowlist``（运行时可变，追加项持久化于
  ``<data_root>/access_addresses.json``），public 模式防 DNS rebinding。
"""

from __future__ import annotations

import functools
import logging
import re
import secrets
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from flask import jsonify, redirect, request, session
from werkzeug.security import check_password_hash, generate_password_hash

from .store import atomic_write_json, read_json

logger = logging.getLogger(__name__)

LOGIN_RATE_LIMIT = 10  # 次/分钟/IP
LOGIN_RATE_WINDOW = 60.0


def load_or_create_secret_key(data_root: object) -> bytes:
    """持久化 session secret key（0600）；已存在则复用（重启不掉线）。"""
    path = Path(str(data_root)) / ".secret_key"
    try:
        raw = path.read_bytes()
        if len(raw) >= 32:
            return raw
    except OSError:
        pass
    key = secrets.token_bytes(48)
    try:
        path.write_bytes(key)
        path.chmod(0o600)
    except OSError as exc:
        logger.warning("monitor: cannot persist secret key to %s: %s", path, exc)
    return key


class UserStore:
    """``users.json`` 的读写（线程安全，整体重写，文件 0600）。

    记录结构：``{username: {"hash": ..., "created_at": ..., "status": ...}}``；
    ``status`` 缺省（旧版记录）视为 ``approved``。
    """

    def __init__(self, data_root: object):
        self.path = Path(str(data_root)) / "users.json"
        self._lock = threading.Lock()

    def _load(self) -> Dict[str, Any]:
        doc = read_json(self.path, default=None)
        return doc if isinstance(doc, dict) else {}

    @staticmethod
    def _status(rec: Any) -> str:
        return (rec or {}).get("status") or "approved"

    def list_users(self) -> Dict[str, Any]:
        """username -> {created_at, status}（不暴露哈希）。"""
        with self._lock:
            return {
                name: {"created_at": (rec or {}).get("created_at"),
                       "status": self._status(rec)}
                for name, rec in self._load().items()
                if isinstance(rec, dict)
            }

    def count(self) -> int:
        with self._lock:
            return len(self._load())

    @staticmethod
    def _validate(username: str, password: str) -> str:
        username = (username or "").strip()
        if not username or any(c in username for c in "/\\\x00"):
            raise ValueError("invalid username")
        if len(username) > 64:
            raise ValueError("username too long")
        if not password or len(password) < 6:
            raise ValueError("password must be at least 6 characters")
        return username

    def add_user(self, username: str, password: str) -> None:
        """管理后台直接创建的账户：立即可登录（approved）。"""
        username = self._validate(username, password)
        with self._lock:
            doc = self._load()
            doc[username] = {
                "hash": generate_password_hash(password),
                "created_at": time.time(),
                "status": "approved",
            }
            self._write(doc)

    def register(self, username: str, password: str) -> str:
        """主站自助注册：首个账户直接 approved，其后写入 pending（管理后台审批后才能
        登录）。返回实际写入的 status。"""
        username = self._validate(username, password)
        with self._lock:
            doc = self._load()
            existing = doc.get(username)
            if isinstance(existing, dict):
                if self._status(existing) == "pending":
                    raise ValueError("该用户名已注册，正在等待审批")
                raise ValueError("该用户名已被使用")
            status = "approved" if not doc else "pending"
            doc[username] = {
                "hash": generate_password_hash(password),
                "created_at": time.time(),
                "status": status,
            }
            self._write(doc)
            return status

    def approve(self, username: str) -> bool:
        with self._lock:
            doc = self._load()
            rec = doc.get(username)
            if not isinstance(rec, dict):
                return False
            rec["status"] = "approved"
            self._write(doc)
            return True

    def reject(self, username: str) -> bool:
        """拒绝注册 = 删除 pending 账户；approved 账户走 delete_user。"""
        with self._lock:
            doc = self._load()
            rec = doc.get(username)
            if not isinstance(rec, dict):
                return False
            if self._status(rec) == "approved":
                raise ValueError("cannot reject an approved account; delete it instead")
            del doc[username]
            self._write(doc)
            return True

    def status_of(self, username: str) -> Optional[str]:
        with self._lock:
            rec = self._load().get(username or "")
        if not isinstance(rec, dict):
            return None
        return self._status(rec)

    def delete_user(self, username: str) -> bool:
        with self._lock:
            doc = self._load()
            if username not in doc:
                return False
            approved = [
                n for n, r in doc.items()
                if isinstance(r, dict) and self._status(r) == "approved"
            ]
            if username in approved and len(approved) <= 1:
                raise ValueError("cannot delete the last account")
            del doc[username]
            self._write(doc)
            return True

    def check(self, username: str, password: str) -> str:
        """登录校验：``ok``（可登录）/ ``pending``（密码对但未审批）/ ``invalid``。"""
        with self._lock:
            rec = self._load().get(username or "")
        if not isinstance(rec, dict) or not rec.get("hash"):
            # 与真实用户走同一条哈希校验路径，避免用户枚举的时间侧信道
            check_password_hash("scrypt:32768:8:1$invalid$invalid", password or "")
            return "invalid"
        try:
            if not check_password_hash(rec["hash"], password or ""):
                return "invalid"
        except ValueError:
            return "invalid"
        return "ok" if self._status(rec) == "approved" else "pending"

    def verify(self, username: str, password: str) -> bool:
        return self.check(username, password) == "ok"

    def _write(self, doc: Dict[str, Any]) -> None:
        atomic_write_json(self.path, doc)
        try:
            self.path.chmod(0o600)
        except OSError:
            pass


_VALID_HOST = re.compile(r"^[A-Za-z0-9._:-]{1,253}$")


class HostAllowlist:
    """运行时 Host 白名单（public 模式防 DNS rebinding）。

    初始集合 = 启动 SAN（服务器证书签名用，不可在线移除）；管理后台可追加
    地址，追加项持久化到 ``<data_root>/access_addresses.json``，加入后即时
    生效（``_security_gate`` 每次请求都查本对象）。
    """

    def __init__(self, data_root: object, base_hosts: Any = ()):
        self.path = Path(str(data_root)) / "access_addresses.json"
        self._lock = threading.Lock()
        self._base = {
            str(h).strip().lower() for h in (base_hosts or []) if str(h).strip()
        }
        raw = read_json(self.path, default=None)
        self._extra = {
            str(h).strip().lower()
            for h in (raw if isinstance(raw, list) else [])
            if isinstance(h, str) and str(h).strip()
        }

    def __contains__(self, host: object) -> bool:
        value = str(host or "").strip().lower()
        if not value:
            return False
        with self._lock:
            return value in self._base or value in self._extra

    def base_hosts(self) -> List[str]:
        with self._lock:
            return sorted(self._base)

    def extra_hosts(self) -> List[str]:
        with self._lock:
            return sorted(self._extra)

    def all_hosts(self) -> List[str]:
        with self._lock:
            return sorted(self._base | self._extra)

    def add(self, host: str) -> bool:
        """追加一个地址（幂等）；已在启动 SAN 中返回 False。非法字符报错。"""
        value = str(host or "").strip().lower()
        if not value or not _VALID_HOST.match(value) or ".." in value:
            raise ValueError(f"invalid host: {host!r}")
        with self._lock:
            if value in self._base or value in self._extra:
                return False
            self._extra.add(value)
            self._persist()
            return True

    def remove(self, host: str) -> bool:
        value = str(host or "").strip().lower()
        with self._lock:
            if value in self._base:
                raise ValueError("启动 SAN 不可在线移除（改 --san 配置并重签证书）")
            if value not in self._extra:
                return False
            self._extra.discard(value)
            self._persist()
            return True

    def _persist(self) -> None:
        atomic_write_json(self.path, sorted(self._extra))
        try:
            self.path.chmod(0o600)
        except OSError:
            pass


class LoginRateLimiter:
    """每 IP 滑动窗口登录尝试限流。"""

    def __init__(self, limit: int = LOGIN_RATE_LIMIT, window: float = LOGIN_RATE_WINDOW):
        self.limit = int(limit)
        self.window = float(window)
        self._hits: Dict[str, list] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            hits = [t for t in self._hits.get(key, []) if now - t < self.window]
            if len(hits) >= self.limit:
                self._hits[key] = hits
                return False
            hits.append(now)
            self._hits[key] = hits
            return True


def wants_json() -> bool:
    return request.path.startswith("/api/") or "application/json" in (
        request.headers.get("Accept") or ""
    )


def login_required(view: Callable) -> Callable:
    """主站路由保护：未登录 → API 401 / 页面 302 到 /login。"""

    @functools.wraps(view)
    def wrapper(*args: Any, **kwargs: Any):
        if not session.get("user"):
            if wants_json():
                return jsonify({"error": "authentication required"}), 401
            return redirect(f"/login?next={request.full_path if request.query_string else request.path}")
        return view(*args, **kwargs)

    return wrapper
