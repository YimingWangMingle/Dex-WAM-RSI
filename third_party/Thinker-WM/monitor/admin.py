"""管理后台（**仅绑定 127.0.0.1**，不对外开放，无登录——绑定地址即安全边界）。

功能：

- 证书：下载内部 CA 公钥证书（``ca.pem``）、导出 Ubuntu 导入说明（文本 /
  ``import-ca.sh`` 脚本）、按新 SAN 重签服务器证书（重签后需重启主站生效）。
- 账户：列出 / 新增 / 删除主站登录账户（禁止删除最后一个 approved 账户）；
  审批自助注册（approve / reject pending 账户）。
- 访问地址：管理主站 Host 白名单（运行时即时生效 + 持久化；可同时按合并
  SAN 重签证书）。
- 服务状态：各 sync engine 状态、dataset helper 池与看门狗状态、证书状态；
  支持手动重启 dataset helper。
- 服务日志：读取主进程 root logger 的内存 ring buffer（过滤/翻页）与
  ``<data_root>/logs/server.log`` 文件下载。

私钥（``ca-key.pem`` / ``server-key.pem``）任何时候都不提供下载。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, Response, request

from .auth import HostAllowlist, UserStore
from .certs import CertError, CertManager, import_guide_text, import_script
from .server import WEB_DIR, _error, _file_bytes, _json, static_file

logger = logging.getLogger(__name__)


def create_admin_app(
    data_root: object,
    *,
    user_store: Optional[UserStore] = None,
    cert_manager: Optional[CertManager] = None,
    sync_engines: Optional[List[Any]] = None,
    dataset_backend: Any = None,
    host_allowlist: Optional[HostAllowlist] = None,
    log_buffer: Any = None,
    log_file: Any = None,
    public_info: Optional[Dict[str, Any]] = None,
) -> Flask:
    # static_folder=None：禁用 Flask 内置 /static/ 路由（用 server.py 的防护路由）
    app = Flask("motus_monitor_admin", static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024

    # ------------------------------------------------------------ 页面/静态
    @app.get("/")
    @app.get("/admin")
    def index():
        return _file_bytes(WEB_DIR / "admin.html")

    @app.get("/static/<path:name>")
    def static_assets(name: str):
        return static_file(name)

    # ------------------------------------------------------------ 状态
    @app.get("/admin/api/status")
    def status() -> Response:
        sync = []
        for engine in sync_engines or []:
            item: Dict[str, Any] = dict(engine.status)
            item["data_version"] = engine.data_version
            sync.append(item)
        dataset = (
            dataset_backend.status()
            if dataset_backend is not None
            else {"enabled": False}
        )
        certs = cert_manager.status() if cert_manager is not None else {"ca_exists": False}
        users = (
            {"count": user_store.count(), "users": user_store.list_users()}
            if user_store is not None
            else {"count": 0, "users": {}}
        )
        return _json({"sync": sync, "dataset": dataset, "certs": certs, "users": users})

    # ------------------------------------------------------------ 证书
    @app.get("/admin/api/ca/download")
    def ca_download() -> Response:
        if cert_manager is None:
            return _error(400, "cert manager not configured")
        try:
            body = cert_manager.ca_cert_bytes()
        except CertError as exc:
            return _error(404, str(exc))
        return Response(
            body,
            content_type="application/x-pem-file",
            headers={"Content-Disposition": "attachment; filename=ca.pem"},
        )

    def _server_hint() -> str:
        """导入说明里的服务地址：优先证书 SAN/访问地址里的第一个非 localhost + public 端口。"""
        info = public_info or {}
        port = info.get("port")
        hosts = list(info.get("sans") or [])
        if host_allowlist is not None:
            try:
                hosts = host_allowlist.all_hosts()
            except Exception:
                pass
        if not hosts and cert_manager is not None:
            hosts = [h.split(":", 1)[-1] if ":" in h else h for h in cert_manager.current_sans()]
        localhost = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
        host = next((h for h in hosts if h and h.lower() not in localhost), "")
        if not host and hosts:
            host = hosts[0]
        if host and port:
            return f"{host}:{int(port)}"
        return host or ""

    @app.get("/admin/api/ca/import-guide")
    def ca_import_guide() -> Response:
        fmt = request.args.get("format") or "text"
        if fmt == "script":
            return Response(
                import_script(),
                content_type="application/x-sh",
                headers={"Content-Disposition": "attachment; filename=import-ca.sh"},
            )
        return _json({"guide": import_guide_text(server=_server_hint()), "server": _server_hint()})

    @app.post("/admin/api/cert/regenerate")
    def cert_regenerate() -> Response:
        if cert_manager is None:
            return _error(400, "cert manager not configured")
        body = request.get_json(silent=True) or {}
        sans = body.get("san") or body.get("sans") or []
        if isinstance(sans, str):
            sans = [s.strip() for s in sans.split(",") if s.strip()]
        if not isinstance(sans, list) or not sans:
            return _error(400, "need san: [<ip-or-hostname>, ...]")
        try:
            cert_manager.ensure_server_cert([str(s) for s in sans], force=True)
        except CertError as exc:
            return _error(400, str(exc))
        return _json(
            {
                "ok": True,
                "sans": cert_manager.current_sans(),
                "note": "服务器证书已重签；重启主站（serve）后生效。",
            }
        )

    # ------------------------------------------------------------ 账户
    @app.get("/admin/api/users")
    def users_list() -> Response:
        if user_store is None:
            return _error(400, "user store not configured")
        return _json({"users": user_store.list_users()})

    @app.post("/admin/api/users")
    def users_add() -> Response:
        if user_store is None:
            return _error(400, "user store not configured")
        body = request.get_json(silent=True) or {}
        username = str(body.get("username") or "")
        password = str(body.get("password") or "")
        try:
            user_store.add_user(username, password)
        except ValueError as exc:
            return _error(400, str(exc))
        return _json({"ok": True, "username": username.strip()})

    @app.route("/admin/api/users/<username>", methods=["DELETE"])
    def users_delete(username: str) -> Response:
        if user_store is None:
            return _error(400, "user store not configured")
        try:
            deleted = user_store.delete_user(username)
        except ValueError as exc:
            return _error(400, str(exc))
        if not deleted:
            return _error(404, f"unknown user: {username}")
        return _json({"ok": True})

    @app.post("/admin/api/users/<username>/approve")
    def users_approve(username: str) -> Response:
        if user_store is None:
            return _error(400, "user store not configured")
        if not user_store.approve(username):
            return _error(404, f"unknown user: {username}")
        return _json({"ok": True, "username": username, "status": "approved"})

    @app.post("/admin/api/users/<username>/reject")
    def users_reject(username: str) -> Response:
        if user_store is None:
            return _error(400, "user store not configured")
        try:
            rejected = user_store.reject(username)
        except ValueError as exc:
            return _error(400, str(exc))
        if not rejected:
            return _error(404, f"unknown user: {username}")
        return _json({"ok": True})

    # ------------------------------------------------------------ 访问地址
    @app.get("/admin/api/access-addresses")
    def access_list() -> Response:
        if host_allowlist is None:
            return _error(400, "host allowlist not configured (public mode only)")
        return _json({
            "base": host_allowlist.base_hosts(),
            "extra": host_allowlist.extra_hosts(),
        })

    @app.post("/admin/api/access-addresses")
    def access_add() -> Response:
        if host_allowlist is None:
            return _error(400, "host allowlist not configured (public mode only)")
        body = request.get_json(silent=True) or {}
        host = str(body.get("host") or "").strip()
        try:
            added = host_allowlist.add(host)
        except ValueError as exc:
            return _error(400, str(exc))
        cert_note = None
        if cert_manager is not None:
            try:
                # 用合并后的 SAN 重签证书（重启主站后新地址才不再有证书警告）
                cert_manager.ensure_server_cert(host_allowlist.all_hosts(), force=True)
                cert_note = "服务器证书已按新地址重签；重启主站（serve）后生效。"
            except CertError as exc:
                cert_note = f"地址已生效，但证书重签失败：{exc}"
        return _json({
            "ok": True,
            "host": host.lower(),
            "added": added,
            "cert_note": cert_note,
            "note": "Host 白名单已即时生效（无需重启）。",
        })

    @app.route("/admin/api/access-addresses/<host>", methods=["DELETE"])
    def access_remove(host: str) -> Response:
        if host_allowlist is None:
            return _error(400, "host allowlist not configured (public mode only)")
        try:
            removed = host_allowlist.remove(host)
        except ValueError as exc:
            return _error(400, str(exc))
        if not removed:
            return _error(404, f"not in added addresses: {host}")
        return _json({"ok": True})

    # ------------------------------------------------------------ helper
    @app.post("/admin/api/helper/restart")
    def helper_restart() -> Response:
        if dataset_backend is None:
            return _error(400, "dataset preview not configured")
        restart = getattr(dataset_backend, "restart", None)
        if restart is None:
            return _error(400, "dataset backend does not support restart")
        try:
            restart()
        except Exception as exc:
            return _error(502, f"{type(exc).__name__}: {exc}")
        return _json({"ok": True, "status": dataset_backend.status()})

    # ------------------------------------------------------------ 服务日志
    @app.get("/admin/api/logs")
    def admin_logs() -> Response:
        """内存 ring buffer 日志：tail 条数（默认 200，上限 2000）、level 级别下限、q 关键字。"""
        if log_buffer is None:
            return _error(400, "log buffer not configured")
        try:
            tail = min(2000, max(1, int(request.args.get("tail", 200))))
        except (TypeError, ValueError):
            tail = 200
        level_name = (request.args.get("level") or "").strip().upper()
        level_no = logging.getLevelName(level_name)
        if not isinstance(level_no, int):
            level_no = None
        query = (request.args.get("q") or "").strip().lower()
        entries = log_buffer.snapshot()
        if level_no is not None:
            entries = [e for e in entries if int(e.get("levelno") or 0) >= level_no]
        if query:
            entries = [
                e for e in entries
                if query in str(e.get("message") or "").lower()
                or query in str(e.get("logger") or "").lower()
            ]
        entries.reverse()  # 倒序：最新在前
        return _json({"logs": entries[:tail]})

    @app.get("/admin/api/logs/download")
    def admin_logs_download() -> Response:
        """下载当前 server.log（文件不存在 404）。"""
        if log_file is None:
            return _error(404, "server.log not configured")
        path = Path(str(log_file))
        try:
            body = path.read_bytes()
        except OSError:
            return _error(404, f"not found: {path.name}")
        return Response(
            body,
            content_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": "attachment; filename=server.log"},
        )

    return app
