"""内部 CA 与服务器证书管理（monitor 独立服务的 HTTPS 支撑）。

用 openssl CLI 生成/签发（固定 argv 列表，不经 shell，无注入面）：

- 内部 CA：``<data_root>/certs/ca-key.pem``（0600，**永不出服务器**）+ ``ca.pem``
  （公钥证书，供管理后台下载，由管理员导入各客户端 Ubuntu 系统信任库）。
- 服务器证书：``server-key.pem``（0600）+ ``server.pem``，SAN 覆盖配置给出的
  IP/主机名列表（浏览器按 SAN 校验）。

同时导出 Ubuntu 导入说明（文本 + 可执行脚本），见 :func:`import_guide_text`
与 :func:`import_script`。
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

CA_DAYS = 3650
SERVER_DAYS = 825  # 主流浏览器对服务器证书的最大接受期约 398 天起，内部 CA 不受限，取 825 天
_SAN_TOKEN = re.compile(r"^[A-Za-z0-9._:-]+$")  # IP / DNS 名白名单字符（防参数注入）

# 证书 DN（内部 CA 与服务器证书统一主体信息）
CA_SUBJ = (
    "/C=CN/ST=Guangdong/L=Shenzhen/O=Ubtech/OU=chen.liao"
    "/CN=Motus Monitor Internal CA/emailAddress=chen.liao@ubtrobot.com"
)
SERVER_SUBJ = (
    "/C=CN/ST=Guangdong/L=Shenzhen/O=Ubtech/OU=chen.liao"
    "/CN=motus-monitor/emailAddress=chen.liao@ubtrobot.com"
)


class CertError(RuntimeError):
    pass


def _openssl() -> str:
    exe = shutil.which("openssl")
    if not exe:
        raise CertError("openssl not found in PATH")
    return exe


def _run(argv: List[str], timeout: int = 60) -> None:
    try:
        proc = subprocess.run(argv, capture_output=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CertError(f"openssl failed to start: {exc}") from exc
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace")[-400:]
        raise CertError(f"openssl exited {proc.returncode}: {stderr.strip()}")


def _chmod_600(path: Path) -> None:
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


class CertManager:
    """管理 ``<data_root>/certs/`` 下的 CA 与服务器证书。"""

    def __init__(self, data_root: object):
        self.dir = Path(str(data_root)) / "certs"
        self.ca_key = self.dir / "ca-key.pem"
        self.ca_cert = self.dir / "ca.pem"
        self.server_key = self.dir / "server-key.pem"
        self.server_cert = self.dir / "server.pem"

    # ------------------------------------------------------------------ CA
    def ensure_ca(self) -> Path:
        """生成内部 CA（已存在则直接返回 ca.pem 路径）。"""
        if self.ca_cert.is_file() and self.ca_key.is_file():
            return self.ca_cert
        self.dir.mkdir(parents=True, exist_ok=True)
        _run(
            [
                _openssl(), "req", "-x509", "-newkey", "rsa:4096",
                "-keyout", str(self.ca_key),
                "-out", str(self.ca_cert),
                "-days", str(CA_DAYS), "-nodes",
                "-subj", CA_SUBJ,
            ]
        )
        _chmod_600(self.ca_key)
        logger.info("monitor: internal CA created at %s", self.ca_cert)
        return self.ca_cert

    # ------------------------------------------------------------ 服务器证书
    @staticmethod
    def _san_config(sans: List[str]) -> str:
        entries = []
        dns_i = ip_i = 0
        for token in sans:
            token = str(token).strip()
            if not token or not _SAN_TOKEN.match(token):
                raise CertError(f"invalid SAN entry: {token!r}")
            # 纯数字+点/冒号视为 IP（v4/v6），其余按 DNS
            if re.fullmatch(r"[0-9.]+", token) or ":" in token:
                ip_i += 1
                entries.append(f"IP.{ip_i} = {token}")
            else:
                dns_i += 1
                entries.append(f"DNS.{dns_i} = {token}")
        if not entries:
            raise CertError("SAN list is empty")
        return (
            "[v3_req]\n"
            "subjectAltName = @alt_names\n"
            "extendedKeyUsage = serverAuth\n"
            "keyUsage = digitalSignature, keyEncipherment\n"
            "[alt_names]\n" + "\n".join(entries) + "\n"
        )

    def _key_cert_match(self) -> bool:
        """server-key.pem 与 server.pem 是否同一把钥匙（模数一致）。

        重签中途崩溃 / 目录被部分同步都可能留下不配对的 key/cert——那种状态
        会让 werkzeug 启动即抛 ``ssl.SSLError: [X509: KEY_VALUES_MISMATCH]``
        且永远无法自愈（两文件都在，SAN 检查不触发重签）。
        """
        try:
            cert = subprocess.run(
                [_openssl(), "x509", "-in", str(self.server_cert), "-noout", "-modulus"],
                capture_output=True, timeout=30,
            )
            key = subprocess.run(
                [_openssl(), "rsa", "-in", str(self.server_key), "-noout", "-modulus"],
                capture_output=True, timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return (
            cert.returncode == 0
            and key.returncode == 0
            and bool(cert.stdout)
            and cert.stdout == key.stdout
        )

    def ensure_server_cert(self, sans: List[str], force: bool = False) -> Path:
        """签发（或重签）服务器证书。

        证书已存在且未 force 时做 SAN 漂移检测：现有证书缺少任一请求的 SAN
        （或根本没有 SAN 扩展，如旧版本遗留证书）即自动重签——Chrome 报
        NET::ERR_CERT_COMMON_NAME_INVALID 多因访问地址不在 SAN 中。
        key/cert 不配对（重签中断/部分同步所致）也自动重签，否则 werkzeug
        启动即抛 KEY_VALUES_MISMATCH。
        """
        if not sans:
            raise CertError("server cert requires at least one SAN (IP or hostname)")
        self.ensure_ca()
        if self.server_cert.is_file() and self.server_key.is_file() and not force:
            if not self._key_cert_match():
                logger.warning(
                    "monitor: server key/cert mismatch (interrupted regen or partial "
                    "sync); re-signing both"
                )
            else:
                current = {self._san_value(e) for e in self.current_sans()}
                wanted = {str(s).strip() for s in sans if str(s).strip()}
                missing = wanted - current
                if not missing:
                    return self.server_cert
                logger.warning(
                    "monitor: server cert SAN drift detected (missing %s, cert has %s); re-signing",
                    sorted(missing), sorted(current) or "<none>",
                )
        self.dir.mkdir(parents=True, exist_ok=True)
        csr = self.dir / "server.csr"
        _run(
            [
                _openssl(), "req", "-newkey", "rsa:2048", "-nodes",
                "-keyout", str(self.server_key),
                "-out", str(csr),
                "-subj", SERVER_SUBJ,
            ]
        )
        ext_path = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", suffix=".cnf", dir=str(self.dir), delete=False
            ) as fp:
                fp.write(self._san_config(sans))
                ext_path = Path(fp.name)
            _run(
                [
                    _openssl(), "x509", "-req",
                    "-in", str(csr),
                    "-CA", str(self.ca_cert),
                    "-CAkey", str(self.ca_key),
                    "-CAcreateserial",
                    "-out", str(self.server_cert),
                    "-days", str(SERVER_DAYS),
                    "-extensions", "v3_req",
                    "-extfile", str(ext_path),
                ]
            )
        finally:
            if ext_path is not None:
                try:
                    ext_path.unlink()
                except OSError:
                    pass
            try:
                csr.unlink()
            except OSError:
                pass
        _chmod_600(self.server_key)
        logger.info("monitor: server cert signed for SAN %s", sans)
        return self.server_cert

    @staticmethod
    def _san_value(entry: str) -> str:
        """``openssl -text`` 的 SAN 条目（``DNS:host`` / ``IP Address:1.2.3.4``）→ 裸值。"""
        return entry.split(":", 1)[1].strip() if ":" in entry else entry.strip()

    def current_sans(self) -> List[str]:
        """读取现有服务器证书的 SAN（无证书返回 []）。"""
        if not self.server_cert.is_file():
            return []
        try:
            proc = subprocess.run(
                [_openssl(), "x509", "-in", str(self.server_cert), "-noout", "-text"],
                capture_output=True, timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            return []
        text = proc.stdout.decode("utf-8", errors="replace")
        match = re.search(r"Subject Alternative Name:\s*\n\s*(.+)", text)
        if not match:
            return []
        return [t.strip() for t in match.group(1).split(",") if t.strip()]

    # ------------------------------------------------------------ 状态/下载
    def status(self) -> Dict[str, Any]:
        return {
            "ca_exists": self.ca_cert.is_file(),
            "server_cert_exists": self.server_cert.is_file(),
            "server_sans": self.current_sans(),
            "certs_dir": str(self.dir),
        }

    def ca_cert_bytes(self) -> bytes:
        if not self.ca_cert.is_file():
            raise CertError("CA 尚未生成（public 模式首次启动时自动创建）")
        return self.ca_cert.read_bytes()


def import_guide_text(ca_name: str = "motus-monitor.crt", server: str = "") -> str:
    """Ubuntu 客户端导入内部 CA 的说明（管理后台展示/下载）。

    ``server`` 为部署的实际地址（``host:port``，通常取证书 SAN 的第一个非
    localhost 地址 + public 端口），给出时验证命令里的地址即为实际值。
    """
    addr = server or "<服务器IP或主机名>:<端口>"
    return f"""\
Ubuntu 导入 Motus Monitor 内部 CA 的步骤
========================================

1. 从管理后台下载 ca.pem（内部 CA 的公钥证书，不含私钥）。

2. 复制到系统信任库并刷新（需要 sudo）：

     sudo cp ca.pem /usr/local/share/ca-certificates/{ca_name}
     sudo update-ca-certificates

   成功后大多数命令行工具（curl、git、Python requests 等）与
   Chrome/Chromium（读取系统信任库）即信任本服务。

3. 验证（本服务地址：{addr}）：

     curl --cacert /dev/stdin https://{addr}/api/health < ca.pem
   或导入后直接：
     curl https://{addr}/api/health

   确认 CA 主体信息（应显示 O=Ubtech / OU=chen.liao /
   emailAddress=chen.liao@ubtrobot.com）：

     openssl x509 -in ca.pem -noout -subject

   Chrome 仍报"不安全"时按错误码排查：
   - NET::ERR_CERT_AUTHORITY_INVALID：CA 未导入或未生效（重做完第 2 步后
     完全退出 Chrome 再开）。
   - NET::ERR_CERT_COMMON_NAME_INVALID：访问的 IP/域名不在服务器证书的
     SAN 里。地址栏"不安全"→ 证书无效 → 详细信息里应能看到
     "Subject Alternative Name" 一项；没有该字段或缺少当前地址时，到管理
     后台"访问地址"添加该地址并重签证书（server 启动时也会自动检测 SAN
     缺失并重签），然后重启主站。

4. Firefox 使用独立信任库，需单独导入：
   设置 → 隐私与安全 → 证书 → 查看证书 → 证书颁发机构 → 导入 → 选择 ca.pem，
   勾选"信任由此证书颁发机构标识的网站"。

5. 卸载：删除 /usr/local/share/ca-certificates/{ca_name} 后再执行
   sudo update-ca-certificates --fresh。

也可直接执行管理后台导出的 import-ca.sh（等价于第 2 步）。
"""


def import_script(ca_name: str = "motus-monitor.crt") -> str:
    """可执行的 Ubuntu 导入脚本（管理后台下载）。"""
    return f"""\
#!/bin/bash
# Motus Monitor 内部 CA 导入脚本（Ubuntu）。用法：
#   chmod +x import-ca.sh && sudo ./import-ca.sh /path/to/ca.pem
set -euo pipefail
CA_SRC="${{1:-ca.pem}}"
if [ ! -f "$CA_SRC" ]; then
  echo "usage: sudo $0 /path/to/ca.pem" >&2
  exit 1
fi
install -m 0644 "$CA_SRC" "/usr/local/share/ca-certificates/{ca_name}"
update-ca-certificates
echo "CA installed: /usr/local/share/ca-certificates/{ca_name}"
"""
