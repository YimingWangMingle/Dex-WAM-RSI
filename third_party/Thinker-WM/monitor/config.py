"""monitor 独立服务的配置文件加载（server 侧；训练侧 client 不依赖本模块）。

配置文件为 JSON（``monitor serve --config monitor_server.json``），所有键都有
默认值且**默认值即当前部署状态**——不带任何新参数时行为与旧版完全一致。
CLI 参数优先于配置文件。

配置结构（全部可选）::

    {
      "host": "127.0.0.1",            // 本机/开发模式监听地址
      "port": 8377,
      "data_dir": null,               // 缺省走 paths.default_data_dir() 探测
      "remotes": [                    // 同步远端列表（多远端合并进同一 data_root/runs/）
        {
          "name": "h200",             // 状态文件 .sync/<name>.json；缺省 "remote<N>"
          "remote": "user@host:/abs/path",   // 或 local:///path
          "ssh_args": "-J jump -p 10023",
          "interval": 2.0,
          "preview": true             // dataset 预览 helper 挂到该远端（缺省第一个 ssh 远端）
        }
      ],
      "dataset": {                    // 集群侧预览默认值=现有 DEFAULT_DATASET_*
        "enabled": true,
        "local": false,               // true = 本机 helper（开发/冒烟，不起 ssh）
        "repo": "/code1/llm_team/chen.liao/ThinkerWM/thinkerwm-pretrain",
        "python": "/home/fengodchen/miniconda3/envs/motus311/bin/python",
        "catalog_root": null,         // 缺省：ssh 模式 DEFAULT_DATASET_CATALOG_ROOT，
                                      // local 模式 micro_data/catalogs（若存在）
        "catalogs": [],               // ["name:path", ...]
        "workers": 2
      },
      "public": {                     // 对外 HTTPS 主站（局域网部署）
        "enabled": false,
        "host": "0.0.0.0",
        "port": 8443,
        "san": ["10.0.0.8", "thinker_wm.monitor.lan"]   // 服务器证书 SAN（IP/DNS），也是 Host 白名单
      },
      "admin": {                      // 管理后台（仅本机）
        "enabled": false,             // public 开启时默认跟随开启
        "host": "127.0.0.1",
        "port": 8378
      }
    }
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 集群侧预览默认值（与 cli.py 既有默认保持一致）
DEFAULT_DATASET_REPO = "/code1/llm_team/chen.liao/ThinkerWM/thinkerwm-pretrain"
DEFAULT_DATASET_PYTHON = "/home/fengodchen/miniconda3/envs/motus311/bin/python"
DEFAULT_DATASET_CATALOG_ROOT = "/data/jiyuan.wang/EpisodeCatalogs"

DEFAULTS: Dict[str, Any] = {
    "host": "127.0.0.1",
    "port": 8377,
    "data_dir": None,
    "remotes": [],
    "dataset": {
        "enabled": True,
        "local": False,
        "repo": DEFAULT_DATASET_REPO,
        "python": DEFAULT_DATASET_PYTHON,
        "catalog_root": None,  # ssh 模式缺省 DEFAULT_DATASET_CATALOG_ROOT；local 模式缺省 micro_data/catalogs
        "catalogs": [],
        "workers": 2,
    },
    "public": {
        "enabled": False,
        "host": "0.0.0.0",
        "port": 8443,
        "san": [],
    },
    "admin": {
        "enabled": False,
        "host": "127.0.0.1",
        "port": 8378,
    },
}


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并：override 中的 dict 与 base 深合并，其余直接覆盖。"""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: Optional[str]) -> Dict[str, Any]:
    """加载配置文件并与默认值合并；path 为 None 时返回纯默认。"""
    cfg = copy.deepcopy(DEFAULTS)
    if not path:
        return cfg
    raw = Path(str(path)).read_text(encoding="utf-8")
    user = json.loads(raw)
    if not isinstance(user, dict):
        raise ValueError(f"config must be a JSON object: {path}")
    return _merge(cfg, user)


def apply_cli_overrides(cfg: Dict[str, Any], args: Any) -> Dict[str, Any]:
    """把 serve 子命令的显式 CLI 参数覆盖到配置（None/缺省值不覆盖）。

    兼容旧用法：``--remote/--remote-target+--remote-dir`` 生成单条 remotes 记录；
    旧参数存在时忽略配置文件的 remotes（避免两份真相）。
    """
    cfg = copy.deepcopy(cfg)

    def _given(name: str) -> Any:
        return getattr(args, name, None)

    if _given("host") is not None:
        cfg["host"] = args.host
    if _given("port") is not None:
        cfg["port"] = int(args.port)
    if _given("data_dir"):
        cfg["data_dir"] = args.data_dir

    remote = getattr(args, "remote", None)
    target = getattr(args, "remote_target", None)
    remote_dir = getattr(args, "remote_dir", None)
    if not remote and target and remote_dir:
        remote = f"{target}:{remote_dir}"
    if remote:
        cfg["remotes"] = [
            {
                "name": "remote0",
                "remote": remote,
                "ssh_args": getattr(args, "ssh_args", "") or "",
                "interval": float(getattr(args, "interval", 2.0) or 2.0),
                "preview": True,
            }
        ]

    dataset = cfg["dataset"]
    if getattr(args, "no_dataset", False):
        dataset["enabled"] = False
    if getattr(args, "dataset_local", False):
        dataset["local"] = True
    if _given("dataset_repo") is not None:
        dataset["repo"] = args.dataset_repo
    if _given("dataset_python") is not None:
        dataset["python"] = args.dataset_python
    if _given("dataset_catalog_root") is not None:
        dataset["catalog_root"] = args.dataset_catalog_root
    if getattr(args, "dataset_catalog", None):
        dataset["catalogs"] = list(args.dataset_catalog)
    if _given("dataset_workers") is not None:
        dataset["workers"] = int(args.dataset_workers)

    public = cfg["public"]
    if getattr(args, "public", False):
        public["enabled"] = True
    if _given("public_host") is not None:
        public["host"] = args.public_host
    if _given("public_port") is not None:
        public["port"] = int(args.public_port)
    if getattr(args, "san", None):
        public["san"] = list(args.san)

    admin = cfg["admin"]
    if getattr(args, "admin", False):
        admin["enabled"] = True
    if _given("admin_port") is not None:
        admin["port"] = int(args.admin_port)
    if public["enabled"] and not admin.get("enabled"):
        # 对外服务没有本机管理面就没法初始化账户/证书：public 开启时默认跟随开启
        admin["enabled"] = True
    return cfg


def normalize_remotes(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """返回规范化的远端列表（补 name/interval 缺省，保证 preview 至多一个）。"""
    remotes: List[Dict[str, Any]] = []
    seen_preview = False
    for i, item in enumerate(cfg.get("remotes") or []):
        entry = {
            "name": str(item.get("name") or f"remote{i}"),
            "remote": str(item.get("remote") or "").strip(),
            "ssh_args": str(item.get("ssh_args") or ""),
            "interval": float(item.get("interval", 2.0) or 2.0),
            "preview": bool(item.get("preview", False)),
        }
        if not entry["remote"]:
            logger.warning("config: remotes[%d] missing 'remote', skipped", i)
            continue
        if entry["preview"]:
            if seen_preview:
                entry["preview"] = False
            else:
                seen_preview = True
        remotes.append(entry)
    return remotes
