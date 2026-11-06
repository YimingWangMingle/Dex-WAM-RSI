"""Command line interface for the Motus monitor.

    python -m motus_monitor serve      [--port 8377] [--data-dir ...]
                                 [--remote user@host:/path | local:///path]
                                 [--ssh-args "-J jump -p 10023"] [--interval 2]
                                 [--config monitor_server.json]
                                 [--public --public-host 0.0.0.0 --public-port 8443
                                  --san 10.0.0.8 --san monitor.lan] [--admin]
    python -m motus_monitor sync-once  (same remote options; single cycle)
    python -m motus_monitor doctor     (check data dir / ssh / remote python3)
    python -m motus_monitor demo-data  (synthesize a demo run for UI development)

所有模式都要求登录（除登录/注册页外）：首个注册账户自动批准，其后注册需
管理后台审批。每个登录用户有独立 workspace（查看路径/自加同步远端互不影响）。
独立服务/局域网部署见 README「独立服务部署」一节：public 模式 = HTTPS + 账户登录，
管理后台只绑 127.0.0.1（证书下载/Ubuntu 导入说明/账户增删/helper 状态与重启）。
"""

from __future__ import annotations

import argparse
import atexit
import json
import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .paths import default_data_dir, repo_root
from .store import read_json
from .sync import SyncDaemon, SyncEngine, parse_remote

logger = logging.getLogger("monitor.cli")

DEFAULT_PORT = 8377

# Dataset 预览（/api/dataset/*）的集群侧默认值
DEFAULT_DATASET_REPO = "/code1/llm_team/chen.liao/ThinkerWM/thinkerwm-pretrain"
DEFAULT_DATASET_PYTHON = "/home/fengodchen/miniconda3/envs/motus311/bin/python"
DEFAULT_DATASET_CATALOG_ROOT = "/data/jiyuan.wang/EpisodeCatalogs"


def _add_remote_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--remote",
        default=None,
        help="Remote monitor data root: 'user@host:/abs/path', 'ssh://host/abs/path' "
        "or 'local:///abs/path'. Omit for pure-local mode.",
    )
    parser.add_argument(
        "--ssh-args",
        default="",
        help="Extra ssh options as one string, e.g. \"-J chen.liao@10.10.94.35 -p 10023\".",
    )
    parser.add_argument(
        "--remote-target",
        default=None,
        help="Alternative to --remote: ssh target only (use with --remote-dir).",
    )
    parser.add_argument(
        "--remote-dir",
        default=None,
        help="Alternative to --remote: remote data dir (use with --remote-target).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Sync poll interval in seconds (default: 2).",
    )


def _resolve_remote(args: argparse.Namespace) -> Optional[str]:
    if getattr(args, "remote", None):
        return args.remote
    target = getattr(args, "remote_target", None)
    remote_dir = getattr(args, "remote_dir", None)
    if target and remote_dir:
        return f"{target}:{remote_dir}"
    return None


def _import_web() -> Any:
    """Flask 相关的 server/admin 模块懒加载（训练侧 client 不依赖 Flask）。"""
    try:
        from .admin import create_admin_app
        from .server import create_app, run_servers
    except ImportError as exc:
        print(
            f"error: monitor web server requires Flask ({exc}); run: pip install flask",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return create_app, create_admin_app, run_servers


def cmd_serve(args: argparse.Namespace) -> int:
    from .config import apply_cli_overrides, load_config, normalize_remotes

    create_app, create_admin_app, run_servers = _import_web()

    cfg = apply_cli_overrides(load_config(getattr(args, "config", None)), args)
    data_dir = Path(cfg["data_dir"]) if cfg["data_dir"] else default_data_dir()

    # ------------------------------------------------------------ 日志落地 + 内存缓冲
    # stderr 输出保留（main 的 basicConfig 已装）；追加滚动文件与 ring buffer。
    from .logbuf import RingBufferHandler

    log_buffer = RingBufferHandler()
    log_file = None
    try:
        from logging.handlers import RotatingFileHandler

        logs_dir = data_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / "server.log"
        file_handler = RotatingFileHandler(
            str(log_file), maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logging.getLogger().addHandler(file_handler)
    except OSError:
        # 数据目录不可写：静默降级为仅 stderr + 内存缓冲
        log_file = None
    logging.getLogger().addHandler(log_buffer)

    # ------------------------------------------------------------ 同步远端
    remotes = normalize_remotes(cfg)
    engines: List[Any] = []
    daemons: List[Any] = []
    for spec in remotes:
        backend = parse_remote(spec["remote"], ssh_args=spec["ssh_args"])
        engine = SyncEngine(data_dir, backend=backend, name=spec["name"])
        daemon = SyncDaemon(engine, interval=spec["interval"])
        daemon.start()
        engines.append(engine)
        daemons.append(daemon)
        logger.info(
            "sync daemon [%s]: %s -> %s (every %.1fs)",
            spec["name"], backend.description, data_dir, spec["interval"],
        )
    if not remotes:
        logger.info("no remote configured; serving local data dir %s", data_dir)

    # 网页端运行时添加的同步远端：旧的全局 .added_remotes.json（拉进启动根）；
    # 各登录用户 workspaces/<user>/workspace.json 里的 added_remotes（拉进各自
    # workspace 的 data/，引擎归该用户所有，serve 重启自动恢复）
    added_remotes = read_json(data_dir / ".added_remotes.json", default=None)
    if isinstance(added_remotes, dict):
        known_names = {spec["name"] for spec in remotes}
        for item in added_remotes.get("remotes") or []:
            if not isinstance(item, dict) or not item.get("remote"):
                continue
            name = str(item.get("name") or "")
            if not name or name in known_names:
                continue
            known_names.add(name)
            try:
                backend = parse_remote(
                    str(item["remote"]), ssh_args=str(item.get("ssh_args") or "")
                )
                engine = SyncEngine(data_dir, backend=backend, name=name)
                daemon = SyncDaemon(engine, interval=float(item.get("interval") or 30.0))
                daemon.start()
                engines.append(engine)
                daemons.append(daemon)
                logger.info("sync daemon [%s]: %s (restored from .added_remotes.json)",
                            name, backend.description)
            except Exception as exc:
                logger.warning("restore added remote %s failed (ignored): %s", name, exc)

    workspaces_dir = data_dir / "workspaces"
    if workspaces_dir.is_dir():
        known_names = {e.name or "" for e in engines}
        for ws_dir in sorted(workspaces_dir.iterdir()):
            doc = read_json(ws_dir / "workspace.json", default=None)
            if not isinstance(doc, dict):
                continue
            for item in doc.get("added_remotes") or []:
                if not isinstance(item, dict) or not item.get("remote"):
                    continue
                name = str(item.get("name") or "")
                if not name or name in known_names:
                    continue
                known_names.add(name)
                try:
                    backend = parse_remote(
                        str(item["remote"]), ssh_args=str(item.get("ssh_args") or "")
                    )
                    engine = SyncEngine(ws_dir / "data", backend=backend, name=name)
                    engine.owner = str(item.get("owner") or ws_dir.name)
                    daemon = SyncDaemon(engine, interval=float(item.get("interval") or 30.0))
                    daemon.start()
                    engines.append(engine)
                    daemons.append(daemon)
                    logger.info(
                        "sync daemon [%s]: %s (restored for user %s)",
                        name, backend.description, engine.owner,
                    )
                except Exception as exc:
                    logger.warning("restore workspace remote %s failed (ignored): %s",
                                   name, exc)

    dataset_backend = _build_dataset_backend(cfg, remotes, data_dir)
    if dataset_backend is not None:
        dataset_backend.start()  # 失败只记 error，不影响 server 其余功能
        atexit.register(dataset_backend.close)

    # ------------------------------------------------------------ 应用集合
    public = cfg["public"]
    admin = cfg["admin"]
    servers: List[Dict[str, Any]] = []
    user_store = None
    cert_manager = None
    # 强制登录：除登录/注册页外全部端点要求会话；首个注册账户自动批准
    from .auth import UserStore

    user_store = UserStore(data_dir)

    if public["enabled"]:
        from .auth import HostAllowlist
        from .certs import CertManager

        cert_manager = CertManager(data_dir)
        cert_manager.ensure_ca()
        sans = [str(s).strip() for s in (public["san"] or []) if str(s).strip()]
        if not sans:
            # 未配置 SAN：只能签 localhost 证书（本机验证用）；局域网访问必须配 --san
            sans = ["127.0.0.1", "localhost"]
            print(
                "warning: public.san 未配置，服务器证书仅对 localhost 有效；\n"
                "         局域网访问请用 --san <服务器IP或主机名> 或配置文件 public.san",
                file=sys.stderr,
            )
        # Host 白名单运行时可由管理后台追加（持久化在 data_dir/access_addresses.json）；
        # 追加过的地址也要并入证书 SAN，否则重启后这些地址又会出现证书警告。
        host_allowlist = HostAllowlist(data_dir, sans)
        cert_manager.ensure_server_cert(host_allowlist.all_hosts())
        app = create_app(
            data_dir,
            sync_engines=engines,
            dataset_backend=dataset_backend,
            user_store=user_store,
            host_allowlist=host_allowlist,
            secure_cookies=True,  # HTTPS 承载才打 Secure；HTTP local 模式打了会被浏览器拒存
            slice_cache_mb=(cfg.get("slices") or {}).get("cache_mb"),
            slice_disk_cache_mb=(cfg.get("slices") or {}).get("disk_cache_mb"),
            cache_idle_s=(cfg.get("cache") or {}).get("idle_s"),
            cache_reap_s=(cfg.get("cache") or {}).get("reap_s"),
            malloc_trim=(cfg.get("cache") or {}).get("malloc_trim"),
        )
        servers.append(
            {
                "name": "public",
                "app": app,
                "host": public["host"],
                "port": int(public["port"]),
                "ssl_context": (
                    str(cert_manager.server_cert),
                    str(cert_manager.server_key),
                ),
            }
        )
    else:
        app = create_app(
            data_dir,
            sync_engines=engines,
            dataset_backend=dataset_backend,
            user_store=user_store,
            slice_cache_mb=(cfg.get("slices") or {}).get("cache_mb"),
            slice_disk_cache_mb=(cfg.get("slices") or {}).get("disk_cache_mb"),
            cache_idle_s=(cfg.get("cache") or {}).get("idle_s"),
            cache_reap_s=(cfg.get("cache") or {}).get("reap_s"),
            malloc_trim=(cfg.get("cache") or {}).get("malloc_trim"),
        )
        servers.append(
            {
                "name": "local",
                "app": app,
                "host": cfg["host"],
                "port": int(cfg["port"]),
                "ssl_context": None,
            }
        )
    if user_store.count() == 0:
        print("note: 还没有账户——打开页面注册首个账户即可使用（自动批准）；"
              "其后注册的账户需管理后台审批")

    if admin.get("enabled"):
        if cert_manager is None:
            from .certs import CertManager

            cert_manager = CertManager(data_dir)
        admin_app = create_admin_app(
            data_dir,
            user_store=user_store,
            cert_manager=cert_manager,
            sync_engines=engines,
            dataset_backend=dataset_backend,
            host_allowlist=host_allowlist if public["enabled"] else None,
            log_buffer=log_buffer,
            log_file=log_file,
            public_info=(
                {
                    "port": int(public["port"]),
                    "sans": list(host_allowlist.all_hosts()) if host_allowlist else [],
                }
                if public["enabled"]
                else None
            ),
        )
        servers.append(
            {
                "name": "admin",
                "app": admin_app,
                "host": admin["host"],
                "port": int(admin["port"]),
                "ssl_context": None,
            }
        )

    # ------------------------------------------------------------ 启动摘要
    for spec in servers:
        scheme = "https" if spec.get("ssl_context") else "http"
        print(f"Motus monitor {spec['name']:6s}: {scheme}://{spec['host']}:{spec['port']}")
    print(f"Data directory: {data_dir}")
    for spec, engine in zip(remotes, engines):
        print(f"Sync [{spec['name']}]: {spec['remote']} (every {spec['interval']}s)")
    print(f"Dataset preview: {_dataset_backend_summary(cfg, remotes, dataset_backend)}")

    if args.open_browser and servers:
        try:
            import webbrowser

            first = servers[0]
            scheme = "https" if first.get("ssl_context") else "http"
            webbrowser.open(f"{scheme}://{first['host']}:{first['port']}")
        except Exception:
            pass
    run_servers(servers, dataset_backend=dataset_backend, daemons=daemons)
    return 0


def _build_dataset_backend(
    cfg: Dict[str, Any], remotes: List[Dict[str, Any]], data_dir: Path
) -> Any:
    """按配置构造 dataset 预览后端（不可用时返回 None）。

    ssh 目标选择：配置里 ``preview: true`` 的远端，否则第一个 ssh 远端。
    """
    ds = cfg["dataset"]
    if not ds.get("enabled", True):
        return None
    from .dataset_preview import DatasetPreviewBackend, ssh_target_from_remote

    cache_dir = data_dir / ".cache" / "dataset"
    workers = max(1, int(ds.get("workers") or 2))
    catalogs = list(ds.get("catalogs") or [])
    if ds.get("local"):
        catalog_root = ds.get("catalog_root")
        if catalog_root is None and not catalogs:
            # 本机开发默认：micro_data/catalogs（若存在）
            micro = repo_root() / "micro_data" / "catalogs"
            if micro.is_dir():
                catalog_root = str(micro)
        return DatasetPreviewBackend(
            mode="local",
            catalog_root=catalog_root,
            catalogs=catalogs,
            workers=workers,
            cache_dir=cache_dir,
        )

    target = None
    ssh_args = ""
    for spec in remotes:
        candidate = ssh_target_from_remote(spec["remote"])
        if candidate and (spec.get("preview") or target is None):
            target, ssh_args = candidate, spec["ssh_args"]
            if spec.get("preview"):
                break
    if target is None:
        return None
    return DatasetPreviewBackend(
        mode="ssh",
        target=target,
        ssh_args=ssh_args,
        remote_python=ds.get("python") or DEFAULT_DATASET_PYTHON,
        remote_repo=ds.get("repo") or DEFAULT_DATASET_REPO,
        catalog_root=ds.get("catalog_root") or DEFAULT_DATASET_CATALOG_ROOT,
        catalogs=catalogs,
        workers=workers,
        cache_dir=cache_dir,
    )


def _dataset_backend_summary(
    cfg: Dict[str, Any], remotes: List[Dict[str, Any]], backend: Any
) -> str:
    if backend is None:
        if not cfg["dataset"].get("enabled", True):
            return "disabled (--no-dataset)"
        return "disabled (need --remote ssh target or --dataset-local)"
    status = backend.status()
    if status.get("error"):
        return f"unavailable: {status['error']}"
    if backend.mode == "local":
        return f"local helper (catalog_root={status.get('catalog_root')})"
    return f"ssh://{backend.target} helper {status.get('helper_path')}"


def cmd_sync_once(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()
    remote = _resolve_remote(args)
    if not remote:
        print("error: sync-once needs --remote (or --remote-target + --remote-dir)", file=sys.stderr)
        return 2
    backend = parse_remote(remote, ssh_args=args.ssh_args)
    # 与 serve 的首个远端同名，共享同一份状态文件（可接管旧版 state.json），
    # 避免手动 sync-once 与服务进程各拉一份、状态发散。
    engine = SyncEngine(data_dir, backend=backend, name="remote0")
    summary = engine.sync_once()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("ok") else 1


def cmd_doctor(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()
    print(f"data dir: {data_dir}")
    print(f"  writable: {data_dir.is_dir() or data_dir.parent.is_dir()}")
    remote = _resolve_remote(args)
    if not remote:
        print("no remote configured; local mode OK")
        return 0
    backend = parse_remote(remote, ssh_args=args.ssh_args)
    print(f"remote: {backend.description}")
    try:
        files = backend.list_files("runs")
        print(f"  ssh + remote python3 OK; {len(files)} files under runs/")
        return 0
    except Exception as exc:
        print(f"  FAILED: {exc}")
        return 1


def cmd_demo_data(args: argparse.Namespace) -> int:
    """Write a synthetic run so the web UI can be developed without training."""
    from .client import MonitorLogger

    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()
    seed = 20260724
    rng = random.Random(seed)

    run_name = args.name or "demo-stage2"
    monitor = MonitorLogger(
        run_name=run_name,
        checkpoint_dir=None,  # force a fresh run id each time
        config={
            "common": {"action_dim": 26, "num_video_frames": 48, "video_height": 480},
            "training": {"batch_size": 12, "max_steps": 300, "learning_rate": 1e-5},
            "dataset": {"type": "mix"},
            "monitor": {"enabled": True},
        },
        max_steps=300,
        data_dir=data_dir,
        enabled=True,
        system_interval=0.2,
        status_interval=0.2,
        collect_system=False,
    )
    steps = int(args.steps)
    sources = ["robotwin", "agibot_beta", "robotmind"]
    ckpt_every = max(1, steps // 3)
    # 每个 ckpt 段的源权重不同，demo 的 ckpt 配比饼图/堆叠条才有可见差异
    seg_weights = [[5, 3, 2], [2, 5, 3], [3, 2, 5]]
    t0 = time.time() - steps * 2.0
    for step in range(1, steps + 1):
        decay = 1.0 / math.sqrt(step + 4)
        seg = min((step - 1) // ckpt_every, len(seg_weights) - 1)
        record = {
            "step": step,
            "epoch": step // 60,
            "metrics": {
                "total_loss": 0.9 * decay + rng.uniform(-0.02, 0.02),
                "video_loss": 0.6 * decay + rng.uniform(-0.015, 0.015),
                "action_loss": 0.3 * decay + rng.uniform(-0.01, 0.01),
            },
            "lr_main": 1e-5 * min(1.0, step / 50),
            "grad_total_norm_before_clip": abs(rng.gauss(0.8, 0.25)),
            "dataloader_next_seconds": rng.uniform(0.02, 0.4),
            "model_training_step_seconds": rng.uniform(0.5, 0.8),
            "backward_seconds": rng.uniform(0.2, 0.4),
            "optimizer_step_seconds": rng.uniform(0.05, 0.12),
            "grad_clip_seconds": rng.uniform(0.01, 0.03),
            "scheduler_step_seconds": rng.uniform(0.001, 0.005),
            "checkpoint_total_seconds": 0.0,
            "validation_seconds": 0.0,
            "step_time_seconds": rng.uniform(0.9, 1.4),
            "loop_total_seconds": rng.uniform(1.0, 1.6),
            "source_dataset": sources[step % len(sources)],
            "source_seen": {s: step // len(sources) for s in sources},
            "ready_queue_get_wait_seconds": rng.uniform(0.0, 0.05),
        }
        # 训练切片元数据（与 train.py step_record 同构），供 /api/runs/{id}/slices
        src = rng.choices(sources, seg_weights[seg])[0]
        record["mix_source_name"] = src
        record["episode_ids"] = [f"{src}:demo:{(step + k) % 7}" for k in range(4)]
        record["frame_indices"] = [
            list(range(10 * ((step + k) % 5), 10 * ((step + k) % 5) + 49)) for k in range(4)
        ]
        record["action_video_fps_ratios"] = [1, 1, 1, 1]
        monitor.log_step_record(record)
        if step % ckpt_every == 0:
            monitor.log_checkpoint(step, f"/demo/checkpoints/{run_name}/checkpoint_step_{step}")
        time.sleep(0.01)
    monitor.set_state("completed")
    monitor.close()

    # A child run "finetuned" from the demo run's step-200 checkpoint.
    child = MonitorLogger(
        run_name=f"{run_name}-ft",
        checkpoint_dir=None,
        config={"training": {"batch_size": 8, "max_steps": 100, "learning_rate": 5e-6}},
        max_steps=100,
        data_dir=data_dir,
        enabled=True,
        system_interval=0.2,
        status_interval=0.2,
        collect_system=False,
    )
    # Patch in the parent link manually (demo has no real checkpoint dirs).
    from .store import atomic_write_json, read_json

    doc = read_json(child.run_dir / "run.json", default={}) or {}
    doc["parent"] = {
        "kind": "finetune",
        "path": f"/demo/checkpoints/{run_name}/checkpoint_step_200",
        "run_id": monitor.run_id,
        "step": 200,
    }
    atomic_write_json(child.run_dir / "run.json", doc)
    for step in range(1, 41):
        child.log_metrics(
            step,
            {
                "total_loss": 0.3 / math.sqrt(step + 2) + rng.uniform(-0.01, 0.01),
                "video_loss": 0.2 / math.sqrt(step + 2),
                "action_loss": 0.1 / math.sqrt(step + 2),
            },
            step_time_seconds=1.1,
            source_dataset="robotwin",
        )
    child.log_checkpoint(40, f"/demo/checkpoints/{run_name}-ft/checkpoint_step_40")
    child.close()

    print(f"demo runs written under {data_dir}/runs")
    print(f"  parent run: {monitor.run_id}")
    print(f"  child run:  {child.run_id} (finetuned from step 200)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m motus_monitor", description=__doc__)
    parser.add_argument("--verbose", "-v", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the web server (+sync daemons)")
    serve.add_argument("--config", default=None,
                       help="JSON 配置文件（独立服务部署；CLI 参数优先于配置文件）")
    serve.add_argument("--port", type=int, default=None, help=f"本机模式端口（默认 {DEFAULT_PORT}）")
    serve.add_argument("--host", default=None, help="本机模式监听地址（默认 127.0.0.1）")
    serve.add_argument("--data-dir", default=None)
    serve.add_argument("--open", dest="open_browser", action="store_true")
    _add_remote_args(serve)
    serve.add_argument(
        "--no-dataset",
        action="store_true",
        help="disable the dataset preview tab backend",
    )
    serve.add_argument(
        "--dataset-local",
        action="store_true",
        help="run the preview helper as a LOCAL subprocess (dev/smoke; no ssh). "
        "Catalogs come from --dataset-catalog, or micro_data/catalogs if present.",
    )
    serve.add_argument(
        "--dataset-repo",
        default=None,
        help="remote repo root giving motus_data to the helper "
        f"(default: {DEFAULT_DATASET_REPO})",
    )
    serve.add_argument(
        "--dataset-python",
        default=None,
        help=f"remote python for the helper (default: {DEFAULT_DATASET_PYTHON})",
    )
    serve.add_argument(
        "--dataset-catalog-root",
        default=None,
        help="Episode Catalog root on the helper side (default: ssh 模式 "
        f"{DEFAULT_DATASET_CATALOG_ROOT}；--dataset-local 模式 micro_data/catalogs 若存在)",
    )
    serve.add_argument(
        "--dataset-catalog",
        action="append",
        default=None,
        metavar="NAME:PATH",
        help="explicit catalog mapping on the helper side (repeatable); overrides discovery",
    )
    serve.add_argument(
        "--dataset-workers",
        type=int,
        default=None,
        help="重渲染 helper 进程数（另加 1 个 control 进程；默认 2）",
    )
    serve.add_argument(
        "--public",
        action="store_true",
        help="对外 HTTPS 主站模式（账户登录 + Host 白名单；需 --san 提供服务器 IP/主机名）",
    )
    serve.add_argument("--public-host", default=None, help="public 模式监听地址（默认 0.0.0.0）")
    serve.add_argument("--public-port", type=int, default=None, help="public 模式端口（默认 8443）")
    serve.add_argument(
        "--san",
        action="append",
        default=None,
        metavar="IP_OR_HOSTNAME",
        help="服务器证书 SAN（可多次），也是 Host 头白名单",
    )
    serve.add_argument(
        "--admin",
        action="store_true",
        help="启用本机管理后台（仅 127.0.0.1；public 模式默认跟随开启）",
    )
    serve.add_argument("--admin-port", type=int, default=None, help="管理后台端口（默认 8378）")
    serve.add_argument(
        "--slice-cache-mb",
        type=float,
        default=None,
        help="切片缓存内存预算 MB（默认 5120；超限 LRU 溢写 .cache/slices/ 磁盘缓存）",
    )
    serve.add_argument(
        "--slice-disk-cache-mb",
        type=float,
        default=None,
        help="切片磁盘溢写缓存容量 MB（默认 4× 内存预算）",
    )
    serve.add_argument(
        "--cache-idle-s",
        type=float,
        default=None,
        help="缓存空闲逐出阈值秒（默认 1800；超龄未访问的缓存条目释放内存，磁盘副本保留，0=禁用）",
    )
    serve.add_argument(
        "--cache-reap-s",
        type=float,
        default=None,
        help="缓存空闲回收周期秒（默认 300）",
    )
    serve.add_argument(
        "--no-malloc-trim",
        action="store_true",
        help="禁用回收后的 glibc malloc_trim（默认开启：把释放的内存还给 OS，RSS 实际回落）",
    )
    serve.set_defaults(func=cmd_serve)

    once = sub.add_parser("sync-once", help="run a single sync cycle and exit")
    once.add_argument("--data-dir", default=None)
    _add_remote_args(once)
    once.set_defaults(func=cmd_sync_once)

    doctor = sub.add_parser("doctor", help="check data dir / ssh / remote python3")
    doctor.add_argument("--data-dir", default=None)
    _add_remote_args(doctor)
    doctor.set_defaults(func=cmd_doctor)

    demo = sub.add_parser("demo-data", help="write a synthetic demo run")
    demo.add_argument("--data-dir", default=None)
    demo.add_argument("--name", default=None)
    demo.add_argument("--steps", type=int, default=300)
    demo.set_defaults(func=cmd_demo_data)
    return parser


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
