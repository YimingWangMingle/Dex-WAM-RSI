#!/usr/bin/env python3
"""End-to-end smoke test for the monitor library (no GPU, no training needed).

Covers:
  1. store layer: chunked JSONL rotation + byte-incremental reads
  2. client: MonitorLogger writes run/metrics/events, NaN sanitization, close
  3. registry: run listing, tree building, checkpoint annotations
  4. sync engine over a local:// remote: pull increments + push annotations
  5. web server: REST endpoints + annotation POST
  6. dataset preview: local-mode helper backend (lifecycle + ops), training
     slice aggregation endpoint (needs micro_data/catalogs; skipped otherwise)

Usage:
    python scripts/smoke_monitor.py [--keep-tmp]
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / "third_party/Thinker-WM/libs" / "motus-monitor"))  # motus_monitor 包

from motus_monitor.client import MonitorLogger, NullMonitor  # noqa: E402
from motus_monitor.registry import Registry  # noqa: E402
from motus_monitor.server import make_server  # noqa: E402
from motus_monitor.store import ChunkedJsonlReader, ChunkedJsonlWriter, atomic_write_json, read_json  # noqa: E402
from motus_monitor.sync import SyncEngine, SyncError, parse_remote  # noqa: E402

CHECKS = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    CHECKS.append((name, condition))
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        raise AssertionError(f"check failed: {name} {detail}")


def http_json(url: str, method: str = "GET", body=None):
    req = urllib.request.Request(url, method=method)
    data = None
    if body is not None:
        req.add_header("Content-Type", "application/json")
        data = json.dumps(body).encode()
    with urllib.request.urlopen(req, data=data, timeout=10) as resp:
        return json.loads(resp.read())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep-tmp", action="store_true", help="keep temp dirs for inspection")
    args = parser.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="motus_monitor_smoke_"))
    remote_root = tmp / "remote"      # simulates the cluster-side data root
    local_root = tmp / "local"        # simulates the laptop-side synced copy
    remote_root.mkdir()
    local_root.mkdir()
    print(f"tmp dir: {tmp}")

    try:
        # ---------------------------------------------------------- 1. store
        chunk_dir = tmp / "chunks"
        writer = ChunkedJsonlWriter(chunk_dir, max_bytes=512)
        for i in range(50):
            writer.write({"i": i, "pad": "x" * 20})
        writer.close()
        n_chunks = len(list(chunk_dir.glob("*.jsonl")))
        check("store rotation creates multiple chunks", n_chunks >= 2, f"chunks={n_chunks}")

        reader = ChunkedJsonlReader(chunk_dir)
        tail = reader.read_tail(max_bytes=256)
        first_lines = [l for l in tail["data"].splitlines() if l.strip()]
        check("store tail returns whole lines only", all(l.startswith("{") for l in first_lines))
        seen = []
        offset = "0:0"
        for _ in range(10):
            out = reader.read_incremental(offset)
            offset = out["offset"]
            if not out["data"]:
                break
            seen.extend(l for l in out["data"].splitlines() if l.strip())
        check("store incremental reads cover all 50 records", len(seen) == 50, f"got {len(seen)}")
        out = reader.read_incremental(offset)
        check("store incremental at end returns empty", out["data"] == "")

        # --------------------------------------------------------- 2. client
        monitor = MonitorLogger(
            run_name="smoke-run",
            checkpoint_dir=None,
            config={"training": {"batch_size": 2}, "nested": {"a": [1, 2, 3]}},
            max_steps=40,
            data_dir=remote_root,
            enabled=True,
            system_interval=0.1,
            status_interval=0.05,
            collect_system=True,
        )
        check("client enabled with run dir", monitor.run_dir is not None and monitor.run_dir.is_dir())
        for step in range(1, 41):
            monitor.log_metrics(
                step,
                {"total_loss": 1.0 / step, "nan_loss": float("nan")},
                step_time_seconds=0.5,
                source_dataset="smoke",
            )
        monitor.log_checkpoint(40, "/ckpts/smoke/checkpoint_step_40", {"checkpoint_total_seconds": 1.2})
        monitor.log_event("custom_event", detail="hello")
        monitor.close("completed")
        check("client wrote metrics chunks", any((monitor.run_dir / "metrics").glob("*.jsonl")))
        check("client wrote events", any((monitor.run_dir / "events").glob("*.jsonl")))
        sys_dirs = list((monitor.run_dir / "system").glob("*")) if (monitor.run_dir / "system").is_dir() else []
        check("client wrote per-node system dir", len(sys_dirs) == 1, f"{sys_dirs}")

        metrics_reader = ChunkedJsonlReader(monitor.run_dir / "metrics")
        records = list(metrics_reader.iter_records())
        check("client 40 step records", len(records) == 40, f"got {len(records)}")
        check("client sanitizes NaN to null", records[0]["metrics"]["nan_loss"] is None)
        raw = (monitor.run_dir / "metrics" / "000000.jsonl").read_text()
        check("client JSONL is strict JSON (no NaN token)", "NaN" not in raw)

        status = json.loads((monitor.run_dir / "status.json").read_text())
        check("client final status has last_step=40", status.get("last_step") == 40, str(status))
        run_doc = json.loads((monitor.run_dir / "run.json").read_text())
        check("client run.json completed + ckpt", run_doc.get("state") == "completed"
              and run_doc["checkpoints"][0]["step"] == 40)

        null = NullMonitor()
        null.log_metrics(1, {"a": 1})
        null.close()
        check("NullMonitor is a no-op", True)

        # ------------------------------------------------------- 3. registry
        registry = Registry(remote_root)
        runs = registry.list_runs()
        check("registry lists the run", len(runs) == 1 and runs[0]["run_id"] == monitor.run_id)
        detail = registry.get_run(monitor.run_id)
        check("registry run detail has config", detail["config"]["training"]["batch_size"] == 2)
        check("registry streams include system nodes", "system" in detail["streams"]
              and len(detail["streams"]["system"]["nodes"]) == 1)
        tree = registry.build_tree()
        check("registry tree has root without parent", len(tree["roots"]) == 1
              and tree["roots"][0]["run_id"] == monitor.run_id)
        check("registry tree exposes ckpt key", tree["roots"][0]["checkpoints"][0]["key"]
              .endswith("__step_40"))

        ann = registry.put_annotation(
            monitor.run_id, 40,
            benchmarks={"simpler_env": {"success_rate": 0.5}},
            notes="smoke", updated_by="smoke_monitor",
        )
        check("registry annotation written", ann["ckpt"]["path"] == "/ckpts/smoke/checkpoint_step_40")
        back = registry.get_annotation(ann["key"])
        check("registry annotation read-back", back["benchmarks"]["simpler_env"]["success_rate"] == 0.5)
        tree2 = registry.build_tree()
        check("registry tree marks annotated ckpt",
              tree2["roots"][0]["checkpoints"][0]["has_annotation"] is True)

        # ------------------------------------------------------------ sync
        backend = parse_remote(f"local://{remote_root}")
        engine = SyncEngine(local_root, backend=backend)
        summary = engine.sync_once()
        check("sync pull ok", summary.get("ok") is True, json.dumps(summary))
        synced_metric = local_root / "runs" / monitor.run_id / "metrics" / "000000.jsonl"
        remote_metric = remote_root / "runs" / monitor.run_id / "metrics" / "000000.jsonl"
        check("sync pulled file bytes identical",
              synced_metric.read_bytes() == remote_metric.read_bytes())

        second = engine.sync_once()
        check("sync second cycle pulls nothing", second.get("pull_bytes") == 0, json.dumps(second))

        # Append more data remotely; next cycle must pull only the increment.
        extra = MonitorLogger(
            run_name="smoke-run",
            checkpoint_dir=None,
            data_dir=remote_root,
            enabled=True,
            collect_system=False,
            run_id=monitor.run_id,
        )
        before = synced_metric.stat().st_size
        extra.log_metrics(41, {"total_loss": 0.01})
        extra.close("completed")
        third = engine.sync_once()
        appended = synced_metric.stat().st_size - before
        check("sync pulls only appended bytes", 0 < third.get("pull_bytes", 0) < 4096
              and appended > 0, json.dumps(third))
        records_after = list(ChunkedJsonlReader(local_root / "runs" / monitor.run_id / "metrics").iter_records())
        check("sync local copy now has 41 records", len(records_after) == 41, f"{len(records_after)}")

        # Local annotation must be pushed back to the remote.
        local_registry = Registry(local_root)
        local_registry.put_annotation(monitor.run_id, 40, {"bench": {"m": 1.0}}, updated_by="local")
        fourth = engine.sync_once()
        pushed_file = remote_root / "annotations" / f"{monitor.run_id}__step_40.json"
        check("sync pushes annotations back", fourth.get("push_files") == 1
              and pushed_file.is_file(), json.dumps(fourth))
        pushed_doc = json.loads(pushed_file.read_text())
        check("sync pushed annotation content", pushed_doc["benchmarks"]["bench"]["m"] == 1.0)
        check("sync data_version bumped", engine.data_version >= 2, f"v={engine.data_version}")

        # SSH wire protocol regression: the remote helper is launched through
        # the REMOTE SHELL, so run the exact command string SshBackend would
        # send through a local shell (covers launcher/base64 quoting).
        from motus_monitor.sync import SshBackend

        ssh_backend = SshBackend("dummy@host", str(remote_root))
        ssh_backend.base_cmd = ["bash", "-c", ssh_backend.base_cmd[-1]]
        ssh_engine = SyncEngine(tmp / "ssh_local", backend=ssh_backend)
        ssh_summary = ssh_engine.sync_once()
        check("ssh remote-cmd shell quoting", ssh_summary.get("ok") is True
              and ssh_summary.get("pull_files", 0) > 0, json.dumps(ssh_summary))
        ssh_metric = tmp / "ssh_local" / "runs" / monitor.run_id / "metrics" / "000000.jsonl"
        check("ssh wire content identical", ssh_metric.read_bytes() == remote_metric.read_bytes())

        # Multi-remote regression: a NAMED engine (serve/sync-once call the
        # first remote "remote0") must adopt the legacy single-remote state
        # file .sync/state.json instead of re-pulling the whole remote.
        legacy_root = tmp / "legacy_local"
        legacy_engine = SyncEngine(legacy_root, backend=parse_remote(f"local://{remote_root}"))
        legacy_engine.sync_once()
        legacy_state = legacy_root / ".sync" / "state.json"
        check("legacy engine wrote state.json", legacy_state.is_file())
        named_engine = SyncEngine(legacy_root, backend=parse_remote(f"local://{remote_root}"),
                                  name="remote0")
        adopted = named_engine.sync_once()
        check("named engine adopts legacy state",
              (legacy_root / ".sync" / "remote0.json").is_file()
              and not legacy_state.exists()
              and adopted.get("pull_files") == 0,
              json.dumps(adopted))

        # Chunked fetch: force tiny chunks so one backlog takes several fetch
        # calls; content must still land byte-identical.
        import motus_monitor.sync as sync_mod

        small_files, small_bytes = sync_mod.FETCH_CHUNK_FILES, sync_mod.FETCH_CHUNK_BYTES
        sync_mod.FETCH_CHUNK_FILES, sync_mod.FETCH_CHUNK_BYTES = 2, 1 << 20
        fetch_calls = []
        phase_seen = []
        chunked_backend = parse_remote(f"local://{remote_root}")
        real_fetch = chunked_backend.fetch
        chunked_engine = None

        def counting_fetch(files):
            fetch_calls.append(len(files))
            # 拉取进行中 engine 必须处于 pulling 相位（前端进度条数据源）
            phase_seen.append(chunked_engine.status.get("phase"))
            return real_fetch(files)

        chunked_backend.fetch = counting_fetch
        try:
            chunked_root = tmp / "chunked_local"
            chunked_engine = SyncEngine(chunked_root, backend=chunked_backend, name="remote0")
            chunked_summary = chunked_engine.sync_once()
        finally:
            sync_mod.FETCH_CHUNK_FILES, sync_mod.FETCH_CHUNK_BYTES = small_files, small_bytes
        chunked_metric = chunked_root / "runs" / monitor.run_id / "metrics" / "000000.jsonl"
        check("chunked fetch splits into multiple calls",
              chunked_summary.get("ok") is True and len(fetch_calls) > 1,
              f"calls={fetch_calls}")
        check("chunked fetch content identical",
              chunked_metric.read_bytes() == remote_metric.read_bytes())
        check("sync progress phase is pulling during fetch",
              bool(phase_seen) and all(p == "pulling" for p in phase_seen),
              f"phases={phase_seen[:3]}")
        check("sync progress counters settle",
              chunked_engine.status["phase"] == "idle"
              and chunked_engine.status["pull_done_bytes"]
              == chunked_engine.status["pull_pending_bytes"]
              and chunked_engine.status["pull_pending_bytes"] > 0,
              json.dumps({k: chunked_engine.status[k] for k in
                          ("phase", "pull_done_bytes", "pull_pending_bytes")}))

        # 中断回归（问题 5 根因）：第 2 个 chunk 的 fetch 失败时，未拉取的
        # 文件绝不能被状态记成已同步（旧 bug 会让这些文件永久缺失）；恢复后
        # 下一周期必须补齐且字节一致；本地文件被删时也要自愈补拉。
        sync_mod.FETCH_CHUNK_FILES, sync_mod.FETCH_CHUNK_BYTES = 2, 1 << 20
        flaky_backend = parse_remote(f"local://{remote_root}")
        real_fetch2 = flaky_backend.fetch
        calls = {"n": 0}

        def flaky_fetch(files):
            calls["n"] += 1
            if calls["n"] == 2:
                raise SyncError("simulated chunk failure")
            return real_fetch2(files)

        flaky_backend.fetch = flaky_fetch
        try:
            flaky_root = tmp / "flaky_local"
            flaky_engine = SyncEngine(flaky_root, backend=flaky_backend, name="remote0")
            bad = flaky_engine.sync_once()
            check("sync chunk failure surfaces", bad.get("ok") is False, json.dumps(bad))
            lying = []
            for rel, ent in flaky_engine._state["files"].items():
                local_file = flaky_root / rel
                local_size = local_file.stat().st_size if local_file.exists() else 0
                if int((ent or {}).get("size", 0)) > local_size:
                    lying.append(rel)
            check("sync state never ahead of local bytes", not lying, ",".join(lying[:3]))
            flaky_backend.fetch = real_fetch2
            healed = flaky_engine.sync_once()
            check("sync resumes after chunk failure", healed.get("ok") is True,
                  json.dumps(healed))
            flaky_metric = flaky_root / "runs" / monitor.run_id / "metrics" / "000000.jsonl"
            check("sync healed content identical",
                  flaky_metric.read_bytes() == remote_metric.read_bytes())
            victim = flaky_root / "runs" / monitor.run_id / "run.json"
            victim.unlink()
            flaky_engine.sync_once()
            check("sync self-heals deleted local file",
                  victim.is_file()
                  and victim.read_bytes()
                  == (remote_root / "runs" / monitor.run_id / "run.json").read_bytes())
        finally:
            sync_mod.FETCH_CHUNK_FILES, sync_mod.FETCH_CHUNK_BYTES = small_files, small_bytes

        # 训练树：父 run 未同步时子 run 平铺为根；父 run 到齐后恢复父子层级
        # （问题 4 的直接验证：残缺是同步不全的后果，build_tree 语义不变）。
        partial = tmp / "tree_partial"
        (partial / "runs").mkdir(parents=True)
        orphan = MonitorLogger(
            run_name="tree-orphan", checkpoint_dir=None, config={}, max_steps=10,
            data_dir=partial, enabled=True, collect_system=False,
        )
        orphan.log_metrics(1, {"total_loss": 0.5})
        orphan.close("completed")
        orphan_doc = read_json(orphan.run_dir / "run.json", default={}) or {}
        orphan_doc["parent"] = {
            "kind": "finetune", "path": "/ckpts/tree-parent/checkpoint_step_10",
            "run_id": "tree-parent__20990101_000000", "step": 10,
        }
        atomic_write_json(orphan.run_dir / "run.json", orphan_doc)
        t1 = Registry(partial).build_tree()
        check("tree orphan child becomes root",
              len(t1["roots"]) == 1 and t1["roots"][0]["run_id"] == orphan.run_id)
        parent_dir = partial / "runs" / "tree-parent__20990101_000000"
        parent_dir.mkdir()
        atomic_write_json(parent_dir / "run.json", {
            "run_id": "tree-parent__20990101_000000", "run_name": "tree-parent",
            "created_at": "2026-01-01T00:00:00", "state": "completed",
            "checkpoints": [{"step": 10, "path": "/ckpts/tree-parent/checkpoint_step_10"}],
        })
        t2 = Registry(partial).build_tree()
        check("tree nests once parent arrives",
              len(t2["roots"]) == 1
              and t2["roots"][0]["run_id"] == "tree-parent__20990101_000000"
              and len(t2["roots"][0]["children"]) == 1
              and t2["roots"][0]["children"][0]["run_id"] == orphan.run_id,
              json.dumps([r["run_id"] for r in t2["roots"]]))

        # ---------------------------------------------------------- server
        server = make_server(local_root, port=0)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        time.sleep(0.2)
        base = f"http://127.0.0.1:{port}"
        try:
            api_runs = http_json(f"{base}/api/runs")["runs"]
            check("server /api/runs", len(api_runs) == 1 and api_runs[0]["state"] == "completed")
            detail = http_json(f"{base}/api/runs/{monitor.run_id}")
            check("server run detail", detail["run_id"] == monitor.run_id)
            tail = http_json(f"{base}/api/runs/{monitor.run_id}/metrics?stream=metrics&tail=1&tail_bytes=2048")
            check("server metrics tail", "total_loss" in tail["data"] and ":" in tail["offset"])
            inc = http_json(f"{base}/api/runs/{monitor.run_id}/metrics?stream=metrics&offset={tail['offset']}")
            check("server metrics incremental", inc["data"] == "")
            system = http_json(f"{base}/api/runs/{monitor.run_id}/metrics?stream=system&tail=1")
            check("server system stream with nodes", "nodes" in system and len(system["nodes"]) == 1,
                  json.dumps(system.get("nodes")))
            events = http_json(f"{base}/api/runs/{monitor.run_id}/metrics?stream=events&tail=1")
            check("server events stream", "custom_event" in events["data"])
            tree_api = http_json(f"{base}/api/tree")
            check("server /api/tree", len(tree_api["roots"]) == 1)
            post = http_json(f"{base}/api/annotations", method="POST", body={
                "ckpt": {"run_id": monitor.run_id, "step": 40},
                "benchmarks": {"libero": {"success": 0.33}},
                "notes": "via api", "updated_by": "smoke",
            })
            check("server POST annotation", post["ok"] is True)
            got = http_json(f"{base}/api/annotations?ckpt={monitor.run_id}__step_40")
            check("server GET annotation", got["benchmarks"]["libero"]["success"] == 0.33)
            status_api = http_json(f"{base}/api/sync/status")
            check("server sync status", status_api["data_root"] == str(local_root))
            help_api = http_json(f"{base}/api/llm/help")
            check("server llm help", len(help_api["endpoints"]) >= 3)
            index = urllib.request.urlopen(f"{base}/", timeout=5).read().decode()
            check("server serves index.html", "<" in index and "html" in index.lower())
        finally:
            server.shutdown()
            server.server_close()

        # ------------------------------------------- 6. dataset preview（可选）
        # 需要本机 micro_data/catalogs（fetch_h200_micro_dataset 产物）；没有则跳过。
        catalogs_dir = Path(__file__).resolve().parent.parent / "micro_data" / "catalogs"
        catalog = (
            next(iter(sorted(catalogs_dir.glob("*.episode_catalog.jsonl"))), None)
            if catalogs_dir.is_dir()
            else None
        )
        if catalog is None:
            print("[SKIP] dataset preview checks -- micro_data/catalogs 不存在")
        else:
            from motus_monitor.dataset_preview import DatasetPreviewBackend

            ds_name = catalog.name[: -len(".episode_catalog.jsonl")]
            backend = DatasetPreviewBackend(
                mode="local",
                catalog_root=None,
                catalogs=[f"{ds_name}:{catalog}"],
                cache_dir=tmp / "ds_cache",
                request_timeout=240,
            )
            backend.start()
            st = backend.status()
            check("dataset backend local mode up", st["running"] and not st["error"],
                  str(st.get("error")))
            helper_pid = st["helper_pid"]
            worker_pids = {w["kind"]: w["pid"] for w in st["workers"]}
            check("dataset backend pool (1 control + 2 heavy)", len(st["workers"]) == 3
                  and len(set(worker_pids.values())) == 3
                  and worker_pids.get("control") is not None, json.dumps(worker_pids))

            payload, _ = backend.request(op="list_datasets")
            check("dataset list_datasets", any(d["name"] == ds_name for d in payload["datasets"]),
                  ds_name)
            payload, _ = backend.request(op="list_episodes", dataset=ds_name, offset=0, limit=2)
            check("dataset list_episodes", payload["total"] > 0 and bool(payload["episode_ids"]))
            eid = payload["episode_ids"][0]
            info, _ = backend.request(op="episode_info", dataset=ds_name, episode_id=eid)
            check("dataset episode_info", info["video_count"] > 0,
                  f"video_count={info['video_count']}")
            meta, blob = backend.request(op="video_frames", dataset=ds_name, episode_id=eid,
                                         start=0, count=2, max_dim=128)
            check("dataset video_frames jpeg", blob[:2] == b"\xff\xd8"
                  and sum(meta["sizes"]) == len(blob))
            meta, blob = backend.request(op="video_mp4", dataset=ds_name, episode_id=eid,
                                         start=0, count=6, fps=4)
            check("dataset video_mp4 ftyp", blob[4:8] == b"ftyp")
            meta, blob = backend.request(op="action_state", dataset=ds_name, episode_id=eid,
                                         max_points=32)
            expect = (len(meta["action_indices"]) * meta["d_action"]
                      + len(meta["state_indices"]) * meta["d_state"]) * 4
            check("dataset action_state blob size", len(blob) == expect, f"{len(blob)} vs {expect}")
            n_sel = min(6, info["video_count"])
            meta, blob = backend.cached_request(
                op="selection_preview", dataset=ds_name, episode_id=eid,
                video_frame_indices=list(range(n_sel)), action_video_fps_ratio=1, max_dim=128)
            check("dataset selection_preview", len(meta["jpeg_sizes"]) == n_sel,
                  f"{len(meta['jpeg_sizes'])} frames")
            check("dataset selection cache file", any((tmp / "ds_cache").glob("*.bin")))

            # 训练切片聚合：补一条带切片元数据的 step 记录并同步
            extra2 = MonitorLogger(
                run_name="smoke-run",
                checkpoint_dir=None,
                data_dir=remote_root,
                enabled=True,
                collect_system=False,
                run_id=monitor.run_id,
            )
            extra2.log_step_record({
                "step": 42,
                "episode_ids": [f"{ds_name}:demo:0", f"{ds_name}:demo:1"],
                "frame_indices": [list(range(0, 49)), list(range(10, 59))],
                "action_video_fps_ratios": [1, 1],
                "mix_source_name": ds_name,
                "source_dataset": "smoke",
            })
            extra2.log_checkpoint(80, "/ckpts/smoke/checkpoint_step_80")
            extra2.close("completed")
            engine.sync_once()

            server2 = make_server(local_root, port=0, dataset_backend=backend)
            port2 = server2.server_address[1]
            thread2 = threading.Thread(target=server2.serve_forever, daemon=True)
            thread2.start()
            time.sleep(0.2)
            base2 = f"http://127.0.0.1:{port2}"
            try:
                st2 = http_json(f"{base2}/api/dataset/status")
                check("server dataset status enabled", st2["enabled"] is True and st2["running"] is True)
                slices = http_json(f"{base2}/api/runs/{monitor.run_id}/slices")
                check("server slices aggregation", slices["total_slices"] == 2
                      and slices["by_source"].get(ds_name, {}).get("count") == 2,
                      json.dumps(slices["by_source"]))
                check("server slices window fields",
                      slices["slices"][0]["window_start"] in (0, 10)
                      and slices["slices"][0]["num_video_frames"] == 48)
                # by_ckpt 分段：ckpt@40 段无切片，ckpt@80 段覆盖 step 42
                check("server slices by_ckpt segments",
                      [c["ckpt_step"] for c in slices["by_ckpt"]] == [80]
                      and slices["by_ckpt"][0]["total"] == 2
                      and slices["by_ckpt"][0]["by_source"].get(ds_name) == 2
                      and slices["by_ckpt"][0]["step_min"] == 41
                      and slices["by_ckpt"][0]["step_max"] == 80,
                      json.dumps(slices["by_ckpt"]))
                steps_only = http_json(f"{base2}/api/runs/{monitor.run_id}/slices?steps_only=1")
                check("server slices steps_only", steps_only["steps"] == [42])
                ranged = http_json(f"{base2}/api/runs/{monitor.run_id}/slices?step_min=1&step_max=41")
                check("server slices step range filter", ranged["total_slices"] == 0)
                exact = http_json(f"{base2}/api/runs/{monitor.run_id}/slices?step_min=42&step_max=42")
                check("server slices exact step filter", exact["total_slices"] == 2)
                packed = urllib.request.urlopen(
                    f"{base2}/api/dataset/frames?dataset={ds_name}&id={urllib.parse.quote(eid)}"
                    "&indices=0,1&max_dim=64", timeout=120).read()
                jl = int.from_bytes(packed[:4], "big")
                check("server packed frames endpoint", packed[4 + jl:4 + jl + 2] == b"\xff\xd8")
                sel = urllib.request.urlopen(
                    f"{base2}/api/dataset/selection?dataset={ds_name}&episode_id={urllib.parse.quote(eid)}"
                    "&start=0&n=5&ratio=1&max_dim=64", timeout=120).read()
                jl = int.from_bytes(sel[:4], "big")
                sel_meta = json.loads(sel[4:4 + jl])
                check("server selection endpoint", len(sel_meta["jpeg_sizes"]) == 6
                      and sel_meta["n_action"] == 5)
                clip = urllib.request.urlopen(
                    f"{base2}/api/dataset/clip?dataset={ds_name}&id={urllib.parse.quote(eid)}"
                    "&start=0&count=6&fps=4", timeout=120).read()
                check("server clip endpoint mp4", clip[4:8] == b"ftyp")
                # stream=1：PRG/LEN 前缀 + packed 成品（urllib 透明处理 chunked）。
                # 用独立参数避开缓存命中（命中时无 PRG 行）
                import re

                streamed = urllib.request.urlopen(
                    f"{base2}/api/dataset/selection?dataset={ds_name}&episode_id={urllib.parse.quote(eid)}"
                    "&start=1&n=5&ratio=1&max_dim=96&stream=1", timeout=120).read()
                m = re.search(rb"(?:^|\n)LEN (\d+)\n", streamed)
                check("server stream=1 framing", m is not None)
                prg_lines = streamed[: m.start()].count(b"PRG ")
                check("server stream=1 emits decode progress", prg_lines >= 1,
                      f"PRG lines={prg_lines}")
                body = streamed[m.end():]
                check("server stream=1 body length", len(body) == int(m.group(1)),
                      f"{len(body)} vs {int(m.group(1))}")
                jl = int.from_bytes(body[:4], "big")
                st_meta = json.loads(body[4:4 + jl])
                check("server stream=1 packed payload", len(st_meta["jpeg_sizes"]) == 6)
                # 轻请求（control）与重渲染（heavy）并发不互堵
                import concurrent.futures

                def _heavy():
                    return backend.cached_request(
                        op="video_mp4", dataset=ds_name, episode_id=eid,
                        start=0, count=8, fps=4)

                def _light():
                    return backend.request(op="list_datasets")

                with concurrent.futures.ThreadPoolExecutor(2) as pool:
                    heavy_future = pool.submit(_heavy)
                    light_future = pool.submit(_light)
                    light_done = light_future.result(timeout=120)
                    heavy_done = heavy_future.result(timeout=240)
                check("dataset pool light/heavy 并发", bool(light_done[0]) and bool(heavy_done[0]))
                busy = [w["kind"] for w in backend.status()["workers"] if w["running"]]
                check("dataset pool workers alive after requests", len(busy) == 3, str(busy))
            finally:
                server2.shutdown()
                server2.server_close()
            backend.close()
            time.sleep(0.5)
            alive = (
                helper_pid
                and subprocess.run(["ps", "-p", str(helper_pid)], capture_output=True).returncode == 0
            )
            check("dataset backend close reaps helper", not alive)

        # --------------------------------------- 7. 认证/白名单/查看根
        from motus_monitor.auth import HostAllowlist, UserStore
        from motus_monitor.server import create_app

        auth_root = tmp / "auth_root"
        auth_root.mkdir()
        store = UserStore(auth_root)
        allowlist = HostAllowlist(auth_root, ["10.0.0.8"])
        auth_client = create_app(
            auth_root, user_store=store, host_allowlist=allowlist
        ).test_client()
        resp = auth_client.get("/api/runs", headers={"Host": "evil.lan"})
        check("public host allowlist blocks unknown host",
              resp.status_code == 403, str(resp.status_code))
        resp = auth_client.post(
            "/register", json={"username": "alice", "password": "secret1"},
            headers={"Host": "10.0.0.8"})
        check("first registered account auto-approved",
              resp.status_code == 200 and resp.get_json().get("pending") is False,
              f"{resp.status_code} {resp.get_json()}")
        resp = auth_client.post(
            "/login", json={"username": "alice", "password": "secret1"},
            headers={"Host": "10.0.0.8"})
        check("first account logs in immediately",
              resp.status_code == 200 and resp.get_json().get("ok") is True,
              f"{resp.status_code} {resp.get_json()}")
        resp = auth_client.post(
            "/register", json={"username": "alice2", "password": "secret1"},
            headers={"Host": "10.0.0.8"})
        check("second registered account stays pending",
              resp.status_code == 200 and resp.get_json().get("pending") is True,
              f"{resp.status_code} {resp.get_json()}")
        resp = auth_client.post(
            "/login", json={"username": "alice2", "password": "secret1"},
            headers={"Host": "10.0.0.8"})
        check("pending account cannot login", resp.status_code == 403,
              str(resp.status_code))
        check("approve enables login", store.approve("alice2") is True)
        resp = auth_client.post(
            "/login", json={"username": "alice2", "password": "secret1"},
            headers={"Host": "10.0.0.8"})
        check("approved account logs in",
              resp.status_code == 200 and resp.get_json().get("ok") is True,
              f"{resp.status_code} {resp.get_json()}")
        resp = auth_client.get("/api/runs", headers={"Host": "10.0.0.8"})
        check("session works on allowed host", resp.status_code == 200,
              str(resp.status_code))
        resp = auth_client.get("/api/runs", headers={"Host": "10.10.29.71"})
        check("host blocked before admin add", resp.status_code == 403,
              str(resp.status_code))
        allowlist.add("10.10.29.71")
        resp = auth_client.post(
            "/login", json={"username": "alice", "password": "secret1"},
            headers={"Host": "10.10.29.71"})
        check("runtime allowlist add takes effect", resp.status_code == 200,
              str(resp.status_code))
        store.register("bob", "secret2")
        check("reject removes pending account",
              store.reject("bob") is True and store.status_of("bob") is None)
        check("allowlist persists added hosts",
              "10.10.29.71" in HostAllowlist(auth_root, ["10.0.0.8"]))

        # 未登录访问控制：除 /login /register /static 外一律 401/302
        anon = create_app(
            auth_root, user_store=store, host_allowlist=allowlist
        ).test_client()
        resp = anon.get("/api/runs", headers={"Host": "10.0.0.8"})
        check("anonymous API blocked", resp.status_code == 401, str(resp.status_code))
        resp = anon.get("/", headers={"Host": "10.0.0.8"})
        check("anonymous page redirected to login",
              resp.status_code == 302 and "/login" in (resp.headers.get("Location") or ""),
              f"{resp.status_code} {resp.headers.get('Location')}")
        resp = anon.get("/login", headers={"Host": "10.0.0.8"})
        check("login page public", resp.status_code == 200, str(resp.status_code))
        resp = anon.get("/register", headers={"Host": "10.0.0.8"})
        check("register page public", resp.status_code == 200, str(resp.status_code))

        # 查看根切换 + 只读目录浏览（无 user_store 的兼容模式：全局切换）
        view_a = tmp / "view_a"
        view_b = tmp / "view_b"
        view_a.mkdir()
        view_b.mkdir()
        (view_b / "runs").mkdir()
        (tmp / "a_secret_file.txt").write_text("top secret", encoding="utf-8")
        view_client = create_app(view_a).test_client()
        resp = view_client.get("/api/browse?path=" + urllib.parse.quote(str(tmp)))
        check("browse lists subdirs",
              resp.status_code == 200
              and any(d["name"] == "view_b" for d in resp.get_json()["dirs"]),
              f"{resp.status_code}")
        check("browse never lists files (read-only tree)",
              all(d["name"] != "a_secret_file.txt" for d in resp.get_json()["dirs"])
              and all(set(d) <= {"name", "hidden", "has_runs_dir"}
                      for d in resp.get_json()["dirs"]),
              json.dumps(resp.get_json())[:200])
        resp = view_client.get(
            "/api/browse?path=" + urllib.parse.quote(str(view_b)))
        check("browse flags runs/ dir",
              resp.status_code == 200 and resp.get_json()["looks_like_data_root"] is True)
        resp = view_client.post("/api/data_root", json={"path": "relative/xx"})
        check("data_root rejects non-absolute", resp.status_code == 400)
        resp = view_client.post("/api/data_root", json={"path": str(tmp / "no_such_dir")})
        check("data_root rejects missing dir", resp.status_code == 400)
        resp = view_client.post("/api/data_root", json={"path": str(view_b)})
        check("data_root switch ok",
              resp.status_code == 200
              and resp.get_json()["data_root"] == str(view_b.resolve()),
              f"{resp.status_code} {resp.get_json()}")
        check("view root persisted", (view_a / ".view_root.json").is_file())
        resp = create_app(view_a).test_client().get("/api/data_root")
        check("view root restored on restart",
              resp.get_json()["data_root"] == str(view_b.resolve()),
              json.dumps(resp.get_json()))

        # run 文件端点：流式 send_file + Range/206（eval 大视频下载路径）；沙箱逃逸拒绝
        run_dir = view_b / "runs" / "r_file"
        (run_dir / "eval").mkdir(parents=True)
        (run_dir / "eval" / "x.bin").write_bytes(b"0123456789" * 100)
        resp = view_client.get("/api/runs/r_file/file?path=eval/x.bin")
        check("run file serves bytes",
              resp.status_code == 200 and len(resp.data) == 1000,
              f"{resp.status_code} {len(resp.data)}")
        check("run file sets download filename",
              "x.bin" in (resp.headers.get("Content-Disposition") or ""),
              resp.headers.get("Content-Disposition") or "<none>")
        resp = view_client.get("/api/runs/r_file/file?path=eval/x.bin",
                               headers={"Range": "bytes=10-19"})
        check("run file honors Range (206)",
              resp.status_code == 206 and resp.data == b"0123456789",
              f"{resp.status_code} {resp.data[:24]!r}")
        resp = view_client.get("/api/runs/r_file/file?path=../../escape.bin")
        check("run file sandbox escape rejected", resp.status_code == 404,
              str(resp.status_code))

        # session cookie 的 Secure 标志按承载scheme区分（HTTP local 模式打了会被拒存）
        plain_app = create_app(view_a, user_store=UserStore(tmp / "plain_auth"))
        check("http local mode cookie not Secure",
              plain_app.config["SESSION_COOKIE_SECURE"] is False)
        secure_app = create_app(view_a, user_store=UserStore(tmp / "sec_auth"),
                                secure_cookies=True)
        check("https public mode cookie Secure",
              secure_app.config["SESSION_COOKIE_SECURE"] is True)

        # 每用户 workspace 隔离：carol 切查看根不影响 dave；重启后按用户恢复
        ws_root = tmp / "ws_root"
        ws_root.mkdir()
        (ws_root / "runs").mkdir()
        ws_store = UserStore(ws_root)
        ws_store.add_user("carol", "secret1")
        ws_store.add_user("dave", "secret2")

        def ws_client():
            return create_app(ws_root, user_store=ws_store).test_client()

        carol = ws_client()
        dave = ws_client()
        for client, name, pw in ((carol, "carol", "secret1"), (dave, "dave", "secret2")):
            resp = client.post("/login", json={"username": name, "password": pw})
            check(f"workspace user {name} logs in", resp.status_code == 200,
                  str(resp.status_code))
        resp = carol.post("/api/data_root", json={"path": str(view_b)})
        check("carol switches own view root", resp.status_code == 200,
              f"{resp.status_code} {resp.get_json()}")
        resp = carol.get("/api/data_root")
        check("carol sees own view root",
              resp.get_json()["data_root"] == str(view_b.resolve()),
              json.dumps(resp.get_json()))
        resp = dave.get("/api/data_root")
        check("dave unaffected by carol's switch",
              resp.get_json()["data_root"] == str(ws_root.resolve()),
              json.dumps(resp.get_json()))
        ws_doc = ws_root / "workspaces" / "carol" / "workspace.json"
        check("carol workspace persisted", ws_doc.is_file())
        carol2 = ws_client()  # 模拟 serve 重启后的新会话
        carol2.post("/login", json={"username": "carol", "password": "secret1"})
        resp = carol2.get("/api/data_root")
        check("carol view root restored per-user after restart",
              resp.get_json()["data_root"] == str(view_b.resolve()),
              json.dumps(resp.get_json()))
        resp = carol2.get("/api/auth/whoami")
        check("whoami reports user and view root",
              resp.get_json().get("user") == "carol"
              and resp.get_json().get("view_root") == str(view_b.resolve()),
              json.dumps(resp.get_json()))

        # 证书 key/cert 不配对自愈（重签中断/部分同步留下不配对文件的场景，
        # 旧行为是 werkzeug 启动即抛 ssl.SSLError KEY_VALUES_MISMATCH 且永不恢复）
        if shutil.which("openssl"):
            import ssl as _ssl

            from motus_monitor.certs import CertManager

            cm = CertManager(tmp / "certs_case")
            cm.ensure_server_cert(["127.0.0.1"])
            check("server cert signed with matching key",
                  cm.server_cert.is_file() and cm._key_cert_match())
            subprocess.run(
                ["openssl", "req", "-newkey", "rsa:2048", "-nodes",
                 "-keyout", str(cm.server_key), "-out", str(tmp / "x.csr"),
                 "-subj", "/CN=x"],
                capture_output=True,
            )
            check("corrupted key/cert pair detected", not cm._key_cert_match())
            cm.ensure_server_cert(["127.0.0.1"])
            healed = cm._key_cert_match()
            if healed:
                try:
                    ssl_ctx = _ssl.SSLContext(_ssl.PROTOCOL_TLS_SERVER)
                    ssl_ctx.load_cert_chain(str(cm.server_cert), str(cm.server_key))
                except _ssl.SSLError:
                    healed = False
            check("mismatched pair self-heals and loads", healed)

        # Final push of the POSTed annotation is left to the daemon in real
        # deployments; here we just note the engine would pick it up.
        print()
        failed = [name for name, ok in CHECKS if not ok]
        print(f"SMOKE MONITOR: {len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
        if args.keep_tmp:
            print(f"kept tmp dir: {tmp}")
        return 0 if not failed else 1
    except Exception:
        traceback.print_exc()
        print(f"\nSMOKE MONITOR FAILED (tmp kept at {tmp})")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
