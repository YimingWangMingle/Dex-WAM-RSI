"""Server-side read model: scan the data root into runs / trees / annotations.

Everything here is read-only except :func:`put_annotation`, which writes the
per-checkpoint benchmark annotations produced by the local web UI (those are
then pushed back to the cluster by the sync engine).
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .paths import annotation_key, sanitize_name
from .store import atomic_write_json, read_json

_SAFE_KEY = re.compile(r"^[A-Za-z0-9._-]+$")


class Registry:
    """Thin read model over ``<data_root>/runs`` and ``<data_root>/annotations``."""

    def __init__(self, data_root: object):
        self.data_root = Path(str(data_root))
        self.runs_dir = self.data_root / "runs"
        self.annotations_dir = self.data_root / "annotations"

    # ------------------------------------------------------------------ runs
    def list_runs(self) -> List[Dict[str, Any]]:
        runs: List[Dict[str, Any]] = []
        if not self.runs_dir.is_dir():
            return runs
        for run_dir in sorted(self.runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            doc = read_json(run_dir / "run.json", default=None)
            if not isinstance(doc, dict):
                continue
            status = read_json(run_dir / "status.json", default={}) or {}
            runs.append(self._run_summary(run_dir.name, doc, status))
        runs.sort(key=lambda r: r.get("created_at") or "", reverse=True)
        return runs

    @staticmethod
    def _run_summary(run_id: str, doc: Dict[str, Any], status: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "run_id": doc.get("run_id", run_id),
            "run_name": doc.get("run_name"),
            "created_at": doc.get("created_at"),
            "last_start_at": doc.get("last_start_at"),
            "state": status.get("state") or doc.get("state") or "unknown",
            "last_step": status.get("last_step"),
            "epoch": status.get("epoch"),
            "max_steps": status.get("max_steps") or doc.get("max_steps"),
            "world_size": doc.get("world_size"),
            "last_total_loss": status.get("last_total_loss"),
            "last_checkpoint_step": status.get("last_checkpoint_step"),
            "status_ts": status.get("ts"),
            "init_phase": status.get("init_phase"),
            "init_elapsed_seconds": status.get("init_elapsed_seconds"),
            "attempt_started_ts": status.get("attempt_started_ts"),
            "lineage_base_step": status.get("lineage_base_step"),
            "restarts": doc.get("restarts", 0),
            "parent": doc.get("parent"),
            "num_checkpoints": len(doc.get("checkpoints") or []),
        }

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        if not _SAFE_KEY.match(run_id or ""):
            return None
        run_dir = self.runs_dir / run_id
        doc = read_json(run_dir / "run.json", default=None)
        if not isinstance(doc, dict):
            return None
        status = read_json(run_dir / "status.json", default={}) or {}
        detail = dict(doc)
        detail["status"] = status
        detail["streams"] = self._streams(run_dir)
        return detail

    @staticmethod
    def _streams(run_dir: Path) -> Dict[str, Any]:
        streams: Dict[str, Any] = {}
        for name in ("metrics", "events", "eval"):
            sub = run_dir / name
            if not sub.is_dir():
                continue
            size = 0
            chunks = 0
            for path in sub.glob("*.jsonl"):
                try:
                    size += path.stat().st_size
                    chunks += 1
                except OSError:
                    continue
            streams[name] = {"chunks": chunks, "bytes": size}
        system_dir = run_dir / "system"
        if system_dir.is_dir():
            nodes = sorted(p.name for p in system_dir.iterdir() if p.is_dir())
            size = 0
            for path in system_dir.rglob("*.jsonl"):
                try:
                    size += path.stat().st_size
                except OSError:
                    continue
            streams["system"] = {"nodes": nodes, "bytes": size}
        return streams

    # ------------------------------------------------------------------ tree
    def build_tree(self) -> Dict[str, Any]:
        """Run -> checkpoint forest following parent links.

        A run whose ``parent.run_id`` matches another known run becomes a
        child of that run (node annotated with the parent checkpoint step);
        everything else is a root. Each run node carries its checkpoints and
        whether each checkpoint already has benchmark annotations.
        """
        runs = {r["run_id"]: r for r in self.list_runs()}
        details: Dict[str, Dict[str, Any]] = {}
        annotated = {a["key"] for a in self.list_annotations()}

        def node_for(run_id: str) -> Dict[str, Any]:
            if run_id not in details:
                detail = self.get_run(run_id) or {}
                ckpts = []
                for ckpt in detail.get("checkpoints") or []:
                    if not isinstance(ckpt, dict) or ckpt.get("step") is None:
                        continue
                    key = annotation_key(run_id, ckpt["step"])
                    ckpts.append(
                        {
                            "step": ckpt.get("step"),
                            "path": ckpt.get("path"),
                            "created_at": ckpt.get("created_at"),
                            "key": key,
                            "has_annotation": key in annotated,
                        }
                    )
                summary = runs.get(run_id, {})
                details[run_id] = {
                    "run_id": run_id,
                    "run_name": summary.get("run_name"),
                    "state": summary.get("state"),
                    "created_at": summary.get("created_at"),
                    "last_step": summary.get("last_step"),
                    "max_steps": summary.get("max_steps"),
                    "parent": detail.get("parent") or summary.get("parent"),
                    "checkpoints": ckpts,
                    "children": [],
                }
            return details[run_id]

        roots: List[Dict[str, Any]] = []
        for run_id in runs:
            node = node_for(run_id)
            parent = node.get("parent") or {}
            parent_run = parent.get("run_id")
            if parent_run and parent_run in runs and parent_run != run_id:
                parent_node = node_for(parent_run)
                parent_node["children"].append(node)
            else:
                roots.append(node)
        return {"roots": roots, "generated_at": time.time()}

    # ------------------------------------------------------------ annotations
    def _annotation_path(self, key: str) -> Optional[Path]:
        if not _SAFE_KEY.match(key or ""):
            return None
        return self.annotations_dir / f"{key}.json"

    def list_annotations(self) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        if not self.annotations_dir.is_dir():
            return items
        for path in sorted(self.annotations_dir.glob("*.json")):
            doc = read_json(path, default=None)
            if isinstance(doc, dict):
                doc.setdefault("key", path.stem)
                items.append(doc)
        return items

    def get_annotation(self, key: str) -> Optional[Dict[str, Any]]:
        path = self._annotation_path(key)
        if path is None:
            return None
        doc = read_json(path, default=None)
        if isinstance(doc, dict):
            doc.setdefault("key", key)
            return doc
        return None

    def put_annotation(
        self,
        run_id: str,
        step: int,
        benchmarks: Dict[str, Any],
        notes: str = "",
        updated_by: str = "user",
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create/replace the annotation document for one checkpoint."""
        key = annotation_key(run_id, step)
        path = self._annotation_path(key)
        if path is None:
            raise ValueError(f"invalid run_id for annotation: {run_id!r}")
        # Link back to the checkpoint path when the run is known locally.
        ckpt_path = None
        detail = self.get_run(run_id)
        for ckpt in (detail or {}).get("checkpoints") or []:
            if isinstance(ckpt, dict) and int(ckpt.get("step", -1)) == int(step):
                ckpt_path = ckpt.get("path")
                break
        doc = {
            "key": key,
            "ckpt": {
                "run_id": run_id,
                "step": int(step),
                "path": ckpt_path,
            },
            "benchmarks": benchmarks or {},
            "notes": notes or "",
            "updated_by": updated_by,
            "updated_at": time.time(),
        }
        if extra:
            doc["extra"] = extra
        atomic_write_json(path, doc)
        return doc

    @staticmethod
    def make_key(run_id: str, step: int) -> str:
        return annotation_key(sanitize_name(run_id), step)
