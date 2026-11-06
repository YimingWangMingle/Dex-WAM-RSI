"""Immutable Data-RSI training manifest from expert data + role-tagged segments.

Rows carry the training mode used by ``models.temporal_training``:
  * ``joint_policy``              -> expert/nominal/recovery or reviewed offline demonstrations
  * ``action_conditioned_world``  -> commanded actions given, only consequences supervised

Sampling weights: ``balanced`` (uniform over rollout rows) or ``error_weighted``
(by failure / prediction error) with the same total rollout budget, so the two
collection strategies can be compared at equal cost. Original demonstrations
are always kept.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from thinker_wm.models.temporal_training import MODE_JOINT, MODE_WORLD
from data.recovery import Segment
from rsi.version_manifest import digest

STRATEGIES = ("balanced", "error_weighted")


def _row(seg: Segment, mode: str, weight: float) -> Dict[str, Any]:
    kind = "offline_recording" if hasattr(seg, "source_record") else "rollout"
    return dict(kind=kind, mode=mode, weight=weight, **seg.to_dict())


def build_manifest(expert_catalogs: Sequence[str], segments: Sequence[Segment], *, strategy: str,
                   rollout_budget: float, parent_checkpoint: str, data_version: str) -> Dict[str, Any]:
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    if any(s.split not in ("train", "development") for s in segments):
        raise ValueError("only train/development data may enter training manifests")
    rows: List[Dict[str, Any]] = []
    raw = []
    for s in segments:
        if s.role == "demonstration" and not getattr(s, "reviewed_demonstration", False):
            raise ValueError("offline demonstrations require explicit reviewed provenance")
        if s.action_supervision and s.role != "demonstration":
            if (getattr(s, "verification_status", "unknown") != "passed"
                    or not getattr(s, "verifier_revision", None)
                    or s.end_step + 1 not in getattr(s, "evidence_steps", ())
                    or s.outcome != "success"):
                raise ValueError("online action supervision requires verified segment provenance")
        if s.action_supervision:
            raw.append((s, MODE_JOINT))
        if s.consequence_supervision:
            raw.append((s, MODE_WORLD))
    if strategy == "balanced":
        w = [1.0] * len(raw)
    else:
        w = [1.0 + (s.outcome != "success") + (s.role == "recovery")
             + (0.0 if s.max_visual_mse is None else min(s.max_visual_mse * 10.0, 2.0)) for s, _ in raw]
    total = sum(w) or 1.0
    for (s, mode), wi in zip(raw, w):
        rows.append(_row(s, mode, rollout_budget * wi / total))
    for r in rows:
        if r["mode"] == MODE_JOINT and r["role"] not in ("nominal", "recovery", "demonstration"):
            raise AssertionError("action supervision on a non-validated segment")
    body = dict(schema="thinkerwam.data-rsi-manifest/v1", strategy=strategy, rollout_budget=rollout_budget,
                parent_checkpoint=parent_checkpoint, data_version=data_version,
                expert_catalogs=list(expert_catalogs), rows=rows)
    return dict(body, manifest_sha256=digest(body))


def write_manifest(manifest: Mapping[str, Any], path: str | Path) -> Path:
    p = Path(path)
    if p.exists():
        raise FileExistsError(f"manifests are immutable: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, sort_keys=True, indent=1) + "\n")
    return p
