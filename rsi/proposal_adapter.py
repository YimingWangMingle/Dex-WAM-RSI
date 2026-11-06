"""Organise execution and optional prediction evidence for the upstream proposer and materialise candidates.

``evidence_bundle`` summarises a development result (upstream
``ExperimentEvaluator.development`` output) plus our trajectory logs:
outcomes by class, recovery decisions and prediction-error statistics.

``materialize_candidate`` turns a concrete edit set into a new physical
harness directory and manifest. Only harness-editable components may change;
weights/model configuration, the verifier and test cases are outside reach.
"""
from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import math
import statistics

from rsi.version_manifest import CLOSURE, build_harness, digest, verify_harness

EDITABLE = ("skill_selection", "skills", "tools", "control", "prompts", "memory_rules")


def evidence_bundle(development: Mapping[str, Any], trajectories: Iterable[Path] = ()) -> Dict[str, Any]:
    rows = development["episodes"]
    outcomes = Counter(r["outcome"] for r in rows)
    failures = [dict(task=r["task"], case=r["case"], outcome=r["outcome"]) for r in rows
                if r["outcome"] != "success"]
    errors, decisions = [], Counter()
    reviews, handoffs, stops = Counter(), Counter(), Counter()
    for t in trajectories:
        with Path(t).open(encoding='utf-8') as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        for rec in records:
            if rec.get('event') == 'review' and rec.get('decision'):
                reviews[rec['decision']] += 1
            if rec.get('event') == 'handoff' and rec.get('from_skill') is not None and rec.get('to_skill'):
                handoffs[rec['to_skill']] += 1
            if rec.get('event') == 'execution' and rec.get('stop_reason'):
                stops[rec['stop_reason']] += 1
            if rec.get("role") == "error" and rec["features"].get("valid_visual"):
                errors.append(rec["features"]["visual_mse_max"])
            if rec.get("role") == "decision":
                decisions[rec["action"]] += 1
    return dict(schema="thinkerwam.evidence/v1", evaluator_revision=development["evaluator_revision"],
                outcomes=dict(outcomes), failures=failures, recovery_decisions=dict(decisions),
                review_decisions=dict(reviews), skill_handoffs=dict(handoffs), execution_stop_reasons=dict(stops),
                visual_mse=None if not errors else dict(mean=statistics.fmean(errors),
                                                        p90=_quantile(errors, 0.9),
                                                        n=len(errors)))


def _quantile(values, q):
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo))


def materialize_candidate(parent: Mapping[str, Any], edits: Mapping[str, Any], *, new_id: str,
                          out_root: str | Path, new_files: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """edits: {relative path: bytes | str | JSON value}; new_files: {relative path: component}."""
    import re
    if not isinstance(new_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", new_id):
        raise ValueError('invalid candidate identifier')
    from rsi.version_manifest import _check_relative
    for name in (*edits, *(new_files or {})):
        _check_relative(name)
    verify_harness(parent)
    src = Path(parent["root"])
    dst = Path(out_root) / new_id
    if dst.exists() or dst.is_symlink():
        raise FileExistsError(dst)
    owner = {name: kind for kind, files in parent["components"].items() for name in files}
    for name, kind in (new_files or {}).items():
        if kind not in EDITABLE:
            raise ValueError(f'new file component {kind!r} is not editable')
        if name not in edits:
            raise ValueError(f'new file {name!r} requires explicit content')
        if name in owner:
            raise ValueError(f"{name} already declared")
        owner[name] = kind
    for name in edits:
        kind = owner.get(name)
        if kind is None:
            raise ValueError(f"edit {name} is not a declared harness file")
        if kind not in EDITABLE:
            raise ValueError(f"component {kind!r} is frozen for Harness-RSI ({name})")
    # Reject file/directory collisions before creating any candidate output.
    for name in owner:
        if any(str(parent) in owner for parent in Path(name).parents if parent != Path('.')):
            raise ValueError(f'candidate file/directory collision: {name}')
    files: Dict[str, list] = {k: [] for k in CLOSURE}
    for name, kind in owner.items():
        target = dst / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if name in edits:
            v = edits[name]
            data = v if isinstance(v, bytes) else (v.encode() if isinstance(v, str)
                                                   else (json.dumps(v, sort_keys=True, indent=1) + "\n").encode())
            target.write_bytes(data)
        else:
            shutil.copyfile(src / name, target)
        files[kind].append(name)
    from rsi.version_manifest import WAM_VERSION_FILE
    h = build_harness(dst, new_id, files,
        require_wam_version=WAM_VERSION_FILE in parent["components"]["configuration"])
    for kind in CLOSURE:
        if kind not in EDITABLE and h['components'][kind] != parent['components'][kind]:
            raise ValueError(f'candidate changed frozen component {kind!r}')
    (dst.parent / f"{new_id}.proposal.json").write_text(json.dumps(dict(
        parent_id=parent["id"], parent_sha256=verify_harness(parent), candidate_sha256=verify_harness(h),
        edited=sorted(edits), edits_sha256=digest({k: (v.decode("latin1") if isinstance(v, bytes) else v)
                                                   for k, v in sorted(edits.items())})), indent=1))
    return h
