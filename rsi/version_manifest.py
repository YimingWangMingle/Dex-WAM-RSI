"""Versioned candidate closure with generic backend and legacy WAM identities.

backend_version.json is the generic contract. Existing wam_version.json manifests
retain their content and revision; neither format is silently rewritten.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

CLOSURE = ("foundation", "skill_selection", "skills", "tools", "control", "prompts",
           "memory_rules", "dependencies", "weights", "assets", "configuration")
BACKEND_VERSION_FILE = "configuration/backend_version.json"
WAM_VERSION_FILE = "configuration/wam_version.json"
WAM_VERSION_KEYS = ("checkpoint", "checkpoint_sha256", "action_stats_version", "history",
                    "camera_transform", "data_version", "model_config_sha256")


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode()


def digest(value) -> str:
    try:
        from PhysicalRSI_core.infra.storage import digest as up
        return up(value)
    except ImportError:
        return hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _check_relative(name: str) -> Path:
    if not isinstance(name, str):
        raise ValueError("candidate paths must be strings")
    p = Path(name)
    if str(p) != name:
        raise ValueError(f"Use a canonical relative path without aliases: {name!r}")
    if not name or p.is_absolute() or ".." in p.parts or p == Path("."):
        raise ValueError(f"Expected nonempty relative path: {name!r}")
    return p


def verify_harness(h: Mapping[str, Any]) -> str:
    import re
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", h["id"]):
        raise ValueError(f"Invalid identifier: {h['id']!r}")
    src = Path(h["root"])
    if src.is_symlink() or not src.is_dir():
        raise ValueError("Harness root must be a physical directory")
    root = src.resolve()
    if set(h["components"]) != set(CLOSURE):
        raise ValueError("Incomplete F/H component declaration")
    owners = {}
    for kind, entries in h["components"].items():
        if not isinstance(entries, dict) or not entries:
            raise ValueError(f"Empty harness component: {kind}")
        for name, sha in entries.items():
            if name in owners:
                raise ValueError(f"candidate file belongs to multiple components: {name}")
            owners[name] = kind
            declared = root / _check_relative(name)
            if any(part.is_symlink() for part in [declared, *declared.parents] if part != root.parent):
                raise ValueError("Harness inputs must be physical files")
            p = declared.resolve()
            if not p.is_relative_to(root) or not p.is_file() or file_digest(p) != sha:
                raise ValueError(f"Changed or missing harness input: {name}")
    if BACKEND_VERSION_FILE in owners:
        if owners[BACKEND_VERSION_FILE] != 'configuration':
            raise ValueError('backend version must belong to configuration')
        _validate_backend_version(json.loads((root / BACKEND_VERSION_FILE).read_text()))
    try:
        from PhysicalRSI_core.self_harness.artifacts import verify_harness as up
    except ImportError:
        return digest({"components": h["components"]})
    return up(dict(h))


def build_harness(root: str | Path, harness_id: str, files: Mapping[str, Iterable[str]], *, require_wam_version: bool | None = None) -> Dict[str, Any]:
    """Hash the declared files (relative to root) per closure component."""
    root = Path(root)
    files = {k: tuple(v) for k, v in files.items()}
    missing = [k for k in CLOSURE if not files.get(k)]
    if missing or set(files) - set(CLOSURE):
        raise ValueError(f"declare files for every closure component; missing {missing}")
    names = [n for values in files.values() for n in values]
    for name in names:
        _check_relative(name)
    if len(names) != len(set(names)):
        raise ValueError('candidate files must have unique ownership')
    comps = {k: {str(_check_relative(n)): file_digest(root / n) for n in files[k]} for k in CLOSURE}
    if require_wam_version is True and WAM_VERSION_FILE not in comps["configuration"]:
        raise ValueError(f"configuration must include {WAM_VERSION_FILE}")
    if require_wam_version is None and not {BACKEND_VERSION_FILE, WAM_VERSION_FILE} & set(comps['configuration']):
        raise ValueError('configuration must include backend_version.json or wam_version.json')
    h = dict(id=harness_id, root=str(root), components=comps)
    verify_harness(h)
    return h


def write_wam_version(root: str | Path, **fields) -> Path:
    missing = [k for k in WAM_VERSION_KEYS if k not in fields]
    if missing:
        raise ValueError(f"wam_version lacks {missing}")
    p = Path(root) / WAM_VERSION_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(canonical(fields) + b"\n")
    return p


def wam_version(h: Mapping[str, Any]) -> Dict[str, Any]:
    verify_harness(h)
    return json.loads((Path(h["root"]) / WAM_VERSION_FILE).read_text())


def _validate_backend_version(fields: Mapping[str, Any]) -> None:
    if not isinstance(fields, dict):
        raise ValueError('backend version must be a JSON object')
    required = ('backend_revision', 'action_contract', 'control_hz', 'runtime_dependencies')
    if any(k not in fields for k in required):
        raise ValueError('backend version requires revision, action contract, frequency and dependencies')
    for key in ('backend_revision', 'action_contract'):
        if not isinstance(fields[key], str) or not fields[key].strip():
            raise ValueError(f'{key} must be a nonempty string')
    hz = fields['control_hz']
    if isinstance(hz, bool) or not isinstance(hz, (int, float)) or not math.isfinite(hz) or hz <= 0:
        raise ValueError('control_hz must be finite and positive')
    if not isinstance(fields['runtime_dependencies'], dict):
        raise ValueError('runtime_dependencies must declare dependency identities')
    canonical(fields)


def write_backend_version(root: str | Path, **fields) -> Path:
    _validate_backend_version(fields)
    path = Path(root) / BACKEND_VERSION_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(fields) + b"\n")
    return path


def backend_version(h: Mapping[str, Any]) -> Dict[str, Any]:
    verify_harness(h)
    if BACKEND_VERSION_FILE not in h['components']['configuration']:
        raise ValueError('no generic backend version declared; legacy WAM metadata is not converted implicitly')
    return json.loads((Path(h['root']) / BACKEND_VERSION_FILE).read_text())
