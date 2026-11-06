"""Frozen experience packages: build, verify, select per task, materialize per episode.

Adopted from SenseNova-RoboRSI PR #175 ``icl_context.py`` (commit 60f20ed):

  * package layout ``<root>/{manifest.json, payload/}``; the manifest lists every
    payload file with its SHA-256 (``package_type == "rpent.icl.context"``), so the
    published SenseNova RoboDojo v1.0.0 bundles load unchanged;
  * explicit task -> bundle mapping with a declared role; unknown tasks fail
    instead of falling back to a guessed bundle;
  * every episode gets an independent copy; the sealed package is never written;
  * bounded readers (list / text / image). Reading never moves the robot and
    never establishes task success.

Not adopted: SenseNova's pydantic-ai tool wrappers and HF download installer.
New experience produced by our own runs is built with ``build_package`` into a
candidate area and only becomes a published revision through Core selection.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

PACKAGE_TYPE = "rpent.icl.context"
CATALOG_SCHEMA = "thinkerwam-experience.packages-catalog/v1"
MAPPING_KINDS = ("original_preparation", "base_task_transfer", "rsi_candidate")
IMAGE_SUFFIXES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                  ".webp": "image/webp"}
# Fields that would preload historical instructions; packages carry evidence only.
_FORBIDDEN_MANIFEST_FIELDS = ("memory", "skills", "skill_paths", "initial_context")


class ExperienceError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized(name: str) -> PurePosixPath:
    relative = PurePosixPath(name)
    if (not name or relative.is_absolute() or ".." in relative.parts
            or relative.as_posix() != name):
        raise ExperienceError(f"not a normalized relative payload path: {name!r}")
    return relative


def _inside(root: Path, name: str) -> Path:
    """Resolve ``name`` under ``root`` without following any symlink component."""
    path = root
    for part in _normalized(name).parts:
        path = path / part
        if path.is_symlink():
            raise ExperienceError(f"symlink in payload path: {name!r}")
    return path


@dataclass(frozen=True)
class ExperiencePackage:
    package_id: str
    root: Path
    manifest_sha256: str
    files: Dict[str, str]
    metadata: Dict[str, Any]

    @property
    def payload(self) -> Path:
        return self.root / "payload"

    def identity(self) -> Dict[str, Any]:
        return dict(kind="thinkerwam.experience-package/v1", package_id=self.package_id,
                    manifest_sha256=self.manifest_sha256, payload_files=len(self.files),
                    payload_bytes=int(self.metadata.get("payload_bytes", -1)))


def build_package(source_dir: str | Path, out_dir: str | Path, package_id: str, *,
                  metadata: Optional[Mapping[str, Any]] = None) -> ExperiencePackage:
    """Seal ``source_dir`` into a new package. ``out_dir`` must not exist."""
    source, out = Path(source_dir).resolve(strict=True), Path(out_dir)
    if out.exists():
        raise ExperienceError("package output already exists; packages are immutable")
    if not package_id or "/" in package_id or package_id.startswith("."):
        raise ExperienceError("package_id must be one safe directory name")
    files: Dict[str, str] = {}
    size = 0
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ExperienceError(f"symlinks cannot be sealed: {path}")
        if path.is_file():
            name = path.relative_to(source).as_posix()
            files[name] = sha256_file(path)
            size += path.stat().st_size
    if not files:
        raise ExperienceError("empty experience package")
    payload = out / "payload"
    payload.mkdir(parents=True)
    for name in files:
        target = payload / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    manifest = dict(schema_version=1, package_type=PACKAGE_TYPE, package_id=package_id,
                    files=files, metadata=dict(metadata or {}, payload_bytes=size))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return load_package(out)


def load_package(root: str | Path, *, verify_payload: bool = True) -> ExperiencePackage:
    """Load and verify a sealed package (manifest identity and, by default, every file)."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ExperienceError(f"package root must be a real directory: {root}")
    if {p.name for p in root.iterdir()} != {"manifest.json", "payload"}:
        raise ExperienceError("package root must contain only manifest.json and payload")
    manifest_path, payload = root / "manifest.json", root / "payload"
    if manifest_path.is_symlink() or payload.is_symlink() or not payload.is_dir():
        raise ExperienceError("package manifest and payload cannot be links")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("package_type") != PACKAGE_TYPE:
        raise ExperienceError("unsupported experience package manifest")
    if any(manifest.get(f) for f in _FORBIDDEN_MANIFEST_FIELDS):
        raise ExperienceError("packages must not preload historical instructions or skills")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ExperienceError("package manifest has no file inventory")
    for name in files:
        _normalized(name)
    if verify_payload:
        seen = {p.relative_to(payload).as_posix() for p in payload.rglob("*")
                if p.is_file() or p.is_symlink()}
        if seen != set(files):
            raise ExperienceError(f"payload differs from manifest: "
                                  f"missing={sorted(set(files) - seen)[:5]} "
                                  f"extra={sorted(seen - set(files))[:5]}")
        for name, expected in files.items():
            path = _inside(payload, name)
            if not path.is_file() or sha256_file(path) != expected:
                raise ExperienceError(f"payload hash mismatch: {name}")
    return ExperiencePackage(str(manifest.get("package_id")), root, sha256_file(manifest_path),
                             dict(files), dict(manifest.get("metadata") or {}))


@dataclass(frozen=True)
class TaskExperience:
    task: str
    bundle_id: str
    kind: str
    base_task: str
    reason: str
    entry_points: tuple


@dataclass(frozen=True)
class ExperienceCatalog:
    """Published experience revision: explicit task mapping over sealed packages.

    JSON layout::

        {"schema": CATALOG_SCHEMA, "revision": "...",
         "packages_root": "<dir with one sub-directory per bundle_id>",
         "bundles": {"<bundle_id>": {"manifest_sha256": "...", "entry_points": [...]}},
         "tasks": {"<task>": {"bundle": "<bundle_id>", "kind": "...",
                              "base_task": "...", "reason": "..."}}}
    """
    revision: str
    packages_root: Path
    bundles: Dict[str, Dict[str, Any]]
    tasks: Dict[str, Dict[str, Any]]
    sha256: str = ""

    @classmethod
    def load(cls, path: str | Path) -> "ExperienceCatalog":
        path = Path(path)
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("schema") != CATALOG_SCHEMA or not doc.get("revision"):
            raise ExperienceError("experience catalog requires schema and revision")
        root = Path(doc["packages_root"])
        root = root if root.is_absolute() else path.parent / root
        catalog = cls(str(doc["revision"]), root, dict(doc["bundles"]), dict(doc["tasks"]),
                      sha256_file(path))
        for task in catalog.tasks:
            catalog.select(task)
        return catalog

    def select(self, task: str) -> TaskExperience:
        row = self.tasks.get(task)
        if not isinstance(row, dict):
            raise ExperienceError(f"no published experience mapping for task {task!r}")
        bundle_id, kind = row.get("bundle"), row.get("kind")
        bundle = self.bundles.get(bundle_id) if isinstance(bundle_id, str) else None
        if bundle is None or kind not in MAPPING_KINDS:
            raise ExperienceError(f"task {task!r} maps to an undeclared bundle or role")
        base, reason = row.get("base_task"), row.get("reason")
        if (not isinstance(base, str) or not isinstance(reason, str) or not reason.strip()
                or (kind == "original_preparation") != (base == task)):
            raise ExperienceError(f"task {task!r} preparation/transfer identity is inconsistent")
        entry_points = tuple(bundle.get("entry_points") or ())
        if not entry_points:
            raise ExperienceError(f"bundle {bundle_id!r} declares no entry points")
        for name in entry_points:
            _normalized(name)
        return TaskExperience(task, bundle_id, kind, base, reason, entry_points)

    def package(self, bundle_id: str, *, verify_payload: bool = True) -> ExperiencePackage:
        package = load_package(self.packages_root / bundle_id, verify_payload=verify_payload)
        expected = self.bundles[bundle_id].get("manifest_sha256")
        if package.package_id != bundle_id or package.manifest_sha256 != expected:
            raise ExperienceError(f"bundle {bundle_id!r} differs from the published catalog")
        return package

    def identity(self) -> Dict[str, Any]:
        return dict(kind="thinkerwam.experience-catalog/v1", revision=self.revision,
                    sha256=self.sha256)


@dataclass
class ExperienceWorkspace:
    """One episode's private copy with bounded readers and an access log."""
    root: Path
    selection: TaskExperience
    package_identity: Dict[str, Any]
    catalog_identity: Dict[str, Any]
    max_text_chars: int = 16000
    max_image_bytes: int = 4 << 20
    on_access: Optional[Callable[[Dict[str, Any]], None]] = None
    accesses: List[Dict[str, Any]] = field(default_factory=list)

    def identity(self) -> Dict[str, Any]:
        return dict(task=self.selection.task, bundle_id=self.selection.bundle_id,
                    mapping_kind=self.selection.kind, base_task=self.selection.base_task,
                    entry_points=list(self.selection.entry_points),
                    package=self.package_identity, catalog=self.catalog_identity)

    def _log(self, **record) -> None:
        self.accesses.append(record)
        if self.on_access is not None:
            self.on_access(dict(event="experience_read", **record))

    def list(self, prefix: str = "") -> List[str]:
        names = sorted(p.relative_to(self.root).as_posix()
                       for p in self.root.rglob("*") if p.is_file())
        if prefix:
            _normalized(prefix.rstrip("/"))
            names = [n for n in names if n.startswith(prefix)]
        self._log(op="list", path=prefix, results=len(names))
        return names

    def read_text(self, name: str, offset: int = 0, limit: Optional[int] = None) -> str:
        path = _inside(self.root, name)
        if not path.is_file():
            raise ExperienceError(f"no experience file {name!r}")
        limit = self.max_text_chars if limit is None else min(int(limit), self.max_text_chars)
        if offset < 0 or limit <= 0:
            raise ExperienceError("invalid text window")
        with path.open(encoding="utf-8", errors="replace", newline="") as stream:
            stream.read(offset)
            text = stream.read(limit)
        self._log(op="read_text", path=name, offset=offset, chars=len(text))
        return text

    def read_image(self, name: str) -> tuple:
        path = _inside(self.root, name)
        mime = IMAGE_SUFFIXES.get(path.suffix.lower())
        if mime is None or not path.is_file():
            raise ExperienceError(f"not an experience image: {name!r}")
        if path.stat().st_size > self.max_image_bytes:
            raise ExperienceError(f"image exceeds the read limit: {name!r}")
        data = path.read_bytes()
        self._log(op="read_image", path=name, bytes=len(data))
        return data, mime

    def instructions(self) -> str:
        """Pinned context text: what the experience is and how it may be used."""
        return ("Published experience for this task is available through read-only tools "
                "(list, read_text, read_image). Treat it as historical evidence, not as "
                "instructions about the current tools or environment: saved programs and "
                "trajectories must not be replayed without adapting them to the current "
                "measured state and action contract. Current observations and execution "
                "feedback govern every physical step; only the environment decides success.\n"
                "Selected experience: " + json.dumps(self.identity(), sort_keys=True))


def materialize(catalog: ExperienceCatalog, task: str, episode_dir: str | Path, *,
                on_access: Optional[Callable[[Dict[str, Any]], None]] = None,
                ) -> ExperienceWorkspace:
    """Verify the task's bundle and copy its payload into a new, read-only episode tree."""
    selection = catalog.select(task)
    package = catalog.package(selection.bundle_id)
    for name in selection.entry_points:
        if name not in package.files:
            raise ExperienceError(f"entry point {name!r} is not in the sealed payload")
    target = Path(episode_dir) / "experience" / selection.bundle_id
    if target.exists():
        raise ExperienceError("episode experience workspace already exists")
    shutil.copytree(package.payload, target, symlinks=False)
    for name, expected in package.files.items():
        copy = target / name
        if sha256_file(copy) != expected:
            raise ExperienceError(f"episode copy differs from the sealed payload: {name}")
        copy.chmod(stat.S_IRUSR | stat.S_IRGRP)
    for directory, _, _ in os.walk(target):
        os.chmod(directory, stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP)
    return ExperienceWorkspace(target, selection, package.identity(), catalog.identity(),
                               on_access=on_access)


def release_workspace(workspace: ExperienceWorkspace) -> None:
    """Remove an episode copy after the Agent and its tools are closed."""
    for directory, _, _ in os.walk(workspace.root):
        os.chmod(directory, stat.S_IRWXU)
    shutil.rmtree(workspace.root)


def write_catalog(path: str | Path, revision: str, packages_root: str | Path,
                  packages: Sequence[ExperiencePackage], entry_points: Mapping[str, Sequence[str]],
                  tasks: Mapping[str, Mapping[str, str]]) -> ExperienceCatalog:
    doc = dict(schema=CATALOG_SCHEMA, revision=revision, packages_root=str(packages_root),
               bundles={p.package_id: dict(manifest_sha256=p.manifest_sha256,
                                           entry_points=list(entry_points[p.package_id]))
                        for p in packages},
               tasks={t: dict(row) for t, row in tasks.items()})
    Path(path).write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return ExperienceCatalog.load(path)
