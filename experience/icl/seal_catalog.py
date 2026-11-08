"""Publish prepared ICL payloads into a new, immutable native catalog."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from uuid import uuid4

from experience.packages import build_package, load_package, write_catalog

from .runtime.memory_pool import MemoryPool
from .runtime.task_knowledge import TaskKnowledge


def _payload_for(source_root: Path, task: str) -> tuple[Path, dict]:
    if not task or task.startswith(".") or "/" in task or "\\" in task or Path(task).name != task:
        raise ValueError(f"invalid task directory name: {task!r}")
    task_dir = source_root / task
    if task_dir.is_symlink() or not task_dir.is_dir():
        raise ValueError(f"expected a task directory: {task_dir}")
    if (task_dir / "manifest.json").exists():
        original = load_package(task_dir)
        if original.package_id != task:
            raise ValueError(f"{task}: source package ID is {original.package_id!r}")
        return original.payload, original.metadata
    return task_dir, {"task": task, "channels": ["icl"]}


def seal_catalog(source_root: Path, output: Path, revision: str, tasks=None) -> Path:
    """Read <task>/ICL.md, pool/, knowledge/ or a verified native <task>/payload/."""
    source_root = source_root.resolve(strict=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"choose a new version directory: {output}")
    output = output.resolve()
    if not source_root.is_dir() or output.is_relative_to(source_root):
        raise ValueError("choose a source directory and an output outside that directory")
    if not revision.strip():
        raise ValueError("supply a nonempty catalog revision")
    if tasks is None:
        tasks = [p.name for p in source_root.iterdir() if p.is_dir() and not p.name.startswith(".")]
    names = sorted(set(tasks))
    if not names:
        raise ValueError(f"no task directories under {source_root}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.building-{uuid4().hex}"
    staging.mkdir()
    try:
        packages, entries, mapping = [], {}, {}
        for task in names:
            source, metadata = _payload_for(source_root, task)
            target = staging / "packages" / task
            package = build_package(source, target, task, metadata=metadata)
            if "ICL.md" not in package.files or not (package.payload / "ICL.md").read_text(encoding="utf-8").strip():
                raise ValueError(f"{task}: missing or empty ICL.md")
            pool = MemoryPool(package.payload / "pool")
            knowledge = TaskKnowledge(package.payload / "knowledge")
            if (knowledge.task != task or knowledge.action_contract != pool.action_contract
                    or not pool.records or any(r.task != task or r.channel != "icl" for r in pool.records)):
                raise ValueError(f"{task}: incompatible task, channel, or action contract")
            packages.append(package)
            entries[task] = ["ICL.md"]
            mapping[task] = dict(bundle=task, kind="original_preparation", base_task=task,
                                 reason="Prepared task-specific ICL demonstrations.")
        write_catalog(staging / "catalog.json", revision, "packages", packages, entries, mapping)
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"version directory appeared during preparation: {output}")
        staging.rename(output)
    finally:
        if staging.is_dir() and not staging.is_symlink() and staging.resolve().parent == output.parent:
            shutil.rmtree(staging)
    return output / "catalog.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path, help="directory of prepared task payloads or packages")
    parser.add_argument("--output", required=True, type=Path, help="new directory in shared storage")
    parser.add_argument("--revision", required=True, help="published catalog revision")
    parser.add_argument("--task", "-t", action="append", help="task to include; repeatable; default: all task directories")
    args = parser.parse_args()
    print(seal_catalog(args.source_root, args.output, args.revision, args.task))


if __name__ == "__main__":
    main()
