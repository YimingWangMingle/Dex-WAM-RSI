"""Build a shared DexJoCo ICL release from local rand_full training recordings."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from uuid import uuid4

from .preparation.guidance import TASKS
from .preparation.raw import discover_episodes
from .seal_catalog import seal_catalog


def build_from_raw(data_root: Path, output: Path, *, tasks=None, workers: int = 4) -> Path:
    data_root = Path(data_root).resolve(strict=True)
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"choose a new output directory: {output}")
    output = output.resolve()
    if workers < 1:
        raise ValueError("workers must be positive")
    # Discover before writing so a wrong or incomplete mount fails immediately.
    names = sorted(TASKS if tasks is None else set(tasks))
    if not names or set(names).difference(TASKS):
        raise ValueError(f"expected DexJoCo tasks from {sorted(TASKS)}")
    episodes = discover_episodes(data_root, names)
    for task in names:
        task_root = episodes[task][0][1].parent.resolve()
        if output.is_relative_to(task_root):
            raise ValueError(f"place output outside the input task directories: {task_root}")
    from .preparation.prepare import prepare_task

    output.parent.mkdir(parents=True, exist_ok=True)
    work = output.parent / f".icl-preparing-{uuid4().hex}"
    work.mkdir()
    try:
        payloads = work / "payloads"
        payloads.mkdir()
        for i, task in enumerate(names, 1):
            print(f"[{i}/{len(names)}] {task}: reading {len(episodes[task])} training episodes", flush=True)
            scratch = work / "scratch" / task
            stats = prepare_task(task, episodes[task], scratch, payloads / task,
                                 workers=workers, revision=output.name)
            print(f"  {stats['numerical_steps']} steps, {stats['independent_representatives']} "
                  f"representatives, {stats['retrieved_windows']} windows", flush=True)
        return seal_catalog(payloads, output, output.name, names)
    finally:
        if work.is_dir() and not work.is_symlink() and work.resolve().parent == output.parent:
            shutil.rmtree(work)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path, help="DexJoCo rand_full training data root")
    parser.add_argument("--output", required=True, type=Path, help="new shared release directory")
    parser.add_argument("--workers", type=int, default=4, help="parallel episode readers per task (default: 4)")
    parser.add_argument("--task", action="append", choices=sorted(TASKS), help="repeatable; defaults to all 11 tasks")
    args = parser.parse_args()
    catalog = build_from_raw(args.data_root, args.output, tasks=args.task, workers=args.workers)
    print(f"ICL catalog: {catalog}", flush=True)


if __name__ == "__main__":
    main()
