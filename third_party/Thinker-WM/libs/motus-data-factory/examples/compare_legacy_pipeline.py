#!/usr/bin/env python3
"""Run legacy/new loaders in isolated processes and compare batch signatures."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--legacy-root", required=True)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--child",
        default=None,
        help="Restrict comparison to one named child dataset.",
    )
    args = parser.parse_args()

    helper = Path(__file__).with_name("dump_batch_signatures.py")
    with tempfile.TemporaryDirectory(prefix="motus_data_compare_") as temp_dir:
        files = {}
        for backend in ("legacy", "new"):
            output = Path(temp_dir) / f"{backend}.json"
            command = [
                sys.executable,
                str(helper),
                "--backend", backend,
                "--config", args.config,
                "--output", str(output),
                "--batches", str(args.batches),
                "--batch-size", str(args.batch_size),
                "--seed", str(args.seed),
            ]
            if backend == "legacy":
                command.extend(["--legacy-root", args.legacy_root])
            if args.child is not None:
                command.extend(["--child", args.child])
            print(
                f"starting backend={backend} child={args.child or 'ALL'}",
                flush=True,
            )
            subprocess.run(command, check=True)
            files[backend] = json.loads(output.read_text(encoding="utf-8"))

    legacy_batches = files["legacy"]["batches"]
    new_batches = files["new"]["batches"]
    if len(legacy_batches) != len(new_batches):
        raise RuntimeError(f"batch count mismatch: {len(legacy_batches)} vs {len(new_batches)}")
    mismatches = []
    for batch_index, (legacy, new) in enumerate(zip(legacy_batches, new_batches)):
        if legacy != new:
            mismatches.append({"batch": batch_index, "legacy": legacy, "new": new})
    report = {
        "equivalent": not mismatches,
        "compared_batches": len(legacy_batches),
        "child": args.child,
        "mismatches": mismatches,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if mismatches:
        raise RuntimeError(f"legacy/new batch mismatch in {len(mismatches)} batch(es)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
