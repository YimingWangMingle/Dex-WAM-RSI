#!/usr/bin/env python3
"""Standalone torchrun full-holdout evaluation; no optimizer or training loop.

Launch on each node using the same inherited inference environment as training
(including MOTUS_TEMPORAL_ATTN=flex). torchrun supplies RANK/WORLD_SIZE/LOCAL_RANK.
All ranks score unpadded finite shards; only rank zero publishes artifacts.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "third_party/Thinker-WM/train", PROJECT_ROOT / "third_party/Thinker-WM/libs" / "motus-monitor"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

LOG = logging.getLogger("eval_distributed_holdout")


class EvalAccelerator:
    """Only the accelerator surface used by the validation functions."""
    def __init__(self, rank: int, world_size: int, local_rank: int, device: Any):
        self.process_index = rank
        self.num_processes = world_size
        self.local_process_index = local_rank
        self.is_main_process = rank == 0
        self.is_local_main_process = local_rank == 0
        self.device = device
        self.state = SimpleNamespace(deepspeed_plugin=None)

    @staticmethod
    def unwrap_model(model: Any) -> Any:
        return model


def validate_checkpoint(checkpoint: Path, step: int) -> dict:
    """Require the exact completed training checkpoint before allocating CUDA."""
    if not checkpoint.is_dir():
        raise ValueError("--checkpoint must be a completed checkpoint directory")
    marker = checkpoint / ".motus_checkpoint_complete.json"
    if not marker.is_file():
        raise ValueError(f"checkpoint completion marker missing: {marker}")
    payload = json.loads(marker.read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or payload.get("step") != step:
        raise ValueError("checkpoint must be complete and its recorded step must match --step")
    candidates = ("pytorch_model_0.bin", "pytorch_model.bin", "mp_rank_00_model_states.pt",
                  "pytorch_model/mp_rank_00_model_states.pt")
    if not any((checkpoint / name).is_file() for name in candidates):
        raise FileNotFoundError("completed checkpoint has no supported model weight file")
    return payload


def _sync_error(error: str | None, stage: str) -> None:
    import torch.distributed as dist
    errors = [error]
    if dist.is_initialized():
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error)
    failures = [f"rank {rank}: {message}" for rank, message in enumerate(errors) if message]
    if failures:
        raise RuntimeError(f"{stage} failed: " + "; ".join(failures))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--monitor-run-dir", required=True, help="Existing run containing run.json; media and eval JSONL append here.")
    parser.add_argument("--output-dir", required=True, help="Unique per-invocation summary directories are created here.")
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=1800, help="Collective timeout, including one slow window or rank-zero publication.")
    args = parser.parse_args(argv)
    if args.step < 0 or args.timeout_seconds <= 0:
        parser.error("step must be nonnegative and timeout must be positive")
    checkpoint = Path(args.checkpoint).resolve()
    validate_checkpoint(checkpoint, args.step)
    run_dir = Path(args.monitor_run_dir).resolve()
    if not (run_dir / "run.json").is_file():
        parser.error("--monitor-run-dir must be an existing monitor run with run.json")
    if not Path(args.config).is_file():
        parser.error("--config does not exist")
    for path in (Path(args.output_dir).resolve(), run_dir):
        if path == checkpoint or checkpoint in path.parents:
            parser.error("output paths must be outside the checkpoint directory")

    # Set before importing model, numpy, or torch libraries. Keep attention,
    # allocator, NCCL, and other launch-time configuration inherited intact.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    import torch
    import torch.distributed as dist
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("invalid torchrun rank/world size")
    if not torch.cuda.is_available():
        raise RuntimeError("standalone holdout evaluation requires CUDA")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s rank={rank} %(levelname)s %(message)s")
    if world_size > 1:
        dist.init_process_group("nccl", init_method="env://", rank=rank, world_size=world_size,
                                timeout=timedelta(seconds=args.timeout_seconds))
    pipeline = None
    try:
        setup_error = None
        try:
            from omegaconf import OmegaConf
            from thinker_wm.scripts.eval_video_ckpt import create_eval_model, load_config, _load_model_weights
            from thinker_wm.data_prepare.finite_validation import FiniteValidationPipeline
            from thinker_wm.data_prepare.history_attach import build_history_attacher
            from distributed_validation import evaluate_distributed
            from thinker_wm.periodic_eval.validation_artifacts import discard_staged_media, publish_validation_artifacts

            config = load_config(args.config)
            OmegaConf.set_struct(config, False)
            config.system.val_full_set = True
            config.system.val_distributed = True
            config.system.val_a2v = True
            config.system.val_depth = True
            if int(config.system.val_num_batches) * int(config.training.batch_size) <= 0:
                raise ValueError("this runner requires a nonempty complete RoboDojo holdout")
            if not bool(config.model.wm.enable_depth):
                raise ValueError("depth validation requires the checkpoint's depth-enabled architecture")
            # Pool construction follows training; inference seeds remain
            # system.val_seed + global window ID inside evaluate_distributed.
            seed = int(config.training.seed)
            pipeline = FiniteValidationPipeline(config, rank=rank, world_size=world_size, seed=seed)
            history_attacher = build_history_attacher(config)
            model = create_eval_model(config, load_pretrained_backbones=False)
            load_info = _load_model_weights(model, checkpoint, strict=True)
            if load_info.get("missing_keys") or load_info.get("unexpected_keys"):
                raise RuntimeError(f"checkpoint load was not exact: {load_info}")
            # The architecture owns mixed precision (world model BF16 and
            # inference autocast). Preserve FP32 normalization/history buffers.
            model.to(device=device)
            model.requires_grad_(False)
            model.eval()
            accelerator = EvalAccelerator(rank, world_size, local_rank, device)
            LOG.info("Loaded complete step %d strictly; %d/%d windows; compute dtype=%s device=%s; CPU threads=1",
                     args.step, len(pipeline.global_ids), pipeline.total_samples, model.dtype, device)
        except Exception as exc:
            LOG.exception("Holdout setup failed")
            setup_error = f"{type(exc).__name__}: {exc}"
        _sync_error(setup_error, "setup")
        started = time.perf_counter()
        with torch.inference_mode():
            metrics, saved = evaluate_distributed(model, pipeline, accelerator, config, history_attacher)
        elapsed = time.perf_counter() - started
        publish_error = None
        if rank == 0:
            try:
                summary = publish_validation_artifacts(
                    config, args.step, metrics, saved, run_dir, seconds=elapsed,
                    output_dir=args.output_dir, world_size=world_size, checkpoint=checkpoint)
                LOG.info("Published media for %d/%d windows: %s",
                         summary["num_saved_samples"], summary["num_eval_samples"], summary["files"]["summary"])
            except Exception as exc:
                LOG.exception("Holdout publication failed")
                publish_error = f"{type(exc).__name__}: {exc}"
            finally:
                discard_staged_media(saved)
        _sync_error(publish_error, "publication")
        return 0
    finally:
        try:
            if pipeline is not None:
                pipeline.close()
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
