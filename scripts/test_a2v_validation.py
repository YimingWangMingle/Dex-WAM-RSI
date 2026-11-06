"""CPU contract tests; execute real evaluator functions with a fake inference model.

AST loading excludes optional plotting/VLM imports; no GPU model is simulated
as a quality test. The inference adapter, metrics and RNG handling are real.
"""
import ast
import json
import tempfile
from datetime import timedelta
import torch.distributed as dist
import torch.multiprocessing as mp
import logging
import random
import sys
import types
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "third_party/Thinker-WM/train/sample.py"
module = types.ModuleType("sample")
module.__dict__.update(torch=torch, F=F, np=np, random=random,
                      logger=logging.getLogger("test"), Dict=dict, Tuple=tuple, List=list, Image=NS(Image=object),
                      Optional=__import__("typing").Optional, _VAL_VIZ_SAMPLES=4)
functions = [n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef)]
exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), module.__dict__)
module._history_inputs = lambda batch, device: batch.get("history_inputs")
sys.modules["sample"] = module
sys.path.insert(0, str(ROOT / "third_party/Thinker-WM/train"))
import distributed_validation as dv


def config(a2v=True):
    return NS(common=NS(num_video_frames=12, action_chunk_size=24, video_action_freq_ratio=2),
              system=NS(val_seed=20261021, val_a2v=a2v), training=NS(batch_size=1),
              model=NS(wm=NS(ar_packed_teacher_forcing=True, ar_latents_per_block=1),
                       inference=NS(num_inference_timesteps=10, guidance_scale=5)))


def batch():
    return dict(first_frame=torch.zeros(1, 3, 4, 4),
                video_frames=torch.zeros(1, 12, 3, 4, 4),
                action_sequence=torch.full((1, 24, 14), 2.25),
                initial_state=torch.ones(1, 14), language_embedding=None, vlm_inputs=None,
                history_inputs={"history_recent_state": torch.ones(1, 5, 14)})


class Model(torch.nn.Module):
    def __init__(self, fail_a2v=False):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))
        self.config = NS(enable_depth=False, temporal_mode="block_causal")
        self.calls = []
        self.fail_a2v = fail_a2v

    def inference_step(self, **kwargs):
        self.calls.append(kwargs)
        assert kwargs["num_blocks"] == 3 and kwargs["guidance_scale"] == 5
        assert "condition_video_frames" not in kwargs
        assert "video_frames" not in kwargs
        torch.rand(())  # Each arm consumes RNG; tests verify isolation.
        video = torch.zeros(1, 3, 12, 4, 4)
        if "condition_actions" in kwargs:
            if self.fail_a2v:
                raise RuntimeError("A2V injected failure")
            video[:, :, :4] = 0.1
            video[:, :, 4:8] = 0.2
            video[:, :, 8:] = 0.3
            return video, None  # No action score is valid for fixed commands.
        return video, torch.zeros(1, 24, 14)


def evaluate(model, batches, enabled=True):
    return module.evaluate_model(model, batches, None, config(enabled),
                                 num_eval_batches=len(batches), visualize=False)



class Pipeline:
    def __init__(self, rank, world, total):
        self.total_samples = total
        self.global_ids = list(range(rank, total, world))

    def set_epoch(self, epoch):
        assert epoch == 0
        self.cursor = 0

    def next(self):
        assert self.cursor < len(self.global_ids)
        self.cursor += 1
        return batch(), {}


def worker(rank, world, rendezvous, folder, total, fail):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=world,
                            init_method="file://" + rendezvous,
                            timeout=timedelta(seconds=30))
    dv.create_video_grid = lambda *a, **k: None
    model = Model(fail_a2v=fail and rank == 2)
    pipe = Pipeline(rank, world, total)
    acc = NS(process_index=rank, num_processes=world, unwrap_model=lambda m: m)
    before = torch.get_rng_state().clone()
    try:
        metrics, saved = dv.evaluate_distributed(model, pipe, acc, config())
        assert not fail
        assert len(model.calls) == len(pipe.global_ids) * 2
        assert len(saved) <= 4 and (rank == 0 or not saved)
        result = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    except RuntimeError as exc:
        assert fail and "rank 2" in str(exc) and "A2V injected failure" in str(exc), str(exc)
        result = {"error_shared": True}
    assert model.training and torch.equal(before, torch.get_rng_state())
    Path(folder, f"{rank}.json").write_text(json.dumps(result))
    dist.destroy_process_group()


def run_gloo(world, total, fail=False):
    with tempfile.TemporaryDirectory() as folder:
        mp.spawn(worker, args=(world, folder + "/rdzv", folder, total, fail),
                 nprocs=world, join=True)
        results = [json.loads(Path(folder, f"{rank}.json").read_text()) for rank in range(world)]
        assert all(row == results[0] for row in results)
        return results[0]


def main():
    model, data = Model(), batch()
    before = torch.get_rng_state().clone()
    # Valid-mask depth excludes padding/invalid pixels from codec MSE.
    pred = torch.ones(1, 12, 3, 4, 4)
    target = torch.zeros_like(pred)
    mask = torch.zeros(1, 12, 4, 4, dtype=torch.bool)
    mask[:, :, 0, 0] = True
    pred[:, :, :, 1:, :] = 99
    assert module.compute_depth_metrics(pred, target, mask)["depth_mse_valid"] == 1.0
    try:
        module.compute_depth_metrics(pred, target, torch.zeros_like(mask))
    except ValueError:
        pass
    else:
        raise AssertionError("empty depth mask accepted")
    # Task identity comes from episode metadata even if instruction wording changes.
    assert dv._window_identity({"episode_ids": ["robodojo:test"],
                               "t5_input_messages": ["changed punctuation!"]},
                              {"robodojo:test": "insert_tubes"})[2] == "insert_tubes"
    metrics, saved = evaluate(model, [data])
    assert model.training and torch.equal(before, torch.get_rng_state())
    assert len(model.calls) == 2
    joint, a2v = model.calls
    assert "condition_actions" not in joint
    assert torch.equal(a2v["condition_actions"], data["action_sequence"])
    assert a2v["condition_actions"].dtype == torch.float32
    assert a2v["history_inputs"] is data["history_inputs"]
    assert a2v["predict_action"] is True
    assert abs(metrics["a2v_video_mse"] - 0.14 / 3) < 1e-6
    for i, expected in enumerate((0.01, 0.04, 0.09), 1):
        assert abs(metrics[f"a2v_video_mse_block_{i}"] - expected) < 1e-6
    assert not any(key.startswith("a2v_action") for key in metrics)
    assert saved[0]["a2v_predicted_frames"].device.type == "cpu"
    comparison = module.a2v_comparison_video(saved)
    assert comparison.shape == (1, 12, 3, 2, 4)
    # Opt-in must leave joint metrics unchanged even across several batches.
    plain, plain_saved = evaluate(Model(), [batch(), batch()], False)
    mixed, _ = evaluate(Model(), [batch(), batch()], True)
    assert plain == {k: v for k, v in mixed.items() if not k.startswith("a2v_")}
    assert "a2v_predicted_frames" not in plain_saved[0]
    for kind in ("length", "nan", "missing_state", "prediction_shape", "prediction_nan", "failure"):
        bad = batch()
        broken = Model(fail_a2v=kind == "failure")
        if kind == "length": bad["action_sequence"] = bad["action_sequence"][:, :-1]
        if kind == "nan": bad["action_sequence"][0, 0, 0] = float("nan")
        if kind == "missing_state": bad["initial_state"] = None
        try:
            if kind.startswith("prediction_"):
                pred = torch.zeros(1, 12, 3, 4, 4)
                if kind == "prediction_shape": pred = pred[:, :-1]
                else: pred[0, 0, 0, 0, 0] = float("nan")
                module.compute_a2v_video_metrics(pred, bad["video_frames"], 3)
            else:
                evaluate(broken, [bad])
        except (ValueError, RuntimeError):
            pass
        else:
            raise AssertionError(f"accepted {kind}")
        assert broken.training and torch.equal(before, torch.get_rng_state())
    # 120 windows / 16 ranks: eight ranks have 8, eight have 7; global mean
    # must weight windows, and reject a missing or duplicated window.
    rows = [{"id": i, "metrics": {"a2v_video_mse": float(i)}}
            for rank in range(16) for i in range(rank, 120, 16)]
    assert dv.aggregate_window_metrics(rows, 120)["a2v_video_mse"] == 59.5
    try:
        dv.aggregate_window_metrics(rows[:-1], 120)
    except RuntimeError:
        pass
    else:
        raise AssertionError("missing window accepted")
    print("PASS raw actions/history/3 blocks/CFG5/no future RGB; four metrics; no A2V action score; opt-in; RNG/mode restore; failures; 120-window aggregation; video layout")


if __name__ == "__main__":
    main()
    assert run_gloo(1, 5) == run_gloo(4, 5)
    run_gloo(4, 2)
    run_gloo(4, 5, True)
    print("PASS real Gloo: serial/distributed metrics match; uneven/empty shards; A2V failure shared on every rank")
