"""Real CPU holdout/history/depth input smoke, with inference replaced by a recorder."""
import os
import sys
from pathlib import Path
from types import SimpleNamespace as NS
import torch
from omegaconf import OmegaConf
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "third_party/Thinker-WM/train"), str(ROOT)]
import sample
from distributed_validation import _catalog_tasks, _window_identity
from thinker_wm.data_prepare.finite_validation import FiniteValidationPipeline
from thinker_wm.data_prepare.history_attach import build_history_attacher
from thinker_wm.utils.vlm_utils import build_vlm_task_message, resolve_vlm_prompt_style

class Recorder(torch.nn.Module):
    def __init__(self, batch, config):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.config = NS(enable_depth=True, depth_condition_mode="rgb_only", temporal_mode="block_causal")
        self.batch = batch
        self.cfg = config
        self.calls = []

    def inference_step(self, **kw):
        self.calls.append(kw)
        assert kw["num_blocks"] == 3 // int(self.cfg.model.wm.ar_latents_per_block) and kw["guidance_scale"] == 5
        assert kw["depth_first"] is None
        assert "condition_video_frames" not in kw and "condition_depth_frames" not in kw
        assert kw["history_inputs"]["history_recent_rgb"].shape[0] == 1
        expected = build_vlm_task_message(self.batch["t5_input_messages"][0],
                                         style=resolve_vlm_prompt_style(self.cfg))
        assert kw["vlm_input_messages"] == [expected]
        video = self.batch["video_frames"].permute(0, 2, 1, 3, 4).clone()
        actions = self.batch["action_sequence"].clone()
        if "condition_actions" in kw:
            assert torch.equal(kw["condition_actions"], actions)
            return video, None
        if kw.get("return_depth"):
            return video, actions, torch.zeros_like(video)
        return video, actions

config = OmegaConf.load(sys.argv[1])
config.system.val_depth = True
config.system.val_a2v = True
pipe = FiniteValidationPipeline(config, rank=0, world_size=1, seed=int(config.training.seed))
history = build_history_attacher(config)
tasks = _catalog_tasks(config)
expected_tasks = set(getattr(config.system, "val_tasks", ("insert_tubes", "pour_liquid_into_cup", "stack_bowls")))
seen = set()
try:
    for offset, index in enumerate(pipe.indices):
        selection = pipe.dataset.sample_pool.generate(index)
        if tasks[selection['episode_id']] in seen:
            continue
        pipe.cursor = offset
        batch, _ = pipe.next()
        task = _window_identity(batch, tasks)[2]
        batch = history.attach(batch, step=0)
        model = Recorder(batch, config)
        metrics, saved = sample.evaluate_model(model, [batch], None, config,
                                               num_eval_batches=1, expected_batch_size=1,
                                               visualize=False, max_saved_samples=1)
        assert len(model.calls) == 2 and model.training
        assert saved[0]["vlm_input_messages"] == model.calls[0]["vlm_input_messages"][0]
        assert saved[0]["vlm_input_messages"] != saved[0]["original_vlm_input_messages"]
        assert tuple(saved[0]["predicted_depth"].shape) == (12, 3, 480, 640)
        assert tuple(saved[0]["depth_valid_mask"].shape) == (12, 480, 640)
        assert metrics["a2v_video_mse"] == 0
        assert 0 <= metrics["depth_mse_valid"] <= 1
        seen.add(task)
        print("PASS",task,"actual prompt/raw24actions/history/depth labels/mask/no GT conditioning",flush=True)
        if seen == expected_tasks:
            break
    assert seen == expected_tasks
finally:
    pipe.close()
