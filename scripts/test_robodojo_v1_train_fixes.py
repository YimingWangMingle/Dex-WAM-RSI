"""Checkpoint prompt migration and multi-epoch task_cycled_ranges."""
import json
import sqlite3
import tempfile
from pathlib import Path
from types import SimpleNamespace

from thinker_wm.data_prepare.task_cycled_ranges import TaskCycledRangesPool
from test_sampling_ranges_stride import _write_ranges, _StubRepository


class Repository(_StubRepository):
    def iter_episode_infos(self):
        for episode, task in (("short", "A"), ("long1", "B"), ("long2", "B")):
            yield SimpleNamespace(episode_id=episode, metadata={"task_name": task})


def check_cycles():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ranges.sqlite"
        _write_ranges(path, layout="video_stride")
        with sqlite3.connect(path) as db:
            db.row_factory = sqlite3.Row
            template = dict(db.execute("SELECT * FROM episode_ranges LIMIT 1").fetchone())
            db.execute("DELETE FROM episode_ranges")
            for i, (eid, count) in enumerate((("short", 80), ("long1", 160), ("long2", 240))):
                row = dict(template, ordinal=i, episode_id=eid,
                           intervals=json.dumps([[0, count - 1]]), episode_len=count + 100)
                db.execute(
                    "INSERT INTO episode_ranges (" + ",".join(row) + ") VALUES ("
                    + ",".join("?" for _ in row) + ")",
                    tuple(row.values()),
                )
        pool = TaskCycledRangesPool(
            path, Repository(), group_size=8, cycle_base_step=0, seed=20261021,
            action_start_offset=0,
        )
        first = [pool.generate(i)["state_indices"][0] for i in range(24)]
        pool.set_epoch(1)
        second = [pool.generate(i)["state_indices"][0] for i in range(24)]
        assert first != second
        pool.set_epoch(0)
        assert first == [pool.generate(i)["state_indices"][0] for i in range(24)]
        try:
            pool.set_epoch(-1)
            raise AssertionError("negative epoch was accepted")
        except Exception as exc:
            assert "epoch" in str(exc)
        print("PASS cycled ranges continues into epoch 1 with a new shuffle")


def check_resume_math():
    from thinker_wm.data_prepare.task_cycled_ranges import install_exact_resume
    from motus_data.indexed_mix import IndexedMixedTrainingDataPipeline

    class Stream:
        def __init__(self, n):
            self.batches_per_epoch = n
        def set_epoch(self, epoch):
            self.epoch = epoch

    class Pipe:
        queue = None
        def __init__(self, n):
            self.stream = Stream(n)

    install_exact_resume(0)
    for step, n, cycle, offset in ((1444, 1443, 1, 1), (722, 721, 1, 1), (10000, 721, 13, 627), (0, 721, 0, 0)):
        pipe = Pipe(n)
        IndexedMixedTrainingDataPipeline.set_epoch(pipe, step)
        assert (pipe._epoch, pipe.stream._resume_skip_steps) == (cycle, offset), (step, pipe._epoch, pipe.stream._resume_skip_steps)

    class Queue:
        def __init__(self):
            self._epoch = 0
            self.started = False

    queued = Pipe(721)
    queued.queue = Queue()
    IndexedMixedTrainingDataPipeline.set_epoch(queued, 2000)
    assert queued.queue._epoch == 2, queued.queue._epoch
    assert queued.stream.epoch == 2
    assert queued.stream._resume_skip_steps == 2000 % 721
    queued.queue.started = True
    try:
        IndexedMixedTrainingDataPipeline.set_epoch(queued, 2716)
        raise AssertionError("a started queue accepted a new epoch")
    except RuntimeError as exc:
        assert "ready-queue" in str(exc)
    print("PASS resume maps steps onto a cycle and syncs the ready-queue epoch")


def _batch(index, batch_size=2):
    import torch
    return {
        "video_frames": torch.zeros(batch_size, 2, 3, 4, 4),
        "first_frame": torch.zeros(batch_size, 3, 4, 4),
        "initial_state": torch.zeros(batch_size, 14),
        "action_sequence": torch.zeros(batch_size, 2, 14),
        "language_embedding": None,
        "vlm_inputs": None,
        "episode_id": [f"ep-{index}-{i}" for i in range(batch_size)],
        "frame_indices": [[index, index + 1] for _ in range(batch_size)],
    }


class _Pulls:
    def __init__(self, batch_size=2):
        self.n = 0
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        index = self.n
        self.n += 1
        return _batch(index, self.batch_size)


def _tiny_model(fail=False):
    import torch
    from types import SimpleNamespace

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.fail = fail
            self.noises = []
            self.config = SimpleNamespace(enable_depth=False)

        def inference_step(self, **kwargs):
            if self.fail:
                raise RuntimeError("injected inference failure")
            self.noises.append(torch.randn(4).clone())
            batch = kwargs["first_frame"].shape[0]
            return torch.zeros(batch, 3, 2, 4, 4), torch.zeros(batch, 2, 14)

    return Tiny()


def _val_config(batch_size=2):
    from types import SimpleNamespace
    inference = SimpleNamespace(
        num_inference_timesteps=1,
        guidance_scale=1.0,
        action_guidance_scale=1.0,
        action_num_inference_timesteps=None,
        action_temporal_smooth_alpha=0.0,
        depth_temporal_smooth_alpha=0.0,
    )
    return SimpleNamespace(
        model=SimpleNamespace(inference=inference, unified_vlm_task_prompt=False),
        training=SimpleNamespace(batch_size=batch_size),
        system=SimpleNamespace(val_seed=20261021),
    )


def check_validation_contract():
    import matplotlib
    matplotlib.use("Agg", force=True)
    import sys
    from pathlib import Path
    import torch
    import torch.distributed as dist
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/Thinker-WM/train"))
    import sample as val

    pred = torch.zeros(2, 1, 2)
    gt = torch.tensor([[[0.0, 2.0]], [[0.0, 2.0]]])
    metrics = val.compute_action_metrics(pred, gt)
    assert abs(metrics["mse_loss"] - 2.0) < 1e-6, metrics
    assert abs(metrics["l2_error"] - 2.0 ** 0.5) < 1e-4, metrics
    print("PASS l2_error is RMSE", round(metrics["l2_error"], 4))

    cfg = _val_config(2)
    torch.manual_seed(123)
    pulls = _Pulls()
    model = _tiny_model()
    result, saved = val.evaluate_model(model, pulls, None, cfg, num_eval_batches=15)
    assert pulls.n == 15, pulls.n
    assert len(saved) == 4, len(saved)
    assert "action_mse_loss" in result and "action_l2_error" in result
    row = saved[0]["predicted_frames"]
    assert row.untyped_storage().size() == row.numel() * row.element_size(), (
        row.untyped_storage().size(), row.numel())
    probe = torch.rand(1)
    torch.manual_seed(123)
    assert torch.equal(probe, torch.rand(1)), "validation did not restore the training RNG"

    again = _tiny_model()
    val.evaluate_model(again, _Pulls(), None, cfg, num_eval_batches=2)
    first = _tiny_model()
    val.evaluate_model(first, _Pulls(), None, cfg, num_eval_batches=2)
    assert torch.allclose(torch.stack(model.noises[:2]), torch.stack(again.noises))
    assert torch.allclose(torch.stack(again.noises), torch.stack(first.noises))
    print("PASS validation takes 15 batches, 4 viz rows, fixed noise, restored RNG")

    failed = _Pulls()
    try:
        val.evaluate_model(_tiny_model(fail=True), failed, None, cfg, num_eval_batches=15)
        raise AssertionError("15 inference failures returned metrics")
    except RuntimeError as exc:
        text = str(exc)
        assert "successes=0" in text and "failures=15" in text, text
    assert failed.n == 15, failed.n
    assert val.share_validation_error("RuntimeError: injected") == "RuntimeError: injected"

    original = (dist.is_available, dist.is_initialized, dist.broadcast_object_list)

    def broadcast(payload, src=0):
        payload[0] = "RuntimeError: injected"

    dist.is_available = lambda: True
    dist.is_initialized = lambda: True
    dist.broadcast_object_list = broadcast
    try:
        heard = val.share_validation_error(None)
    finally:
        dist.is_available, dist.is_initialized, dist.broadcast_object_list = original
    assert heard == "RuntimeError: injected", heard
    print("PASS a failed validation round raises and is visible to every rank")


def _install_light_reads():
    import torch
    from thinker_wm.data_prepare.robodojo import RoboDojoEpisodeReader

    def read_video(self, record, indices, *, output_format, video_size=None):
        count = len(indices)
        if output_format == "rgb_uint8_hwc":
            return torch.zeros(count, 8, 8, 3, dtype=torch.uint8)
        if output_format == "rgb_float32_chw":
            return torch.zeros(count, 3, 8, 8)
        raise AssertionError(output_format)

    def read_state(self, record, indices):
        return torch.zeros(len(indices), 14)

    def read_action(self, record, indices):
        return torch.zeros(len(indices), 14)

    RoboDojoEpisodeReader.read_video = read_video
    RoboDojoEpisodeReader.read_state = read_state
    RoboDojoEpisodeReader.read_action = read_action


def _light_config():
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("/code1/zxw_data/pretrain/ThinkerWAM-RSI/configs/rsi_robodojo_v1.yaml")
    cfg.system.num_workers = 0
    cfg.system.ready_batch_queue_clone_tensors = False
    cfg.system.ready_batch_queue_size = 2
    cfg.model.wm.history_enable = False
    cfg.model.wm.enable_depth = False
    return cfg


def _identity(batch):
    import torch
    episodes = batch["episode_ids"]
    frames = batch["frame_indices"]
    if torch.is_tensor(episodes):
        episodes = [str(item) for item in episodes.tolist()]
    rows = []
    for episode, frame in zip(episodes, frames):
        if torch.is_tensor(frame):
            frame = frame.tolist()
        rows.append((str(episode), int(frame[0]), int(frame[-1])))
    return tuple(rows)


def check_holdout_windows_repeat():
    from thinker_wm.data_prepare.factory import create_eval_data_pipeline
    _install_light_reads()
    pipe = create_eval_data_pipeline(_light_config(), rank=0, world_size=1, seed=20261021)
    try:
        pipe.set_epoch(0)
        first = []
        for _ in range(15):
            batch, stats = pipe.next()
            assert int(stats["mix_epoch"]) == 0, stats["mix_epoch"]
            first.append(_identity(batch))
        extra_batch, extra_stats = pipe.next()
        assert int(extra_stats["mix_epoch"]) == 1, extra_stats["mix_epoch"]
        assert _identity(extra_batch) != first[0]
        pipe.set_epoch(0)
        second = [_identity(pipe.next()[0]) for _ in range(15)]
    finally:
        pipe.close()
    assert first == second, "resetting the eval pipeline did not repeat the holdout windows"
    assert len(first) == 15 and sum(len(row) for row in first) == 120
    print("PASS holdout repeats the same 120 windows after set_epoch(0)")


def check_resume_crosses_cycle():
    import time
    from thinker_wm.data_prepare.factory import create_training_data_pipeline
    _install_light_reads()
    cfg = _light_config()

    def consume(pipe, count, keep):
        kept = {}
        epochs = {}
        started = time.time()
        for index in range(count):
            batch, stats = pipe.next()
            if index == 0 and "mix_epoch" not in stats:
                raise AssertionError("mix_epoch missing from " + str(sorted(stats)))
            if index in keep:
                kept[index] = _identity(batch)
                epochs[index] = int(stats["mix_epoch"])
            if index and index % 200 == 0:
                print(f"  step {index} epoch {stats['mix_epoch']} {time.time() - started:.1f}s", flush=True)
        return kept, epochs

    per_epoch = 721
    # The 2164th consumed batch is index 2163, the first batch of cycle 3.
    boundary = 3 * per_epoch
    assert boundary == 2163
    watch = set(range(2000, boundary + 1))
    watch.add(per_epoch)
    continuous = create_training_data_pipeline(cfg, rank=0, world_size=16, seed=20261021)
    try:
        assert int(continuous.stream.batches_per_epoch) == per_epoch
        kept, epochs = consume(continuous, boundary + 1, watch)
    finally:
        continuous.close()
    assert epochs[per_epoch] == 1, epochs
    assert epochs[2000] == 2, epochs
    assert epochs[boundary - 1] == 2, epochs
    assert epochs[boundary] == 3, epochs
    assert kept[boundary] != kept[per_epoch]

    resumed = create_training_data_pipeline(cfg, rank=0, world_size=16, seed=20261021)
    try:
        resumed.set_epoch(2000)
        assert int(resumed.queue._epoch) == 2, resumed.queue._epoch
        assert int(resumed.stream._resume_skip_steps) == 2000 % per_epoch
        again, again_epochs = consume(resumed, boundary - 2000 + 1, set(range(boundary - 2000 + 1)))
    finally:
        resumed.close()
    for index in range(2000, boundary + 1):
        local = index - 2000
        assert again[local] == kept[index], (index, again_epochs[local], epochs[index], again[local][:1], kept[index][:1])
        assert again_epochs[local] == epochs[index], (index, again_epochs[local], epochs[index])
    assert again_epochs[boundary - 2000] == 3
    print("PASS resume at step 2000 matches the uninterrupted sequence through epoch 3")



def check_warmup_starts_before_first_update():
    import torch
    from thinker_wm.utils.scheduler import LambdaLinearScheduler

    def make(lr=1e-5):
        opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=lr)
        sched = LambdaLinearScheduler(
            opt, warm_up_steps=500, cycle_length=1_000_000,
            f_max=0.99, f_min=0.4, f_start=1e-6,
        )
        return opt, sched

    opt, sched = make()
    used = []
    for _ in range(3):
        used.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert abs(used[0] - 1e-11) < 1e-18, used
    assert abs(used[1] - 1.980998e-8) < 1e-12, used
    assert abs(used[2] - 3.960996e-8) < 1e-12, used
    assert used[0] < used[1] < used[2]

    opt2, sched2 = make()
    assert abs(opt2.param_groups[0]["lr"] - 1e-11) < 1e-18
    sched2.load_state_dict(sched.state_dict())
    assert sched2.step_count == 3
    assert abs(opt2.param_groups[0]["lr"] - opt.param_groups[0]["lr"]) < 1e-18
    print("PASS warmup initial learning rate is applied before the first update")


def _window_config():
    from types import SimpleNamespace
    inference = SimpleNamespace(
        num_inference_timesteps=1,
        guidance_scale=1.0,
        action_guidance_scale=1.0,
        action_num_inference_timesteps=None,
        action_temporal_smooth_alpha=0.0,
        depth_temporal_smooth_alpha=0.0,
    )
    return SimpleNamespace(
        common=SimpleNamespace(
            num_video_frames=12, action_chunk_size=24, video_action_freq_ratio=2),
        model=SimpleNamespace(
            inference=inference,
            unified_vlm_task_prompt=False,
            wm=SimpleNamespace(ar_packed_teacher_forcing=True, ar_latents_per_block=1),
        ),
        training=SimpleNamespace(batch_size=2),
        system=SimpleNamespace(val_seed=20261021),
    )


def _window_batch():
    import torch
    return {
        "video_frames": torch.zeros(2, 12, 3, 2, 2),
        "first_frame": torch.zeros(2, 3, 2, 2),
        "initial_state": torch.zeros(2, 14),
        "action_sequence": torch.zeros(2, 24, 14),
        "language_embedding": None,
        "vlm_inputs": None,
        "episode_ids": ["a", "b"],
        "frame_indices": [[0, 1], [0, 1]],
    }


class _WindowPulls:
    def __init__(self):
        self.n = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.n += 1
        return _window_batch()


def _window_model(mode):
    import torch
    from types import SimpleNamespace

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.blocks = None
            self.config = SimpleNamespace(enable_depth=False)

        def inference_step(self, **kwargs):
            self.blocks = kwargs.get("num_blocks")
            batch = kwargs["first_frame"].shape[0]
            if mode == "short":
                return torch.zeros(batch, 3, 4, 2, 2), torch.zeros(batch, 8, 14)
            if mode == "nan":
                actions = torch.zeros(batch, 24, 14)
                actions[0, 0, 0] = float("nan")
                return torch.zeros(batch, 3, 12, 2, 2), actions
            if mode == "noact":
                return torch.zeros(batch, 3, 12, 2, 2), None
            return torch.zeros(batch, 3, 12, 2, 2), torch.zeros(batch, 24, 14)

    return Model()


def check_validation_window():
    import matplotlib
    matplotlib.use("Agg", force=True)
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/Thinker-WM/train"))
    import sample as val

    cfg = _window_config()
    window = val._validation_window(cfg)
    assert window == {"blocks": 3, "frames": 12, "actions": 24, "first_block_actions": 8}, window

    model = _window_model("full")
    metrics, _saved = val.evaluate_model(model, _WindowPulls(), None, cfg, num_eval_batches=1)
    assert model.blocks == 3, model.blocks
    assert "action_mse_loss" in metrics and "action_first_block_mse_loss" in metrics

    for mode, needle in (("short", "video length 4"), ("nan", "non-finite"), ("noact", "no predicted actions")):
        try:
            val.evaluate_model(_window_model(mode), _WindowPulls(), None, cfg, num_eval_batches=1)
            raise AssertionError(mode + " was accepted")
        except RuntimeError as exc:
            assert needle in str(exc), (mode, exc)
    print("PASS validation generates 3 blocks and rejects short, NaN, and missing actions")


def check_checkpoint_precedes_validation():
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / "third_party/Thinker-WM/train/train.py").read_text()
    save = text.index("if self.global_step % self.save_interval == 0:")
    val = text.index("if self.val_dataloader is not None and val_interval > 0")
    assert save < val, "checkpoint save is still after validation"
    print("PASS checkpoint save runs before validation")



def check_finetune_schedule():
    import torch
    from omegaconf import OmegaConf
    from thinker_wm.utils.scheduler import LambdaLinearScheduler

    cfg = OmegaConf.load("/code1/zxw_data/pretrain/ThinkerWAM-RSI/configs/rsi_robodojo_v1.yaml")
    assert int(cfg.training.warmup_steps) == 100
    assert float(cfg.training.f_start) == 0.1
    assert float(cfg.training.f_max) == 1.0
    assert float(cfg.training.learning_rate) == 1.0e-5
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1.0e-5)
    sched = LambdaLinearScheduler(
        opt, warm_up_steps=int(cfg.training.warmup_steps),
        cycle_length=int(cfg.training.cycle_length),
        f_max=float(cfg.training.f_max), f_min=float(cfg.training.f_min),
        f_start=float(cfg.training.f_start),
    )
    assert abs(opt.param_groups[0]["lr"] - 1.0e-6) < 1e-12
    for _ in range(100):
        sched.step()
    assert sched.step_count == 100
    assert abs(opt.param_groups[0]["lr"] - 1.0e-5) < 1e-12
    print("PASS finetune warmup reaches 1e-5 after 100 steps")


if __name__ == "__main__":
    check_cycles()
    check_resume_math()
    check_warmup_starts_before_first_update()
    check_finetune_schedule()
    check_validation_contract()
    check_validation_window()
    check_checkpoint_precedes_validation()
    check_holdout_windows_repeat()
    check_resume_crosses_cycle()
