#!/usr/bin/env python3
"""Local CPU contract tests; no checkpoint/model/data/GPU or remote access."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Candidate overlays do not duplicate the unchanged monitor library.
MONITOR = ROOT / "third_party/Thinker-WM/libs" / "motus-monitor"
if not MONITOR.exists():
    MONITOR = ROOT.parent / "source" / "third_party/Thinker-WM/libs" / "motus-monitor"
sys.path.insert(0, str(MONITOR))

import numpy as np
import torch
from thinker_wm.periodic_eval import validation_artifacts as publisher
from thinker_wm.periodic_eval import render
from scripts import eval_distributed_holdout as runner


COUNTS = (24, 48, 48)


def window_sample(window_id, task):
    frames = torch.zeros(3, 3, 32, 32)
    frames[0, 0] = .25
    frames[1, 1] = .5
    frames[2, 2] = .75
    mask = torch.ones(3, 32, 32, dtype=torch.bool)
    mask[:, -1, -1] = False
    return dict(task=task, task_name=task, task_text=f"Instruction for {task}",
                episode_id=f"episode_{task}_{window_id // 8}", window_id=window_id,
                frame_indices=[window_id, window_id + 2, window_id + 4],
                metrics={"video_mse": .1, "mse_loss": .2, "depth_mse_valid": .3, "a2v_video_mse": .4},
                gt_first_frame=frames[0], gt_frames=frames,
                predicted_frames=frames.clone(), gt_depth_frames=frames.clone(),
                predicted_depth=frames.clone(), depth_valid_mask=mask,
                a2v_predicted_frames=frames.clone(),
                gt_actions=torch.zeros(24, 14), predicted_actions=torch.ones(24, 14),
                vlm_input_messages=f"ACTUAL MODEL ARGUMENT: {task}",
                original_vlm_input_messages="original adapter prompt; not the model prompt")


def fixture():
    windows, samples, by_task = [], [], {}
    for task, count in zip(publisher.TASKS, COUNTS):
        stats = {"video_mse": .1, "mse_loss": .2, "depth_mse_valid": .3, "a2v_video_mse": .4}
        by_task[task] = dict(stats, num_samples=count)
        for _ in range(count):
            sample = window_sample(len(windows), task)
            windows.append({"id": sample["window_id"], "task": task, "episode_id": sample["episode_id"],
                            "frame_indices": sample["frame_indices"], "metrics": stats.copy()})
            samples.append(sample)
    metrics = dict(by_task[publisher.TASKS[0]], num_samples=120, _windows=windows, _by_task=by_task,
                   visualization=torch.zeros(1, 3, 32, 32))
    return SimpleNamespace(depth=SimpleNamespace(mode="log", z_near=.01, z_far=2.0, invalid_to="far")), metrics, samples


class PublisherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT, prefix=".publisher-test-")
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name) / "run"
        self.run.mkdir()
        (self.run / "run.json").write_text('{}\n')
        self.config, self.metrics, self.samples = fixture()
        self.encodes = []

    def encode(self, path, frames, fps):
        self.assertEqual(fps, 12.5)
        self.assertEqual(frames.shape, (3, 32, 96, 3))
        self.encodes.append((Path(path).name, frames.copy()))
        Path(path).write_bytes(b"fake encoded video")

    def stage(self):
        root = Path(tempfile.mkdtemp(prefix="holdout-", dir=self.tmp.name))
        with patch.object(render, "write_video", self.encode):
            return [publisher.stage_window_media(sample, root, rank=sample["window_id"] % 16)
                    for sample in self.samples]

    def publish(self, saved):
        return publisher.publish_validation_artifacts(
            self.config, 2000, self.metrics, saved, self.run,
            seconds=12, output_dir=Path(self.tmp.name) / "exports", world_size=16)

    def test_append_revisions_complete_coverage_and_actual_prompts(self):
        historical = self.run / "eval" / "step_0002000" / "rank0" / "sample_0" / "compare.mp4"
        historical.parent.mkdir(parents=True)
        historical.write_bytes(b"historical-video-must-stay")
        stream = self.run / "eval" / "000000.jsonl"
        old = b'{"type":"eval_round","step":1000,"num_samples":1}\n'
        stream.write_bytes(old)
        saved = self.stage()
        self.assertTrue(all(isinstance(v, (str, int, float, type(None))) for e in saved for v in e.values()))
        first = self.publish(saved)
        self.assertEqual(list(Path(saved[0]["media_dir"]).parent.iterdir()), [])  # all moved into the run
        media = {path: path.read_bytes() for path in (self.run / first["files"]["summary"]).parent.rglob("*") if path.is_file()}
        prefix = stream.read_bytes()
        second = self.publish(self.stage())
        self.assertNotEqual(first["revision"], second["revision"])
        self.assertTrue(stream.read_bytes().startswith(prefix))
        self.assertTrue(prefix.startswith(old))
        self.assertEqual(historical.read_bytes(), b"historical-video-must-stay")
        for path, before in media.items():
            self.assertEqual(path.read_bytes(), before)
        records = [json.loads(line) for path in sorted((self.run / "eval").glob("*.jsonl")) for line in path.read_text().splitlines()]
        self.assertEqual(len(records), 1 + 2 * 121)
        self.assertEqual(records[121]["type"], "eval_round")             # completion after all samples
        for key in ("num_samples", "num_saved_samples", "viz_samples", "num_eval_samples", "num_windows"):
            self.assertEqual(first[key], 120, key)
        self.assertEqual(len(first["_windows"]), 120)
        self.assertEqual(first["world_size"], 16)
        self.assertGreaterEqual(first["media_seconds_total"], 0.0)
        self.assertNotIn("visualization", first["metrics"])
        self.assertEqual(set(first["metrics_by_task"]), set(publisher.TASKS))
        for task, count in zip(publisher.TASKS, COUNTS):
            self.assertEqual(first["metrics_by_task"][task]["num_windows"], count)
            self.assertEqual(first["metrics_by_task"][task]["metric_counts"]["depth_mse_valid"], count)
        samples = [row for row in records[1:122] if row["type"] == "eval_sample"]
        self.assertEqual(sorted(row["window_id"] for row in samples), list(range(120)))
        self.assertEqual(len({(row["rank"], row["sample_index"]) for row in samples}), 120)
        for row in samples:
            self.assertEqual(row["source"], "robodojo_holdout")
            self.assertEqual(row["split"], "holdout")
            self.assertEqual(row["sample_index"], row["window_id"])
            self.assertEqual(row["eval_rank"], row["window_id"] % 16)
            self.assertEqual(row["vlm_prompt"], f"ACTUAL MODEL ARGUMENT: {row['task_name']}")
            self.assertEqual(row["prompt_source"], "model_argument")
            self.assertIn(f"/{row['task']}/window_{row['window_id']:04d}/", row["files"]["meta"])
            self.assertNotIn("gt_actions", row)
            meta = json.loads((self.run / row["files"]["meta"]).read_text())
            self.assertEqual((len(meta["gt_actions"]), len(meta["gt_actions"][0])), (24, 14))
            self.assertEqual(meta["step"], 2000)
            self.assertFalse(meta["a2v_action_score_available"])
            self.assertEqual(meta["depth_metric_space"], "codec_[0,1]")
            self.assertNotIn("predicted_frames", meta)
            self.assertNotIn("depth_valid_mask", meta)
            for relative in row["files"].values():
                self.assertTrue((self.run / relative).is_file())
        self.assertLess(len(stream.read_bytes()), 1_000_000)
        self.assertEqual(len(self.encodes), 2 * 120 * 3)
        for name, frames in self.encodes:
            if name == "depth_compare.mp4":
                self.assertEqual(frames[:, -1, 63].sum(), 0)
                self.assertEqual(frames[:, -1, 95].sum(), 0)

    def test_six_task_240_window_publication(self):
        tasks = publisher.TASKS + ("plug_in_charger", "insert_key", "fold_clothes")
        self.config.system = SimpleNamespace(val_tasks=list(tasks), val_num_batches=30)
        self.config.training = SimpleNamespace(batch_size=8)
        windows = self.metrics['_windows']
        for task in tasks[3:]:
            stats = dict(video_mse=.1, mse_loss=.2, depth_mse_valid=.3, a2v_video_mse=.4)
            self.metrics['_by_task'][task] = dict(stats, num_samples=40)
            for _ in range(40):
                sample = window_sample(len(windows), task)
                windows.append(dict(id=sample['window_id'], task=task, episode_id=sample['episode_id'],
                                    frame_indices=sample['frame_indices'], metrics=stats.copy()))
                self.samples.append(sample)
        self.metrics['num_samples'] = 240
        record = self.publish(self.stage())
        self.assertEqual(record['num_samples'], 240)
        self.assertEqual(record['num_saved_samples'], 240)
        self.assertEqual(set(record['metrics_by_task']), set(tasks))
        self.assertEqual(len(self.encodes), 720)
        for task in tasks[3:]:
            self.assertEqual(record['metrics_by_task'][task]['num_windows'], 40)

    def test_six_tasks_balanced_120_and_reject_wrong_quota(self):
        tasks = publisher.TASKS + ("plug_in_charger", "insert_key", "fold_clothes")
        self.config.system = SimpleNamespace(val_tasks=list(tasks), val_num_batches=15, val_windows_per_task=20)
        self.config.training = SimpleNamespace(batch_size=8)
        self.samples = []
        windows, by_task = [], {}
        for task in tasks:
            stats = dict(video_mse=.1, mse_loss=.2, depth_mse_valid=.3, a2v_video_mse=.4)
            by_task[task] = dict(stats, num_samples=20)
            for _ in range(20):
                sample = window_sample(len(windows), task)
                windows.append(dict(id=sample['window_id'], task=task, episode_id=sample['episode_id'],
                                    frame_indices=sample['frame_indices'], metrics=stats.copy()))
                self.samples.append(sample)
        self.metrics.update(num_samples=120, _windows=windows, _by_task=by_task)
        saved = self.stage()
        self.config.system.val_windows_per_task = 19
        with self.assertRaisesRegex(ValueError, 'per-task quota'):
            self.publish(saved)
        self.assertFalse((self.run / 'eval').exists())
        self.config.system.val_windows_per_task = 20
        record = self.publish(saved)
        self.assertEqual(record['num_samples'], 120)
        self.assertEqual(record['num_saved_samples'], 120)
        self.assertEqual(len(self.encodes), 360)
        for task in tasks:
            self.assertEqual(record['metrics_by_task'][task]['num_windows'], 20)

    def test_eight_tasks_balanced_120_and_reject_wrong_quota(self):
        tasks = ("insert_tubes", "plug_in_charger", "insert_key", "fold_clothes",
                 "fill_pen_holder", "organize_table", "match_and_pick_from_conveyor", "sweep_blocks")
        self.config.system = SimpleNamespace(val_tasks=list(tasks), val_num_batches=15, val_windows_per_task=15)
        self.config.training = SimpleNamespace(batch_size=8)
        self.samples = []
        windows, by_task = [], {}
        for task in tasks:
            stats = dict(video_mse=.1, mse_loss=.2, depth_mse_valid=.3, a2v_video_mse=.4)
            by_task[task] = dict(stats, num_samples=15)
            for _ in range(15):
                sample = window_sample(len(windows), task)
                windows.append(dict(id=sample['window_id'], task=task, episode_id=sample['episode_id'],
                                    frame_indices=sample['frame_indices'], metrics=stats.copy()))
                self.samples.append(sample)
        self.metrics.update(num_samples=120, _windows=windows, _by_task=by_task)
        saved = self.stage()
        self.config.system.val_windows_per_task = 14
        with self.assertRaisesRegex(ValueError, 'per-task quota'):
            self.publish(saved)
        self.assertFalse((self.run / 'eval').exists())
        self.config.system.val_windows_per_task = 15
        record = self.publish(saved)
        self.assertEqual(record['num_samples'], 120)
        self.assertEqual(record['num_saved_samples'], 120)
        self.assertEqual(len(self.encodes), 360)
        for task in tasks:
            self.assertEqual(record['metrics_by_task'][task]['num_windows'], 15)

    def test_reject_partial_or_inconsistent_results_before_writing(self):
        for change in ("window", "missing_entry", "identity", "media_file", "duplicate"):
            with self.subTest(change=change):
                self.config, self.metrics, self.samples = fixture()
                saved = self.stage()
                if change == "window":
                    self.metrics["_windows"].pop()
                elif change == "missing_entry":
                    saved.pop()
                elif change == "identity":
                    meta_path = Path(saved[5]["media_dir"]) / "meta.json"
                    meta = json.loads(meta_path.read_text())
                    meta["episode_id"] = "someone_else"
                    meta_path.write_text(json.dumps(meta))
                elif change == "media_file":
                    (Path(saved[7]["media_dir"]) / "a2v_compare.mp4").unlink()
                else:
                    saved[1] = dict(saved[1], window_id=0)
                with self.assertRaises(ValueError):
                    self.publish(saved)
                self.assertFalse((self.run / "eval").exists())
                publisher.discard_staged_media(saved)
                self.assertFalse(Path(saved[0]["media_dir"]).parent.exists())

    def test_stage_rejects_bad_window_without_writing(self):
        for change in ("prompt", "task", "shape"):
            with self.subTest(change=change):
                sample = window_sample(3, publisher.TASKS[0])
                if change == "prompt":
                    sample["vlm_input_messages"] = None
                elif change == "task":
                    sample["task"] = sample["task_name"] = "unknown"
                else:
                    sample["predicted_depth"] = torch.zeros(1, 3, 3, 32, 32)
                root = Path(tempfile.mkdtemp(dir=self.tmp.name))
                with self.assertRaises(ValueError):
                    publisher.stage_window_media(sample, root, rank=0)
                self.assertEqual(list(root.iterdir()), [])

    def test_explicit_short_video_layout(self):
        video = torch.zeros(3, 3, 4, 4)
        video[0, 2] = 1
        result = publisher._frames(video)
        self.assertTrue((result[0, :, :, 2] == 255).all())
        self.assertEqual(result[1:].sum(), 0)

    def test_nonzero_rank_cannot_publish(self):
        with patch.dict(os.environ, RANK="1"):
            with self.assertRaises(RuntimeError):
                self.publish([])

    def test_real_fractional_fps_encoder(self):
        import imageio.v2 as imageio
        path = Path(self.tmp.name) / "fps.mp4"
        render.write_video(path, np.zeros((5, 32, 96, 3), dtype=np.uint8), fps=12.5)
        with imageio.get_reader(str(path)) as reader:
            self.assertAlmostEqual(reader.get_meta_data()["fps"], 12.5)
            self.assertEqual(reader.count_frames(), 5)


class RunnerTests(unittest.TestCase):
    def test_checkpoint_completion_and_step(self):
        with tempfile.TemporaryDirectory(dir=ROOT, prefix=".runner-test-") as tmp:
            root = Path(tmp)
            marker = root / ".motus_checkpoint_complete.json"
            with self.assertRaises(ValueError):
                runner.validate_checkpoint(root, 2000)
            marker.write_text(json.dumps({"status": "complete", "step": 2000}))
            with self.assertRaises(FileNotFoundError):
                runner.validate_checkpoint(root, 2000)
            (root / "pytorch_model_0.bin").write_bytes(b"fake weights")
            self.assertEqual(runner.validate_checkpoint(root, 2000)["step"], 2000)
            with self.assertRaises(ValueError):
                runner.validate_checkpoint(root, 1000)
            marker.write_text(json.dumps({"status": "writing", "step": 2000}))
            with self.assertRaises(ValueError):
                runner.validate_checkpoint(root, 2000)

    def test_accelerator_surface(self):
        accelerator = runner.EvalAccelerator(9, 16, 1, "cuda:1")
        model = object()
        self.assertIs(accelerator.unwrap_model(model), model)
        self.assertFalse(accelerator.is_main_process)
        self.assertEqual(accelerator.num_processes, 16)
        self.assertEqual(accelerator.local_process_index, 1)


if __name__ == "__main__":
    unittest.main()
