from __future__ import annotations

import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from motus_data import (
    ContractError,
    build_data_pipeline,
    build_dataset,
    validate_batch,
    validate_motus_model_batch_v1,
)
from motus_data.collate import collate_samples
from motus_data.baseline import batch_signature, selection_records, sample_signature
from motus_data.datasets.agibot_native import AgibotDataset, split_agibot_episodes
from motus_data.datasets.agibot_common import action_text_for_start
from motus_data.datasets.agibot_beta import AgibotBetaBatchSampler, AgibotBetaDataset
from motus_data.datasets.manifest import MotusManifestDataset
from motus_data.datasets.robotmind import ROBOTMIND_SCHEMAS, RobotMindDataset
from motus_data.verification import run_optimizer_smoke
from motus_data.sampling import ContiguousWindowSampler, EpisodeRange


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.config = OmegaConf.load(
            Path(__file__).resolve().parent / "synthetic_mix.yaml"
        )

    def test_video_only_collate_is_explicit(self):
        dataset = build_dataset(self.config)
        video_only = dataset.datasets[1]
        batch = collate_samples([video_only[0], video_only[1]])
        validate_batch(batch, expected_batch_size=2)
        self.assertIsNone(batch["action_sequence"])
        self.assertNotIn("initial_state", batch)
        self.assertFalse(batch["action_sequence_valid"].any())
        self.assertEqual(batch["source_children"], ["synthetic_b", "synthetic_b"])

    def test_current_motus_model_batch_v1_contract(self):
        dataset = build_dataset(self.config)
        full_modal = dataset.datasets[0]
        batch = collate_samples([full_modal[0], full_modal[1]])
        validate_motus_model_batch_v1(
            batch,
            expected_batch_size=2,
            num_video_frames=4,
            video_action_freq_ratio=1,
        )
        video_only = collate_samples([dataset.datasets[1][0], dataset.datasets[1][1]])
        with self.assertRaises(ContractError):
            validate_motus_model_batch_v1(video_only, expected_batch_size=2)

    def test_baseline_signatures_freeze_samples_and_batch(self):
        dataset = build_dataset(self.config).datasets[0]
        samples = [dataset[0], dataset[1]]
        batch = collate_samples(samples)

        first_sample = sample_signature(samples[0])
        repeated_sample = sample_signature(samples[0])
        self.assertEqual(first_sample, repeated_sample)
        self.assertEqual(
            first_sample["tensors"]["video_frames"]["sha256"],
            repeated_sample["tensors"]["video_frames"]["sha256"],
        )
        changed = dict(samples[0])
        changed["video_frames"] = samples[0]["video_frames"].clone()
        changed["video_frames"][0, 0, 0, 0] += 1
        self.assertNotEqual(
            first_sample["tensors"]["video_frames"]["sha256"],
            sample_signature(changed)["tensors"]["video_frames"]["sha256"],
        )

        selections = selection_records(samples)
        self.assertEqual(selections[0]["episode_id"], samples[0]["episode_id"])
        self.assertEqual(
            selections[0]["video_frame_indices"], samples[0]["frame_indices"]
        )
        frozen_batch = batch_signature(batch)
        self.assertEqual(frozen_batch["tensors"]["first_frame"]["shape"][0], 2)
        self.assertEqual(frozen_batch["lists"]["episode_ids"], batch["episode_ids"])

    def test_mixture_batches_are_child_homogeneous(self):
        dataset = build_dataset(self.config)
        sampler = dataset.create_batch_sampler(batch_size=4, seed=7)
        for step, indices in enumerate(sampler):
            offsets = dataset._offsets
            children = [max(i for i, offset in enumerate(offsets) if offset <= index) for index in indices]
            self.assertEqual(len(set(children)), 1)
            if step >= 20:
                break

    def test_epoch_changes_dynamic_sample_seed(self):
        dataset = build_dataset(self.config).datasets[0]
        first = dataset[3]["video_frames"]
        dataset.set_epoch(1)
        second = dataset[3]["video_frames"]
        self.assertFalse(torch.equal(first, second))

    def test_runtime_window_sampler_is_bounded_and_reproducible(self):
        sampler = ContiguousWindowSampler(num_frames=5, frame_step=2)
        episode = EpisodeRange("episode-1", first_frame=10, last_frame=30)
        first_gen = torch.Generator().manual_seed(123)
        second_gen = torch.Generator().manual_seed(123)
        first = sampler.sample(episode, generator=first_gen, training=True)
        second = sampler.sample(episode, generator=second_gen, training=True)
        self.assertEqual(first, second)
        self.assertGreaterEqual(first.start_frame, 10)
        self.assertLessEqual(first.frame_indices[-1], 30)
        self.assertEqual(first.frame_indices, tuple(range(first.start_frame, first.start_frame + 10, 2)))

        evaluation = sampler.sample(episode, training=False)
        self.assertEqual(evaluation.start_frame, 16)

    def test_queue_and_optimizer_boundary(self):
        pipeline = build_data_pipeline(self.config, seed=11)
        result = run_optimizer_smoke(pipeline, steps=8, device="cpu")
        self.assertTrue(result["all_losses_finite"])
        self.assertTrue(result["parameters_updated"])
        self.assertEqual(sum(result["source_counts"].values()), 8)

    def test_robotmind_manifest_hdf5_decode_and_alignment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            h5_path = root / "trajectory.hdf5"
            encoded_frames = []
            for index in range(5):
                image = Image.fromarray(
                    np.full((8, 10, 3), fill_value=index * 30, dtype=np.uint8)
                )
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG")
                encoded_frames.append(np.frombuffer(buffer.getvalue(), dtype=np.uint8))
            with h5py.File(h5_path, "w") as handle:
                dtype = h5py.vlen_dtype(np.dtype("uint8"))
                images = handle.create_dataset("camera/rgb", shape=(5,), dtype=dtype)
                for index, encoded in enumerate(encoded_frames):
                    images[index] = encoded
                handle.create_dataset(
                    "puppet/joints", data=np.arange(15, dtype=np.float32).reshape(5, 3)
                )

            record = {
                "schema_version": "motus_manifest_v1",
                "id": "robotmind:test:episode0:window0",
                "source_dataset": "robotmind",
                "split": "train",
                "task": "fixture task",
                "text": {"t5": "Task: fixture", "vlm": "Task: fixture", "en": "fixture"},
                "video": {
                    "storage_backend": "robotmind_hdf5_jpeg",
                    "uri": str(h5_path),
                    "key": "camera/rgb",
                    "frame_indices": [0, 1, 2, 3, 4],
                },
                "robot": {"name": "robotmind-fixture", "state_dim": 3, "action_dim": 3},
                "state": {
                    "storage_backend": "robotmind_hdf5",
                    "uri": str(h5_path),
                    "keys": {"joint": "puppet/joints"},
                    "index": 0,
                },
                "action": {
                    "storage_backend": "robotmind_hdf5",
                    "uri": str(h5_path),
                    "keys": {"joint": "puppet/joints"},
                    "frame_indices": [0, 1, 2, 3, 4],
                    "video_action_freq_ratio": 1,
                },
                "loss": {"video_weight": 1.0, "action_weight": 1.0},
                "metadata": {"state_action_keys_order": ["joint"]},
            }
            manifest_path = root / "manifest.jsonl"
            manifest_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            dataset = MotusManifestDataset(
                manifest_path=str(manifest_path),
                split="train",
                video_size=(8, 10),
                num_video_frames=4,
                action_dim=3,
                state_dim=3,
                source_dataset="robotmind",
            )
            sample = dataset[0]
            self.assertEqual(tuple(sample["first_frame"].shape), (3, 8, 10))
            self.assertEqual(tuple(sample["video_frames"].shape), (4, 3, 8, 10))
            self.assertTrue(torch.equal(sample["initial_state"], torch.tensor([0.0, 1.0, 2.0])))
            self.assertTrue(
                torch.equal(
                    sample["action_sequence"],
                    torch.arange(3, 15, dtype=torch.float32).reshape(4, 3),
                )
            )
            self.assertEqual(sample["t5_input_message"], "Task: fixture")
            self.assertEqual(sample["vlm_input_message"], "Task: fixture")
            dataset.close()

    def test_robotmind_direct_runtime_window_and_factory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_dir = root / "pick_object"
            data_dir = task_dir / "success_episodes" / "episode_001" / "data"
            data_dir.mkdir(parents=True)
            (task_dir / "zh_description.txt").write_text("拿起物体", encoding="utf-8")
            h5_path = data_dir / "trajectory.hdf5"
            encoded_frames = []
            for index in range(9):
                image = Image.fromarray(
                    np.full((8, 10, 3), fill_value=index * 20, dtype=np.uint8)
                )
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG")
                encoded_frames.append(np.frombuffer(buffer.getvalue(), dtype=np.uint8))

            schema = ROBOTMIND_SCHEMAS["tianyi"]
            field_dims = [7, 7, 1, 1]
            with h5py.File(h5_path, "w") as handle:
                dtype = h5py.vlen_dtype(np.dtype("uint8"))
                images = handle.create_dataset(schema.rgb_key, shape=(9,), dtype=dtype)
                for index, encoded in enumerate(encoded_frames):
                    images[index] = encoded
                expected_arrays = []
                for field_index, ((_, key), dim) in enumerate(
                    zip(schema.state_action_keys, field_dims)
                ):
                    values = (
                        np.arange(9 * dim, dtype=np.float32).reshape(9, dim)
                        + field_index * 1000
                    )
                    handle.create_dataset(key, data=values)
                    expected_arrays.append(values)
                metadata = handle.create_group("metadata")
                metadata.attrs["trajectory_length"] = 9
                metadata.attrs["language_instruction"] = "pick up the object"

            dataset = RobotMindDataset(
                data_root=str(root),
                embodiment="tianyi",
                split="train",
                video_size=(8, 10),
                num_video_frames=3,
                video_frame_step=2,
                action_video_fps_ratio=1,
                instruction_lang="en",
                test_fraction=0.0,
                virtual_samples_per_episode=2,
                max_decode_retries=1,
                seed=17,
            )
            sample = dataset.load_window(0, 1)
            self.assertEqual(sample["frame_indices"], [1, 3, 5, 7])
            self.assertEqual(tuple(sample["video_frames"].shape), (3, 3, 8, 10))
            self.assertTrue(
                torch.equal(
                    sample["initial_state"],
                    torch.from_numpy(
                        np.concatenate([values[1] for values in expected_arrays])
                    ).float(),
                )
            )
            self.assertTrue(
                torch.equal(
                    sample["action_sequence"],
                    torch.from_numpy(
                        np.concatenate(
                            [values[[3, 5, 7]] for values in expected_arrays], axis=-1
                        )
                    ).float(),
                )
            )
            self.assertEqual(sample["t5_input_message"], "Task: pick up the object.\n")
            self.assertEqual(sample["source_dataset"], "robotmind")
            self.assertEqual(sample["source_child"], "robotmind_tianyi")
            self.assertEqual(sample["action_video_fps_ratio"], 1)

            config = OmegaConf.create(
                {
                    "common": {
                        "video_height": 8,
                        "video_width": 10,
                        "num_video_frames": 3,
                        "video_action_freq_ratio": 1,
                    },
                    "dataset": {
                        "type": "robotmind",
                        "name": "robotmind_tianyi_fixture",
                        "source_dataset": "robotmind",
                        "embodiment": "tianyi",
                        "data_root": str(root),
                        "split": "train",
                        "instruction_lang": "en",
                        "video_frame_step": 1,
                        "test_fraction": 0.0,
                        "virtual_samples_per_episode": 2,
                    },
                    "training": {"batch_size": 2},
                    "system": {
                        "num_workers": 0,
                        "pin_memory": False,
                        "persistent_workers": False,
                        "prefetch_factor": None,
                        "ready_batch_queue_enabled": True,
                        "ready_batch_queue_size": 2,
                        "ready_batch_queue_timeout_seconds": 30,
                    },
                }
            )
            pipeline = build_data_pipeline(config, seed=23)
            try:
                batch, _ = pipeline.next()
                validate_motus_model_batch_v1(
                    batch,
                    expected_batch_size=2,
                    num_video_frames=3,
                    video_action_freq_ratio=1,
                )
                self.assertEqual(batch["source_children"], [
                    "robotmind_tianyi_fixture",
                    "robotmind_tianyi_fixture",
                ])
            finally:
                pipeline.close()

    def test_robotmind_embodiment_schemas_are_explicit(self):
        self.assertEqual(ROBOTMIND_SCHEMAS["ark"].state_action_dim, 14)
        self.assertEqual(ROBOTMIND_SCHEMAS["agilex"].state_action_dim, 14)
        self.assertEqual(
            ROBOTMIND_SCHEMAS["agilex"].rgb_key,
            "camera_observations/color_images/camera_front",
        )
        self.assertEqual(ROBOTMIND_SCHEMAS["tianyi"].state_action_dim, 16)
        self.assertEqual(ROBOTMIND_SCHEMAS["tienkung"].state_action_dim, 16)
        self.assertEqual(ROBOTMIND_SCHEMAS["arkmobile"].state_action_dim, 20)
        self.assertEqual(
            [name for name, _ in ROBOTMIND_SCHEMAS["ark"].state_action_keys],
            ["left_arm", "left_gripper", "right_arm", "right_gripper"],
        )

    def test_agibot_native_runtime_window_aligns_all_modalities(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            annotation_root = root / "new_task_info"
            data_root = root / "agibot"
            annotation_root.mkdir()
            h5_path = data_root / "proprio_stats" / "1" / "2" / "proprio_stats.h5"
            h5_path.parent.mkdir(parents=True)
            timesteps = 60
            with h5py.File(h5_path, "w") as handle:
                positions = np.arange(timesteps * 2 * 3, dtype=np.float32).reshape(
                    timesteps, 2, 3
                )
                orientations = np.zeros((timesteps, 2, 4), dtype=np.float32)
                orientations[..., 3] = 1.0
                effectors = np.arange(timesteps * 2, dtype=np.float32).reshape(
                    timesteps, 2
                )
                handle.create_dataset("state/end/position", data=positions)
                handle.create_dataset("state/end/orientation", data=orientations)
                handle.create_dataset("state/effector/position", data=effectors)

            labels = {
                str(index): {"action_text": f"move at {index}"}
                for index in range(timesteps)
            }
            annotation_item = {
                "task_id": 1,
                "episode_id": 2,
                "task_name": "fixture task",
                "init_scene_text": "fixture scene",
                "action_video_fps_ratio": 2,
                "label_info": {"flatten_action_config": labels},
                "effective_frames": [2, 20],
            }
            (annotation_root / "task_1.json").write_text(
                json.dumps([annotation_item]), encoding="utf-8"
            )

            decoded_indices = []

            def fake_decode(_path, indices, video_size):
                decoded_indices.append(list(indices))
                height, width = video_size
                return torch.stack(
                    [torch.full((3, height, width), index / 100.0) for index in indices]
                )

            dataset = AgibotDataset(
                [str(annotation_root)],
                [str(data_root)],
                ["agibot-world"],
                video_size=(8, 10),
                num_video_frames=4,
                test_fraction=0.0,
                virtual_samples_per_episode=3,
                max_decode_retries=1,
                seed=123,
            )
            self.assertEqual(len(dataset), 3)
            with mock.patch(
                "motus_data.datasets.agibot_native.decode_video_frames",
                side_effect=fake_decode,
            ):
                first = dataset[0]
                second = dataset[1]
                repeated = dataset[0]

            self.assertEqual(first["frame_indices"], decoded_indices[0])
            self.assertEqual(first["frame_indices"], decoded_indices[2])
            self.assertEqual(first["frame_indices"], list(range(first["frame_indices"][0], first["frame_indices"][0] + 5)))
            self.assertIn(
                f"Action: move at {first['frame_indices'][0]}", first["t5_input_message"]
            )
            self.assertEqual(first["t5_input_message"], first["t5_input_message"].strip())
            self.assertEqual(first["vlm_input_message"], first["vlm_input_message"].strip())
            self.assertEqual(tuple(first["video_frames"].shape), (4, 3, 8, 10))
            self.assertEqual(tuple(first["initial_state"].shape), (26,))
            self.assertEqual(tuple(first["action_sequence"].shape), (8, 26))
            self.assertEqual(first["action_video_fps_ratio"], 2)
            self.assertNotEqual(first["manifest_id"], second["manifest_id"])

            batch = collate_samples([first, repeated])
            validate_motus_model_batch_v1(
                batch,
                expected_batch_size=2,
                num_video_frames=4,
                # Per-sample ratio=2 must override a heterogeneous mix default.
                video_action_freq_ratio=1,
            )
            self.assertEqual(batch["action_video_fps_ratios"], [2, 2])

    def test_agibot_episode_split_is_exact_and_disjoint_without_an_index_file(self):
        episodes = [
            {"episode_uid": f"agibot:1:{index}", "task_id": 1, "episode_id": index}
            for index in range(100)
        ]
        train = split_agibot_episodes(episodes, "train", 0.01, 20260720)
        test = split_agibot_episodes(episodes, "test", 0.01, 20260720)
        self.assertEqual(len(train), 99)
        self.assertEqual(len(test), 1)
        self.assertFalse(
            {item["episode_uid"] for item in train}
            & {item["episode_uid"] for item in test}
        )

    def test_agibot_direct_dataset_requires_the_old_start_label_invariant(self):
        with self.assertRaises(KeyError):
            action_text_for_start({}, 17, require_key=True)

    def test_agibot_beta_sampler_interleaves_episodes_before_virtual_rounds(self):
        dataset = mock.Mock()
        dataset.virtual_samples_per_episode = 4
        dataset.episodes = [{}, {}, {}]
        dataset.source_child = "agibot_beta"
        sampler = AgibotBetaBatchSampler(
            dataset,
            batch_size=2,
            drop_last=True,
            shuffle=False,
            num_replicas=1,
            rank=0,
            seed=7,
        )
        batches = list(sampler)
        first_round_episodes = [batch[0] // 4 for batch in batches[:3]]
        second_round_episodes = [batch[0] // 4 for batch in batches[3:6]]
        self.assertEqual(first_round_episodes, [0, 1, 2])
        self.assertEqual(second_round_episodes, [0, 1, 2])

    def test_agibot_beta_direct_tar_batch_aligns_every_modality(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "agibot_beta"
            task_info = data_root / "task_info"
            catalog = root / "catalog"
            task_info.mkdir(parents=True)
            catalog.mkdir()

            h5_buffer = io.BytesIO()
            timesteps = 24
            with h5py.File(h5_buffer, "w") as handle:
                positions = np.zeros((timesteps, 2, 3), dtype=np.float32)
                positions[..., 0] = 0.66
                positions[..., 2] = 0.65
                positions[:, 1, 1] = 0.27
                orientations = np.zeros((timesteps, 2, 4), dtype=np.float32)
                orientations[..., 3] = 1.0
                effectors = np.full((timesteps, 2), 0.5, dtype=np.float32)
                effectors[:, 0] += np.arange(timesteps, dtype=np.float32) / 1000
                handle.create_dataset("state/end/position", data=positions)
                handle.create_dataset("state/end/orientation", data=orientations)
                handle.create_dataset("state/effector/position", data=effectors)

            tar_path = root / "beta.tar"
            members = {
                "7/videos/head_color.mp4": b"fixture-video-member",
                "3/7/proprio_stats.h5": h5_buffer.getvalue(),
            }
            with tarfile.open(tar_path, "w") as archive:
                for name, payload in members.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))
            with tarfile.open(tar_path, "r") as archive:
                refs = {
                    member.name: (member.offset_data, member.size)
                    for member in archive.getmembers()
                }

            def write_catalog(name, member):
                offset, size = refs[member]
                row = {
                    "task_id": 3,
                    "episode_id": 7,
                    "uri": str(tar_path),
                    "member": member,
                    "tar_offset_data": offset,
                    "tar_size": size,
                }
                path = catalog / name
                path.write_text(json.dumps(row) + "\n", encoding="utf-8")
                path.with_suffix(path.suffix + ".done").write_text(
                    "status=complete\n", encoding="utf-8"
                )

            write_catalog("video_task_3.jsonl", "7/videos/head_color.mp4")
            write_catalog("proprio.jsonl", "3/7/proprio_stats.h5")
            annotation = [{
                "episode_id": 7,
                "task_name": "put the cup away",
                "init_scene_text": "a cup is on the table",
                "label_info": {
                    "action_config": [
                        {
                            "start_frame": 2,
                            "end_frame": 10,
                            "action_text": "reach for the cup",
                        },
                        {
                            "start_frame": 10,
                            "end_frame": 20,
                            "action_text": "move the cup",
                        },
                    ]
                },
            }]
            (task_info / "task_3.json").write_text(
                json.dumps(annotation), encoding="utf-8"
            )

            decode_calls = []

            def fake_decode(_path, indices, video_size, *, backend):
                indices = list(indices)
                decode_calls.append((indices, backend))
                height, width = video_size
                return torch.stack([
                    torch.full((3, height, width), index / 100.0)
                    for index in indices
                ])

            dataset = AgibotBetaDataset(
                str(data_root),
                str(catalog),
                split="train",
                video_size=(8, 10),
                num_video_frames=4,
                action_video_fps_ratio=1,
                test_fraction=0.0,
                virtual_samples_per_episode=4,
                max_decode_retries=1,
                video_decode_backend="ffmpeg",
                action_magnitude_guard_threshold=100,
                seed=19,
            )
            with mock.patch(
                "motus_data.datasets.agibot_beta.decode_video_frames",
                side_effect=fake_decode,
            ):
                samples = dataset.get_grouped_samples([0, 1])

            self.assertEqual(len(decode_calls), 1)
            self.assertEqual(decode_calls[0][1], "ffmpeg")
            for sample in samples:
                start = sample["metadata"]["start_frame"]
                self.assertEqual(sample["frame_indices"], list(range(start, start + 5)))
                self.assertAlmostEqual(float(sample["first_frame"][0, 0, 0]), start / 100)
                expected_action = (
                    "reach for the cup" if start < 10 else "move the cup"
                )
                self.assertIn(f"Action: {expected_action}", sample["t5_input_message"])
                self.assertEqual(tuple(sample["initial_state"].shape), (26,))
                self.assertEqual(tuple(sample["action_sequence"].shape), (4, 26))
                self.assertEqual(sample["action_video_fps_ratio"], 1)

            sampler = dataset.create_batch_sampler(
                batch_size=2, shuffle=False, num_replicas=1, rank=0
            )
            for indices in sampler:
                episode_indices = {
                    index // dataset.virtual_samples_per_episode for index in indices
                }
                self.assertEqual(len(episode_indices), 1)

            config = OmegaConf.create({
                "common": {
                    "video_height": 8,
                    "video_width": 10,
                    "num_video_frames": 4,
                    "video_action_freq_ratio": 1,
                    "action_dim": 26,
                    "state_dim": 26,
                },
                "dataset": {
                    "type": "agibot_beta",
                    "name": "agibot_beta_fixture",
                    "source_dataset": "agibot_beta",
                    "data_root": str(data_root),
                    "tar_catalog_dir": str(catalog),
                    "split": "train",
                    "test_fraction": 0.0,
                    "virtual_samples_per_episode": 4,
                    "max_decode_retries": 1,
                    "video_decode_backend": "ffmpeg",
                    "action_magnitude_guard_threshold": 100,
                },
                "training": {"batch_size": 2},
                "system": {
                    "num_workers": 0,
                    "pin_memory": False,
                    "ready_batch_queue_enabled": True,
                    "ready_batch_queue_size": 2,
                    "ready_batch_queue_clone_tensors": True,
                    "ready_batch_queue_timeout_seconds": 10,
                },
            })
            with mock.patch(
                "motus_data.datasets.agibot_beta.decode_video_frames",
                side_effect=fake_decode,
            ):
                pipeline = build_data_pipeline(config, seed=23)
                with pipeline:
                    batch, queue_stats = pipeline.next()
            validate_motus_model_batch_v1(
                batch,
                expected_batch_size=2,
                num_video_frames=4,
                video_action_freq_ratio=1,
            )
            self.assertEqual(batch["source_datasets"], ["agibot_beta"] * 2)
            self.assertEqual(batch["action_video_fps_ratios"], [1, 1])
            self.assertTrue(queue_stats["ready_queue_enabled"])


if __name__ == "__main__":
    unittest.main()
