from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from motus_data.batch_adapters import (
    IndexedBatchCollator,
    build_indexed_batch_adapter,
)
from motus_data.catalog.libero import build_libero_episode_catalog
from motus_data.contracts import validate_motus_native_multimodal_batch_v1
from motus_data.libero import LiberoParquetEpisodeReader
from motus_data.repository import EpisodeRepository
from motus_data.sample_pool import SamplePoolDataset


PYARROW_AVAILABLE = importlib.util.find_spec("pyarrow") is not None


def _png(value: int, *, height: int = 4, width: int = 5) -> bytes:
    array = np.full((height, width, 3), value, dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


@unittest.skipUnless(PYARROW_AVAILABLE, "pyarrow is not installed")
class LiberoTests(unittest.TestCase):
    @staticmethod
    def _fixture(root: Path) -> tuple[Path, np.ndarray, np.ndarray]:
        import pyarrow as pa
        import pyarrow.parquet as pq

        meta = root / "meta"
        data = root / "data" / "chunk-000"
        meta.mkdir(parents=True)
        data.mkdir(parents=True)

        tasks = [
            {"task_index": index, "task": f"LIBERO task {index}"}
            for index in range(40)
        ]
        episodes = []
        total_frames = 0
        for episode_index in range(1693):
            if episode_index < 379:
                task_index = episode_index % 10
            elif episode_index < 807:
                task_index = 10 + (episode_index - 379) % 10
            elif episode_index < 1261:
                task_index = 20 + (episode_index - 807) % 10
            else:
                task_index = 30 + (episode_index - 1261) % 10
            length = 4 if episode_index == 0 else 1
            episodes.append(
                {
                    "episode_index": episode_index,
                    "tasks": [f"LIBERO task {task_index}"],
                    "length": length,
                }
            )
            total_frames += length

        info = {
            "codebase_version": "v2.0",
            "robot_type": "panda",
            "total_episodes": len(episodes),
            "total_frames": total_frames,
            "total_tasks": len(tasks),
            "total_videos": 0,
            "total_chunks": 2,
            "chunks_size": 1000,
            "fps": 10,
            "data_path": (
                "data/chunk-{episode_chunk:03d}/"
                "episode_{episode_index:06d}.parquet"
            ),
        }
        (meta / "info.json").write_text(
            json.dumps(info),
            encoding="utf-8",
        )
        for name, rows in (("tasks.jsonl", tasks), ("episodes.jsonl", episodes)):
            (meta / name).write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
        statistics = {
            "state": {
                "mean": list(range(8)),
                "std": [2.0] * 8,
            },
            "actions": {
                "mean": list(range(100, 107)),
                "std": [4.0] * 7,
            },
        }
        (meta / "stats.json").write_text(
            json.dumps(statistics) + "\n",
            encoding="utf-8",
        )

        state = np.arange(32, dtype=np.float32).reshape(4, 8)
        action = (100 + np.arange(28, dtype=np.float32)).reshape(4, 7)
        image_type = pa.struct(
            [pa.field("bytes", pa.binary()), pa.field("path", pa.string())]
        )
        table = pa.table(
            {
                "image": pa.array(
                    [
                        {
                            "bytes": _png(index * 10),
                            "path": f"image/{index:06d}.png",
                        }
                        for index in range(4)
                    ],
                    type=image_type,
                ),
                "wrist_image": pa.array(
                    [
                        {
                            "bytes": _png(100 + index * 10),
                            "path": f"wrist_image/{index:06d}.png",
                        }
                        for index in range(4)
                    ],
                    type=image_type,
                ),
                "state": pa.array(
                    state.tolist(),
                    type=pa.list_(pa.float32(), 8),
                ),
                "actions": pa.array(
                    action.tolist(),
                    type=pa.list_(pa.float32(), 7),
                ),
                "timestamp": pa.array(
                    np.arange(4, dtype=np.float32) / 10,
                    type=pa.float32(),
                ),
                "frame_index": pa.array(np.arange(4), type=pa.int64()),
                "episode_index": pa.array(np.zeros(4), type=pa.int64()),
                "index": pa.array(np.arange(4), type=pa.int64()),
                "task_index": pa.array(np.zeros(4), type=pa.int64()),
            }
        )
        parquet_path = data / "episode_000000.parquet"
        pq.write_table(table, parquet_path, row_group_size=2)
        return parquet_path, state, action

    def test_catalog_and_exact_episode_reads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            parquet_path, state, action = self._fixture(root)
            catalog = root / "libero_10.episode_catalog.jsonl"
            progress = []
            metadata = build_libero_episode_catalog(
                root,
                "libero_10",
                catalog,
                catalog_id="libero_10_fixture_v1",
                data_version="fixture",
                max_episodes=1,
                progress_callback=lambda value: progress.append(dict(value)),
            )
            self.assertEqual(metadata["source"], "libero")
            self.assertEqual(metadata["child"], "libero_10")
            self.assertEqual(metadata["episodes"], 1)
            self.assertEqual(metadata["meta"]["build"]["frames"], 4)
            self.assertEqual(
                metadata["meta"]["normalization"]["default_mode"],
                "zscore",
            )
            self.assertEqual(
                metadata["meta"]["normalization"]["modalities"]["state"][
                    "dimension"
                ],
                8,
            )
            self.assertEqual(
                metadata["meta"]["normalization"]["modalities"]["action"][
                    "dimension"
                ],
                7,
            )
            self.assertTrue(progress[-1]["done"])

            record = json.loads(catalog.read_text(encoding="utf-8"))
            self.assertEqual(
                record["episode_id"],
                "libero:libero_10:000000",
            )
            self.assertEqual(record["video"]["storage"]["type"], "parquet")
            self.assertEqual(
                record["video"]["storage"]["path"],
                str(parquet_path),
            )
            self.assertEqual(record["state"]["meta"]["dimension"], 8)
            self.assertEqual(record["action"]["meta"]["dimension"], 7)

            repository = EpisodeRepository(
                catalog,
                readers={
                    "libero": LiberoParquetEpisodeReader(
                        normalization_mode="raw"
                    )
                },
            )
            try:
                episode = repository.get_episode(record["episode_id"])
                self.assertEqual(
                    episode.video.schema.metadata["camera_mode"],
                    "main",
                )
                self.assertEqual(
                    episode.state.schema.metadata["normalization"],
                    "raw",
                )
                selection = episode.read_selection(
                    {
                        "episode_id": record["episode_id"],
                        "video_frame_indices": [3, 0, 3],
                        "state_indices": [2, 0],
                        "action_indices": [1, 3],
                        "text_indices": [0],
                    }
                )
                self.assertEqual(tuple(selection.video.shape), (3, 4, 5, 3))
                self.assertEqual(
                    selection.video[:, 0, 0, 0].tolist(),
                    [30, 0, 30],
                )
                self.assertTrue(
                    torch.equal(selection.state, torch.from_numpy(state[[2, 0]]))
                )
                self.assertTrue(
                    torch.equal(selection.action, torch.from_numpy(action[[1, 3]]))
                )
                self.assertEqual(selection.text[0].text, "LIBERO task 0")
            finally:
                repository.close()

    def test_camera_modes_and_aspect_preserving_resize(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._fixture(root)
            catalog = root / "libero_10.episode_catalog.jsonl"
            build_libero_episode_catalog(
                root,
                "libero_10",
                catalog,
                catalog_id="libero_camera_fixture_v1",
                data_version="fixture",
                max_episodes=1,
            )
            episode_id = "libero:libero_10:000000"
            wrist_repository = EpisodeRepository(
                catalog,
                readers={
                    "libero": LiberoParquetEpisodeReader(
                        camera_mode="wrist"
                    )
                },
            )
            try:
                wrist = wrist_repository.get_episode(episode_id).read_video([1])
                self.assertEqual(tuple(wrist.shape), (1, 4, 5, 3))
                self.assertEqual(int(wrist[0, 0, 0, 0]), 110)
            finally:
                wrist_repository.close()

            concat_repository = EpisodeRepository(
                catalog,
                readers={
                    "libero": LiberoParquetEpisodeReader(
                        camera_mode="main_wrist_horizontal"
                    )
                },
            )
            try:
                episode = concat_repository.get_episode(episode_id)
                concat = episode.read_video([2])
                self.assertEqual(tuple(concat.shape), (1, 4, 10, 3))
                self.assertEqual(int(concat[0, 0, 0, 0]), 20)
                self.assertEqual(int(concat[0, 0, 9, 0]), 120)
                resized = episode.read_video(
                    [2],
                    output_format="rgb_float32_chw",
                    video_size=(4, 5),
                )
                self.assertEqual(tuple(resized.shape), (1, 3, 4, 5))
                self.assertEqual(
                    int(torch.count_nonzero(resized[:, :, 0, :])),
                    0,
                )
            finally:
                concat_repository.close()

    def test_reader_options_validate_camera_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "camera_mode"):
            LiberoParquetEpisodeReader(camera_mode="unsupported")
        with self.assertRaisesRegex(ValueError, "resize_mode"):
            LiberoParquetEpisodeReader(resize_mode="unsupported")
        with self.assertRaisesRegex(ValueError, "normalization_mode"):
            LiberoParquetEpisodeReader(normalization_mode="unsupported")

    def test_zscore_is_default_raw_is_optional_and_stats_are_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _, state, action = self._fixture(root)
            catalog = root / "libero_10.episode_catalog.jsonl"
            build_libero_episode_catalog(
                root,
                "libero_10",
                catalog,
                catalog_id="libero_normalization_fixture_v1",
                data_version="fixture",
                max_episodes=1,
            )
            repository = EpisodeRepository(catalog)
            try:
                episode = repository.get_episode(
                    "libero:libero_10:000000"
                )
                self.assertEqual(
                    episode.state.schema.metadata["normalization"],
                    "zscore",
                )
                expected_state = (
                    torch.from_numpy(state[[0, 2]])
                    - torch.arange(8, dtype=torch.float32)
                ) / 2.0
                expected_action = (
                    torch.from_numpy(action[[0, 1]])
                    - torch.arange(100, 107, dtype=torch.float32)
                ) / 4.0
                self.assertTrue(
                    torch.equal(
                        episode.state.read([0, 2]),
                        expected_state,
                    )
                )
                self.assertTrue(
                    torch.equal(
                        episode.action.read([0, 1]),
                        expected_action,
                    )
                )
            finally:
                repository.close()

            stats_path = root / "meta" / "stats.json"
            stats_path.write_text(
                stats_path.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            tampered = EpisodeRepository(catalog)
            try:
                with self.assertRaisesRegex(
                    ValueError, "SHA256 differs"
                ):
                    tampered.get_episode(
                        "libero:libero_10:000000"
                    ).state.read([0])
            finally:
                tampered.close()

    def test_native_adapter_stacks_independent_state_action_dims(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._fixture(root)
            catalog = root / "libero_10.episode_catalog.jsonl"
            build_libero_episode_catalog(
                root,
                "libero_10",
                catalog,
                catalog_id="libero_native_adapter_fixture_v1",
                data_version="fixture",
                max_episodes=1,
            )
            episode_id = "libero:libero_10:000000"
            rows = [
                {
                    "format": "motus_sample_pool_v2",
                    "pool_id": "libero_native_adapter_pool",
                    "catalog_id": "libero_native_adapter_fixture_v1",
                    "samples": 2,
                },
                {
                    "episode_id": episode_id,
                    "video_frame_indices": [0, 1, 2],
                    "state_indices": [0],
                    "action_indices": [0, 1],
                    "text_indices": [0],
                },
                {
                    "episode_id": episode_id,
                    "video_frame_indices": [1, 2, 3],
                    "state_indices": [1],
                    "action_indices": [1, 2],
                    "text_indices": [0],
                },
            ]
            pool = root / "libero.sample_pool.jsonl"
            pool.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            repository = EpisodeRepository(catalog)
            try:
                dataset = SamplePoolDataset(
                    repository,
                    pool,
                    video_output_format="rgb_float32_chw",
                    video_size=(4, 5),
                )
                adapter = build_indexed_batch_adapter(
                    "motus_native_multimodal_batch_v1",
                    num_video_frames=2,
                    action_video_fps_ratio=1,
                )
                loader = DataLoader(
                    dataset,
                    batch_sampler=dataset.create_batch_sampler(
                        batch_size=2,
                        drop_last=True,
                        shuffle=False,
                    ),
                    collate_fn=IndexedBatchCollator(adapter),
                    num_workers=0,
                )
                batch = next(iter(loader))
                validate_motus_native_multimodal_batch_v1(
                    batch,
                    expected_batch_size=2,
                    num_video_frames=2,
                    video_action_freq_ratio=1,
                    expected_state_dimension=8,
                    expected_action_dimension=7,
                )
                self.assertEqual(
                    tuple(batch["initial_state"].shape),
                    (2, 8),
                )
                self.assertEqual(
                    tuple(batch["action_sequence"].shape),
                    (2, 2, 7),
                )
                self.assertEqual(
                    batch["normalization_modes"],
                    {"state": "zscore", "action": "zscore"},
                )
            finally:
                repository.close()


if __name__ == "__main__":
    unittest.main()
