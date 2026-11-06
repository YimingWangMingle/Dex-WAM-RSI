from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

from motus_data import (
    AgibotEpisodeReader,
    EpisodeRepository,
    RobotmindEpisodeReader,
    RobotwinEpisodeReader,
)
from tests.test_repository import _hdf5_bytes


def _publish_catalog(
    root: Path,
    *,
    source: str,
    child: str,
    record: dict,
) -> Path:
    path = root / f"{child}.episode_catalog.jsonl"
    payload = (
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    path.write_bytes(payload)
    path.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "format": "motus_episode_catalog_v2",
                "catalog_id": f"{child}_fixture",
                "source": source,
                "child": child,
                "episodes": 1,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    return path


def _base_record(
    source: str,
    robot: str,
    video: dict,
    state: dict,
    action: dict,
) -> dict:
    return {
        "episode_id": f"{source}:fixture:0",
        "robot": robot,
        "video": video,
        "state": state,
        "action": action,
        "texts": [
            {
                "index": 0,
                "kind": "instruction_en",
                "text": "Move the object.",
            }
        ],
        "meta": {"fixture": True},
    }


class SourceReaderTests(unittest.TestCase):
    @staticmethod
    def _video_decoder(path: str, indices):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        return {
            int(index): np.full((4, 5, 3), int(index), dtype=np.uint8)
            for index in indices
        }

    def test_robotwin_reader_preserves_qpos_indices(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video_path = root / "episode.mp4"
            video_path.write_bytes(b"fixture")
            qpos_path = root / "qpos.pt"
            torch.save(torch.arange(30).reshape(6, 5).float(), qpos_path)
            record = _base_record(
                "robotwin",
                "fixture-robot",
                {
                    "count": 6,
                    "storage": {"type": "file", "path": str(video_path)},
                },
                {
                    "count": 6,
                    "storage": {"type": "file", "path": str(qpos_path)},
                    "meta": {"dimension": 5},
                },
                {
                    "count": 6,
                    "storage": {"type": "file", "path": str(qpos_path)},
                    "meta": {"dimension": 5},
                },
            )
            catalog = _publish_catalog(
                root, source="robotwin", child="robotwin_fixture", record=record
            )
            repository = EpisodeRepository(
                catalog,
                readers={
                    "robotwin": RobotwinEpisodeReader(
                        video_decoder=self._video_decoder
                    )
                },
            )
            try:
                episode = repository.get_episode(record["episode_id"])
                selected = episode.read_selection(
                    {
                        "episode_id": record["episode_id"],
                        "video_frame_indices": [4, 1, 4],
                        "state_indices": [2],
                        "action_indices": [3, 1],
                        "text_indices": [0],
                    },
                    video_output_format="rgb_float32_chw",
                    video_size=(10, 10),
                )
                self.assertEqual(tuple(selected.video.shape), (3, 3, 10, 10))
                self.assertTrue(torch.count_nonzero(selected.video[0, :, 0]) == 0)
                self.assertTrue(torch.count_nonzero(selected.video[0, :, 9]) == 0)
                self.assertTrue(
                    torch.allclose(
                        selected.video[0, :, 1:9],
                        torch.full((3, 8, 10), 4.0 / 255.0),
                    )
                )
                self.assertTrue(
                    torch.equal(selected.state, torch.arange(10, 15).reshape(1, 5).float())
                )
                self.assertTrue(
                    torch.equal(
                        selected.action,
                        torch.tensor(
                            [[15, 16, 17, 18, 19], [5, 6, 7, 8, 9]],
                            dtype=torch.float32,
                        ),
                    )
                )
            finally:
                repository.close()

    def test_robotwin_reader_allows_legacy_stretch_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video_path = root / "episode.mp4"
            video_path.write_bytes(b"fixture")
            qpos_path = root / "qpos.pt"
            torch.save(torch.zeros((1, 2)), qpos_path)
            record = _base_record(
                "robotwin",
                "fixture-robot",
                {
                    "count": 1,
                    "storage": {"type": "file", "path": str(video_path)},
                },
                {
                    "count": 1,
                    "storage": {"type": "file", "path": str(qpos_path)},
                    "meta": {"dimension": 2},
                },
                {
                    "count": 1,
                    "storage": {"type": "file", "path": str(qpos_path)},
                    "meta": {"dimension": 2},
                },
            )
            catalog = _publish_catalog(
                root, source="robotwin", child="robotwin_fixture", record=record
            )
            repository = EpisodeRepository(
                catalog,
                readers={
                    "robotwin": RobotwinEpisodeReader(
                        video_decoder=self._video_decoder,
                        resize_mode="interpolate",
                    )
                },
            )
            try:
                video = repository.get_episode(record["episode_id"]).read_video(
                    [0],
                    output_format="rgb_uint8_hwc",
                    video_size=(10, 10),
                )
                self.assertEqual(tuple(video.shape), (1, 10, 10, 3))
            finally:
                repository.close()

        with self.assertRaisesRegex(ValueError, "resize_mode"):
            RobotwinEpisodeReader(resize_mode="unknown")

    def test_robotmind_reader_uses_catalog_key_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            h5_path = root / "trajectory.hdf5"
            with h5py.File(h5_path, "w") as handle:
                dtype = h5py.vlen_dtype(np.dtype("uint8"))
                images = handle.create_dataset("camera/rgb", shape=(6,), dtype=dtype)
                for index in range(6):
                    image = Image.fromarray(
                        np.full((4, 5, 3), index * 10, dtype=np.uint8)
                    )
                    buffer = io.BytesIO()
                    image.save(buffer, format="JPEG")
                    images[index] = np.frombuffer(
                        buffer.getvalue(), dtype=np.uint8
                    )
                handle.create_dataset(
                    "puppet/left",
                    data=np.arange(12, dtype=np.float32).reshape(6, 2),
                )
                handle.create_dataset(
                    "puppet/right",
                    data=100 + np.arange(6, dtype=np.float32).reshape(6, 1),
                )
            numeric_meta = {
                "dimension": 3,
                "keys": ["puppet/left", "puppet/right"],
            }
            record = _base_record(
                "robotmind",
                "robotmind-fixture",
                {
                    "count": 6,
                    "storage": {"type": "hdf5", "path": str(h5_path)},
                    "meta": {"key": "camera/rgb"},
                },
                {
                    "count": 6,
                    "storage": {"type": "hdf5", "path": str(h5_path)},
                    "meta": numeric_meta,
                },
                {
                    "count": 6,
                    "storage": {"type": "hdf5", "path": str(h5_path)},
                    "meta": numeric_meta,
                },
            )
            catalog = _publish_catalog(
                root, source="robotmind", child="robotmind_fixture", record=record
            )
            repository = EpisodeRepository(
                catalog, readers={"robotmind": RobotmindEpisodeReader()}
            )
            try:
                episode = repository.get_episode(record["episode_id"])
                selected = episode.read_selection(
                    {
                        "episode_id": record["episode_id"],
                        "video_frame_indices": [3, 1, 3],
                        "state_indices": [2],
                        "action_indices": [4, 0],
                        "text_indices": [0],
                    },
                    video_output_format="rgb_float32_chw",
                    video_size=(2, 3),
                )
                self.assertEqual(tuple(selected.video.shape), (3, 3, 2, 3))
                self.assertEqual(selected.state.tolist(), [[4.0, 5.0, 102.0]])
                self.assertEqual(
                    selected.action.tolist(),
                    [[8.0, 9.0, 104.0], [0.0, 1.0, 100.0]],
                )
            finally:
                repository.close()

    def test_agibot_reader_keeps_video_and_proprio_index_spaces_independent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video_path = root / "head_color.mp4"
            video_path.write_bytes(b"fixture")
            proprio_path = root / "proprio.h5"
            proprio_path.write_bytes(_hdf5_bytes(12))
            record = _base_record(
                "agibot",
                "agibot-a2-d",
                {
                    "count": 6,
                    "storage": {"type": "file", "path": str(video_path)},
                },
                {
                    "count": 12,
                    "storage": {"type": "file", "path": str(proprio_path)},
                },
                {
                    "count": 12,
                    "storage": {"type": "file", "path": str(proprio_path)},
                },
            )
            catalog = _publish_catalog(
                root, source="agibot", child="agibot", record=record
            )
            repository = EpisodeRepository(
                catalog,
                readers={
                    "agibot": AgibotEpisodeReader(
                        video_decoder=self._video_decoder
                    )
                },
            )
            try:
                episode = repository.get_episode(record["episode_id"])
                selected = episode.read_selection(
                    {
                        "episode_id": record["episode_id"],
                        "video_frame_indices": [1, 4],
                        "state_indices": [6],
                        "action_indices": [7, 8],
                        "text_indices": [0],
                    }
                )
                self.assertEqual(tuple(selected.state.shape), (1, 26))
                self.assertEqual(tuple(selected.action.shape), (2, 26))
                self.assertEqual(selected.video_frame_indices, (1, 4))
                self.assertEqual(selected.state_indices, (6,))
            finally:
                repository.close()


if __name__ == "__main__":
    unittest.main()
