from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from motus_data.batch_adapters import (
    BatchAdapterError,
    IndexedBatchCollator,
    MotusModelBatchV1Adapter,
)
from motus_data.catalog.openvid import (
    build_openvid_episode_catalog,
    discover_openvid_archives,
)
from motus_data.openvid import OpenVidEpisodeReader, open_openvid_zip
from motus_data.repository import EpisodeRepository
from motus_data.sample_pool import SamplePoolDataset


CSV_FIELDS = (
    "video",
    "caption",
    "aesthetic score",
    "motion score",
    "temporal consistency score",
    "camera motion",
    "frame",
    "fps",
    "seconds",
)


def _zip_payload(member: str, payload: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.writestr(member, payload)
    return buffer.getvalue()


def _write_csv(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(
            [
                {
                    "video": "normal.mp4",
                    "caption": "A normal archive video.",
                    "aesthetic score": "5.5",
                    "motion score": "3.0",
                    "temporal consistency score": "0.9",
                    "camera motion": "static",
                    "frame": "12",
                    "fps": "6",
                    "seconds": "2",
                },
                {
                    "video": "split.mp4",
                    "caption": "A split archive video.",
                    "aesthetic score": "6",
                    "motion score": "4",
                    "temporal consistency score": "0.8",
                    "camera motion": "pan",
                    "frame": "18",
                    "fps": "6",
                    "seconds": "3",
                },
            ]
        )


def _write_pool(path: Path, catalog_id: str) -> None:
    rows = [
        {
            "format": "motus_sample_pool_v2",
            "pool_id": "openvid_fixture_pool",
            "catalog_id": catalog_id,
            "samples": 2,
        },
        {
            "episode_id": "openvid:split.mp4",
            "video_frame_indices": [0, 2, 4],
            "state_indices": [],
            "action_indices": [],
            "text_indices": [0],
        },
        {
            "episode_id": "openvid:split.mp4",
            "video_frame_indices": [2, 4, 6],
            "state_indices": [],
            "action_indices": [],
            "text_indices": [0],
        },
    ]
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


class OpenVidTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path]:
        data_root = root / "OpenVid"
        data_root.mkdir()
        (data_root / "OpenVid_part0.zip").write_bytes(
            _zip_payload(
                "mnt/source/normal.mp4",
                b"normal-mp4-placeholder",
            )
        )
        split_payload = _zip_payload(
            "mnt/source/split.mp4",
            b"split-mp4-placeholder",
        )
        split_at = len(split_payload) // 2
        (data_root / "OpenVid_part1_partaa").write_bytes(
            split_payload[:split_at]
        )
        (data_root / "OpenVid_part1_partab").write_bytes(
            split_payload[split_at:]
        )
        # The HD collection is a duplicate subset and must never be discovered.
        (data_root / "OpenVidHD_part0.zip").write_bytes(
            _zip_payload("duplicate.mp4", b"duplicate")
        )
        csv_path = root / "OpenVid-1M.csv"
        _write_csv(csv_path)
        return data_root, csv_path

    def test_split_zip_is_seekable_without_reconstruction(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root, csv_path = self._fixture(root)
            archives = discover_openvid_archives(data_root)
            self.assertEqual(
                [archive.storage_type for archive in archives],
                ["zip", "zip_split"],
            )
            split = archives[1]
            with open_openvid_zip(
                split.logical_path,
                parts=split.parts,
            ) as archive:
                self.assertEqual(
                    archive.read("mnt/source/split.mp4"),
                    b"split-mp4-placeholder",
                )
            output = root / "split_only.episode_catalog.jsonl"
            metadata = build_openvid_episode_catalog(
                data_root=data_root,
                csv_path=csv_path,
                output_path=output,
                catalog_id="openvid_split_only",
                data_version="fixture",
                archive_numbers=[1],
                max_episodes=1,
            )
            self.assertEqual(metadata["episodes"], 1)
            self.assertEqual(
                metadata["meta"]["build"]["archive_filtered_rows"],
                1,
            )
            record = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(record["episode_id"], "openvid:split.mp4")

    def test_complete_video_catalog_to_grouped_model_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root, csv_path = self._fixture(root)
            catalog_path = root / "openvid.episode_catalog.jsonl"
            progress = []
            metadata = build_openvid_episode_catalog(
                data_root=data_root,
                csv_path=csv_path,
                output_path=catalog_path,
                catalog_id="openvid_fixture_v1",
                data_version="fixture",
                progress_callback=lambda state: progress.append(dict(state)),
            )
            self.assertEqual(metadata["episodes"], 2)
            self.assertEqual(
                metadata["meta"]["build"]["zip_inventory"]["split_archives"],
                1,
            )
            self.assertTrue(progress[-1]["done"])
            records = [
                json.loads(line)
                for line in catalog_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(
                [record["episode_id"] for record in records],
                ["openvid:normal.mp4", "openvid:split.mp4"],
            )
            self.assertEqual(records[1]["video"]["storage"]["type"], "zip_split")
            self.assertEqual(len(records[1]["video"]["storage"]["parts"]), 2)
            for record in records:
                self.assertEqual(record["state"]["count"], 0)
                self.assertEqual(record["action"]["count"], 0)
                self.assertEqual(record["state"]["storage"]["type"], "none")
                self.assertEqual(record["action"]["storage"]["type"], "none")

            decode_calls = []

            def video_decoder(path: str, indices):
                self.assertTrue(Path(path).is_file())
                decode_calls.append(tuple(indices))
                return {
                    int(index): np.full(
                        (4, 5, 3),
                        int(index),
                        dtype=np.uint8,
                    )
                    for index in indices
                }

            repository = EpisodeRepository(
                catalog_path,
                readers={
                    "openvid": OpenVidEpisodeReader(
                        video_decoder=video_decoder
                    )
                },
            )
            try:
                pool_path = root / "openvid.sample_pool.jsonl"
                _write_pool(pool_path, repository.catalog_id)
                dataset = SamplePoolDataset(
                    repository,
                    pool_path,
                    video_output_format="rgb_float32_chw",
                    video_size=(4, 5),
                )
                sampler = dataset.create_batch_sampler(
                    batch_size=2,
                    drop_last=True,
                    shuffle=False,
                )
                adapter = MotusModelBatchV1Adapter(
                    num_video_frames=2,
                    action_video_fps_ratio=1,
                    action_loss_weight=0.0,
                    state_action_policy="absent",
                    allow_missing_state_action=True,
                    placeholder_state_action_dim=26,
                )
                loader = DataLoader(
                    dataset,
                    batch_sampler=sampler,
                    collate_fn=IndexedBatchCollator(adapter),
                    num_workers=0,
                )
                batch = next(iter(loader))
                self.assertEqual(decode_calls, [(0, 2, 4, 6)])
                self.assertEqual(
                    tuple(batch["video_frames"].shape),
                    (2, 2, 3, 4, 5),
                )
                self.assertEqual(
                    tuple(batch["initial_state"].shape),
                    (2, 26),
                )
                self.assertEqual(
                    tuple(batch["action_sequence"].shape),
                    (2, 2, 26),
                )
                self.assertTrue(
                    torch.equal(
                        batch["action_loss_weights"],
                        torch.zeros(2),
                    )
                )
                self.assertTrue(
                    all(
                        item["synthetic_state_action"]
                        for item in batch["metadata"]
                    )
                )

                strict_adapter = MotusModelBatchV1Adapter(
                    num_video_frames=2,
                    action_loss_weight=0.0,
                )
                indexed = dataset.get_grouped_batch([0, 1])
                with self.assertRaisesRegex(
                    BatchAdapterError,
                    "does not allow missing state/action",
                ):
                    strict_adapter.adapt(indexed)
            finally:
                repository.close()


if __name__ == "__main__":
    unittest.main()
