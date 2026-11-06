from __future__ import annotations

import io
import json
import struct
import tarfile
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from motus_data.catalog.agibot_beta import (
    build_agibot_beta_episode_catalog,
    validate_catalog_file_bindings,
)
from motus_data.iso_bmff import read_tar_member_video_sample_count


def box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", len(payload) + 8, kind) + payload


def fixture_mp4(frame_count: int) -> bytes:
    hdlr = box(b"hdlr", b"\x00\x00\x00\x00" + b"\x00" * 4 + b"vide")
    stsz = box(
        b"stsz",
        b"\x00\x00\x00\x00" + struct.pack(">II", 1, int(frame_count)),
    )
    stbl = box(b"stbl", stsz)
    minf = box(b"minf", stbl)
    mdia = box(b"mdia", hdlr + minf)
    trak = box(b"trak", mdia)
    moov = box(b"moov", trak)
    return box(b"ftyp", b"isom\x00\x00\x00\x00isom") + box(b"mdat", b"x" * 32) + moov


def hdf5_bytes(timesteps: int) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".h5") as temporary:
        with h5py.File(temporary.name, "w") as handle:
            handle.create_dataset(
                "state/end/position",
                data=np.zeros((timesteps, 2, 3), dtype=np.float32),
            )
            orientation = np.zeros((timesteps, 2, 4), dtype=np.float32)
            orientation[..., 3] = 1
            handle.create_dataset("state/end/orientation", data=orientation)
            handle.create_dataset(
                "state/effector/position",
                data=np.zeros((timesteps, 2), dtype=np.float32),
            )
        temporary.seek(0)
        return temporary.read()


def add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


class AgibotBetaCatalogBuilderTests(unittest.TestCase):
    def test_iso_bmff_reads_video_count_from_bounded_tar_member(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tar_path = Path(temp_dir) / "video.tar"
            with tarfile.open(tar_path, "w") as archive:
                add_bytes(archive, "episode/video.mp4", fixture_mp4(37))
            with tarfile.open(tar_path, "r") as archive:
                member = archive.getmember("episode/video.mp4")
            self.assertEqual(
                read_tar_member_video_sample_count(
                    tar_path, member.offset_data, member.size
                ),
                37,
            )

    def test_builds_atomic_catalog_with_independent_modalities_and_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "agibot_beta"
            task_root = data_root / "task_info"
            cache_root = root / "cache"
            task_root.mkdir(parents=True)
            cache_root.mkdir()
            task_id = 327
            episode_ids = (648642, 648649)

            video_tar = root / "videos.tar"
            proprio_tar = root / "proprio.tar"
            with tarfile.open(video_tar, "w") as archive:
                for episode_id in episode_ids:
                    add_bytes(
                        archive,
                        f"{episode_id}/videos/head_color.mp4",
                        fixture_mp4(6),
                    )
            with tarfile.open(proprio_tar, "w") as archive:
                for episode_id in episode_ids:
                    add_bytes(
                        archive,
                        f"{task_id}/{episode_id}/proprio_stats.h5",
                        hdf5_bytes(6),
                    )

            video_rows = []
            proprio_rows = []
            with tarfile.open(video_tar, "r") as video_archive, tarfile.open(
                proprio_tar, "r"
            ) as proprio_archive:
                for episode_id in episode_ids:
                    video_member = video_archive.getmember(
                        f"{episode_id}/videos/head_color.mp4"
                    )
                    proprio_member = proprio_archive.getmember(
                        f"{task_id}/{episode_id}/proprio_stats.h5"
                    )
                    video_rows.append(
                        {
                            "task_id": task_id,
                            "episode_id": episode_id,
                            "uri": str(video_tar),
                            "member": video_member.name,
                            "tar_offset_data": video_member.offset_data,
                            "tar_size": video_member.size,
                        }
                    )
                    proprio_rows.append(
                        {
                            "task_id": task_id,
                            "episode_id": episode_id,
                            "uri": str(proprio_tar),
                            "member": proprio_member.name,
                            "tar_offset_data": proprio_member.offset_data,
                            "tar_size": proprio_member.size,
                        }
                    )

            for path, rows in (
                (cache_root / f"video_task_{task_id}.jsonl", video_rows),
                (cache_root / "proprio.jsonl", proprio_rows),
            ):
                path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
                path.with_suffix(path.suffix + ".done").touch()

            annotations = []
            for episode_id in episode_ids:
                annotations.append(
                    {
                        "episode_id": episode_id,
                        "task_name": "Water the flowers",
                        "init_scene_text": "Fixture scene",
                        "label_info": {
                            "action_config": [
                                {
                                    "start_frame": 0,
                                    "end_frame": 3,
                                    "action_text": "First action.",
                                    "skill": "FirstSkill",
                                },
                                {
                                    "start_frame": 3,
                                    "end_frame": 6,
                                    "action_text": "Second action.",
                                    "skill": "SecondSkill",
                                },
                            ]
                        },
                    }
                )
            annotations.append(json.loads(json.dumps(annotations[0])))
            (task_root / f"task_{task_id}.json").write_text(
                json.dumps(annotations), encoding="utf-8"
            )

            progress = []
            output = root / "agibot_beta.episode_catalog.jsonl"
            metadata = build_agibot_beta_episode_catalog(
                data_root=data_root,
                tar_catalog_dir=cache_root,
                output_path=output,
                catalog_id="fixture_catalog_v2",
                data_version="fixture_data_v2",
                fail_on_episode_error=True,
                progress_callback=lambda state: progress.append(dict(state)),
            )

            metadata_path = output.with_suffix(".meta.json")
            self.assertTrue(output.is_file())
            self.assertTrue(metadata_path.is_file())
            self.assertFalse(list(root.glob(".*.tmp.*")))
            self.assertEqual(metadata["episodes"], 2)
            self.assertEqual(
                metadata["format"],
                "motus_episode_catalog_v2",
            )
            self.assertNotIn("split_policy", metadata)
            self.assertNotIn("schema_version", metadata)
            self.assertNotIn("record_schema_version", metadata)
            self.assertEqual(
                set(metadata),
                {
                    "format",
                    "catalog_id",
                    "source",
                    "child",
                    "episodes",
                    "sha256",
                    "meta",
                },
            )
            build = metadata["meta"]["build"]
            self.assertNotIn("split_counts", build)
            source_scan = build["source_scan"]
            self.assertEqual(source_scan["annotation_episodes"], 3)
            self.assertEqual(source_scan["unique_annotation_episodes"], 2)
            self.assertEqual(source_scan["duplicate_annotation_exact"], 1)
            self.assertEqual(source_scan["excluded_bad_task"], 0)
            self.assertEqual(progress[-1]["processed"], 2)
            self.assertEqual(validate_catalog_file_bindings(output, metadata), 2)

            records = [json.loads(line) for line in output.read_text().splitlines()]
            first = records[0]
            self.assertNotIn("schema_version", first)
            self.assertNotIn("source_dataset", first)
            self.assertNotIn("source_child", first)
            self.assertNotIn("split", first)
            self.assertEqual(
                set(first),
                {
                    "episode_id",
                    "robot",
                    "video",
                    "state",
                    "action",
                    "texts",
                    "meta",
                },
            )
            self.assertEqual(first["video"]["count"], 6)
            self.assertEqual(first["state"]["count"], 6)
            self.assertEqual(first["action"]["count"], 6)
            self.assertEqual(len(first["texts"]), 3)
            action_item = first["texts"][1]
            self.assertEqual(action_item["video_range"], [0, 3])
            self.assertEqual(action_item["meta"]["action_text"], "First action.")
            self.assertEqual(action_item["meta"]["skill"], "FirstSkill")
            self.assertEqual(action_item["meta"]["source_segment_index"], 0)
            self.assertEqual(first["meta"]["annotation_occurrence_count"], 2)
            self.assertEqual(len(first["meta"]["annotation_locations"]), 2)

            annotations[-1]["task_name"] = "Conflicting task"
            (task_root / f"task_{task_id}.json").write_text(
                json.dumps(annotations), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
                build_agibot_beta_episode_catalog(
                    data_root=data_root,
                    tar_catalog_dir=cache_root,
                    output_path=root / "conflict.episode_catalog.jsonl",
                    catalog_id="conflict_catalog_v2",
                    data_version="fixture_data_v2",
                    fail_on_episode_error=True,
                )


if __name__ == "__main__":
    unittest.main()
