from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from motus_data.catalog.egoverse import (
    build_egoverse_episode_catalog,
    scan_egoverse_episode_inventory,
)
from motus_data.egoverse import (
    EGOVERSE_PROFILES,
    EgoVerseEpisodeReader,
    compose_egoverse_text,
    get_egoverse_profile,
)
from motus_data.repository import EpisodeRepository, TextItem


class EgoVerseTextComposerTests(unittest.TestCase):
    """compose_egoverse_text：可选 episode 级任务描述 + 窗口内全部叙述片段。"""

    def test_task_and_all_annotations(self) -> None:
        items = [
            TextItem(0, "task_description_en", "Make coffee", None, {}),
            TextItem(1, "annotation_en", "person opens the drawer", (0, 90), {}),
            TextItem(2, "annotation_en", "person takes out a pod.", (90, 150), {}),
        ]
        self.assertEqual(
            compose_egoverse_text(items),
            "Task: Make coffee.\n"
            "Action: person opens the drawer.\n"
            "Action: person takes out a pod.",
        )

    def test_missing_task_description_is_skipped(self) -> None:
        items = [
            TextItem(1, "annotation_en", "person opens the drawer", (0, 90), {}),
        ]
        self.assertEqual(
            compose_egoverse_text(items),
            "Action: person opens the drawer.",
        )

    def test_unknown_kind_is_kept_verbatim(self) -> None:
        items = [TextItem(0, "mystery_kind", "raw text", None, {})]
        self.assertEqual(compose_egoverse_text(items), "raw text.")

    def test_empty_input_composes_empty(self) -> None:
        self.assertEqual(compose_egoverse_text([]), "")

    def test_registered_for_egoverse_source(self) -> None:
        import motus_data

        self.assertIs(
            motus_data.resolve_text_composer("egoverse"),
            compose_egoverse_text,
        )


class _FakeArray:
    def __init__(self, values: Any):
        self.values = np.asarray(values)
        self.shape = self.values.shape
        self.dtype = self.values.dtype

    def __getitem__(self, key: Any) -> Any:
        return self.values[key]


class _BrokenArray(_FakeArray):
    def __getitem__(self, key: Any) -> Any:
        raise ValueError("checksum mismatch")


class _FakeGroup(dict[str, _FakeArray]):
    def __init__(self, values: dict[str, Any], attrs: dict[str, Any]):
        super().__init__(
            (key, _FakeArray(value)) for key, value in values.items()
        )
        self.attrs = dict(attrs)


def _jpeg(value: int) -> bytes:
    image = Image.fromarray(
        np.full((4, 5, 3), value, dtype=np.uint8)
    )
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def _write_zarr_root(
    root: Path,
    *,
    episode_name: str = "fixture_episode",
    embodiment: str = "aria_bimanual",
    total_frames: int = 6,
) -> Path:
    store = root / f"{episode_name}.zarr"
    store.mkdir(parents=True)
    attributes = {
        "embodiment": embodiment,
        "total_frames": total_frames,
        "fps": 30,
        "task_description": "Pick up the red cup.",
        "task_name": "pick_cup",
        "features": {
            "images.front_1": {},
            "left.obs_ee_pose": {},
            "right.obs_ee_pose": {},
        },
    }
    (store / "zarr.json").write_text(
        json.dumps(
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": attributes,
            }
        ),
        encoding="utf-8",
    )
    return store


def _aria_group(
    *,
    total_frames: int = 6,
    physical_frames: int = 8,
) -> _FakeGroup:
    attributes = {
        "embodiment": "aria_bimanual",
        "total_frames": total_frames,
        "fps": 30,
        "task_description": "Pick up the red cup.",
        "task_name": "pick_cup",
        "features": {
            "images.front_1": {},
            "left.obs_ee_pose": {},
            "right.obs_ee_pose": {},
        },
    }
    annotations = [
        json.dumps(
            {
                "start_idx": 1,
                "end_idx": 5,
                "text": "Reach for the cup.",
            }
        ).encode("utf-8")
    ]
    return _FakeGroup(
        {
            "images.front_1": [
                _jpeg(index * 10) for index in range(physical_frames)
            ],
            "left.obs_ee_pose": np.arange(
                physical_frames * 7, dtype=np.float32
            ).reshape(physical_frames, 7),
            "right.obs_ee_pose": (
                1000
                + np.arange(physical_frames * 7, dtype=np.float32)
            ).reshape(physical_frames, 7),
            "annotations": np.asarray(annotations, dtype=object),
        },
        attributes,
    )


class EgoVerseTests(unittest.TestCase):
    def test_profiles_freeze_raw_state_action_dimensions(self) -> None:
        expected = {
            "aria_bimanual": (14, 14),
            "aria_left_arm": (7, 7),
            "aria_right_arm": (7, 7),
            "eva_bimanual": (16, 16),
            "eva_left_arm": (8, 8),
            "eva_right_arm": (8, 8),
            "scale_bimanual": (14, 14),
            "mecka_freeform_bimanual": (14, 14),
            "mecka_flagship_bimanual": (14, 14),
        }
        self.assertEqual(set(EGOVERSE_PROFILES), set(expected))
        for name, dimensions in expected.items():
            profile = get_egoverse_profile(f"egoverse-{name}")
            self.assertEqual(
                (profile.state_dim, profile.action_dim),
                dimensions,
            )
            self.assertEqual(
                profile.action_is_control_command,
                name.startswith("eva_"),
            )
        with self.assertRaisesRegex(ValueError, "resize_mode"):
            EgoVerseEpisodeReader(resize_mode="unknown")

    def test_builder_and_reader_use_total_frames_and_model_indices(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = _write_zarr_root(root)
            group = _aria_group()
            groups = {str(store): group}
            opener = lambda path: groups[str(path)]
            output = root / "aria.episode_catalog.jsonl"
            progress = []

            metadata = build_egoverse_episode_catalog(
                data_root=root,
                profile="aria_bimanual",
                output_path=output,
                catalog_id="egoverse_aria_fixture",
                data_version="fixture",
                group_opener=opener,
                progress_callback=lambda state: progress.append(dict(state)),
            )

            self.assertEqual(metadata["episodes"], 1)
            self.assertEqual(metadata["source"], "egoverse")
            self.assertTrue(progress[-1]["done"])
            build = metadata["meta"]["build"]
            self.assertEqual(build["quarantine"]["episodes"], 0)
            self.assertTrue(
                output.with_suffix(".quarantine.jsonl").is_file()
            )
            self.assertEqual(
                build["quality"]["text_source_counts"],
                {
                    "annotation_and_episode_text": 1,
                    "annotation_only": 0,
                    "episode_text_only": 0,
                    "no_text": 0,
                },
            )
            self.assertEqual(build["quality"]["total_frames"], 6)
            self.assertEqual(
                build["profile_contract"]["action_source_signal"],
                "observed_ee_pose_timeline",
            )
            self.assertFalse(
                build["profile_contract"]["action_is_control_command"]
            )
            record = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(record["video"]["count"], 6)
            self.assertEqual(record["state"]["count"], 6)
            self.assertEqual(record["action"]["count"], 6)
            self.assertEqual(record["state"]["meta"]["dimension"], 14)
            self.assertEqual(
                record["state"]["meta"]["keys"],
                ["left.obs_ee_pose", "right.obs_ee_pose"],
            )
            self.assertEqual(
                [item["kind"] for item in record["texts"]],
                ["task_description_en", "annotation_en"],
            )
            self.assertEqual(record["texts"][1]["video_range"], [1, 5])
            self.assertEqual(
                record["action"]["meta"]["source_signal"],
                "observed_ee_pose_timeline",
            )
            self.assertFalse(
                record["action"]["meta"]["is_control_command"]
            )
            self.assertEqual(
                record["action"]["meta"]["indices"],
                "model_provided",
            )

            repository = EpisodeRepository(
                output,
                readers={
                    "egoverse": EgoVerseEpisodeReader(
                        group_opener=opener
                    )
                },
            )
            try:
                episode = repository.get_episode(record["episode_id"])
                self.assertEqual(episode.info.video_count, 6)
                self.assertEqual(episode.info.state_count, 6)
                selection = episode.read_selection(
                    {
                        "episode_id": record["episode_id"],
                        "video_frame_indices": [4, 1, 4],
                        "state_indices": [0, 3],
                        "action_indices": [2, 5],
                        "text_indices": [1],
                    },
                    video_output_format="rgb_float32_chw",
                    video_size=(2, 3),
                )
                self.assertEqual(
                    tuple(selection.video.shape), (3, 3, 2, 3)
                )
                self.assertEqual(
                    int(torch.count_nonzero(selection.video[0, :, :, 2])),
                    0,
                )
                self.assertEqual(tuple(selection.state.shape), (2, 14))
                self.assertEqual(tuple(selection.action.shape), (2, 14))
                self.assertEqual(selection.text[0].frame_range, (1, 5))
                expected_state = torch.cat(
                    (
                        torch.from_numpy(group["left.obs_ee_pose"].values[[0, 3]]),
                        torch.from_numpy(group["right.obs_ee_pose"].values[[0, 3]]),
                    ),
                    dim=1,
                )
                expected_action = torch.cat(
                    (
                        torch.from_numpy(group["left.obs_ee_pose"].values[[2, 5]]),
                        torch.from_numpy(group["right.obs_ee_pose"].values[[2, 5]]),
                    ),
                    dim=1,
                )
                self.assertTrue(torch.equal(selection.state, expected_state))
                self.assertTrue(torch.equal(selection.action, expected_action))
            finally:
                repository.close()

    def test_scan_can_report_an_incomplete_store(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_zarr_root(root)
            (root / "incomplete.zarr").mkdir()

            with self.assertRaisesRegex(
                FileNotFoundError, "lacks root zarr.json"
            ):
                scan_egoverse_episode_inventory(
                    root,
                    "aria_bimanual",
                )

            result = scan_egoverse_episode_inventory(
                root,
                "aria_bimanual",
                fail_on_store_error=False,
            )
            self.assertEqual(result.candidate_count, 2)
            self.assertEqual(len(result.episodes), 1)
            self.assertEqual(len(result.issues), 1)
            self.assertEqual(
                result.issues[0].error_type,
                "FileNotFoundError",
            )

    def test_resumable_builder_restarts_from_fsynced_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = _write_zarr_root(root, episode_name="episode_001")
            second = _write_zarr_root(root, episode_name="episode_002")
            groups = {
                str(first): _aria_group(),
                str(second): _aria_group(),
            }
            output = root / "resume.episode_catalog.jsonl"

            class InterruptOnce:
                def __init__(self) -> None:
                    self.interrupted = False

                def __call__(self, path: str) -> _FakeGroup:
                    if str(path) == str(second) and not self.interrupted:
                        self.interrupted = True
                        raise KeyboardInterrupt()
                    return groups[str(path)]

            with self.assertRaises(KeyboardInterrupt):
                build_egoverse_episode_catalog(
                    data_root=root,
                    profile="aria_bimanual",
                    output_path=output,
                    catalog_id="egoverse_resume_fixture",
                    data_version="fixture",
                    group_opener=InterruptOnce(),
                    resume=True,
                    checkpoint_interval=1,
                )

            partial = root / f".{output.name}.build.partial"
            state = root / f".{output.name}.build.state.json"
            self.assertTrue(partial.is_file())
            self.assertTrue(state.is_file())
            checkpoint = json.loads(state.read_text(encoding="utf-8"))
            self.assertEqual(checkpoint["processed"], 1)
            self.assertEqual(checkpoint["written"], 1)

            metadata = build_egoverse_episode_catalog(
                data_root=root,
                profile="aria_bimanual",
                output_path=output,
                catalog_id="egoverse_resume_fixture",
                data_version="fixture",
                group_opener=lambda path: groups[str(path)],
                resume=True,
                checkpoint_interval=1,
            )
            rows = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(rows), 2)
            self.assertEqual(metadata["episodes"], 2)
            self.assertEqual(
                metadata["meta"]["build"]["quality"]["episodes"],
                2,
            )
            self.assertFalse(partial.exists())
            self.assertFalse(state.exists())

    def test_builder_rejects_array_shorter_than_total_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = _write_zarr_root(root, total_frames=6)
            group = _aria_group(total_frames=6, physical_frames=5)
            with self.assertRaisesRegex(
                ValueError, "shorter than total_frames"
            ):
                build_egoverse_episode_catalog(
                    data_root=root,
                    profile="aria_bimanual",
                    output_path=root / "bad.episode_catalog.jsonl",
                    catalog_id="bad",
                    data_version="bad",
                    group_opener=lambda path: {
                        str(store): group
                    }[str(path)],
                )

    def test_builder_reports_corrupt_numeric_key_and_range(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = _write_zarr_root(root)
            group = _aria_group()
            group["right.obs_ee_pose"] = _BrokenArray(
                np.zeros((8, 7), dtype=np.float32)
            )
            with self.assertRaisesRegex(
                ValueError,
                r"key='right\.obs_ee_pose'.*rows \[0,6\).*checksum",
            ):
                build_egoverse_episode_catalog(
                    data_root=root,
                    profile="aria_bimanual",
                    output_path=root / "corrupt.episode_catalog.jsonl",
                    catalog_id="corrupt",
                    data_version="fixture",
                    group_opener=lambda path: {
                        str(store): group
                    }[str(path)],
                )

    def test_builder_can_publish_explicit_quarantine_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bad_store = _write_zarr_root(
                root,
                episode_name="episode_bad",
            )
            good_store = _write_zarr_root(
                root,
                episode_name="episode_good",
            )
            bad_group = _aria_group()
            bad_group["left.obs_ee_pose"] = _BrokenArray(
                np.zeros((8, 7), dtype=np.float32)
            )
            groups = {
                str(bad_store): bad_group,
                str(good_store): _aria_group(),
            }
            output = root / "quarantine.episode_catalog.jsonl"
            metadata = build_egoverse_episode_catalog(
                data_root=root,
                profile="aria_bimanual",
                output_path=output,
                catalog_id="quarantine",
                data_version="fixture",
                group_opener=lambda path: groups[str(path)],
                fail_on_episode_error=False,
                resume=True,
                checkpoint_interval=1,
            )

            build = metadata["meta"]["build"]
            self.assertEqual(metadata["episodes"], 1)
            self.assertEqual(build["skipped_episodes"], 1)
            self.assertEqual(build["quarantine"]["episodes"], 1)
            self.assertFalse(build["fail_on_episode_error"])
            quarantine_path = output.with_suffix(".quarantine.jsonl")
            entries = [
                json.loads(line)
                for line in quarantine_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(len(entries), 1)
            self.assertEqual(
                entries[0]["episode_id"],
                "egoverse:aria_bimanual:episode_bad",
            )
            self.assertIn("left.obs_ee_pose", entries[0]["error"])


if __name__ == "__main__":
    unittest.main()
