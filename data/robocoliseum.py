"""RoboColiseum offline evidence -> RSI training manifests and ICL candidates.

No native task success, skill identity, scene seed or recovery event is invented.
The recorded expert actions may be demonstrations; they are not new robot trials.
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from thinker_wm.data_prepare.robocoliseum_core import CONTRACT, digest, read_window


@dataclass(frozen=True)
class OfflineSegment:
    episode: str
    split: str
    task: str
    decision_step: int
    end_step: int
    command_hash: str
    command_attachment: str
    decision: str
    role: str
    action_supervision: bool
    consequence_supervision: bool
    outcome: str
    max_visual_mse: None
    instruction: str
    source_record: dict
    action_offset: int
    reviewed_demonstration: bool

    def to_dict(self):
        return asdict(self)


def training_segments(records, *, action_offset, allow_reviewed_demonstrations=False):
    """Train/development only. BC is an explicit reviewed-demo opt-in.

    Call on the desired training subset; passing validation/test/holdout is an
    error, not a silently dropped record. Native outcome stays unverified.
    """
    if type(action_offset) is not int or action_offset not in (0, 1):
        raise ValueError("action_offset must be explicitly 0 or 1")
    result = []
    for record in records:
        if record["split"] not in {"train", "development"}:
            raise ValueError(f"cannot train on {record['split']} episode {record['episode_id']}")
        if digest({k: v for k, v in record.items() if k != "record_sha256"}) != record["record_sha256"]:
            raise ValueError("source record digest mismatch")
        if record["action_contract"] != CONTRACT or not all(
                record["validation"].get(k) for k in ("rows", "video_headers")):
            raise ValueError("training export requires validated rows and video headers")
        for segment in record["instruction_segments"]:
            start = segment["start"]
            stop = min(segment["stop"] - action_offset, record["length"] - 1)
            if start >= stop:
                continue
            reviewed = segment["reviewed_demonstration"]
            bc = allow_reviewed_demonstrations and reviewed
            result.append(OfflineSegment(
                record["episode_id"], record["split"], record["task"], start, stop - 1,
                digest(dict(parquet=record["parquet_sha256"], start=start + action_offset,
                            stop=stop + action_offset, contract=CONTRACT)),
                record["parquet"], "offline_recording", "demonstration" if bc else "unverified",
                bc, True, "unverified", None, segment["instruction"], record, action_offset, reviewed))
    return result


def training_manifest(records, *, action_offset, parent_checkpoint, data_version,
                      allow_reviewed_demonstrations=False):
    from data.training_manifest import build_manifest

    segments = training_segments(records, action_offset=action_offset,
                                 allow_reviewed_demonstrations=allow_reviewed_demonstrations)
    if not segments or not parent_checkpoint.strip() or not data_version.strip():
        raise ValueError("nonempty training data, parent checkpoint identity and data version required")
    return build_manifest([], segments, strategy="balanced", rollout_budget=1.0,
                          parent_checkpoint=parent_checkpoint, data_version=data_version)


def load_training_row(row):
    """Materialize aligned numeric arrays for an existing RSI manifest row.

    RGB remains lazy: use read_rgb(source_record, view, observation_indices).
    The trainer/provider must honor row.mode/weight when constructing batches.
    """
    record = row["source_record"]
    if row["episode"] != record["episode_id"] or row["split"] != record["split"]:
        raise ValueError("manifest/source identity mismatch")
    if row["mode"] not in {"joint_policy", "action_conditioned_world"}:
        raise ValueError("unsupported training mode")
    if row["mode"] == "joint_policy" and not row["action_supervision"]:
        raise ValueError("action labels forbidden for consequence-only data")
    candidates = training_segments([record], action_offset=row["action_offset"],
                                   allow_reviewed_demonstrations=row["action_supervision"])
    if not any(s.command_hash == row["command_hash"] and s.instruction == row["instruction"]
               and s.decision_step == row["decision_step"] and s.end_step == row["end_step"]
               and s.action_supervision == row["action_supervision"] and s.role == row["role"]
               for s in candidates):
        raise ValueError("manifest row is not a source instruction segment")
    return read_window(record, row["decision_step"], row["end_step"] + 1,
                       action_offset=row["action_offset"])


def experience_candidate(records, output, *, package_id):
    """Seal a single-task evidence package for existing ExperienceCatalog/Runner.

    This contains cited annotations, not generated success claims or executable
    programs. Publishing/selecting it remains the responsibility of Core.
    """
    from experience.packages import build_package

    if not records or len({r["task"] for r in records}) != 1:
        raise ValueError("experience candidate must contain exactly one task")
    if any(r["split"] not in {"train", "development"} for r in records):
        raise ValueError("held-out episodes cannot become task experience")
    if any(digest({k: v for k, v in r.items() if k != "record_sha256"}) != r["record_sha256"]
           for r in records):
        raise ValueError("source record digest mismatch")
    with tempfile.TemporaryDirectory(prefix="robocoliseum-evidence-") as tmp:
        root = Path(tmp)
        (root / "episodes").mkdir()
        lines = ["# RoboColiseum recorded task evidence", "",
                 "Source review approval is not a native task success verdict.",
                 "These are historical demonstrations, not current scene instructions.",
                 "No WAM/direct/VLA identity or successful recovery is inferred.",
                 "Read the cited episode JSON for instruction spans and video references.", ""]
        for r in records:
            name = f"episodes/{r['episode_index']:06d}.json"
            (root / name).write_text(json.dumps(r, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            texts = " | ".join(s["instruction"] for s in r["instruction_segments"])
            lines.append(f"- {name}: {texts}")
        (root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return build_package(root, output, package_id, metadata=dict(
            source="robocoliseum", task=records[0]["task"], status="candidate",
            episodes=len(records), source_records_sha256=digest(records),
            native_outcome="unverified", entry_points=["README.md"]))
