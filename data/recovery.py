"""Assign data roles to executed rollout segments (Data-RSI).

One segment = one executed command chunk (its executed prefix only).

Roles
  nominal       success episode, no recovery decision         -> verified action + consequence
  error_prefix  chunks up to the detected deviation, or every
                chunk of a failed episode                     -> consequence only
  recovery      chunks after a non-``continue`` decision in an
                episode the independent verifier marked success -> verified action + consequence
  unverified    uncertain / invalid outcome                   -> consequence only (measured data is real)

Action supervision additionally requires an exact executed-segment verification
with a versioned verifier and final measured evidence; legacy logs without it
remain consequence-only even when the whole episode succeeded.
Wrong actions never receive action supervision; predictions never become
targets (consequence = commanded actions + *measured* frames). Rollouts from
the formal test split are refused.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping

from thinker_wm.runtime.recovery_controller import CONTINUE
from execution.trajectory import read_trajectory

ROLE_NOMINAL, ROLE_ERROR, ROLE_RECOVERY, ROLE_UNVERIFIED = "nominal", "error_prefix", "recovery", "unverified"
ACTION_ROLES = (ROLE_NOMINAL, ROLE_RECOVERY)
ALLOWED_SPLITS = ("train", "development")


@dataclass(frozen=True)
class Segment:
    episode: str
    split: str
    task: str
    decision_step: int
    end_step: int                 # last executed row (inclusive)
    command_hash: str
    command_attachment: str
    decision: str
    role: str
    action_supervision: bool
    consequence_supervision: bool
    outcome: str
    max_visual_mse: float | None
    verification_status: str = "unknown"
    verifier_revision: str | None = None
    evidence_steps: tuple = ()

    def to_dict(self):
        return asdict(self)


def segments_from_trajectory(path: str | Path, *, outcome: str, split: str, task: str) -> List[Segment]:
    if split not in ALLOWED_SPLITS:
        raise ValueError("only train/development rollouts may enter collection")
    recs = read_trajectory(path)
    header = recs[0]
    if header["role"] != "header":
        raise ValueError("trajectory lacks header")
    episode = header["episode_id"]
    cmds = [r for r in recs if r["role"] == "command"]
    decisions = {r["step"]: r["action"] for r in recs if r["role"] == "decision"}
    measured = [r for r in recs if r["role"] == "measured"]
    if any("executed_row" not in r for r in measured):
        raise ValueError("trajectory lacks measured.executed_row (pre-2026-10-16 log); re-collect it")
    # A row counts as executed only once its feedback observation was measured.
    executed = sorted(r["executed_row"] for r in measured if r["executed_row"] is not None)
    verifications = {}
    for record in recs:
        if record["role"] == "segment_verification":
            key = record.get("decision_step")
            if type(key) is not int or key < 0 or key in verifications:
                raise ValueError("invalid or duplicate segment verification")
            verifications[key] = record
    if any(step not in {c["decision_step"] for c in cmds} for step in verifications):
        raise ValueError("verification has no corresponding command")
    consumed_verifications = set()
    errors: Dict[int, float] = {}
    for r in recs:
        if r["role"] == "error" and r["features"].get("valid_visual"):
            errors[r["step"]] = r["features"]["visual_mse_max"]
    first_recovery = next((i for i, c in enumerate(cmds)
                           if decisions.get(c["decision_step"], CONTINUE) != CONTINUE), None)
    out: List[Segment] = []
    for i, c in enumerate(cmds):
        d = c["decision_step"]
        nxt = cmds[i + 1]["decision_step"] if i + 1 < len(cmds) else float("inf")
        rows = [r for r in executed if d <= r < nxt]
        if not rows:
            continue
        end = rows[-1]
        if outcome not in ("success", "failure"):
            role = ROLE_UNVERIFIED
        elif outcome == "failure":
            role = ROLE_ERROR
        elif first_recovery is None:
            role = ROLE_NOMINAL
        elif i < first_recovery:
            role = ROLE_ERROR
        else:
            role = ROLE_RECOVERY
        record = verifications.get(d)
        status, revision, evidence_steps = "unknown", None, ()
        if record is not None:
            consumed_verifications.add(d)
            status = record.get("verdict")
            revision = record.get("verifier_revision")
            evidence_steps = tuple(record.get("evidence_steps", ()))
            if (record.get("command_hash") != c["hash"] or record.get("end_step") != end
                    or record.get("episode_id") != episode
                    or rows != list(range(d, end + 1))
                    or status not in ("passed", "failed", "unknown")
                    or not isinstance(revision, str) or not revision.strip()):
                raise ValueError("verification does not bind the exact executed segment")
            if (any(type(step) is not int or step < d or step > end + 1
                    for step in evidence_steps)
                    or any(step not in {m["step"] for m in measured} for step in evidence_steps)
                    or (status != "unknown" and (end + 1 not in evidence_steps
                        or not any(m["step"] == end + 1 and m["executed_row"] == end
                                   for m in measured)))):
                raise ValueError("verification requires final measured segment evidence")
            if status == "failed":
                role = ROLE_ERROR
        errs = [v for s, v in errors.items() if d < s <= end + 1]
        out.append(Segment(episode, split, task, d, end, c["hash"], c.get("attachment", ""),
                           decisions.get(d, CONTINUE), role,
                           role in ACTION_ROLES and status == "passed", True, outcome,
                           max(errs) if errs else None, status, revision, evidence_steps))
    if consumed_verifications != set(verifications):
        raise ValueError("verification refers to an unexecuted segment")
    return out


def collect(rollouts: List[Mapping[str, str]]) -> List[Segment]:
    """rollouts: [{trajectory, outcome, split, task}] (outcome from the upstream verdict)."""
    segs: List[Segment] = []
    for r in rollouts:
        segs.extend(segments_from_trajectory(r["trajectory"], outcome=r["outcome"],
                                             split=r["split"], task=r["task"]))
    return segs
