"""Backend-independent adapter with an explicit commit/discard model lifecycle.

Adopted from WorldScape R²SI PR #162 ``harness/wam.py`` (commit 9b474a6):

  * the model sees only measured observations (``update_obs``);
  * a proposal is held as pending until execution feedback arrives;
  * a fully executed chunk is committed *before* the final observation is
    forwarded, so the model's causal memory contains only executed actions;
  * a partially executed or rejected chunk is discarded (with whether execution
    started), never committed; its unexecuted rows never enter history;
  * ``reset`` clears model context. TaskSkillRunner resets a backend whenever
    control is handed to another skill, so an inactive model does not accumulate
    stale observations and restarts from the current measured state.

The model protocol mirrors WSP's native explicit-commit interface. The concrete
ThinkerWAM binding (checkpoint loading, RoboDojo action contract, history
window) implements ``ExplicitCommitModel``; it is not part of this module.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

from skills.contracts import (
    ActionRows, BackendProposal, ExecutionFeedback, MeasuredObservation, SkillRequest,
)


class ExplicitCommitModel(Protocol):
    def reset(self) -> None: ...
    def update_obs(self, observation: MeasuredObservation) -> None: ...
    def get_action(self, request: SkillRequest) -> Tuple[Sequence[Sequence[float]], Optional[str]]: ...
    def commit_action(self, executed_rows: ActionRows) -> None: ...
    def discard_action(self, execution_started: bool) -> None: ...
    def close(self) -> None: ...


class CommittedChunkBackend:
    """ControlBackend for any generator exposing an explicit commit lifecycle."""

    def __init__(self, model: ExplicitCommitModel, *, requires_full_history: bool = False):
        self.model = model
        self.requires_full_history = requires_full_history
        self.episode_id: Optional[str] = None
        self.latest: Optional[MeasuredObservation] = None
        self.pending: Optional[Tuple[int, ActionRows]] = None   # (base step, proposed rows)
        self.log: List[Dict[str, Any]] = []

    def reset(self, episode_id: str) -> None:
        if self.pending is not None:
            self.model.discard_action(False)
            self.log.append(dict(event="discard", reason="reset", execution_started=False))
        self.pending = None
        self.model.reset()
        self.episode_id, self.latest = episode_id, None
        self.log.append(dict(event="reset", episode_id=episode_id))

    def _feed(self, observation: MeasuredObservation) -> None:
        if observation.source != "measured":
            raise ValueError("only measured observations can enter model history")
        if self.latest is not None and observation.control_step <= self.latest.control_step:
            if observation.observation_id == self.latest.observation_id:
                return
            raise ValueError("observation is not newer than the model's latest history")
        self.model.update_obs(observation)
        self.latest = observation
        self.log.append(dict(event="update_obs", observation_id=observation.observation_id,
                             control_step=observation.control_step))

    def propose(self, request: SkillRequest) -> BackendProposal:
        if self.episode_id is None or request.observation.episode_id != self.episode_id:
            raise RuntimeError("reset the backend for this episode first")
        if self.pending is not None:
            raise RuntimeError("WAM has an unfinished proposal")
        if self.latest is None and self.requires_full_history:
            # A history-conditioned model started mid-episode is rebuilt from the
            # measured history the runner supplies, never from predictions.
            for item in request.history:
                self._feed(item)
        self._feed(request.observation)
        actions, prediction_ref = self.model.get_action(request)
        rows = tuple(tuple(float(v) for v in row) for row in actions)
        if not rows or len(rows) > request.max_actions:
            self.model.discard_action(False)
            raise ValueError(f"WAM chunk of {len(rows)} rows violates max_actions={request.max_actions}")
        self.pending = (request.observation.control_step, rows)
        self.log.append(dict(event="propose", control_step=request.observation.control_step,
                             rows=len(rows), prediction_ref=prediction_ref))
        return BackendProposal(rows, prediction_ref)

    def observe(self, feedback: ExecutionFeedback) -> None:
        if self.pending is None:
            raise RuntimeError("no WAM proposal awaits feedback")
        _, proposed = self.pending
        executed = feedback.executed_actions
        # Rows the model did not propose (e.g. changed by the executor) are never committed.
        matches = executed == proposed[:len(executed)]
        if matches and len(executed) == len(proposed):
            # Commit before forwarding the final frame, as in R²SI.
            self.model.commit_action(executed)
            self.log.append(dict(event="commit", executed_steps=len(executed)))
        else:
            self.model.discard_action(bool(executed))
            self.log.append(dict(event="discard", executed_steps=len(executed),
                                 proposed_steps=len(proposed), execution_started=bool(executed),
                                 reason="partial" if matches else "executed_rows_differ"))
        self.pending = None
        self._feed(feedback.final_observation)

    def close(self) -> None:
        try:
            if self.pending is not None:
                self.model.discard_action(False)
        finally:
            self.pending = None
            self.model.close()
