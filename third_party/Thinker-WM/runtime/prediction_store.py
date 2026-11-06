"""Bind forecasts to the final command and track what was actually executed.

Rules:
  * a joint forecast is usable only if post-processing left the model actions
    unchanged (same hash) - otherwise it is invalidated and must be re-predicted
    with ``predict_outcome`` on the final command;
  * a forecast must exist before the first step of its command is dispatched;
  * an interrupted command is evaluated only on its executed prefix;
  * a forecast expires when the controller moves past its last predicted step.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from thinker_wm.models.inference_api import KIND_CONDITIONED, Forecast
from thinker_wm.runtime.action_postprocess import Command


@dataclass
class Binding:
    command: Command
    forecast: Optional[Forecast]
    dispatched: List[int] = field(default_factory=list)      # native steps sent
    observed: Dict[int, float] = field(default_factory=dict)  # step -> observation time
    status: str = "pending"            # pending | executing | completed | interrupted
    invalid_reason: Optional[str] = None

    @property
    def executed_prefix(self) -> int:
        return len(self.dispatched)


class PredictionStore:
    def __init__(self):
        self._by_decision: Dict[int, Binding] = {}
        self.active: Optional[int] = None

    def bind(self, command: Command, forecast: Optional[Forecast]) -> Binding:
        if not command.finalized:
            raise ValueError("bind only post-processed commands")
        reason = None
        if forecast is not None:
            if forecast.decision_step != command.decision_step:
                raise ValueError("forecast and command come from different decision steps")
            expected = command.model_hash if forecast.kind == KIND_CONDITIONED else command.source_hash
            if forecast.action_hash != expected:
                raise ValueError("forecast was produced for different actions")
            if forecast.kind != KIND_CONDITIONED and command.model_hash != command.source_hash:
                reason = "command rewritten by post-processing; re-predict with predict_outcome"
                forecast = None
        if self.active is not None and self._by_decision[self.active].status == "executing":
            self.interrupt("superseded by a new command")
        b = Binding(command, forecast, invalid_reason=reason)
        self._by_decision[command.decision_step] = b
        self.active = command.decision_step
        return b

    def attach_forecast(self, decision_step: int, forecast: Forecast) -> None:
        b = self._by_decision[decision_step]
        if b.dispatched:
            raise ValueError("forecast must be produced before execution starts")
        if forecast.kind != KIND_CONDITIONED or forecast.action_hash != b.command.model_hash:
            raise ValueError("only a forecast conditioned on the final command can be attached")
        b.forecast, b.invalid_reason = forecast, None

    def dispatch(self, step: int, k: int) -> None:
        b = self._by_decision[self.active]
        if k != b.executed_prefix:
            raise ValueError(f"command row {k} dispatched out of order (prefix {b.executed_prefix})")
        if b.command.decision_step + k != step:
            raise ValueError("dispatch step does not match the command time base")
        b.dispatched.append(int(step))
        b.status = "executing"

    def observe(self, step: int, time_s: float) -> None:
        if self.active is not None:
            self._by_decision[self.active].observed[int(step)] = float(time_s)

    def complete(self) -> None:
        if self.active is not None:
            self._by_decision[self.active].status = "completed"

    def interrupt(self, reason: str) -> None:
        if self.active is not None:
            b = self._by_decision[self.active]
            b.status, b.invalid_reason = "interrupted", reason

    def usable(self, decision_step: int, step: int) -> Optional[Forecast]:
        """Forecast valid for comparing against the observation at ``step``."""
        b = self._by_decision.get(decision_step)
        if b is None or b.forecast is None:
            return None
        if step not in b.observed:
            return None
        if not b.dispatched or step > b.dispatched[-1] + 1:
            return None
        if step > b.forecast.frame_steps[-1]:
            return None
        return b.forecast

    def binding(self, decision_step: int) -> Binding:
        return self._by_decision[decision_step]

    def expire_before(self, step: int) -> int:
        old = [d for d, b in self._by_decision.items()
               if d != self.active and (b.forecast is None or b.forecast.frame_steps[-1] < step)]
        for d in old:
            del self._by_decision[d]
        return len(old)

    def reset(self) -> None:
        self._by_decision.clear()
        self.active = None
