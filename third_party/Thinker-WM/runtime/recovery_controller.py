"""Recovery decision from prediction-error features.

Thresholds are explicit; with none set the controller is record-only and
always answers ``continue`` (the default until thresholds are calibrated on
held-out rollouts).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional

from thinker_wm.runtime.prediction_error import PredictionErrorFeatures

CONTINUE, REOBSERVE, LOCAL_RECOVERY, SWITCH_SKILL, STOP = (
    "continue", "reobserve", "local_recovery", "switch_skill", "stop")
DECISIONS = (CONTINUE, REOBSERVE, LOCAL_RECOVERY, SWITCH_SKILL, STOP)


@dataclass(frozen=True)
class RecoveryThresholds:
    track_max: Optional[float] = None          # -> reobserve (replan from fresh observation)
    visual_mse: Optional[float] = None         # -> local_recovery
    forecast_age_s: Optional[float] = None     # stale forecast -> reobserve
    max_local_recoveries: Optional[int] = None  # exceeded -> switch_skill
    max_switches: Optional[int] = None         # exceeded -> stop

    @property
    def record_only(self) -> bool:
        return all(v is None for v in asdict(self).values())


@dataclass(frozen=True)
class Decision:
    action: str
    reason: str
    features: Dict[str, float]


class RecoveryController:
    def __init__(self, thresholds: RecoveryThresholds = RecoveryThresholds()):
        self.th = thresholds
        self.reset()

    def reset(self) -> None:
        self.local_recoveries = 0
        self.switches = 0

    def decide(self, f: PredictionErrorFeatures) -> Decision:
        v, th = f.values, self.th
        if th.record_only:
            return Decision(CONTINUE, "record_only", dict(v))
        if v.get("valid_visual") and th.visual_mse is not None and v["visual_mse_max"] > th.visual_mse:
            self.local_recoveries += 1
            if th.max_local_recoveries is not None and self.local_recoveries > th.max_local_recoveries:
                self.local_recoveries = 0
                self.switches += 1
                if th.max_switches is not None and self.switches > th.max_switches:
                    return Decision(STOP, "recoveries exhausted", dict(v))
                return Decision(SWITCH_SKILL, "repeated visual deviation", dict(v))
            return Decision(LOCAL_RECOVERY, "visual deviation", dict(v))
        if v.get("valid_track") and th.track_max is not None and v["track_max"] > th.track_max:
            return Decision(REOBSERVE, "action tracking error", dict(v))
        if v.get("valid_forecast") and th.forecast_age_s is not None and v["forecast_age_s"] > th.forecast_age_s:
            return Decision(REOBSERVE, "forecast expired", dict(v))
        return Decision(CONTINUE, "within thresholds", dict(v))
