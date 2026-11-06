"""Record-only task-progress signals. Never makes a control decision."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


class ProgressChecker:
    def __init__(self, stall_window_steps: int = 50, stall_eps: float = 1e-3):
        self.stall_window_steps, self.stall_eps = int(stall_window_steps), float(stall_eps)
        self.reset()

    def reset(self) -> None:
        self._states: List[np.ndarray] = []
        self._steps: List[int] = []
        self.records: List[Dict] = []

    def update(self, step: int, state: np.ndarray, info: Optional[Dict] = None) -> Dict:
        s = np.asarray(state, dtype=np.float64).reshape(-1)
        self._states.append(s)
        self._steps.append(int(step))
        rec: Dict = dict(step=int(step))
        if len(self._states) > 1:
            rec["state_delta"] = float(np.linalg.norm(s - self._states[-2]))
        if len(self._states) > self.stall_window_steps:
            old = self._states[-1 - self.stall_window_steps]
            rec["window_motion"] = float(np.linalg.norm(s - old))
            rec["stalled"] = rec["window_motion"] < self.stall_eps
        if info:
            rec["env"] = {k: info[k] for k in sorted(info) if isinstance(info[k], (bool, int, float, str))}
        self.records.append(rec)
        return rec
