"""Explicit action semantics and measured observations across simulators."""
from dataclasses import dataclass
import math
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class ActionContract:
    name: str
    channels: tuple[str, ...]
    units: tuple[str, ...]
    control_hz: float
    representation: str
    frame: str
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    max_delta: tuple[float, ...] = ()

    def __post_init__(self):
        for key in ('channels', 'units', 'lower', 'upper', 'max_delta'):
            object.__setattr__(self, key, tuple(getattr(self, key)))
        n = len(self.channels)
        if not self.name or not self.frame or self.representation not in ('absolute', 'delta'):
            raise ValueError('versioned action name, frame and absolute/delta representation required')
        if n == 0 or len(set(self.channels)) != n or any(len(x) != n for x in (self.units,self.lower,self.upper)):
            raise ValueError('explicit unique channels, units and per-channel bounds required')
        if not math.isfinite(self.control_hz) or self.control_hz <= 0:
            raise ValueError('positive control frequency required')
        if any(not math.isfinite(v) for v in (*self.lower,*self.upper,*self.max_delta)):
            raise ValueError('finite bounds required')
        if any(a > b for a,b in zip(self.lower,self.upper)):
            raise ValueError('inverted action bounds')
        if self.max_delta and (len(self.max_delta) != n or any(v <= 0 for v in self.max_delta)):
            raise ValueError('positive per-channel delta bounds required')

    @property
    def dimension(self):
        return len(self.channels)

    def validate(self, row, previous=None):
        if len(row) != self.dimension or not all(math.isfinite(v) for v in row):
            raise ValueError('action dimension or finiteness differs from contract')
        if any(not lo <= v <= hi for v,lo,hi in zip(row,self.lower,self.upper)):
            raise ValueError('action outside contract bounds')
        if self.max_delta:
            if previous is None or len(previous) != self.dimension:
                raise ValueError('measured reference required for delta bounds')
            if any(abs(v-p) > d+1e-9 for v,p,d in zip(row,previous,self.max_delta)):
                raise ValueError('action exceeds delta bounds')


@dataclass(frozen=True)
class Observation:
    instruction: str
    state: tuple[float, ...]
    images: Mapping[str, Any]  # uint8 HWC RGB; no simulator ground truth
    step: int
    time_s: float


@dataclass(frozen=True)
class Outcome:
    success: bool | None
    terminated: bool
    score: float | None = None


class Environment(Protocol):
    contract: ActionContract
    def identity(self) -> Mapping[str, Any]: ...
    def reset(self, case: Mapping[str, Any]) -> None: ...
    def observe(self) -> Observation: ...
    def execute(self, row: tuple[float, ...]) -> None: ...
    def is_terminal(self) -> bool: ...
    def outcome(self) -> Outcome: ...
    def close(self) -> None: ...
