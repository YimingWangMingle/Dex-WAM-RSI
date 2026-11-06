"""Contract double of the PhysicalRSI core checks our adapters must satisfy.

Mirrors commit 3c8cf7f (contracts.py / embodiment.py / timing.py /
experiments.ExperimentRuntime._execute) only where the adapters are
constrained: binding equality, System-2 revision binding, JSON freezing,
reset/step/verdict shapes, termination classes and policy cleanup. It does not
write evidence, lease devices or reconcile; it is NOT a replacement for the
core. Tests use the real ``PhysicalRSI_core`` whenever it is importable.
"""
from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace


class Cancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class Contract:
    name: str
    unit: str = ""
    frame: str = ""
    embodiment: str = ""

    def __post_init__(self):
        if not self.name:
            raise ValueError("Contract name is required")


@dataclass(frozen=True)
class ControlTiming:
    period_seconds: float
    max_chunk_steps: int
    max_observation_age_seconds: float
    execution_slack_seconds: float = 0.25

    def __post_init__(self):
        if not (self.period_seconds > 0 and self.max_chunk_steps >= 1 and self.max_observation_age_seconds > 0):
            raise ValueError("invalid control timing")


@dataclass(frozen=True)
class Dependency:
    name: str
    revision: str
    kind: str

    def __post_init__(self):
        if self.kind not in {"model", "skill", "tool", "memory"}:
            raise ValueError("dependency kind")
        if self.kind == "memory" and not re.fullmatch(r"[0-9a-f]{64}", self.revision):
            raise ValueError("memory revisions must be sha256")


@dataclass(frozen=True)
class Revision:
    name: str
    revision: str


@dataclass(frozen=True)
class Embodiment:
    name: str
    revision: str
    mode: str
    observation: Contract
    action: Contract
    resources: tuple = ()
    timing: ControlTiming | None = None

    def __post_init__(self):
        if self.mode not in {"software", "simulation", "physical"}:
            raise ValueError("mode")
        if self.mode == "physical" and not self.resources:
            raise ValueError("physical embodiments declare resources")


@dataclass(frozen=True)
class System1:
    name: str
    revision: str
    observation: Contract
    action: Contract
    dependencies: tuple = ()
    timing: ControlTiming | None = None


@dataclass(frozen=True)
class ExecutionBinding:
    embodiment: Embodiment
    system1: System1

    def __post_init__(self):
        e, s = self.embodiment, self.system1
        if e.observation != s.observation or e.action != s.action or e.timing != s.timing:
            raise ValueError("System 1 interface differs from the embodiment")


@dataclass(frozen=True)
class System2Trial:
    system2: Revision
    candidate_id: str
    candidate_sha256: str
    split: str
    comparison_sha256: str | None = None
    cohort_sha256: str | None = None

    def bind(self, system1: System1):
        if system1.revision != self.candidate_sha256:
            raise ValueError("Executing System 1 differs from the System 2 candidate")


class Context:
    def __init__(self, episode, deadline=math.inf):
        self.episode, self.deadline, self.cancelled = episode, deadline, False

    def check(self):
        if self.cancelled:
            raise Cancelled(self.episode)


def _json(v):
    return json.loads(json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False))


def _binding(env, policy):
    decl = [getattr(a, "describe", None) for a in (env, policy)]
    if all(d is None for d in decl):
        return None
    if not all(callable(d) for d in decl):
        raise ValueError("Both environment and System 1 must declare their interfaces")
    b = ExecutionBinding(*(d() for d in decl))
    if not callable(getattr(policy, "begin_episode", None)):
        raise ValueError("Declared System 1 requires begin_episode")
    if b.embodiment.resources and not callable(getattr(env, "quiesce", None)):
        raise ValueError("Leased environments require quiesce")
    return b


def run(*, task, case, environment, policy, verifier, max_steps, system2=None):
    binding = _binding(environment, policy)
    if system2 is not None:
        system2.bind(binding.system1)
    ids = _json(dict(e=environment.identity(), p=policy.identity(), v=verifier.identity()))
    spec_binding = _json(repr(binding))
    ctx = Context("run")

    def stable():
        if _json(dict(e=environment.identity(), p=policy.identity(), v=verifier.identity())) != ids \
                or _json(repr(_binding(environment, policy))) != spec_binding:
            raise ValueError("An adapter changed during the experiment")

    trace, started = [], False
    initial = _json(environment.reset(deepcopy(case), ctx))
    if type(initial.get("ready")) is not bool:
        raise ValueError("Reset must declare readiness")
    if not initial["ready"]:
        return dict(outcome="invalid", termination="reset_not_ready", steps=0, trace=trace, policy=None)
    trace.append(dict(observation=initial["observation"]))
    if binding:
        started = True
        policy.begin_episode(task, deepcopy(case), deepcopy(initial["observation"]), ctx)
    termination = "environment" if initial.get("terminated", False) else "step_budget"
    for step in range(0 if termination == "environment" else max_steps):
        stable()
        action = _json(policy.act(deepcopy(trace[-1]["observation"]), ctx))
        result = _json(environment.step(deepcopy(action), ctx))
        if (type(result.get("terminated")) is not bool or "observation" not in result
                or {"step", "action"}.intersection(result)):
            raise ValueError("Environment must return observation and termination without action metadata")
        trace.append(dict(result, step=step, action=action))
        if result["terminated"]:
            termination = "environment"
            break
    report = _json(policy.end_episode(ctx)) if started else dict(stopped=True, started=False)
    if report.get("stopped") is not True:
        raise RuntimeError("Policy worker termination is unconfirmed")
    verdict = _json(verifier.verify(deepcopy(case), deepcopy(trace)))
    if verdict.get("outcome") not in {"success", "failure", "uncertain"}:
        raise ValueError("Verifier must return success, failure or uncertain")
    if not verdict.get("reason") or not isinstance(verdict.get("measurements"), dict):
        raise ValueError("Verifier must supply a reason and measured evidence")
    return dict(outcome=verdict["outcome"], verdict=verdict, termination=termination,
                steps=max(0, len(trace) - 1), trace=trace, policy=report)


CORE = SimpleNamespace(Contract=Contract, Embodiment=Embodiment, System1=System1, Dependency=Dependency,
                       ControlTiming=ControlTiming, Revision=Revision, System2Trial=System2Trial)
