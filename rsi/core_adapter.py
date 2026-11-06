"""ThinkerWAM / RoboDojo-sim adapters for the upstream PhysicalRSI core.

Implements the upstream protocols (commit 3c8cf7f) without re-implementing
the core:

  * ``ThinkerWAMSuite``  -> ``self_harness.evaluation.ExperimentSuite``
  * ``SimEnvironment``   -> ``experiments.EmbodiedEnvironment`` (simulation, no leased resources)
  * ``WAMSkill``         -> ``experiments.System1Policy`` (skills.wam.core)
  * ``TaskVerifier``     -> ``experiments.Verifier``

``ExperimentEvaluator`` / ``ImprovementCampaign`` are used unchanged. The core
types are imported lazily (``load_core``) so this module imports without the
upstream checkout; tests may pass a contract double with the same names.

Outcome separation: reset failure -> ``ready: False`` (upstream records
``invalid``/``reset_not_ready``); no success signal -> ``uncertain``; a policy
stop or termination without success -> ``failure``.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from skills.wam.core import ImageStore
ACTION_SCHEMA = "thinkerwam.command/v1"

SPLITS_FOR_RSI = ("development", "validation")


def load_core():
    from PhysicalRSI_core import contracts, embodiment, timing
    return SimpleNamespace(Contract=contracts.Contract, Embodiment=embodiment.Embodiment,
                           System1=embodiment.System1, Dependency=embodiment.Dependency,
                           ControlTiming=timing.ControlTiming, Revision=embodiment.Revision)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


@dataclass(frozen=True)
class InterfaceSpec:
    """One declaration shared by Embodiment and System1 (ExecutionBinding equality)."""
    robot: str
    camera_transform: str
    action_semantics: str
    action_unit: str
    control_hz: float
    execute_steps: int
    max_observation_age_s: float = 0.2
    execution_slack_s: float = 0.25

    def contracts(self, core):
        obs = core.Contract("thinkerwam.observation/v1", unit="rgb_uint8;state_raw;depth_mm",
                            frame=self.camera_transform, embodiment=self.robot)
        act = core.Contract(f"{ACTION_SCHEMA}:{self.action_semantics}", unit=self.action_unit,
                            frame="", embodiment=self.robot)
        timing = core.ControlTiming(1.0 / self.control_hz, self.execute_steps,
                                    self.max_observation_age_s, self.execution_slack_s)
        return obs, act, timing


class SimEnvironment:
    """``sim``: ``reset(case) -> live | None``; ``step(np action) -> (live, done, info)``;
    ``identity() -> dict``. ``info["success"]`` (bool) is the simulator's own judgement."""

    def __init__(self, *, sim, adapter, store: ImageStore, interface: InterfaceSpec,
                 embodiment_name: str, embodiment_revision: str, core=None):
        self.sim, self.adapter, self.store, self.iface = sim, adapter, store, interface
        self.name, self.revision = embodiment_name, embodiment_revision
        self.core = core or load_core()
        self._step = 0
        self.resets = 0
        self.on_observation: Optional[Callable[[Dict[str, Any]], None]] = None

    def identity(self) -> Dict[str, Any]:
        return dict(kind="thinkerwam.sim-env/v1", sim=self.sim.identity(),
                    camera_transform=self.adapter.camera_transform, embodiment=self.revision)

    def describe(self):
        obs, act, timing = self.iface.contracts(self.core)
        return self.core.Embodiment(self.name, self.revision, "simulation", obs, act, (), timing)

    def _obs(self, live, info=None):
        o, depth = self.adapter.observation(live, self._step)
        task = {k: info[k] for k in ("success",) if info and k in info}
        from skills.wam.core import encode_observation
        return encode_observation(self.store, o, depth, self.adapter.camera_transform,
                                  dict(task=task) if task else None)

    def reset(self, case: dict, context) -> Dict[str, Any]:
        context.check()
        self.resets += 1
        self._step = 0
        live = self.sim.reset(dict(case))
        if live is None:
            return dict(ready=False, observation=None)
        return dict(ready=True, observation=self._obs(live), terminated=False)

    def step(self, action, context) -> Dict[str, Any]:
        context.check()
        if not isinstance(action, Mapping) or action.get("schema") != ACTION_SCHEMA:
            raise ValueError("unexpected action payload")
        if action.get("stop"):
            return dict(observation=None, terminated=True, policy_stop=str(action.get("reason", "")))
        cmd = np.asarray(action["command"], dtype=np.float32)
        live, done, info = self.sim.step(cmd)
        self._step += 1
        obs = self._obs(live, info)
        if self.on_observation is not None:
            self.on_observation(obs)
        return dict(observation=obs, terminated=bool(done))


class TaskVerifier:
    def __init__(self, task: str, revision: str = "success-signal-v1"):
        self.task, self.revision = task, revision

    def identity(self) -> Dict[str, Any]:
        return dict(kind="thinkerwam.task-verifier/v1", task=self.task, revision=self.revision)

    def verify(self, case: dict, trace: List[dict]) -> Dict[str, Any]:
        steps = max(0, len(trace) - 1)
        signals = [e["observation"]["task"]["success"] for e in trace
                   if isinstance(e.get("observation"), Mapping) and "success" in (e["observation"].get("task") or {})]
        m = dict(steps=float(steps))
        if any(signals):
            return dict(outcome="success", reason="simulator success signal", measurements=dict(m, success=1.0))
        if trace and trace[-1].get("policy_stop") is not None:
            return dict(outcome="failure", reason="policy stopped: " + trace[-1]["policy_stop"],
                        measurements=dict(m, success=0.0))
        if signals:
            return dict(outcome="failure", reason="no success before termination/budget",
                        measurements=dict(m, success=0.0))
        return dict(outcome="uncertain", reason="simulator reported no success signal", measurements=m)


@dataclass(frozen=True)
class TaskSpec:
    instruction: str
    development_seeds: Sequence[int]
    validation_pool: Sequence[int]
    validation_episodes: int
    test_seeds: Sequence[int] = ()       # reserved: never sampled by RSI


@dataclass
class ThinkerWAMSuite:
    name: str
    tasks: Dict[str, TaskSpec]
    env_factory: Callable[[str], SimEnvironment]
    skill_factory: Callable[[dict, str], Any]        # (candidate, task) -> WAMSkill
    frozen_components: Sequence[str] = ("weights", "configuration")
    frozen_reference: Optional[Dict[str, Dict[str, str]]] = None
    verifier_revision: str = "success-signal-v1"
    _skills: Dict[Any, Any] = field(default_factory=dict, repr=False)
    _active: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        for t, s in self.tasks.items():
            pools = [set(s.development_seeds), set(s.validation_pool), set(s.test_seeds)]
            if pools[0] & pools[1] or pools[0] & pools[2] or pools[1] & pools[2]:
                raise ValueError(f"task {t}: development/validation/test seeds overlap")
            if s.validation_episodes > len(s.validation_pool):
                raise ValueError(f"task {t}: validation pool smaller than episodes")

    def identity(self) -> Dict[str, Any]:
        return dict(kind="thinkerwam.suite/v1", name=self.name, verifier=self.verifier_revision,
                    frozen_components=list(self.frozen_components),
                    tasks={t: dict(instruction=s.instruction, development=list(s.development_seeds),
                                   validation_pool=list(s.validation_pool),
                                   validation_episodes=s.validation_episodes,
                                   test_sha256=_sha(repr(sorted(s.test_seeds))))
                           for t, s in sorted(self.tasks.items())})

    def admit(self, candidate: dict) -> Dict[str, Any]:
        """Model weights/config stay fixed during a Harness comparison."""
        if self.frozen_reference is not None:
            for kind in self.frozen_components:
                if candidate["components"].get(kind) != self.frozen_reference.get(kind):
                    raise ValueError(f"candidate changes frozen component {kind!r}")
        return dict(admitted=True, frozen_components=list(self.frozen_components))

    def _case(self, task, seed, split):
        return dict(task=task, seed=int(seed), split=split, instruction=self.tasks[task].instruction)

    def development_cases(self, parent: dict) -> Dict[str, List[dict]]:
        return {t: [self._case(t, s, "development") for s in spec.development_seeds]
                for t, spec in sorted(self.tasks.items())}

    def validation_cases(self, comparison: dict) -> Dict[str, List[dict]]:
        from rsi.version_manifest import digest
        key = digest(comparison)
        out = {}
        for t, spec in sorted(self.tasks.items()):
            pool = sorted(spec.validation_pool)
            order = sorted(pool, key=lambda s: _sha(f"{key}:{t}:{s}"))
            out[t] = [self._case(t, s, "validation") for s in order[:spec.validation_episodes]]
        return out

    def environment(self, task: str):
        env = self.env_factory(task)
        # The core hands the terminal observation only to the verifier; route
        # post-step observations to the task's policy so it gets the last feedback.
        env.on_observation = lambda obs: self._feedback(task, obs)
        return env

    def _feedback(self, task: str, observation) -> None:
        skill = self._active.get(task)
        if skill is not None and hasattr(skill, "on_environment_observation"):
            skill.on_environment_observation(observation)

    def policy(self, candidate: dict, task: str):
        from rsi.version_manifest import verify_harness
        sha = verify_harness(candidate)
        key = (sha, task)
        if key not in self._skills:
            skill = self.skill_factory(candidate, task)
            if skill.revision != sha:
                raise ValueError("System 1 revision must equal the candidate sha256")
            self._skills[key] = skill
        self._active[task] = self._skills[key]
        return self._skills[key]

    def verifier(self, task: str):
        return TaskVerifier(task, self.verifier_revision)
