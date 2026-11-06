"""Parameterized read-only measurements; no robot/model dependencies."""
from dataclasses import dataclass
from typing import Any, Callable, Mapping
from execution.feedback import OperationReport


@dataclass(frozen=True)
class MeasurementResult:
    report: OperationReport
    data: Mapping[str, Any]


@dataclass(frozen=True)
class MeasurementTool:
    name: str
    revision: str
    description: str
    parameters: Mapping[str, str]
    run: Callable[..., MeasurementResult]

    def interface(self):
        return dict(name=self.name, revision=self.revision, description=self.description,
                    parameters=dict(self.parameters), moves_robot=False)
