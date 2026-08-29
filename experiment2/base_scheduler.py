from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Tuple

from experiment2.common import (
    Workflow,
    Resource,
    SystemState,
    ExecutionProfile,
    ScheduleResult,
)


ProfileKey = Tuple[str, str, str]
# (model, stage, resource_id)


class BaseScheduler(ABC):

    name = "base"

    def __init__(
        self,
        profiles: Dict[ProfileKey, ExecutionProfile],
    ):
        self.profiles = profiles

    def runtime(
        self,
        model: str,
        stage: str,
        resource_id: str,
    ) -> float:

        key = (model, stage, resource_id)

        if key not in self.profiles:
            raise KeyError(
                f"No Experiment-1 execution profile for {key}"
            )

        return self.profiles[key].runtime_sec

    @abstractmethod
    def schedule(
        self,
        workflow: Workflow,
        resources: Dict[str, Resource],
        state: SystemState,
    ) -> ScheduleResult:
        pass
