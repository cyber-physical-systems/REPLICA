from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set
import json


# ============================================================
# TASK
# ============================================================

@dataclass(frozen=True)
class Task:
    task_id: str
    model: str
    stage: str

    # Workflow structure
    predecessors: tuple[str, ...] = ()

    # Capability requirements
    requires_gpu: bool = False
    min_cpu_cores: float = 0.0
    min_ram_mb: float = 0.0
    min_vram_mb: float = 0.0

    # Artifact / transfer properties
    input_size_mb: float = 0.0
    output_size_mb: float = 0.0

    # Optional constraints
    allowed_resource_types: tuple[str, ...] = ()
    target_edge: Optional[str] = None


# ============================================================
# RESOURCE
# ============================================================

@dataclass(frozen=True)
class Resource:
    resource_id: str
    resource_type: str

    cpu_cores: float
    ram_mb: float

    has_gpu: bool = False
    gpu_name: Optional[str] = None
    vram_mb: float = 0.0

    # Nominal network properties
    network_mbps: float = 0.0

    # Capability labels
    capabilities: tuple[str, ...] = ()


# ============================================================
# CURRENT SYSTEM STATE
# ============================================================

@dataclass
class ResourceState:
    resource_id: str

    available: bool = True

    cpu_available_cores: Optional[float] = None
    ram_available_mb: Optional[float] = None
    vram_available_mb: Optional[float] = None

    network_available: bool = True

    # Optional current queue/load information
    busy_until_sec: float = 0.0


@dataclass
class SystemState:
    time_sec: float
    resources: Dict[str, ResourceState]


# ============================================================
# EMPIRICAL EXECUTION PROFILE
# Experiment 1 feeds this object.
# ============================================================

@dataclass(frozen=True)
class ExecutionProfile:
    model: str
    stage: str
    resource_id: str

    runtime_sec: float

    cpu_peak_cores: float = 0.0
    ram_peak_mb: float = 0.0
    gpu_mean_pct: float = 0.0
    gpu_peak_pct: float = 0.0
    vram_peak_mb: float = 0.0

    disk_read_mb: float = 0.0
    disk_write_mb: float = 0.0

    artifact_size_mb: float = 0.0


# ============================================================
# WORKFLOW
# ============================================================

@dataclass
class Workflow:
    workflow_id: str
    tasks: Dict[str, Task]

    def ready_tasks(self, completed: Set[str]) -> List[Task]:
        ready = []

        for task in self.tasks.values():
            if task.task_id in completed:
                continue

            if all(pred in completed for pred in task.predecessors):
                ready.append(task)

        return ready


# ============================================================
# SCHEDULER OUTPUT
# ============================================================

@dataclass
class Assignment:
    task_id: str
    resource_id: str

    start_sec: float
    end_sec: float

    estimated_runtime_sec: float


@dataclass
class ScheduleResult:
    scheduler: str
    success: bool

    assignments: List[Assignment] = field(default_factory=list)

    scheduling_overhead_sec: float = 0.0
    makespan_sec: Optional[float] = None

    infeasible_tasks: List[str] = field(default_factory=list)

    notes: str = ""

    def to_dict(self):
        d = asdict(self)
        return d

    def save_json(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# ============================================================
# FEASIBILITY
# Same function must be used by ALL schedulers.
# ============================================================

def task_resource_feasible(
    task: Task,
    resource: Resource,
    state: ResourceState,
) -> bool:

    if not state.available:
        return False

    if task.requires_gpu and not resource.has_gpu:
        return False

    if task.min_cpu_cores > resource.cpu_cores:
        return False

    if task.min_ram_mb > resource.ram_mb:
        return False

    if task.min_vram_mb > resource.vram_mb:
        return False

    if (
        task.allowed_resource_types
        and resource.resource_type not in task.allowed_resource_types
    ):
        return False

    if state.cpu_available_cores is not None:
        if task.min_cpu_cores > state.cpu_available_cores:
            return False

    if state.ram_available_mb is not None:
        if task.min_ram_mb > state.ram_available_mb:
            return False

    if task.requires_gpu and state.vram_available_mb is not None:
        if task.min_vram_mb > state.vram_available_mb:
            return False

    return True
