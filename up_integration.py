from __future__ import annotations

from typing import Dict, Any

from up_model_builder import build_planning_problem
from up_planner_runner import plan_assignments


STAGES = [
    "adversarial_training",
    "perturbation_generation",
    "metric_computation",
    "rais_scoring",
    "pruning_decision",
    "recovery_finetuning",
    "model_evaluation",
    "deploy_updated_model",
]

CAPABILITIES = {
    "adversarial_training": ["ncps", "gpu", "gpu_2"],
    "perturbation_generation": ["gpu", "gpu_2", "ncps", "jetson"],
    "metric_computation": ["gpu", "gpu_2", "ncps", "jetson"],
    "rais_scoring": ["gpu", "gpu_2", "ncps", "jetson"],
    "pruning_decision": ["gpu", "gpu_2", "ncps", "jetson"],
    "recovery_finetuning": ["ncps", "gpu", "gpu_2"],
    "model_evaluation": ["gpu", "gpu_2", "jetson", "ncps"],
    "deploy_updated_model": ["pi5a", "pi5d", "jetson"],
}

THRESHOLDS = {
    "adversarial_training": {
        "min_free_ram_gb": 4.0,
        "max_cpu_load": 0.98,
        "max_gpu_util": 1.00,
        "max_latency_ms": 1500.0,
        "min_reliability": 0.60,
    },
    "perturbation_generation": {
        "min_free_ram_gb": 2.0,
        "max_cpu_load": 0.98,
        "max_gpu_util": 1.00,
        "max_latency_ms": 1500.0,
        "min_reliability": 0.60,
    },
    "metric_computation": {
        "min_free_ram_gb": 2.0,
        "max_cpu_load": 0.98,
        "max_gpu_util": 1.00,
        "max_latency_ms": 1500.0,
        "min_reliability": 0.60,
    },
    "rais_scoring": {
        "min_free_ram_gb": 1.0,
        "max_cpu_load": 0.98,
        "max_gpu_util": 1.00,
        "max_latency_ms": 1500.0,
        "min_reliability": 0.60,
    },
    "pruning_decision": {
        "min_free_ram_gb": 1.0,
        "max_cpu_load": 0.98,
        "max_gpu_util": 1.00,
        "max_latency_ms": 1500.0,
        "min_reliability": 0.60,
    },
    "recovery_finetuning": {
        "min_free_ram_gb": 8.0,
        "max_cpu_load": 0.98,
        "max_gpu_util": 1.00,
        "max_latency_ms": 1500.0,
        "min_reliability": 0.70,
    },
    "model_evaluation": {
        "min_free_ram_gb": 1.0,
        "max_cpu_load": 0.99,
        "max_gpu_util": 1.00,
        "max_latency_ms": 1500.0,
        "min_reliability": 0.50,
    },
    "deploy_updated_model": {
        "min_free_ram_gb": 0.5,
        "max_cpu_load": 0.99,
        "max_gpu_util": 1.00,
        "max_latency_ms": 1500.0,
        "min_reliability": 0.50,
    },
}


def get_up_assignments(
    raw_telemetry: Dict[str, Dict[str, Any]],
    completed_stages: list[str] | None = None,
) -> Dict[str, str]:
    normalized = raw_telemetry

    problem, _, _ = build_planning_problem(
        telemetry=normalized,
        stages=STAGES,
        capabilities=CAPABILITIES,
        thresholds=THRESHOLDS,
        completed_stages=completed_stages or [],
    )

    _, assignments = plan_assignments(problem)
    return assignments