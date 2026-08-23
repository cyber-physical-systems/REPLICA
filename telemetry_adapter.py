from __future__ import annotations

from typing import Any, Dict, List, Tuple

from up_model_builder import telemetry_stage_cost



def get_exact_telemetry_optimal_assignments(
    telemetry: Dict[str, Dict[str, Any]],
    stages: List[str],
    capabilities: Dict[str, List[str]],
    thresholds: Dict[str, Dict[str, float]] | None = None,
    completed_stages: List[str] | None = None,
) -> Tuple[Dict[str, str], Dict[str, int]]:
    """
    Exact optimizer for the current model:
    each stage is assigned independently, and total cost is additive.
    Therefore, the global optimum is obtained by choosing the minimum-cost
    feasible host for each remaining stage independently.
    """
    thresholds = thresholds or {}
    completed_stages = set(completed_stages or [])

    assignments: Dict[str, str] = {}
    chosen_costs: Dict[str, int] = {}

    for sname in stages:
        if sname in completed_stages:
            continue

        allowed_hosts = capabilities.get(sname, [])
        stage_thresh = thresholds.get(sname, {})

        min_free_ram_gb = float(stage_thresh.get("min_free_ram_gb", 0.0))
        max_cpu_load = float(stage_thresh.get("max_cpu_load", 1.0))
        max_gpu_util = float(stage_thresh.get("max_gpu_util", 1.0))
        max_latency_ms = float(stage_thresh.get("max_latency_ms", 100000.0))
        min_reliability = float(stage_thresh.get("min_reliability", 0.0))

        best_host = None
        best_cost = None

        for hname in allowed_hosts:
            if hname not in telemetry:
                continue

            h = telemetry[hname]

            feasible = (
                bool(h.get("available", False))
                and float(h.get("free_ram_gb", 0.0)) >= min_free_ram_gb
                and float(h.get("cpu_load", 1.0)) <= max_cpu_load
                and float(h.get("gpu_util", 1.0)) <= max_gpu_util
                and float(h.get("latency_ms", 100000.0)) <= max_latency_ms
                and float(h.get("reliability", 0.0)) >= min_reliability
            )

            if not feasible:
                continue

            cost = int(telemetry_stage_cost(sname, h))

            if best_cost is None or cost < best_cost:
                best_host = hname
                best_cost = cost

        if best_host is None or best_cost is None:
            raise RuntimeError(
                f"No feasible host found for stage '{sname}' under current telemetry/thresholds."
            )

        assignments[sname] = best_host
        chosen_costs[sname] = best_cost

    return assignments, chosen_costs