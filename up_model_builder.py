from __future__ import annotations

from typing import Any, Dict, List, Tuple

from unified_planning.shortcuts import (
    Problem,
    UserType,
    Object,
    Fluent,
    BoolType,
    IntType,
    InstantaneousAction,
    GE,
    LE,
    Not,
    MinimizeActionCosts,
)


# --------------------------
# Integer scaling helpers
# --------------------------
def cpu_to_milli(x: float) -> int:
    return int(round(max(0.0, min(1.0, x)) * 1000))


def gpu_to_milli(x: float) -> int:
    return int(round(max(0.0, min(1.0, x)) * 1000))


def rel_to_milli(x: float) -> int:
    return int(round(max(0.0, min(1.0, x)) * 1000))


def ram_gb_to_mb(x: float) -> int:
    return int(round(max(0.0, x) * 1024))


def lat_to_int(x: float) -> int:
    return int(round(max(0.0, x)))


def threshold_cpu_to_milli(x: float) -> int:
    return cpu_to_milli(x)


def threshold_gpu_to_milli(x: float) -> int:
    return gpu_to_milli(x)


def threshold_rel_to_milli(x: float) -> int:
    return rel_to_milli(x)


def threshold_ram_gb_to_mb(x: float) -> int:
    return ram_gb_to_mb(x)


# --------------------------
# Cost helpers (integer only)
# --------------------------
def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def normalize_latency_milli(latency_ms: int, max_ms: int = 500) -> int:
    """
    Return normalized latency in [0, 1000].
    """
    if max_ms <= 0:
        return 0
    return clamp_int(int(round((latency_ms / max_ms) * 1000)), 0, 1000)


def normalize_ram_pressure_milli(free_ram_mb: int, target_ram_mb: int) -> int:
    """
    0 means RAM target satisfied.
    1000 means severe RAM shortage.
    """
    if target_ram_mb <= 0:
        return 0
    if free_ram_mb >= target_ram_mb:
        return 0
    shortage_ratio = (target_ram_mb - free_ram_mb) / target_ram_mb
    return clamp_int(int(round(shortage_ratio * 1000)), 0, 1000)


def estimate_stage_runtime_from_telemetry_int(
    stage_name: str,
    cpu_milli: int,
    gpu_milli: int,
    free_ram_mb: int,
    latency_ms: int,
) -> int:
    """
    Integer-only runtime estimate.
    Lower is better.
    """

    stage_base = {
        "adversarial_training": 120,
        "perturbation_generation": 60,
        "metric_computation": 45,
        "rais_scoring": 25,
        "pruning_decision": 20,
        "recovery_finetuning": 90,
        "model_evaluation": 35,
        "deploy_updated_model": 10,
    }
    base = stage_base.get(stage_name, 50)

    cpu_weight_milli = {
        "adversarial_training": 800,
        "perturbation_generation": 600,
        "metric_computation": 500,
        "rais_scoring": 400,
        "pruning_decision": 300,
        "recovery_finetuning": 700,
        "model_evaluation": 400,
        "deploy_updated_model": 200,
    }.get(stage_name, 500)

    gpu_weight_milli = {
        "adversarial_training": 1200,
        "perturbation_generation": 1000,
        "metric_computation": 800,
        "rais_scoring": 500,
        "pruning_decision": 300,
        "recovery_finetuning": 1000,
        "model_evaluation": 600,
        "deploy_updated_model": 0,
    }.get(stage_name, 500)

    latency_weight_milli = {
        "adversarial_training": 100,
        "perturbation_generation": 200,
        "metric_computation": 200,
        "rais_scoring": 200,
        "pruning_decision": 100,
        "recovery_finetuning": 100,
        "model_evaluation": 200,
        "deploy_updated_model": 400,
    }.get(stage_name, 200)

    ram_target_mb = {
        "adversarial_training": 16 * 1024,
        "perturbation_generation": 8 * 1024,
        "metric_computation": 6 * 1024,
        "rais_scoring": 4 * 1024,
        "pruning_decision": 4 * 1024,
        "recovery_finetuning": 12 * 1024,
        "model_evaluation": 4 * 1024,
        "deploy_updated_model": 2 * 1024,
    }.get(stage_name, 4 * 1024)

    ram_pressure_milli = normalize_ram_pressure_milli(free_ram_mb, ram_target_mb)
    latency_norm_milli = normalize_latency_milli(latency_ms)

    # multiplier_milli = 1000 + weighted normalized pressures
    multiplier_milli = (
        1000
        + (cpu_weight_milli * cpu_milli) // 1000
        + (gpu_weight_milli * gpu_milli) // 1000
        + (1500 * ram_pressure_milli) // 1000
        + (latency_weight_milli * latency_norm_milli) // 1000
    )

    return (base * multiplier_milli) // 1000


def telemetry_stage_cost(stage_name: str, host_data: Dict[str, Any]) -> int:
    """
    Final integer cost for assigning stage_name to this host.
    Lower is better.
    """
    cpu_milli = cpu_to_milli(float(host_data.get("cpu_load", 1.0)))
    gpu_milli = gpu_to_milli(float(host_data.get("gpu_util", 1.0)))
    free_ram_mb = ram_gb_to_mb(float(host_data.get("free_ram_gb", 0.0)))
    latency_ms = lat_to_int(float(host_data.get("latency_ms", 1000.0)))
    reliability_milli = rel_to_milli(float(host_data.get("reliability", 0.0)))

    runtime_est = estimate_stage_runtime_from_telemetry_int(
        stage_name=stage_name,
        cpu_milli=cpu_milli,
        gpu_milli=gpu_milli,
        free_ram_mb=free_ram_mb,
        latency_ms=latency_ms,
    )

    reliability_penalty = (100 * (1000 - reliability_milli)) // 1000
    latency_penalty = (10 * normalize_latency_milli(latency_ms)) // 1000
    cpu_penalty = (20 * cpu_milli) // 1000
    gpu_penalty = (15 * gpu_milli) // 1000

    return int(runtime_est + reliability_penalty + latency_penalty + cpu_penalty + gpu_penalty)


# --------------------------
# Problem builder
# --------------------------
def build_planning_problem(
    telemetry: Dict[str, Dict[str, Any]],
    stages: List[str],
    capabilities: Dict[str, List[str]],
    thresholds: Dict[str, Dict[str, float]] | None = None,
    completed_stages: List[str] | None = None,
) -> Tuple[Problem, Dict[str, Object], Dict[str, Object]]:
    thresholds = thresholds or {}
    completed_stages = set(completed_stages or [])

    problem = Problem("resilient_orchestration_numeric")

    Host = UserType("Host")
    Stage = UserType("Stage")

    host_objects = {h: Object(h, Host) for h in telemetry.keys()}
    stage_objects = {s: Object(s, Stage) for s in stages}

    problem.add_objects(list(host_objects.values()))
    problem.add_objects(list(stage_objects.values()))

    available = Fluent("available", BoolType(), h=Host)

    cpu_load = Fluent("cpu_load", IntType(0, 1000), h=Host)
    free_ram_mb = Fluent("free_ram_mb", IntType(0, 4_194_304), h=Host)
    gpu_util = Fluent("gpu_util", IntType(0, 1000), h=Host)
    latency_ms = Fluent("latency_ms", IntType(0, 100000), h=Host)
    reliability = Fluent("reliability", IntType(0, 1000), h=Host)

    done = Fluent("done", BoolType(), s=Stage)
    assigned_to = Fluent("assigned_to", BoolType(), s=Stage, h=Host)
    expected_runtime_sec = Fluent(
        "expected_runtime_sec",
        IntType(0, 1_000_000),
        s=Stage,
        h=Host,
    )

    problem.add_fluent(available, default_initial_value=False)
    problem.add_fluent(cpu_load, default_initial_value=1000)
    problem.add_fluent(free_ram_mb, default_initial_value=0)
    problem.add_fluent(gpu_util, default_initial_value=1000)
    problem.add_fluent(latency_ms, default_initial_value=100000)
    problem.add_fluent(reliability, default_initial_value=0)
    problem.add_fluent(done, default_initial_value=False)
    problem.add_fluent(assigned_to, default_initial_value=False)
    problem.add_fluent(expected_runtime_sec, default_initial_value=999999)

    # --------------------------
    # Initial host telemetry
    # --------------------------
    for hname, hdata in telemetry.items():
        h = host_objects[hname]

        problem.set_initial_value(
            available(h),
            bool(hdata.get("available", False)),
        )
        problem.set_initial_value(
            cpu_load(h),
            cpu_to_milli(float(hdata.get("cpu_load", 1.0))),
        )
        problem.set_initial_value(
            free_ram_mb(h),
            ram_gb_to_mb(float(hdata.get("free_ram_gb", 0.0))),
        )
        problem.set_initial_value(
            gpu_util(h),
            gpu_to_milli(float(hdata.get("gpu_util", 1.0))),
        )
        problem.set_initial_value(
            latency_ms(h),
            lat_to_int(float(hdata.get("latency_ms", 100000.0))),
        )
        problem.set_initial_value(
            reliability(h),
            rel_to_milli(float(hdata.get("reliability", 0.0))),
        )

    # --------------------------
    # Initial stage state
    # --------------------------
    for sname in stages:
        s = stage_objects[sname]

        if sname in completed_stages:
            problem.set_initial_value(done(s), True)
        else:
            problem.set_initial_value(done(s), False)

        for hname in telemetry.keys():
            h = host_objects[hname]
            problem.set_initial_value(assigned_to(s, h), False)

            est = telemetry.get(hname, {}).get("expected_runtime_sec", {})
            if isinstance(est, dict):
                val = int(round(float(est.get(sname, 999999.0))))
            else:
                val = 999999

            problem.set_initial_value(expected_runtime_sec(s, h), val)

    action_costs = {}

    # --------------------------
    # Assignment actions
    # --------------------------
    for sname in stages:
        if sname in completed_stages:
            continue

        s = stage_objects[sname]
        allowed_hosts = capabilities.get(sname, [])
        stage_thresh = thresholds.get(sname, {})

        min_free_ram_mb = threshold_ram_gb_to_mb(
            float(stage_thresh.get("min_free_ram_gb", 0.0))
        )
        max_cpu_load = threshold_cpu_to_milli(
            float(stage_thresh.get("max_cpu_load", 1.0))
        )
        max_gpu_util = threshold_gpu_to_milli(
            float(stage_thresh.get("max_gpu_util", 1.0))
        )
        max_latency = lat_to_int(
            float(stage_thresh.get("max_latency_ms", 100000.0))
        )
        min_reliability = threshold_rel_to_milli(
            float(stage_thresh.get("min_reliability", 0.0))
        )

        feasible_action_count = 0

        for hname in allowed_hosts:
            if hname not in host_objects:
                continue

            h = host_objects[hname]
            action_name = f"assign__{sname}__to__{hname}"
            act = InstantaneousAction(action_name)

            act.add_precondition(Not(done(s)))
            act.add_precondition(available(h))
            act.add_precondition(GE(free_ram_mb(h), min_free_ram_mb))
            act.add_precondition(LE(cpu_load(h), max_cpu_load))
            act.add_precondition(LE(gpu_util(h), max_gpu_util))
            act.add_precondition(LE(latency_ms(h), max_latency))
            act.add_precondition(GE(reliability(h), min_reliability))

            act.add_effect(done(s), True)
            act.add_effect(assigned_to(s, h), True)

            problem.add_action(act)
            feasible_action_count += 1

            host_data = telemetry[hname]
            cost = telemetry_stage_cost(
                stage_name=sname,
                host_data=host_data,
            )

            print(
                f"[up-cost] stage={sname:24s} host={hname:8s} cost={cost} "
                f"cpu={host_data.get('cpu_load', 1.0):.3f} "
                f"gpu={host_data.get('gpu_util', 1.0):.3f} "
                f"ram={host_data.get('free_ram_gb', 0.0):.2f} "
                f"lat={host_data.get('latency_ms', 1000.0):.1f} "
                f"rel={host_data.get('reliability', 0.0):.2f}"
            )

            action_costs[act] = cost

        if feasible_action_count == 0:
            print(
                f"[up-warn] no candidate actions generated for required stage: {sname}"
            )

    for sname in stages:
        if sname in completed_stages:
            continue

        problem.add_goal(done(stage_objects[sname]))

    if action_costs:
        problem.add_quality_metric(MinimizeActionCosts(action_costs))

    return problem, host_objects, stage_objects