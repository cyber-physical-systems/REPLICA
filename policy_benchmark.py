#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from up_model_builder import build_planning_problem, telemetry_stage_cost
from up_planner_runner import plan_assignments
from up_integration import STAGES, CAPABILITIES, THRESHOLDS


OUT = Path("policy_benchmark_results_real")
OUT.mkdir(exist_ok=True)

SEED = 7
random.seed(SEED)
np.random.seed(SEED)

HPC3_PATH = "data/HPC3/hpcdata_20170502.csv"
HPC4_PATH = "data/HPC4/multi_cloud_service_dataset.csv"
HPC5_PATH = "data/HPC5/cloud_task_scheduling_dataset.csv"

POLICIES = [
    "static",
    "round_robin",
    "greedy",
    "rule_based",
    "adaptive_planner",
]

DEVICES = ["a100", "rtx5090", "rtx4080s", "gpu_2", "jetson"]

BASE_TELEMETRY = {
    "a100": {
        "available": True,
        "cpu_load": 0.12,
        "gpu_util": 0.20,
        "free_ram_gb": 36.0,
        "latency_ms": 80.0,
        "reliability": 0.96,
    },
    "rtx5090": {
        "available": True,
        "cpu_load": 0.10,
        "gpu_util": 0.15,
        "free_ram_gb": 82.0,
        "latency_ms": 70.0,
        "reliability": 0.95,
    },
    "rtx4080s": {
        "available": True,
        "cpu_load": 0.14,
        "gpu_util": 0.18,
        "free_ram_gb": 28.0,
        "latency_ms": 75.0,
        "reliability": 0.94,
    },
    "gpu_2": {
        "available": True,
        "cpu_load": 0.22,
        "gpu_util": 0.12,
        "free_ram_gb": 24.0,
        "latency_ms": 25.0,
        "reliability": 0.93,
    },
    "jetson": {
        "available": True,
        "cpu_load": 0.35,
        "gpu_util": 0.10,
        "free_ram_gb": 4.5,
        "latency_ms": 20.0,
        "reliability": 0.90,
    },
}

STATIC_ASSIGNMENTS = {
    "adversarial_training": "a100",
    "perturbation_generation": "a100",
    "metric_computation": "rtx5090",
    "rais_scoring": "gpu_2",
    "pruning_decision": "gpu_2",
    "recovery_finetuning": "rtx5090",
    "model_evaluation": "rtx4080s",
    "deploy_updated_model": "jetson",
}

RULE_FALLBACKS = {
    "adversarial_training": ["a100", "rtx5090", "rtx4080s", "gpu_2"],
    "perturbation_generation": ["a100", "rtx5090", "rtx4080s", "gpu_2"],
    "metric_computation": ["rtx5090", "rtx4080s", "gpu_2", "a100", "jetson"],
    "rais_scoring": ["gpu_2", "rtx4080s", "rtx5090", "jetson"],
    "pruning_decision": ["gpu_2", "rtx4080s", "rtx5090", "jetson"],
    "recovery_finetuning": ["rtx5090", "a100", "rtx4080s", "gpu_2"],
    "model_evaluation": ["rtx4080s", "rtx5090", "gpu_2", "a100", "jetson"],
    "deploy_updated_model": ["jetson", "gpu_2", "rtx4080s"],
}

BENCHMARK_STAGES = [
    "adversarial_training",
    "perturbation_generation",
    "metric_computation",
    "rais_scoring",
    "pruning_decision",
    "recovery_finetuning",
    "model_evaluation",
    "deploy_updated_model",
]

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def allowed_devices(stage: str) -> List[str]:
    allowed = CAPABILITIES.get(stage, DEVICES)
    allowed = [d for d in allowed if d in DEVICES]
    return allowed if allowed else DEVICES

def threshold(stage: str, *names: str, default: float) -> float:
    t = THRESHOLDS.get(stage, {})
    for name in names:
        if name in t:
            return float(t[name])
    return default

def device_feasible(stage, device, telemetry):

    if device not in telemetry:
        return False

    t = telemetry[device]

    if not bool(t.get("available", False)):
        return False

    max_cpu = threshold(stage, "max_cpu_load", "max_cpu", default=0.90)
    max_gpu = threshold(stage, "max_gpu_util", "max_gpu", default=0.95)
    min_ram = threshold(stage, "min_free_ram_gb", "min_ram_gb", default=2.0)
    max_latency = threshold(stage, "max_latency_ms", default=1000.0)
    min_reliability = threshold(stage, "min_reliability", default=0.0)

    return (
        float(t.get("cpu_load", 1.0)) <= max_cpu
        and float(t.get("gpu_util", 1.0)) <= max_gpu
        and float(t.get("free_ram_gb", 0.0)) >= min_ram
        and float(t.get("latency_ms", 1e9)) <= max_latency
        and float(t.get("reliability", 0.0)) >= min_reliability
    )

def assignment_cost(stage: str, device: str, telemetry: Dict[str, Dict[str, Any]]) -> float:
    try:
        return float(telemetry_stage_cost(stage, telemetry[device]))
    except Exception:
        t = telemetry[device]
        return (
            1000.0
            + 700.0 * float(t.get("cpu_load", 1.0))
            + 600.0 * float(t.get("gpu_util", 1.0))
            - 10.0 * float(t.get("free_ram_gb", 0.0))
            + float(t.get("latency_ms", 1000.0))
            + 500.0 * (1.0 - float(t.get("reliability", 0.0)))
        )

def compute_depletion(stage: str, device: str) -> Tuple[float, float, float]:
    """Calculates realistic cluster consumption values based on target architecture hardware classes."""
    if stage in ["adversarial_training", "recovery_finetuning", "perturbation_generation"]:
        ram_drain = 6.0 if device in ["rtx5090", "a100"] else 3.5
        cpu_step = 0.12 if device in ["rtx5090", "a100"] else 0.40
        gpu_step = 0.25 if device in ["rtx5090", "a100"] else 0.70
    else:
        ram_drain = 1.0 if device in ["rtx5090", "a100"] else 0.5
        cpu_step = 0.04 if device in ["rtx5090", "a100"] else 0.12
        gpu_step = 0.04 if device in ["rtx5090", "a100"] else 0.15
    return cpu_step, gpu_step, ram_drain

def evaluate_assignment(
    policy: str,
    assignments: Dict[str, str],
    telemetry: Dict[str, Dict[str, Any]],
    planner_status: str = "",
) -> Dict[str, Any]:
    missing = []
    infeasible = []
    total_cost = 0.0

    for stage in BENCHMARK_STAGES:
        device = assignments.get(stage)

        if device is None:
            missing.append(stage)
            continue

        if not device_feasible(stage, device, telemetry):
            infeasible.append(f"{stage}->{device}")
            continue

        total_cost += assignment_cost(stage, device, telemetry)

    success = len(missing) == 0 and len(infeasible) == 0
    status_upper = str(planner_status).upper()

    if success:
        failure_reason = "success"
    elif policy == "adaptive_planner" and (
        "UNSOLVABLE" in status_upper
        or "TIMEOUT" in status_upper
        or "EXCEPTION" in status_upper
        or "FAILED" in status_upper
    ):
        failure_reason = "planner_unsolved"
    elif len(missing) > 0:
        failure_reason = "missing_stage"
    elif len(infeasible) > 0:
        failure_reason = "constraint_violation"
    else:
        failure_reason = "other_failure"

    return {
        "policy": policy,
        "success": int(success),
        "failure_reason": failure_reason,
        "num_missing_stages": len(missing),
        "num_infeasible_assignments": len(infeasible),
        "total_cost": total_cost if success else np.nan,
        "planner_status": planner_status,
        "assignments": json.dumps(assignments, sort_keys=True),
        "missing_stages": json.dumps(missing),
        "infeasible_assignments": json.dumps(infeasible),
    }

def static_policy(telemetry: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    assignments = {}
    for stage in BENCHMARK_STAGES:
        if stage in STATIC_ASSIGNMENTS:
            assignments[stage] = STATIC_ASSIGNMENTS[stage]
        else:
            candidates = allowed_devices(stage)
            assignments[stage] = candidates[0]
    return assignments

def round_robin_policy(telemetry: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    assignments = {}
    cluster_state = json.loads(json.dumps(telemetry))
    cursor = 0

    for stage in BENCHMARK_STAGES:
        candidates = allowed_devices(stage)
        chosen = None

        for k in range(len(candidates)):
            device = candidates[(cursor + k) % len(candidates)]
            if device_feasible(stage, device, cluster_state):
                chosen = device
                cursor = (cursor + k + 1) % len(candidates)
                break

        if chosen is not None:
            assignments[stage] = chosen
            cpu_s, gpu_s, ram_d = compute_depletion(stage, chosen)
            cluster_state[chosen]["cpu_load"] = clamp(cluster_state[chosen]["cpu_load"] + cpu_s, 0.0, 1.0)
            cluster_state[chosen]["gpu_util"] = clamp(cluster_state[chosen]["gpu_util"] + gpu_s, 0.0, 1.0)
            cluster_state[chosen]["free_ram_gb"] = max(0.0, cluster_state[chosen]["free_ram_gb"] - ram_d)
    return assignments

def greedy_policy(telemetry: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    assignments = {}
    cluster_state = json.loads(json.dumps(telemetry))

    for stage in BENCHMARK_STAGES:
        candidates = [d for d in allowed_devices(stage) if device_feasible(stage, d, cluster_state)]
        if not candidates:
            continue

        chosen = min(candidates, key=lambda d: assignment_cost(stage, d, cluster_state))
        assignments[stage] = chosen

        cpu_s, gpu_s, ram_d = compute_depletion(stage, chosen)
        cluster_state[chosen]["cpu_load"] = clamp(cluster_state[chosen]["cpu_load"] + cpu_s, 0.0, 1.0)
        cluster_state[chosen]["gpu_util"] = clamp(cluster_state[chosen]["gpu_util"] + gpu_s, 0.0, 1.0)
        cluster_state[chosen]["free_ram_gb"] = max(0.0, cluster_state[chosen]["free_ram_gb"] - ram_d)
    return assignments

def rule_based_policy(telemetry: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    assignments = {}
    cluster_state = json.loads(json.dumps(telemetry))

    for stage in BENCHMARK_STAGES:
        fallbacks = RULE_FALLBACKS.get(stage, allowed_devices(stage))
        valid_fallbacks = [d for d in fallbacks if d in allowed_devices(stage)]
        if not valid_fallbacks:
            valid_fallbacks = allowed_devices(stage)

        chosen = None
        for device in valid_fallbacks:
            if device_feasible(stage, device, cluster_state):
                chosen = device
                break

        if chosen is not None:
            assignments[stage] = chosen
            cpu_s, gpu_s, ram_d = compute_depletion(stage, chosen)
            cluster_state[chosen]["cpu_load"] = clamp(cluster_state[chosen]["cpu_load"] + cpu_s, 0.0, 1.0)
            cluster_state[chosen]["gpu_util"] = clamp(cluster_state[chosen]["gpu_util"] + gpu_s, 0.0, 1.0)
            cluster_state[chosen]["free_ram_gb"] = max(0.0, cluster_state[chosen]["free_ram_gb"] - ram_d)
    return assignments

def adaptive_planner_policy(
    telemetry: Dict[str, Dict[str, Any]],
    engine_name: str,
) -> Tuple[Dict[str, str], str, Dict[str, Dict[str, Any]]]:
    try:

        planner_telemetry = json.loads(json.dumps(telemetry))
        constrained_thresholds = json.loads(json.dumps(THRESHOLDS))
        
        for stage in BENCHMARK_STAGES:
            if stage not in constrained_thresholds:
                constrained_thresholds[stage] = {}
            if stage in ["adversarial_training", "recovery_finetuning", "perturbation_generation"]:
                constrained_thresholds[stage]["min_free_ram_gb"] = 7.5
                constrained_thresholds[stage]["max_cpu_load"] = 0.85

        problem, _, _ = build_planning_problem(
            telemetry=planner_telemetry, # <-- Pass the reservation telemetry footprint
            stages=BENCHMARK_STAGES,
            capabilities=CAPABILITIES,
            thresholds=constrained_thresholds,
            completed_stages=[],
        )

        result, assignments = plan_assignments(problem, engine_name=engine_name)
        status = str(result.status)

        if result.plan is None:
            return {}, status, planner_telemetry
        
        return dict(assignments), status, planner_telemetry

    except Exception as e:
        return {}, f"planner_exception: {type(e).__name__}: {e}", telemetry

def load_hpc3_rows(n: int) -> List[Dict[str, Any]]:
    df = pd.read_csv(HPC3_PATH, sep="\t")
    df = df.sample(min(n, len(df)), random_state=SEED).reset_index(drop=True)
    rows = []
    for _, r in df.iterrows():
        cpu_load = 1.0 - float(r["cpu_idle"]) / 100.0
        mem_pressure = 1.0 - clamp(float(r["mem_free"]) / 1e9, 0.0, 1.0)
        temp = max(float(r["CPU1_Temp"]), float(r["CPU2_Temp"]), float(r["System_Temp"]))
        network_pressure = np.log1p(float(r["bytes_in"]) + float(r["bytes_out"])) / 25.0

        rows.append({
            "dataset": "hpc_node_telemetry",
            "cpu_pressure": clamp(cpu_load, 0.0, 1.0),
            "gpu_pressure": clamp(cpu_load * 0.65, 0.0, 1.0),
            "mem_pressure": clamp(mem_pressure, 0.0, 1.0),
            "latency_pressure": clamp(network_pressure, 0.0, 1.0),
            "thermal_pressure": clamp((temp - 35.0) / 50.0, 0.0, 1.0),
            "workload_variability": clamp(float(r["load_five"]) / 12.0, 0.0, 1.0),
        })
    return rows

def load_hpc4_rows(n: int) -> List[Dict[str, Any]]:
    df = pd.read_csv(HPC4_PATH)
    df = df.sample(min(n, len(df)), random_state=SEED).reset_index(drop=True)
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "dataset": "multi_cloud_service",
            "cpu_pressure": clamp(float(r["CPU_Utilization (%)"]) / 100.0, 0.0, 1.0),
            "gpu_pressure": clamp(float(r["CPU_Utilization (%)"]) / 120.0, 0.0, 1.0),
            "mem_pressure": clamp(float(r["Memory_Usage (MB)"]) / 10000.0, 0.0, 1.0),
            "latency_pressure": clamp(float(r["Service_Latency (ms)"]) / 200.0, 0.0, 1.0),
            "thermal_pressure": 0.0,
            "workload_variability": clamp(float(r["Workload_Variability"]) / 10.0, 0.0, 1.0),
            "qos": clamp(float(r["QoS_Score"]), 0.0, 1.0),
            "optimal_label": int(r["Optimal_Service_Placement"]),
        })
    return rows

def load_hpc5_rows(n: int) -> List[Dict[str, Any]]:
    df = pd.read_csv(HPC5_PATH)
    df = df.sample(min(n, len(df)), random_state=SEED).reset_index(drop=True)

    rows = []

    for _, r in df.iterrows():
        rows.append({
            "dataset": "cloud_task_scheduling",
            "cpu_pressure": clamp(float(r["CPU_Usage (%)"]) / 100.0, 0.0, 1.0),
            "gpu_pressure": clamp(float(r["CPU_Usage (%)"]) / 120.0, 0.0, 1.0),
            "mem_pressure": clamp(float(r["RAM_Usage (MB)"]) / 16000.0, 0.0, 1.0),
            "latency_pressure": clamp(float(r["Network_IO (MB/s)"]) / 100.0, 0.0, 1.0),
            "thermal_pressure": 0.0,
            "workload_variability": clamp(float(r["Priority"]) / 3.0, 0.0, 1.0),
            "execution_time": float(r["Execution_Time (s)"]),
            "optimal_label": int(r["Target (Optimal Scheduling)"]),
        })

    return rows

def make_telemetry_from_row(row: Dict[str, Any], scenario_id: int) -> Tuple[Dict[str, Dict[str, Any]], str, str]:
    telemetry = json.loads(json.dumps(BASE_TELEMETRY))
    stressed_device = DEVICES[scenario_id % len(DEVICES)]

    cpu_p = float(row["cpu_pressure"])
    gpu_p = float(row["gpu_pressure"])
    mem_p = float(row["mem_pressure"])
    lat_p = float(row["latency_pressure"])
    therm_p = float(row["thermal_pressure"])
    var_p = float(row["workload_variability"])

    event = "normal"
    if cpu_p > 0.85: event = "cpu_pressure"
    if mem_p > 0.85: event = "memory_pressure"
    if lat_p > 0.85: event = "latency_pressure"
    if var_p > 0.85: event = "workload_variability"
    if therm_p > 0.80: event = "thermal_pressure"

    telemetry[stressed_device]["cpu_load"] = max(telemetry[stressed_device]["cpu_load"], cpu_p)
    telemetry[stressed_device]["gpu_util"] = max(telemetry[stressed_device]["gpu_util"], gpu_p)
    telemetry[stressed_device]["free_ram_gb"] = max(0.25, telemetry[stressed_device]["free_ram_gb"] * (1.0 - 0.90 * mem_p))
    telemetry[stressed_device]["latency_ms"] = telemetry[stressed_device]["latency_ms"] * (1.0 + 6.0 * lat_p)
    telemetry[stressed_device]["reliability"] = clamp(telemetry[stressed_device]["reliability"] - 0.25 * var_p - 0.20 * therm_p, 0.05, 1.0)

    injected = random.choices(
        ["none", "device_unavailable", "resource_exhaustion", "latency_spike", "combined"],
        weights=[0.55, 0.02, 0.30, 0.10, 0.03], 
    )[0]

    if injected == "device_unavailable":
        telemetry[stressed_device]["available"] = False
        event = "device_unavailable"
    elif injected == "resource_exhaustion":
        telemetry[stressed_device]["cpu_load"] = 0.92
        telemetry[stressed_device]["gpu_util"] = 0.94
        telemetry[stressed_device]["free_ram_gb"] = 2.1 
        event = "resource_exhaustion"
    elif injected == "latency_spike":
        telemetry[stressed_device]["latency_ms"] = 1500.0
        event = "latency_spike"
    elif injected == "combined":
        telemetry[stressed_device]["available"] = False
        other = DEVICES[(scenario_id + 1) % len(DEVICES)]
        telemetry[other]["cpu_load"] = 0.96
        telemetry[other]["gpu_util"] = 0.96
        telemetry[other]["free_ram_gb"] = 0.75
        event = "combined_failure"

    return telemetry, event, stressed_device

def make_scenarios(n_per_dataset: int) -> List[Dict[str, Any]]:
    raw_rows = []
    raw_rows.extend(load_hpc3_rows(n_per_dataset))
    raw_rows.extend(load_hpc4_rows(n_per_dataset))
    raw_rows.extend(load_hpc5_rows(n_per_dataset))
    scenarios = []
    for i, row in enumerate(raw_rows):
        telemetry, event, stressed_device = make_telemetry_from_row(row, i)
        scenarios.append({
            "scenario_id": i,
            "dataset": row["dataset"],
            "event": event,
            "stressed_device": stressed_device,
            "telemetry": telemetry,
        })
    return scenarios

def run_policy(
    policy: str,
    telemetry: Dict[str, Dict[str, Any]],
    engine_name: str,
) -> Dict[str, Any]:
    if policy == "static":
        assignments = static_policy(telemetry)
        return evaluate_assignment(policy, assignments, telemetry)

    if policy == "round_robin":
        assignments = round_robin_policy(telemetry)
        return evaluate_assignment(policy, assignments, telemetry)

    if policy == "greedy":
        assignments = greedy_policy(telemetry)
        return evaluate_assignment(policy, assignments, telemetry)

    if policy == "rule_based":
        assignments = rule_based_policy(telemetry)
        return evaluate_assignment(policy, assignments, telemetry)

    if policy == "adaptive_planner":
        assignments, status, eval_telemetry = adaptive_planner_policy(
            telemetry,
            engine_name,
        )
        return evaluate_assignment(
            policy,
            assignments,
            eval_telemetry,
            planner_status=status,
        )

    raise ValueError(policy)

def save_figures(summary: pd.DataFrame) -> None:
    order = ["static", "round_robin", "greedy", "rule_based", "adaptive_planner"]

    plt.figure(figsize=(10, 5))
    pivot = summary.pivot(index="policy", columns="dataset", values="success_rate")
    pivot = pivot.reindex(order)
    pivot.plot(kind="bar", ax=plt.gca())
    plt.ylabel("Success rate (%)")
    plt.title("Policy success rate across real-derived scenarios")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUT / "success_rate_by_policy.png", dpi=300)

    plt.figure(figsize=(10, 5))
    pivot = summary.pivot(index="policy", columns="dataset", values="mean_infeasible_assignments")
    pivot = pivot.reindex(order)
    pivot.plot(kind="bar", ax=plt.gca())
    plt.ylabel("Mean infeasible stage assignments")
    plt.title("Policy-specific infeasible assignments")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUT / "infeasible_assignments_by_policy.png", dpi=300)

    plt.figure(figsize=(10, 5))
    pivot = summary.pivot(index="policy", columns="dataset", values="mean_total_cost")
    pivot = pivot.reindex(order)
    pivot.plot(kind="bar", ax=plt.gca())
    plt.ylabel("Mean planner cost")
    plt.title("Cost of successful assignments")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUT / "assignment_cost_by_policy.png", dpi=300)

    plt.figure(figsize=(10, 5))
    pivot = summary.pivot(index="policy", columns="dataset", values="mean_missing_stages")
    pivot = pivot.reindex(order)
    pivot.plot(kind="bar", ax=plt.gca())
    plt.ylabel("Mean missing stages")
    plt.title("Policy-specific missing workflow stages")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUT / "missing_stages_by_policy.png", dpi=300)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-dataset", type=int, default=100)
    parser.add_argument("--up-engine", type=str, default="tamer")
    parser.add_argument("--skip-planner", action="store_true")
    args = parser.parse_args()

    policies = list(POLICIES)
    if args.skip_planner:
        policies.remove("adaptive_planner")

    scenarios = make_scenarios(args.n_per_dataset)
    rows = []

    for s in scenarios:
        for policy in policies:
            result = run_policy(
                policy=policy,
                telemetry=s["telemetry"],
                engine_name=args.up_engine,
            )
            rows.append({
                "scenario_id": s["scenario_id"],
                "dataset": s["dataset"],
                "event": s["event"],
                "stressed_device": s["stressed_device"],
                **result,
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "policy_results.csv", index=False)

    summary = (
        df.groupby(["dataset", "policy"])
        .agg(
            success_rate=("success", lambda x: 100.0 * float(np.mean(x))),
            mean_missing_stages=("num_missing_stages", "mean"),
            mean_infeasible_assignments=("num_infeasible_assignments", "mean"),
            mean_total_cost=("total_cost", "mean"),
            n=("success", "count"),
        )
        .reset_index()
    )
    summary.to_csv(OUT / "policy_summary.csv", index=False)

    event_summary = (
        df.groupby(["dataset", "event", "policy"])
        .agg(
            success_rate=("success", lambda x: 100.0 * float(np.mean(x))),
            mean_infeasible_assignments=("num_infeasible_assignments", "mean"),
            n=("success", "count"),
        )
        .reset_index()
    )
    event_summary.to_csv(OUT / "policy_event_summary.csv", index=False)
    
    failure_summary = (
        df[df["success"] == 0]
        .groupby(["dataset", "policy", "failure_reason"])
        .size()
        .reset_index(name="count")
    )
    failure_summary.to_csv(OUT / "policy_failure_reason_summary.csv", index=False)

    print("\n=== POLICY SUMMARY ===")
    print(summary)

    print("\n=== PLANNER STATUSES ===")
    if "planner_status" in df.columns:
        print(df[df["policy"] == "adaptive_planner"]["planner_status"].value_counts())

    save_figures(summary)

    print("\nSaved:")
    print(OUT / "policy_results.csv")
    print(OUT / "policy_summary.csv")
    print(OUT / "policy_event_summary.csv")
    print(OUT / "success_rate_by_policy.png")
    print(OUT / "infeasible_assignments_by_policy.png")
    print(OUT / "assignment_cost_by_policy.png")

if __name__ == "__main__":
    main()