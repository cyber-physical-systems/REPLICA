#!/usr/bin/env python3

import copy
import json
from pathlib import Path

import pandas as pd

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
    "adversarial_training": ["a100", "rtx5090", "rtx4080s", "gpu_2"],
    "perturbation_generation": ["a100", "rtx5090", "rtx4080s", "gpu_2"],
    "metric_computation": ["a100", "rtx5090", "rtx4080s", "gpu_2", "jetson"],
    "rais_scoring": ["a100", "rtx5090", "rtx4080s", "gpu_2", "jetson"],
    "pruning_decision": ["a100", "rtx5090", "rtx4080s", "gpu_2", "jetson"],
    "recovery_finetuning": ["a100", "rtx5090", "rtx4080s", "gpu_2"],
    "model_evaluation": ["a100", "rtx5090", "rtx4080s", "gpu_2", "jetson"],
    "deploy_updated_model": ["jetson", "gpu_2", "rtx4080s"],
}


THRESHOLDS = {
    s: {
        "min_free_ram_gb": 2.0,
        "max_cpu_load": 0.95,
        "max_gpu_util": 0.95,
        "max_latency_ms": 500.0,
        "min_reliability": 0.0,
    }
    for s in STAGES
}


STATIC_ASSIGNMENTS = {
    "adversarial_training": "a100",
    "perturbation_generation": "rtx5090",
    "metric_computation": "rtx5090",
    "rais_scoring": "rtx5090",
    "pruning_decision": "rtx5090",
    "recovery_finetuning": "gpu_2",
    "model_evaluation": "rtx5090",
    "deploy_updated_model": "rtx4080s",
}


FALLBACK_ORDER = {
    "a100": ["rtx5090", "rtx4080s", "gpu_2"],
    "rtx5090": ["a100", "rtx4080s", "gpu_2"],
    "rtx4080s": ["rtx5090", "gpu_2"],
    "gpu_2": ["rtx4080s", "rtx5090"],
    "jetson": ["gpu_2", "rtx4080s"],
}


STAGE_RUNTIME_SEC = {
    "adversarial_training": {
        "a100": 900,
        "rtx5090": 1200,
        "rtx4080s": 1600,
        "gpu_2": 2400,
    },

    "perturbation_generation": {
        "a100": 180,
        "rtx5090": 240,
        "rtx4080s": 300,
        "gpu_2": 500,
    },

    "metric_computation": {
        "a100": 120,
        "rtx5090": 160,
        "rtx4080s": 200,
        "gpu_2": 300,
        "jetson": 700,
    },

    "rais_scoring": {
        "a100": 40,
        "rtx5090": 50,
        "rtx4080s": 60,
        "gpu_2": 90,
        "jetson": 180,
    },

    "pruning_decision": {
        "a100": 30,
        "rtx5090": 40,
        "rtx4080s": 50,
        "gpu_2": 80,
        "jetson": 150,
    },

    "recovery_finetuning": {
        "a100": 800,
        "rtx5090": 1100,
        "rtx4080s": 1400,
        "gpu_2": 2200,
    },

    "model_evaluation": {
        "a100": 180,
        "rtx5090": 220,
        "rtx4080s": 280,
        "gpu_2": 400,
        "jetson": 800,
    },

    "deploy_updated_model": {
        "jetson": 60,
        "gpu_2": 45,
        "rtx4080s": 40,
    },
}
def normalize_telemetry(raw):
    out = {}

    for name, rec in raw.items():
        if rec.get("status") != "success":
            out[name] = {
                "available": False,
                "cpu_load": 1.0,
                "gpu_util": 1.0,
                "free_ram_gb": 0.0,
                "latency_ms": 99999.0,
                "reliability": 0.0,
            }
            continue

        out[name] = {
            "available": True,
            "cpu_load": float(rec.get("cpu_load_percent") or 100.0) / 100.0,
            "gpu_util": float(rec.get("gpu_util_percent") or 0.0) / 100.0,
            "free_ram_gb": float(rec.get("ram_available_gb") or 0.0),
            "latency_ms": 50.0,
            "reliability": 0.95,
        }

    return out


def feasible(device, telemetry):
    t = telemetry.get(device)
    if not t:
        return False

    return (
        t["available"]
        and t["cpu_load"] <= 0.95
        and t["gpu_util"] <= 0.95
        and t["free_ram_gb"] >= 2.0
        and t["latency_ms"] <= 500.0
        and t["reliability"] >= 0.0
    )


def plan_runtime(assignments):
    total = 0
    missing = []

    for stage, device in assignments.items():
        stage_times = STAGE_RUNTIME_SEC.get(stage, {})
        if device not in stage_times:
            missing.append((stage, device))
            total += 999999
        else:
            total += stage_times[device]

    return total, missing


def static_policy(telemetry):
    for stage, device in STATIC_ASSIGNMENTS.items():
        if not feasible(device, telemetry):
            runtime, missing = plan_runtime(STATIC_ASSIGNMENTS)
            return {
                "success": False,
                "assignments": STATIC_ASSIGNMENTS,
                "runtime_sec": None,
                "missing_runtime": missing,
                "reason": f"{stage} assigned to infeasible device {device}",
            }

    runtime, missing = plan_runtime(STATIC_ASSIGNMENTS)
    return {
        "success": True,
        "assignments": STATIC_ASSIGNMENTS,
        "runtime_sec": runtime,
        "missing_runtime": missing,
        "reason": "all static assignments feasible",
    }


def rule_based_policy(telemetry):
    assignments = {}

    for stage, original in STATIC_ASSIGNMENTS.items():
        if feasible(original, telemetry):
            assignments[stage] = original
            continue

        replacement = None
        for candidate in FALLBACK_ORDER.get(original, []):
            if candidate in CAPABILITIES.get(stage, []) and feasible(candidate, telemetry):
                replacement = candidate
                break

        if replacement is None:
            runtime, missing = plan_runtime(assignments)
            return {
                "success": False,
                "assignments": assignments,
                "runtime_sec": None,
                "missing_runtime": missing,
                "reason": f"no fallback for {stage} from {original}",
            }

        assignments[stage] = replacement

    runtime, missing = plan_runtime(assignments)
    return {
        "success": True,
        "assignments": assignments,
        "runtime_sec": runtime,
        "missing_runtime": missing,
        "reason": "fallback rules found feasible devices",
    }


def planner_policy(telemetry):
    try:
        problem, _, _ = build_planning_problem(
            telemetry=telemetry,
            stages=STAGES,
            capabilities=CAPABILITIES,
            thresholds=THRESHOLDS,
            completed_stages=[],
        )

        result, assignments = plan_assignments(problem, engine_name="tamer")
        success = all(stage in assignments for stage in STAGES)

        runtime, missing = plan_runtime(assignments)

        return {
            "success": success,
            "assignments": assignments,
            "runtime_sec": runtime if success else None,
            "missing_runtime": missing,
            "reason": str(result.status),
        }

    except Exception as e:
        return {
            "success": False,
            "assignments": {},
            "runtime_sec": None,
            "missing_runtime": [],
            "reason": str(e),
        }


def make_scenarios(base):
    scenarios = {}

    scenarios["nominal"] = copy.deepcopy(base)

    s = copy.deepcopy(base)
    s["a100"]["available"] = False
    scenarios["a100_failure"] = s

    s = copy.deepcopy(base)
    s["jetson"]["cpu_load"] = 0.99
    s["jetson"]["free_ram_gb"] = 1.0
    scenarios["jetson_degraded"] = s

    s = copy.deepcopy(base)
    s["a100"]["available"] = False
    s["rtx5090"]["cpu_load"] = 0.99
    s["rtx5090"]["gpu_util"] = 0.99
    scenarios["combined_a100_down_5090_busy"] = s

    return scenarios


def main():
    raw_path = Path("telemetry_runs/latest_telemetry.json")
    if not raw_path.exists():
        raise FileNotFoundError(
            "Missing telemetry_runs/latest_telemetry.json. "
            "Run collect_telemetry.py first."
        )

    raw = json.loads(raw_path.read_text())
    base = normalize_telemetry(raw)
    scenarios = make_scenarios(base)

    rows = []
    details = {}

    policies = {
        "Static": static_policy,
        "Rule-Based": rule_based_policy,
        "Planner": planner_policy,
    }

    for scenario_name, telemetry in scenarios.items():
        details[scenario_name] = {}

        for policy_name, policy_fn in policies.items():
            print(f"\n=== {scenario_name} | {policy_name} ===")
            result = policy_fn(telemetry)

            details[scenario_name][policy_name] = result

            rows.append({
                "scenario": scenario_name,
                "policy": policy_name,
                "success": result["success"],
                "runtime_sec": result["runtime_sec"],
                "runtime_min": (
                    result["runtime_sec"] / 60.0
                    if result["runtime_sec"] is not None
                    else None
                ),
                "reason": result["reason"],
                "assignments": json.dumps(result["assignments"]),
                "missing_runtime": json.dumps(result["missing_runtime"]),
            })

            print("success:", result["success"])
            print("runtime_sec:", result["runtime_sec"])
            print("reason:", result["reason"])
            print("assignments:", result["assignments"])

    out_dir = Path("telemetry_runs")
    out_dir.mkdir(exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "policy_comparison_results.csv", index=False)

    Path(out_dir / "policy_comparison_results.json").write_text(
        json.dumps(details, indent=2)
    )

    success_table = df.pivot(
        index="scenario",
        columns="policy",
        values="success",
    )

    runtime_table = df.pivot(
        index="scenario",
        columns="policy",
        values="runtime_sec",
    )

    print("\n=== SUCCESS SUMMARY ===")
    print(success_table)

    print("\n=== RUNTIME SUMMARY SEC ===")
    print(runtime_table)

    success_table.to_csv(out_dir / "policy_comparison_success_summary.csv")
    runtime_table.to_csv(out_dir / "policy_comparison_runtime_summary.csv")

    print("\nSaved:")
    print("telemetry_runs/policy_comparison_results.csv")
    print("telemetry_runs/policy_comparison_results.json")
    print("telemetry_runs/policy_comparison_success_summary.csv")
    print("telemetry_runs/policy_comparison_runtime_summary.csv")


if __name__ == "__main__":
    main()