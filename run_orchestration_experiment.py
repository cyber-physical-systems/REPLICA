import json
import subprocess
import pandas as pd
from device_config import DEVICE_CONFIG


CONF_THRESH = 0.75
LATENCY_BUDGET = 1.25
BATTERY_BUDGET = 0.85


SCENARIOS = [
    {
        "name": "nominal",
        "offline_devices": [],
        "attacked_devices": [],
        "resource_degraded_devices": [],
    },
    {
        "name": "resource_failure",
        "offline_devices": ["a100"],
        "attacked_devices": [],
        "resource_degraded_devices": [],
    },
    {
        "name": "adversarial_degradation",
        "offline_devices": [],
        "attacked_devices": ["a100"],
        "resource_degraded_devices": [],
    },
    {
        "name": "resource_degradation",
        "offline_devices": [],
        "attacked_devices": [],
        "resource_degraded_devices": ["a100"],
    },
    {
        "name": "combined_failure",
        "offline_devices": ["a100"],
        "attacked_devices": ["gpu_2"],
        "resource_degraded_devices": ["jetson"],
    },
]


STATIC_WORKFLOW = ["a100"]


RULE_BASED_ORDER = [
    "a100",
    "gpu_2",
    "jetson",
    "rtx5090",
    "rtx4080s",
]


def run_remote(device_label, scenario):
    cfg = DEVICE_CONFIG[device_label]

    cmd = [
        "ssh",
        cfg["ssh"],
        f"cd {cfg['project_dir']} && "
        f"{cfg['python']} worker_agent.py "
        f"--device {device_label}"
        f"{' --attacked' if device_label in scenario['attacked_devices'] else ''}"
        f"{' --resource-degraded' if device_label in scenario['resource_degraded_devices'] else ''}"
    ]

    try:
        output = subprocess.check_output(
            cmd,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=20,
        )
        return json.loads(output.strip().splitlines()[-1])

    except Exception as e:
        return {
            "device": device_label,
            "success": False,
            "error": str(e),
            "confidence": 0.0,
            "latency": 999.0,
            "battery_cost": 999.0,
        }


def satisfies_constraints(result):
    return (
        result["confidence"] >= CONF_THRESH
        and result["latency"] <= LATENCY_BUDGET
        and result["battery_cost"] <= BATTERY_BUDGET
    )


def static_pipeline(scenario):
    selected = STATIC_WORKFLOW[0]

    if selected in scenario["offline_devices"]:
        return False, selected, "offline"

    result = run_remote(selected, scenario)

    if satisfies_constraints(result):
        return True, selected, "success"

    return False, selected, "constraint_violation"


def rule_based_orchestrator(scenario):
    for device in RULE_BASED_ORDER:
        if device in scenario["offline_devices"]:
            continue

        result = run_remote(device, scenario)

        if satisfies_constraints(result):
            return True, device, "success"

    return False, None, "no_rule_satisfied"


def pddl_style_planner(scenario):
    candidates = []

    for device in DEVICE_CONFIG:
        if device in scenario["offline_devices"]:
            continue

        result = run_remote(device, scenario)

        if satisfies_constraints(result):
            candidates.append(result)

    if not candidates:
        return False, None, "no_valid_plan"

    best = max(candidates, key=lambda r: r["confidence"])
    return True, best["device"], "success"


def run_all():
    rows = []

    orchestrators = {
        "static_pipeline": static_pipeline,
        "rule_based": rule_based_orchestrator,
        "pddl_style_planner": pddl_style_planner,
    }

    for scenario in SCENARIOS:
        for orch_name, orch_fn in orchestrators.items():
            success, selected_device, reason = orch_fn(scenario)

            rows.append({
                "scenario": scenario["name"],
                "orchestrator": orch_name,
                "success": success,
                "selected_device": selected_device,
                "reason": reason,
                "offline_devices": ",".join(scenario["offline_devices"]),
                "attacked_devices": ",".join(scenario["attacked_devices"]),
                "resource_degraded_devices": ",".join(
                    scenario["resource_degraded_devices"]
                ),
            })

    df = pd.DataFrame(rows)
    df.to_csv("orchestration_pre_experiment_results.csv", index=False)

    print(df.to_string(index=False))
    print("\nSaved: orchestration_pre_experiment_results.csv")


if __name__ == "__main__":
    run_all()