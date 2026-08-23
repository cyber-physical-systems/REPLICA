import argparse
import json
import random
import time


DEVICE_PROFILES = {
    "a100": {
        "capability": "classification",
        "base_confidence": 0.88,
        "latency": 0.45,
        "battery_cost": 0.35,
    },
    "gpu_2": {
        "capability": "classification",
        "base_confidence": 0.82,
        "latency": 0.55,
        "battery_cost": 0.40,
    },
    "jetson": {
        "capability": "classification",
        "base_confidence": 0.74,
        "latency": 0.75,
        "battery_cost": 0.15,
    },
    "rtx5090": {
        "capability": "robust_classification",
        "base_confidence": 0.91,
        "latency": 0.80,
        "battery_cost": 0.70,
    },
    "rtx4080s": {
        "capability": "fusion",
        "base_confidence": 0.86,
        "latency": 0.65,
        "battery_cost": 0.55,
    },
}


def run_task(device, attacked=False, resource_degraded=False):
    profile = DEVICE_PROFILES[device].copy()

    confidence = profile["base_confidence"]
    latency = profile["latency"]
    battery_cost = profile["battery_cost"]

    if attacked:
        confidence -= random.uniform(0.25, 0.45)

    if resource_degraded:
        latency *= random.uniform(1.5, 2.5)
        battery_cost *= random.uniform(1.2, 1.8)

    time.sleep(min(latency, 2.0))

    return {
        "device": device,
        "capability": profile["capability"],
        "confidence": round(confidence, 3),
        "latency": round(latency, 3),
        "battery_cost": round(battery_cost, 3),
        "attacked": attacked,
        "resource_degraded": resource_degraded,
        "success": confidence >= 0.75,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument("--attacked", action="store_true")
    parser.add_argument("--resource-degraded", action="store_true")
    args = parser.parse_args()

    result = run_task(
        device=args.device,
        attacked=args.attacked,
        resource_degraded=args.resource_degraded,
    )

    print(json.dumps(result))