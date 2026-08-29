from pathlib import Path
import subprocess
import json
import sys
import os
import yaml


from discover_devices import discover_online_devices
from devices_config import executable_device_labels

FAILED_DEVICES = {
    x.strip().lower()
    for x in os.environ.get("FAILED_DEVICES", "").split(",")
    if x.strip()
}


SHARED_ROOT = "/home/widney/Documents/rais_venv/Security-aware-Model-Compression/sc26/shared_runs"
RUN_ID = "run_001"
RUN_ROOT = Path(f"{SHARED_ROOT}/{RUN_ID}")
RUN_STATE_PATH = RUN_ROOT / "orchestrator_state.json"

STAGE_ORDER = [
    "adversarial_training",
    "perturbation_generation",
    "metric_computation",
    "rais_scoring",
    "pruning_decision",
    "recovery_finetuning",
    "model_evaluation",
    "deploy_updated_model",
]

def device_predicates(dev: dict) -> list[str]:
    # Use 'dev' consistently (fixed the 'd' bug)
    label = dev["label"]
    preds = []

    # 1. Handle Connectivity & Orchestrator-level failure
    # If the probe failed, the device is unreachable
    if dev.get("probe_failed") or dev.get("telemetry_failed"):
        return [f"(device-unreachable {label})"]

    # Check if the device is currently in the global 'isolated' list
    # Use label.lower() to prevent case-sensitivity issues
    if any(f.lower() == label.lower() for f in FAILED_DEVICES):
        return [f"(device-failed {label})"]

    # If it survived the checks above, it's available
    preds.append(f"(available {label})")

    # --------------------------------------------------
    # Static Capabilities & Roles
    # --------------------------------------------------
    is_workstation = any(x in label.lower() for x in ["gpu", "gpu_2", "ncps"])
    has_cuda = bool(dev.get("has_cuda", False))

    if has_cuda:
        preds.append(f"(has-gpu {label})")
    else:
        preds.append(f"(cpu-device {label})")

    # Role Assignment (VIP Logic)
    if is_workstation and has_cuda:
        preds.append(f"(workstation-gpu {label})")
        # Preference Logic: Only ncps and gpu_2 get the 'preferred' tag
        if label.lower() in ["ncps", "gpu_2"]:
            preds.append(f"(preferred-workstation {label})")
    
    if label.lower().startswith("pi"):
        preds.append(f"(edge-device {label})")
    
    if "jetson" in label.lower():
        # Jetson is a hybrid: has-gpu but also an edge-device
        preds.append(f"(edge-device {label})")

    # --------------------------------------------------
    # Telemetry → Symbolic Predicates
    # --------------------------------------------------
    
    # CPU Logic
    cpu_load = dev.get("cpu_load_percent", 0)
    if is_workstation:
        preds.append(f"(cpu-free {label})") # Workstations are never 'too busy' for us
    elif cpu_load < 50:
        preds.append(f"(cpu-free {label})")
    else:
        preds.append(f"(cpu-busy {label})")

    # RAM Logic (The Fix)
    ram_free = dev.get("free_ram_gb", 0.0)
    
    if is_workstation:
        # Force high-mem for your big machines so the planner doesn't choke 
        # even if the telemetry probe returns a weird value.
        preds.append(f"(high-mem {label})")
    elif ram_free > 4.0:
        preds.append(f"(high-mem {label})")
    else:
        preds.append(f"(low-mem {label})")

    # Latency Logic
    # Filter out None values before calculating min()
    latencies = [v for k, v in dev.items() if k.startswith("latency_ms") and v is not None]
    if latencies:
        best_lat = min(latencies)
        if best_lat < 30:
            preds.append(f"(low-latency {label})")
        else:
            preds.append(f"(high-latency {label})")
    else:
        # Default to low-latency if unknown to keep the plan moving
        preds.append(f"(low-latency {label})")

    return preds

def build_problem_text(devices: list[dict]) -> str:
    # 1. LOAD STATE FIRST
    state = {}
    if RUN_STATE_PATH.exists():
        state = json.loads(RUN_STATE_PATH.read_text())
    
    device_objs = " ".join(d["label"] for d in devices)
    
    preds = []
    
    # 2. Add Security Predicate if the model is flagged
    if state.get("model_vulnerable"):
        preds.append("(model-vulnerable)")
        
    # 3. Add Device Predicates
    for d in devices:
        preds.extend(device_predicates(d))

    preds.append("(= (total-cost) 0)")

    # 4. Handle Workflow Stage Logic
    completed_stages = set(state.get("completed_stages", []))
    
    for stage in STAGE_ORDER:
        if stage in completed_stages:
            preds.append(f"(completed {stage})")

    remaining = [s for s in STAGE_ORDER if s not in completed_stages]
    if remaining:
        preds.append(f"(ready {remaining[0]})")

    init_block = "\n    ".join(preds)

    return f"""(define (problem rais_run_01)
  (:domain rais_orchestration)

  (:objects
    {device_objs} - device
  )

  (:init
    {init_block}
  )

  (:goal
    (completed deploy_updated_model)
  )

  (:metric minimize (total-cost))
)
"""
    

def enrich_with_telemetry(devices: list[dict]) -> list[dict]:
    enriched = []

    HOSTS = {
        "jetson": "cpslab@100.77.214.32",
        "ncps":   "NCPS@100.114.8.110",
        "gpu_2":  "widney@100.104.178.47",
        "gpu":    "widney@100.97.5.96",
        "pi5a":   "widney@100.127.157.32",
        "pi5d":   "pi5d@100.73.148.79",
    }

    PY_PATH = {
        "jetson": "/home/cpslab/Documents/rais_venv/bin/python",
        "ncps":   "/home/NCPS/miniconda3/bin/python3",
        "gpu_2":  "/home/widney/Documents/rais_venv/bin/python",
        "gpu":    "/home/widney/Documents/rais_venv/bin/python",
        "pi5a":   "/home/widney/rais_project/bin/python",
        "pi5d":   "/home/pi5d/rais_project/bin/python3",
    }

    SCRIPT_PATH = {
        "jetson": "/home/cpslab/Documents/rais_venv/Security-aware-Model-Compression/sc26/device_telemetry.py",
        "ncps":   "/home/NCPS/sc26/device_telemetry.py",
        "gpu_2":  "/home/widney/Documents/rais_venv/Security-aware-Model-Compression/sc26/device_telemetry.py",
        "gpu":    "/home/widney/Documents/rais_venv/Security-aware-Model-Compression/sc26/device_telemetry.py",
        "pi5a":   "/home/widney/Documents/rais_venv/Security-aware-Model-Compression/sc26/device_telemetry.py",
        "pi5d":   "/home/pi5d/Documents/rais_venv/Security-aware-Model-Compression/sc26/device_telemetry.py",
    }


    for d in devices:
        label = d["label"]
    
        # Skip labels you have not configured yet
        if label not in HOSTS:
            enriched.append(d)
            continue
    
        telemetry = None  
    
        try:
            cmd = f'{PY_PATH[label]} {SCRIPT_PATH[label]} --label {label}'
    
            result = subprocess.run(
                ["ssh", HOSTS[label], cmd],
                capture_output=True,
                text=True,
                timeout=15,
            )
    
            if result.returncode != 0:
                print(f"[warn] telemetry failed for {label}: {result.stderr.strip()}")
                d["telemetry_failed"] = True
            else:
                telemetry = json.loads(result.stdout.strip()) if result.stdout.strip() else {}
        
        except Exception as e:
            print(f"[warn] telemetry failed for {label}: {e}")
            d["telemetry_failed"] = True
            
        if telemetry is not None:
            d.update(telemetry)
        else:
            d["telemetry_missing"] = True
    
        enriched.append(d)
    
    return enriched


# --------------------------
# Main
# --------------------------
def main():
    online_devices = discover_online_devices()
    trusted = executable_device_labels()

    devices = [
        d for d in online_devices
        if d["label"].lower() in trusted
    ]

    if not devices:
        raise RuntimeError("No valid devices after filtering.")

    devices = enrich_with_telemetry(devices)

    print("\n[orchestrate] Online devices:")
    for d in devices:
        print(f"  {d['label']}")

    out = Path("problem.pddl")
    out.write_text(build_problem_text(devices))
    print(f"Wrote {out.resolve()}")


if __name__ == "__main__":
    main()
