#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import time
import csv
from pathlib import Path
from typing import Dict, List, Tuple
import yaml

from discover_devices import discover_online_devices
from up_integration import get_up_assignments
from telemetry_adapter import get_exact_telemetry_optimal_assignments
from build_problem import enrich_with_telemetry
from devices_config import executable_device_labels
from up_model_builder import build_planning_problem
from up_planner_runner import plan_assignments
from up_integration import STAGES, CAPABILITIES, THRESHOLDS
from event_detector import EventDetector
from dataclasses import dataclass
from typing import Optional, Dict, Any
import datetime


# --------------------------------------------------
# Device inventory (SC26 rebuttal)
# --------------------------------------------------

DEVICE_CONFIG = {
    "a100": {
        "ssh": "root@47.161.216.236",
        "ssh_opts": ["-p", "40381"],
        "python": "/venv/sc26_py311/bin/python",
        "project_dir": "/workspace/sc26_rebuttal",
        "role": "primary_classifier",
    },
    "gpu_2": {
        "ssh": "widney@100.87.63.83",
        "ssh_opts": [],
        "python": "/home/widney/Documents/rais_venv/bin/python",
        "project_dir": "/home/widney/Documents/rais_venv/Security-aware-Model-Compression/sc26_rebuttal",
        "role": "backup_classifier",
    },
    "jetson": {
        "ssh": "cpslab@100.114.155.116",
        "ssh_opts": ["-o", "ConnectTimeout=10"],
        "python": "/home/cpslab/Documents/rais_venv/bin/python",
        "project_dir": "/home/cpslab/Documents/rais_venv/Security-aware-Model-Compression/sc26_rebuttal",
        "role": "edge_low_power",
    },
    "rtx5090": {
        "ssh": "root@162.142.111.64",
        "ssh_opts": ["-p", "23373"],
        "python": "/venv/sc26_py311/bin/python",
        "project_dir": "/workspace/sc26_rebuttal",
        "role": "robust_model",
    },
    "rtx4080s": {
        "ssh": "root@216.49.136.176",
        "ssh_opts": ["-p", "31192"],
        "python": "/venv/sc26_py311/bin/python",
        "project_dir": "/workspace/sc26_rebuttal",
        "role": "fusion_controller",
    },
}


def executable_device_labels() -> set[str]:
    return set(DEVICE_CONFIG.keys())


# --------------------------------------------------
# Convenience maps
# --------------------------------------------------

SSH_TARGETS = {
    k: v["ssh"]
    for k, v in DEVICE_CONFIG.items()
}

PY_BY_HOST = {
    k: v["python"]
    for k, v in DEVICE_CONFIG.items()
}

REMOTE_PROJECT_DIRS = {
    k: v["project_dir"]
    for k, v in DEVICE_CONFIG.items()
}

SSH_OPTS_BY_HOST = {
    k: v.get("ssh_opts", [])
    for k, v in DEVICE_CONFIG.items()
}


SHARED_ROOT = "/home/widney/Documents/rais_venv/Security-aware-Model-Compression/sc26/shared_runs"
RUN_ID = "run_001"
RUN_ROOT = f"{SHARED_ROOT}/{RUN_ID}"
STAGE_METRICS_PATH = Path(RUN_ROOT) / "orchestrator_stage_runtime_metrics.csv"
WORKFLOW_METRICS_PATH = Path(RUN_ROOT) / "workflow_metrics.json"
MAX_REPLANS = 15

LAST_TRIGGERED_EVENTS = set()

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

ASSIGN_RE = re.compile(r"^\s*\(?assign-([a-zA-Z0-9_-]+)\s+([a-zA-Z0-9_-]+)")
RUN_RE = re.compile(r"^\s*\(?run-([a-zA-Z0-9_-]+)\s+([a-zA-Z0-9_-]+)")

JETSON_YOLO_EVENT_PATH = Path("/tmp/jetson_yolo_event.json")
PI_RESNET_EVENT_PATH = Path("/tmp/pi_resnet_event.json")
DOMAIN_PATH = Path("domain.pddl")
PROBLEM_PATH = Path("problem.pddl")
PLAN_PATH = Path("sas_plan")
RUN_STATE_PATH = Path(RUN_ROOT) / "orchestrator_state.json"
EVAL_FAMILIES = {
    "gradient_based": ["fgsm", "pgd", "cw"],
    "localized": ["patch", "square"],
    "transformation_based": ["spatial", "color_shift"],
    "query_based": ["boundary"],
    "distribution_shift": ["universal"],
}

EVAL_FAMILY_ASSIGNMENTS = {
    "a100": [
        "gradient_based",
        "query_based",
    ],

    "rtx5090": [
        "gradient_based",
        "localized",
        "distribution_shift",
        "query_based",
    ],

    "rtx4080s": [
        "localized",
        "distribution_shift",
        "transformation_based",
    ],

    "gpu_2": [
        "gradient_based",
        "localized",
        "distribution_shift",
        "transformation_based",
        "query_based",
    ],

    "jetson": [
        "transformation_based",
    ],
}

EVAL_FAMILY_CANDIDATES = {

    "gradient_based": [
        "a100",
        "rtx5090",
        "gpu_2",
        "rtx4080s",
    ],

    "localized": [
        "rtx5090",
        "rtx4080s",
        "gpu_2",
        "a100",
    ],

    "distribution_shift": [
        "rtx5090",
        "rtx4080s",
        "gpu_2",
        "a100",
    ],

    "query_based": [
        "a100",
        "rtx5090",
        "gpu_2",
    ],

    "transformation_based": [
        "jetson",
        "gpu_2",
        "rtx4080s",
        "rtx5090",
    ],
}

PLANNER_CMD = [
    "./downward/fast-downward.py",
    str(DOMAIN_PATH),
    str(PROBLEM_PATH),
    "--search",
    "astar(lmcut())",
]

## For Experiment B

FAULT_INJECTION = {
    "gpu": {
        "cpu_load_percent": 70.0,
        "gpu_util_percent": 85.0,
        "ram_available_gb": 20.0,
    },
    "gpu_2": {
        "cpu_load_percent": 2.0,
        "gpu_util_percent": 5.0,
        "ram_available_gb": 120.0,
    },
    "jetson": {
        "cpu_load_percent": 10.0,
        "gpu_util_percent": 15.0,
        "ram_available_gb": 4.5,
    },
}


@dataclass
class RecoveryRequest:
    event: str
    severity: str
    source: str
    details: Dict[str, Any]

@dataclass
class ModelEvent:
    event: str
    severity: str
    source: str
    model_type: str
    device: str
    details: Dict[str, Any]
    
RECOVERY_REQUEST_PATH = Path("/tmp/rais_recovery_request.json")

def load_model_event(path: Path) -> ModelEvent | None:
    if not path.exists():
        return None

    try:
        raw = json.loads(path.read_text())
        return ModelEvent(
            event=raw["event"],
            severity=raw["severity"],
            source=raw["source"],
            model_type=raw["model_type"],
            device=raw["device"],
            details=raw.get("details", {}),
        )
    except Exception as e:
        print(f"[warn] failed to load model event from {path}: {e}")
        return None


def clear_model_event(path: Path) -> None:
    if path.exists():
        path.unlink()


def load_active_model_events() -> list[ModelEvent]:
    events = []

    jetson_evt = load_model_event(JETSON_YOLO_EVENT_PATH)
    if jetson_evt is not None:
        events.append(jetson_evt)

    pi_evt = load_model_event(PI_RESNET_EVENT_PATH)
    if pi_evt is not None:
        events.append(pi_evt)

    return events


def clear_active_model_events() -> None:
    clear_model_event(JETSON_YOLO_EVENT_PATH)
    clear_model_event(PI_RESNET_EVENT_PATH)
    
def load_recovery_request(path: Path = RECOVERY_REQUEST_PATH) -> RecoveryRequest | None:
    if not path.exists():
        return None

    try:
        raw = json.loads(path.read_text())
        return RecoveryRequest(
            event=raw["event"],
            severity=raw["severity"],
            source=raw.get("source", "unknown"),
            details=raw.get("details", {}),
        )
    except Exception as e:
        print(f"[warn] failed to load recovery request from {path}: {e}")
        return None


def clear_recovery_request(path: Path = RECOVERY_REQUEST_PATH) -> None:
    if path.exists():
        path.unlink()


def apply_recovery_request(assignments: dict[str, str], req: RecoveryRequest) -> dict[str, str]:
    updated = dict(assignments)

    if req.event == "request_retraining":
        print(f"[recovery] applying retraining request from {req.source}")
        # force the recovery path to exist
        updated.setdefault("adversarial_training", "gpu")
        updated.setdefault("perturbation_generation", "gpu")
        updated.setdefault("metric_computation", "gpu")
        updated.setdefault("rais_scoring", "gpu")
        updated.setdefault("pruning_decision", "gpu")
        updated.setdefault("recovery_finetuning", "gpu_2")
        updated.setdefault("model_evaluation", "gpu")
        updated.setdefault("deploy_updated_model", "pi5a")

        # clear completed stages so pipeline can rerun the training/recovery path
        state = load_run_state()
        to_reopen = {
            "adversarial_training",
            "perturbation_generation",
            "metric_computation",
            "rais_scoring",
            "pruning_decision",
            "recovery_finetuning",
            "model_evaluation",
            "deploy_updated_model",
        }
        state["completed_stages"] = [s for s in state.get("completed_stages", []) if s not in to_reopen]
        save_run_state(state)

    elif req.event == "request_clean_model":
        print(f"[recovery] applying clean-model request from {req.source}")
        updated["deploy_updated_model"] = "pi5a"

        state = load_run_state()
        to_reopen = {"model_evaluation", "deploy_updated_model"}
        state["completed_stages"] = [s for s in state.get("completed_stages", []) if s not in to_reopen]
        save_run_state(state)

    return updated

@dataclass
class Trigger:
    trigger_type: str
    name: str
    severity: str
    details: Dict[str, Any]

def ensure_local_run_root() -> None:
    run_root = Path(RUN_ROOT)
    config_dir = run_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    local_cfg = CONFIG_PATH
    target_cfg = config_dir / "rais_pipeline_prune_config.yaml"

    if local_cfg.exists():
        target_cfg.write_text(local_cfg.read_text())

def load_model_source_path() -> Path:
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    model_path = Path(cfg["model_dir"]).expanduser()

    if not model_path.is_absolute():
        model_path = (Path.cwd() / model_path).resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Configured model_dir not found: {model_path}")

    return model_path

def load_external_trigger(trigger_path: str = "/tmp/rais_trigger.json") -> Trigger | None:
    path = Path(trigger_path)
    if not path.exists():
        return None

    try:
        raw = json.loads(path.read_text())
        return Trigger(
            trigger_type=raw["trigger_type"],
            name=raw["name"],
            severity=raw["severity"],
            details=raw.get("details", {}),
        )
    except Exception as e:
        print(f"[warn] failed to load external trigger: {e}")
        return None


def clear_external_trigger(trigger_path: str = "/tmp/rais_trigger.json") -> None:
    path = Path(trigger_path)
    if path.exists():
        path.unlink()

def ensure_model_available(device: str) -> None:
    if device not in SSH_TARGETS or device not in REMOTE_PROJECT_DIRS:
        raise KeyError(f"Device not configured for model staging: {device}")

    model_src = load_model_source_path()

    host = SSH_TARGETS[device]
    remote_project_dir = REMOTE_PROJECT_DIRS[device]
    remote_run_root = f"{remote_project_dir}/shared_runs/{RUN_ID}"
    remote_model_dir = f"{remote_run_root}/models"
    remote_model_path = f"{remote_model_dir}/best_model.pth"
    remote_config_path = f"{remote_run_root}/config/rais_pipeline_prune_config.yaml"

    subprocess.run(
        ["ssh", host, f"mkdir -p {shlex.quote(remote_model_dir)}"],
        check=True,
    )

    subprocess.run(
        ["rsync", "-av", str(model_src), f"{host}:{remote_model_path}"],
        check=True,
    )

    patch_cmd = (
        f"{PY_BY_HOST[device]} - <<'PY'\n"
        "from pathlib import Path\n"
        "import yaml\n"
        f"cfg_path = Path({remote_config_path!r})\n"
        "cfg = yaml.safe_load(cfg_path.read_text()) or {}\n"
        f"cfg['model_dir'] = {remote_model_path!r}\n"
        "cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))\n"
        "print('[model] updated model_dir ->', cfg['model_dir'])\n"
        "PY"
    )

    subprocess.run(
        ["ssh", host, f"cd {shlex.quote(remote_project_dir)} && {patch_cmd}"],
        check=True,
    )

    print(f"[model] staged {model_src} -> {device}:{remote_model_path}")
    

def probe_device_ready(device: str, timeout_sec: int = 8) -> bool:
    """
    Cheap availability probe.
    Only checks whether we can SSH to the device.
    Does NOT rsync anything.
    """
    if device not in SSH_TARGETS:
        return False

    host = SSH_TARGETS[device]

    try:
        subprocess.run(
            [
                "ssh",
                "-o", "BatchMode=yes",
                "-o", f"ConnectTimeout={timeout_sec}",
                host,
                "echo ok",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError:
        return False
    except Exception:
        return False

def detect_time_based_trigger(now: Optional[datetime.datetime] = None) -> Trigger | None:
    now = now or datetime.datetime.now()
    hour = now.hour

    if hour >= 20 or hour < 6:
        return Trigger(
            trigger_type="time_based",
            name="night_shift",
            severity="medium",
            details={"hour": hour},
        )
    return None


def detect_performance_based_trigger(eval_results: dict | None) -> Trigger | None:
    if not eval_results:
        return None

    clean_acc = eval_results.get("clean", {}).get("acc")
    fgsm_acc = eval_results.get("fgsm", {}).get("acc")
    pgd_acc = eval_results.get("pgd", {}).get("acc")

    if clean_acc is not None and clean_acc < 0.80:
        return Trigger(
            trigger_type="performance_based",
            name="accuracy_drop",
            severity="high",
            details={"metric": "clean.acc", "value": clean_acc, "threshold": 0.80},
        )

    if fgsm_acc is not None and fgsm_acc < 0.75:
        return Trigger(
            trigger_type="performance_based",
            name="gradient_attack_risk",
            severity="high",
            details={"metric": "fgsm.acc", "value": fgsm_acc, "threshold": 0.75},
        )

    if pgd_acc is not None and pgd_acc < 0.75:
        return Trigger(
            trigger_type="performance_based",
            name="gradient_attack_risk",
            severity="high",
            details={"metric": "pgd.acc", "value": pgd_acc, "threshold": 0.75},
        )

    return None


def detect_system_based_trigger(telemetry: dict | None) -> Trigger | None:
    if not telemetry:
        return None

    cpu = telemetry.get("cpu_load_percent")
    ram = telemetry.get("ram_available_gb")
    latency = telemetry.get("latency_ms")

    if cpu is not None and cpu > 90:
        return Trigger(
            trigger_type="system_based",
            name="resource_pressure",
            severity="medium",
            details={"metric": "cpu_load_percent", "value": cpu, "threshold": 90},
        )

    if ram is not None and ram < 2.0:
        return Trigger(
            trigger_type="system_based",
            name="low_memory",
            severity="high",
            details={"metric": "ram_available_gb", "value": ram, "threshold": 2.0},
        )

    if latency is not None and latency > 200:
        return Trigger(
            trigger_type="system_based",
            name="latency_spike",
            severity="medium",
            details={"metric": "latency_ms", "value": latency, "threshold": 200},
        )

    return None


def detect_trigger(
    eval_results: dict | None = None,
    telemetry: dict | None = None,
) -> Trigger | None:
    checks = [
        detect_performance_based_trigger(eval_results),
        detect_system_based_trigger(telemetry),
        detect_time_based_trigger(),
    ]

    priority = {"high": 3, "medium": 2, "low": 1}
    valid = [c for c in checks if c is not None]

    if not valid:
        return None

    return sorted(valid, key=lambda x: priority.get(x.severity, 0), reverse=True)[0]


def edge_self_evaluate_trigger(device: str, telemetry: dict | None = None) -> Trigger | None:
    trigger = detect_trigger(telemetry=telemetry)
    if trigger:
        print(f"[edge-trigger] {device} detected {trigger.name} ({trigger.trigger_type}) | details={trigger.details}")
    return trigger


def prioritize_model_evaluation(assignments: dict[str, str], preferred_device: str = "gpu") -> dict[str, str]:
    updated = dict(assignments)
    if "model_evaluation" not in updated:
        updated["model_evaluation"] = preferred_device
    return updated

def maybe_detect_event(stage: str, device: str):
    detector = EventDetector()

    eval_results = None
    telemetry = None
    rais_metrics = None

    try:
        if stage.startswith("model_evaluation__"):
            raw_eval = load_eval_results_for_substage(stage)
            eval_results = adapt_eval_results(raw_eval)

            if eval_results is not None:
                print(f"[anomaly] loaded eval results for {stage}: {list(eval_results.keys())}")

        event = detector.highest_priority_event(
            eval_results=eval_results,
            telemetry=telemetry,
            rais_metrics=rais_metrics,
        )

        if event:
            print(f"[anomaly] Detected {event.event} at {stage} on {device} | details={event.details}")

        return event

    except Exception as e:
        print(f"[warn] anomaly detection failed for {stage}@{device}: {e}")
        return None

        

def load_eval_results_for_substage(substage_name: str) -> dict | None:
    """
    Load the evaluation JSON for a given model_evaluation substage
    from the local run root after results have been synced back.
    """
    run_root = Path(RUN_ROOT)

    file_map = {
        "model_evaluation__gradient_based":
            run_root / "stage_outputs" / substage_name / "evaluation_fgsm_pgd_cw.json",
        "model_evaluation__localized":
            run_root / "stage_outputs" / substage_name / "evaluation_patch_square.json",
        "model_evaluation__transformation_based":
            run_root / "stage_outputs" / substage_name / "evaluation_spatial_color_shift.json",
        "model_evaluation__query_based":
            run_root / "stage_outputs" / substage_name / "evaluation_boundary.json",
        "model_evaluation__distribution_shift":
            run_root / "stage_outputs" / substage_name / "evaluation_universal.json",
    }

    path = file_map.get(substage_name)
    if path is None:
        print(f"[warn] no eval file mapping for {substage_name}")
        return None

    if not path.exists():
        print(f"[warn] eval file not found for {substage_name}: {path}")
        return None

    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"[warn] failed to load eval results for {substage_name}: {e}")
        return None


def adapt_eval_results(raw: dict | None) -> dict | None:
    """
    Convert your evaluation JSON into the shape expected by EventDetector.
    Input:
      {
        "clean_test": {...},
        "named_attacks": {"fgsm": {...}, "pgd": {...}, ...},
        "boundary_probe": {...}
      }

    Output:
      {
        "clean": {...},
        "fgsm": {...},
        "pgd": {...},
        ...
      }
    """
    if not raw:
        return None

    adapted = {}

    if isinstance(raw.get("clean_test"), dict):
        adapted["clean"] = raw["clean_test"]

    named_attacks = raw.get("named_attacks", {})
    if isinstance(named_attacks, dict):
        for attack_name, vals in named_attacks.items():
            if isinstance(vals, dict):
                adapted[attack_name] = vals

    if isinstance(raw.get("boundary_probe"), dict):
        adapted["boundary_probe"] = raw["boundary_probe"]

    return adapted


def pull_run_root_from_device(device: str) -> None:
    """
    Pull the latest run artifacts from a remote device back to the local RUN_ROOT
    so anomaly detection can read the freshly written evaluation JSON files.
    """
    if device not in SSH_TARGETS or device not in REMOTE_PROJECT_DIRS:
        return

    host = SSH_TARGETS[device]
    remote_run_root = f"{REMOTE_PROJECT_DIRS[device]}/shared_runs/{RUN_ID}"

    subprocess.run(
        ["rsync", "-av", f"{host}:{remote_run_root}/", f"{RUN_ROOT}/"],
        check=True,
    )


def load_telemetry_for_device(device: str) -> dict | None:
    """
    Minimal placeholder for now.
    Later this can pull current live telemetry if you want device-driven anomalies too.
    """
    return None


def load_rais_metrics_for_stage(stage: str) -> dict | None:
    """
    Placeholder until RAIS metrics are written to a standard location.
    """
    return None


  

def pick_device_for_family(
    family_name: str,
    unavailable_devices: set[str],
) -> str | None:
    candidates = EVAL_FAMILY_CANDIDATES.get(family_name, [])

    for device in candidates:
        if device in unavailable_devices:
            continue
        if device not in SSH_TARGETS or device not in PY_BY_HOST or device not in REMOTE_PROJECT_DIRS:
            continue

        if not probe_device_ready(device):
            print(f"[warn] candidate {device} not reachable for {family_name}")
            unavailable_devices.add(device)
            continue

        return device

    return None
    
def pull_substage_outputs_from_device(device: str, substage_name: str, timeout_sec: int = 300) -> None:
    """
    Pull only the outputs for one completed substage instead of the whole run root.
    """
    if device not in SSH_TARGETS or device not in REMOTE_PROJECT_DIRS:
        return

    host = SSH_TARGETS[device]
    remote_subdir = (
        f"{REMOTE_PROJECT_DIRS[device]}/shared_runs/{RUN_ID}/stage_outputs/{substage_name}/"
    )
    local_subdir = Path(RUN_ROOT) / "stage_outputs" / substage_name
    local_subdir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        ["rsync", "-av", f"{host}:{remote_subdir}", f"{local_subdir.parent}/"],
        check=True,
        timeout=timeout_sec,
    )
    
def run_parallel_model_evaluation(timeout_sec: int = 3600) -> list[str]:
    procs = []
    failed_devices = []
    unavailable_devices = set(load_run_state().get("failed_devices", []))

    def cleanup_remaining_procs(current_procs) -> None:
        for p in current_procs:
            proc = p["proc"]
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                except Exception as e:
                    print(f"[warn] failed to clean up proc for {p['family_name']} on {p['device']}: {e}")

    for family_name, attacks in EVAL_FAMILIES.items():
        if not attacks:
            continue

        device = pick_device_for_family(family_name, unavailable_devices)
        if device is None:
            print(f"[warn] no available device could be found for family {family_name}")
            continue

        host = SSH_TARGETS[device]
        py = PY_BY_HOST[device]
        remote_project_dir = REMOTE_PROJECT_DIRS[device]
        remote_run_root = f"{remote_project_dir}/shared_runs/{RUN_ID}"
        remote_config_path = f"{remote_run_root}/config/rais_pipeline_prune_config.yaml"

        attack_arg = ",".join(attacks)

        remote_cmd = (
            f'cd {remote_project_dir} && '
            f'{py} -u rais_pipeline_with_yaml.py '
            f'--config {shlex.quote(remote_config_path)} '
            f'--output-dir {shlex.quote(remote_run_root)} '
            f'--mode atomic-pipeline '
            f'--pipeline-force-stage '
            f'--pipeline-start-stage model_evaluation '
            f'--pipeline-end-stage model_evaluation '
            f'--eval-attacks {shlex.quote(attack_arg)} '
            f'--stage-suffix {shlex.quote(family_name)}'
        )

        ssh_cmd = ["ssh", "-o", "BatchMode=yes", host, remote_cmd]

        substage_name = f"model_evaluation__{family_name}"
        assigned_time = time.perf_counter()
        start_time = assigned_time

        mark_stage_running(substage_name, device)
        proc = subprocess.Popen(ssh_cmd)

        procs.append({
            "device": device,
            "family_name": family_name,
            "proc": proc,
            "assigned_time": assigned_time,
            "start_time": start_time,
        })

        print(f"[parallel] assigned {family_name} -> {device}")

    for item in procs:
        substage_name = f"model_evaluation__{item['family_name']}"
        device = item["device"]

        try:
            rc = item["proc"].wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            print(f"[warn] timeout waiting for {substage_name} on {device}")
            item["proc"].kill()
            end_time = time.perf_counter()
            run_time_sec = end_time - item["start_time"]

            mark_stage_failed(substage_name, device)
            append_stage_metric(
                stage=substage_name,
                device=device,
                status="failed",
                assigned_time=item["assigned_time"],
                start_time=item["start_time"],
                end_time=end_time,
                queue_time_sec=0.0,
                run_time_sec=run_time_sec,
                relay_time_sec=0.0,
                replan_time_sec=0.0,
            )

            if device not in failed_devices:
                failed_devices.append(device)

            cleanup_remaining_procs(procs)
            return failed_devices

        end_time = time.perf_counter()
        run_time_sec = end_time - item["start_time"]

        if rc != 0:
            print(f"[warn] {substage_name} failed on {device} (rc={rc})")
            mark_stage_failed(substage_name, device)
            append_stage_metric(
                stage=substage_name,
                device=device,
                status="failed",
                assigned_time=item["assigned_time"],
                start_time=item["start_time"],
                end_time=end_time,
                queue_time_sec=0.0,
                run_time_sec=run_time_sec,
                relay_time_sec=0.0,
                replan_time_sec=0.0,
            )

            if device not in failed_devices:
                failed_devices.append(device)

            cleanup_remaining_procs(procs)
            return failed_devices

        mark_stage_completed(substage_name)
        append_stage_metric(
            stage=substage_name,
            device=device,
            status="completed",
            assigned_time=item["assigned_time"],
            start_time=item["start_time"],
            end_time=end_time,
            queue_time_sec=0.0,
            run_time_sec=run_time_sec,
            relay_time_sec=0.0,
            replan_time_sec=0.0,
        )

        try:
            pull_substage_outputs_from_device(device, substage_name)
        except Exception as e:
            print(f"[warn] failed to pull outputs for {substage_name} from {device}: {e}")
        
        event = maybe_detect_event(substage_name, device)
        
        if event is not None and event.severity == "high":
            state = load_run_state()
            handled = set(tuple(x) for x in state.get("handled_events", []))
        
            event_key = (event.event, device)
        
            if event_key in handled:
                print(f"[anomaly] event {event.event} already handled for {device}, skipping")
            else:
                print(f"[replan] triggered by event {event.event} at {substage_name}@{device}")
        
                handled.add(event_key)
                state["handled_events"] = list(handled)
                save_run_state(state)
        
                cleanup_remaining_procs(procs)
        
                raise RuntimeError(
                    f"EVENT_TRIGGER::{event.event}::{device}::{substage_name}"
                )

    return failed_devices

def init_stage_metrics_csv() -> None:
    STAGE_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with STAGE_METRICS_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "stage",
            "device_label",
            "status",
            "assigned_time",
            "start_time",
            "end_time",
            "queue_time_sec",
            "run_time_sec",
            "relay_time_sec",
            "replan_time_sec",
        ])
        
def append_stage_metric(
    stage: str,
    device: str,
    status: str,
    assigned_time: float | None,
    start_time: float | None,
    end_time: float | None,
    queue_time_sec: float,
    run_time_sec: float,
    relay_time_sec: float = 0.0,
    replan_time_sec: float = 0.0,
) -> None:
    with STAGE_METRICS_PATH.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            stage,
            device,
            status,
            assigned_time,
            start_time,
            end_time,
            queue_time_sec,
            run_time_sec,
            relay_time_sec,
            replan_time_sec,
        ])

def write_workflow_metrics(orchestrator_start: float, orchestrator_end: float) -> None:
    import pandas as pd

    if not STAGE_METRICS_PATH.exists():
        print(f"[warn] Stage metrics file not found: {STAGE_METRICS_PATH}")
        return

    try:
        df = pd.read_csv(STAGE_METRICS_PATH)
    except pd.errors.ParserError as e:
        print(f"[warn] Failed to parse {STAGE_METRICS_PATH}: {e}")
        return
    except Exception as e:
        print(f"[warn] Failed to read {STAGE_METRICS_PATH}: {e}")
        return

    if df.empty:
        print(f"[warn] Stage metrics file is empty: {STAGE_METRICS_PATH}")
        return

    required_cols = {
        "status",
        "run_time_sec",
        "queue_time_sec",
        "relay_time_sec",
        "replan_time_sec",
    }

    missing = required_cols - set(df.columns)
    if missing:
        print(
            f"[warn] Stage metrics file has wrong schema: {STAGE_METRICS_PATH}. "
            f"Missing columns: {sorted(missing)}. "
            f"Found columns: {list(df.columns)}"
        )
        return

    makespan_sec = max(0.0, orchestrator_end - orchestrator_start)
    completed_df = df[df["status"] == "completed"]

    completed_stages = int(len(completed_df))
    total_stage_runtime_sec = float(pd.to_numeric(completed_df["run_time_sec"], errors="coerce").fillna(0).sum())
    total_queue_time_sec = float(pd.to_numeric(df["queue_time_sec"], errors="coerce").fillna(0).sum())
    avg_queue_time_sec = (
        float(pd.to_numeric(df["queue_time_sec"], errors="coerce").fillna(0).mean())
        if len(df) else 0.0
    )
    total_relay_time_sec = float(pd.to_numeric(df["relay_time_sec"], errors="coerce").fillna(0).sum())
    total_replan_time_sec = float(pd.to_numeric(df["replan_time_sec"], errors="coerce").fillna(0).sum())

    throughput_stages_per_sec = completed_stages / makespan_sec if makespan_sec > 0 else 0.0
    sequential_utilization = total_stage_runtime_sec / makespan_sec if makespan_sec > 0 else 0.0

    orchestration_overhead_sec = max(
        0.0,
        makespan_sec - total_stage_runtime_sec - total_queue_time_sec
    )

    metrics = {
        "makespan_sec": makespan_sec,
        "makespan_hr": makespan_sec / 3600.0,
        "completed_stages": completed_stages,
        "throughput_stages_per_sec": throughput_stages_per_sec,
        "throughput_stages_per_hr": throughput_stages_per_sec * 3600.0,
        "total_stage_runtime_sec": total_stage_runtime_sec,
        "total_queue_time_sec": total_queue_time_sec,
        "avg_queue_time_sec": avg_queue_time_sec,
        "total_relay_time_sec": total_relay_time_sec,
        "total_replan_time_sec": total_replan_time_sec,
        "sequential_utilization": sequential_utilization,
        "orchestration_overhead_sec": orchestration_overhead_sec,
    }

    WORKFLOW_METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"[metrics] wrote {WORKFLOW_METRICS_PATH}")
    
# --------------------------
# Run state helpers
# --------------------------
def load_run_state() -> dict:
    if RUN_STATE_PATH.exists():
        return json.loads(RUN_STATE_PATH.read_text())

    state = {
        "completed_stages": [],
        "failed_devices": [],
        "queue": [],
        "replan_count": 0,
        "handled_events": [],
    }
    save_run_state(state)
    return state


def save_run_state(state: dict) -> None:
    RUN_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUN_STATE_PATH.write_text(json.dumps(state, indent=2))


def mark_stage_running(stage: str, device: str) -> None:
    state = load_run_state()
    if state.get("replan_count", 0) >= MAX_REPLANS:
        raise RuntimeError(f"Exceeded maximum replans ({MAX_REPLANS}). Aborting safely.")
    for item in state["queue"]:
        if item["stage"] == stage:
            item["device"] = device
            item["status"] = "running"
            item["start_time"] = time.time()
            save_run_state(state)
            return

    state["queue"].append({
        "stage": stage,
        "device": device,
        "status": "running",
        "start_time": time.time(),
    })
    save_run_state(state)


def mark_stage_completed(stage: str) -> None:
    state = load_run_state()
    if state.get("replan_count", 0) >= MAX_REPLANS:
        raise RuntimeError(f"Exceeded maximum replans ({MAX_REPLANS}). Aborting safely.")

    if stage in STAGE_ORDER and stage not in state["completed_stages"]:
        state["completed_stages"].append(stage)

    for item in state["queue"]:
        if item["stage"] == stage:
            item["status"] = "completed"
            item["end_time"] = time.time()

    save_run_state(state)

def mark_stage_failed(stage: str, device: str) -> None:
    state = load_run_state()
    if state.get("replan_count", 0) >= MAX_REPLANS:
        raise RuntimeError(f"Exceeded maximum replans ({MAX_REPLANS}). Aborting safely.")

    for item in state["queue"]:
        if item["stage"] == stage:
            item["device"] = device
            item["status"] = "failed"
            item["failed_time"] = time.time()
            save_run_state(state)
            return

    state["queue"].append({
        "stage": stage,
        "device": device,
        "status": "failed",
        "failed_time": time.time(),
    })
    save_run_state(state)

def mark_device_failed(device: str) -> None:
    state = load_run_state()
    if device not in state["failed_devices"]:
        state["failed_devices"].append(device)
    save_run_state(state)

def increment_replan_count() -> None:
    state = load_run_state()
    if state.get("replan_count", 0) >= MAX_REPLANS:
        raise RuntimeError(f"Exceeded maximum replans ({MAX_REPLANS}). Aborting safely.")
    state["replan_count"] += 1
    save_run_state(state)

def remaining_stages(stage_order: list[str]) -> list[str]:
    state = load_run_state()
    completed = set(state["completed_stages"])
    return [s for s in stage_order if s not in completed]

def mark_model_vulnerable(status: bool):
    state = load_run_state()
    state["model_vulnerable"] = status
    save_run_state(state)

def trigger_replan(event, device: str | None = None) -> dict[str, str]:
    print(f"[replan] triggered by event: {event.event} (Severity: {event.severity})")

    resource_critical = {
        "device_unreachable",
        "hard_failure",
        "low_memory",
        "resource_pressure",
    }

    security_critical = {
        "accuracy_drop",
        "robustness_drop",
        "model_degradation",
        "gradient_attack_risk",
    }
    if event.event in resource_critical and device is not None:
        print(f"[replan] Action: Isolating {device}")
        mark_device_failed(device)

    if event.event in security_critical:
        print(f"[replan] Action: Flagging model as VULNERABLE")
        mark_model_vulnerable(True)

    increment_replan_count()

    rebuild_problem_with_failures()

    ok = rerun_planner()
    if not ok:
        raise RuntimeError(
            f"Replanning failed after event '{event.event}': no valid plan could be generated."
        )

    _, new_assignments = parse_plan(PLAN_PATH)
    return new_assignments

# --------------------------
# Plan parsing
# --------------------------
def normalize_stage_name(name: str) -> str:
    return name.replace("-", "_")


def normalize_run_stage_name(name: str) -> str:
    """
    run-action names may include suffixes like:
      run-metric-computation-gpu
      run-adversarial-training-workstation
    We map them back to canonical pipeline stages.
    """
    norm = normalize_stage_name(name)

    suffixes = [
        "_workstation",
        "_edge",
        "_gpu",
        "_cpu",
    ]
    for suf in suffixes:
        if norm.endswith(suf):
            return norm[:-len(suf)]
    return norm


def parse_plan(plan_file: Path) -> tuple[list[tuple[str, str, str]], dict[str, str]]:
    actions: list[tuple[str, str, str]] = []
    assignments: dict[str, str] = {}

    for raw_line in plan_file.read_text().splitlines():
        line = raw_line.strip().lower()
        if not line or line.startswith(";") or line.startswith("["):
            continue

        m = ASSIGN_RE.match(line)
        if m:
            stage = normalize_stage_name(m.group(1))
            device = m.group(2)
            actions.append(("assign", stage, device))
            assignments[stage] = device
            continue

        m = RUN_RE.match(line)
        if m:
            stage = normalize_run_stage_name(m.group(1))
            device = m.group(2)
            actions.append(("run", stage, device))
            continue

    return actions, assignments


# --------------------------
# Command builders
# --------------------------
def build_stage_command(stage: str, device: str) -> str:
    remote_project_dir = REMOTE_PROJECT_DIRS[device]
    remote_run_root = f"{remote_project_dir}/shared_runs/{RUN_ID}"
    remote_config_path = f"{remote_run_root}/config/rais_pipeline_prune_config.yaml"

    base_cmd = (
        f"rais_pipeline_with_yaml.py "
        f"--config {shlex.quote(remote_config_path)} "
        f"--output-dir {shlex.quote(remote_run_root)} "
        f"--mode atomic-pipeline "
        f"--pipeline-force-stage "
    )
    return base_cmd + f"--pipeline-start-stage {stage} --pipeline-end-stage {stage}"


def stage_output_subdir(stage: str) -> str:
    return f"stage_outputs/{stage}"


# --------------------------
# Relay
# --------------------------
def relay_stage_outputs(src_device: str, stage: str, dst_device: str) -> float:
    if src_device == dst_device:
        return 0.0

    if src_device not in SSH_TARGETS:
        raise KeyError(f"No SSH target configured for source device '{src_device}'")
    if dst_device not in SSH_TARGETS:
        raise KeyError(f"No SSH target configured for destination device '{dst_device}'")

    t0 = time.perf_counter()

    src_host = SSH_TARGETS[src_device]
    dst_host = SSH_TARGETS[dst_device]

    src_remote_root = f"{REMOTE_PROJECT_DIRS[src_device]}/shared_runs/{RUN_ID}"
    dst_remote_root = f"{REMOTE_PROJECT_DIRS[dst_device]}/shared_runs/{RUN_ID}"

    print(f"[relay] pull {stage} outputs from {src_device} -> control node")
    subprocess.run(
        ["rsync", "-av", f"{src_host}:{src_remote_root}/", f"{RUN_ROOT}/"],
        check=True,
    )

    print(f"[relay] push run root -> {dst_device}")
    subprocess.run(
        ["ssh", dst_host, f"mkdir -p {shlex.quote(dst_remote_root)}"],
        check=True,
    )
    subprocess.run(
        ["rsync", "-av", f"{RUN_ROOT}/", f"{dst_host}:{dst_remote_root}/"],
        check=True,
    )

    return time.perf_counter() - t0

# --------------------------
# Replanning helpers
# --------------------------
def rebuild_problem_with_failures() -> None:
    state = load_run_state()
    if state.get("replan_count", 0) >= MAX_REPLANS:
        raise RuntimeError(f"Exceeded maximum replans ({MAX_REPLANS}). Aborting safely.")
    failed_devices = ",".join(state["failed_devices"])

    env = os.environ.copy()
    env["FAILED_DEVICES"] = failed_devices

    print(f"[replan] FAILED_DEVICES={failed_devices}")
    subprocess.run(["python", "build_problem.py"], check=True, env=env)


def rerun_planner() -> bool:
    state = load_run_state()
    failed_devices = state.get("failed_devices", [])

    try:
        print(f"[replan] Attempting to generate new plan with FAILED_DEVICES={failed_devices}...")
        result = subprocess.run(
            PLANNER_CMD,
            check=True,
            capture_output=True,
            text=True,
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        print("[replan] New plan generated successfully.")
        return True

    except subprocess.CalledProcessError as e:
        print("\n" + "!" * 60)
        print("CRITICAL: PLANNER FAILED TO FIND A SOLUTION!")
        print("Reason: The remaining devices cannot satisfy the workflow requirements.")
        print(f"Current Isolation List: {failed_devices}")
        if e.stdout:
            print("\n[planner stdout]")
            print(e.stdout)
        if e.stderr:
            print("\n[planner stderr]")
            print(e.stderr)
        print("!" * 60 + "\n")
        return False


def filter_assignments_to_remaining(assignments: dict[str, str]) -> dict[str, str]:
    rem = set(remaining_stages(STAGE_ORDER))
    return {stage: dev for stage, dev in assignments.items() if stage in rem}


# --------------------------
# Reporting
# --------------------------
def print_assignment_table(assignments: dict[str, str]) -> None:
    print("\n[stage assignments]")
    for stage in STAGE_ORDER:
        dev = assignments.get(stage, "<unassigned>")
        print(f"  {stage:24s} -> {dev}")


def dry_run_commands(assignments: dict[str, str]) -> None:
    print("\n[dry-run commands]")
    assignments = filter_assignments_to_remaining(assignments)

    for stage in STAGE_ORDER:
        if stage not in assignments:
            print(f"  [skip] {stage}: no assignment")
            continue

        device = assignments[stage]

        if device not in SSH_TARGETS:
            print(f"  [skip] {stage}: no SSH target configured for {device}")
            continue
        if device not in PY_BY_HOST:
            print(f"  [skip] {stage}: no python configured for {device}")
            continue
        if device not in REMOTE_PROJECT_DIRS:
            print(f"  [skip] {stage}: no project dir configured for {device}")
            continue

        host = SSH_TARGETS[device]
        py = PY_BY_HOST[device]
        remote_project_dir = REMOTE_PROJECT_DIRS[device]
        stage_cmd = build_stage_command(stage, device)
        remote_cmd = f'cd {remote_project_dir} && {py} -u {stage_cmd}'
        ssh_cmd = f'ssh -o BatchMode=yes {host} "{remote_cmd}"'

        print(f"\n  [{stage} -> {device}]\n  {ssh_cmd}")


# --------------------------
# Execution
# --------------------------
def ensure_run_root_on_device(device: str, timeout_sec: int = 300) -> None:
    """
    Stage only the minimal run structure needed for remote execution.
    Avoid copying bulky outputs/checkpoints/logs every time.
    """
    host = SSH_TARGETS[device]
    remote_run_root = f"{REMOTE_PROJECT_DIRS[device]}/shared_runs/{RUN_ID}"

    subprocess.run(
        ["ssh", host, f"mkdir -p {shlex.quote(remote_run_root)}/config"],
        check=True,
        timeout=timeout_sec,
    )

    # Copy only config directory by default
    local_config_dir = Path(RUN_ROOT) / "config"
    if local_config_dir.exists():
        subprocess.run(
            [
                "rsync",
                "-av",
                f"{local_config_dir}/",
                f"{host}:{remote_run_root}/config/",
            ],
            check=True,
            timeout=timeout_sec,
        )
    
def execute_plan(assignments: dict[str, str]) -> None:
    assignments = filter_assignments_to_remaining(assignments)

    prev_stage = None
    prev_device = None

    for stage in STAGE_ORDER:
        if stage not in assignments:
            continue

        assigned_time = time.perf_counter()
        device = assignments[stage]

        # --------------------------------------------------
        # Validate target device
        # --------------------------------------------------
        if (
            device not in SSH_TARGETS
            or device not in PY_BY_HOST
            or device not in REMOTE_PROJECT_DIRS
        ):
            print(f"[warn] device '{device}' is not executable by orchestrator")
            mark_device_failed(device)
            increment_replan_count()

            replan_t0 = time.perf_counter()
            rebuild_problem_with_failures()
            ok = rerun_planner()
            replan_time_sec = time.perf_counter() - replan_t0

            append_stage_metric(
                stage=stage,
                device=device,
                status="failed",
                assigned_time=assigned_time,
                start_time=None,
                end_time=time.perf_counter(),
                queue_time_sec=0.0,
                run_time_sec=0.0,
                relay_time_sec=0.0,
                replan_time_sec=replan_time_sec,
            )

            if not ok:
                raise RuntimeError("Replanning failed: no valid plan could be generated after device failure.")
            
            print("[replan] reparsing plan")
            _, new_assignments = parse_plan(PLAN_PATH)
            return execute_plan(new_assignments)

        # --------------------------------------------------
        # Ensure run root/config exists on target device
        # IMPORTANT: do this only once here, not again later
        # --------------------------------------------------
        if prev_device is None or prev_device != device:
            ensure_run_root_on_device(device)

        # --------------------------------------------------
        # Ensure model exists before model-dependent stage
        # --------------------------------------------------
        if stage == "perturbation_generation":
            ensure_model_available(device)

        # --------------------------------------------------
        # Relay outputs if stage moves across devices
        # --------------------------------------------------
        relay_time_sec = 0.0
        if prev_stage is not None and prev_device is not None and prev_device != device:
            relay_time_sec = relay_stage_outputs(prev_device, prev_stage, device)

        # --------------------------------------------------
        # Special-case: parallelize model_evaluation
        # --------------------------------------------------
        if stage == "model_evaluation":
            start_time = time.perf_counter()
            queue_time_sec = start_time - assigned_time

            print(f"[launch] {stage} -> parallel family branches")
            mark_stage_running(stage, "parallel")

            try:
                failed_parallel_devices = run_parallel_model_evaluation(timeout_sec=1800)
            except RuntimeError as e:
                msg = str(e)

                if msg.startswith("EVENT_TRIGGER::"):
                    _, event_name, event_device, substage_name = msg.split("::", 3)

                    print(f"[replan] handling event-triggered replan for {event_name} at {substage_name}@{event_device}")

                    mark_stage_failed(stage, "parallel")
                    increment_replan_count()

                    replan_t0 = time.perf_counter()
                    rebuild_problem_with_failures()
                    ok = rerun_planner()
                    replan_time_sec = time.perf_counter() - replan_t0

                    append_stage_metric(
                        stage=stage,
                        device="parallel_families",
                        status="failed",
                        assigned_time=assigned_time,
                        start_time=start_time,
                        end_time=time.perf_counter(),
                        queue_time_sec=queue_time_sec,
                        run_time_sec=time.perf_counter() - start_time,
                        relay_time_sec=0.0,
                        replan_time_sec=replan_time_sec,
                    )

                    if not ok:
                        raise RuntimeError("Replanning failed after event-triggered anomaly.")

                    print("[replan] reparsing plan")
                    _, new_assignments = parse_plan(PLAN_PATH)
                    return execute_plan(new_assignments)

                raise

            end_time = time.perf_counter()
            run_time_sec = end_time - start_time

            if failed_parallel_devices:
                print(f"[warn] stage '{stage}' parallel execution failed on devices: {failed_parallel_devices}")

                mark_stage_failed(stage, "parallel")
                for failed_device in failed_parallel_devices:
                    mark_device_failed(failed_device)

                increment_replan_count()

                replan_t0 = time.perf_counter()
                rebuild_problem_with_failures()
                ok = rerun_planner()
                replan_time_sec = time.perf_counter() - replan_t0

                append_stage_metric(
                    stage=stage,
                    device="parallel_families",
                    status="failed",
                    assigned_time=assigned_time,
                    start_time=start_time,
                    end_time=end_time,
                    queue_time_sec=queue_time_sec,
                    run_time_sec=run_time_sec,
                    relay_time_sec=0.0,
                    replan_time_sec=replan_time_sec,
                )

                if not ok:
                    raise RuntimeError("Replanning failed after parallel model_evaluation failure.")

                print("[replan] reparsing plan")
                _, new_assignments = parse_plan(PLAN_PATH)
                return execute_plan(new_assignments)

            mark_stage_completed(stage)

            append_stage_metric(
                stage=stage,
                device="parallel_families",
                status="completed",
                assigned_time=assigned_time,
                start_time=start_time,
                end_time=end_time,
                queue_time_sec=queue_time_sec,
                run_time_sec=run_time_sec,
                relay_time_sec=0.0,
                replan_time_sec=0.0,
            )

            prev_stage = None
            prev_device = None
            continue
            
        # --------------------------------------------------
        # Normal single-stage execution
        # --------------------------------------------------
        host = SSH_TARGETS[device]
        py = PY_BY_HOST[device]
        remote_project_dir = REMOTE_PROJECT_DIRS[device]

        stage_cmd = build_stage_command(stage, device)
        remote_cmd = f'cd {remote_project_dir} && {py} -u {stage_cmd}'
        ssh_cmd = f'ssh -o BatchMode=yes {host} "{remote_cmd}"'

        start_time = time.perf_counter()
        queue_time_sec = start_time - assigned_time

        print(f"[launch] {stage} -> {device}")
        mark_stage_running(stage, device)

        rc = subprocess.call(ssh_cmd, shell=True)

        end_time = time.perf_counter()
        run_time_sec = end_time - start_time

        if rc != 0:
            print(f"[warn] stage '{stage}' failed on '{device}' (rc={rc})")

            mark_stage_failed(stage, device)
            mark_device_failed(device)
            increment_replan_count()

            replan_t0 = time.perf_counter()
            rebuild_problem_with_failures()
            ok = rerun_planner()
            replan_time_sec = time.perf_counter() - replan_t0

            append_stage_metric(
                stage=stage,
                device=device,
                status="failed",
                assigned_time=assigned_time,
                start_time=start_time,
                end_time=end_time,
                queue_time_sec=queue_time_sec,
                run_time_sec=run_time_sec,
                relay_time_sec=relay_time_sec,
                replan_time_sec=replan_time_sec,
            )

            if not ok:
                raise RuntimeError("Replanning failed: no valid plan could be generated after device failure.")
            
            print("[replan] reparsing plan")
            _, new_assignments = parse_plan(PLAN_PATH)
            return execute_plan(new_assignments)
            
        mark_stage_completed(stage)

        event = maybe_detect_event(stage, device)
        if event is not None and event.severity == "high":
            print(f"[replan] triggered by event {event.event} at {stage}@{device}")
            new_assignments = trigger_replan(event, device=device)
            return execute_plan(new_assignments)

        append_stage_metric(
            stage=stage,
            device=device,
            status="completed",
            assigned_time=assigned_time,
            start_time=start_time,
            end_time=end_time,
            queue_time_sec=queue_time_sec,
            run_time_sec=run_time_sec,
            relay_time_sec=relay_time_sec,
            replan_time_sec=0.0,
        )

        prev_stage = stage
        prev_device = device
        

def reset_run_state() -> None:
    state = {
        "completed_stages": [],
        "failed_devices": [],
        "queue": [],
        "replan_count": 0,
        "handled_events": [],
    }
    save_run_state(state)
    

def adapt_telemetry(raw):
    adapted = {}

    for name, t in raw.items():

        # Base values
        cpu = float(t.get("cpu_load_percent", 100.0))
        ram = float(t.get("ram_available_gb", 0.0))
        gpu = t.get("gpu_util_percent")

        # Inject faults if configured
        if name in FAULT_INJECTION:
            overrides = FAULT_INJECTION[name]

            cpu = overrides.get("cpu_load_percent", cpu)
            ram = overrides.get("ram_available_gb", ram)
            gpu = overrides.get("gpu_util_percent", gpu)

            print(f"[debug] Injecting telemetry fault for {name}: {overrides}")

        adapted[name] = {
            "available": True,
            "cpu_load": cpu / 100.0,
            "gpu_util": (gpu / 100.0) if gpu is not None else 0.0,
            "free_ram_gb": ram,
            "latency_ms": 50.0,
            "reliability": 0.95,
        }

    return adapted
    

def debug_stage_feasibility(normalized, stages, capabilities, thresholds):
    for stage in stages:
        allowed = capabilities.get(stage, [])
        feasible = []
        for host in allowed:
            if host not in normalized: continue
            t = normalized[host]
            
            # Identify workstations
            is_workstation = any(kw in host.lower() for kw in ["gpu", "ncps"])
            
            # If it's a workstation, let it pass even if RAM looks like 0 
            # (until the probe is fixed)
            if is_workstation:
                ok = t["available"] 
            else:
                # Strict check for Pis/Jetsons
                ok = (t["available"] and 
                      t["free_ram_gb"] >= thresholds.get(stage, {}).get("min_free_ram_gb", 0))
            
            if ok: feasible.append(host)

        print(f"  {stage}: {feasible}")


def load_live_event(event_path: str = "/tmp/rais_live_event.json") -> dict | None:
    path = Path(event_path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"[warn] failed to load live event: {e}")
        return None


def clear_live_event(event_path: str = "/tmp/rais_live_event.json") -> None:
    path = Path(event_path)
    if path.exists():
        path.unlink()


# --------------------------
# Main
# --------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Parse a plan and map stages to devices.")
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-discovery-check", action="store_true")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--fault", type=str, default=None)
    parser.add_argument(
        "--planner-backend",
        choices=["pddl", "up", "telemetry_opt"],
        default="pddl",
    )
    parser.add_argument(
    "--up-engine",
    type=str,
    default="tamer",
    choices=["tamer", "enhsp", "enhsp-opt"],
)
    parser.add_argument("--fresh-run", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    global CONFIG_PATH
    
    if args.config is not None:
        CONFIG_PATH = Path(args.config)
    else:
        CONFIG_PATH = Path("rais_pipeline_prune_config.yaml")
    
    global RUN_ID
    if args.run_id is not None:
        RUN_ID = args.run_id
    
    global RUN_ROOT, STAGE_METRICS_PATH, WORKFLOW_METRICS_PATH, RUN_STATE_PATH
    RUN_ROOT = f"{SHARED_ROOT}/{RUN_ID}"
    STAGE_METRICS_PATH = Path(RUN_ROOT) / "orchestrator_stage_runtime_metrics.csv"
    WORKFLOW_METRICS_PATH = Path(RUN_ROOT) / "workflow_metrics.json"
    RUN_STATE_PATH = Path(RUN_ROOT) / "orchestrator_state.json"
    
    global FAULT_INJECTION
    if args.fault:
        FAULT_INJECTION = json.loads(args.fault)
    else:
        FAULT_INJECTION = {}
    
    print(f"[run] RUN_ID={RUN_ID}")
    print(f"[run] FAULT_INJECTION={FAULT_INJECTION}")
    
    if args.fresh_run or not STAGE_METRICS_PATH.exists():
        init_stage_metrics_csv()
    
    if args.fresh_run:
        reset_run_state()

    orchestrator_start = time.perf_counter()

    state = load_run_state()
    failed_from_state = state.get("failed_devices", [])
    
    FAILED_DEVICES = failed_from_state

    discovered_online = None
    if not args.skip_discovery_check:
        discovered_online = discover_online_devices()
        labels = sorted(d["label"] for d in discovered_online)
        print(f"[discover] online devices: {labels}")

    assignments = {}

    if args.planner_backend == "pddl":
        if not args.plan.exists():
            raise FileNotFoundError(f"Plan file not found: {args.plan}")

        actions, assignments = parse_plan(args.plan)

        if not actions:
            raise RuntimeError("No actions parsed from plan file.")

        print("\n[plan actions]")
        for kind, stage, device in actions:
            print(f"  {kind:6s} stage={stage:24s} device={device}")

    else:
        online_devices = discovered_online if discovered_online is not None else discover_online_devices()
        trusted = executable_device_labels()

        devices = [
            d for d in online_devices
            if d["label"].lower() in trusted
        ]

        devices = [
            d for d in devices
            if not d.get("probe_failed", False)
        ]

        if not devices:
            raise RuntimeError("No trusted online devices found for planning.")

        devices = enrich_with_telemetry(devices)

        raw_telemetry = {d["label"]: d for d in devices}
        
        completed = load_run_state().get("completed_stages", [])

        normalized = adapt_telemetry(raw_telemetry)

        print(f"[up] completed stages from run state: {completed}")
        print("[up] normalized telemetry:")
        for host, vals in normalized.items():
            print(f"  {host}: {vals}")

        remaining = [s for s in STAGES if s not in completed]
        print(f"[up] remaining stages: {remaining}")

        if not remaining:
            print("[up] no assignments needed: all stages already completed.")
            orchestrator_end = time.perf_counter()
            write_workflow_metrics(orchestrator_start, orchestrator_end)
            print(f"\n[pddl_orchestrator] done in {orchestrator_end - orchestrator_start:.2f}s")
            return

        debug_stage_feasibility(normalized, STAGES, CAPABILITIES, THRESHOLDS)

        if args.planner_backend == "up":
            problem, _, _ = build_planning_problem(
                telemetry=normalized,
                stages=STAGES,
                capabilities=CAPABILITIES,
                thresholds=THRESHOLDS,
                completed_stages=completed,
            )

            result, assignments = plan_assignments(problem, engine_name=args.up_engine)

            if result.plan is None:
                raise RuntimeError(f"Unified Planning failed. Solver status: {result.status}")

            if not assignments:
                raise RuntimeError(
                    f"Unified Planning returned an empty plan or no assignment actions. "
                    f"Solver status: {result.status}. Remaining stages: {remaining}"
                )

            print("\n[up assignments]")
            for stage, device in assignments.items():
                print(f"  stage={stage:24s} device={device}")

        else:  # telemetry_opt
            assignments, chosen_costs = get_exact_telemetry_optimal_assignments(
                telemetry=normalized,
                stages=STAGES,
                capabilities=CAPABILITIES,
                thresholds=THRESHOLDS,
                completed_stages=completed,
            )

            print("\n[telemetry-opt assignments]")
            for stage, device in assignments.items():
                print(
                    f"  stage={stage:24s} device={device:8s} cost={chosen_costs[stage]}"
                )

    edge_trigger = edge_self_evaluate_trigger(
        device="jetson",
        telemetry=FAULT_INJECTION.get("jetson")
    )

    if edge_trigger is not None:
        print(f"[trigger] promoting model_evaluation due to {edge_trigger.name}")
        assignments = prioritize_model_evaluation(assignments, preferred_device="gpu")

    external_trigger = load_external_trigger()

    if external_trigger is not None:
        print(
            f"[trigger] external trigger detected: "
            f"{external_trigger.name} ({external_trigger.trigger_type}) | "
            f"details={external_trigger.details}"
        )
        assignments = prioritize_model_evaluation(assignments, preferred_device="gpu")
        clear_external_trigger()

    live_event = load_live_event()
    if live_event is not None:
        print(
            f"[trigger] live event detected: "
            f"{live_event['event']} | details={live_event.get('details', {})}"
        )
        assignments = prioritize_model_evaluation(assignments, preferred_device="gpu")
        clear_live_event()

    recovery_req = load_recovery_request()
    if recovery_req is not None:
        print(
            f"[recovery] external request detected: "
            f"{recovery_req.event} from {recovery_req.source} | "
            f"details={recovery_req.details}"
        )
        assignments = apply_recovery_request(assignments, recovery_req)
        clear_recovery_request()

    print_assignment_table(assignments)

    if args.dry_run:
        dry_run_commands(assignments)
    else:
        ensure_local_run_root()
        execute_plan(assignments)

    orchestrator_end = time.perf_counter()
    write_workflow_metrics(orchestrator_start, orchestrator_end)

    print(f"\n[pddl_orchestrator] done in {orchestrator_end - orchestrator_start:.2f}s")

    
if __name__ == "__main__":
    main()