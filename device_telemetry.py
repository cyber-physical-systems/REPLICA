#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Optional dependencies
try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
except ImportError:
    torch = None

try:
    import pynvml  # pip install nvidia-ml-py
except ImportError:
    pynvml = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def safe_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def get_hostname() -> str:
    return platform.node()


def get_cpu_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "cpu_cores": None,
        "cpu_logical": None,
        "cpu_load_percent": None,
        "cpu_freq_mhz": None,
        "cpu_per_core_percent": None,
    }
    if psutil is None:
        return info

    try:
        info["cpu_cores"] = psutil.cpu_count(logical=False)
        info["cpu_logical"] = psutil.cpu_count(logical=True)
    except Exception:
        pass

    try:
        # Small interval gives a meaningful instantaneous sample
        info["cpu_load_percent"] = psutil.cpu_percent(interval=0.2)
    except Exception:
        pass

    try:
        freq = psutil.cpu_freq()
        if freq is not None:
            info["cpu_freq_mhz"] = freq.current
    except Exception:
        pass

    try:
        info["cpu_per_core_percent"] = psutil.cpu_percent(interval=0.2, percpu=True)
    except Exception:
        pass

    return info


def get_memory_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "ram_total_gb": None,
        "ram_available_gb": None,
        "ram_used_gb": None,
        "ram_used_percent": None,
        "swap_total_gb": None,
        "swap_used_gb": None,
        "swap_used_percent": None,
    }
    if psutil is None:
        return info

    try:
        vm = psutil.virtual_memory()
        info["ram_total_gb"] = vm.total / (1024 ** 3)
        info["ram_available_gb"] = vm.available / (1024 ** 3)
        info["ram_used_gb"] = vm.used / (1024 ** 3)
        info["ram_used_percent"] = vm.percent
    except Exception:
        pass

    try:
        sm = psutil.swap_memory()
        info["swap_total_gb"] = sm.total / (1024 ** 3)
        info["swap_used_gb"] = sm.used / (1024 ** 3)
        info["swap_used_percent"] = sm.percent
    except Exception:
        pass

    return info


def get_battery_info() -> Dict[str, Any]:
    info = {
        "battery_present": None,
        "battery_percent": None,
        "battery_secs_left": None,
        "battery_power_plugged": None,
    }
    if psutil is None or not hasattr(psutil, "sensors_battery"):
        return info

    try:
        batt = psutil.sensors_battery()
        if batt is None:
            info["battery_present"] = False
        else:
            info["battery_present"] = True
            info["battery_percent"] = batt.percent
            info["battery_secs_left"] = batt.secsleft
            info["battery_power_plugged"] = batt.power_plugged
    except Exception:
        pass
    return info


def get_loadavg_info() -> Dict[str, Any]:
    info = {"loadavg_1m": None, "loadavg_5m": None, "loadavg_15m": None}
    try:
        a1, a5, a15 = os.getloadavg()  # noqa: F821
        info["loadavg_1m"] = a1
        info["loadavg_5m"] = a5
        info["loadavg_15m"] = a15
    except Exception:
        pass
    return info


def get_cuda_info_torch() -> Dict[str, Any]:
    info = {
        "has_cuda": False,
        "gpu_name": None,
        "gpu_mem_total_gb": None,
    }
    if torch is None:
        return info

    try:
        has = torch.cuda.is_available()
    except Exception:
        has = False

    info["has_cuda"] = bool(has)
    if not has:
        return info

    try:
        prop = torch.cuda.get_device_properties(0)
        info["gpu_name"] = getattr(prop, "name", None)
        info["gpu_mem_total_gb"] = getattr(prop, "total_memory", 0) / (1024 ** 3)
    except Exception:
        pass

    return info


def get_gpu_info_nvml() -> Dict[str, Any]:
    info = {
        "gpu_util_percent": None,
        "gpu_mem_used_gb": None,
        "gpu_mem_free_gb": None,
        "gpu_mem_total_gb_nvml": None,
        "gpu_temp_c": None,
        "gpu_power_w": None,
        "gpu_sm_clock_mhz": None,
    }
    if pynvml is None:
        return info

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

        info["gpu_util_percent"] = getattr(util, "gpu", None)
        info["gpu_mem_used_gb"] = mem.used / (1024 ** 3)
        info["gpu_mem_free_gb"] = mem.free / (1024 ** 3)
        info["gpu_mem_total_gb_nvml"] = mem.total / (1024 ** 3)

        try:
            info["gpu_temp_c"] = pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            )
        except Exception:
            pass

        try:
            # milliwatts -> watts
            info["gpu_power_w"] = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        except Exception:
            pass

        try:
            info["gpu_sm_clock_mhz"] = pynvml.nvmlDeviceGetClockInfo(
                handle, pynvml.NVML_CLOCK_SM
            )
        except Exception:
            pass

    except Exception:
        pass
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    return info


def get_jetson_tegrastats() -> Dict[str, Any]:
    """
    Best-effort parse of a single tegrastats sample.
    Only works if tegrastats is installed and accessible.
    """
    info = {
        "tegrastats_available": False,
        "jetson_ram_used_mb": None,
        "jetson_ram_total_mb": None,
        "jetson_gpu_util_percent": None,
        "jetson_cpu_util_percent_avg": None,
    }

    tegrastats = shutil.which("tegrastats")
    if tegrastats is None:
        return info

    try:
        # Collect one line, then exit
        proc = subprocess.run(
            [tegrastats, "--interval", "1000"],
            capture_output=True,
            text=True,
            timeout=2.5,
        )
        text = (proc.stdout or proc.stderr or "").strip().splitlines()
        if not text:
            return info

        line = text[0]
        info["tegrastats_available"] = True

        m_ram = re.search(r"RAM\s+(\d+)/(\d+)MB", line)
        if m_ram:
            info["jetson_ram_used_mb"] = int(m_ram.group(1))
            info["jetson_ram_total_mb"] = int(m_ram.group(2))

        m_gr3d = re.search(r"GR3D_FREQ\s+(\d+)%", line)
        if m_gr3d:
            info["jetson_gpu_util_percent"] = int(m_gr3d.group(1))

        cpu_matches = re.findall(r"(\d+)%@", line)
        if cpu_matches:
            vals = [int(v) for v in cpu_matches]
            info["jetson_cpu_util_percent_avg"] = sum(vals) / len(vals)

    except Exception:
        pass

    return info


def ping_latency_ms(host: str, timeout_sec: float = 1.5) -> Optional[float]:
    """
    Linux/macOS ping parser, single probe.
    """
    try:
        proc = subprocess.run(
            ["ping", "-c", "1", "-W", str(int(max(timeout_sec, 1))), host],
            capture_output=True,
            text=True,
            timeout=timeout_sec + 1.0,
        )
        txt = (proc.stdout or "") + "\n" + (proc.stderr or "")
        m = re.search(r"time[=<]([\d.]+)\s*ms", txt)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return None


def read_queue_backlog(queue_state_path: Optional[str]) -> Dict[str, Any]:
    info = {
        "queue_backlog_jobs": None,
        "queue_ready_jobs": None,
        "queue_running_jobs": None,
    }
    if not queue_state_path:
        return info

    try:
        path = Path(queue_state_path)
        if not path.exists():
            return info
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            if "jobs" in data and isinstance(data["jobs"], list):
                jobs = data["jobs"]
                info["queue_backlog_jobs"] = len(jobs)
                info["queue_ready_jobs"] = sum(1 for j in jobs if j.get("status") == "ready")
                info["queue_running_jobs"] = sum(1 for j in jobs if j.get("status") == "running")
            else:
                info["queue_backlog_jobs"] = data.get("queue_backlog_jobs")
                info["queue_ready_jobs"] = data.get("queue_ready_jobs")
                info["queue_running_jobs"] = data.get("queue_running_jobs")
    except Exception:
        pass
    return info


def read_reliability_score(history_log: Optional[str], lookback_last_n: int = 50) -> Dict[str, Any]:
    info = {
        "reliability_success_rate": None,
        "reliability_samples": 0,
    }
    if not history_log:
        return info

    path = Path(history_log)
    if not path.exists():
        return info

    successes: List[int] = []
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            status = rec.get("status")
            if status in {"success", "ok", "completed"}:
                successes.append(1)
            elif status in {"failed", "timeout", "error"}:
                successes.append(0)

        if successes:
            tail = successes[-lookback_last_n:]
            info["reliability_samples"] = len(tail)
            info["reliability_success_rate"] = sum(tail) / len(tail)
    except Exception:
        pass

    return info


def flatten_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            for kk, vv in flatten_dict(v).items():
                out[f"{k}_{kk}"] = vv
        else:
            out[k] = v
    return out


def collect_telemetry(
    label: Optional[str],
    ping_hosts: List[str],
    queue_state_path: Optional[str],
    reliability_log: Optional[str],
) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "timestamp_utc": utc_now_iso(),
        "hostname": get_hostname(),
        "label": label,
        "platform": platform.platform(),
        "python_executable": sys.executable,
    }

    info.update(get_cpu_info())
    info.update(get_memory_info())
    info.update(get_battery_info())
    info.update(get_cuda_info_torch())
    info.update(get_gpu_info_nvml())
    info.update(get_jetson_tegrastats())
    info.update(read_queue_backlog(queue_state_path))
    info.update(read_reliability_score(reliability_log))

    if ping_hosts:
        latencies = {}
        for host in ping_hosts:
            latencies[f"latency_ms__{host.replace('.', '_').replace(':', '_')}"] = ping_latency_ms(host)
        info.update(latencies)

    return info


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def append_csv(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(record.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(record)


def main():
    parser = argparse.ArgumentParser(description="Collect live device telemetry for orchestration.")
    parser.add_argument("--label", type=str, default=None, help="Logical label for this device, e.g. gpu, jetson, pi5a")
    parser.add_argument("--ping-host", action="append", default=[], help="Host/IP to ping for latency. Can be repeated.")
    parser.add_argument("--queue-state-path", type=str, default=None, help="Optional JSON file containing queue/backlog state.")
    parser.add_argument("--reliability-log", type=str, default=None, help="Optional JSONL event log for reliability score.")
    parser.add_argument("--json-out", type=str, default=None, help="Write latest telemetry snapshot as JSON to this path.")
    parser.add_argument("--jsonl-out", type=str, default=None, help="Append telemetry snapshot as JSONL to this path.")
    parser.add_argument("--csv-out", type=str, default=None, help="Append telemetry snapshot as CSV row to this path.")
    parser.add_argument("--watch", action="store_true", help="Continuously sample telemetry.")
    parser.add_argument("--interval-sec", type=float, default=5.0, help="Sampling interval for --watch mode.")
    args = parser.parse_args()

    def emit_once() -> Dict[str, Any]:
        rec = collect_telemetry(
            label=args.label,
            ping_hosts=args.ping_host,
            queue_state_path=args.queue_state_path,
            reliability_log=args.reliability_log,
        )

        if args.json_out:
            out = Path(args.json_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(rec, indent=2))

        if args.jsonl_out:
            append_jsonl(Path(args.jsonl_out), rec)

        if args.csv_out:
            append_csv(Path(args.csv_out), rec)

        print(json.dumps(rec))
        return rec

    if not args.watch:
        emit_once()
        return

    while True:
        emit_once()
        time.sleep(max(0.1, args.interval_sec))


if __name__ == "__main__":
    import os

    main()
