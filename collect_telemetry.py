#!/usr/bin/env python3

import json
import subprocess
from pathlib import Path
import pandas as pd

from device_config import DEVICE_CONFIG


OUT_DIR = Path("telemetry_runs")
OUT_DIR.mkdir(exist_ok=True)


def run_remote_telemetry(device_name: str) -> dict:
    cfg = DEVICE_CONFIG[device_name]

    remote_cmd = (
        f"cd {cfg['project_dir']} && "
        f"{cfg['python']} device_telemetry.py --label {device_name}"
    )

    ssh_cmd = ["ssh"]
    ssh_cmd.extend(cfg.get("ssh_opts", []))
    ssh_cmd.extend([cfg["ssh"], remote_cmd])

    try:
        output = subprocess.check_output(
            ssh_cmd,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=90,
        )

        # Some machines print warnings before JSON, so take the last JSON-looking line.
        lines = [x.strip() for x in output.splitlines() if x.strip()]
        json_line = None

        for line in reversed(lines):
            if line.startswith("{") and line.endswith("}"):
                json_line = line
                break

        if json_line is None:
            raise RuntimeError(f"No JSON found in output:\n{output}")

        rec = json.loads(json_line)
        rec["device"] = device_name
        rec["status"] = "success"
        return rec

    except Exception as e:
        return {
            "device": device_name,
            "label": device_name,
            "status": "failed",
            "error": str(e),
            "ssh_cmd": " ".join(ssh_cmd),
        }


def collect_all() -> dict:
    telemetry = {}

    for device_name in DEVICE_CONFIG:
        print(f"[telemetry] collecting {device_name}")
        telemetry[device_name] = run_remote_telemetry(device_name)

    return telemetry


def save_outputs(telemetry: dict) -> None:
    latest_json = OUT_DIR / "latest_telemetry.json"
    latest_csv = OUT_DIR / "latest_telemetry.csv"

    latest_json.write_text(json.dumps(telemetry, indent=2))

    rows = list(telemetry.values())
    df = pd.DataFrame(rows)
    df.to_csv(latest_csv, index=False)

    print(f"\nSaved {latest_json}")
    print(f"Saved {latest_csv}")


def main() -> None:
    telemetry = collect_all()

    print("\n[summary]")
    for name, rec in telemetry.items():
        status = rec.get("status")
        cpu = rec.get("cpu_load_percent")
        ram = rec.get("ram_available_gb")
        gpu = rec.get("gpu_name")
        gpu_util = rec.get("gpu_util_percent")

        print(
            f"{name:10s} status={status:8s} "
            f"cpu={cpu} ram={ram} gpu={gpu} gpu_util={gpu_util}"
        )

    save_outputs(telemetry)


if __name__ == "__main__":
    main()