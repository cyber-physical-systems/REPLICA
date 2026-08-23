# run_all.py

import json
import subprocess
import pandas as pd

from device_config import DEVICE_CONFIG


def run_device_telemetry(device_name):
    cfg = DEVICE_CONFIG[device_name]

    remote_cmd = (
        f"cd {cfg['project_dir']} && "
        f"{cfg['python']} device_telemetry.py "
        f"--label {device_name}"
    )

    ssh_cmd = ["ssh"]
    ssh_cmd.extend(cfg.get("ssh_opts", []))
    ssh_cmd.extend([cfg["ssh"], remote_cmd])

    try:
        output = subprocess.check_output(
            ssh_cmd,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=60,
        )

        result = json.loads(output.strip().splitlines()[-1])
        result["device"] = device_name
        result["status"] = "success"
        return result

    except Exception as e:
        return {
            "device": device_name,
            "label": device_name,
            "status": "failed",
            "error": str(e),
            "ssh_cmd": " ".join(ssh_cmd),
        }


def main():
    rows = []

    for device in DEVICE_CONFIG:
        print(f"Collecting telemetry from {device}")
        rows.append(run_device_telemetry(device))

    df = pd.DataFrame(rows)

    print("\nTelemetry")
    print(df)

    df.to_csv("distributed_telemetry.csv", index=False)
    print("\nSaved distributed_telemetry.csv")


if __name__ == "__main__":
    main()