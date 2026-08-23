# sync_all.py

import subprocess

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



LOCAL_PROJECT = "."


EXCLUDES = [
    ".git",
    "__pycache__",
    "*.pyc",
    "*.pyo",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "runs",
    "wandb",
    ".venv",
    "venv",
]


def sync_device(name, cfg):
    ssh_transport = "ssh"
    if cfg.get("ssh_opts"):
        ssh_transport += " " + " ".join(cfg["ssh_opts"])

    mkdir_cmd = ["ssh"]
    mkdir_cmd.extend(cfg.get("ssh_opts", []))
    mkdir_cmd.extend([
        cfg["ssh"],
        f"mkdir -p {cfg['project_dir']}"
    ])

    rsync_cmd = ["rsync", "-avz"]

    for pattern in EXCLUDES:
        rsync_cmd.extend(["--exclude", pattern])

    rsync_cmd.extend([
        "-e",
        ssh_transport,
        f"{LOCAL_PROJECT}/",
        f"{cfg['ssh']}:{cfg['project_dir']}/",
    ])

    try:
        print(f"\n=== Syncing {name} ===")
        print(f"Creating {cfg['project_dir']} on {name}")
        subprocess.run(mkdir_cmd, check=True)

        print(" ".join(rsync_cmd))
        subprocess.run(rsync_cmd, check=True)

        print(f"[OK] {name}")

    except subprocess.CalledProcessError as e:
        print(f"[FAIL] {name}: {e}")


def main():

    for name, cfg in DEVICE_CONFIG.items():
        sync_device(name, cfg)


if __name__ == "__main__":
    main()