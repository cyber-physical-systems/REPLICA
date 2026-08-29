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