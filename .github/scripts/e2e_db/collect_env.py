#!/usr/bin/env python3
"""Write the environment JSON consumed by load_results.py --meta (host/container/torch/python).

Run it in the test environment. Inside a container, mount the host's /etc/hostname and
/etc/os-release under --host-etc so the host row records the real machine.
"""
import argparse
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

import torch


def run(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=True).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def read(path):
    path = Path(path)
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def os_name(etc):
    match = re.search(r'^PRETTY_NAME="?([^"\n]*)"?', read(Path(etc) / "os-release"), re.M)
    return match.group(1) if match else None


def cpu():
    match = re.search(r"^model name\s*:\s*(.+)$", read("/proc/cpuinfo"), re.M)
    return match.group(1).strip() if match else platform.processor() or None


def memory():
    match = re.search(r"^MemTotal:\s*(\d+) kB", read("/proc/meminfo"), re.M)
    return f"{round(int(match.group(1)) / 1024**2)}G"


def disk(path, mount_only=True):
    if not os.path.isdir(path) or (mount_only and not os.path.ismount(path)):
        return None
    return f"{round(shutil.disk_usage(path).total / 1024**3)}G"


def deb(*names):
    for name in names:
        version = run("dpkg-query", "-W", "-f=${Version}", name)
        if version:
            return version
    return None


def dist(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def gpu():
    count = torch.xpu.device_count() if torch.xpu.is_available() else 0
    return f"{count}x {torch.xpu.get_device_name(0)}" if count else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--host-etc", default="/etc", help="directory holding the host's hostname and os-release")
    parser.add_argument("--container-image", default=os.environ.get("E2E_CONTAINER_IMAGE"),
                        help="container image name; omit on bare metal")
    parser.add_argument("--torch-name", default="pytorch", help="torch source label, e.g. main or nightly_wheel")
    parser.add_argument("--pytorch-dir", default="pytorch")
    parser.add_argument("--torchbench-dir", default="benchmark")
    args = parser.parse_args()

    stack = {
        "compute_runtime": deb("intel-opencl-icd"),
        "igc": deb("intel-igc-core-2", "intel-igc-core", "libigc2", "libigc1"),
        "l0": deb("libze1", "libze-intel-gpu1", "level-zero"),
        "gcc": run("gcc", "-dumpfullversion"),
    }
    in_container = bool(args.container_image)
    host_os = os_name(args.host_etc)
    host = {
        "hostname": read(Path(args.host_etc) / "hostname").strip() or platform.node(),
        "cpu": cpu(),
        "gpu": gpu(),
        "memory": memory(),
        "disk_root": disk("/", mount_only=False),
        "disk_home": None if in_container else disk("/home"),
        "disk_opt": None if in_container else disk("/opt"),
        "disk_var": None if in_container else disk("/var"),
        "os": host_os,
        "kernel": platform.release(),
        # Inside a container the user-space driver stack belongs to the image, not the host.
        **({} if in_container else stack),
    }
    container = {"name": args.container_image, "os": os_name("/etc"), **stack} if in_container \
        else {"name": "none", "os": host_os}
    meta = {
        "host": host,
        "container": container,
        "torch": {
            "name": args.torch_name,
            "version": torch.__version__,
            "commit": torch.version.git_version,
            "repo": run("git", "-C", args.pytorch_dir, "remote", "get-url", "origin"),
            "torchvision": dist("torchvision"),
            "torchaudio": dist("torchaudio"),
            "torchbench": run("git", "-C", args.torchbench_dir, "rev-parse", "HEAD"),
            "transformers": dist("transformers"),
            "timm": dist("timm"),
            "pytest": dist("pytest"),
        },
        "python": {"version": platform.python_version()},
    }
    args.output.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
