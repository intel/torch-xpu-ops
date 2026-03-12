#!/usr/bin/env python3
"""
Summarize test environment from collect_env.py and printenv outputs.
Usage:
    python summarize_env.py --collect-env collect_env.txt --printenv printenv.txt --output summary.md
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any


def parse_collect_env(text: str) -> Dict[str, Any]:
    """Parse the output of PyTorch's collect_env.py into a dictionary."""
    data = {"_raw": text}  # store raw text for later inclusion
    lines = text.splitlines()

    # State for multi-line sections
    in_gpu_driver = False
    in_gpu_detected = False
    in_cpu = False
    in_versions = False
    gpu_driver_lines = []
    gpu_detected_lines = []
    cpu_lines = []
    pip_packages = []

    for idx, line in enumerate(lines):
        line = line.rstrip()

        # Detect section headers
        if line.startswith("Intel GPU driver version:"):
            in_gpu_driver = True
            in_gpu_detected = False
            in_cpu = False
            in_versions = False
            gpu_driver_lines = [line]
            continue
        if line.startswith("Intel GPU models detected:"):
            in_gpu_detected = True
            in_gpu_driver = False
            in_cpu = False
            in_versions = False
            gpu_detected_lines = [line]
            continue
        if line.startswith("CPU:"):
            in_cpu = True
            in_gpu_driver = False
            in_gpu_detected = False
            in_versions = False
            cpu_lines = [line]
            continue
        if line.startswith("Versions of relevant libraries:"):
            in_versions = True
            in_gpu_driver = False
            in_gpu_detected = False
            in_cpu = False
            continue

        # Handle multi-line sections
        if in_gpu_driver:
            gpu_driver_lines.append(line)
            # Section ends when we hit an empty line or next section
            if line == "" or (idx < len(lines)-1 and lines[idx+1] and not lines[idx+1].startswith("*")):
                in_gpu_driver = False
                data["gpu_driver_version"] = "\n".join(gpu_driver_lines)
        elif in_gpu_detected:
            gpu_detected_lines.append(line)
            if line == "" or (idx < len(lines)-1 and lines[idx+1] and not lines[idx+1].startswith("*")):
                in_gpu_detected = False
                data["gpu_detected"] = "\n".join(gpu_detected_lines)
        elif in_cpu:
            cpu_lines.append(line)
            # CPU block ends with an empty line
            if line == "":
                in_cpu = False
                data["cpu_info"] = "\n".join(cpu_lines)
        elif in_versions:
            if line.startswith("[pip3]"):
                # Extract package and version
                match = re.match(r"\[pip3\] (\S+)==(\S+)", line)
                if match:
                    pip_packages.append((match.group(1), match.group(2)))
            # Stop at next empty line or end of section
            if line == "":
                in_versions = False
                data["pip_packages"] = pip_packages
        else:
            # Regular key: value lines
            if ": " in line and not line.startswith(" "):
                key, val = line.split(": ", 1)
                data[key.strip()] = val.strip()

    # Post-process some fields
    if "pip_packages" not in data:
        data["pip_packages"] = pip_packages
    if "cpu_info" in data:
        # Extract useful CPU details
        cpu_data = {}
        for line in data["cpu_info"].splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                cpu_data[k.strip()] = v.strip()
        data["cpu_summary"] = cpu_data

    return data


def parse_printenv(text: str) -> Dict[str, str]:
    """Parse printenv output into a dictionary."""
    env = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, val = line.split("=", 1)
        env[key] = val
    return env


def filter_pip_packages(packages: List[tuple]) -> List[tuple]:
    """Keep only packages matching patterns: intel-cmplr-*, torch, torch-*, *triton*."""
    filtered = []
    for pkg, ver in packages:
        if (pkg.startswith("intel-cmplr-") or
            pkg == "torch" or
            pkg.startswith("torch-") or
            "triton" in pkg):
            filtered.append((pkg, ver))
    return filtered


def extract_gpu_device_info(gpu_detected_text: str) -> tuple:
    """Extract first GPU name and total memory from the detected GPUs block."""
    name = "Unknown"
    memory = "?"
    # Look for line starting with "* [0]"
    for line in gpu_detected_text.splitlines():
        if line.startswith("* [0]"):
            # Extract name
            name_match = re.search(r"name='([^']+)'", line)
            if name_match:
                name = name_match.group(1)
            # Extract total_memory
            mem_match = re.search(r"total_memory=(\d+)MB", line)
            if mem_match:
                memory = mem_match.group(1)
            break
    return name, memory


def extract_gpu_driver_version(gpu_driver_text: str) -> str:
    """Extract intel-opencl-icd version from driver block."""
    for line in gpu_driver_text.splitlines():
        if "intel-opencl-icd:" in line:
            parts = line.split(":", 1)
            if len(parts) == 2:
                return parts[1].strip()
    return "N/A"


def generate_markdown(collect_env: Dict[str, Any], printenv: Dict[str, str]) -> str:
    """Generate GitHub Flavored Markdown summary with emojis."""
    lines = []
    lines.append("# 🧪 Test Environment Summary")
    lines.append("")

    # --- PyTorch / Collect Env Section ---
    lines.append("## 🔥 PyTorch Environment (`collect_env.py`)")
    lines.append("")

    # Basic PyTorch info (Core)
    lines.append("### 🔧 Core")
    lines.append("")
    lines.append(f"- **PyTorch version**: `{collect_env.get('PyTorch version', 'N/A')}`")
    lines.append(f"- **XPU available**: `{collect_env.get('Is XPU available', 'N/A')}`")
    lines.append(f"- **XPU build version**: `{collect_env.get('XPU used to build PyTorch', 'N/A')}`")

    # Add Device summary
    if "gpu_detected" in collect_env:
        gpu_name, gpu_mem = extract_gpu_device_info(collect_env["gpu_detected"])
        lines.append(f"- **🎮 Device**: `{gpu_name} – {gpu_mem} MB`")
    else:
        lines.append("- **🎮 Device**: N/A")
    # Add GPU driver version
    if "gpu_driver_version" in collect_env:
        driver_ver = extract_gpu_driver_version(collect_env["gpu_driver_version"])
        lines.append(f"- **🛠️ Intel GPU Driver**: `{driver_ver}`")
    else:
        lines.append("- **🛠️ Intel GPU Driver**: N/A")
    # Add ZE_AFFINITY_MASK from printenv
    ze_mask = printenv.get("ZE_AFFINITY_MASK", "N/A")
    lines.append(f"- **⚙️ ZE_AFFINITY_MASK**: `{ze_mask}`")
    lines.append("")

    # System info (OS, Python, etc.)
    lines.append("### 🖥️ System")
    lines.append("")
    lines.append(f"- **OS**: `{collect_env.get('OS', 'N/A')}`")
    lines.append(f"- **🐍 Python version**: `{collect_env.get('Python version', 'N/A')}`")
    lines.append(f"- **Python platform**: `{collect_env.get('Python platform', 'N/A')}`")
    lines.append(f"- **GCC version**: `{collect_env.get('GCC version', 'N/A')}`")
    lines.append(f"- **CMake version**: `{collect_env.get('CMake version', 'N/A')}`")
    lines.append("")

    # CPU summary
    if "cpu_summary" in collect_env:
        cs = collect_env["cpu_summary"]
        lines.append("### 🧠 CPU")
        lines.append("")
        lines.append(f"- **Model**: `{cs.get('Model name', 'N/A')}`")
        lines.append(f"- **CPU(s)**: `{cs.get('CPU(s)', 'N/A')}`")
        lines.append(f"- **NUMA node(s)**: `{cs.get('NUMA node(s)', 'N/A')}`")
        lines.append(f"- **Architecture**: `{cs.get('Architecture', 'N/A')}`")
        lines.append("")

    # Filtered pip packages
    if "pip_packages" in collect_env and collect_env["pip_packages"]:
        filtered_pips = filter_pip_packages(collect_env["pip_packages"])
        if filtered_pips:
            lines.append("### 📦 Relevant pip packages")
            lines.append("")
            lines.append("| Package | Version |")
            lines.append("|---------|---------|")
            for pkg, ver in filtered_pips:
                lines.append(f"| {pkg} | {ver} |")
            lines.append("")

    # --- printenv Section ---
    lines.append("## 🌐 Environment Variables (`printenv`)")
    lines.append("")

    # GitHub Actions specific keys
    gh_keys = ["GITHUB_ACTOR", "GITHUB_EVENT_NAME", "GITHUB_REF_NAME",
               "GITHUB_REPOSITORY", "GITHUB_RUN_ID", "GITHUB_SHA", "GITHUB_WORKFLOW"]
    present_gh = {k: printenv.get(k, "N/A") for k in gh_keys if k in printenv}
    if present_gh:
        lines.append("### 🤖 GitHub Actions")
        lines.append("")
        for key in sorted(present_gh.keys()):
            val = present_gh[key]
            # Mask if it looks like a token (though input already has ***)
            if "TOKEN" in key or "SECRET" in key:
                val = "***"
            lines.append(f"- **{key}**: `{val}`")
        lines.append("")

    # Other relevant env vars (non-proxy)
    other_keys = ["CI", "PATH", "LD_LIBRARY_PATH", "PYTHONPATH"]
    for key in other_keys:
        if key in printenv:
            lines.append(f"- **{key}**: `{printenv[key]}`")
    if any(k in printenv for k in other_keys):
        lines.append("")

    # Optionally include full printenv in a collapsible section
    lines.append("<details>")
    lines.append("<summary><b>📋 Full <code>printenv</code> output</b></summary>")
    lines.append("")
    lines.append("```")
    for k, v in sorted(printenv.items()):
        # Mask any line that looks like a token (basic check)
        if "TOKEN" in k or "SECRET" in k or "PASSWORD" in k:
            v = "***"
        lines.append(f"{k}={v}")
    lines.append("```")
    lines.append("</details>")
    lines.append("")

    # Include full collect_env output in a collapsible section
    if "_raw" in collect_env:
        lines.append("<details>")
        lines.append("<summary><b>📋 Full <code>collect_env.py</code> output</b></summary>")
        lines.append("")
        lines.append("```")
        lines.append(collect_env["_raw"].rstrip())
        lines.append("```")
        lines.append("</details>")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Summarize test environment for GitHub Markdown.")
    parser.add_argument("-c", "--collect-env", required=True, help="Path to file containing output of collect_env.py")
    parser.add_argument("-p", "--printenv", required=True, help="Path to file containing output of printenv")
    parser.add_argument("-o", "--output", required=True, help="Path to output Markdown file")
    args = parser.parse_args()

    collect_text = Path(args.collect_env).read_text()
    printenv_text = Path(args.printenv).read_text()

    collect_data = parse_collect_env(collect_text)
    printenv_data = parse_printenv(printenv_text)

    markdown = generate_markdown(collect_data, printenv_data)

    Path(args.output).write_text(markdown)
    print(f"Summary written to {args.output}")


if __name__ == "__main__":
    main()