#!/usr/bin/env python3
"""
Test Environment Summary Generator

Parses a combined log file containing output from PyTorch's collect_env.py
and printenv (with possible command lines like '+ python ...') and produces
a concise Markdown summary with emojis.

Usage:
    # Single file mode
    ./summary.py --input combined.log --output summary.md

    # Comparison mode (baseline vs target)
    ./summary.py --baseline baseline.log --target target.log --output comparison.md
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional


# ==============================================================================
# Constants and Emojis
# ==============================================================================

EMOJI = {
    "test_tube": "🧪",
    "fire": "🔥",
    "wrench": "🔧",
    "computer": "🖥️",
    "brain": "🧠",
    "clipboard": "📋",
    "snake": "🐍",
    "package": "📦",  # for dependencies
}

# Core package names of interest
PACKAGES_OF_INTEREST = {
    "oneapi": "intel-cmplr-lib-rt",
    "triton": "triton-xpu",
}

# Dependency package names
DEPENDENCY_PACKAGES = {
    "torchvision": "torchvision",
    "torchaudio": "torchaudio",
    "torchao": "torchao",
}

# Dependency environment variables
DEPENDENCY_ENV_VARS = {
    "Transformers": "TRANSFORMERS_VERSION",
    "Timm": "TIMM_COMMIT_ID",
    "TorchBench": "TORCHBENCH_COMMIT_ID",
}


# ==============================================================================
# Log Preprocessing
# ==============================================================================

def strip_command_lines(raw_log: str) -> str:
    """
    Remove lines that start with '+' (after stripping whitespace).
    These are typically shell command prompts.
    """
    lines = raw_log.splitlines()
    cleaned = []
    for line in lines:
        if line.lstrip().startswith('+'):
            continue
        cleaned.append(line)
    return '\n'.join(cleaned)


# ==============================================================================
# CollectEnv Parser
# ==============================================================================

def parse_collect_env_output(cleaned_text: str) -> Dict[str, Any]:
    """
    Extract structured information from the collect_env.py output.
    Returns a dictionary with keys like 'PyTorch version', 'OS', etc.,
    plus special keys for multi-line sections:
        'gpu_driver_version', 'gpu_detected', 'cpu_info', 'cpu_summary',
        'pip_packages', '_raw' (the cleaned text).
    """
    data = {"_raw": cleaned_text}
    lines = cleaned_text.splitlines()

    # State machine for parsing multi-line sections
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

        # Section headers
        if line.startswith("Intel GPU driver version:"):
            in_gpu_driver, in_gpu_detected, in_cpu, in_versions = True, False, False, False
            gpu_driver_lines = [line]
            continue
        if line.startswith("Intel GPU models detected:"):
            in_gpu_detected, in_gpu_driver, in_cpu, in_versions = True, False, False, False
            gpu_detected_lines = [line]
            continue
        if line.startswith("CPU:"):
            in_cpu, in_gpu_driver, in_gpu_detected, in_versions = True, False, False, False
            cpu_lines = [line]
            continue
        if line.startswith("Versions of relevant libraries:"):
            in_versions, in_gpu_driver, in_gpu_detected, in_cpu = True, False, False, False
            continue

        # Multi-line section accumulation
        if in_gpu_driver:
            gpu_driver_lines.append(line)
            # End of section: empty line or next non-bullet line
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
            if line == "":
                in_cpu = False
                data["cpu_info"] = "\n".join(cpu_lines)
        elif in_versions:
            if line.startswith("[pip3]"):
                match = re.match(r"\[pip3\] (\S+)==(\S+)", line)
                if match:
                    pip_packages.append((match.group(1), match.group(2)))
            if line == "":
                in_versions = False
                data["pip_packages"] = pip_packages
        else:
            # Regular key: value lines
            if ": " in line and not line.startswith(" "):
                key, val = line.split(": ", 1)
                data[key.strip()] = val.strip()

    # Ensure pip_packages is present
    if "pip_packages" not in data:
        data["pip_packages"] = pip_packages

    # Post-process CPU info into a summary dictionary
    if "cpu_info" in data:
        cpu_summary = {}
        for line in data["cpu_info"].splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                cpu_summary[k.strip()] = v.strip()
        data["cpu_summary"] = cpu_summary

    return data


# ==============================================================================
# Printenv Parser
# ==============================================================================

def parse_printenv_output(cleaned_text: str) -> Dict[str, str]:
    """
    Extract environment variables from lines containing '='.
    Returns a dictionary of variable name → value.
    """
    env = {}
    for line in cleaned_text.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, val = line.split("=", 1)
        env[key] = val
    return env


# ==============================================================================
# Specific Extractors
# ==============================================================================

def get_pip_version(pip_packages: List[tuple], package_name: str) -> str:
    """Return version of a pip package, or 'N/A' if not found."""
    for pkg, ver in pip_packages:
        if pkg == package_name:
            return ver
    return "N/A"


def clean_version_string(version_str: str) -> str:
    """Remove common prefixes like 'version ' from version strings."""
    if not version_str or version_str == "N/A":
        return version_str
    # Remove leading "version " (case-insensitive)
    cleaned = re.sub(r'^version\s+', '', version_str, flags=re.IGNORECASE)
    return cleaned.strip()


def extract_gpu_device_info(gpu_detected_text: str) -> Tuple[str, str]:
    """
    From the 'Intel GPU models detected' block, extract the name and total memory
    of the first GPU (index 0).
    """
    name = "Unknown"
    memory = "?"
    for line in gpu_detected_text.splitlines():
        if line.startswith("* [0]"):
            name_match = re.search(r"name='([^']+)'", line)
            if name_match:
                name = name_match.group(1)
            mem_match = re.search(r"total_memory=(\d+)MB", line)
            if mem_match:
                memory = mem_match.group(1)
            break
    return name, memory


def extract_gpu_driver_version(gpu_driver_text: str) -> str:
    """
    From the 'Intel GPU driver version' block, extract the version of
    intel-opencl-icd.
    """
    for line in gpu_driver_text.splitlines():
        if "intel-opencl-icd:" in line:
            parts = line.split(":", 1)
            if len(parts) == 2:
                return parts[1].strip()
    return "N/A"


def extract_onednn_version(full_raw_log: str) -> str:
    """
    Scan the raw log for a line containing 'onednn_verbose' and extract
    the version part after 'oneDNN '.
    Example: 'onednn_verbose,v1,info,oneDNN v3.10.2 (commit ...)'
    Returns version string or 'N/A'.
    """
    for line in full_raw_log.splitlines():
        if "onednn_verbose" in line and "oneDNN" in line:
            match = re.search(r"oneDNN\s+(.+)", line)
            if match:
                return match.group(1).strip()
    return "N/A"


def extract_memory_info(cleaned_text: str) -> str:
    """
    Extract total memory from a line starting with 'Mem:'.
    Example line: "Mem:            31Gi       5.6Gi        15Gi       221Mi        10Gi        25Gi"
    Returns the total value (second field) as a string, e.g., "31Gi", or "N/A".
    """
    for line in cleaned_text.splitlines():
        if line.startswith("Mem:"):
            parts = line.split()
            if len(parts) >= 2:
                return parts[1]  # total memory
    return "N/A"


def extract_disk_info(cleaned_text: str) -> str:
    """
    Find the line where mount point is '/' and extract 'Avail / Size'.
    Looks for lines with fields like: "/dev/mapper/vgubuntu-root  467G  325G  119G  74% /"
    Returns string like "119G / 467G" or "N/A".
    """
    lines = cleaned_text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().endswith('/'):
            # This line might be the mount point line; parse fields
            fields = line.split()
            # Expected: filesystem size used avail use% mountpoint
            if len(fields) >= 5 and fields[-1] == '/':
                size = fields[1]   # Size column
                avail = fields[3]  # Avail column
                return f"{avail} / {size}"
    return "N/A"


# ==============================================================================
# Markdown Generation (Single Summary)
# ==============================================================================

def generate_single_summary(
    collect_data: Dict[str, Any],
    env_vars: Dict[str, str],
    raw_log: str
) -> str:
    """Create a Markdown summary for a single environment."""
    lines = []
    lines.append(f"# {EMOJI['test_tube']} Test Environment Summary")
    lines.append("")

    # ----- Core Section -----
    lines.append(f"### {EMOJI['wrench']} Core")
    lines.append(f"- **PyTorch version**: `{collect_data.get('PyTorch version', 'N/A')}`")

    pkgs = collect_data.get("pip_packages", [])
    oneapi_ver = get_pip_version(pkgs, PACKAGES_OF_INTEREST["oneapi"])
    lines.append(f"- **oneAPI**: `{oneapi_ver}`")

    onednn_ver = extract_onednn_version(raw_log)
    lines.append(f"- **oneDNN**: `{onednn_ver}`")

    triton_ver = get_pip_version(pkgs, PACKAGES_OF_INTEREST["triton"])
    lines.append(f"- **Triton**: `{triton_ver}`")

    if "gpu_driver_version" in collect_data:
        driver_ver = extract_gpu_driver_version(collect_data["gpu_driver_version"])
    else:
        driver_ver = "N/A"
    lines.append(f"- **Driver**: `{driver_ver}`")
    lines.append("")

    # ----- Dependencies Section -----
    lines.append(f"### {EMOJI['package']} Dependencies")
    # torch packages
    for name, pkg in DEPENDENCY_PACKAGES.items():
        ver = get_pip_version(pkgs, pkg)
        lines.append(f"- **{name}**: `{ver}`")
    # env vars
    for name, env_var in DEPENDENCY_ENV_VARS.items():
        val = env_vars.get(env_var, "N/A")
        lines.append(f"- **{name}**: `{val}`")
    lines.append("")

    # ----- System Section -----
    lines.append(f"### {EMOJI['computer']} System")
    lines.append(f"- **OS**: `{collect_data.get('OS', 'N/A')}`")
    lines.append(f"- **Kernel**: `{collect_data.get('Kernel version', 'N/A')}`")
    lines.append(f"- **{EMOJI['snake']} Python version**: `{collect_data.get('Python version', 'N/A')}`")
    lines.append(f"- **GCC version**: `{collect_data.get('GCC version', 'N/A')}`")
    cmake_ver = clean_version_string(collect_data.get('CMake version', 'N/A'))
    lines.append(f"- **CMake version**: `{cmake_ver}`")

    # Memory and Disk from the raw log
    memory = extract_memory_info(raw_log)  # use raw to include all lines
    lines.append(f"- **Memory**: `{memory}`")
    disk = extract_disk_info(raw_log)
    lines.append(f"- **Disk**: `{disk}`")
    lines.append("")

    # ----- CPU & GPU Section -----
    lines.append(f"### {EMOJI['brain']} CPU & GPU")

    # CPU details
    if "cpu_summary" in collect_data:
        cpu = collect_data["cpu_summary"]
        lines.append(f"- **CPU Model**: `{cpu.get('Model name', 'N/A')}`")
        lines.append(f"- **CPU(s)**: `{cpu.get('CPU(s)', 'N/A')}`")
        lines.append(f"- **Architecture**: `{cpu.get('Architecture', 'N/A')}`")
    else:
        lines.append("- **CPU Model**: N/A")

    # GPU Model
    if "gpu_detected" in collect_data:
        gpu_name, gpu_mem = extract_gpu_device_info(collect_data["gpu_detected"])
        lines.append(f"- **GPU Model**: `{gpu_name} – {gpu_mem} MB`")
    else:
        lines.append("- **GPU Model**: N/A")

    # GPU(s) from ZE_AFFINITY_MASK – show count and list
    ze_mask = env_vars.get("ZE_AFFINITY_MASK", "N/A")
    if ze_mask != "N/A" and ze_mask.strip():
        gpu_indices = [idx.strip() for idx in ze_mask.split(',') if idx.strip()]
        count = len(gpu_indices)
        lines.append(f"- **GPU(s)**: `{count} ({', '.join(gpu_indices)})`")
    else:
        lines.append("- **GPU(s)**: N/A")
    lines.append("")

    # ----- Full Log (Collapsible) -----
    lines.append("<details>")
    lines.append(f"<summary><b>{EMOJI['clipboard']} Full combined log</b></summary>")
    lines.append("")
    lines.append("```")
    lines.append(raw_log.rstrip())
    lines.append("```")
    lines.append("</details>")
    lines.append("")

    return "\n".join(lines)


# ==============================================================================
# Markdown Generation (Comparison Summary)
# ==============================================================================

def _format_value(val: Any) -> str:
    """Convert a value to string, handling None/N/A."""
    if val is None or val == "N/A":
        return "N/A"
    return str(val)


def _add_comparison_row(
    lines: List[str],
    description: str,
    target_val: Any,      # Note: order swapped: target first, baseline second
    baseline_val: Any,
    fmt: str = "`{}`"
) -> None:
    """Helper to add a row to the comparison table, marking differences.
       Columns: Target | Baseline (Target on left)."""
    t_str = fmt.format(_format_value(target_val))
    b_str = fmt.format(_format_value(baseline_val))
    marker = "" if t_str == b_str else " 🔄"
    lines.append(f"| **{description}** | {t_str} | {b_str} |{marker}")


def generate_comparison_summary(
    baseline_collect: Dict[str, Any],
    target_collect: Dict[str, Any],
    baseline_env: Dict[str, str],
    target_env: Dict[str, str],
    raw_baseline: str,
    raw_target: str
) -> str:
    """Create a Markdown comparison table between target and baseline (target left)."""
    lines = []
    lines.append(f"# {EMOJI['test_tube']} Test Environment Comparison")
    lines.append("")
    lines.append(f"| | {EMOJI['fire']} Target | {EMOJI['fire']} Baseline |")
    lines.append("| --- | --- | --- |")

    # Helper to get package version
    def pkg_ver(data, pkg):
        return get_pip_version(data.get("pip_packages", []), pkg)

    # ----- Core -----
    lines.append(f"| **{EMOJI['wrench']} Core** | | |")
    _add_comparison_row(lines, "PyTorch version",
                        target_collect.get('PyTorch version'),
                        baseline_collect.get('PyTorch version'))
    _add_comparison_row(lines, "oneAPI",
                        pkg_ver(target_collect, PACKAGES_OF_INTEREST["oneapi"]),
                        pkg_ver(baseline_collect, PACKAGES_OF_INTEREST["oneapi"]))
    _add_comparison_row(lines, "oneDNN",
                        extract_onednn_version(raw_target),
                        extract_onednn_version(raw_baseline))
    _add_comparison_row(lines, "Triton",
                        pkg_ver(target_collect, PACKAGES_OF_INTEREST["triton"]),
                        pkg_ver(baseline_collect, PACKAGES_OF_INTEREST["triton"]))

    t_driver = extract_gpu_driver_version(target_collect.get("gpu_driver_version", "")) if "gpu_driver_version" in target_collect else "N/A"
    b_driver = extract_gpu_driver_version(baseline_collect.get("gpu_driver_version", "")) if "gpu_driver_version" in baseline_collect else "N/A"
    _add_comparison_row(lines, "Driver", t_driver, b_driver)

    # ----- Dependencies -----
    lines.append(f"| **{EMOJI['package']} Dependencies** | | |")
    # torch packages
    for name, pkg in DEPENDENCY_PACKAGES.items():
        _add_comparison_row(lines, name,
                            pkg_ver(target_collect, pkg),
                            pkg_ver(baseline_collect, pkg))
    # env vars
    for name, env_var in DEPENDENCY_ENV_VARS.items():
        _add_comparison_row(lines, name,
                            target_env.get(env_var, "N/A"),
                            baseline_env.get(env_var, "N/A"))

    # ----- System -----
    lines.append(f"| **{EMOJI['computer']} System** | | |")
    _add_comparison_row(lines, "OS",
                        target_collect.get('OS'),
                        baseline_collect.get('OS'))
    _add_comparison_row(lines, "Kernel",
                        target_collect.get('Kernel version'),
                        baseline_collect.get('Kernel version'))
    _add_comparison_row(lines, "Python version",
                        target_collect.get('Python version'),
                        baseline_collect.get('Python version'))
    _add_comparison_row(lines, "GCC version",
                        target_collect.get('GCC version'),
                        baseline_collect.get('GCC version'))
    cmake_t = clean_version_string(target_collect.get('CMake version', 'N/A'))
    cmake_b = clean_version_string(baseline_collect.get('CMake version', 'N/A'))
    _add_comparison_row(lines, "CMake version", cmake_t, cmake_b)

    # Memory and Disk
    mem_t = extract_memory_info(raw_target)
    mem_b = extract_memory_info(raw_baseline)
    _add_comparison_row(lines, "Memory", mem_t, mem_b)
    disk_t = extract_disk_info(raw_target)
    disk_b = extract_disk_info(raw_baseline)
    _add_comparison_row(lines, "Disk", disk_t, disk_b)

    # ----- CPU & GPU -----
    lines.append(f"| **{EMOJI['brain']} CPU & GPU** | | |")

    # CPU
    t_cpu = target_collect.get("cpu_summary", {})
    b_cpu = baseline_collect.get("cpu_summary", {})
    _add_comparison_row(lines, "CPU Model",
                        t_cpu.get('Model name'),
                        b_cpu.get('Model name'))
    _add_comparison_row(lines, "CPU(s)",
                        t_cpu.get('CPU(s)'),
                        b_cpu.get('CPU(s)'))
    _add_comparison_row(lines, "Architecture",
                        t_cpu.get('Architecture'),
                        b_cpu.get('Architecture'))

    # GPU Model
    if "gpu_detected" in target_collect and "gpu_detected" in baseline_collect:
        t_name, t_mem = extract_gpu_device_info(target_collect["gpu_detected"])
        b_name, b_mem = extract_gpu_device_info(baseline_collect["gpu_detected"])
        _add_comparison_row(lines, "GPU Model",
                            f"{t_name} – {t_mem} MB",
                            f"{b_name} – {b_mem} MB")
    else:
        _add_comparison_row(lines, "GPU Model", "N/A", "N/A")

    # GPU(s) from ZE_AFFINITY_MASK – show count and list
    def format_gpus(mask):
        if mask == "N/A" or not mask or not mask.strip():
            return "N/A"
        indices = [idx.strip() for idx in mask.split(',') if idx.strip()]
        return f"{len(indices)} ({', '.join(indices)})"

    _add_comparison_row(lines, "GPU(s)",
                        format_gpus(target_env.get("ZE_AFFINITY_MASK", "N/A")),
                        format_gpus(baseline_env.get("ZE_AFFINITY_MASK", "N/A")))

    # ----- Full Logs (Collapsible) -----
    lines.append("")
    lines.append("<details>")
    lines.append(f"<summary><b>{EMOJI['clipboard']} Target full log</b></summary>")
    lines.append("")
    lines.append("```")
    lines.append(raw_target.rstrip())
    lines.append("```")
    lines.append("</details>")
    lines.append("")

    lines.append("<details>")
    lines.append(f"<summary><b>{EMOJI['clipboard']} Baseline full log</b></summary>")
    lines.append("")
    lines.append("```")
    lines.append(raw_baseline.rstrip())
    lines.append("```")
    lines.append("</details>")
    lines.append("")

    return "\n".join(lines)


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate a Markdown summary from a combined test environment log."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", "-i", help="Single combined log file (normal summary)")
    group.add_argument("--baseline", help="Baseline log file (for comparison)")
    parser.add_argument("--target", help="Target log file (required with --baseline)")
    parser.add_argument("--output", "-o", required=True, help="Output Markdown file")

    args = parser.parse_args()

    if args.input:
        # Single mode
        raw = Path(args.input).read_text()
        cleaned = strip_command_lines(raw)
        collect = parse_collect_env_output(cleaned)
        env = parse_printenv_output(cleaned)
        markdown = generate_single_summary(collect, env, raw)
    else:
        # Comparison mode
        if not args.target:
            parser.error("--target is required when --baseline is used")
        raw_base = Path(args.baseline).read_text()
        raw_tgt = Path(args.target).read_text()

        cleaned_base = strip_command_lines(raw_base)
        cleaned_tgt = strip_command_lines(raw_tgt)

        collect_base = parse_collect_env_output(cleaned_base)
        collect_tgt = parse_collect_env_output(cleaned_tgt)
        env_base = parse_printenv_output(cleaned_base)
        env_tgt = parse_printenv_output(cleaned_tgt)

        markdown = generate_comparison_summary(
            collect_base, collect_tgt,
            env_base, env_tgt,
            raw_base, raw_tgt
        )

    Path(args.output).write_text(markdown)
    print(f"Summary written to {args.output}")


if __name__ == "__main__":
    main()
