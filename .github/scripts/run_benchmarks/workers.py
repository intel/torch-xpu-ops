"""CPU topology detection and worker (GPU/CPU) generation."""

import os
import re
import subprocess
import sys
from functools import lru_cache

from .config import IS_WINDOWS
from .log import log


@lru_cache(maxsize=1)
def get_cpu_topology() -> int:
    """Return number of physical cores."""
    if IS_WINDOWS:
        return _get_cpu_topology_windows()
    return _get_cpu_topology_linux()


def _get_cpu_topology_windows() -> int:
    """Return number of physical cores on Windows via wmic or os.cpu_count fallback."""
    try:
        output = subprocess.check_output(["wmic", "cpu", "get", "NumberOfCores"], text=True)
        for line in output.splitlines():
            line = line.strip()
            if line.isdigit():
                physical_cores = int(line)
                logical = os.cpu_count() or physical_cores
                log(f"CPU topology (Windows): {logical} logical, {physical_cores} physical cores")
                return physical_cores
    except Exception:
        pass
    logical = os.cpu_count() or 1
    physical_cores = max(logical // 2, 1)
    log(f"CPU topology (Windows fallback): {logical} logical, ~{physical_cores} physical cores")
    return physical_cores


def _parse_cpu_range(range_str: str) -> int:
    """Count CPUs from a range string like '0-7,12-19'."""
    total = 0
    for seg in range_str.split(','):
        seg = seg.strip()
        if '-' in seg:
            lo, hi = map(int, seg.split('-'))
            total += hi - lo + 1
        else:
            total += 1
    return total


def _get_cpu_topology_linux() -> int:
    """Return number of physical cores on Linux via lscpu."""
    try:
        output = subprocess.check_output(["lscpu"], text=True)
    except Exception as e:
        sys.exit(f"ERROR: Could not run lscpu: {e}")

    fields: dict[str, str] = {}
    for line in output.splitlines():
        if ':' in line:
            key, _, val = line.partition(':')
            fields[key.strip()] = val.strip()

    try:
        online_cpus = _parse_cpu_range(fields["On-line CPU(s) list"])
        threads_per_core = int(fields["Thread(s) per core"])
    except (KeyError, ValueError) as e:
        sys.exit(f"ERROR: Could not parse CPU topology: {e}")

    physical_cores = online_cpus // threads_per_core
    log(f"CPU topology: {online_cpus} logical, {threads_per_core} threads/core → {physical_cores} physical cores")
    return physical_cores


def _cores_to_range_str(cores: list[int]) -> str:
    """Convert a sorted list of core IDs to a compact range string like '0-7,12-19'."""
    if not cores:
        return ""
    ranges: list[str] = []
    start = prev = cores[0]
    for c in cores[1:]:
        if c == prev + 1:
            prev = c
        else:
            ranges.append(f"{start}-{prev}" if start != prev else str(start))
            start = prev = c
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(ranges)


def _parse_ze_affinity_mask(num_gpus: int) -> list[int]:
    """Parse ZE_AFFINITY_MASK into a list of GPU indices, defaulting to 0..num_gpus-1."""
    ze_mask = os.environ.get("ZE_AFFINITY_MASK", "")
    if not ze_mask:
        log(f"ZE_AFFINITY_MASK not set — using all {num_gpus} GPUs")
        return list(range(num_gpus))

    gpu_list: list[int] = []
    try:
        for part in ze_mask.split(','):
            part = part.strip()
            if not part:
                continue
            if '-' in part:
                lo, hi = part.split('-', maxsplit=1)
                gpu_list.extend(range(int(lo), int(hi) + 1))
            else:
                gpu_list.append(int(part))
    except (ValueError, IndexError) as e:
        sys.exit(f"ERROR: Invalid ZE_AFFINITY_MASK format '{ze_mask}': {e}")

    if not gpu_list:
        sys.exit("ERROR: ZE_AFFINITY_MASK produced no GPUs")
    log(f"ZE_AFFINITY_MASK → GPUs {gpu_list}")
    return gpu_list


def generate_gpu_workers(num_gpus: int) -> list[tuple[int, str, dict]]:
    """Return (card, cmd_prefix, env_vars) per GPU with numactl CPU binding.

    GPU list comes from ZE_AFFINITY_MASK (if set) or 0..num_gpus-1.
    CPU cores are always distributed evenly by num_gpus.
    """
    gpu_list = _parse_ze_affinity_mask(num_gpus)
    physical_cores = get_cpu_topology()
    cores_per_gpu = max(physical_cores // num_gpus, 1)

    workers: list[tuple[int, str, dict]] = []
    for gpu in gpu_list:
        start = gpu * cores_per_gpu
        end = min(start + cores_per_gpu - 1, physical_cores - 1)
        if IS_WINDOWS:
            log(f"  GPU {gpu}: OMP_NUM_THREADS={cores_per_gpu} (numactl N/A on Windows)")
            workers.append((gpu, "", {"OMP_NUM_THREADS": str(cores_per_gpu)}))
        else:
            core_range = f"{start}-{end}"
            log(f"  GPU {gpu}: cores {core_range}, OMP_NUM_THREADS={cores_per_gpu}")
            workers.append((gpu, f"numactl -l -C {core_range}", {"OMP_NUM_THREADS": str(cores_per_gpu)}))
    return workers


def _get_numa_nodes() -> list[list[int]]:
    """Return a list of NUMA nodes, each containing its physical core IDs."""
    if IS_WINDOWS:
        n = get_cpu_topology()
        return [list(range(n))]

    try:
        output = subprocess.check_output(["lscpu", "--parse=CPU,NODE,CORE"], text=True)
    except Exception:
        n = get_cpu_topology()
        return [list(range(n))]

    seen_cores: set[tuple[int, int]] = set()
    node_cpus: dict[int, list[int]] = {}
    for line in output.splitlines():
        if line.startswith('#') or not line.strip():
            continue
        parts = line.split(',')
        if len(parts) < 3:
            continue
        try:
            cpu_id, node_id, core_id = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            continue
        key = (node_id, core_id)
        if key not in seen_cores:
            seen_cores.add(key)
            node_cpus.setdefault(node_id, []).append(cpu_id)

    if not node_cpus:
        n = get_cpu_topology()
        return [list(range(n))]

    result = [sorted(cpus) for _, cpus in sorted(node_cpus.items())]
    for i, cores in enumerate(result):
        log(f"  NUMA node {i}: {len(cores)} physical cores")
    return result


def generate_cpu_workers(cores_per_instance: int | None = None) -> list[tuple[int, str, dict]]:
    """Return (worker_id, cmd_prefix, env_vars) for CPU-only benchmarking.

    Workers are created per NUMA node. If *cores_per_instance* is set,
    total physical cores are split into chunks of that size instead.
    """
    if IS_WINDOWS:
        physical_cores = get_cpu_topology()
        cpi = cores_per_instance or physical_cores
        num_workers = max(physical_cores // cpi, 1)
        workers: list[tuple[int, str, dict]] = []
        for i in range(num_workers):
            log(f"  CPU worker {i}: OMP_NUM_THREADS={cpi} (numactl N/A on Windows)")
            workers.append((i, "", {"OMP_NUM_THREADS": str(cpi)}))
        return workers

    numa_nodes = _get_numa_nodes()

    if cores_per_instance is not None:
        all_cores: list[int] = []
        for cores in numa_nodes:
            all_cores.extend(cores)
        all_cores.sort()
        num_workers = max(len(all_cores) // cores_per_instance, 1)
        workers = []
        for i in range(num_workers):
            start = i * cores_per_instance
            end = min(start + cores_per_instance, len(all_cores))
            chunk = all_cores[start:end]
            if not chunk:
                break
            core_range = _cores_to_range_str(chunk)
            log(f"  CPU worker {i}: cores {core_range}, OMP_NUM_THREADS={len(chunk)}")
            workers.append((i, f"numactl -l -C {core_range}", {"OMP_NUM_THREADS": str(len(chunk))}))
        return workers

    # Default: one worker per NUMA node
    workers = []
    for node_idx, cores in enumerate(numa_nodes):
        core_range = _cores_to_range_str(cores)
        log(f"  CPU worker {node_idx} (NUMA {node_idx}): cores {core_range}, OMP_NUM_THREADS={len(cores)}")
        workers.append((node_idx, f"numactl -m {node_idx} -C {core_range}", {"OMP_NUM_THREADS": str(len(cores))}))
    return workers


def parse_numactl_args(numactl_args: str, num_gpus: int) -> list[tuple[int, str, dict]]:
    """Parse user-provided --numactl-args into worker tuples."""
    prefix_strings = [p.strip().rstrip(';') for p in numactl_args.split(';') if p.strip()]
    if not prefix_strings:
        sys.exit("ERROR: --numactl-args produced no valid prefixes.")
    if len(prefix_strings) < num_gpus:
        prefix_strings += [prefix_strings[-1]] * (num_gpus - len(prefix_strings))
    workers: list[tuple[int, str, dict]] = []
    for i, ps in enumerate(prefix_strings):
        m = re.search(r'ZE_AFFINITY_MASK=(\d+)', ps)
        workers.append((int(m.group(1)) if m else i, ps, {}))
    log(f"Using user-provided NUMACTL_ARGS ({len(workers)} workers)")
    return workers


def get_num_gpus(required: bool = True) -> int | None:
    """Read and validate the NUM_GPUS environment variable."""
    raw = os.getenv('NUM_GPUS')
    if raw is None:
        if required:
            sys.exit("ERROR: Environment variable NUM_GPUS is not set.")
        return None
    try:
        n = int(raw)
    except ValueError:
        sys.exit(f"ERROR: NUM_GPUS must be an integer, got '{raw}'.")
    if n <= 0:
        sys.exit(f"ERROR: NUM_GPUS must be positive, got {n}.")
    return n
