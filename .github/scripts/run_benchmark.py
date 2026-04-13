#!/usr/bin/env python3
"""
PyTorch Dynamo benchmark runner - unified script with automatic result aggregation.
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("dynamo_bench")

# ----------------------------------------------------------------------
# Constants and defaults
# ----------------------------------------------------------------------
DEFAULT_PYTHON_VERSION = "3.12"
UV_INSTALL_URL = "https://astral.sh/uv/install.sh"
UV_INSTALL_DIR_DEFAULT = ".uv"

PYTORCH_REPO = "https://github.com/pytorch/pytorch"
TORCH_XPU_OPS_REPO = "https://github.com/intel/torch-xpu-ops"
BENCHMARK_REPO = "https://github.com/pytorch/benchmark"

VENV_DIR_NAME = "myvenv"
PYTORCH_DIR_NAME = "pytorch-src"
BENCHMARK_REPO_NAME = "benchmark-src"
INDUCTOR_LOG_BASE = "inductor_log"
SUMMARY_CSV_NAME = "summary_results.csv"

ALL_SUITES = ["huggingface", "timm_models", "torchbench"]
ALL_DTYPES = ["float32", "float16", "bfloat16", "amp_bf16", "amp_fp16"]
ALL_MODES = ["inference", "training"]
ALL_SCENARIOS = ["accuracy", "performance"]

# Default package versions (used only when no requirements.txt is given)
DEFAULT_VERSIONS = {
    "numpy": "1.26.4",
    "transformers": "4.55.2",
    "timm": "1.0.19",
    "accelerate": None,
    "pandas": None,
    "psutil": None,
    "scipy": None,
    "requests": None,
    "pyre_extensions": None,
}

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def run_cmd(cmd, desc=None, env=None, cwd=None, check=True, capture=False, shell=False):
    """Run a command and log output."""
    if desc:
        logger.info(desc)
    cmd_str = cmd if isinstance(cmd, str) else " ".join(cmd)
    logger.debug(f"Running: {cmd_str}")
    try:
        if capture:
            result = subprocess.run(
                cmd,
                shell=shell or isinstance(cmd, str),
                env=env,
                cwd=cwd,
                check=check,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()
        else:
            subprocess.run(
                cmd,
                shell=shell or isinstance(cmd, str),
                env=env,
                cwd=cwd,
                check=check,
            )
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        if check:
            sys.exit(e.returncode)


def get_env_version(pkg: str) -> str | None:
    """Return version from environment variable PKG_VERSION."""
    env_var = f"{pkg.upper().replace('-', '_')}_VERSION"
    return os.environ.get(env_var)


def load_requirements(req_path: Path) -> dict[str, str | None]:
    """
    Parse a requirements.txt file and return a dict of {package: version}.
    Lines with extras (e.g., package[extra]==version) are simplified to the base package name.
    """
    versions = {}
    if not req_path or not req_path.exists():
        return versions
    with open(req_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Remove any trailing comments
            if " #" in line:
                line = line[: line.index(" #")].strip()
            # Handle package with version
            if "==" in line:
                pkg, ver = line.split("==", 1)
                # Strip extras like [extra]
                if "[" in pkg:
                    pkg = pkg[: pkg.index("[")]
                versions[pkg] = ver
            elif ">=" in line or "<=" in line or "~=" in line:
                # Not pinning exactly; we'll just take the package name without version
                pkg = line.split()[0]
                if "[" in pkg:
                    pkg = pkg[: pkg.index("[")]
                versions[pkg] = None
            else:
                # No version specifier
                pkg = line
                if "[" in pkg:
                    pkg = pkg[: pkg.index("[")]
                versions[pkg] = None
    return versions


def get_package_spec(pkg: str, versions: dict[str, str | None]) -> str:
    """Return package==version if version is known, else just package name."""
    env_ver = get_env_version(pkg)
    if env_ver:
        return f"{pkg}=={env_ver}"
    ver = versions.get(pkg)
    if ver:
        return f"{pkg}=={ver}"
    return pkg


def find_uv(workspace: Path) -> str:
    """Return path to uv executable, installing it if necessary."""
    uv_in_path = shutil.which("uv")
    if uv_in_path:
        logger.info(f"Using system uv: {uv_in_path}")
        return "uv"

    uv_dir = workspace / UV_INSTALL_DIR_DEFAULT
    uv_bin = uv_dir / "bin"
    uv_exe = uv_bin / "uv"
    if uv_exe.exists():
        logger.info(f"Using workspace uv: {uv_exe}")
        return str(uv_exe)

    logger.info("uv not found, installing into workspace...")
    env = os.environ.copy()
    env["UV_INSTALL_DIR"] = str(uv_bin)
    run_cmd(
        f'curl -LsSf {UV_INSTALL_URL} | sh',
        "Installing uv",
        env=env,
        shell=True,
    )
    if not uv_exe.exists():
        raise RuntimeError("uv installation failed")
    logger.info(f"uv installed to {uv_exe}")
    return str(uv_exe)


def parse_benchmark_entries(path: Path):
    """
    Generator that yields (suite, dtype, mode, model, result) for each valid row in the benchmark file.
    Handles both old (three-column) and new (five-column with optional header) formats.
    """
    if not path.exists():
        return
    with open(path) as f:
        lines = f.readlines()

    if not lines:
        return

    # Check for header (first line starting with "suite" case-insensitive)
    header_skipped = False
    first_line = lines[0].strip()
    if first_line.lower().startswith("suite"):
        header_skipped = True
        lines = lines[1:]

    for line_num, line in enumerate(lines, start=2 if header_skipped else 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        # New format: 5 columns
        if len(parts) == 5:
            suite, dtype, mode, model, result = parts
            yield suite, dtype, mode, model, result
        # Old format: at least 3 columns
        elif len(parts) >= 3:
            suite_dt_mode, model, result = parts[0], parts[1], parts[2]
            # Parse suite_dt_mode (e.g., "torchbench_float32_inference")
            for suite in ALL_SUITES:
                if suite_dt_mode.startswith(suite + "_"):
                    remainder = suite_dt_mode[len(suite) + 1 :]
                    dtype_mode = remainder.rsplit("_", 1)
                    if len(dtype_mode) != 2:
                        logger.warning(f"Line {line_num}: invalid suite_dt_mode format '{suite_dt_mode}'")
                        break
                    dtype, mode = dtype_mode
                    if mode not in ALL_MODES:
                        logger.warning(f"Line {line_num}: unknown mode '{mode}'")
                        break
                    yield suite, dtype, mode, model, result
                    break
            else:
                logger.warning(f"Line {line_num}: unknown suite in '{suite_dt_mode}'")
        else:
            logger.warning(f"Line {line_num}: skipping line with {len(parts)} columns")


# ----------------------------------------------------------------------
# Core classes
# ----------------------------------------------------------------------
@dataclass
class Environment:
    """Manages the Python environment, dependencies, and source checkouts."""

    workspace: Path
    python_version: str
    pytorch_spec: str
    versions: dict[str, str | None] = field(default_factory=lambda: DEFAULT_VERSIONS.copy())
    venv_path: Path = field(init=False)
    pytorch_dir: Path = field(init=False)
    benchmark_dir: Path = field(init=False)
    uv_cmd: str = field(init=False)
    python_exe: Path = field(init=False)        # Path to the Python interpreter to use
    use_venv: bool = field(init=False)          # Whether we created a virtual env

    def __post_init__(self):
        self.workspace = self.workspace.resolve()
        self.venv_path = self.workspace / VENV_DIR_NAME
        self.pytorch_dir = self.workspace / PYTORCH_DIR_NAME
        self.benchmark_dir = self.workspace / BENCHMARK_REPO_NAME
        self.uv_cmd = find_uv(self.workspace)

        # Determine Python interpreter
        if self.python_version:   # --setup-python was used
            self.use_venv = True
            self.python_exe = self.venv_path / "bin" / "python"
        else:
            self.use_venv = False
            # Use system Python
            system_python = shutil.which("python3") or shutil.which("python")
            if not system_python:
                raise RuntimeError("No system Python found (python3/python not in PATH)")
            self.python_exe = Path(system_python)
            logger.info(f"Using system Python: {self.python_exe}")

    def create_venv(self):
        """Create a virtual environment with the specified Python version."""
        if not self.use_venv:
            raise RuntimeError("create_venv called but use_venv is False")
        logger.info("Creating virtual environment")
        run_cmd(
            f"{self.uv_cmd} venv {self.venv_path} --python {self.python_version} --clear",
            "Creating venv",
        )
        self._install_base_packages()

    def _install_base_packages(self):
        """Install pip, setuptools, wheel, numpy."""
        logger.info("Installing base packages")
        run_cmd(
            f"{self.uv_cmd} pip install --python {self.python_exe} pip numpy 'setuptools<81' wheel",
            "Base packages",
        )

    def install_pytorch(self, args):
        """Install PyTorch family from channel, wheels, or local directory."""
        logger.info(f"Installing PyTorch from spec: {self.pytorch_spec}")

        # Local wheel directory?
        if os.path.isdir(self.pytorch_spec):
            wheel_dir = Path(self.pytorch_spec).resolve()
            wheels = list(wheel_dir.glob("*.whl"))
            if not wheels:
                raise ValueError(f"No wheel files in {wheel_dir}")
            cmd = [self.uv_cmd, "pip", "install", "--python", str(self.python_exe)] + [str(w) for w in wheels]
            run_cmd(cmd, "Installing from local wheels")
            return

        # Channel[=version] format
        if "=" in self.pytorch_spec:
            channel, version = self.pytorch_spec.split("=", 1)
            version_spec = f"=={version}"
        else:
            channel = self.pytorch_spec
            version_spec = ""

        channel = channel.lower()
        index_map = {
            "release": f"https://download.pytorch.org/whl/{args.device}",
            "nightly": f"https://download.pytorch.org/whl/nightly/{args.device}",
            "rc": f"https://download.pytorch.org/whl/test/{args.device}",
        }
        if channel not in index_map:
            raise ValueError(f"Unknown channel: {channel}")
        index_url = index_map[channel]

        packages = [
            f"torch{version_spec}",
            "torchvision",
            "torchaudio",
            "torchao",
        ]
        if channel == 'nightly':
            packages.append("--pre")
        cmd = [self.uv_cmd, "pip", "install", "--python", str(self.python_exe)] + packages
        if index_url:
            cmd.extend(["--index-url", index_url])
        run_cmd(cmd, f"Installing PyTorch ({channel}{version_spec})")

    def get_torch_commit(self) -> str:
        """
        Return a tuple (git_commit_hash_of_torch, triton_version).
        Each value is 'unknown' if not available.
        """
        try:
            # Run a single command to get both values.
            # Use double braces to escape f-string literal inside the command.
            cmd = (
                f"{self.python_exe} -c 'import torch; print(torch.version.git_version)'"
            )
            torch_commit = run_cmd(cmd, capture=True).strip()
            cmd = (
                f"{self.uv_cmd} pip list --python {self.python_exe} 2>&1 |grep triton-xpu |tail -n 1 |sed 's/.* //'"
            )
            triton_version = run_cmd(cmd, capture=True).strip()
            return torch_commit, triton_version
        except Exception:  # Catch any error from run_cmd or subprocess
            pass
        # Fallback if anything went wrong
        return "unknown", "unknown"

    def clone_pytorch(self, commit: str):
        """Clone PyTorch at exact commit and set up torch-xpu-ops."""
        if not self.pytorch_dir.exists():
            run_cmd(f"git clone {PYTORCH_REPO} {self.pytorch_dir}", "Cloning PyTorch")
        run_cmd("git reset --hard && git clean -df", cwd=self.pytorch_dir)
        run_cmd("git fetch origin -a", cwd=self.pytorch_dir)
        run_cmd(f"git checkout {commit}", cwd=self.pytorch_dir)
        # run_cmd("rm -rf ./torch", cwd=self.pytorch_dir, check=False)

        # Clone torch-xpu-ops and copy benchmark files
        run_cmd("rm -rf torch-xpu-ops", cwd=self.pytorch_dir, check=False)
        run_cmd(f"git clone {TORCH_XPU_OPS_REPO}", cwd=self.pytorch_dir)
        run_cmd(
            "rsync -avz ./torch-xpu-ops/.ci/benchmarks/ ./benchmarks/dynamo/",
            cwd=self.pytorch_dir,
        )

        # Read pin files from PyTorch CI
        self._read_pin_files()

    def _read_pin_files(self):
        """Read pin files from PyTorch source and store relevant specs."""
        self.hf_req_file = self.pytorch_dir / ".ci/docker/ci_commit_pins/huggingface-requirements.txt"
        self.timm_pin_file = self.pytorch_dir / ".ci/docker/ci_commit_pins/timm.txt"
        self.torchbench_pin_file = self.pytorch_dir / ".ci/docker/ci_commit_pins/torchbench.txt"
        self.timm_pinned_version = None
        self.torchbench_commit = None

        if self.timm_pin_file.exists():
            with open(self.timm_pin_file) as f:
                self.timm_pinned_version = f.read().strip()
        if self.torchbench_pin_file.exists():
            with open(self.torchbench_pin_file) as f:
                self.torchbench_commit = f.read().strip()

    def install_additional_packages(self):
        """Install all remaining Python packages with version control (fallback if no requirements.txt)."""
        logger.info("Installing additional packages (fallback mode)")

        # Install huggingface requirements if available (for transformers, etc.)
        if hasattr(self, 'hf_req_file') and self.hf_req_file.exists():
            logger.info(f"Installing pinned huggingface requirements from {self.hf_req_file}")
            run_cmd(
                f"{self.uv_cmd} pip install --python {self.python_exe} -r {self.hf_req_file}",
                "Installing huggingface pinned requirements",
            )
        else:
            # Fallback to installing transformers & accelerate individually
            run_cmd(
                f"{self.uv_cmd} pip install --python {self.python_exe} {get_package_spec('transformers', self.versions)} {get_package_spec('accelerate', self.versions)}",
                "Installing transformers & accelerate",
            )

        # Install timm with pinned version if available
        timm_spec = get_package_spec('timm', self.versions)
        # If no env/version and we have a pinned version, use that
        if timm_spec == 'timm' and hasattr(self, 'timm_pinned_version') and self.timm_pinned_version:
            timm_spec = f"timm=={self.timm_pinned_version}"
            logger.info(f"Using pinned timm version: {self.timm_pinned_version}")
        run_cmd(
            f"{self.uv_cmd} pip install --python {self.python_exe} {timm_spec}",
            "Installing timm",
        )

        # Continue with other packages
        pkgs = [
            get_package_spec("pandas", self.versions),
            get_package_spec("psutil", self.versions),
            get_package_spec("scipy", self.versions),
            get_package_spec("requests", self.versions),
        ]
        run_cmd(
            f"{self.uv_cmd} pip install --python {self.python_exe} " + " ".join(pkgs),
            "Installing pandas, psutil, scipy, requests",
        )

        # numpy separately (upgrade)
        run_cmd(
            f"{self.uv_cmd} pip install --python {self.python_exe} -U {get_package_spec('numpy', self.versions)}",
            "Installing numpy",
        )

        # pyre-extensions
        run_cmd(
            f"{self.uv_cmd} pip install --python {self.python_exe} {get_package_spec('pyre_extensions', self.versions)}",
            "Installing pyre-extensions",
        )

        # DLRM requirements (skip torch packages)
        dlrm_req = run_cmd(
            "curl -fsSL https://raw.githubusercontent.com/facebookresearch/dlrm/refs/heads/torchrec-dlrm/requirements.txt",
            capture=True,
            shell=True,
        )
        if dlrm_req:
            for line in dlrm_req.splitlines():
                pkg = line.strip()
                if pkg and not pkg.startswith("#") and not pkg.startswith("torch"):
                    run_cmd(
                        f"{self.uv_cmd} pip install --python {self.python_exe} {pkg}",
                        check=False,
                    )

        # custom gym fork (optional)
        run_cmd(
            f"{self.uv_cmd} pip install --python {self.python_exe} git+https://github.com/nocoding03/gym@fix-np",
            "Installing custom gym",
            check=False,
        )

    def install_from_requirements(self, req_file: Path):
        """Install all packages listed in a requirements.txt file."""
        logger.info(f"Installing packages from {req_file}")
        run_cmd(
            f"{self.uv_cmd} pip install --python {self.python_exe} -r {req_file}",
            "Installing from requirements.txt",
        )

    def install_benchmark_repo(self, torchbench_models: list[str] | None = None, torchbench_commit: str | None = None):
        """
        Clone benchmark repo and optionally install only specified TorchBench models.
        If torchbench_models is None, the entire suite is installed.
        If torchbench_commit is provided, checkout that commit after cloning.
        """
        if self.benchmark_dir.exists():
            shutil.rmtree(self.benchmark_dir)
        run_cmd(f"git clone {BENCHMARK_REPO} {self.benchmark_dir}", "Cloning benchmark repo")

        if torchbench_commit:
            logger.info(f"Checking out torchbench at commit {torchbench_commit}")
            run_cmd(f"git checkout {torchbench_commit}", cwd=self.benchmark_dir)

        if torchbench_models is not None:
            # Install either specific models or all (if list is empty)
            install_cmd = [str(self.python_exe), "install.py", "--continue_on_fail"]
            if torchbench_models:  # non-empty list
                install_cmd.extend(torchbench_models)
                desc = f"Installing TorchBench models: {', '.join(torchbench_models)}"
            else:
                desc = "Installing all TorchBench models"
            run_cmd(install_cmd, cwd=self.benchmark_dir, desc=desc)
        else:
            logger.info("Benchmark repo cloned but no model installation requested.")

    def prepare_pythonpath(self):
        """Set PYTHONPATH to include benchmark repo if it exists."""
        if self.benchmark_dir.exists():
            pythonpath = str(self.benchmark_dir) + os.pathsep + os.environ.get("PYTHONPATH", "")
            os.environ["PYTHONPATH"] = pythonpath
            logger.debug(f"PYTHONPATH set to include {self.benchmark_dir}")
        else:
            logger.debug("Benchmark repo not present, not setting PYTHONPATH")


@dataclass
class BenchmarkSpec:
    """One benchmark run specification."""
    suite: str
    dtype: str
    mode: str
    scenario: str
    model: str


class BenchmarkRunner:
    """Executes benchmarks based on command line arguments."""

    def __init__(self, args, env: Environment):
        self.args = args
        self.env = env
        self.log_base = env.workspace / INDUCTOR_LOG_BASE

    def run_all(self):
        """Main entry point: run from file or expanded combinations."""
        if self.args.from_file:
            specs = self._parse_benchmark_file(Path(self.args.from_file))
        else:
            specs = self._expand_combinations()

        logger.info(f"Preparing to run {len(specs)} benchmark(s)")
        for i, spec in enumerate(specs, 1):
            logger.info(f"[{i}/{len(specs)}] Running {spec}")
            self._run_single(spec)

        # After all runs, generate summary CSV
        self._generate_summary()

    def _expand_combinations(self) -> list[BenchmarkSpec]:
        suites = self._expand_arg(self.args.suite, ALL_SUITES)
        dtypes = self._expand_arg(self.args.dt, ALL_DTYPES)
        modes = self._expand_arg(self.args.mode, ALL_MODES)
        scenarios = self._expand_arg(self.args.scenario, ALL_SCENARIOS)
        specs = []
        for s in suites:
            for d in dtypes:
                for m in modes:
                    for sc in scenarios:
                        specs.append(BenchmarkSpec(s, d, m, sc, self.args.model_only or ""))
        return specs

    @staticmethod
    def _expand_arg(value, all_list):
        return all_list if value == "all" else [value]

    def _parse_benchmark_file(self, path: Path) -> list[BenchmarkSpec]:
        specs = []
        for suite, dtype, mode, model, result in parse_benchmark_entries(path):
            # Determine scenario from result: if it's a number (possibly negative, decimal) -> performance
            if re.match(r'^-?\d+(\.\d+)?$', result.strip()):
                scenario = "performance"
            else:
                scenario = "accuracy"
            specs.append(BenchmarkSpec(suite, dtype, mode, scenario, model))
        return specs

    def _run_single(self, spec: BenchmarkSpec):
        """Execute one benchmark and augment its CSV."""
        log_dir = self.log_base / torch_commit / spec.suite / spec.dtype
        log_dir.mkdir(parents=True, exist_ok=True)

        cmd, csv_path = self._build_command(spec, log_dir)
        logger.info("Checking: torch and triton")
        torch_check, triton_check = self.env.get_torch_commit()
        logger.info(f"Torch: {torch_check}, Triton: {triton_check}")
        if torch_commit != torch_check or triton_version != triton_check:
            logger.error(f"Torch or Triton has been re-installed! Torch: {torch_commit}/{torch_check}, Triton: {triton_version}/{triton_check}")
            sys.exit(1)
        run_cmd('sudo rm -rf ~/.triton/ /tmp/torch* /tmp/tmp*', shell=True, check=False)
        run_cmd(cmd, cwd=self.env.pytorch_dir, shell=True, check=False)  # Do not exit on benchmark failure

        # If the benchmark did not produce the expected CSV, create a fallback
        if not csv_path.exists():
            self._create_fallback_csv(csv_path, spec)

        # Augment the CSV by adding suite, dt, mode, scenario columns at the front
        self._augment_csv(csv_path, spec)

    def _create_fallback_csv(self, csv_path: Path, spec: BenchmarkSpec):
        """
        Create a minimal CSV indicating the benchmark crashed.
        The columns match the typical output of the benchmark script.
        """
        logger.warning(f"Benchmark did not produce CSV, creating fallback: {csv_path}")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        # Columns expected in a normal benchmark CSV (based on example)
        fieldnames = [
            'dev', 'name', 'batch_size', 'accuracy',
            'calls_captured', 'unique_graphs', 'graph_breaks', 'unique_graph_breaks',
            'autograd_captures', 'autograd_compiles', 'cudagraph_skips', 'compilation_latency'
        ]
        with open(csv_path, 'w+', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            row = {
                'dev': self.args.device,
                'name': spec.model,
                'batch_size': 0,
                'accuracy': 'crash',
                'calls_captured': 0,
                'unique_graphs': 0,
                'graph_breaks': 0,
                'unique_graph_breaks': 0,
                'autograd_captures': 0,
                'autograd_compiles': 0,
                'cudagraph_skips': 0,
                'compilation_latency': 0,
            }
            writer.writerow(row)

    def _build_command(self, spec: BenchmarkSpec, log_dir: Path) -> tuple[str, Path]:
        """Build shell command string and return output CSV path."""
        # Determine dtype and extra flags
        real_dt = spec.dtype
        dt_extra = []
        if spec.dtype == "amp_bf16":
            real_dt = "amp"
            dt_extra = ["--amp-dtype", "bfloat16"]
        elif spec.dtype == "amp_fp16":
            real_dt = "amp"
            dt_extra = ["--amp-dtype", "float16"]

        mode_extra = ["--training"] if spec.mode == "training" else ["--inference"]
        shape_extra = (
            ["--dynamic-shapes", "--dynamic-batch-only"]
            if self.args.shape == "dynamic"
            else []
        )
        partition_flags = (
            [
                "--total-partitions",
                str(self.args.num_shards),
                "--partition-id",
                str(self.args.shard_id),
            ]
            if self.args.num_shards and self.args.shard_id is not None
            else []
        )
        model_extra = []
        if spec.model:
            if " -k " in spec.model:
                model_extra = [spec.model]
            else:
                model_extra = ["--only", spec.model]

        # Output files
        timestamp = datetime.now().timestamp()
        csv_name = f"inductor_{spec.suite}_{spec.dtype}_{spec.mode}_{self.args.device}_{timestamp}_{spec.scenario}.csv"
        csv_path = log_dir / csv_name
        log_path = log_dir / csv_name.replace(".csv", ".log")

        script = self.env.pytorch_dir / "benchmarks" / "dynamo" / f"{spec.suite}.py"
        cmd_parts = [
            str(self.env.python_exe),
            str(script),
            f"--{spec.scenario}",
            f"--{real_dt}",
            *dt_extra,
            *mode_extra,
            "-d",
            self.args.device,
            "-n",
            str(self.args.iterations),
            *shape_extra,
            *partition_flags,
            *model_extra,
            "--backend=inductor",
            "--cold-start-latency",
            "--timeout",
            str(self.args.timeout),
            "--disable-cudagraphs",
            "--output",
            str(csv_path),
        ]
        cmd_str = " ".join(cmd_parts) + f" 2>&1 | tee -a {log_path}"
        return cmd_str, csv_path

    def _augment_csv(self, csv_path: Path, spec: BenchmarkSpec):
        """
        Add suite, dt, mode, scenario columns at the beginning of the CSV.
        Also filter out rows that do not have the expected device.
        """
        if not csv_path.exists():
            logger.warning(f"CSV not found: {csv_path}")
            return
        try:
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if not rows:
                    logger.warning(f"Empty CSV: {csv_path}")
                    return
                base_fields = reader.fieldnames
                new_cols = ['suite', 'dt', 'mode', 'scenario']
                # Only add columns that are not already present
                cols_to_add = [col for col in new_cols if col not in base_fields]
                fieldnames = cols_to_add + base_fields

            # Add the new columns to every row
            for row in rows:
                row['suite'] = spec.suite
                row['dt'] = spec.dtype
                row['mode'] = spec.mode
                row['scenario'] = spec.scenario

            # ---- NEW: Filter out rows with incorrect device ----
            expected_device = self.args.device
            original_rows = rows[:]  # keep a copy for fallback
            rows = [row for row in rows if row.get('dev') == expected_device]
            if not rows:
                logger.warning(f"No rows with device '{expected_device}' found in {csv_path}; keeping all rows.")
                rows = original_rows
            else:
                logger.debug(f"Filtered {len(original_rows) - len(rows)} malformed row(s) from {csv_path}")
            # ----------------------------------------------------

            with open(csv_path, 'w+', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            logger.debug(f"Augmented {csv_path} (columns inserted at front)")
        except Exception as e:
            logger.error(f"Failed to augment {csv_path}: {e}")

    def _generate_summary(self):
        """Collect all individual CSVs and write a summary CSV with a fixed schema."""
        logger.info("Generating summary CSV...")
        csv_files = list(self.log_base.rglob("*.csv"))
        if not csv_files:
            logger.warning("No CSV files found to combine.")
            return

        # Filter out files we want to exclude
        filtered_files = []
        for f in csv_files:
            if not f.name.endswith(('_accuracy.csv', '_performance.csv')):
                continue
            filtered_files.append(f)

        if not filtered_files:
            logger.warning("No relevant CSV files after filtering.")
            return

        # Define the target columns for the summary CSV
        target_fieldnames = [
            'suite', 'dt', 'mode', 'scenario', 'dev', 'name', 'batch_size',
            'accuracy', 'eager_latency', 'inductor_latency', 'speedup',
            'torch_commit', 'triton_version', 'shape'
        ]

        summary_rows = []
        for csv_file in filtered_files:
            try:
                with open(csv_file) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Build a new row with only the target columns
                        new_row = dict.fromkeys(target_fieldnames, '')

                        # Copy values that exist in the original row
                        for key in row:
                            if key in new_row:
                                new_row[key] = row[key]

                        # Add metadata
                        new_row['torch_commit'] = torch_commit
                        new_row['triton_version'] = triton_version
                        new_row['shape'] = 'dynamic' if self.args.shape == 'dynamic' else 'static'

                        # Compute derived latency fields if this is a performance run
                        scenario = new_row.get('scenario', '')
                        if scenario == 'performance':
                            abs_latency = row.get('abs_latency', '')
                            speedup = row.get('speedup', '')
                            # Try to compute eager_latency = abs_latency * speedup
                            try:
                                if abs_latency and speedup:
                                    eager = float(abs_latency) * float(speedup)
                                    new_row['eager_latency'] = str(eager)
                            except (ValueError, TypeError):
                                pass
                            # inductor_latency is abs_latency
                            new_row['inductor_latency'] = abs_latency
                            # speedup is already copied
                        # For accuracy runs, eager_latency and inductor_latency remain empty

                        summary_rows.append(new_row)
            except Exception as e:
                logger.error(f"Error processing {csv_file}: {e}")

        if not summary_rows:
            logger.warning("No data extracted.")
            return

        # Sort rows by scenario, suite, mode, dt (in that order)
        summary_rows.sort(key=lambda r: (
            r.get('scenario', ''),
            r.get('suite', ''),
            r.get('mode', ''),
            r.get('dt', ''),
            r.get('name', ''),
            r.get('batch_size', '')
        ))

        summary_path = self.log_base / SUMMARY_CSV_NAME
        with open(summary_path, "w+", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=target_fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

        logger.info(f"Summary written to {summary_path}")


def check_torchbench_needed(args) -> tuple[bool, list[str] | None]:
    """
    Determine whether TorchBench models are needed and, if so,
    return (True, list_of_models) where list_of_models may be None (install all)
    or a list of specific model names.
    """
    if args.from_file:
        models = set()
        for suite, _, _, model, _ in parse_benchmark_entries(args.from_file):
            if suite == "torchbench":
                models.add(model)
        if models:
            return True, list(models)
        else:
            return False, None

    # Existing logic for command-line arguments
    suites = []
    if args.suite == "all":
        suites = ALL_SUITES
    elif args.suite in ALL_SUITES:
        suites = [args.suite]
    else:
        suites = []

    if "torchbench" in suites:
        return True, None

    if args.model_only and " -k " not in args.model_only:
        return True, [args.model_only]

    return False, None


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Unified PyTorch Dynamo benchmark script with automatic result aggregation."
    )
    parser.add_argument("--workspace", default=".", help="Base directory for all artifacts (default: current)")

    # Setup flags
    parser.add_argument("--setup-python", nargs="?", const=DEFAULT_PYTHON_VERSION, default=None,
                        help="Create Python virtual environment. Optionally specify Python version (default: 3.12). If omitted, system Python is used.")
    parser.add_argument("--setup-deps", action="store_true",
                        help="Install PyTorch, dependencies, and clone repositories (benchmark repo only if TorchBench is needed).")
    parser.add_argument("--pytorch", default="release",
                        help="PyTorch source: channel[=version] (e.g., release, nightly=2.1.0.dev) or path to directory containing .whl files. (Used with --setup-deps)")
    parser.add_argument("--config", type=Path,
                        help="Path to a requirements.txt file specifying additional dependencies. (Used with --setup-deps)")

    # Benchmark parameters
    parser.add_argument("--suite", default="huggingface",
                        help="Suite: huggingface, timm_models, torchbench, or 'all' (ignored if --from-file used)")
    parser.add_argument("--dt", default="float32",
                        help="Data type: float32, float16, bfloat16, amp_bf16, amp_fp16, or 'all' (ignored if --from-file used)")
    parser.add_argument("--mode", default="inference",
                        help="Mode: inference, training, or 'all' (ignored if --from-file used)")
    parser.add_argument("--scenario", default="accuracy",
                        help="Scenario: accuracy, performance, or 'all' (ignored if --from-file used)")
    parser.add_argument("--model-only",
                        help="Specific model name or pytest -k expression (ignored if --from-file used)")
    parser.add_argument("--from-file", type=Path,
                        help="Path to a text file with benchmark specifications (overrides suite/dt/mode/scenario/model-only)")

    parser.add_argument("--device", default="xpu",
                        help="Device: xpu or cuda")
    parser.add_argument("--shape", default="static", choices=["static", "dynamic"],
                        help="Shape: static or dynamic")
    parser.add_argument("--num-shards", type=int,
                        help="Number of shards for parallel execution")
    parser.add_argument("--shard-id", type=int,
                        help="Shard ID (0-based) when using sharding")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of iterations (default: 10)")
    parser.add_argument("--timeout", type=int, default=10800,
                        help="Timeout in seconds (default: 10800)")

    args = parser.parse_args()
    workspace = Path(args.workspace).resolve()
    global torch_commit, triton_version

    # Determine if we will run benchmarks (to know if we need benchmark repo)
    need_torchbench, torchbench_models = check_torchbench_needed(args)

    # ------------------------------------------------------------------
    # 1. Setup Python environment (if requested)
    # ------------------------------------------------------------------
    if args.setup_python is not None:
        logger.info("=== Setting up Python environment ===")
        versions = DEFAULT_VERSIONS.copy()
        if args.config and args.config.exists():
            req_versions = load_requirements(args.config)
            versions.update(req_versions)
        env = Environment(workspace, args.setup_python, args.pytorch, versions)
        env.create_venv()
    else:
        logger.info("=== Using system Python ===")
        # Create environment without creating venv; detect system python inside Environment
        env = Environment(workspace, "", "")
        # Check that python_exe is valid
        if not env.python_exe.exists():
            logger.error(f"System Python not found: {env.python_exe}")
            sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Install dependencies (if requested)
    # ------------------------------------------------------------------
    if args.setup_deps or args.setup_python is not None:
        logger.info("=== Installing dependencies ===")
        if args.setup_python is None:
            # Ensure versions are loaded for system python case
            versions = DEFAULT_VERSIONS.copy()
            if args.config and args.config.exists():
                req_versions = load_requirements(args.config)
                versions.update(req_versions)
            env.versions = versions

        # Check Python availability (should be set)
        if not env.python_exe.exists():
            logger.error(f"Python interpreter not found: {env.python_exe}")
            sys.exit(1)

        env.install_pytorch(args)
        torch_commit, triton_version = env.get_torch_commit()
        env.clone_pytorch(torch_commit)

        # Install additional packages
        if args.config and args.config.exists():
            env.install_from_requirements(args.config)
        else:
            env.install_additional_packages()

        # Clone benchmark repo only if TorchBench is needed
        if need_torchbench:
            logger.info("TorchBench models required - cloning benchmark repository.")
            env.install_benchmark_repo(torchbench_models, torchbench_commit=env.torchbench_commit if hasattr(env, 'torchbench_commit') else None)
        else:
            logger.info("TorchBench not required - skipping benchmark repo clone.")
    else:
        torch_commit, triton_version = env.get_torch_commit()

    # ------------------------------------------------------------------
    # 3. Run benchmarks (if requested implicitly)
    # ------------------------------------------------------------------
    ran_benchmarks = False
    if True:  # Always try to run benchmarks if args are provided
        logger.info("=== Starting benchmarks ===")
        if not env.python_exe.exists():
            logger.error(f"Python interpreter not found: {env.python_exe}")
            sys.exit(1)
        if not env.pytorch_dir.exists():
            logger.error(f"PyTorch source not found at {env.pytorch_dir}. Run with --setup-deps first.")
            sys.exit(1)

        # If TorchBench is needed but benchmark repo is missing, error
        if need_torchbench and not env.benchmark_dir.exists():
            logger.error("TorchBench models are required but benchmark repository is not cloned. Run --setup-deps with appropriate arguments.")
            sys.exit(1)

        env.prepare_pythonpath()
        runner = BenchmarkRunner(args, env)
        runner.run_all()
        ran_benchmarks = True

    if not (args.setup_python or args.setup_deps or ran_benchmarks):
        parser.print_help()


if __name__ == "__main__":
    main()
