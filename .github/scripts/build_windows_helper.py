#!/usr/bin/env python3
"""Windows build helper: runs PyTorch's upstream build pipeline in-process.

This avoids cmd.exe quoting/path issues by keeping everything in Python.
The upstream scripts (build_env_setup.py, build_install_deps.py, build_wheel.py)
are designed to emit env to a file for bash to source. Since we're in cmd,
we instead run them in-process and apply env changes directly.

Must be run from the pytorch source root (WORKSPACE/pytorch).
"""

import os
import subprocess
import sys
from pathlib import Path

PYTORCH_ROOT = Path.cwd()
WIN_CI_DIR = PYTORCH_ROOT / ".ci" / "pytorch" / "windows"

# Add the Windows CI dir to path so we can import helpers
sys.path.insert(0, str(WIN_CI_DIR))


def main():
    # Set required env vars for XPU build
    os.environ["GPU_ARCH_TYPE"] = "xpu"
    os.environ["DESIRED_CUDA"] = "xpu"
    os.environ["CUDA_VERSION"] = "xpu"
    os.environ["USE_SCCACHE"] = "0"
    os.environ["USE_XCCL"] = "0"
    os.environ["TORCH_XPU_ARCH_LIST"] = "mtl-h,bmg,lnl-m"
    os.environ["SKIP_SETUP_CLEAN"] = "1"
    os.environ["WERROR"] = "1"

    # Set PYTORCH_EXTRA_INSTALL_REQUIREMENTS from upstream source
    sys.path.insert(0, str(PYTORCH_ROOT / ".github" / "scripts"))
    from generate_binary_build_matrix import PYTORCH_EXTRA_INSTALL_REQUIREMENTS

    os.environ["PYTORCH_EXTRA_INSTALL_REQUIREMENTS"] = (
        PYTORCH_EXTRA_INSTALL_REQUIREMENTS["xpu"]
    )

    # Step 1: Environment setup
    print("=== Step 1: build_env_setup ===", flush=True)
    import build_env_setup

    # Simulate the main() logic but apply env directly
    vc_year = os.environ.get("VC_YEAR", "2022")
    env_out = {**build_env_setup.COMMON_BUILD_ENV}

    vcvarsall = build_env_setup.find_vcvarsall(vc_year)
    vs_install_dir = vcvarsall.parents[3]
    env_out.update(build_env_setup.setup_xpu(vs_install_dir))

    # MKL prefix
    mkl_prefix = str(Path(sys.prefix) / "Library")
    device_prefix = env_out.get("CMAKE_PREFIX_PATH", "")
    env_out["CMAKE_PREFIX_PATH"] = (
        f"{mkl_prefix};{device_prefix}" if device_prefix else mkl_prefix
    )

    os.environ.update(env_out)

    # vcvars
    print(f"Sourcing {vcvarsall}", flush=True)
    vsdevcmd_args = os.environ.get("VSDEVCMD_ARGS", "")
    vcvars_env = build_env_setup.capture_vcvars_env(
        vcvarsall, f"x64 {vsdevcmd_args}".strip()
    )
    os.environ.update(vcvars_env)
    print("build_env_setup complete", flush=True)

    # Step 2: Install build dependencies
    print("=== Step 2: build_install_deps ===", flush=True)
    import build_install_deps

    build_install_deps.pip_install("-q", f"numpy=={build_install_deps.numpy_pin()}")
    build_install_deps.pip_install("-q", *build_install_deps.PIP_PACKAGES)
    # Install libuv via conda (upstream uses 7z which may not be available)
    subprocess.run(["conda", "install", "-y", "libuv"], check=True)
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    os.environ["libuv_ROOT"] = str(Path(conda_prefix) / "Library")
    print(f"libuv_ROOT={os.environ['libuv_ROOT']}", flush=True)
    print("build_install_deps complete", flush=True)

    # Step 3: Build wheel
    print("=== Step 3: build_wheel ===", flush=True)
    dist_dir = PYTORCH_ROOT / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(dist_dir),
        ],
        check=True,
    )
    print("build_wheel complete", flush=True)


if __name__ == "__main__":
    main()
