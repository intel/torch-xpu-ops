#!/usr/bin/env python3
"""Build PyTorch XPU wheel on Windows using PyTorch's upstream build scripts.

Runs the upstream pipeline (build_env_setup.py, build_install_deps.py,
build_wheel.py) in-process, avoiding cmd.exe quoting/path issues.

Assumes PyTorch source is already prepared (via prepare_pytorch.py) at
WORKSPACE/pytorch.

Environment variables (set by caller):
    WORKSPACE   - workspace root (parent of pytorch/)
    XPU_VERSION - XPU bundle version (for xpu_install.bat)
"""

import os
import subprocess
import sys
from pathlib import Path


def main():
    workspace = Path(os.environ["WORKSPACE"])
    pytorch_root = workspace / "pytorch"
    os.chdir(pytorch_root)

    win_ci_dir = pytorch_root / ".ci" / "pytorch" / "windows"
    sys.path.insert(0, str(win_ci_dir))

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
    sys.path.insert(0, str(pytorch_root / ".github" / "scripts"))
    from generate_binary_build_matrix import PYTORCH_EXTRA_INSTALL_REQUIREMENTS

    os.environ["PYTORCH_EXTRA_INSTALL_REQUIREMENTS"] = (
        PYTORCH_EXTRA_INSTALL_REQUIREMENTS["xpu"]
    )

    # Step 1: Environment setup (sources oneAPI + vcvars)
    print("=== Step 1: build_env_setup ===", flush=True)
    import build_env_setup

    vc_year = os.environ.get("VC_YEAR", "2022")
    env_out = {**build_env_setup.COMMON_BUILD_ENV}

    vcvarsall = build_env_setup.find_vcvarsall(vc_year)
    vs_install_dir = vcvarsall.parents[3]
    env_out.update(build_env_setup.setup_xpu(vs_install_dir))

    mkl_prefix = str(Path(sys.prefix) / "Library")
    device_prefix = env_out.get("CMAKE_PREFIX_PATH", "")
    env_out["CMAKE_PREFIX_PATH"] = (
        f"{mkl_prefix};{device_prefix}" if device_prefix else mkl_prefix
    )
    os.environ.update(env_out)

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
    # Install libuv via conda (upstream uses 7z which may not be on our runners)
    conda_exe = os.environ.get("CONDA_EXE", "conda")
    subprocess.run([conda_exe, "install", "-y", "libuv"], check=True)
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    os.environ["libuv_ROOT"] = str(Path(conda_prefix) / "Library")
    print(f"libuv_ROOT={os.environ['libuv_ROOT']}", flush=True)
    print("build_install_deps complete", flush=True)

    # Step 3: Build wheel
    print("=== Step 3: build_wheel ===", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "build"], check=True)
    dist_dir = pytorch_root / "dist"
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
    print("Build completed successfully", flush=True)


if __name__ == "__main__":
    main()
