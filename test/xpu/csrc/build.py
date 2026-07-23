#!/usr/bin/env python3
"""
Build XPU custom ops as shared libraries for Python usage.

Usage:
    python build.py          # build all .so outputs
    python build.py clean    # remove build artifacts
"""

import os
import shlex
import subprocess
import sys
import sysconfig

import torch


def get_build_config():
    """Gather all paths and flags needed for building."""
    torch_dir = os.path.dirname(torch.__file__)
    torch_include = os.path.join(torch_dir, "include")
    torch_include_csrc = os.path.join(torch_include, "torch", "csrc", "api", "include")
    torch_lib = os.path.join(torch_dir, "lib")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    project_src = os.path.join(project_root, "src")

    cxx_abi = "1" if torch._C._GLIBCXX_USE_CXX11_ABI else "0"

    python_include = sysconfig.get_path("include")

    return {
        "torch_include": torch_include,
        "torch_include_csrc": torch_include_csrc,
        "torch_lib": torch_lib,
        "project_src": project_src,
        "cxx_abi": cxx_abi,
        "python_include": python_include,
    }


def get_mpi_flags():
    try:
        output = subprocess.check_output(
            ["pkg-config", "--cflags", "--libs", "impi"],
            text=True,
        ).strip()
        if output:
            return shlex.split(output)
    except (OSError, subprocess.CalledProcessError):
        pass
    return [
        "-I/opt/intel/oneapi/mpi/2021.18/include",
        "-L/opt/intel/oneapi/mpi/2021.18/lib",
        "-lmpi",
        "-lmpicxx",
        "-lmpifort",
    ]


def get_ishmem_extra_link_flags():
    flags = []
    for pkg in ("hwloc",):
        try:
            output = subprocess.check_output(
                ["pkg-config", "--libs", pkg],
                text=True,
            ).strip()
            if output:
                flags.extend(shlex.split(output))
        except (OSError, subprocess.CalledProcessError):
            pass

    # Match ISHMEM's own test binaries: static libishmem.a may still need these
    # system libraries resolved when loaded as a Python extension.
    for flag in ("-ldl", "-lrt", "-lpthread"):
        if flag not in flags:
            flags.append(flag)
    return flags


def get_ishmem_config():
    env_root = os.environ.get("ISHMEM_HOME") or os.environ.get("ISHMEM_ROOT")
    candidates = [
        env_root,
        "/root/cherry/ishmem_ws/ishmem_ibgda/build/_install",
        "/root/cherry/ishmem_ws/ishmem_ibgda/build",
    ]
    for root in candidates:
        if not root:
            continue
        include_dir = os.path.join(root, "include")
        lib_dir = os.path.join(root, "lib")
        static_lib = os.path.join(lib_dir, "libishmem.a")
        if os.path.isdir(include_dir) and os.path.exists(static_lib):
            return {
                "root": root,
                "include_dir": include_dir,
                "lib_dir": lib_dir,
                "static_lib": static_lib,
            }
    raise RuntimeError(
        "Unable to locate ISHMEM install. Set ISHMEM_HOME or ISHMEM_ROOT to a "
        "prefix containing include/ and lib/libishmem.a."
    )


def build_one(cfg, src_name, out_name, label):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(base_dir, src_name)
    out = os.path.join(base_dir, out_name)

    cmd = [
        "icpx",
        "-fsycl",
        "-std=c++17",
        "-shared",
        "-fPIC",
        "-O2",
        f"-D_GLIBCXX_USE_CXX11_ABI={cfg['cxx_abi']}",
        f"-I{cfg['torch_include']}",
        f"-I{cfg['torch_include_csrc']}",
        f"-I{cfg['project_src']}",
        f"-I{cfg['python_include']}",
        f"-L{cfg['torch_lib']}",
        "-ltorch",
        "-ltorch_cpu",
        "-lc10",
        "-Wl,-rpath," + cfg["torch_lib"],
        src,
        "-o",
        out,
    ]

    print(f"Building {label} extension...")
    print(" ".join(cmd))
    subprocess.check_call(cmd)
    print(f"Build successful: {out}\n")
    return out


def build_one_ishmem(cfg, ishmem_cfg, src_name, out_name, label):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(base_dir, src_name)
    out = os.path.join(base_dir, out_name)

    cmd = [
        "icpx",
        "-fsycl",
        "-std=c++17",
        "-shared",
        "-fPIC",
        "-O2",
        f"-D_GLIBCXX_USE_CXX11_ABI={cfg['cxx_abi']}",
        f"-I{cfg['torch_include']}",
        f"-I{cfg['torch_include_csrc']}",
        f"-I{cfg['project_src']}",
        f"-I{cfg['python_include']}",
        f"-I{ishmem_cfg['include_dir']}",
        f"-L{cfg['torch_lib']}",
        "-ltorch",
        "-ltorch_cpu",
        "-lc10",
        "-Wl,-rpath," + cfg["torch_lib"],
        src,
        ishmem_cfg["static_lib"],
        *get_mpi_flags(),
        *get_ishmem_extra_link_flags(),
        "-lze_loader",
        "-o",
        out,
    ]

    print(f"Building {label} extension...")
    print(" ".join(cmd))
    subprocess.check_call(cmd)
    print(f"Build successful: {out}\n")
    return out


def build():
    cfg = get_build_config()
    ishmem_cfg = get_ishmem_config()
    outputs = []
    outputs.append(
        build_one(cfg, "LocalPermuteCopy.cpp", "liblocal_permute_copy.so", "LocalPermuteCopy")
    )
    outputs.append(build_one(cfg, "EpDispatch.cpp", "libep_dispatch.so", "EpDispatch"))
    outputs.append(build_one(cfg, "NotifyDispatch.cpp", "libnotify_dispatch.so", "NotifyDispatch"))
    outputs.append(build_one(cfg, "EpCombine.cpp", "libep_combine.so", "EpCombine"))
    outputs.append(
        build_one(
            cfg,
            "AllgatherWithSymmMem.cpp",
            "liballgather_with_symm_mem.so",
            "AllgatherWithSymmMem",
        )
    )
    outputs.append(
        build_one(
            cfg,
            "UnpermuteReduceScatter.cpp",
            "libunpermute_reduce_scatter.so",
            "UnpermuteReduceScatter",
        )
    )
    outputs.append(
        build_one(
            cfg,
            "RingAllgather.cpp",
            "libring_allgather.so",
            "RingAllgather",
        )
    )
    outputs.append(
        build_one(
            cfg,
            "RingReduceScatter.cpp",
            "libring_reduce_scatter.so",
            "RingReduceScatter",
        )
    )
    outputs.append(
        build_one(
            cfg,
            "RingAllgatherPermute.cpp",
            "libring_allgather_permute.so",
            "RingAllgatherPermute",
        )
    )
    outputs.append(
        build_one(
            cfg,
            "RingReduceScatterUnpermute.cpp",
            "libring_reduce_scatter_unpermute.so",
            "RingReduceScatterUnpermute",
        )
    )
    outputs.append(
        build_one(
            cfg,
            "RingReduceScatterUnpermuteTwoStage.cpp",
            "libring_reduce_scatter_unpermute_two_stage.so",
            "RingReduceScatterUnpermuteTwoStage",
        )
    )
    outputs.append(
        build_one_ishmem(
            cfg,
            ishmem_cfg,
            "AllgatherPermuteIshmem.cpp",
            "liballgather_permute_ishmem.so",
            "AllgatherPermuteIshmem",
        )
    )
    outputs.append(
        build_one_ishmem(
            cfg,
            ishmem_cfg,
            "RingAllgatherIshmem.cpp",
            "libring_allgather_ishmem.so",
            "RingAllgatherIshmem",
        )
    )
    outputs.append(
        build_one_ishmem(
            cfg,
            ishmem_cfg,
            "RingReduceScatterIshmem.cpp",
            "libring_reduce_scatter_ishmem.so",
            "RingReduceScatterIshmem",
        )
    )
    return outputs


def clean():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    outputs = [
        os.path.join(base_dir, "liblocal_permute_copy.so"),
        os.path.join(base_dir, "libep_dispatch.so"),
        os.path.join(base_dir, "libnotify_dispatch.so"),
        os.path.join(base_dir, "libep_combine.so"),
        os.path.join(base_dir, "liballgather_with_symm_mem.so"),
        os.path.join(base_dir, "libunpermute_reduce_scatter.so"),
        os.path.join(base_dir, "libring_allgather.so"),
        os.path.join(base_dir, "libring_reduce_scatter.so"),
        os.path.join(base_dir, "libring_allgather_permute.so"),
        os.path.join(base_dir, "libring_reduce_scatter_unpermute.so"),
        os.path.join(base_dir, "libring_reduce_scatter_unpermute_two_stage.so"),
        os.path.join(base_dir, "liballgather_permute_ishmem.so"),
        os.path.join(base_dir, "libring_allgather_ishmem.so"),
        os.path.join(base_dir, "libring_reduce_scatter_ishmem.so"),
    ]

    removed = False
    for out in outputs:
        if os.path.exists(out):
            os.remove(out)
            print(f"Removed {out}")
            removed = True
    if not removed:
        print("Nothing to clean.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "clean":
        clean()
    else:
        build()
