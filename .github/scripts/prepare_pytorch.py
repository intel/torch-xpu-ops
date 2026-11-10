#!/usr/bin/env python3
"""Prepare PyTorch source tree with torch-xpu-ops for XPU builds.

Cross-platform script (Linux + Windows) that replaces the separate
prepare_pytorch.sh and prepare_pytorch.bat scripts.

Usage:
    python prepare_pytorch.py [options]

All options can also be set via environment variables.
After running, CWD is <workspace>/pytorch with source ready to build.
"""

import argparse
import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

from checkout_torch_xpu_ops import place as place_torch_xpu_ops


def _force_remove_readonly(func, path, _exc_info):
    """Handle read-only files on Windows during rmtree."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def run(cmd, **kwargs):
    """Run a command, printing it first."""
    print(f"+ {cmd if isinstance(cmd, str) else ' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=True, **kwargs)


def git(*args, **kwargs):
    run(["git", *args], **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        default=os.environ.get("WORKSPACE", "/tmp"),
        help="Workspace root directory",
    )
    parser.add_argument(
        "--pytorch-repo",
        default=os.environ.get(
            "PYTORCH_REPO", "https://github.com/pytorch/pytorch.git"
        ),
    )
    parser.add_argument(
        "--pytorch-commit",
        default=os.environ.get("PYTORCH_COMMIT", "main"),
    )
    parser.add_argument(
        "--torch-xpu-ops-repo",
        default=os.environ.get(
            "TORCH_XPU_OPS_REPO", "https://github.com/intel/torch-xpu-ops.git"
        ),
    )
    parser.add_argument(
        "--torch-xpu-ops-commit",
        default=os.environ.get("TORCH_XPU_OPS_COMMIT", "main"),
    )
    parser.add_argument(
        "--onednn-commit",
        default=os.environ.get("ONEDNN_COMMIT", ""),
        help="Optional oneDNN override: commit, or repo@commit",
    )
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    pytorch_dir = workspace / "pytorch"
    github_event = os.environ.get("GITHUB_EVENT_NAME", "")

    # Clone PyTorch
    if pytorch_dir.exists():
        shutil.rmtree(pytorch_dir, onerror=_force_remove_readonly)
    git("clone", args.pytorch_repo, str(pytorch_dir))
    os.chdir(pytorch_dir)
    git("checkout", args.pytorch_commit)
    git("remote", "-v")
    git("branch")
    git("show", "-s")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    (workspace / "pytorch.commit").write_text(commit + "\n")

    # Place torch-xpu-ops into third_party/: the PR merge checkout as-is,
    # otherwise the requested ref (``pinned`` reads third_party/xpu.txt).
    xpu_pin = pytorch_dir / "third_party" / "xpu.txt"
    xpu_pin_ref = xpu_pin.read_text().strip()
    source = workspace / "torch-xpu-ops" if github_event == "pull_request" else None
    torch_xpu_ops_sha = place_torch_xpu_ops(
        pytorch_dir,
        args.torch_xpu_ops_repo,
        args.torch_xpu_ops_commit,
        source=source,
    )

    # Extra PyTorch cherry-picks are not needed for now (re-add `import sys`).
    # run([sys.executable, "-m", "pip", "install", "requests"])
    # apply_script = (
    #     pytorch_dir / "third_party" / "torch-xpu-ops" / ".github" / "scripts"
    #     / "apply_torch_pr.py"
    # )
    # run([sys.executable, str(apply_script)])

    # Init submodules
    git("submodule", "sync")
    git("submodule", "update", "--init", "--recursive")

    # Optional oneDNN override
    onednn = args.onednn_commit
    if onednn and onednn.lower() != "pinned":
        find_mkldnn = pytorch_dir / "cmake" / "Modules" / "FindMKLDNN.cmake"
        content = find_mkldnn.read_text()
        if "@" in onednn:
            repo_url, ref = onednn.rsplit("@", 1)
        else:
            repo_url = re.search(r"GIT_REPOSITORY\s+(\S+)", content).group(1)
            ref = onednn

        # A PR merge/head commit only lives under refs/pull/*, which the plain
        # clone done by ExternalProject never fetches; mirror it onto a local
        # branch and build from that clone instead.
        onednn_dir = workspace / "onednn"
        if onednn_dir.exists():
            shutil.rmtree(onednn_dir, onerror=_force_remove_readonly)
        git("clone", repo_url, str(onednn_dir))
        fetched = subprocess.run(
            ["git", "-C", str(onednn_dir), "fetch", "--no-tags", "origin", ref],
            check=False,
        )
        if fetched.returncode != 0:
            git(
                "-C", str(onednn_dir), "fetch", "--no-tags", "origin",
                "+refs/pull/*/head:refs/pull/*/head",
                "+refs/pull/*/merge:refs/pull/*/merge",
            )
        git("-C", str(onednn_dir), "checkout", "-B", "acceptance", ref)
        git("-C", str(onednn_dir), "show", "-s")

        content = re.sub(
            r"GIT_REPOSITORY\s+\S+",
            f"GIT_REPOSITORY {onednn_dir.as_posix()}",
            content,
        )
        content = re.sub(r"GIT_TAG\s+\S+", f"GIT_TAG {ref}", content)
        find_mkldnn.write_text(content)
        print(f"oneDNN override: {onednn}")
        for line in content.splitlines():
            if "GIT_REPOSITORY" in line or "GIT_TAG" in line:
                print(f"  {line.strip()}")

    # Pin CMake to the prepared checkout, including PR merge commits. Done last
    # so nothing above sees a modified PyTorch tree.
    xpu_pin.write_bytes((torch_xpu_ops_sha + "\n").encode())

    print(f"\nPyTorch source prepared at: {pytorch_dir}")
    print(f"  pytorch commit: {commit}")
    print(f"  torch-xpu-ops pin: {xpu_pin_ref} -> {torch_xpu_ops_sha}")


if __name__ == "__main__":
    main()
