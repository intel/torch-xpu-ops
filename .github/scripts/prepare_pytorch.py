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
import sys
from pathlib import Path


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

    # Resolve pinned torch-xpu-ops
    torch_xpu_ops_repo = args.torch_xpu_ops_repo
    torch_xpu_ops_commit = args.torch_xpu_ops_commit
    if torch_xpu_ops_commit.lower() == "pinned":
        torch_xpu_ops_repo = "https://github.com/intel/torch-xpu-ops.git"
        torch_xpu_ops_commit = (
            (pytorch_dir / "third_party" / "xpu.txt").read_text().strip()
        )

    # Place torch-xpu-ops into third_party/
    third_party_dir = pytorch_dir / "third_party" / "torch-xpu-ops"
    if third_party_dir.exists():
        shutil.rmtree(third_party_dir, onerror=_force_remove_readonly)

    if github_event == "pull_request":
        # Copy the checked-out PR source
        src = workspace / "torch-xpu-ops"
        shutil.copytree(src, third_party_dir, symlinks=True)
    else:
        git("clone", torch_xpu_ops_repo, str(third_party_dir))
        git("-C", str(third_party_dir), "checkout", torch_xpu_ops_commit)

    # Show torch-xpu-ops info
    git("-C", str(third_party_dir), "remote", "-v")
    git("-C", str(third_party_dir), "branch")
    git("-C", str(third_party_dir), "show", "-s")

    # Apply cherry-picks
    run([sys.executable, "-m", "pip", "install", "requests"])
    apply_script = third_party_dir / ".github" / "scripts" / "apply_torch_pr.py"
    run([sys.executable, str(apply_script)])

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
            content = re.sub(
                r"GIT_REPOSITORY\s+\S+",
                f"GIT_REPOSITORY {repo_url}",
                content,
            )
        else:
            ref = onednn
        content = re.sub(r"GIT_TAG\s+\S+", f"GIT_TAG {ref}", content)
        find_mkldnn.write_text(content)
        print(f"oneDNN override: {onednn}")
        for line in content.splitlines():
            if "GIT_REPOSITORY" in line or "GIT_TAG" in line:
                print(f"  {line.strip()}")

    # Patch CMakeLists.txt to skip torch-xpu-ops checkout
    cmakelists = pytorch_dir / "caffe2" / "CMakeLists.txt"
    text = cmakelists.read_text()
    text = text.replace("checkout --quiet ${TORCH_XPU_OPS_COMMIT}", "log -n 1")
    cmakelists.write_text(text)

    print(f"\nPyTorch source prepared at: {pytorch_dir}")
    print(f"  pytorch commit: {commit}")
    print(f"  torch-xpu-ops: {torch_xpu_ops_commit}")


if __name__ == "__main__":
    main()
