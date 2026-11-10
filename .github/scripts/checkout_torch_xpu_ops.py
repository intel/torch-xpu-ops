#!/usr/bin/env python3
# Copyright 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0

"""Prepare torch-xpu-ops for source builds and wheel tests.

--source preserves the PR checkout and verifies --ref when it is a full SHA.
Otherwise, use --repo and --ref; pinned selects the PyTorch source pin.
--prefer-wheel uses installed torch/xpu.txt first, since a source release
branch may have advanced since the wheel was built. For fork commits, supply
the fork URL with --repo and its branch or commit with --ref.
"""

import argparse
import os
import re
import shutil
import stat
import subprocess
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

DEFAULT_REPO = "https://github.com/intel/torch-xpu-ops.git"


def force_remove_readonly(func, path, _exc_info):
    """Handle read-only files on Windows during rmtree."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def git(*args, check=True):
    cmd = ["git", *args]
    print(f"+ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, check=check, stdout=subprocess.PIPE, text=True)
    if result.stdout:
        print(result.stdout, end="", flush=True)
    return result.stdout.strip() if result.returncode == 0 else ""


def resolve_pinned(pytorch_dir, prefer_wheel):
    if prefer_wheel:
        try:
            path = Path(distribution("torch").locate_file("torch/xpu.txt"))
        except PackageNotFoundError:
            pass
        else:
            sha = path.read_text().strip() if path.is_file() else ""
            if sha:
                print(f"Using torch-xpu-ops pin from installed wheel: {sha}")
                return sha
        # Older wheels lack this record; a source branch cannot recover their exact SHA.
        print("Installed wheel has no torch/xpu.txt; using source pin (may be a moving branch)")
    ref = (Path(pytorch_dir) / "third_party" / "xpu.txt").read_text().strip()
    print(f"Using torch-xpu-ops pin from source: {ref}")
    return ref


def prepare_checkout(pytorch_dir, repo, ref, source=None, prefer_wheel=False):
    """Populate <pytorch_dir>/third_party/torch-xpu-ops and return its HEAD SHA."""
    repo = repo or DEFAULT_REPO
    ref = ref or ("pinned" if prefer_wheel else "main")
    pytorch_dir = Path(pytorch_dir)
    dest = pytorch_dir / "third_party" / "torch-xpu-ops"
    if dest.exists():
        shutil.rmtree(dest, onerror=force_remove_readonly)

    if source:
        print(f"Using PR checkout {source}")
        shutil.copytree(source, dest, symlinks=True)
    else:
        if ref.lower() == "pinned":
            repo = DEFAULT_REPO
            ref = resolve_pinned(pytorch_dir, prefer_wheel)
        git("clone", "--no-checkout", repo, str(dest))
        if ref.startswith("release/"):
            ref = f"refs/heads/{ref}"
        for candidate in (ref, f"refs/remotes/origin/{ref.removeprefix('refs/heads/')}"):
            sha = git("-C", str(dest), "rev-parse", "--verify", "--quiet", f"{candidate}^{{commit}}", check=False)
            if sha:
                break
        if not sha:
            try:
                git("-C", str(dest), "fetch", "--no-tags", "origin", ref)
            except subprocess.CalledProcessError:
                raise SystemExit(
                    f"Cannot fetch torch-xpu-ops ref '{ref}' from {repo}. "
                    "For a fork-only commit, pass the fork repository URL and ref."
                ) from None
            sha = git("-C", str(dest), "rev-parse", "--verify", "FETCH_HEAD^{commit}")
        git("-C", str(dest), "checkout", "--quiet", "--detach", sha)

    git("-C", str(dest), "show", "-s")
    git("-C", str(dest), "status", "--porcelain")
    sha = git("-C", str(dest), "rev-parse", "HEAD")
    if source and re.fullmatch(r"[0-9a-fA-F]{40}", ref) and sha != ref.lower():
        raise SystemExit(f"PR checkout mismatch: expected {ref}, got {sha}")
    return sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pytorch-dir", required=True)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--ref", default="")
    parser.add_argument("--source", default="", help="Copy this checkout (PR)")
    parser.add_argument("--prefer-wheel", action="store_true")
    args = parser.parse_args()
    sha = prepare_checkout(
        args.pytorch_dir, args.repo, args.ref,
        source=args.source or None, prefer_wheel=args.prefer_wheel,
    )
    print(f"torch-xpu-ops commit: {sha}")


if __name__ == "__main__":
    main()
