#!/usr/bin/env python3
# Copyright 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0

"""Place torch-xpu-ops into a PyTorch source tree.

Shared by prepare_pytorch.py (source builds) and the test environment
actions (wheel tests) so every CI flow resolves refs the same way.

Ref handling:
  * ``--source``: copy that checkout as-is (PR merge commit); the requested
    ref is ignored, including ``pinned`` passed for disable_build runs.
  * ``pinned`` resolves to the commit PyTorch pins. With ``--prefer-wheel``
    the SHA recorded in the installed wheel (``torch/xpu.txt``) wins, since
    the source ``third_party/xpu.txt`` may name a release branch whose head
    has moved since the wheel was built.
  * Any branch, tag or SHA in the given repo, including commits that only
    live under ``refs/pull/*``.
  * A ref missing from the repo fails; a fork must be passed as
    ``<repo-url>@<ref>``.

Usage:
    python checkout_torch_xpu_ops.py --pytorch-dir DIR --ref REF
        [--repo URL] [--source DIR] [--prefer-wheel]
"""

import argparse
import os
import shutil
import stat
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

DEFAULT_REPO = "https://github.com/intel/torch-xpu-ops.git"


def force_remove_readonly(func, path, _exc_info):
    """Handle read-only files on Windows during rmtree."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def run(cmd, check=True, quiet=False):
    print(f"+ {' '.join(cmd)}", flush=True)
    return subprocess.run(
        cmd, check=check, stderr=subprocess.DEVNULL if quiet else None
    )


def git_out(*args):
    cmd = ["git", *args]
    print(f"+ {' '.join(cmd)}", flush=True)
    return subprocess.check_output(cmd, text=True).strip()


def wheel_pin():
    """Return the torch-xpu-ops SHA recorded in the installed torch wheel."""
    try:
        path = Path(distribution("torch").locate_file("torch/xpu.txt"))
    except PackageNotFoundError:
        return ""
    return path.read_text().strip() if path.is_file() else ""


def resolve_pinned(pytorch_dir, prefer_wheel):
    if prefer_wheel:
        sha = wheel_pin()
        if sha:
            print(f"Using torch-xpu-ops pin from installed wheel: {sha}")
            return sha
        # TODO: make this an error once nightly XPU wheels ship torch/xpu.txt
        # (pytorch/pytorch#202608); falling back can drift on release branches.
        print("Installed wheel has no torch/xpu.txt; using source pin")
    ref = (Path(pytorch_dir) / "third_party" / "xpu.txt").read_text().strip()
    print(f"Using torch-xpu-ops pin from source: {ref}")
    return ref


def resolve_local(dest, ref):
    """Return the commit for ref if the clone already has it, else ''."""
    for candidate in (ref, f"origin/{ref}"):
        res = subprocess.run(
            ["git", "-C", dest, "rev-parse", "--verify", "--quiet",
             f"{candidate}^{{commit}}"],
            capture_output=True, text=True,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    return ""


def checkout_ref(repo, ref, dest):
    """Clone repo into dest and detach at ref."""
    dest = str(dest)
    run(["git", "clone", repo, dest])
    sha = resolve_local(dest, ref)
    if not sha:
        # Not reachable from the clone: fetch it directly (SHA or full ref),
        # then fall back to PR refs, which a plain clone never fetches.
        fetch = ["git", "-C", dest, "fetch", "--no-tags", "origin"]
        if run([*fetch, ref], check=False, quiet=True).returncode == 0:
            sha = git_out("-C", dest, "rev-parse", "FETCH_HEAD")
        else:
            run(
                [*fetch, "+refs/pull/*/head:refs/pull/*/head",
                 "+refs/pull/*/merge:refs/pull/*/merge"],
                check=False,
            )
            sha = resolve_local(dest, ref)
    if not sha:
        sys.exit(
            f"error: torch-xpu-ops ref '{ref}' not found in {repo}. "
            "If it only exists in a fork, pass '<fork-repo-url>@<ref>'."
        )
    run(["git", "-C", dest, "checkout", "--quiet", "--detach", sha])


def place(pytorch_dir, repo, ref, source=None, prefer_wheel=False):
    """Populate <pytorch_dir>/third_party/torch-xpu-ops and return its HEAD SHA."""
    pytorch_dir = Path(pytorch_dir)
    dest = pytorch_dir / "third_party" / "torch-xpu-ops"
    if dest.exists():
        shutil.rmtree(dest, onerror=force_remove_readonly)

    if source:
        print(f"Using PR checkout {source}; ignoring requested ref '{ref}'")
        shutil.copytree(source, dest, symlinks=True)
    else:
        if ref.lower() == "pinned":
            repo = DEFAULT_REPO
            ref = resolve_pinned(pytorch_dir, prefer_wheel)
        checkout_ref(repo or DEFAULT_REPO, ref or "main", dest)

    run(["git", "-C", str(dest), "remote", "-v"])
    run(["git", "-C", str(dest), "show", "-s"])
    run(["git", "-C", str(dest), "status", "--porcelain"])
    return git_out("-C", str(dest), "rev-parse", "HEAD")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pytorch-dir", required=True)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--ref", default="main")
    parser.add_argument("--source", default="", help="Copy this checkout (PR)")
    parser.add_argument("--prefer-wheel", action="store_true")
    args = parser.parse_args()
    sha = place(
        args.pytorch_dir, args.repo, args.ref,
        source=args.source or None, prefer_wheel=args.prefer_wheel,
    )
    print(f"torch-xpu-ops commit: {sha}")


if __name__ == "__main__":
    main()
