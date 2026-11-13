#!/usr/bin/env python3
"""Point pip/uv/HF/torch at the persistent ~/.cache and keep the cache within budget.

Runs once per job from setup-python. Best effort: problems are reported as warnings, never failures.
Budget (env): CI_CACHE_MAX_GB (40% of disk), CI_CACHE_MIN_FREE_GB (15% of disk), CI_CACHE_MAX_AGE_DAYS (14).
"""

from __future__ import annotations

import fcntl
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

DAY = 86_400
GIB = 1 << 30
NOW = time.time()
CACHE = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")


def info(msg: str) -> None:
    print(f"[ci_cache] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"::warning title=ci_cache::{msg}", flush=True)


def export(**env: object) -> None:
    for key, value in env.items():
        os.environ[key] = str(value)
        info(f"{key}={value}")
        if os.environ.get("GITHUB_ENV"):
            with open(os.environ["GITHUB_ENV"], "a") as fh:
                fh.write(f"{key}={value}\n")


def last_used(path: Path) -> float:
    st = path.lstat()
    if path.is_dir():  # listing a dir (as prune does) bumps its atime
        return st.st_mtime
    return max(st.st_atime, st.st_mtime)


def remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def du(path: Path) -> int:
    out = subprocess.run(["du", "-sb", str(path)], capture_output=True, text=True).stdout.split()
    return int(out[0]) if out else 0


def uv_cache(*args: str) -> None:
    # uv locks its cache, so this waits for concurrent installs instead of breaking them
    if shutil.which("uv"):
        subprocess.run(["timeout", "900", "uv", "cache", *args])


# ── setup ────────────────────────────────────────────────────────────────────

def setup_env() -> None:
    pip_cache = CACHE / "pip"
    pip_cache.mkdir(parents=True, exist_ok=True)
    if os.geteuid() == 0:
        os.chown(pip_cache, 0, 0)  # pip as root ignores a cache dir it doesn't own
    export(
        PIP_CACHE_DIR=pip_cache,
        UV_CACHE_DIR=CACHE / "uv",
        UV_LINK_MODE="copy",  # cache and venv sit on different mounts
        HF_HOME=os.environ.get("HF_HOME") or CACHE / "huggingface",
        TORCH_HOME=os.environ.get("TORCH_HOME") or CACHE / "torch",
    )


# ── prune ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Entry:
    path: Path
    size: int
    used: float


def hf_repos() -> list[Path]:
    hub = Path(os.environ["HF_HOME"]) / "hub"
    return [p for p in hub.iterdir() if p.is_dir() and "--" in p.name] if hub.is_dir() else []


def drop_superseded_hf() -> None:
    """Snapshots no ref points to anymore are older upstream commits; drop them and the blobs only they used."""
    for repo in hf_repos():
        refs = {p.read_text().strip() for p in (repo / "refs").rglob("*") if p.is_file()}
        snapshots = repo / "snapshots"
        if not refs or not snapshots.is_dir():  # fetched by commit hash only: nothing is superseded
            continue
        links = {s: {Path(os.readlink(f)).name for f in s.rglob("*") if f.is_symlink()} for s in snapshots.iterdir()}
        blobs = repo / "blobs"

        def idle(snapshot: Path) -> bool:  # reads go through the symlinks, so blob atime is the real last use
            times = [last_used(blobs / b) for b in links[snapshot] if (blobs / b).exists()]
            return max([last_used(snapshot), *times]) < NOW - 2 * DAY

        stale = [s for s in links if s.name not in refs and idle(s)]
        keep = set().union(*(names for s, names in links.items() if s not in stale))
        for snapshot in stale:
            remove(snapshot)
        for blob in blobs.glob("*") if stale else ():
            if blob.name not in keep and last_used(blob) < NOW - 2 * DAY:
                remove(blob)


def drop_stale_uv_pointers(max_age: float) -> None:
    """Old nightlies and per-build wheels; `uv cache prune` then frees the archives they pointed to."""
    groups: dict[Path, list[Path]] = defaultdict(list)
    for ptr in (CACHE / "uv").glob("wheels-v*/**/*"):
        if ptr.is_file():
            groups[ptr.with_suffix("")].append(ptr)  # .http/.msgpack/.rev of one wheel go together
    for files in groups.values():
        if max(map(last_used, files)) < NOW - max_age:
            for ptr in files:
                remove(ptr)


def lru_entries() -> list[Entry]:
    """Evictable units, least recently used first: HF repos and pip/torch-hub files. uv manages its own."""
    entries = []
    for repo in hf_repos():
        blobs = list((repo / "blobs").glob("*"))
        used = max([last_used(repo), *map(last_used, blobs)])
        entries.append(Entry(repo, sum(b.lstat().st_size for b in blobs), used))
    for root in (CACHE / "pip", Path(os.environ["TORCH_HOME"]) / "hub" / "checkpoints"):
        files = [p for p in root.rglob("*") if p.is_file()] if root.is_dir() else []
        entries += [Entry(p, p.lstat().st_size, last_used(p)) for p in files]
    return sorted(entries, key=lambda e: e.used)


def prune() -> None:
    stamp = CACHE / ".ci_cache.pruned"
    with open(CACHE / ".ci_cache.lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return  # another job on this host is pruning
        if stamp.exists() and stamp.stat().st_mtime > NOW - DAY / 2:
            return

        disk = shutil.disk_usage(CACHE).total
        max_bytes = float(os.environ.get("CI_CACHE_MAX_GB") or 0.40 * disk / GIB) * GIB
        min_free = float(os.environ.get("CI_CACHE_MIN_FREE_GB") or 0.15 * disk / GIB) * GIB
        max_age = float(os.environ.get("CI_CACHE_MAX_AGE_DAYS") or 14) * DAY

        # superseded versions and anything idle for max_age
        drop_superseded_hf()
        drop_stale_uv_pointers(max_age)
        uv_cache("prune")
        entries = []
        for entry in lru_entries():
            if entry.used < NOW - max_age:
                remove(entry.path)
            else:
                entries.append(entry)

        # over budget: uv downloads (cheapest to refetch), then LRU sparing today's, then all of uv
        uv_size = du(CACHE / "uv")

        def over() -> bool:
            used = uv_size + sum(e.size for e in entries)
            return used > max_bytes or shutil.disk_usage(CACHE).free < min_free

        if over():
            uv_cache("prune", "--ci")
            uv_size = du(CACHE / "uv")
        while over() and entries and entries[0].used < NOW - DAY:
            remove(entries.pop(0).path)
        if over():
            uv_cache("clean")
            uv_size = du(CACHE / "uv")

        used = uv_size + sum(e.size for e in entries)
        (warn if over() else info)(f"cache {used / GIB:.1f}G of {max_bytes / GIB:.0f}G, "
                                   f"disk free {shutil.disk_usage(CACHE).free / GIB:.0f}G of min {min_free / GIB:.0f}G")
        stamp.touch()


def main() -> int:
    if os.environ.get("RUNNER_ENVIRONMENT") == "github-hosted":
        info("GitHub-hosted runner: ~/.cache does not persist, skipping")
        return 0
    for step in (setup_env, prune):
        try:
            step()
        except Exception as exc:  # a cache problem must never fail the job
            warn(f"{step.__name__} failed: {exc!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
