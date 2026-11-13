#!/usr/bin/env python3
"""Persistent CI caches and CA bundle for Linux jobs.

Commands:
  setup                                     point pip/uv/HF/torch at ~/.cache, export a merged CA bundle
  torchbench {restore,save} REPO [MODEL..]  reuse torchbench weights and S3 inputs across runs
  prune                                     drop superseded and idle entries, keep the cache within budget

Budget (env): CI_CACHE_MAX_GB (40% of disk), CI_CACHE_MIN_FREE_GB (15% of disk), CI_CACHE_MAX_AGE_DAYS (14).
Everything is best effort: a cache problem is reported as a warning and never fails the job.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import os
import shutil
import ssl
import subprocess
import sys
import tarfile
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

DAY = 86_400
GIB = 1 << 30
NOW = time.time()

CACHE = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
TORCHBENCH = CACHE / "torchbench"
SYSTEM_CA = ("/etc/ssl/certs/ca-certificates.crt", "/etc/pki/tls/certs/ca-bundle.crt")
TLS_PROBES = (
    "https://pypi.org/simple/",
    "https://download.pytorch.org/whl/",
    "https://huggingface.co",
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
)


# ── helpers ──────────────────────────────────────────────────────────────────

def info(msg: str) -> None:
    print(f"[ci_cache] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"::warning title=ci_cache::{msg}", flush=True)


def export(**env: object) -> None:
    """Set for this process and for later steps of the job."""
    github_env = os.environ.get("GITHUB_ENV")
    for key, value in env.items():
        os.environ[key] = str(value)
        info(f"{key}={value}")
        if github_env:
            with open(github_env, "a") as fh:
                fh.write(f"{key}={value}\n")


def last_used(path: Path) -> float:
    st = path.lstat()
    return max(st.st_atime, st.st_mtime)


def size_of(path: Path) -> int:
    if path.is_dir() and not path.is_symlink():
        return sum(p.lstat().st_size for p in path.rglob("*") if p.is_symlink() or not p.is_dir())
    return path.lstat().st_size


def remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def copy_atomic(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def du(path: Path) -> int:
    out = subprocess.run(["du", "-sb", str(path)], capture_output=True, text=True).stdout.split()
    return int(out[0]) if out else 0


def uv_cache(*args: str) -> None:
    # uv locks its cache, so this waits for concurrent installs instead of breaking them
    if shutil.which("uv"):
        subprocess.run(["timeout", "900", "uv", "cache", *args], env={**os.environ, "UV_CACHE_DIR": str(CACHE / "uv")})


@contextmanager
def exclusive(lock: Path) -> Iterator[bool]:
    with open(lock, "w") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
        else:
            yield True


# ── setup ────────────────────────────────────────────────────────────────────

def setup(_: argparse.Namespace) -> None:
    pip_cache = CACHE / "pip"
    pip_cache.mkdir(parents=True, exist_ok=True)
    if os.geteuid() == 0:
        # pip as root ignores a cache dir it doesn't own; job cleanup chowns it back to 1000
        os.chown(pip_cache, 0, 0)
    export(
        PIP_CACHE_DIR=pip_cache,
        UV_CACHE_DIR=CACHE / "uv",
        UV_LINK_MODE="copy",  # cache and venv sit on different mounts
        HF_HOME=os.environ.get("HF_HOME") or CACHE / "huggingface",
        TORCH_HOME=os.environ.get("TORCH_HOME") or CACHE / "torch",
    )
    if bundle := build_ca_bundle():
        export(SSL_CERT_FILE=bundle, REQUESTS_CA_BUNDLE=bundle)
        probe_tls(bundle)


def build_ca_bundle() -> Path | None:
    """System store (corporate/proxy CAs) + latest certifi (Mozilla roots); requests alone only reads certifi."""
    if shutil.which("uv") and os.environ.get("VIRTUAL_ENV"):
        subprocess.run(["uv", "pip", "install", "-q", "-U", "certifi"])
    certifi = subprocess.run([sys.executable, "-c", "import certifi; print(certifi.where())"],
                             capture_output=True, text=True).stdout.strip()
    candidates = (ssl.get_default_verify_paths().cafile, *SYSTEM_CA, certifi)
    sources = dict.fromkeys(Path(p).resolve() for p in candidates if p and Path(p).is_file())
    if not sources:
        warn("no CA bundle found, keeping defaults")
        return None
    bundle = Path(os.environ.get("RUNNER_TEMP") or "/tmp") / "ci-ca-bundle.pem"
    bundle.write_text("\n".join(p.read_text() for p in sources))
    info(f"CA bundle: {bundle.read_text().count('BEGIN CERTIFICATE')} certs from {', '.join(map(str, sources))}")
    return bundle


def probe_tls(bundle: Path) -> None:
    context = ssl.create_default_context(cafile=str(bundle))

    def probe(url: str) -> str | None:
        try:
            urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=15, context=context)
        except Exception as exc:  # an HTTP status error still proves the TLS handshake worked
            return None if getattr(exc, "code", None) else f"{url}: {exc}"
        return None

    with ThreadPoolExecutor(len(TLS_PROBES)) as pool:
        failures = [f for f in pool.map(probe, TLS_PROBES) if f]
    clock = time.strftime("%F %T UTC", time.gmtime())
    for failure in failures:
        warn(f"TLS check failed (host clock {clock}): {failure}")


# ── torchbench ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Torchbench:
    repo: Path

    def models(self, names: list[str]) -> Iterator[tuple[Path, Path]]:
        """(models/<m>/.data, cache dir) keyed by install.py, so a changed download recipe starts fresh."""
        root = self.repo / "torchbenchmark" / "models"
        dirs = [root / n for n in names] if names else sorted(p for p in root.iterdir() if p.is_dir())
        for model in dirs:
            install = model / "install.py"
            key = hashlib.sha256(install.read_bytes()).hexdigest()[:12] if install.is_file() else "none"
            yield model / ".data", TORCHBENCH / "models" / f"{model.name}-{key}"

    def inputs(self) -> Iterator[tuple[Path, Path]]:
        """S3 tarballs/pickles from data/index.yaml; torchbench skips the download when the file exists."""
        index = self.repo / "torchbenchmark" / "data" / "index.yaml"
        targets = {"INPUT_TARBALLS": "data", "MODEL_PKLS": "models"}
        section = None
        for raw in index.read_text().splitlines() if index.is_file() else ():
            line = raw.split("#")[0].rstrip()
            if line and not line[0].isspace() and line.endswith(":"):
                section = line[:-1]
            elif section in targets and line.lstrip().startswith("- "):
                name = line.lstrip()[2:].strip()
                yield self.repo / "torchbenchmark" / targets[section] / name, TORCHBENCH / "inputs" / section / name


def is_valid_input(path: Path) -> bool:
    if path.stat().st_size == 0:
        return False
    if path.name.endswith(".tar.gz"):
        return tarfile.is_tarfile(path)
    with path.open("rb") as fh:
        return not fh.read(5).startswith(b"<?xml")  # S3 error page written as data


def store_new(local: Path, cached: Path) -> int:
    """Add files the cache doesn't have yet; entries are immutable per key."""
    added = 0
    for src in local.rglob("*") if local.is_dir() else ():
        dst = cached / src.relative_to(local)
        if src.is_file() and not src.is_symlink() and src.stat().st_size and not dst.exists():
            copy_atomic(src, dst)
            added += 1
    return added


def mirror(cached: Path, local: Path) -> None:
    """The cached copy wins: fills missing files and undoes truncated re-downloads."""
    for src in cached.rglob("*"):
        dst = local / src.relative_to(cached)
        if src.is_file() and (not dst.is_file() or dst.stat().st_size != src.stat().st_size):
            copy_atomic(src, dst)
    os.utime(cached)


def torchbench(args: argparse.Namespace) -> None:
    bench, saving = Torchbench(Path(args.repo).resolve()), args.action == "save"
    hits = added = 0
    for local, cached in bench.models(args.models):
        if saving:
            added += store_new(local, cached)
        if cached.is_dir():
            mirror(cached, local)
            hits += 1
    for local, cached in bench.inputs():
        if saving and local.is_file() and not cached.exists() and is_valid_input(local):
            copy_atomic(local, cached)
            added += 1
        elif not saving and cached.is_file():
            if not local.exists():
                copy_atomic(cached, local)
            os.utime(cached)
            hits += 1
    info(f"torchbench {args.action}: {hits} cache hits, {added} files added")


# ── prune ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Entry:
    path: Path
    size: int
    used: float

    @classmethod
    def of(cls, path: Path, used: float | None = None) -> Entry:
        return cls(path, size_of(path), last_used(path) if used is None else used)


@dataclass(frozen=True)
class Budget:
    max_bytes: float
    min_free: float
    max_age: float

    @classmethod
    def from_env(cls) -> Budget:
        disk = shutil.disk_usage(CACHE).total
        env = lambda key, default: float(os.environ.get(key) or default)  # noqa: E731
        return cls(
            max_bytes=env("CI_CACHE_MAX_GB", 0.40 * disk / GIB) * GIB,
            min_free=env("CI_CACHE_MIN_FREE_GB", 0.15 * disk / GIB) * GIB,
            max_age=env("CI_CACHE_MAX_AGE_DAYS", 14) * DAY,
        )

    def exceeded(self, used: int) -> bool:
        return used > self.max_bytes or shutil.disk_usage(CACHE).free < self.min_free


def hf_repos() -> list[Path]:
    hub = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME") or CACHE / "huggingface") / "hub")
    return [p for p in hub.iterdir() if p.is_dir() and "--" in p.name] if hub.is_dir() else []


def drop_superseded_hf(min_idle: float) -> None:
    """Remove snapshots no ref points to anymore (an older upstream commit) and the blobs only they used."""
    for repo in hf_repos():
        refs = {p.read_text().strip() for p in (repo / "refs").rglob("*") if p.is_file()}
        if not refs:  # fetched by commit hash only, nothing marks a revision as superseded
            continue
        blobs = repo / "blobs"
        links = {s: {Path(os.readlink(ln)).name for ln in s.rglob("*") if ln.is_symlink()}
                 for s in (repo / "snapshots").iterdir() if s.is_dir()}

        def idle(snapshot: Path) -> bool:
            times = [last_used(blobs / b) for b in links[snapshot] if (blobs / b).exists()]
            return max([snapshot.stat().st_mtime, *times]) < NOW - min_idle

        stale = [s for s in links if s.name not in refs and idle(s)]
        if not stale:
            continue
        keep = set().union(*(names for s, names in links.items() if s not in stale))
        for snapshot in stale:
            remove(snapshot)
        for blob in blobs.iterdir():
            if blob.name not in keep and last_used(blob) < NOW - min_idle:
                remove(blob)
        info(f"HF {repo.name}: removed {len(stale)} superseded revision(s)")


def drop_stale_uv_pointers(max_age: float) -> None:
    """Superseded nightlies and per-build wheels; `uv cache prune` then frees the archives they held."""
    groups: dict[Path, list[Path]] = defaultdict(list)
    for ptr in (CACHE / "uv").glob("wheels-v*/**/*"):
        if ptr.is_file():
            groups[ptr.with_suffix("")].append(ptr)  # .http/.msgpack/.rev of one wheel go together
    for files in groups.values():
        if max(map(last_used, files)) < NOW - max_age:
            for ptr in files:
                remove(ptr)


def lru_entries() -> list[Entry]:
    """Evictable units, least recently used first. uv is left to its own commands."""
    entries = []
    for repo in hf_repos():
        blobs = list((repo / "blobs").glob("*"))
        used = max([repo.stat().st_mtime, *map(last_used, blobs)])
        entries.append(Entry(repo, sum(b.lstat().st_size for b in blobs), used))
    checkpoints = Path(os.environ.get("TORCH_HOME") or CACHE / "torch") / "hub" / "checkpoints"
    for root in (CACHE / "pip", checkpoints, TORCHBENCH / "inputs"):
        entries += [Entry.of(p) for p in root.rglob("*") if p.is_file()] if root.is_dir() else []
    models = TORCHBENCH / "models"
    entries += [Entry.of(d, d.stat().st_mtime) for d in models.iterdir() if d.is_dir()] if models.is_dir() else []
    return sorted(entries, key=lambda e: e.used)


def prune(_: argparse.Namespace) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    stamp = CACHE / ".ci_cache.pruned"
    with exclusive(CACHE / ".ci_cache.lock") as locked:
        if not locked:
            info("another job is pruning, skip")
            return
        if stamp.exists() and stamp.stat().st_mtime > NOW - DAY / 2 and not os.environ.get("CI_CACHE_FORCE_PRUNE"):
            info("pruned within the last 12h, skip")
            return
        budget = Budget.from_env()

        # 1. superseded versions  2. idle for max_age
        drop_superseded_hf(min_idle=2 * DAY)
        drop_stale_uv_pointers(budget.max_age)
        uv_cache("prune")
        entries = []
        for entry in lru_entries():
            if entry.used < NOW - budget.max_age:
                remove(entry.path)
            else:
                entries.append(entry)

        # 3. over budget: uv downloads (cheapest to refetch) -> LRU, sparing today's -> all of uv
        uv_size = du(CACHE / "uv")
        usage = lambda: uv_size + sum(e.size for e in entries)  # noqa: E731
        if budget.exceeded(usage()):
            uv_cache("prune", "--ci")
            uv_size = du(CACHE / "uv")
        while budget.exceeded(usage()) and entries and entries[0].used < NOW - DAY:
            remove(entries.pop(0).path)
        if budget.exceeded(usage()):
            uv_cache("clean")
            uv_size = du(CACHE / "uv")

        free = shutil.disk_usage(CACHE).free
        summary = (f"cache {usage() / GIB:.1f}G of {budget.max_bytes / GIB:.0f}G (uv {uv_size / GIB:.1f}G), "
                   f"disk free {free / GIB:.0f}G of min {budget.min_free / GIB:.0f}G")
        (warn if budget.exceeded(usage()) else info)(summary)
        stamp.touch()


# ── cli ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("setup").set_defaults(run=setup)
    commands.add_parser("prune").set_defaults(run=prune)
    bench = commands.add_parser("torchbench")
    bench.add_argument("action", choices=("restore", "save"))
    bench.add_argument("repo", help="pytorch/benchmark checkout")
    bench.add_argument("models", nargs="*", help="models to sync (default: all)")
    bench.set_defaults(run=torchbench)
    args = parser.parse_args(argv)
    try:
        args.run(args)
    except Exception as exc:  # a cache problem must never fail the job
        warn(f"{args.command} failed: {exc!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
