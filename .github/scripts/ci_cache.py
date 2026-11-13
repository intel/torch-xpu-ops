#!/usr/bin/env python3
"""Shared CI caches and CA bundle for Linux jobs. Stdlib only, best effort: never fails the job.

  setup                                    export cache dirs and a merged CA bundle to GITHUB_ENV
  torchbench restore|save REPO [MODEL ...] reuse torchbench weights and S3 inputs across runs
  prune                                    drop superseded/unused entries, keep the cache within budget

Budget knobs: CI_CACHE_MAX_GB (default 40% of the disk), CI_CACHE_MIN_FREE_GB (default 15% of the disk),
CI_CACHE_MAX_AGE_DAYS (default 14). Last use is max(atime, mtime), which relatime refreshes daily.
"""

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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
TB = ROOT / "torchbench"
DAY = 86400
NOW = time.time()
GB = 1 << 30
TLS_PROBES = (
    "https://pypi.org/simple/",
    "https://download.pytorch.org/whl/",
    "https://huggingface.co",
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
)


def log(msg):
    print(f"[ci_cache] {msg}", flush=True)


def warn(msg):
    print(f"::warning::[ci_cache] {msg}", flush=True)


def set_env(**kv):
    for key, value in kv.items():
        os.environ[key] = str(value)
        log(f"{key}={value}")
        if os.environ.get("GITHUB_ENV"):
            with open(os.environ["GITHUB_ENV"], "a") as f:
                f.write(f"{key}={value}\n")


def hf_hub():
    return Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME") or ROOT / "huggingface") / "hub")


def last_used(path):
    st = path.lstat()
    return max(st.st_atime, st.st_mtime)


def tree_size(path):
    return sum(p.lstat().st_size for p in path.rglob("*") if not p.is_dir() or p.is_symlink()) if path.is_dir() else 0


def remove(path):
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def copy_atomic(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def uv_cache(*args):
    if shutil.which("uv"):
        # uv locks its cache, so this waits for concurrent installs instead of breaking them
        subprocess.run(["timeout", "900", "uv", "cache", *args], env={**os.environ, "UV_CACHE_DIR": str(ROOT / "uv")})


# ---------------------------------------------------------------- setup
def cmd_setup():
    pip_dir = ROOT / "pip"
    pip_dir.mkdir(parents=True, exist_ok=True)
    if os.geteuid() == 0:
        # pip running as root ignores a cache dir it doesn't own; job cleanup chowns it back to 1000
        os.chown(pip_dir, 0, 0)
    set_env(
        PIP_CACHE_DIR=pip_dir,
        UV_CACHE_DIR=ROOT / "uv",
        UV_LINK_MODE="copy",  # cache and venv sit on different mounts, hardlinks always fall back
        HF_HOME=os.environ.get("HF_HOME") or ROOT / "huggingface",
        TORCH_HOME=os.environ.get("TORCH_HOME") or ROOT / "torch",
    )
    bundle = build_ca_bundle()
    if bundle:
        set_env(SSL_CERT_FILE=bundle, REQUESTS_CA_BUNDLE=bundle)
        check_tls(bundle)


def build_ca_bundle():
    """System store (corporate/proxy CAs) + latest certifi (fresh Mozilla roots); requests alone uses certifi only."""
    if shutil.which("uv") and os.environ.get("VIRTUAL_ENV"):
        subprocess.run(["uv", "pip", "install", "-q", "-U", "certifi"])
    system = (ssl.get_default_verify_paths().cafile, "/etc/ssl/certs/ca-certificates.crt", "/etc/pki/tls/certs/ca-bundle.crt")
    sources = [Path(p) for p in system if p]
    try:
        out = subprocess.run([sys.executable, "-c", "import certifi; print(certifi.where())"],
                             capture_output=True, text=True).stdout.strip()
        sources.append(Path(out))
    except OSError:
        pass
    pems = [p.read_text() for p in dict.fromkeys(s.resolve() for s in sources) if p.is_file()]
    if not pems:
        warn("no CA bundle found, keeping defaults")
        return None
    bundle = Path(os.environ.get("RUNNER_TEMP") or "/tmp") / "ci-ca-bundle.pem"
    bundle.write_text("\n".join(pems))
    log(f"CA bundle: {bundle.read_text().count('BEGIN CERTIFICATE')} certs")
    return bundle


def check_tls(bundle):
    ctx = ssl.create_default_context(cafile=str(bundle))

    def probe(url):
        try:
            urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=15, context=ctx)
            return None
        except Exception as e:  # HTTP status errors still prove TLS works
            return None if getattr(e, "code", None) else f"{url}: {e}"

    with ThreadPoolExecutor(len(TLS_PROBES)) as pool:
        failures = [f for f in pool.map(probe, TLS_PROBES) if f]
    for f in failures:
        warn(f"TLS check failed (host clock {time.strftime('%F %T UTC', time.gmtime())}): {f}")


# ---------------------------------------------------------------- torchbench
def tb_models(repo, names):
    models = repo / "torchbenchmark" / "models"
    dirs = [models / n for n in names] if names else sorted(p for p in models.iterdir() if p.is_dir())
    for d in dirs:
        install = d / "install.py"
        # keyed by install.py so a changed download recipe starts a fresh entry and the old one ages out
        key = hashlib.sha256(install.read_bytes()).hexdigest()[:12] if install.is_file() else "none"
        yield d / ".data", TB / "models" / f"{d.name}-{key}"


def tb_inputs(repo):
    """S3 tarballs/pickles listed in torchbenchmark/data/index.yaml; install skips the download if present."""
    index = repo / "torchbenchmark" / "data" / "index.yaml"
    section = None
    for line in index.read_text().splitlines() if index.is_file() else []:
        line = line.split("#")[0].rstrip()
        if line.endswith(":") and not line.startswith(" "):
            section = line[:-1]
        elif line.strip().startswith("- ") and section in ("INPUT_TARBALLS", "MODEL_PKLS"):
            name = line.strip()[2:].strip()
            sub = "data" if section == "INPUT_TARBALLS" else "models"
            yield repo / "torchbenchmark" / sub / name, TB / "inputs" / section / name


def valid_input(path):
    if path.stat().st_size == 0:
        return False
    if path.name.endswith(".tar.gz"):
        return tarfile.is_tarfile(path)
    with path.open("rb") as f:
        return not f.read(5).startswith(b"<?xml")  # S3 error body saved as data


def sync_into(cached, data):
    """The cached copy is authoritative for its key: fill missing files and undo truncated re-downloads."""
    for c in cached.rglob("*"):
        if c.is_file():
            f = data / c.relative_to(cached)
            if not f.is_file() or f.stat().st_size != c.stat().st_size:
                copy_atomic(c, f)
    os.utime(cached)


def cmd_torchbench(action, repo, names):
    repo = Path(repo).resolve()
    names = [n for n in names if not n.startswith("-")]
    hits = added = 0
    for data, cached in tb_models(repo, names):
        if action == "save" and data.is_dir():
            for f in data.rglob("*"):
                c = cached / f.relative_to(data)
                if f.is_file() and not f.is_symlink() and f.stat().st_size and not c.exists():
                    copy_atomic(f, c)
                    added += 1
        if cached.is_dir():
            sync_into(cached, data)
            hits += 1
    for path, cached in tb_inputs(repo):
        if action == "save" and path.is_file() and not cached.exists() and valid_input(path):
            copy_atomic(path, cached)
            added += 1
        if action == "restore" and cached.is_file():
            if not path.exists():
                copy_atomic(cached, path)
            os.utime(cached)
            hits += 1
    log(f"torchbench {action}: {hits} cached entries used, {added} files added")


# ---------------------------------------------------------------- prune
def prune_superseded_hf(min_idle):
    """Drop snapshots no ref points to anymore (an older upstream commit) plus blobs only they used."""
    freed = 0
    for repo in hub_repos():
        refs = {p.read_text().strip() for p in (repo / "refs").rglob("*") if p.is_file()}
        snaps = [s for s in (repo / "snapshots").iterdir() if s.is_dir()] if (repo / "snapshots").is_dir() else []
        if not refs:  # downloaded by commit hash only, nothing marks a revision as superseded
            continue
        blobs_of = {s: {os.path.basename(os.readlink(ln)) for ln in s.rglob("*") if ln.is_symlink()} for s in snaps}
        blob_dir = repo / "blobs"

        def idle(s):
            times = [last_used(blob_dir / b) for b in blobs_of[s] if (blob_dir / b).exists()]
            return max(times + [s.stat().st_mtime]) < NOW - min_idle

        stale = [s for s in snaps if s.name not in refs and idle(s)]
        if not stale:
            continue
        keep = set().union(*(blobs_of[s] for s in snaps if s not in stale))
        for s in stale:
            remove(s)
        for b in blob_dir.iterdir() if blob_dir.is_dir() else []:
            if b.name not in keep and last_used(b) < NOW - min_idle:
                freed += b.lstat().st_size
                remove(b)
        log(f"HF {repo.name}: removed {len(stale)} superseded revisions")
    return freed


def hub_repos():
    hub = hf_hub()
    return [p for p in hub.iterdir() if p.is_dir() and "--" in p.name] if hub.is_dir() else []


def lru_entries():
    """(last_used, size, path) for every evictable unit except uv, which manages its own cache."""
    entries = []
    for repo in hub_repos():
        blobs = list((repo / "blobs").glob("*"))
        used = max([last_used(b) for b in blobs] + [repo.stat().st_mtime])
        entries.append((used, sum(b.lstat().st_size for b in blobs), repo))
    torch_ckpt = Path(os.environ.get("TORCH_HOME") or ROOT / "torch") / "hub" / "checkpoints"
    files = [p for d in (ROOT / "pip", torch_ckpt, TB / "inputs") if d.is_dir() for p in d.rglob("*") if p.is_file()]
    entries += [(last_used(p), p.lstat().st_size, p) for p in files]
    if (TB / "models").is_dir():
        entries += [(d.stat().st_mtime, tree_size(d), d) for d in (TB / "models").iterdir() if d.is_dir()]
    return sorted(entries, key=lambda e: e[0])


def du(path):
    out = subprocess.run(["du", "-sb", str(path)], capture_output=True, text=True).stdout.split()
    return int(out[0]) if out else 0


def cmd_prune():
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = open(ROOT / ".ci_cache.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log("another job is pruning, skip")
        return
    stamp = ROOT / ".ci_cache.pruned"
    if stamp.exists() and stamp.stat().st_mtime > NOW - DAY / 2 and not os.environ.get("CI_CACHE_FORCE_PRUNE"):
        log("pruned within the last 12h, skip")
        return

    disk = shutil.disk_usage(ROOT)
    max_bytes = float(os.environ.get("CI_CACHE_MAX_GB") or 0.40 * disk.total / GB) * GB
    min_free = float(os.environ.get("CI_CACHE_MIN_FREE_GB") or 0.15 * disk.total / GB) * GB
    max_age = float(os.environ.get("CI_CACHE_MAX_AGE_DAYS") or 14) * DAY

    # 1. superseded versions, 2. entries unused for max_age
    prune_superseded_hf(min_idle=2 * DAY)
    entries = lru_entries()
    for _, _, path in [e for e in entries if e[0] < NOW - max_age]:
        remove(path)
    entries = [e for e in entries if e[0] >= NOW - max_age]
    # uv wheel pointers (superseded nightlies, per-build wheels); prune then drops the archives they held
    pointers = {}
    for ptr in (ROOT / "uv").glob("wheels-v*/**/*"):
        if ptr.is_file():
            pointers.setdefault(ptr.with_suffix(""), []).append(ptr)
    for group in pointers.values():  # .http/.msgpack/.rev of one wheel go together
        if max(map(last_used, group)) < NOW - max_age:
            for ptr in group:
                remove(ptr)
    uv_cache("prune")

    # 3. over budget: drop downloaded uv wheels first (cheapest to refetch), then LRU, then all of uv
    uv_size = du(ROOT / "uv")
    usage = lambda: uv_size + sum(e[1] for e in entries)  # noqa: E731
    over = lambda: usage() > max_bytes or shutil.disk_usage(ROOT).free < min_free  # noqa: E731
    if over():
        uv_cache("prune", "--ci")
        uv_size = du(ROOT / "uv")
    while over() and entries and entries[0][0] < NOW - DAY:  # anything touched today may be in use
        remove(entries.pop(0)[2])
    if over():
        uv_cache("clean")
        uv_size = du(ROOT / "uv")
    if over():
        warn(f"cache still over budget: {usage() / GB:.0f}G used, {shutil.disk_usage(ROOT).free / GB:.0f}G free")
    stamp.touch()
    log(f"cache {usage() / GB:.1f}G (budget {max_bytes / GB:.0f}G, uv {uv_size / GB:.1f}G), "
        f"disk free {shutil.disk_usage(ROOT).free / GB:.0f}G (min {min_free / GB:.0f}G)")


def main(argv):
    try:
        if argv[:1] == ["setup"]:
            cmd_setup()
        elif argv[:1] == ["prune"]:
            cmd_prune()
        elif argv[:1] == ["torchbench"] and len(argv) >= 3 and argv[1] in ("restore", "save"):
            cmd_torchbench(argv[1], argv[2], argv[3:])
        else:
            sys.exit(__doc__)
    except Exception as e:  # a cache problem must never fail the job
        warn(f"{' '.join(argv)} failed: {e!r}")


if __name__ == "__main__":
    main(sys.argv[1:])
