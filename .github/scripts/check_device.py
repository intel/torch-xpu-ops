#!/usr/bin/env python3
"""Check an XPU CI machine: system acceptance, network and H2D/D2H bandwidth.

SYS  New machine acceptance (Linux only): Intel GPU on PCI, OS version and kernel
     driver per GPU family (BMG: Ubuntu 26.04 + xe, PVC: Ubuntu 24.04 + i915),
     compute runtime packages, tools, GPU visible to clinfo/xpu-smi, / > 800 GB
     plus disk bandwidth, user jenkins with UID 1001.
NET  Download speed of pytorch/pytorch git, intel/omix docker layer, torch XPU
     wheel, Hugging Face model and torch weights, all forced through the proxy.
BW   H2D/D2H copy bandwidth per device, pageable and pinned (reference only).
     Uses the current Python if its torch sees XPU (probe only, never modified);
     otherwise a temporary uv venv with torch XPU nightly, always removed.

Levels: PASS, WARN (works but slow), FAIL, ERROR (check could not run).
Exit:   0 PASS/WARN, 1 FAIL, 2 ERROR.
"""

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

MIN_DISK_GB = 800
JENKINS_UID = 1001
# GPU family -> (Ubuntu VERSION_ID, kernel driver)
GPU_SPEC = {"BMG": ("26.04", "xe"), "PVC": ("24.04", "i915")}
TOOLS = ("xpu-smi", "clinfo", "lspci", "ocloc")
# Each entry is satisfied by any one of its alternatives (PPA vs legacy names).
PACKAGES = (
    ("intel-opencl-icd",),
    ("libze-intel-gpu1", "intel-level-zero-gpu"),
    ("libze1", "level-zero"),
    ("intel-ocloc", "libigc-tools"),
)
NET_LIMIT_MIB = 256
NET_TIMEOUT = 60
# Below these speeds the check still passes but is flagged as a warning.
NET_WARN_MBPS = 10
DISK_WARN_MBPS = 500
UV_INSTALLER = "https://astral.sh/uv/install.sh"
UV_PYTHON = "3.12"
TORCH_INDEX = "https://download.pytorch.org/whl/nightly/xpu"
BW_TAG = "CHECK_DEVICE_BW "
# Kept out of the temporary venv so an active venv/conda cannot leak in.
SCRUB_ENV = ("VIRTUAL_ENV", "CONDA_PREFIX", "PYTHONPATH", "PYTHONHOME")

LEVELS = ("PASS", "WARN", "FAIL", "ERROR")
EXIT = {"PASS": 0, "WARN": 0, "FAIL": 1, "ERROR": 2}
RULE = "-" * 60
_COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ
_STYLE = {"PASS": "32", "WARN": "33", "FAIL": "31", "ERROR": "1;31"}
# Every recorded check as {"section", "name", "level", "detail"}.
CHECKS = []
_SECTION = "-"


def tag(level):
    return f"\033[{_STYLE[level]}m{level}\033[0m" if _COLOR else level


def worst(levels):
    return max(levels, key=LEVELS.index, default="PASS")


def line(text, i=0):
    """Line i of text (-1 for last), '' if empty."""
    lines = str(text).strip().splitlines()
    return lines[i] if lines else ""


def note(level, name, detail=""):
    print(f"{name:<10}: {tag(level)}" + (f" - {detail}" if detail else ""))
    CHECKS.append({"section": _SECTION, "name": name, "level": level, "detail": detail})


def check(name, ok, detail):
    note("PASS" if ok else "FAIL", name, detail)


def speed(name, detail, value, floor, unit):
    """PASS, or WARN when value is below floor (working but slow)."""
    slow = value < floor
    note("WARN" if slow else "PASS", name, detail + (f" (< {floor} {unit})" if slow else ""))


def run(cmd, timeout=600, env=None):
    """Run cmd, return (returncode, stdout+stderr). Missing binary => rc 127."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        return p.returncode, p.stdout + p.stderr
    except FileNotFoundError:
        return 127, f"{cmd[0]}: not found"
    except subprocess.TimeoutExpired:
        return 124, f"{cmd[0]}: timeout"


def detect_gpus():
    """Return [(bdf, family, driver, desc)] of Intel display-class PCI devices."""
    _, out = run(["lspci", "-nnk", "-d", "8086:"])
    gpus = []
    for block in re.split(r"\n(?=\S)", out):
        head = block.splitlines()[0] if block else ""
        if not re.search(r"\[03(00|02|80)\]", head):
            continue
        family = None
        if re.search(r"Battlemage|\[8086:e2[0-9a-f]{2}\]", head):
            family = "BMG"
        elif re.search(r"Ponte Vecchio|Data Center GPU Max|\[8086:0bd[0-9a-f]\]", head):
            family = "PVC"
        m = re.search(r"Kernel driver in use:\s*(\S+)", block)
        gpus.append((head.split()[0], family, m.group(1) if m else None, head))
    return gpus


def os_release():
    try:
        text = Path("/etc/os-release").read_text()
    except OSError:
        return {}
    return dict(re.findall(r'^(\w+)="?([^"\n]*)"?$', text, re.M))


def installed_packages(pkgs):
    """Subset of pkgs installed per dpkg, in one query."""
    _, out = run(["dpkg-query", "-W", "-f=${Package} ${db:Status-Abbrev}\n", *pkgs])
    return {f[0] for f in map(str.split, out.splitlines()) if f[1:2] == ["ii"]}


def clinfo_gpus():
    """Devices on Intel OpenCL GPU platforms; CPU OpenCL runtimes are not counted."""
    _, out = run(["clinfo", "-l"])
    n, gpu_platform = 0, False
    for ln in out.splitlines():
        if ln.startswith("Platform #"):
            gpu_platform = "Intel" in ln and "Graphics" in ln
        elif gpu_platform and re.search(r"Device #\d+:", ln):
            n += 1
    return n


def disk_bw(path, mib=2048):
    """Sequential direct-IO write/read MB/s via dd, bypassing page cache."""
    f = Path(path) / f"check_device_{os.getpid()}.bin"
    rates = []
    try:
        for cmd in (
            ["dd", "if=/dev/zero", f"of={f}", "bs=1M", f"count={mib}", "oflag=direct", "conv=fdatasync"],
            ["dd", f"if={f}", "of=/dev/null", "bs=1M", "iflag=direct"],
        ):
            start = time.perf_counter()
            rc, out = run(cmd)
            if rc:
                raise RuntimeError(line(out, -1))
            rates.append(mib * 2**20 / (time.perf_counter() - start) / 1e6)
    finally:
        f.unlink(missing_ok=True)
    return rates


def check_sys():
    if platform.system() != "Linux":
        return note("ERROR", "SYS", "Linux only")

    missing = [t for t in TOOLS if not shutil.which(t)]
    check("Tools", not missing, f"missing {missing}" if missing else ", ".join(TOOLS))

    # Only BMG/PVC are CI GPUs; other Intel display devices (iGPU) are ignored.
    all_gpus = detect_gpus()
    gpus = [g for g in all_gpus if g[1]]
    families = sorted({g[1] for g in gpus})
    check("GPU", bool(gpus), f"{len(gpus)} x {families}")
    for g in all_gpus:
        print(f"{'':<12}{g[3]}{'' if g[1] else '  (ignored)'}")
    if len(families) > 1:
        check("GPU mix", False, "mixed GPU families")

    osr = os_release()
    os_desc = f"{osr.get('NAME', '?')} {osr.get('VERSION_ID', '?')}, kernel {platform.release()}"
    spec = GPU_SPEC.get(families[0]) if len(families) == 1 else None
    if spec:
        want_os, want_drv = spec
        check("OS", osr.get("ID") == "ubuntu" and osr.get("VERSION_ID") == want_os,
              f"{os_desc} (want Ubuntu {want_os})")
        drivers = sorted({str(g[2]) for g in gpus})
        check("Driver", drivers == [want_drv], f"kernel driver {drivers} (want {want_drv})")
    else:
        # Family unknown: accept any supported OS so the GPU failure is not reported twice.
        known = sorted({v for v, _ in GPU_SPEC.values()})
        check("OS", osr.get("ID") == "ubuntu" and osr.get("VERSION_ID") in known,
              f"{os_desc} (GPU family unknown, want Ubuntu {'/'.join(known)})")

    have = installed_packages([p for alts in PACKAGES for p in alts])
    bad = ["|".join(alts) for alts in PACKAGES if not have.intersection(alts)]
    check("Packages", not bad, f"missing {bad}" if bad else "compute runtime installed")

    n_cl = clinfo_gpus()
    rc, out = run(["xpu-smi", "discovery", "-j"])
    try:
        n_smi = len(json.loads(out).get("device_list", [])) if rc == 0 else 0
    except ValueError:
        n_smi = 0
    detail = f"clinfo {n_cl}, xpu-smi {n_smi}, lspci {len(gpus)}"
    check("Visible", n_cl > 0 and n_smi > 0, detail)
    if n_cl and n_smi and len({n_cl, n_smi, len(gpus)}) > 1:
        note("WARN", "Visible", f"device count mismatch: {detail}")

    total_gb = shutil.disk_usage("/").total / 1e9
    check("Disk /", total_gb > MIN_DISK_GB, f"{total_gb:.0f} GB (want > {MIN_DISK_GB} GB)")
    try:
        w, r = disk_bw("/var/tmp")
        speed("Disk BW", f"write {w:.0f} MB/s, read {r:.0f} MB/s", min(w, r), DISK_WARN_MBPS, "MB/s")
    except Exception as e:
        check("Disk BW", False, line(e))

    import pwd  # Unix only; imported here so net/bw still run elsewhere.

    try:
        uid = pwd.getpwnam("jenkins").pw_uid
        check("User", uid == JENKINS_UID, f"jenkins uid {uid} (want {JENKINS_UID})")
    except KeyError:
        check("User", False, "jenkins not found")


def http_open(url, headers=None):
    # Unredirected so the registry token is not forwarded to the blob CDN.
    req = urllib.request.Request(url, headers={"User-Agent": "check_device"})
    for k, v in (headers or {}).items():
        req.add_unredirected_header(k, v)
    return urllib.request.urlopen(req, timeout=NET_TIMEOUT)


def http_get(url, headers=None, limit_mib=NET_LIMIT_MIB):
    """Stream up to limit_mib from url, return (bytes, seconds). Honors *_proxy env."""
    limit, got = limit_mib * 2**20, 0
    start = time.perf_counter()
    with http_open(url, headers) as resp:
        while got < limit and (chunk := resp.read(min(2**20, limit - got))):
            got += len(chunk)
    return got, time.perf_counter() - start


def http_json(url, headers=None):
    with http_open(url, headers) as resp:
        return json.load(resp)


def net_git():
    d = tempfile.mkdtemp(prefix="check_device_git_")
    try:
        start = time.perf_counter()
        # --no-checkout: time only the transfer, not writing the work tree.
        rc, out = run(["git", "clone", "-q", "--depth", "1", "--no-checkout",
                       "https://github.com/pytorch/pytorch", d])
        if rc:
            raise RuntimeError(line(out, -1))
        size = sum(f.stat().st_size for f in Path(d, ".git").rglob("*") if f.is_file())
        return size, time.perf_counter() - start
    finally:
        shutil.rmtree(d, ignore_errors=True)


def net_docker(repo="intel/omix"):
    """Pull the largest layer blob of the newest tag straight from the registry."""
    tags = http_json(f"https://hub.docker.com/v2/repositories/{repo}/tags?page_size=1&ordering=last_updated")
    name = tags["results"][0]["name"]
    token = http_json(f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull")
    auth = {"Authorization": f"Bearer {token['token']}"}
    accept = ", ".join([
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    ])
    base = f"https://registry-1.docker.io/v2/{repo}"
    manifest = http_json(f"{base}/manifests/{name}", {**auth, "Accept": accept})
    if "manifests" in manifest:
        digest = next((m["digest"] for m in manifest["manifests"]
                       if m.get("platform", {}).get("architecture") == "amd64"), None)
        if not digest:
            raise RuntimeError(f"no amd64 image for {repo}:{name}")
        manifest = http_json(f"{base}/manifests/{digest}", {**auth, "Accept": accept})
    layer = max(manifest["layers"], key=lambda x: x["size"])
    return http_get(f"{base}/blobs/{layer['digest']}", auth)


def net_pypi(index="https://download.pytorch.org/whl/xpu/torch/"):
    with http_open(index) as resp:
        links = re.findall(r'href="([^"]+linux_x86_64\.whl[^"]*)"', resp.read().decode())
    if not links:
        raise RuntimeError(f"no linux wheel in {index}")
    return http_get(urllib.request.urljoin(index, links[-1]))


NET_TARGETS = (
    ("Git", net_git),
    ("Docker", net_docker),
    ("PyPI", net_pypi),
    ("HF model", lambda: http_get(
        "https://huggingface.co/google-bert/bert-base-uncased/resolve/main/model.safetensors")),
    ("Weights", lambda: http_get("https://download.pytorch.org/models/resnet50-0676ba61.pth")),
)


@contextmanager
def proxy_only():
    """Drop no_proxy for the block so every target (git subprocess too) uses the proxy."""
    saved = {k: os.environ.pop(k) for k in ("no_proxy", "NO_PROXY") if k in os.environ}
    try:
        yield
    finally:
        os.environ.update(saved)


def check_net():
    with proxy_only():
        for name, fn in NET_TARGETS:
            try:
                size, secs = fn()
            except Exception as e:
                check(name, False, f"{type(e).__name__}: {line(e)}")
                continue
            mb = size / 1e6
            detail = f"{mb:8.1f} MB in {secs:6.1f} s = {mb / secs:8.1f} MB/s"
            if size:
                speed(name, detail, mb / secs, NET_WARN_MBPS, "MB/s")
            else:
                check(name, False, detail)


def bench_bw(k, pinned, mib=256, iters=20):
    import torch

    n = mib * 2**20
    h = torch.empty(n, dtype=torch.uint8)
    if pinned:
        h = h.pin_memory()
    d = torch.empty(n, dtype=torch.uint8, device=f"xpu:{k}")
    for _ in range(3):
        d.copy_(h, non_blocking=pinned)
        h.copy_(d, non_blocking=pinned)
    torch.xpu.synchronize(k)
    rates = []
    for dst, src in ((d, h), (h, d)):
        start = time.perf_counter()
        for _ in range(iters):
            dst.copy_(src, non_blocking=pinned)
        torch.xpu.synchronize(k)
        rates.append(n * iters / (time.perf_counter() - start) / 1e9)
    return rates


def bw_child():
    """Internal `_bw` mode: bench in this interpreter, print one tagged JSON line."""
    import torch

    res = {"torch": torch.__version__, "python": sys.executable, "devices": []}
    if torch.xpu.is_available():
        for k in range(torch.xpu.device_count()):
            torch.xpu.set_device(k)
            res["devices"].append([torch.xpu.get_device_name(k), *bench_bw(k, False), *bench_bw(k, True)])
    print(BW_TAG + json.dumps(res), flush=True)
    return 0


def bench_in(py, env=None):
    """Run the bench with interpreter py in a subprocess, return its JSON result."""
    rc, out = run([str(py), os.path.abspath(__file__), "_bw"], env=env)
    m = re.search(rf"^{re.escape(BW_TAG)}(.*)$", out, re.M)
    if rc or not m:
        raise RuntimeError(f"bench: {line(out, -1)}")
    return json.loads(m.group(1))


def find_uv(tmp, env):
    uv = shutil.which("uv")
    if uv:
        return uv
    installer = tmp / "uv-install.sh"
    with http_open(UV_INSTALLER) as resp:
        installer.write_bytes(resp.read())
    # Unmanaged install: no shell profile or PATH changes.
    rc, out = run(["sh", str(installer)], env={**env, "UV_UNMANAGED_INSTALL": str(tmp / "uv")})
    uv = next((str(p) for p in (tmp / "uv" / "uv", tmp / "uv" / "bin" / "uv") if p.exists()), None)
    if rc or not uv:
        raise RuntimeError(f"uv install: {line(out, -1)}")
    return uv


def venv_bench(tmp):
    """Bench in a fresh uv venv with torch XPU nightly; uv, cache, Python and venv stay in tmp."""
    # Drop user UV_* settings (e.g. UV_EXTRA_INDEX_URL) so only TORCH_INDEX is used.
    env = {k: v for k, v in os.environ.items() if k not in SCRUB_ENV and not k.startswith("UV_")}
    env.update(UV_CACHE_DIR=str(tmp / "cache"), UV_PYTHON_INSTALL_DIR=str(tmp / "python"))
    uv = find_uv(tmp, env)
    note("PASS", "uv", uv)
    py = tmp / "venv" / "bin" / "python"
    for step, cmd in (
        ("venv", [uv, "venv", "-q", "--no-config", "--python", UV_PYTHON, str(py.parent.parent)]),
        ("torch", [uv, "pip", "install", "-q", "--no-config", "--python", str(py), "--pre", "torch",
                   "--index-url", TORCH_INDEX]),
    ):
        start = time.perf_counter()
        rc, out = run(cmd, timeout=3600, env=env)
        if rc:
            raise RuntimeError(f"{step}: {line(out, -1)}")
        print(f"{'Setup':<10}: {step} done in {time.perf_counter() - start:.0f} s")
    return bench_in(py, env)


def bw_result():
    """Bench in the current env if its torch sees XPU (probe only), else in a temporary venv."""
    import importlib.util

    reason = "torch not installed"
    if importlib.util.find_spec("torch"):
        try:
            res = bench_in(sys.executable)
            if res["devices"]:
                print(f"{'Setup':<10}: current env {sys.executable}")
                return res
            reason = f"torch {res['torch']} sees no XPU"
        except Exception as e:
            reason = line(e)
    print(f"{'Setup':<10}: current env unusable ({reason}), using temporary uv venv")
    tmp = Path(tempfile.mkdtemp(prefix="check_device_bw_", dir="/var/tmp"))
    try:
        return venv_bench(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"{'Setup':<10}: removed {tmp}")


def check_bw():
    res = bw_result()
    note("PASS", "Torch", f"{res['torch']} ({res['python']})")
    if not res["devices"]:
        return note("ERROR", "BW", "torch sees no XPU")
    for k, (name, ph2d, pd2h, nh2d, nd2h) in enumerate(res["devices"]):
        note("PASS", f"xpu:{k}", f"H2D/D2H pageable {ph2d:5.1f}/{pd2h:5.1f}, "
                                 f"pinned {nh2d:5.1f}/{nd2h:5.1f} GB/s  {name}")


SECTIONS = {
    "sys": ("SYS", "System Acceptance", check_sys),
    "net": ("NET", f"Network Bandwidth (<= {NET_LIMIT_MIB} MiB each, WARN < {NET_WARN_MBPS} MB/s)", check_net),
    "bw": ("BW", "H2D/D2H Bandwidth (reference only)", check_bw),
}


def run_section(key, title, fn):
    """Run one section; any uncaught exception becomes an ERROR. Return (level, seconds)."""
    global _SECTION
    _SECTION = key
    print(f"{RULE}\n{title}\n{RULE}")
    start = time.perf_counter()
    try:
        fn()
    except Exception as e:
        note("ERROR", key, f"{type(e).__name__}: {line(e)}")
    return worst(c["level"] for c in CHECKS if c["section"] == key), time.perf_counter() - start


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("modes", nargs="*", metavar="MODE", help="all, sys, net, bw (default: all)")
    ap.add_argument("--json", metavar="FILE", help="also write all results to FILE as JSON")
    args = ap.parse_args(argv)
    unknown = sorted(set(args.modes) - {"all", *SECTIONS})
    if unknown:
        ap.error(f"unknown MODE {unknown}")
    picked = [m for m in SECTIONS if not args.modes or "all" in args.modes or m in args.modes]

    print(f"{'=' * 60}\nXPU Machine Check\n{'=' * 60}")
    print(f"{'Host':<10}: {platform.node()}")
    print(f"{'OS':<10}: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"{'Python':<10}: {platform.python_version()} ({sys.executable})")
    print(f"{'Modes':<10}: {', '.join(picked)}")

    sections = {}
    for m in picked:
        key, title, fn = SECTIONS[m]
        sections[key] = run_section(key, title, fn)
    result = worst(lv for lv, _ in sections.values())

    print(f"{'=' * 60}\nSummary\n{RULE}")
    for key, (lv, secs) in sections.items():
        print(f"{key:<10}: {tag(lv)} ({secs:.0f} s)")
    for level in ("ERROR", "FAIL", "WARN"):
        items = [c for c in CHECKS if c["level"] == level]
        if items:
            print(f"{RULE}\n{tag(level)} ({len(items)})")
            for c in items:
                print(f"  {c['section']:<4} {c['name']:<10}: {c['detail'].strip()}")
    print(f"{RULE}\n{'RESULT':<10}: {tag(result)}\n{'=' * 60}")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "host": platform.node(),
            "result": result,
            "sections": {k: {"level": lv, "seconds": round(s, 1)} for k, (lv, s) in sections.items()},
            "checks": CHECKS,
        }, indent=2))
    return EXIT[result]


if __name__ == "__main__":
    sys.exit(bw_child() if sys.argv[1:2] == ["_bw"] else main())
