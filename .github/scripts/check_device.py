#!/usr/bin/env python3
"""Check an XPU CI machine: system acceptance, network and H2D/D2H bandwidth.

SYS: new machine acceptance (Linux only). Intel GPU on PCI, OS version per GPU
(BMG: Ubuntu 26.04, PVC: Ubuntu 24.04), kernel driver (BMG: xe, PVC: i915) and
compute runtime packages, tools (xpu-smi, clinfo, lspci, ocloc), GPU visible to
clinfo/xpu-smi, / size > 800 GB plus disk bandwidth, user jenkins with UID 1001.

NET: download speed of pytorch/pytorch git, intel/omix docker image, torch XPU
wheel, Hugging Face model and torch weights. Unreachable => FAIL, slow => WARN.

Summary lists every ERROR, FAIL and WARN item (colored on a terminal).

BW: H2D/D2H copy bandwidth per device, with pageable and pinned host memory.
Needs torch with XPU. Reported for reference only.

Usage: python check_device.py [MODE]
       MODE = all | sys | net | bw (default all)
Exit:  0 PASS, 1 FAIL, 2 error
"""

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
from pathlib import Path

MODES = ("all", "sys", "net", "bw")
MODE = sys.argv[1] if len(sys.argv) > 1 else "all"
if MODE not in MODES:
    print(f"Usage: python {Path(sys.argv[0]).name} [{'|'.join(MODES)}]")
    sys.exit(2)

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

# (level, name, detail) of every FAIL/WARN/ERROR, listed again in the summary.
ISSUES = []
_COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ
_STYLE = {"PASS": "32", "WARN": "33", "FAIL": "31", "ERROR": "1;31"}


def tag(level):
    return f"\033[{_STYLE[level]}m{level}\033[0m" if _COLOR else level


def note(level, name, detail):
    """Print one check line; record anything that is not PASS for the summary."""
    print(f"{name:<10}: {tag(level)} - {detail}")
    if level != "PASS":
        ISSUES.append((level, name, detail))


def run(cmd, timeout=600):
    """Run cmd, return (returncode, stdout+stderr). Missing binary => rc 127."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout + p.stderr
    except FileNotFoundError:
        return 127, f"{cmd[0]}: not found"
    except subprocess.TimeoutExpired:
        return 124, f"{cmd[0]}: timeout"


def report(name, ok, detail):
    note("PASS" if ok else "FAIL", name, detail)
    return ok


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
                raise RuntimeError(out.strip().splitlines()[-1])
            rates.append(mib * 2**20 / (time.perf_counter() - start) / 1e6)
    finally:
        f.unlink(missing_ok=True)
    return rates


def check_sys():
    print("-" * 60 + "\nSystem Acceptance Check\n" + "-" * 60)
    if platform.system() != "Linux":
        note("ERROR", "SYS", "Linux only")
        return 2
    results = []

    missing = [t for t in TOOLS if not shutil.which(t)]
    results.append(report("Tools", not missing, f"missing {missing}" if missing else ", ".join(TOOLS)))

    # Only BMG/PVC are CI GPUs; other Intel display devices (iGPU) are ignored.
    all_gpus = detect_gpus()
    gpus = [g for g in all_gpus if g[1]]
    families = {g[1] for g in gpus}
    results.append(report("GPU", bool(gpus), f"{len(gpus)} x {sorted(families)}"))
    for gpu in all_gpus:
        print(f"            {gpu[3]}{'' if gpu[1] else '  (ignored)'}")
    family = next(iter(families)) if len(families) == 1 else None
    if len(families) > 1:
        results.append(report("GPU mix", False, "mixed GPU families"))

    osr = os_release()
    os_desc = f"{osr.get('NAME', '?')} {osr.get('VERSION_ID', '?')}, kernel {platform.release()}"
    if family in GPU_SPEC:
        want_os, want_drv = GPU_SPEC[family]
        results.append(report("OS", osr.get("ID") == "ubuntu" and osr.get("VERSION_ID") == want_os,
                              f"{os_desc} (want Ubuntu {want_os})"))
        drivers = {g[2] for g in gpus}
        results.append(report("Driver", drivers == {want_drv},
                              f"kernel driver {sorted(map(str, drivers))} (want {want_drv})"))
    else:
        results.append(report("OS", False, f"{os_desc} (unknown GPU family)"))

    bad = []
    for alts in PACKAGES:
        if not any(run(["dpkg-query", "-W", "-f=${Status}", p])[1].startswith("install ok installed") for p in alts):
            bad.append("|".join(alts))
    results.append(report("Packages", not bad, f"missing {bad}" if bad else "compute runtime installed"))

    _, out = run(["clinfo", "-l"])
    n_cl = len(re.findall(r"Device #\d+:.*Intel", out))
    rc, out = run(["xpu-smi", "discovery", "-j"])
    try:
        n_smi = len(json.loads(out).get("device_list", [])) if rc == 0 else 0
    except ValueError:
        n_smi = 0
    detail = f"clinfo {n_cl}, xpu-smi {n_smi}, lspci {len(gpus)}"
    results.append(report("Visible", n_cl > 0 and n_smi > 0, detail))
    if n_cl > 0 and n_smi > 0 and len({n_smi, len(gpus)}) > 1:
        note("WARN", "Visible", f"device count mismatch: {detail}")

    total_gb = shutil.disk_usage("/").total / 1e9
    results.append(report("Disk /", total_gb > MIN_DISK_GB, f"{total_gb:.0f} GB (want > {MIN_DISK_GB} GB)"))
    try:
        w, r = disk_bw("/var/tmp")
        slow = min(w, r) < DISK_WARN_MBPS
        note("WARN" if slow else "PASS", "Disk BW",
             f"write {w:.0f} MB/s, read {r:.0f} MB/s" + (f" (< {DISK_WARN_MBPS} MB/s)" if slow else ""))
    except Exception as e:
        results.append(report("Disk BW", False, str(e)))

    import pwd

    try:
        uid = pwd.getpwnam("jenkins").pw_uid
        results.append(report("User", uid == JENKINS_UID, f"jenkins uid {uid} (want {JENKINS_UID})"))
    except KeyError:
        results.append(report("User", False, "jenkins not found"))

    return 0 if all(results) else 1


def http_get(url, headers=None, limit_mib=NET_LIMIT_MIB):
    """Stream up to limit_mib from url, return (bytes, seconds). Honors *_proxy env."""
    req = urllib.request.Request(url, headers={"User-Agent": "check_device", **(headers or {})})
    limit, got = limit_mib * 2**20, 0
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=NET_TIMEOUT) as resp:
        while got < limit and (chunk := resp.read(min(2**20, limit - got))):
            got += len(chunk)
    return got, time.perf_counter() - start


def http_json(url, headers=None):
    req = urllib.request.Request(url, headers={"User-Agent": "check_device", **(headers or {})})
    with urllib.request.urlopen(req, timeout=NET_TIMEOUT) as resp:
        return json.load(resp)


def net_git():
    d = tempfile.mkdtemp(prefix="check_device_git_")
    try:
        start = time.perf_counter()
        rc, out = run(["git", "clone", "-q", "--depth", "1", "https://github.com/pytorch/pytorch", d])
        if rc:
            raise RuntimeError(out.strip().splitlines()[-1])
        size = sum(f.stat().st_size for f in Path(d, ".git").rglob("*") if f.is_file())
        return size, time.perf_counter() - start
    finally:
        shutil.rmtree(d, ignore_errors=True)


def net_docker(repo="intel/omix"):
    """Pull the largest layer blob of the newest tag straight from the registry."""
    tag = http_json(f"https://hub.docker.com/v2/repositories/{repo}/tags?page_size=1&ordering=last_updated")
    tag = tag["results"][0]["name"]
    token = http_json(f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull")
    auth = {"Authorization": f"Bearer {token['token']}"}
    accept = ", ".join([
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    ])
    base = f"https://registry-1.docker.io/v2/{repo}"
    manifest = http_json(f"{base}/manifests/{tag}", {**auth, "Accept": accept})
    if "manifests" in manifest:
        digest = next(m["digest"] for m in manifest["manifests"]
                      if m.get("platform", {}).get("architecture") == "amd64")
        manifest = http_json(f"{base}/manifests/{digest}", {**auth, "Accept": accept})
    layer = max(manifest["layers"], key=lambda x: x["size"])
    return http_get(f"{base}/blobs/{layer['digest']}", auth)


def net_pypi(index="https://download.pytorch.org/whl/xpu/torch/"):
    with urllib.request.urlopen(index, timeout=NET_TIMEOUT) as resp:
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


def check_net():
    print("-" * 60 + f"\nNetwork Bandwidth (<= {NET_LIMIT_MIB} MiB each, WARN < {NET_WARN_MBPS} MB/s)\n" + "-" * 60)
    # Force every target through the proxy; also inherited by the git subprocess.
    for var in ("no_proxy", "NO_PROXY"):
        os.environ.pop(var, None)
    results = []
    for name, fn in NET_TARGETS:
        try:
            size, secs = fn()
            mb = size / 1e6
            slow = mb / secs < NET_WARN_MBPS
            detail = f"{mb:8.1f} MB in {secs:6.1f} s = {mb / secs:8.1f} MB/s"
            if size == 0:
                results.append(report(name, False, detail))
            else:
                note("WARN" if slow else "PASS", name, detail + (f" (< {NET_WARN_MBPS} MB/s)" if slow else ""))
        except Exception as e:
            msg = str(e).strip().splitlines()
            results.append(report(name, False, f"{type(e).__name__}: {msg[0] if msg else ''}"))
    return 0 if all(results) else 1


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


def check_bw():
    print("-" * 60 + "\nMemory Bandwidth (GB/s, reference only)\n" + "-" * 60)
    try:
        import torch
    except ImportError:
        note("ERROR", "BW", "torch not installed")
        return 2
    if not torch.xpu.is_available():
        note("ERROR", "BW", "XPU not available")
        return 2
    print(f"{'Device':<8}{'Pageable H2D':>14}{'D2H':>8}{'Pinned H2D':>14}{'D2H':>8}")
    try:
        for k in range(torch.xpu.device_count()):
            torch.xpu.set_device(k)
            ph2d, pd2h = bench_bw(k, pinned=False)
            nh2d, nd2h = bench_bw(k, pinned=True)
            print(f"{'xpu:' + str(k):<8}{ph2d:>14.1f}{pd2h:>8.1f}{nh2d:>14.1f}{nd2h:>8.1f}")
    except Exception as e:
        msg = str(e).strip().splitlines()
        note("ERROR", "BW", f"{type(e).__name__}: {msg[0] if msg else ''}")
        return 2
    print("BW        : DONE")
    return 0


def main():
    print("=" * 60 + "\nXPU Machine Check\n" + "=" * 60)
    print(f"Mode      : {MODE}")
    print(f"OS        : {platform.system()} {platform.release()} ({platform.machine()})")

    codes = {}
    if MODE in ("all", "sys"):
        codes["SYS"] = check_sys()
    if MODE in ("all", "net"):
        codes["NET"] = check_net()
    if MODE in ("all", "bw"):
        codes["BW"] = check_bw()
    code = max(codes.values())
    print("=" * 60 + "\nSummary\n" + "-" * 60)
    for name, c in codes.items():
        print(f"{name:<10}: {tag(('PASS', 'FAIL', 'ERROR')[c])}")
    for level, title in (("ERROR", "Errors"), ("FAIL", "Failures"), ("WARN", "Warnings")):
        items = [(n, d) for lv, n, d in ISSUES if lv == level]
        if items:
            print("-" * 60 + f"\n{tag(level)} {title} ({len(items)})")
            for n, d in items:
                print(f"  {n:<10}: {d.strip()}")
    print("-" * 60)
    print(f"RESULT    : {tag(('PASS', 'FAIL', 'ERROR')[code])}")
    print("=" * 60)
    return code


if __name__ == "__main__":
    sys.exit(main())
