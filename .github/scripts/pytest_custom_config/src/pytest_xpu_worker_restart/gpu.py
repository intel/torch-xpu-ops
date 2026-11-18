"""XPU card discovery, health probes, card recovery and health events.

Workers are pinned with ZE_AFFINITY_MASK; the runtime orders devices by PCI
BDF, so index -> BDF is fixed while all cards are present. A lost card shifts
every later index, so a slot remembers its cards by BDF and is re-indexed
before its worker is respawned.
"""

import json
import os
import platform
import re
import shlex
import subprocess
import sys
import time

CLINFO_TIMEOUT = 60
PROBE_TIMEOUT = 120
RECOVER_TIMEOUT = 600
_BDF_LINE = re.compile(
    r"PCI bus info \(KHR\)\s+PCI-E,\s*([0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7])"
)
# One line per device that ran a kernel, so a hang points at its device.
_PROBE = """
import torch
for i in range(torch.xpu.device_count()):
    x = torch.ones(1 << 20, device=f"xpu:{i}")
    assert int(x.sum().item()) == 1 << 20, f"wrong result on xpu:{i}"
    torch.xpu.synchronize(i)
    print(f"PROBE_OK {i}", flush=True)
"""


def run(cmd, timeout, env=None):
    """Return (rc, output); a child stuck in the driver (D state) is abandoned, not awaited."""
    try:
        p = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env
        )
    except FileNotFoundError:
        return 127, f"{cmd[0]}: not found"
    try:
        out, _ = p.communicate(timeout=timeout)
        return p.returncode, out
    except subprocess.TimeoutExpired:
        p.kill()
        try:
            out, _ = p.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            out = ""
        return 124, f"{out or ''}{os.path.basename(cmd[0])}: timeout after {timeout}s"


def _unmasked_env():
    return {
        k: v
        for k, v in os.environ.items()
        if k not in ("ZE_AFFINITY_MASK", "ONEAPI_DEVICE_SELECTOR")
    }


def visible_cards():
    """Sorted BDFs of Intel GPUs the compute runtime enumerates; None if clinfo failed."""
    rc, out = run(["clinfo"], CLINFO_TIMEOUT, _unmasked_env())
    if rc != 0:
        return None
    bdfs, gpu_platform = set(), False
    for ln in out.splitlines():
        m = re.match(r"\s*Platform Name\s+(.*)", ln)
        if m:
            gpu_platform = "Intel" in m[1] and "Graphics" in m[1]
        elif gpu_platform and (m := _BDF_LINE.search(ln)):
            bdfs.add(m[1].lower())
    return sorted(bdfs)


def probe(mask, timeout=PROBE_TIMEOUT):
    """Run a tiny kernel on every device of ZE_AFFINITY_MASK=mask.

    Return (ok, detail, index of the first device that failed or None).
    """
    env = _unmasked_env()
    if mask is not None:
        env["ZE_AFFINITY_MASK"] = mask
    rc, out = run([sys.executable, "-c", _PROBE], timeout, env)
    if rc == 0:
        return True, "probe ok", None
    passed = out.count("PROBE_OK")
    lines = [ln for ln in out.strip().splitlines() if not ln.startswith("PROBE_OK")]
    return (
        False,
        f"probe rc={rc} on device {passed}: {lines[-1] if lines else ''}",
        passed,
    )


class Slot:
    """The GPUs (card BDF + tile) one xdist worker is pinned to."""

    def __init__(self, mask, cards, tiles=1):
        self.mask = mask
        self.current = mask
        self.entries = []  # (card, tile), in device order
        self.excluded = set()
        self.bad = []
        # PVC exposes each tile as a device: masks index tiles, not cards.
        self.tiles = tiles
        if not cards:
            return
        count = len(cards) * self.tiles
        if mask is None:
            indexes = range(count)
        elif all(p.strip().isdigit() for p in mask.split(",")):
            indexes = sorted({int(p) for p in mask.split(",")})
        else:
            indexes = []
        self.entries = [
            (cards[i // self.tiles], i % self.tiles) for i in indexes if i < count
        ]

    @property
    def cards(self):
        return list(dict.fromkeys(c for c, _ in self.entries if c not in self.excluded))

    def __str__(self):
        cards = self.cards
        names = ", ".join(cards) if len(cards) <= 2 else f"{len(cards)} cards"
        return f"{names or 'unknown card'} (ZE_AFFINITY_MASK={self.mask if self.mask is not None else '-'})"

    def mask_for(self, visible):
        """Mask for the slot's usable cards given the cards enumerated now."""
        if not self.entries:
            return self.mask
        return ",".join(
            str(visible.index(c) * self.tiles + t)
            for c, t in self.entries
            if c in visible and c not in self.excluded
        )

    def health(self):
        """(ok, detail); sets self.bad, and self.current (the mask to respawn with) on success."""
        self.bad = []
        if self.entries:
            visible = visible_cards()
            if visible is None:
                return False, "clinfo failed or timed out"
            self.bad = [c for c in self.cards if c not in visible]
            if self.bad:
                return (
                    False,
                    f"{', '.join(self.bad)} lost (not enumerated by the runtime)",
                )
            self.current = self.mask_for(visible)
        ok, detail, failed = probe(self.current)
        if not ok:
            active = [(c, t) for c, t in self.entries if c not in self.excluded]
            if failed is not None and failed < len(active):
                self.bad = [active[failed][0]]
            return False, detail
        return True, detail

    def exclude(self, cards):
        """Drop cards from a multi-card slot; True if any card is left to run on."""
        self.excluded.update(cards)
        return bool(self.cards)


def recover(card, command=None):
    """Recover card with `detect-devices.sh --recover`; return (status, detail)."""
    command = command or os.environ.get("XPU_RECOVER_CMD")
    if not command:
        return "unavailable", "no recovery command (XPU_RECOVER_CMD unset)"
    rc, out = run([*shlex.split(command), card], RECOVER_TIMEOUT)
    for ln in reversed(out.strip().splitlines()):
        if ln.startswith("RESULT "):
            _, status, *detail = ln.split(" ", 2)
            return status, detail[0] if detail else ""
    lines = out.strip().splitlines()
    return "failed", f"rc={rc}: {lines[-1] if lines else ''}"


def health_event(event, card, detail, path=None):
    """Append a GPU health event for the UT report (no-op unless XPU_HEALTH_LOG is set)."""
    path = path or os.environ.get("XPU_HEALTH_LOG")
    if not path:
        return
    record = {
        "time": int(time.time()),
        "machine": os.environ.get("RUNNER_NAME") or platform.node(),
        "card": card,
        "event": event,
        "detail": detail,
        "source": "pytest",
    }
    try:
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        pass
