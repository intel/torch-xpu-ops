"""xdist controller side: keep runs going when a GPU hangs or is lost.

When a worker goes down (crash, GPU hang found at pytest-timeout, or a hang the
watchdog kills), its GPUs are health-checked before the worker is respawned:

* healthy  -> the worker is respawned right away;
* broken   -> the worker stays offline while `detect-devices.sh --recover`
              recovers the card, and comes back online once it is healthy.
              A card not back within --xpu-offline-timeout is excluded: a
              multi-card worker (-n 1 distributed) continues on the other
              cards, a single-card worker stays offline.

The test that was running is rerun once on another worker; if it is unhealthy
again it is reported as a failure of the test itself, with every attempt and
the GPU verdict after it in the report (and so in the JUnit XML). A plain test
timeout (GPU still responsive) is a normal failure, not rerun.
"""

import os
import signal
import threading
import time
import types

import pytest

from . import gpu
from .plugin import GPU_HANG_TAG, TEST_TIMEOUT_TAG, is_fatal

TICK = 10  # watchdog period, seconds
POLL = 15  # health re-check period while a worker is offline, seconds
RETRY = 300  # retry recovering a card this often while it is still bad
# Recovery outcomes that will not change by retrying.
FINAL = ("reboot-required", "no-privilege", "unavailable", "refused")
# Outcome of reports already accounted for elsewhere: not counted, not in XML.
HIDDEN = "xpu-hidden"


class _Offline:
    """Stands in for an offline worker so xdist does not end the run while it recovers."""

    def __init__(self, node):
        self.node = node

    def __repr__(self):
        return f"<offline {self.node.gateway.id}>"


class XpuController:
    def __init__(self, config, dsession):
        self.config = config
        self.ds = dsession
        self.reruns = config.getoption("xpu_crash_reruns")
        self.offline_timeout = config.getoption("xpu_offline_timeout")
        hang = config.getoption("xpu_hang_timeout")
        if hang is None:
            timeout = config.getoption("timeout", None) or 0
            # Workers classify their own timeouts; this only catches ones that cannot.
            hang = int(timeout) + 300 if timeout else 0
        self.hang_limit = hang
        self.recover_cmd = config.getoption("xpu_recover_cmd")
        self.attempts = {}  # nodeid -> [{"worker", "gpu", "reason", "after"}]
        self.fatal = {}  # node -> (nodeid it already reported, rerun it?)
        self.running = {}  # node -> (nodeid, start)
        self.killed = {}  # node -> kill time
        self.down_reason = {}
        self.cards = None
        self.tiles = 1
        self.armed = False
        self.stop = threading.Event()
        self._card_locks = {}
        self._recovered_at = {}
        self._locks_guard = threading.Lock()

        self._clone = dsession._clone_node
        dsession._clone_node = self._guard(self._defer_clone, fallback=self._clone)
        dsession.worker_xpu_log = dsession.report_line
        dsession.worker_xpu_respawn = self._guard(self._respawn)
        dsession.worker_xpu_watchdog = self._guard(self._watchdog)
        logstart, logfinish = dsession.worker_logstart, dsession.worker_logfinish

        def worker_logstart(node, nodeid, location):
            self.running[node] = (nodeid, time.monotonic())
            logstart(node=node, nodeid=nodeid, location=location)

        def worker_logfinish(node, nodeid, location):
            self.running.pop(node, None)
            logfinish(node=node, nodeid=nodeid, location=location)

        dsession.worker_logstart = worker_logstart
        dsession.worker_logfinish = worker_logfinish
        if self.reruns:
            _drop_rerunfailures_crash_hook(config.pluginmanager)

    def _guard(self, fn, fallback=None):
        """A recovery bug must never abort the test run: log it and fall back to xdist."""

        def call(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                self.ds.report_line(
                    f"[xpu-recovery] internal error in {fn.__name__}: {e!r}"
                )
                return fallback(*args, **kwargs) if fallback else None

        return call

    def start(self):
        self.cards = gpu.visible_cards()
        self.tiles = int(os.environ.get("XPU_TILES_PER_CARD") or 1)
        # Without a usable XPU a failed probe means nothing: never take workers offline.
        self.armed = bool(self.cards) or gpu.probe(None)[0]
        self._log(
            f"[xpu-recovery] cards {self.cards}, GPU health checks {'on' if self.armed else 'off'}, "
            f"hang watchdog {f'{self.hang_limit}s' if self.hang_limit else 'off'}, "
            f"recovery {self.recover_cmd or os.environ.get('XPU_RECOVER_CMD') or 'unavailable'}"
        )
        threading.Thread(target=self._tick, name="xpu-watchdog", daemon=True).start()

    # -- hooks -----------------------------------------------------------------

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_logreport(self, report):
        node = getattr(report, "node", None)
        if node is None or report.when == "???" or report.outcome in ("rerun", HIDDEN):
            return
        if is_fatal(report):
            # The worker restarts itself after this report: an unhealthy attempt.
            crash = getattr(report.longrepr, "reprcrash", None)
            lines = str(report.longrepr).strip().splitlines()
            msg = next(
                (ln for ln in lines if ln.startswith(GPU_HANG_TAG)),
                crash.message if crash else lines[-1],
            )
            n = self._record(
                report.nodeid, node, f"fatal error in {report.when}: {msg[:200]}"
            )
            self.fatal[node] = (report.nodeid, n <= self.reruns)
            if n <= self.reruns:
                report.outcome = "rerun"
            else:
                _explain(report, self._history(report.nodeid))
        elif TEST_TIMEOUT_TAG in str(report.longrepr):
            # A real timeout, GPU fine: the worker exits, the result stands.
            self.fatal[node] = (report.nodeid, False)
            if report.nodeid in self.attempts:
                _explain(report, self._history(report.nodeid))
        elif report.when == "teardown" and report.nodeid in self.attempts:
            report.user_properties.append(
                ("xpu_recovery", self._history(report.nodeid))
            )

    @pytest.hookimpl(tryfirst=True)
    def pytest_handlecrashitem(self, crashitem, report, sched):
        node = getattr(report, "node", None)
        reason = self.down_reason.pop(node, None)
        reported, rerun = self.fatal.pop(node, (None, False))
        if reported == crashitem:
            # Already reported (and counted) from the worker before it exited.
            report.outcome = HIDDEN
            if rerun:
                self._reschedule(crashitem, sched)
            return
        n = self._record(
            crashitem,
            node,
            reason or "worker process died (crash, abort or pytest-timeout)",
        )
        if n <= self.reruns and self._reschedule(crashitem, sched):
            report.outcome = "rerun"
            report.rerun = n - 1
            return
        report.longrepr = f"{report.longrepr}\n{self._history(crashitem)}"

    def pytest_report_teststatus(self, report):
        if report.outcome == "rerun":
            return "rerun", "R", ("RERUN", {"yellow": True})
        if report.outcome == HIDDEN:
            return "", "", ""
        return None

    def pytest_testnodedown(self, node, error):
        self.running.pop(node, None)
        self.killed.pop(node, None)
        if error and node not in self.down_reason:
            self.down_reason[node] = f"worker process died ({error})"

    @pytest.hookimpl(tryfirst=True)
    def pytest_sessionfinish(self, session):
        self.stop.set()
        _prune_junit(session.config)

    def _record(self, nodeid, node, reason):
        attempts = self.attempts.setdefault(nodeid, [])
        attempts.append(
            {
                "worker": node.gateway.id,
                "gpu": str(_slot(node, self)),
                "reason": reason,
                "after": None,
            }
        )
        return len(attempts)

    def _reschedule(self, nodeid, sched):
        try:
            sched.mark_test_pending(nodeid)
        except NotImplementedError:
            return False
        self._log(
            f"[xpu-recovery] rerun {nodeid} on another worker ({len(self.attempts[nodeid])}/{self.reruns})"
        )
        return True

    def _history(self, nodeid):
        attempts = self.attempts[nodeid]
        lines = [
            f"[xpu-recovery] {len(attempts)} unhealthy run(s), rerun limit {self.reruns}"
            + (
                ": treated as a problem of this test"
                if len(attempts) > self.reruns
                else ""
            )
        ]
        for i, a in enumerate(attempts, 1):
            after = a["after"] or "GPU check pending"
            lines.append(
                f"  attempt {i}: worker {a['worker']} on {a['gpu']}: {a['reason']}; {after}"
            )
        return "\n".join(lines)

    # -- worker lifecycle (main thread) ----------------------------------------

    def _defer_clone(self, node):
        """Replaces DSession._clone_node: respawn only after the GPU checks out."""
        if not self.armed:
            for attempts in self.attempts.values():
                for a in attempts:
                    if a["worker"] == node.gateway.id and a["after"] is None:
                        a["after"] = "GPU checks off"
            return self._clone(node)
        placeholder = _Offline(node)
        self.ds._active_nodes.add(placeholder)
        threading.Thread(
            target=self._heal, args=(node, placeholder), daemon=True
        ).start()

    def _respawn(self, node, placeholder, healthy, detail, mask, was_offline, cause):
        self.ds._active_nodes.discard(placeholder)
        wid, slot = node.gateway.id, _slot(node, self)
        if not was_offline:
            after = "GPU healthy afterwards"
        elif healthy and slot.excluded:
            after = f"GPU broken ({cause}), excluded {', '.join(sorted(slot.excluded))}"
        else:
            after = f"GPU broken ({cause}), " + (
                "recovered" if healthy else "not recovered"
            )
        for attempts in self.attempts.values():
            for a in attempts:
                if a["worker"] == wid and a["after"] is None:
                    a["after"] = after
        if (
            self.stop.is_set()
            or self.ds.sched.tests_finished
            or getattr(self.ds, "shouldstop", False)
        ):
            if healthy and was_offline:
                self._log(
                    f"[xpu-recovery] {slot} recovered ({detail}); no tests left for its worker"
                )
            return
        if not healthy:
            self._log(
                f"[xpu-recovery] worker {wid} stays OFFLINE for the rest of the run: {slot}: {detail}"
            )
            for card in slot.cards:
                gpu.health_event("excluded", card, f"worker {wid} offline: {detail}")
            if not self.ds._active_nodes:
                self.ds.shouldstop = (
                    "all XPU workers are offline (GPUs could not be recovered)"
                )
            return
        env = node.gateway.spec.env
        if (
            mask is not None
            and env.get("ZE_AFFINITY_MASK", os.environ.get("ZE_AFFINITY_MASK")) != mask
        ):
            self._log(
                f"[xpu-recovery] worker for {slot} now runs with ZE_AFFINITY_MASK={mask}"
            )
            env["ZE_AFFINITY_MASK"] = mask
        # Same as xdist's own respawn: tests are left, so the session is not ending.
        self.ds.shuttingdown = False
        self._clone(node)
        if was_offline:
            self._log(f"[xpu-recovery] worker for {slot} back ONLINE ({detail})")

    def _watchdog(self):
        offline = [n for n in self.ds._active_nodes if isinstance(n, _Offline)]
        if (
            offline
            and len(offline) == len(self.ds._active_nodes)
            and self.ds.sched.tests_finished
        ):
            self._log(
                f"[xpu-recovery] all tests done; not waiting for {len(offline)} offline worker(s)"
            )
            self.ds._active_nodes.difference_update(offline)
            self.stop.set()
            return
        if not self.hang_limit:
            return
        now = time.monotonic()
        for node, (nodeid, start) in list(self.running.items()):
            if now - start < self.hang_limit:
                continue
            killed = self.killed.get(node)
            if killed is None:
                reason = f"hung: {nodeid} ran {now - start:.0f}s > {self.hang_limit}s"
                self.down_reason[node] = reason
                self.killed[node] = now
                self._log(
                    f"[xpu-recovery] worker {node.gateway.id} {reason}; killing it"
                )
                pid = getattr(getattr(node.gateway, "_cache_rinfo", None), "pid", None)
                if pid:
                    try:
                        os.kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))
                    except OSError:
                        pass
            elif now - killed > TICK / 2 and not node._down:
                # Killed but still alive: stuck in the GPU driver. Drop it and move on.
                self.running.pop(node, None)
                node._down = True
                node.putevent = lambda event: None
                self.ds.queue.put(
                    (
                        "errordown",
                        {"node": node, "error": self.down_reason.get(node, "hung")},
                    )
                )

    # -- background threads ----------------------------------------------------

    def _heal(self, node, placeholder):
        slot = _slot(node, self)
        ok, detail = slot.health()
        offline, cause = not ok, detail
        if offline:
            self._log(
                f"[xpu-recovery] worker {node.gateway.id} OFFLINE: {slot}: {detail}"
            )
            for card in slot.bad or slot.cards:
                gpu.health_event("unhealthy", card, detail)
            ok, detail = self._recover(slot, node.gateway.id, detail)
        self._post(
            "xpu_respawn",
            node=node,
            placeholder=placeholder,
            healthy=ok,
            detail=detail,
            mask=slot.current,
            was_offline=offline,
            cause=cause,
        )

    def _recover(self, slot, wid, detail):
        deadline = time.monotonic() + self.offline_timeout
        give_up = False
        while True:
            ok, detail = slot.health()
            if ok:
                if slot.excluded:
                    return True, f"without {', '.join(sorted(slot.excluded))}, {detail}"
                return True, f"recovered, {detail}"
            if self.stop.is_set():
                return False, detail
            bad = slot.bad
            if give_up or time.monotonic() > deadline:
                if bad and len(slot.cards) > len(bad) and slot.exclude(bad):
                    for card in bad:
                        gpu.health_event("excluded", card, detail)
                    self._log(
                        f"[xpu-recovery] worker {wid}: excluded {', '.join(bad)}, "
                        f"continuing on {len(slot.cards)} card(s)"
                    )
                    give_up, deadline = False, time.monotonic() + self.offline_timeout
                    continue
                return False, detail
            for card in bad:
                status = self._recover_card(card, wid)
                give_up = give_up or status in FINAL
            self.stop.wait(POLL)

    def _recover_card(self, card, wid):
        """Run detect-devices.sh --recover on card unless it was just tried."""
        with self._lock_for(card):
            last = self._recovered_at.get(card)
            if last and time.monotonic() - last[0] < RETRY:
                return last[1]
            self._log(f"[xpu-recovery] worker {wid}: recovering {card}")
            status, info = gpu.recover(card, self.recover_cmd)
            self._recovered_at[card] = (time.monotonic(), status)
        self._log(f"[xpu-recovery] {card} recovery: {status}: {info}")
        gpu.health_event("recovered" if status == "ok" else status, card, info)
        return status

    def _lock_for(self, card):
        with self._locks_guard:
            return self._card_locks.setdefault(card, threading.Lock())

    def _tick(self):
        while not self.stop.wait(TICK):
            self._post("xpu_watchdog")

    def _post(self, name, **kwargs):
        self.ds.queue.put((name, kwargs))

    def _log(self, line):
        if threading.current_thread() is threading.main_thread():
            self.ds.report_line(line)
        else:
            self._post("xpu_log", line=line)


def _slot(node, ctl):
    spec = node.gateway.spec
    slot = getattr(spec, "_xpu_slot", None)
    if slot is None:
        mask = spec.env.get("ZE_AFFINITY_MASK", os.environ.get("ZE_AFFINITY_MASK"))
        slot = spec._xpu_slot = gpu.Slot(mask, ctl.cards, ctl.tiles)
    return slot


def _explain(report, text):
    """Append text to a failure so it shows in the terminal and the JUnit XML."""
    if hasattr(report.longrepr, "addsection"):
        report.longrepr.addsection("xpu-recovery", text)
    else:
        report.longrepr = f"{report.longrepr}\n{text}"


def _prune_junit(config):
    """Drop the empty <testcase> junitxml opens for reports it does not record (reruns)."""
    try:
        from _pytest.junitxml import xml_key

        xml = config.stash.get(xml_key, None)
        if xml is not None:
            # Finalized reporters have no attrs left; only unfinalized ones can be empty.
            xml.node_reporters_ordered[:] = [
                r
                for r in xml.node_reporters_ordered
                if "name" in getattr(r, "attrs", {"name": None})
            ]
    except Exception:
        pass


def _drop_rerunfailures_crash_hook(pm):
    """This plugin owns crash reruns; keep pytest-rerunfailures from rescheduling too."""
    for plugin in pm.get_plugins():
        cls = type(plugin)
        if cls.__module__ == "pytest_rerunfailures" and hasattr(
            plugin, "pytest_handlecrashitem"
        ):
            name = pm.get_name(plugin)
            pm.unregister(plugin)
            hooks = {
                n: getattr(plugin, n) for n in dir(plugin) if n.startswith("pytest_")
            }
            hooks.pop("pytest_handlecrashitem")
            pm.register(types.SimpleNamespace(**hooks), name)
