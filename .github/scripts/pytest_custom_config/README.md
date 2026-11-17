# pytest-xpu-worker-restart

A tiny [pytest-xdist](https://pypi.org/project/pytest-xdist/) plugin that restarts
an XPU worker process when a fatal failure is detected (device crash, out of
memory, segmentation fault, `ur_result_error`, etc.).

When such a failure is seen in a test report, the plugin:

1. Attempts to clean up the XPU device (`gc.collect()`, `torch.xpu.synchronize()`,
   `torch.xpu.empty_cache()`).
2. Forwards the real failure report to the xdist controller.
3. Forcibly exits the worker with code `101` so xdist respawns a fresh worker.

## Install

```bash
pip install .
```

Or, for development:

```bash
pip install -e .
```

## Usage

No configuration needed. Once installed, the plugin is auto-registered via its
`pytest11` entry point and activates automatically when running under xdist
(`pytest -n <N> ...`). This replaces the previous `conftest.py` +
`PYTHONPATH=.github/scripts/pytest_config` approach.

To disable it for a run:

```bash
pytest -p no:xpu_worker_restart ...
```

## GPU recovery (xdist controller)

A hung or lost GPU used to stall a run or keep crashing every worker respawned
on it. When a worker goes down, the controller:

1. **Reruns the test once on another worker** when the worker was unhealthy:
   crash, the fatal failures above (device lost, OOM, ...), or a GPU hang. If the
   test is unhealthy again it is reported as failed with "treated as a problem of
   this test". This replaces pytest-rerunfailures' crash reruns
   (`--only-rerun crashed.*while.*running`), which are disabled while the plugin
   is active; normal failures are untouched.
2. **Health-checks the worker's GPUs** (cards still enumerated by `clinfo`, a
   small kernel runs on every device) before respawning the worker.
   - Healthy: the worker is respawned right away.
   - Broken: the worker goes **offline**, the card is recovered with
     `detect-devices.sh --recover`, and the worker comes back **online** once the
     card checks out; other workers keep running meanwhile. A card not back
     within `--xpu-offline-timeout` (or that cannot be recovered at all) is
     **excluded**: a multi-card worker (`-n 1` distributed runs) continues on its
     other cards, a single-card worker stays offline and its remaining tests run
     on the other workers.
3. **Classifies pytest-timeout** (`--timeout_method=thread`): at the timeout the
   worker dumps every thread's stack and runs a kernel on its GPU.
   - GPU does not respond: `[xpu-timeout] gpu hang`, handled as unhealthy (1, 2).
   - GPU responds: `[xpu-timeout] test timeout`, a normal failure with the stacks
     and captured output in the XML; not rerun.
4. **Kills workers stuck on one test** past `--xpu-hang-timeout` (default
   `--timeout + 300`), e.g. a process hung in the GPU driver that cannot even
   run the timeout handler. If it cannot be killed, it is dropped and handled as
   a crash.

Workers keep their cards even when another card disappears: `ZE_AFFINITY_MASK`
indexes cards in PCI order, so a respawned worker gets its mask re-computed from
the cards' PCI addresses. GPU checks are off when the machine has no usable XPU.

Results stay one entry per test in the JUnit XML: the unhealthy attempt that is
rerun is not recorded, and the final entry lists every attempt, the worker and
card it ran on, what went wrong, and the GPU verdict after it. A test that
passed on rerun gets it as an `xpu_recovery` property; a test that failed twice
has it in its failure/error text, e.g.:

```text
[xpu-recovery] 2 unhealthy run(s), rerun limit 1: treated as a problem of this test
  attempt 1: worker gw4 on 0000:19:00.0 (ZE_AFFINITY_MASK=0): fatal error in call: RuntimeError: UR_RESULT_ERROR_DEVICE_LOST; GPU healthy afterwards
  attempt 2: worker gw5 on 0000:1d:00.0 (ZE_AFFINITY_MASK=1): fatal error in call: RuntimeError: UR_RESULT_ERROR_DEVICE_LOST; GPU check pending
```

Lines prefixed `[xpu-recovery]` in the log show each step. Options:

| Option | Default | |
|---|---|---|
| `--xpu-crash-reruns N` | 1 | reruns of an unhealthy test, 0 disables |
| `--xpu-hang-timeout S` | `--timeout` + 300 | kill a worker stuck this long, 0 disables |
| `--xpu-offline-timeout S` | 600 | how long a broken card may take to recover before it is excluded |
| `--xpu-recover-cmd CMD` | `$XPU_RECOVER_CMD` | command run with a card's PCI BDF to recover it |

### Card recovery and the UT report

[`detect-devices.sh`](../detect-devices.sh) (run by the `linux-testenv` action)
checks every Intel GPU on PCI before the tests: a card the compute runtime
cannot use is recovered, or left out of `ZE_AFFINITY_MASK` if that fails. It
exports `XPU_RECOVER_CMD` (`detect-devices.sh --recover`) for the plugin and
`XPU_HEALTH_LOG`, where both log GPU health events.

`detect-devices.sh --recover <BDF>` kills the processes holding the card,
unbinds it from `xe`/`i915`, PCI-resets it and binds it again. It needs root or
passwordless sudo and a writable `/sys`: bare-metal jobs have that, unprivileged
containers do not (status `no-privilege`, the card is excluded instead). If the
kernel already oopsed or the unbind hangs, the status is `reboot-required`.

`linux-uttest` copies the events into the UT artifact (`ut_log/env/xpu_health.jsonl`)
and the UT report lists them under **GPU health**: machine, card, event
(`unhealthy`, `recovered`, `excluded`, `reboot-required`, ...) and detail.
