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

A hung or lost GPU used to stall a run, or keep crashing every worker that
xdist respawned on it. With this plugin a run keeps going, the results stay
accurate, and broken cards are recovered or set aside.

### What it does

| Situation | Action |
|---|---|
| Worker crashes, hits a fatal XPU error, or its GPU hangs | Test is **rerun once on another worker**; the worker's GPUs are checked before it is respawned |
| Worker's GPU is broken | Worker goes **offline**, the card is **recovered**, the worker comes back **online** |
| Card not back within 10 min, or not recoverable | Card is **excluded**; remaining tests run on the other GPUs |
| pytest-timeout fires, GPU still works | **Real timeout**: normal failure with stacks in the XML, no rerun |
| Test is unhealthy again on its rerun | Reported as a **problem of the test**, with every attempt in the XML |

### Architecture

```mermaid
flowchart LR
  subgraph setup["linux-testenv"]
    DD["detect-devices.sh<br/>pre-test card check"]
  end
  subgraph run["pytest -n N / --tx ... (linux-uttest)"]
    C["xdist controller<br/>+ XpuController"]
    W0["worker gw0<br/>ZE_AFFINITY_MASK=0"]
    W1["worker gw1<br/>ZE_AFFINITY_MASK=1"]
    Wn["worker gwN ..."]
    C <--> W0 & W1 & Wn
  end
  subgraph gpus["Host GPUs"]
    G0[(card 0)]
    G1[(card 1)]
    Gn[(card N)]
  end
  W0 --- G0
  W1 --- G1
  Wn --- Gn
  DD -- "ZE_AFFINITY_MASK<br/>XPU_RECOVER_CMD<br/>XPU_HEALTH_LOG" --> C
  C -- "detect-devices.sh --recover BDF" --> gpus
  DD -- "recover / exclude" --> gpus
  DD & C -- events --> H[("xpu_health.jsonl")]
  H --> A["UT artifact<br/>ut_log/env/"] --> R["ut_results.py<br/>UT report: GPU health"]
```

| Piece | Where | Role |
|---|---|---|
| `plugin.py` | every pytest process | options; on workers: restart on fatal errors, classify pytest-timeout |
| `controller.py` | xdist controller | worker lifecycle, reruns, watchdog, JUnit XML accuracy |
| `gpu.py` | controller and workers | card discovery (`clinfo`), GPU probe, recovery call, health events |
| [`detect-devices.sh`](../detect-devices.sh) | CI setup step | pre-test card check, `--recover <BDF>`, exports the env vars above |
| [`ut_results.py`](../ut_results.py) | UT report job | renders the GPU health events |

### Worker lifecycle

Each xdist worker is pinned to one or more GPUs (a *slot*: card PCI address +
tile). When a worker goes down, its slot is checked before it comes back:

```mermaid
stateDiagram-v2
  [*] --> Online
  Online --> Checking: worker down (crash / fatal error / GPU hang / killed)
  Checking --> Online: GPUs healthy, respawn now
  Checking --> Offline: GPU broken
  Offline --> Recovering: detect-devices.sh --recover BDF
  Recovering --> Online: card healthy again
  Recovering --> Offline: retry every 5 min
  Offline --> Excluded: 10 min passed, or reboot-required / no-privilege
  Excluded --> Online: multi-card worker, without the bad card
  Excluded --> [*]: single-card worker, stays offline
```

- **Health check**: the card is still listed by `clinfo`, and a small kernel
  runs on every device of the slot.
- **While offline**, the other workers keep taking tests. The offline worker
  keeps the run alive only while tests are left.
- **Distributed runs** use `-n 1` with all cards in one worker: a bad card is
  excluded and the worker continues on the others.
- **Mask remap**: `ZE_AFFINITY_MASK` indexes cards in PCI order, so a lost card
  shifts every later index. Slots remember their cards by PCI address and get a
  recomputed mask when respawned.

### Rerun and result accuracy

```mermaid
flowchart TD
  U["unhealthy attempt<br/>(crash, fatal error, GPU hang, killed)"] --> N{"first time<br/>for this test?"}
  N -- yes --> RR["hidden from XML (outcome RERUN)<br/>rerun on another worker"]
  N -- no --> F["FAILED: treated as a problem of this test<br/>+ every attempt in the failure text"]
  RR --> P{"rerun result"}
  P -- "pass / normal failure" --> OK["recorded as is<br/>+ xpu_recovery property with the history"]
  P -- unhealthy again --> F
```

The JUnit XML keeps exactly one entry per test: no empty `<testcase>` for the
rerun attempt and no FAILED+PASSED pair for the same test. A failure after
reruns explains itself:

```text
[xpu-recovery] 2 unhealthy run(s), rerun limit 1: treated as a problem of this test
  attempt 1: worker gw4 on 0000:19:00.0 (ZE_AFFINITY_MASK=0): fatal error in call: RuntimeError: UR_RESULT_ERROR_DEVICE_LOST; GPU healthy afterwards
  attempt 2: worker gw5 on 0000:1d:00.0 (ZE_AFFINITY_MASK=1): fatal error in call: RuntimeError: UR_RESULT_ERROR_DEVICE_LOST; GPU check pending
```

This replaces pytest-rerunfailures' crash reruns
(`--only-rerun crashed.*while.*running`), which are disabled while the plugin is
active; normal-failure reruns are untouched.

### Timeouts: GPU hang or slow test?

With `--timeout_method=thread`, the worker handles pytest-timeout itself:

```mermaid
flowchart TD
  T["pytest-timeout fires"] --> S["dump the stacks of all threads"]
  S --> P{"run a kernel on<br/>this worker's GPU"}
  P -- "fails / hangs" --> H["[xpu-timeout] gpu hang<br/>unhealthy: rerun + GPU recovery"]
  P -- works --> R["[xpu-timeout] test timeout<br/>normal failure (stacks, captured output), no rerun"]
  H & R --> X["report sent, worker exits"]
```

A worker that cannot even run this (stuck in the GPU driver holding the GIL) is
killed by the controller's watchdog after `--timeout + 300` s and handled as a
hang. If it cannot be killed, it is dropped and the run moves on.

### Card recovery

`detect-devices.sh --recover <BDF>` prints `RESULT <status> <detail>`:

```mermaid
flowchart LR
  A{"kernel oopsed?<br/>(taint D)"} -- yes --> RB["reboot-required"]
  A -- no --> B{"root/sudo +<br/>writable /sys?"}
  B -- no --> NP["no-privilege"]
  B -- yes --> K["kill processes<br/>holding the card"] --> UB["unbind from xe/i915"]
  UB -- hangs --> RB
  UB --> PR["PCI reset"] --> BD["bind"] --> V{"listed by clinfo?"}
  V -- yes --> OK["ok"]
  V -- no --> FL["failed"]
```

- Sysfs writes run in the background with a timeout, so a driver stuck in the
  kernel cannot hang CI; that case is `reboot-required`.
- `reboot-required` and `no-privilege` are final: the card is excluded right away
  instead of waiting 10 minutes.
- Bare-metal jobs have the needed rights (passwordless sudo); the UT container
  runs `--privileged` for it.

Before the tests, `detect-devices.sh` compares the Intel GPUs on PCI with what
the compute runtime lists, recovers the missing ones, and leaves those it cannot
recover out of `ZE_AFFINITY_MASK`.

### GPU health in the UT report

`detect-devices.sh` and the plugin append events to `$XPU_HEALTH_LOG`;
`linux-uttest` copies it into the UT artifact (`ut_log/env/xpu_health.jsonl`) and
the UT report shows it:

| Machine | Card | Event | UT job | Detail |
|---|---|---|---|---|
| runner-01 | 0000:19:00.0 | unhealthy | upstream_distributed | on PCI but not enumerated by the compute runtime |
| runner-01 | 0000:19:00.0 | reboot-required | upstream_distributed | kernel oopsed (taint D): unbinding 0000:19:00.0 would hang, reboot the node |
| runner-01 | 0000:19:00.0 | excluded | upstream_distributed | not usable before the tests started |

Events: `unhealthy`, `recovered`, `excluded`, `reboot-required`, `no-privilege`,
`failed`.

### Options

| Option | Default | |
|---|---|---|
| `--xpu-crash-reruns N` | 1 | reruns of an unhealthy test, 0 disables |
| `--xpu-hang-timeout S` | `--timeout` + 300 | kill a worker stuck this long, 0 disables |
| `--xpu-offline-timeout S` | 600 | how long a broken card may take to recover before it is excluded |
| `--xpu-recover-cmd CMD` | `$XPU_RECOVER_CMD` | command run with a card's PCI BDF to recover it |

Every step is logged with an `[xpu-recovery]` prefix. GPU checks turn off by
themselves on a machine without a usable XPU, and only apply under xdist.
