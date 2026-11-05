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
