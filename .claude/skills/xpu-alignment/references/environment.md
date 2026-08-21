# Environment Contract

The caller provisions the environment; this skill verifies and records it. Do
not create a virtual environment, install a nightly, or change drivers unless an
interactive user explicitly approves that separate action.

Before scanning or executing, establish:

- the exact Python executable and `torch.__version__`;
- `torch.xpu.is_available()`, device count, and device name;
- enough free disk space for artifacts;
- a read-only GitHub channel that can paginate the required endpoints;
- the GitHub API quota remaining at the start of collection/review.

Save `python -m torch.utils.collect_env` as `artifacts/collect_env.txt` and put a
concise environment summary in `scan_manifest.json` for automation. The report
must distinguish the build that ran the repro from any upstream build discussed
in source material.

An unavailable XPU, unusable interpreter, missing dependency, or inadequate API
quota is a blocker. Interactive runs may ask the user how to repair it.
Automation records the blocker and stops the affected phase without attempting
installation or credential repair.
