# pytorch-agent

Autonomous pipeline that watches intel/torch-xpu-ops for CI failure issues,
generates fixes, and submits them to pytorch/pytorch.

## Quick start

```bash
cp .env.example .env   # fill in tokens and repo names
python scripts/run_pipeline.py --issue 3509
```

## Slash commands (in issue / PR comments)

| Command        | Effect                          |
|----------------|---------------------------------|
| `/agent pause` | Stop processing this issue      |
| `/agent resume`| Resume processing               |

## Stage machine

```
DISCOVERED → IMPLEMENTING → IN_REVIEW → PUBLIC_PR → CI_WATCH → MERGED → DONE
                                                                       ↘ SKIPPED
                                                                       ↘ NEEDS_HUMAN
```

## Hard rules for the coding agent

- **Never** use `@skipIfXpu`, `@skip`, or any skip decorator — fix the test.
- **Never** commit `third_party/*` submodule pointer changes.
- **Never** force-push a branch that has been reviewed.
