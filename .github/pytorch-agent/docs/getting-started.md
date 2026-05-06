# Getting Started

## Prerequisites

- `gh` CLI authenticated
- `opencode` CLI installed
- Python 3.11+
- A fork of pytorch/pytorch as your private review repo

## Setup

1. Copy `.env.example` to `.env` and fill in values.
2. Add a cron entry:
   ```
   */15 * * * * /path/to/scripts/cron.sh >> /path/to/logs/cron.log 2>&1
   ```
3. Label an issue with `agent:new` to trigger the pipeline.

## Running manually

```bash
# Single issue
./scripts/run_oneshot.sh --issue 3509

# One cron cycle
python scripts/run_pipeline.py --once
```
