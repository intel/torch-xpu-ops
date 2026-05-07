# pytorch-agent — Getting Started

## Prerequisites

1. **Python 3.12+** with `pytest` (tests only)
2. **`gh` CLI** — authenticated with access to the repos below
3. **`opencode` CLI** — the agent backend that does the actual coding
4. **PyTorch checkout** at `~/pytorch` (or set `PYTORCH_DIR`) with remotes configured:

```bash
cd ~/pytorch
git remote -v
# Should show:
#   origin    <your-pytorch-fork>          (your fork)
#   upstream  https://github.com/pytorch/pytorch.git  (upstream)
#   review    <private-review-fork>        (private review)

# Add review remote if missing:
git remote add review <your-private-review-fork-url>
```

5. **Labels created** on `intel/torch-xpu-ops`:

```bash
REPO=intel/torch-xpu-ops
for label in agent:active agent:blocked agent:done agent:skipped agent:needs-human agent:paused; do
  gh label create "$label" --repo $REPO --color ededed --force
done
```

---

## Environment Variables

All repo references are configured via environment. **Set these before running:**

| Variable | Default | Description |
|----------|---------|-------------|
| `PRIVATE_REVIEW_REPO` | *(required)* | Private review fork, e.g. `yourorg/pytorch` |
| `UPSTREAM_ISSUE_REPO` | `intel/torch-xpu-ops` | Source repo for `ai_generated` issues |
| `PUBLIC_TARGET_REPO` | `pytorch/pytorch` | Upstream repo for public PRs |
| `PYTORCH_DIR` | `~/pytorch` | Local pytorch checkout path |
| `AGENT_BACKEND` | `opencode` | Agent backend (`opencode` or `copilot`) |
| `OPENCODE_CMD` | `opencode` | Path to opencode binary |
| `POLL_INTERVAL` | `60` | Seconds between polling cycles |

Example `.env` setup:
```bash
export PRIVATE_REVIEW_REPO="yourorg/pytorch"
export PYTORCH_DIR="$HOME/pytorch"
```

---

## Directory Layout

```
torch-xpu-ops/.github/pytorch-agent/
├── AGENTS.md                         # Architecture overview
├── pytorch_agent/
│   ├── utils/
│   │   ├── config.py                 # All constants & env vars
│   │   ├── logger.py                 # Simple file logger
│   │   ├── github_client.py          # gh CLI wrappers
│   │   ├── state.py                  # Issue-comment state tracking
│   │   ├── agent_backend.py          # OpenCode/Copilot dispatch
│   │   └── review_handler.py         # PR review parsing
│   ├── issue_discovery.py            # Find new ai_generated issues
│   ├── issue_triaging_agent.py       # Decide: pytorch or skip?
│   ├── issue_fixing_agent.py         # Orchestrate fixing steps
│   └── fixing_steps/
│       ├── implement.py              # Branch, code, push
│       ├── private_review.py         # Handle review on private fork
│       ├── public_submit.py          # Cross-fork PR to pytorch/pytorch
│       ├── ci_watch.py               # Monitor CI, fix failures
│       └── close_issue.py            # Close issue, cleanup branch
├── scripts/
│   ├── run_pipeline.py               # Polling loop (temporary)
│   └── status_report.py              # Print status table
├── tests/                            # 34 unit tests
└── logs/                             # Runtime logs
```

---

## How to Run

### Option A: Process a single issue end-to-end

```bash
cd ~/torch-xpu-ops/.github/pytorch-agent

# Step 1: Start tracking an issue
python -m pytorch_agent.issue_discovery --issue 123

# Step 2: Triage it (decides pytorch vs skip)
python -m pytorch_agent.issue_triaging_agent --issue 123

# Step 3: Advance through fixing stages (run repeatedly)
python -m pytorch_agent.issue_fixing_agent --issue 123
# Re-run after review / CI completes to advance to next stage
```

### Option B: Automated polling loop

```bash
cd ~/torch-xpu-ops/.github/pytorch-agent

# Single cycle — discover all new issues, advance all active ones
python scripts/run_pipeline.py --once

# Continuous loop (every 60s by default)
python scripts/run_pipeline.py

# Custom interval
python scripts/run_pipeline.py --interval 300
```

### Check status

```bash
python scripts/status_report.py
```

Output:
```
    #  Stage            Branch                       PR    Pub  Rev  Att  Title
----------------------------------------------------------------------------------------------------
  123  IN_REVIEW        agent/issue-123               5      -    1    1  Fix XPU conv dispatch
  456  SKIPPED          agent/issue-456               -      -    0    0  Missing XPU kernel
```

### Run tests

```bash
cd ~/torch-xpu-ops/.github/pytorch-agent
pytest tests/ -v
```

---

## Stage Flow & What Happens at Each Stage

```
DISCOVERED ──→ TRIAGING ──→ IMPLEMENTING ──→ IN_REVIEW ──→ PUBLIC_PR ──→ CI_WATCH ──→ DONE
                  │               │              │
                  ▼               ▼              ▼
               SKIPPED       NEEDS_HUMAN    NEEDS_HUMAN
```

| Stage | What happens | Needs human? |
|-------|-------------|--------------|
| **DISCOVERED** | Issue found, state comment created on issue | No |
| **TRIAGING** | Agent reads issue, decides pytorch vs skip | No |
| **IMPLEMENTING** | Agent branches `agent/issue-N`, codes fix, pushes to `review` remote, creates draft PR on private review fork | No |
| **IN_REVIEW** | Waits for human review on private PR. If `changes_requested`, agent addresses feedback and re-pushes. Max 3 iterations. | **Yes — review the PR** |
| **PUBLIC_PR** | Agent creates cross-fork PR from review fork to `pytorch/pytorch:main` | No |
| **CI_WATCH** | Polls CI checks. If failures related to the change, agent tries to fix. If merged, advances to DONE. | No |
| **DONE** | Comments on source issue with link to merged PR, closes issue, deletes agent branch | No |
| **SKIPPED** | Triage decided fix doesn't belong in pytorch | No |
| **NEEDS_HUMAN** | Agent exceeded 3 attempts or 3 review iterations | **Yes — take over** |

---

## Automated Monitoring (cron)

The agent ships with a cron wrapper that runs one cycle every N minutes:

```bash
# 1. Create .env with your config
cp .env.example .env
# Edit .env — set PRIVATE_REVIEW_REPO at minimum

# 2. Test manually
./scripts/cron.sh

# 3. Install crontab (every 5 minutes)
crontab -l 2>/dev/null; echo "*/5 * * * * $HOME/torch-xpu-ops/.github/pytorch-agent/scripts/cron.sh"
# Review, then:
(crontab -l 2>/dev/null; echo "*/5 * * * * $HOME/torch-xpu-ops/.github/pytorch-agent/scripts/cron.sh") | crontab -
```

**What it does each cycle:**
1. Discovers new `ai_generated` issues
2. Triages any DISCOVERED issues
3. Advances all active issues (IMPLEMENTING → IN_REVIEW → PUBLIC_PR → CI_WATCH → DONE)

**Safety:**
- File lock prevents overlapping runs
- Cycle logs in `logs/cycle-*.log` (last 100 kept)
- Summary in `logs/cron.log`

**Check it's working:**
```bash
# Recent cron runs
tail -20 ~/torch-xpu-ops/.github/pytorch-agent/logs/cron.log

# Latest cycle detail
ls -1t ~/torch-xpu-ops/.github/pytorch-agent/logs/cycle-*.log | head -1 | xargs cat

# All tracked issues
python scripts/status_report.py
```

---

## Troubleshooting

**Issue not being picked up?**
- Check it has the `ai_generated` label on the upstream issue repo
- Run `python -m pytorch_agent.issue_discovery --poll` manually

**Stuck at IN_REVIEW?**
- Review the PR on the private review fork — approve or request changes
- Then run `python -m pytorch_agent.issue_fixing_agent --issue N`

**NEEDS_HUMAN?**
- Check logs in `logs/` for what went wrong
- Fix manually, then update the state comment on the issue to change stage

**Agent backend errors?**
- Verify `opencode` works: `opencode run --dir ~/pytorch "echo hello"`
- Check `OPENCODE_CMD` env var if opencode is not on PATH

**PRIVATE_REVIEW_REPO not set?**
- The agent will fail at any step that touches the review fork
- Set it: `export PRIVATE_REVIEW_REPO="yourorg/pytorch"`
