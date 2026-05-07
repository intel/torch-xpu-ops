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

5. **Labels created** on the issue repo:

```bash
REPO=ZhaoqiongZ/torch-xpu-ops-exp  # or your ISSUE_REPO
for label in agent:active agent:needs-human; do
  gh label create "$label" --repo $REPO --color ededed --force
done
```

---

## Environment Variables

All repo references are configured via environment. **Set these before running:**

| Variable | Default | Description |
|----------|---------|-------------|
| `ISSUE_REPO` | `ZhaoqiongZ/torch-xpu-ops-exp` | Source repo for issues to process |
| `UPSTREAM_ISSUE_REPO` | `intel/torch-xpu-ops` | Upstream torch-xpu-ops repo |
| `PRIVATE_REVIEW_REPO` | *(required)* | Private review fork, e.g. `yourorg/pytorch` |
| `PUBLIC_TARGET_REPO` | `pytorch/pytorch` | Upstream repo for public PRs |
| `PYTORCH_DIR` | `~/pytorch` | Local pytorch checkout path |
| `AGENT_BACKEND` | `opencode` | Agent backend (`opencode` or `copilot`) |
| `OPENCODE_CMD` | `opencode` | Path to opencode binary |
| `POLL_INTERVAL` | `60` | Seconds between polling cycles |
| `GH_TOKEN` | - | Token for upstream repos |
| `REVIEW_GH_TOKEN` | - | Token for private review, public target, and issue repos |

Example `.env` setup:
```bash
export ISSUE_REPO="ZhaoqiongZ/torch-xpu-ops-exp"
export PRIVATE_REVIEW_REPO="yourorg/pytorch"
export PYTORCH_DIR="$HOME/pytorch"
export GH_TOKEN="ghp_..."
export REVIEW_GH_TOKEN="ghp_..."
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
│   │   ├── git.py                    # Git + GitHub CLI wrappers
│   │   ├── issue_body.py             # Issue body read/write helpers
│   │   ├── json_utils.py             # JSON extraction from agent output
│   │   ├── agent_backend.py          # OpenCode/Copilot dispatch
│   │   ├── notify.py                 # Notification helpers
│   │   └── review_handler.py         # PR review parsing
│   ├── discovery_agent.py            # Format raw issues into template
│   ├── triage_agent.py               # Root cause analysis + verdict
│   ├── issue_fixing_agent.py         # Orchestrate all stages
│   └── fixing_steps/
│       ├── implement.py              # Branch, code, push, PR
│       ├── private_review.py         # Handle review on private fork
│       ├── public_submit.py          # Cross-fork PR to pytorch/pytorch
│       ├── ci_watch.py               # Monitor CI, fix failures
│       └── close_issue.py            # Close issue, cleanup branch
├── scripts/
│   ├── run_pipeline.py               # Polling loop
│   ├── cron.sh                       # Cron wrapper
│   └── status_report.py              # Print status table
└── logs/                             # Runtime logs
```

---

## How to Run

### Option A: Process a single issue end-to-end

```bash
cd ~/torch-xpu-ops/.github/pytorch-agent
source .env

# Run full pipeline for one issue (discovery → triage → fix → review → PR)
python scripts/run_pipeline.py --issue 123
```

Each stage reads the issue body, does its work, updates the issue body,
and advances to the next stage automatically.

You can also run individual agents:
```bash
# Just format a raw issue
python -m pytorch_agent.discovery_agent --issue 123

# Just triage a formatted issue
python -m pytorch_agent.triage_agent --issue 123

# Just advance one step
python -m pytorch_agent.issue_fixing_agent --issue 123
```

### Option B: Automated polling loop

```bash
cd ~/torch-xpu-ops/.github/pytorch-agent
source .env

# Single cycle — check all open issues and advance them
python scripts/run_pipeline.py --once

# Continuous loop (every 60s by default)
python scripts/run_pipeline.py

# Custom interval
python scripts/run_pipeline.py --interval 300
```

---

## Stage Flow

```
(no status) ──→ DISCOVERED ──→ TRIAGING ──→ IMPLEMENTING ──→ IN_REVIEW ──→ PUBLIC_PR ──→ CI_WATCH ──→ DONE
                                   │             │               │
                                   ▼             ▼               ▼
                               NEEDS_HUMAN   NEEDS_HUMAN     NEEDS_HUMAN
```

All state is tracked in the issue body via `<!-- agent:status:STAGE -->` markers.

| Stage | Agent | What happens | Needs human? |
|-------|-------|-------------|--------------|
| **(no status)** | discovery_agent | Formats raw issue into structured template | No |
| **TRIAGING** | triage_agent | Determines root cause, writes fix strategy, decides IMPLEMENTING vs NEEDS_HUMAN | No |
| **IMPLEMENTING** | implement.py | Branches `agent/issue-N`, calls LLM with fix skill, pushes, creates PR on review fork | No |
| **IN_REVIEW** | private_review.py | Waits for human review. Addresses feedback if changes requested (max 3 iterations) | **Yes — review the PR** |
| **PUBLIC_PR** | public_submit.py | Creates cross-fork PR from review fork to pytorch/pytorch | No |
| **CI_WATCH** | ci_watch.py | Polls CI checks. Fixes related failures. Advances to DONE on merge | No |
| **DONE** | close_issue.py | Comments on issue with merged PR link, closes issue | No |
| **NEEDS_HUMAN** | — | Agent exceeded attempts or can't fix | **Yes — take over** |

---

## Automated Monitoring (cron)

```bash
# 1. Create .env with your config
cp .env.example .env
# Edit .env — set ISSUE_REPO and PRIVATE_REVIEW_REPO at minimum

# 2. Test manually
./scripts/cron.sh

# 3. Install crontab (every 5 minutes)
(crontab -l 2>/dev/null; echo "*/5 * * * * $HOME/torch-xpu-ops/.github/pytorch-agent/scripts/cron.sh") | crontab -
```

**What it does each cycle:**
1. Lists all open issues in `ISSUE_REPO` with the tracking label
2. For each issue, reads its status from the body and advances one step
3. Discovery → Triage → Implement → Review → Public PR → CI Watch → Done

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
```

---

## Skills

The agent uses skills (in `.github/skills/`) to guide LLM behavior at each stage:

| Skill | Used by | Purpose |
|-------|---------|---------|
| `pytorch-issue-discovery` | discovery_agent | How to extract structured info from raw issues |
| `pytorch-triage-ut` | triage_agent | How to triage unit test failures |
| `pytorch-triage-e2e` | triage_agent | How to triage end-to-end test failures |
| `pytorch-fix` | implement.py | How to fix XPU CI failures |
| `pytorch-ci-triage` | ci_watch.py | How to triage CI failures on PRs |
| `pytorch-review-fix` | private_review.py | How to address code review feedback |
| `xpu-ops-pr-creation` | implement.py | PR format and conventions |

---

## Troubleshooting

**Issue not being picked up?**
- Check it exists in `ISSUE_REPO` (default: `ZhaoqiongZ/torch-xpu-ops-exp`)
- Check it has the tracking label (default: `ai_generated`)

**Stuck at a stage?**
- Read the issue body — action items show what's done and what's pending
- Check `<details>` logs for each agent's output
- Run `python -m pytorch_agent.issue_fixing_agent --issue N` to retry

**NEEDS_HUMAN?**
- Check the triage/fix logs in the issue body
- Fix manually, then edit the issue body to set `<!-- agent:status:DONE -->`

**Agent backend errors?**
- Verify `opencode` works: `opencode run --dir ~/pytorch "echo hello"`
- Check `OPENCODE_CMD` env var if opencode is not on PATH

**Token issues?**
- `REVIEW_GH_TOKEN` is used for `ISSUE_REPO`, `PRIVATE_REVIEW_REPO`, and `PUBLIC_TARGET_REPO`
- `GH_TOKEN` is used for `UPSTREAM_ISSUE_REPO`
