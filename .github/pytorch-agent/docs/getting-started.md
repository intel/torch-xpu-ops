# Issue Handler Agent — Getting Started

## Prerequisites

1. **Python 3.12+**
2. **`gh` CLI** — authenticated with repo access
3. **`opencode` CLI** — agent backend
4. **PyTorch checkout** at `~/pytorch` with remotes:

```bash
cd ~/pytorch
git remote add review <your-private-review-fork-url>
# Remotes: origin (your fork), upstream (pytorch/pytorch), review (private review)
```

5. **Labels** on the issue repo:

```bash
REPO=ZhaoqiongZ/torch-xpu-ops-exp
for label in agent:active agent:done agent:needs-human; do
  gh label create "$label" --repo $REPO --color ededed --force
done
```

---

## Configuration

All constants in `config/agent_config.yml`. Env vars override YAML defaults.

| Variable | Default | Description |
|----------|---------|-------------|
| `ISSUE_REPO` | `ZhaoqiongZ/torch-xpu-ops-exp` | Source repo for issues |
| `UPSTREAM_ISSUE_REPO` | `intel/torch-xpu-ops` | Upstream torch-xpu-ops repo |
| `PRIVATE_REVIEW_REPO` | *(required)* | Private review fork, e.g. `yourorg/pytorch` |
| `PUBLIC_TARGET_REPO` | `pytorch/pytorch` | Upstream repo for public PRs |
| `PYTORCH_DIR` | `~/pytorch` | Local pytorch checkout path |
| `AGENT_BACKEND` | `opencode` | Agent backend (`opencode` or `copilot`) |
| `GH_TOKEN` | - | Token for upstream repos |
| `REVIEW_GH_TOKEN` | - | Token for private review, public target, and issue repos |

Example `.env`:
```bash
export ISSUE_REPO="ZhaoqiongZ/torch-xpu-ops-exp"
export PRIVATE_REVIEW_REPO="yourorg/pytorch"
export PYTORCH_DIR="$HOME/pytorch"
export GH_TOKEN="***"
export REVIEW_GH_TOKEN="***"
```

---

## Directory Layout

```
torch-xpu-ops/.github/
├── ISSUE_TEMPLATE/
│   ├── agent-issue.yml              # GitHub issue form
│   ├── agent-issue-body.yml         # Programmatic issue body template
│   └── agent-pr-body.yml           # PR description template
├── skills/                          # LLM skill prompts
└── pytorch-agent/
    ├── AGENTS.md
    ├── config/
    │   └── agent_config.yml         # All tunable constants
    ├── issue_handler/
    │   ├── format_agent.py       # Format raw issues
    │   ├── triage_agent.py          # Root cause analysis
    │   ├── orchestrator.py    # Stage orchestrator
    │   ├── fixing_steps/
    │   │   ├── code_fix.py          # Branch, code, push, PR
    │   │   ├── private_review.py    # Handle review feedback
    │   │   ├── public_submit.py     # Cross-fork PR
    │   │   ├── ci_watch.py          # Monitor CI
    │   │   └── close_issue.py       # Close issue, cleanup
    │   └── utils/
    │       ├── config.py            # Loads agent_config.yml + env
    │       ├── git.py               # Git + GitHub API
    │       ├── body_templates.py    # Issue body parsing & templates
    │       ├── agent_backend.py     # OpenCode/Copilot dispatch
    │       ├── json_utils.py        # JSON extraction
    │       ├── logger.py            # Structured logging
    │       ├── notify.py            # Notifications
    │       └── review_handler.py    # PR review parsing
    ├── scripts/
    │   ├── run_pipeline.py          # Polling loop
    │   ├── cron.sh                  # Cron wrapper
    │   └── status_report.py         # Status table
    └── logs/
```

---

## How to Run

### Single issue end-to-end

```bash
cd ~/torch-xpu-ops/.github/pytorch-agent
source .env
python scripts/run_pipeline.py --issue 123
```

### Individual agents

```bash
# Format raw issue → structured template (input: raw issue body, output: formatted issue body)
python -m issue_handler.format_agent --issue 123

# Analyze root cause → verdict (input: formatted issue body, output: root cause + fix strategy + IMPLEMENTING/NEEDS_HUMAN)
python -m issue_handler.triage_agent --issue 123

# Advance one stage (input: issue at any stage, output: issue advanced to next stage)
python -m issue_handler.orchestrator --issue 123
```

### Automated polling

```bash
python scripts/run_pipeline.py --once          # Single cycle
python scripts/run_pipeline.py                 # Continuous (60s default)
python scripts/run_pipeline.py --interval 300  # Custom interval
```

---

## Stage Flow

```
(raw) → DISCOVERED → TRIAGING → IMPLEMENTING → IN_REVIEW → PUBLIC_PR → CI_WATCH → DONE
                          ↓            ↓            ↓                       ↓
                      NEEDS_HUMAN  NEEDS_HUMAN  NEEDS_HUMAN            NEEDS_HUMAN
```

| Stage | Agent | What happens |
|-------|-------|-------------|
| *(raw)* | format_agent | Formats issue into structured template |
| TRIAGING | triage_agent | Root cause + fix strategy → IMPLEMENTING or NEEDS_HUMAN |
| IMPLEMENTING | code_fix.py | Branch, LLM fix, push, create PR on review fork |
| IN_REVIEW | private_review.py | Wait for review, address feedback (max 3 iterations) |
| PUBLIC_PR | public_submit.py | Cross-fork PR to pytorch/pytorch |
| CI_WATCH | ci_watch.py | Monitor CI, fix failures |
| DONE | close_issue.py | Close issue with merged PR link |
| NEEDS_HUMAN | — | Agent exceeded attempts |

---

## Cron Setup

```bash
cp .env.example .env  # Edit with your config
./scripts/cron.sh     # Test manually

# Install (every 5 min)
(crontab -l 2>/dev/null; echo "*/5 * * * * $HOME/torch-xpu-ops/.github/pytorch-agent/scripts/cron.sh") | crontab -
```

Check logs:
```bash
tail -20 logs/cron.log
ls -1t logs/cycle-*.log | head -1 | xargs cat
```

---

## Troubleshooting

- **Not picked up?** Check issue exists in `ISSUE_REPO` with `ai_generated` label.
- **Stuck?** Read issue body action items and `<details>` logs. Retry with `python -m issue_handler.orchestrator --issue N`.
- **NEEDS_HUMAN?** Check logs, fix manually, set `<!-- agent:status:DONE -->`.
- **Backend errors?** Verify `opencode run --dir ~/pytorch "echo hello"` works.
- **Token issues?** `REVIEW_GH_TOKEN` → issue/review/public repos. `GH_TOKEN` → upstream repo.
