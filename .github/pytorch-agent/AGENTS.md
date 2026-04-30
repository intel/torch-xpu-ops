# AGENTS.md — PyTorch Agent (pytorch-agent)

Autonomous agent system for triaging and fixing `ai_generated` issues from
`intel/torch-xpu-ops` in `pytorch/pytorch`.

## Architecture

```
issue_discovery.py → issue_triaging_agent.py → issue_fixing_agent.py
                                                     ↓
                                              fixing_steps/
                                  implement → private_review → public_submit
                                                                    ↓
                                                            ci_watch → close_issue
```

## Entry Points

| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/run_pipeline.py` | Polling loop (temporary) | `python scripts/run_pipeline.py --once` |
| `scripts/status_report.py` | Print tracked issue status | `python scripts/status_report.py` |
| `issue_discovery.py` | Discover new issues | `python -m pytorch_agent.issue_discovery --poll` |
| `issue_triaging_agent.py` | Triage a specific issue | `python -m pytorch_agent.issue_triaging_agent --issue 123` |
| `issue_fixing_agent.py` | Advance a tracked issue | `python -m pytorch_agent.issue_fixing_agent --issue 123` |

## State Tracking

State is stored on **source issues** in `intel/torch-xpu-ops`:
- **Labels:** `agent:tracking`, `agent:implementing`, `agent:in-review`, etc.
- **JSON state:** Hidden HTML comment in a dedicated issue comment
- **Stage transitions:** Human-readable comments posted on each change

## Stage Flow

```
DISCOVERED → TRIAGING → IMPLEMENTING → IN_REVIEW → PUBLIC_PR → CI_WATCH → DONE
                ↓              ↓            ↓
              SKIPPED    NEEDS_HUMAN   NEEDS_HUMAN
```

## Repos & Remotes

| Repo | Role |
|------|------|
| `intel/torch-xpu-ops` | Source issues (`ai_generated` label) |
| `$PRIVATE_REVIEW_REPO` | Private review PRs (remote: `review`) |
| `pytorch/pytorch` | Public PRs (remote: `upstream`) |

## Escalation

- Implementation: max 3 attempts → `needs:human` label + NEEDS_HUMAN stage
- Review: max 3 iterations → `needs:human` label + NEEDS_HUMAN stage

## Configuration

Environment variables override defaults in `pytorch_agent/utils/config.py`:
- `PYTORCH_DIR` — local pytorch checkout (default: `~/pytorch`)
- `AGENT_BACKEND` — `opencode` (default) or `copilot`
- `POLL_INTERVAL` — seconds between polling cycles (default: 60)

## Skills

Agent prompts reference skills in `.github/skills/`:
- `pytorch-triage` — issue triage criteria
- `xpu-ops-pr-creation` — implementation & PR conventions
- `xpu-ops-pr-review` — review feedback handling
- `pytorch-ci-triage` — CI failure analysis
