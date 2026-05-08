# AGENTS.md — PyTorch Agent (pytorch-agent)

Autonomous agent for triaging and fixing `ai_generated` issues in `pytorch/pytorch`.
Issues sourced from `ISSUE_REPO` (default: `ZhaoqiongZ/torch-xpu-ops-exp`).

## Architecture

```
format_agent.py → triage_agent.py → orchestrator.py
                                              ↓
                                       fixing_steps/
                           code_fix → private_review → public_submit
                                                            ↓
                                                    ci_watch → close_issue
```

## Entry Points

| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/run_pipeline.py` | Polling loop (temporary) | `python scripts/run_pipeline.py --once` |
| `scripts/status_report.py` | Print tracked issue status | `python scripts/status_report.py` |
| `format_agent.py` | Format raw issue into structured template | `python -m issue_handler.format_agent --issue 123` |
| `triage_agent.py` | Analyze issue, determine root cause & verdict | `python -m issue_handler.triage_agent --issue 123` |
| `orchestrator.py` | Advance a tracked issue through stages | `python -m issue_handler.orchestrator --issue 123` |

## State Tracking

All state lives in **issue bodies** in `ISSUE_REPO`:
- **Status:** `<!-- agent:status:STAGE -->` HTML comment
- **Metadata:** `<!-- tracking_pr: #N -->`, `<!-- last_push_sha: abc -->`, etc.
- **Logs:** `<details>` blocks appended per stage
- **Labels:** `agent:active` (processing), `agent:done` (finished), `agent:needs-human` (escalation)

## Stage Flow

```
DISCOVERED → TRIAGING → IMPLEMENTING → IN_REVIEW → PUBLIC_PR → CI_WATCH → MERGED → DONE
                    ↓            ↓            ↓                       ↓
              NEEDS_HUMAN  NEEDS_HUMAN  NEEDS_HUMAN              NEEDS_HUMAN
```

Terminal stages (`DONE`, `SKIPPED`, `NEEDS_HUMAN`) are defined in `config/agent_config.yml`.

## Configuration

All constants live in `config/agent_config.yml`. Python `config.py` loads from it.
Environment variables override YAML defaults:

| Variable | Default | Description |
|----------|---------|-------------|
| `ISSUE_REPO` | `ZhaoqiongZ/torch-xpu-ops-exp` | Source repo for issues |
| `PRIVATE_REVIEW_REPO` | *(required)* | Private review fork |
| `PYTORCH_DIR` | `~/pytorch` | Local pytorch checkout |
| `AGENT_BACKEND` | `opencode` | Backend (`opencode` or `copilot`) |

## Skills

Agent prompts reference skills in `.github/skills/`:

| Skill | Used by |
|-------|---------|
| `pytorch-issue-discovery` | format_agent |
| `pytorch-triage-ut` / `pytorch-triage-e2e` | triage_agent |
| `pytorch-fix` | code_fix |
| `pytorch-review-fix` | private_review |
| `pytorch-review-task-extraction` | private_review |
| `pytorch-ci-triage` | ci_watch |
| `xpu-ops-pr-creation` | code_fix |

## Templates

Issue/PR body templates in `.github/ISSUE_TEMPLATE/`:
- `agent-issue.yml` — GitHub issue form
- `agent-issue-body.yml` — programmatic issue body template
- `agent-pr-body.yml` — PR description template

Loaded via `build_body()` in `issue_body.py`.

## Utilities

| Module | Purpose |
|--------|---------|
| `utils/config.py` | Loads `config/agent_config.yml` + env overrides |
| `utils/git.py` | Git CLI + GitHub API operations |
| `utils/issue_body.py` | Issue body parsing, template rendering, metadata |
| `utils/agent_backend.py` | LLM agent dispatch (OpenCode / Copilot) |
| `utils/json_utils.py` | JSON extraction from agent output |
| `utils/logger.py` | Structured logging |
| `utils/notify.py` | Notifications |
| `utils/review_handler.py` | PR review state parsing |
