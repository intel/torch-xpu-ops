# AGENTS.md — PyTorch Agent (pytorch-agent)

Autonomous agent system for triaging and fixing `ai_generated` issues from
`ZhaoqiongZ/torch-xpu-ops-exp` (configurable via `ISSUE_REPO`) in `pytorch/pytorch`.

## Architecture

```
discovery_agent.py → triage_agent.py → issue_fixing_agent.py
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
| `discovery_agent.py` | Format raw issue into structured template | `python -m pytorch_agent.discovery_agent --issue 123` |
| `triage_agent.py` | Analyze issue, determine root cause & verdict | `python -m pytorch_agent.triage_agent --issue 123` |
| `issue_fixing_agent.py` | Advance a tracked issue through stages | `python -m pytorch_agent.issue_fixing_agent --issue 123` |

## State Tracking

State is stored on **issue bodies** in the issue repo (`ISSUE_REPO`):
- **Status:** HTML comment `<!-- agent:status:STAGE -->` in issue body
- **Metadata:** HTML comments `<!-- tracking_pr: #N -->`, `<!-- last_push_sha: abc -->`, etc.
- **Action log:** `<details>` blocks appended to issue body
- **Labels:** `agent:active` (during processing), `agent:needs-human` (escalation)

## Stage Flow

```
DISCOVERED → TRIAGING → IMPLEMENTING → IN_REVIEW → PUBLIC_PR → CI_WATCH → MERGED → DONE
                    ↓            ↓            ↓                       ↓
              NEEDS_HUMAN  NEEDS_HUMAN  NEEDS_HUMAN              NEEDS_HUMAN
```

## Repos & Remotes

| Repo | Role |
|------|------|
| `$ISSUE_REPO` | Source issues (default: `ZhaoqiongZ/torch-xpu-ops-exp`) |
| `$PRIVATE_REVIEW_REPO` | Private review PRs (remote: `review`) |
| `pytorch/pytorch` | Public PRs (remote: `upstream`) |

## Escalation

- Implementation: max 3 attempts → `agent:needs-human` label + NEEDS_HUMAN stage
- Review: max 3 iterations → `agent:needs-human` label + NEEDS_HUMAN stage
- CI watch: max 3 iterations → `agent:needs-human` label + NEEDS_HUMAN stage

## Configuration

Environment variables override defaults in `pytorch_agent/utils/config.py`:
- `PYTORCH_DIR` — local pytorch checkout (default: `~/pytorch`)
- `ISSUE_REPO` — issue tracking repo (default: `ZhaoqiongZ/torch-xpu-ops-exp`)
- `AGENT_BACKEND` — `opencode` (default) or `copilot`
- `POLL_INTERVAL` — seconds between polling cycles (default: 60)

## Skills

Agent prompts reference skills in `.github/skills/`:
- `pytorch-issue-discovery` — format raw issues into structured template
- `pytorch-triage-ut` — triage unit test failures
- `pytorch-triage-e2e` — triage end-to-end test failures
- `pytorch-fix` — implement fixes for triaged issues
- `pytorch-review-fix` — address code review feedback
- `pytorch-ci-triage` — CI failure analysis
- `xpu-ops-pr-creation` — implementation & PR conventions
- `xpu-ops-pr-review` — review feedback handling

## Utilities

| Module | Purpose |
|--------|---------|
| `utils/git.py` | All git CLI + GitHub API operations |
| `utils/issue_body.py` | Parse/update issue body (status, sections, metadata, logs) |
| `utils/config.py` | Configuration constants |
| `utils/agent_backend.py` | LLM agent dispatch (OpenCode / Copilot) |
| `utils/json_utils.py` | JSON extraction from agent output |
| `utils/logger.py` | Structured logging |
| `utils/notify.py` | Notifications |
| `utils/review_handler.py` | PR review state parsing |
| `config/label_mapping.yml` | Label prefix → field name mapping |
