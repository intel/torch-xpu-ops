# pytorch-agent: autonomous CI issue fixing pipeline

An agent pipeline that monitors `agent:new` labeled issues, triages them, generates fixes via an AI coding backend, submits PRs through a private review fork, and watches CI until merge.

## File Structure

```
.github/pytorch-agent/
├── scripts/
│   ├── cron.sh                  # Cron entry point (periodic polling)
│   ├── run_oneshot.sh           # Single-issue ad-hoc run
│   ├── run_pipeline.py          # CLI: --once | --issue N | continuous loop
│   └── status_report.py         # Print state of all tracked issues
├── pytorch_agent/
│   ├── issue_discovery.py       # Scan for `agent:new` labeled issues
│   ├── issue_triaging_agent.py  # AI triage: accept/skip + extract metadata
│   ├── issue_fixing_agent.py    # Stage router: advance(issue) → next step
│   ├── fixing_steps/
│   │   ├── _issue_format.py     # Shared: parse_issue_sections, build_pr_body
│   │   ├── implement.py         # Generate fix on private fork branch
│   │   ├── private_review.py    # AI self-review on chuanqi129/pytorch
│   │   ├── public_submit.py     # Open cross-fork PR to pytorch/pytorch
│   │   ├── ci_watch.py          # Monitor CI, auto-fix failures (max 3 iters)
│   │   └── close_issue.py       # Close source issue on merge
│   └── utils/
│       ├── config.py            # Env vars, repos, stage labels, timeouts
│       ├── state.py             # TrackedIssue dataclass + JSON persistence
│       ├── github_client.py     # gh CLI wrapper: issues, PRs, labels, checks
│       ├── git.py               # git(), git_out(), add_and_commit()
│       ├── agent_backend.py     # OpenCode backend: run prompts, parse events
│       ├── notify.py            # Post session/completion comments to issues
│       ├── review_handler.py    # Parse /agent commands from issue comments
│       └── logger.py            # Structured logging
├── docs/getting-started.md
├── AGENTS.md                    # Agent instructions (skills, conventions)
├── .env.example
└── .gitignore
```

## Workflow

```
┌─────────────┐     ┌──────────┐     ┌───────────┐     ┌────────────┐
│  DISCOVERED │────▶│  TRIAGE  │────▶│ IMPLEMENT │────▶│ IN_REVIEW  │
│ (new label) │     │ (accept?)│     │ (gen fix) │     │ (self-rev) │
└─────────────┘     └──────────┘     └───────────┘     └────────────┘
                         │                                    │
                     SKIPPED                                  ▼
                                                    ┌────────────────┐
                    ┌──────┐     ┌──────────┐       │ PUBLIC_SUBMIT  │
                    │ DONE │◀───│  MERGED   │◀──────│ (cross-fork PR)│
                    └──────┘    └──────────┘       └────────────────┘
                                      ▲                     │
                                      │              ┌──────▼──────┐
                                      └──────────────│  CI_WATCH   │
                                                     │ (≤3 rounds) │
                                                     └─────────────┘
```

1. **Cron** runs `scripts/cron.sh` → discovers issues labeled `agent:new`
2. **Triage** — AI decides accept/skip, extracts error logs & metadata
3. **Implement** — AI coding agent generates a fix on `chuanqi129/pytorch` branch
4. **Private review** — AI self-reviews the diff, iterates if needed
5. **Public submit** — opens a cross-fork PR to `pytorch/pytorch`
6. **CI watch** — monitors checks; on failure, AI fixes and re-pushes (max 3 iterations)
7. **Merge → Done** — closes source issue with summary

Each stage is idempotent — cron re-enters `advance()` safely. Issues can be paused via `/agent pause` comment.

## Example

Cron fires every 15 minutes, calling `run_pipeline.py --once`. Each cycle runs a loop: `advance(issue)` until the stage stops changing or hits a terminal state.

```
cron fires → discover #3509 (agent:new)
           → triage: ACCEPT
           → implement: AI writes fix, pushes to chuanqi129/pytorch
           → private_review: AI self-reviews diff, approves
           → public_submit: opens PR #181987 to pytorch/pytorch
           → ci_watch: CI still running → stop, wait for next cron
         ┌─── next cron ───┐
         │ ci_watch:        │
         │   CI passed?  ───┼──▶ MERGED → close issue → DONE ✅
         │   CI failed?  ───┼──▶ AI fixes code, pushes, stays in CI_WATCH
         │                  │    (repeat up to 3 times)
         │   3 failures? ───┼──▶ NEEDS_HUMAN — agent gives up 🛑
         └──────────────────┘
```

Pause: human comments `/agent pause` → cron skips the issue until `/agent resume`.
