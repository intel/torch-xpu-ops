---
name: ci-automation
description: Use when the pytorch-test-refactoring orchestrator emits "schedule_cron" or "need_agent" with phase "ci_debug", when a CronCreate cron job fires and you need to run orchestrator.py --ci-check, or when the user says "look after the ci", "monitor ci", "check ci status", "watch the pr", or "debug ci failures".
---

# CI Automation

Monitors CI for an existing PR, classifies failures, and spawns a debugger agent to fix regressions. PR creation and pushing is handled by the user — this skill only looks after CI.

Reference for the CI loop. The root SKILL.md covers the status vocabulary — this file has the debugger format and allowlist.

## Actions by Status

### "done" + phase: "ci_done"

All green. Run `gh pr ready <N>` to mark the draft PR ready for review, then `CronDelete` all CI cron jobs. Truly finished.

### "schedule_cron" + phase: "ci_monitor"

CI still running. `CronCreate` with the `on_complete` fields (cron_interval, prompt, durable=true). Save the job ID, session exits.

### "need_agent" + phase: "ci_debug"

1. Spawn debugger via `Agent` tool: `run_in_background=true`, `mode=bypassPermissions`
2. Capture `agent_id` from result
3. Wait for completion, then pipe: `echo '{"agent_id":"...","agent_name":"debugger",...}' | python orchestrator.py <file> --ci-check --feed debugger`

## Debugger Result Format

```json
{
  "agent_id": "c3fa28...",
  "agent_name": "debugger",
  "fixes_applied": [
    {
      "check_name": "pull / linux-bionic-cuda12.1 / test_ops",
      "failure": "Description of the failure",
      "verdict": "caused_by_us",
      "rationale": "Why the refactoring caused this",
      "change": "What code change was made"
    }
  ],
  "unrelated": [
    {
      "check_name": "pull / win-vs2019 / test_ops",
      "failure": "Description",
      "verdict": "unrelated",
      "rationale": "Why this is not our fault"
    }
  ]
}
```

- `agent_id`: **Required** after spawn.
- `agent_name`: Must be `"debugger"`.

## Max Fix Rounds

5 rounds max. On limit: orchestrator leaves a PR comment and marks ready.

## Required Bash Allowlist

```json
{
  "permissions": {
    "allow": [
      "Bash(gh:*)",
      "Bash(git:*)",
      "Bash(lintrunner:*)",
      "Bash(python orchestrator.py:*)"
    ]
  }
}
```

## Related

- CI state machine: `ci_ops.py`
- CI operations: `scripts/ci.py`
- Debugger prompt: `agent/prompts/debugger.md`
