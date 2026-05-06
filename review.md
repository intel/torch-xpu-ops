# Review #5 — post round-4 cleanup

Round 4 (commit `aa468f4f`) did most of what was suggested:

| Issue | Status |
|---|---|
| `_save_agent_log` / `_get_latest_session_id` dead | ✅ deleted |
| `import time`, `STAGE_TIMEOUTS`, `Path` unused imports | ✅ removed |
| Duplicate `_parse_issue_sections` in `implement.py` | ✅ removed (re-imported from `_issue_format`) |
| `build_pr_body` not wired up | ✅ both `implement.py` and `public_submit.py` now call it |
| `notify.post_session_started` not used in triage | ✅ migrated |
| `notify.post_agent_completed` helper | ✅ added, 3 call sites use it |
| `implement._git` shim | ✅ deleted; `git_out()` added to utils |
| `_STAGE_CHECKLIST` dup literals | ✅ DRYed via `_ALL_ITEMS` slicing |
| `cron_run.sh` confusing name | ✅ renamed to `run_oneshot.sh` |
| Empty `requirements.txt` | ✅ deleted |
| `subprocess` import noise | ✅ narrowed to `from subprocess import CalledProcessError` |
| `parse_opencode_events` extracted | ✅ — but only the offline copy in `private_review`. `OpenCodeBackend.run` still inlines its own streaming JSON parser (2g below) |

Below: bugs introduced or untouched by round 4.

---

## 🔴 Carry-overs round 4 didn't address

### R5-1. `public_submit.py:66` still has the broken `except (CalledProcessError, Exception)`

This was flagged in **rounds 3 and 4**. Untouched.

```python
import subprocess
...
except (subprocess.CalledProcessError, Exception) as exc:
    # 422 "PR already exists" — find it
    existing = gh._gh_api(...)
    if existing:
        pr = existing[0]
    else:
        raise
```

Three problems still present:
1. `Exception` swallows `CalledProcessError`; the tuple's first element is
   dead code.
2. `exc` is bound but never read — no `log("ERROR", …, exc=exc)`. If the
   "already exists" path is taken with a *different* exception (network,
   auth, etc.), it silently retries the same API call and either returns
   a stale PR or re-raises with no context.
3. `gh._gh_api` doesn't raise `CalledProcessError` at all — it raises
   `urllib.error.HTTPError` / `RuntimeError`. So the `subprocess` import
   on line 10 exists *purely* to spell the wrong exception type.

Fix:
```python
except gh.GHAPIError as exc:        # or whatever gh raises for 422
    if "already exists" not in str(exc):
        raise
    existing = gh._gh_api(...)
    if not existing:
        raise
    pr = existing[0]
```
…and drop `import subprocess`.

### R5-2. `PAUSED` is still half-wired

State machine still inconsistent:

| Place | Setting |
|---|---|
| `config.py:19` | `STAGE_TO_LABEL["PAUSED"] = "agent:paused"` |
| `run_pipeline.py:23` | `TERMINAL_STAGES = {…, "PAUSED"}` |
| `state.py:45` | `paused: bool = False` (dataclass field) |
| `private_review.py:155-156` | `tracked.paused = True` (sets bool, **not** stage) |

`AGENTS.md` documents:
> `/agent pause` command → PAUSED (stays in IN_REVIEW, cron skips)

Code does *neither*:
- Stage is never set to `"PAUSED"` anywhere → `STAGE_TO_LABEL["PAUSED"]`
  and the `"PAUSED"` entry in `TERMINAL_STAGES` are dead.
- `tracked.paused = True` is set, but `run_pipeline.run_cycle()` only
  filters on `stage in TERMINAL_STAGES`. **Paused issues are still
  picked up** and fed back into `advance()` on every cycle.

Pick one:
- A) Honour the doc: `if tracked.paused: continue` in
  `run_pipeline.run_cycle` and `run_single`.
- B) Promote pause to a real stage: when `/agent pause` arrives, call
  `update_stage(tracked, "PAUSED", …)`. Then drop the `paused` boolean.

Either way, the current state has dead config **and** a runtime bug.

---

## 🟠 Issues introduced by round 4

### R5-3. `notify.py:21` annotates `log_path: "Path"` without importing `Path`

```python
def post_agent_completed(
    repo: str, issue: int, header: str, log_path: "Path",
    output: str, *, tail: int = 50,
) -> None:
```

Works at runtime because string annotations aren't evaluated, but:
- type-checkers (`mypy`, `pyright`) flag it as undefined name
- the file *has* `from __future__ import annotations`, so the quotes are
  redundant — and would fail if removed because `Path` truly isn't
  imported

Fix:
```python
from pathlib import Path
...
def post_agent_completed(
    repo: str, issue: int, header: str, log_path: Path,
    ...
```

### R5-4. `run_oneshot.sh` misses `set -a` so `.env` vars don't propagate

```bash
# scripts/run_oneshot.sh
cd "$AGENT_DIR"
source .env
python3 scripts/run_pipeline.py "$@" 2>&1
```

vs. `cron.sh:13`:
```bash
[[ -f "$AGENT_DIR/.env" ]] && set -a && source "$AGENT_DIR/.env" && set +a
```

If `.env` uses bare `KEY=value` (the convention used by python-dotenv,
docker-compose, the agent's own README), those vars are **not**
exported into Python's `os.environ`. `cron.sh` works because of `set -a`;
`run_oneshot.sh` silently runs with an empty environment for tokens.

Symptom: ad-hoc `./scripts/run_oneshot.sh --issue 3509` will hit
`gh: GH_TOKEN missing` while cron runs cleanly.

Fix: copy the `set -a / source / set +a` block from `cron.sh`, or
require `.env` to use `export` (and document it).

### R5-5. `implement.py:45` keeps a misleading `as _parse_issue_sections` alias

```python
# Issue section parsing — use the shared helper from _issue_format
from ._issue_format import parse_issue_sections as _parse_issue_sections
```

The original purpose of the `_` prefix was "private to this module". Now
that the implementation lives elsewhere, the alias just hides the import
source from `git grep`. Drop the `as` alias and call
`parse_issue_sections(...)` directly. (Same alias trick is *not* used in
`public_submit.py` after the round-4 rewrite — only `implement.py` is
inconsistent.)

### R5-6. Three function-local imports that should be top-level

After round 4, three modules import notify/format helpers inside the
function body:

| File | Line | Import |
|---|---|---|
| `implement.py` | 175 | `from ..utils.notify import post_agent_completed` |
| `implement.py` | 243 | `from ._issue_format import build_pr_body` |
| `ci_watch.py` | 152 | `from ..utils.notify import post_agent_completed` |
| `private_review.py` | 296 | `from ..utils.notify import post_agent_completed` |
| `public_submit.py` | 36 | `from ._issue_format import build_pr_body` |
| `issue_triaging_agent.py` | 65 | `from .utils.notify import post_session_started` |

There's no circular-import reason for any of these — `notify.py` only
imports from `github_client`, and `_issue_format.py` has no internal
imports. Move them to the top-of-file import block; right now each
function call does a fresh `importlib` lookup and the imports are
invisible to static analysis.

### R5-7. `build_pr_body` produces orphan `---` separator when no reviewer

`_issue_format.py:63-65`:
```python
body += "---\n\n"
if reviewer:
    body += f"cc @{reviewer}\n"
```

When called from `implement.py` (private review) `reviewer=""`, so the
PR body ends with a literal `---\n\n` and nothing after it. Renders as
a horizontal rule with no content beneath. Move the `---` inside the
`if reviewer:` block:

```python
if reviewer:
    body += f"---\n\ncc @{reviewer}\n"
```

### R5-8. `parse_opencode_events` and the streaming parser duplicate logic

Round 4 extracted `parse_opencode_events()` into `agent_backend.py` and
called it from `private_review._build_task_list`. ✅

But the **streaming** version inside `OpenCodeBackend.run`
(`agent_backend.py:108-135`) still inlines:

```python
try:
    event = json.loads(line)
except json.JSONDecodeError:
    continue
...
if event.get("type") == "text":
    part = event.get("part", {})
    text_parts.append(part.get("text", ""))
```

It can't trivially call `parse_opencode_events()` because the streaming
loop also extracts session-id and writes to a log file. But the two
parsers can drift — note the offline one falls back to
`event.get("content", "")` when `part.text` is empty, while the
streaming one does not. Either:
- give the streaming loop a `parse_event(line) -> dict | None` helper
  shared with the offline parser, or
- accept the dup and add a comment "intentional — streaming side
  effects" in both places.

---

## 🟡 Smaller cleanup carried over

### R5-9. `add_and_commit` mishandles `R old -> new` porcelain (still present)

`utils/git.py:43`:
```python
parts = line.split(maxsplit=1)
...
fname = parts[1].strip()
```

For a renamed file, porcelain prints `R  old/path -> new/path`. After
`split(maxsplit=1)` you get `parts[1] = "old/path -> new/path"`, and the
subsequent `git add -- "old/path -> new/path"` will fail with "pathspec
did not match any files". `git add -A` would handle this; or skip the
`-- files` list and `git add -u` for tracked-only.

### R5-10. `ci_watch.ci_iteration` only increments on successful push

`fixing_steps/ci_watch.py:140-160` (post round-4 indices):
```python
try:
    git("push", REVIEW_REMOTE, branch, ...)
    tracked.ci_iteration += 1
    save_state(tracked)
    ...
except CalledProcessError as e:
    log("ERROR", ...)
```

Push is the *last* thing that can fail. If it fails transiently, the
counter stays at N forever; next cron cycle increments to N again
(because the agent ran again), and the `>= MAX_AGENT_ATTEMPTS` cap
never kicks in. Move `tracked.ci_iteration += 1` to the *top* of the
function (right after `if tracked.stage != "CI_WATCH": return`).

### R5-11. Function-local `from ..utils.git import ... ` still exists

`add_and_commit` in `git.py:55-56` calls `git("add", "--", *files)`
with a `*files` splat. If `files` is empty after filtering, it becomes
`git add --` which **errors** with "Nothing specified". Round 4
guarded with `if not files: return False` — ✅ that's actually fine.

(Removing this entry — verified safe.)

---

## Summary table

| Severity | Item | File | Effort |
|---|---|---|---|
| 🔴 carry | R5-1 broken `except (CalledProcessError, Exception)` | `public_submit.py:66` | trivial |
| 🔴 carry | R5-2 `PAUSED` half-wired (cron doesn't skip) | `run_pipeline.py`, `private_review.py` | small |
| 🟠 new   | R5-3 missing `Path` import in `notify.py` | `notify.py:21` | trivial |
| 🟠 new   | R5-4 `run_oneshot.sh` lacks `set -a` | `scripts/run_oneshot.sh` | trivial |
| 🟠 new   | R5-5 misleading `as _parse_issue_sections` alias | `implement.py:45` | trivial |
| 🟠 new   | R5-6 6 function-local imports → move to top | 5 files | small |
| 🟠 new   | R5-7 orphan `---` when no reviewer | `_issue_format.py:63` | trivial |
| 🟡 carry | R5-8 streaming JSON parser still inlined | `agent_backend.py` | small |
| 🟡 carry | R5-9 `add_and_commit` rename porcelain | `git.py:43` | small |
| 🟡 carry | R5-10 `ci_iteration` increment guard | `ci_watch.py` | trivial |

The two real bugs are **R5-1** (broken except still posts no log on
real PR-create failure) and **R5-2** (paused issues continue to be
processed). The rest is polish.
