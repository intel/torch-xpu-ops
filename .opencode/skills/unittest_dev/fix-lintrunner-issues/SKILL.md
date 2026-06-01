---
name: fix-lintrunner-issues
description: Diagnose and fix lintrunner / preci-lint-check failures on PyTorch and torch-xpu-ops code. Accepts either a PR link (fetches the failing CI log) or a local folder / file path (runs the relevant adapters locally). Use when a PR's preci-lint-check job is failing, when asked to "fix lint issues" on a PR or a local path, or when porting tests triggers linter violations. Covers fetching CI logs via gh, reproducing failures locally even without lintrunner installed, and selecting the right fix strategy (code edit vs. noqa vs. `.lintrunner.toml` exclude) based on how the underlying adapter actually works.
---

# Fix Lintrunner Issues

## Skill Integration

Follow `agent-guidelines` on top of this skill: deep semantic analysis over pattern matching, atomic commits, never commit without explicit user permission.

Per `AGENTS.md`: do not auto-install tooling. If `uv`, `lintrunner`, or the env is missing, **stop and ask the user** before running build/install commands. You can still do most of the diagnostic work without lintrunner — see Step 3B (adapter-only invocation) and Step 6 (verification).

## Input Modes

This skill supports two input modes. Determine which before starting.

### Mode A: PR link or PR reference

Examples: `https://github.com/intel/torch-xpu-ops/pull/3383`, `intel/torch-xpu-ops#3383`, or "PR 3383 on torch-xpu-ops".

Parse into `(owner, repo, pr_number)`. Proceed to **Step 1A**.

### Mode B: Local folder or file path

Examples: `third_party/torch-xpu-ops`, `/home/.../my_clone`, `test/xpu/dynamo/test_misc_xpu.py`, or "fix lint in the current directory".

The user wants to lint a working copy directly — no CI log exists. Proceed to **Step 1B**.

If the input is ambiguous (e.g. just a repo name), ask the user which mode they mean.

## High-Level Flow

```
Mode A (PR):                                  Mode B (local path):
  1A. Fetch PR + failing job log via gh         1B. Identify the lintrunner-enabled repo root
  2A. Parse lint report from log                    (ascend to the nearest .lintrunner.toml)
  3A. Locate/clone the authoritative tree       2B. Select target files (explicit path, or
                                                    changed files vs. base branch)
                                                3B. Run lintrunner (or adapters directly)
                                                    and parse the report

Shared from here on:
  4.  Look up each failing rule in .lintrunner.toml: include/exclude scope + adapter
  4.5 Deep-analyze the failing code: intent, real vs. incidental vs. scope
      mismatch; launch `explore` subagent for cross-file / cross-repo checks
  5.  Decide fix strategy per rule (Fix Strategy Matrix)
  6.  Ask the user to choose between viable strategies when there is a meaningful trade-off
  7.  Apply fix; verify by re-running the adapter or lintrunner against the same file
  8.  Ask the user before committing; for Mode A push to the PR head branch
```

## Step 1A (PR): Fetch PR + Failing Job Log

Prefer `gh` over `curl` — it handles auth, pagination, and redirects.

```bash
# Overview: find the failing check and its job id
gh pr view <PR> --repo <owner>/<repo> \
  --json number,title,headRefName,headRepositoryOwner,headRepository,baseRefName,statusCheckRollup

# Pull the raw job log (2-3 MB is fine; let the tool truncate to file if huge)
gh api repos/<owner>/<repo>/actions/jobs/<jobId>/logs
```

Notes:
- `gh run view --log-failed` and `gh run view --log` sometimes return empty for check-suite jobs on forks. Fall back to `gh api .../jobs/<jobId>/logs`.
- Save long logs to `agent_space/` only if you need to grep repeatedly; otherwise `tail -300` on the tool output is usually enough to capture the lintrunner section.

## Step 2A (PR): Parse the Lint Report from the Log

Each lintrunner error block in the log looks like:

```
>>> Lint for <path>:
  Error (<RULE_CODE>) <short name>
    <long description>
    >>> <lineno>  |<offending line>
```

Extract a table of `(rule_code, path, line, message)`. The GitHub Actions `##[error]...` summary at the bottom of the log lists the same errors; either is fine.

## Step 3A (PR): Locate the Authoritative Source Tree

Before editing anything, confirm where the PR's code actually lives.

```bash
gh api repos/<owner>/<repo>/pulls/<PR> --jq '.head.ref,.head.repo.clone_url'
```

Decision:

- **PR is against the repo you're currently in** (same `remote -v`): check out the PR locally with `gh pr checkout <PR>` on a throwaway branch, or fetch the PR ref without switching.
- **PR is against a different repo** (common: `intel/torch-xpu-ops` when you're in `pytorch/pytorch`; torch-xpu-ops ships as a submodule whose pinned commit lags the PR): **do not** edit the submodule or attempt `gh pr checkout`. Clone the fork shallowly into `agent_space/`:

  ```bash
  git clone --depth 1 --branch <head_ref> <head_clone_url> agent_space/<repo>-pr<PR>
  ```

All subsequent edits, diffs, and pushes happen inside that clone. The submodule in the main repo is irrelevant for fixing the PR.

### Always use a clean clone

Every invocation of this skill must start from a **freshly cloned** directory. Do not reuse an `agent_space/<repo>-pr<PR>/` that already exists from a prior session — it may contain stale local commits, a detached HEAD behind the remote, or edits from a previous attempt that would silently taint the diff.

```bash
# If agent_space/<repo>-pr<PR> exists, pick a fresh suffix or remove it.
# Prefer a new suffix so any prior work remains available for inspection:
CLONE_DIR="agent_space/<repo>-pr<PR>-clean"
[ -e "$CLONE_DIR" ] && CLONE_DIR="agent_space/<repo>-pr<PR>-$(date +%Y%m%d-%H%M)"
git clone --depth 1 --branch <head_ref> <head_clone_url> "$CLONE_DIR"
```

Immediately verify the clone is at the PR head: `git rev-parse HEAD` should match `gh api repos/<owner>/<repo>/pulls/<PR> --jq .head.sha`. If it doesn't, stop — something is wrong with the clone.

## Step 1B (local): Identify the lintrunner-enabled repo root

`.lintrunner.toml` and the adapter scripts under `tools/linter/adapters/` live at a repo root. Starting from the given path, ascend until both exist:

```bash
TARGET="<path the user gave>"
DIR="$(cd "$(dirname "$TARGET" 2>/dev/null || echo "$TARGET")" && pwd)"
while [ "$DIR" != "/" ] && [ ! -f "$DIR/.lintrunner.toml" ]; do
  DIR="$(dirname "$DIR")"
done
# $DIR is now the repo root; verify adapters exist
test -d "$DIR/tools/linter/adapters" || echo "No adapter directory found — ask user."
```

If no `.lintrunner.toml` is found on the way up, stop and ask the user whether they meant a different path or want to run raw flake8/ruff instead. Do **not** operate on a path that lies inside a git submodule unless the submodule itself has its own `.lintrunner.toml` — edits there are usually a mistake (see Step 3A's submodule warning).

Common cases:
- User points at `third_party/torch-xpu-ops` from a pytorch checkout → repo root is `third_party/torch-xpu-ops` *only if* its `.lintrunner.toml` is present and the user actually wants to edit the submodule working copy (ask). Otherwise they probably mean a PR; re-ask.
- User points at a file under `test/xpu/...` → ascend to the torch-xpu-ops root.
- User points at a directory like `/home/me/my_clone` → that's the root if `.lintrunner.toml` is directly there.

## Step 2B (local): Select target files

```bash
cd "$DIR"
```

Choose the set of files to lint:

- **Explicit file**: the exact path the user gave.
- **Explicit directory**: all files under it. Prefer `lintrunner --paths-cmd` over passing thousands of files manually; otherwise use `git ls-files -- <dir>`.
- **"Files changed in my branch"**: `git diff <base>...HEAD --name-only --diff-filter=AMR` where `<base>` is typically `origin/main` or `origin/master`. Confirm the base branch with `git remote show origin | grep 'HEAD branch'` if unsure.
- **Whole repo** (matches CI): `lintrunner --all-files` — usually too noisy; ask before choosing.

## Step 3B (local): Run lintrunner (or adapters) and parse

Preferred, if available:

```bash
# One-time per clone
lintrunner init

# Lint specific files with machine-readable output
lintrunner --tee-json=lint.json <files>
# Or apply auto-fixes for formatters
lintrunner -a <files>
```

`lint.json` is one JSON object per line, matching the CI format — parse the same way as Step 2A (each has `code`, `path`, `line`, `name`, `description`).

**If `lintrunner` is not installed**, per `AGENTS.md` do not auto-install. Ask the user whether to install it (typically via `uv pip install lintrunner && lintrunner init`) or to proceed adapter-by-adapter. For the adapter-only path:

```bash
# For each [[linter]] block in .lintrunner.toml whose include_patterns match the target file,
# invoke its `command` with @{{PATHSFILE}} replaced by a file listing the target paths.
printf '%s\n' <file1> <file2> > /tmp/paths.txt
python3 tools/linter/adapters/<adapter>.py <flags from .lintrunner.toml> -- $(cat /tmp/paths.txt)
```

This is how you reproduce CI failures locally without lintrunner installed (we used it on PR #3383 with `grep_linter.py`). It's tedious for many rules but sufficient to verify the specific rules that are failing.

From here, Mode A and Mode B converge.

## Step 4: Understand Each Failing Rule (the linter side)

Open `.lintrunner.toml` *in the working tree you're about to edit* (Mode A: the PR clone; Mode B: the repo root from 1B — don't read pytorch's config by mistake; configs diverge) and find the `[[linter]]` block with the failing `code`. Capture:

- `include_patterns` and `exclude_patterns` — scope
- `command = [...]` — which adapter in `tools/linter/adapters/` runs
- Adapter flags — especially whether `--allowlist-pattern` is set

### Does the rule honor `# noqa: <CODE>`?

Not all linters do. Read the adapter to find out. Quick reference for common adapters in PyTorch/torch-xpu-ops:

| Adapter | Honors `# noqa: <CODE>`? | Notes |
|---|---|---|
| `flake8_linter.py` (FLAKE8, B950, E731, ...) | Yes | Standard flake8 semantics |
| `ruff_linter.py` (RUFF, PYFMT via ruff) | Yes | Standard ruff semantics |
| `grep_linter.py` (META_NO_CREATE_UNBACKED, many custom rules) | **No** by default. Pass `--allowlist-pattern` to enable an allowlist token; bare `# noqa: ...` is *not* recognized. |
| `mypy_linter.py` (MYPY) | Yes via `# type: ignore[...]` | Not `# noqa` |
| `black_linter.py`, `clangformat_linter.py` | N/A (formatters) | |

**Key consequence**: for any rule backed by `grep_linter.py` without `--allowlist-pattern`, adding `# noqa: <CODE>` comments is *dead code*. The linter will still fire, and the comments mislead future readers. Either drop them or choose a different fix strategy.

### Confirm scope vs upstream

If the failing rule is a torch-xpu-ops custom rule, compare with `pytorch/pytorch`'s `.lintrunner.toml`. torch-xpu-ops sometimes broadens `include_patterns` to `**/*.py` for rules pytorch only applies to a single file (e.g. `META_NO_CREATE_UNBACKED` → `torch/_meta_registrations.py`). Such over-broad scopes are a frequent root cause when porting tests.

## Step 4.5: Understand the Failing Code (deep analysis)

Step 4 tells you how the rule fires. Step 4.5 tells you *whether the match is a real problem*. Do this **before** picking a fix strategy — otherwise you're pattern-matching, not engineering.

For each reported error, answer these in order. If an answer requires poking around more than the immediate file, use the `explore` subagent (see below) rather than manual ripgrep.

1. **Read the failing site with real context.** Not 3 lines — load at least 20 lines above and below. For errors inside a class, also read the class docstring/signature. For errors inside an `assertExpectedInline` / triple-quoted block, read the whole string.
2. **What is this code trying to do?** State it in one sentence (e.g. "this lambda is a Triton kernel grid function", "this is captured Dynamo output asserted inline"). If you can't, stop and explore more.
3. **Is the violation real, incidental, or a rule-scope mismatch?**
   - *Real*: the code is wrong by the rule's actual intent (e.g. a missing type annotation, a genuinely dead import).
   - *Incidental*: the code does the right thing but triggers the rule mechanically (Triton grid lambda hitting E731; `# noqa` suppresses it where the adapter honors noqa).
   - *Scope mismatch*: the rule was never meant to apply here (port of an upstream test hitting a linter aimed at `torch/_meta_registrations.py`). Confirm by reading the *upstream* file if one exists.
4. **Does the same pattern already exist elsewhere in this repo, and how is it handled?** If another file uses the exact construct and passes lint, copy that approach. This is the single highest-leverage check and is where `explore` pays off.
5. **Will the candidate fix change runtime behavior?** If yes, it's not a lint fix anymore — escalate to the user.

Only after these five questions have answers do you go to Step 5.

### When to launch the `explore` subagent

Use `task` with `subagent_type: explore` instead of doing it yourself when any of these is true:

- You need to check **more than ~3 files** to answer question 4 (how is this pattern handled elsewhere).
- You need to find **all occurrences** of a pattern across the repo to scope a fix (e.g. "find every `grid = lambda meta:` so we can confirm E731 noqa is the repo-wide convention").
- You need to **compare two repos** (e.g. this PR's `.lintrunner.toml` vs. `pytorch/pytorch`'s, or this file vs. its upstream source).
- You need to verify a claim about an **adapter's behavior** by reading its source plus its call sites across several rules.
- The lint report has **more than ~5 distinct rule codes** and you need a triaged summary before deciding strategies.

Do **not** launch `explore` for:
- A single file with a handful of errors of one rule code.
- Anything you can answer by reading one adapter and one `.lintrunner.toml` block.
- Iterative edit/verify cycles — do those yourself with `edit` + the adapter invocation from Step 3B.

Explore prompt template (keep it tight and ask for concrete deliverables):

```
Task(
  description="classify lint rule scope",
  subagent_type="explore",
  prompt="""
In the repo at <path>, for rule <RULE_CODE>:

1. Open .lintrunner.toml and extract include_patterns, exclude_patterns, and
   the adapter command for this rule.
2. Read tools/linter/adapters/<adapter>.py and report whether it honors
   `# noqa: <RULE_CODE>` (look for noqa handling or --allowlist-pattern).
3. Search the repo for existing occurrences of the failing pattern that are
   NOT flagged (i.e. legitimate uses). For each, report file:line and whether
   it uses noqa, getattr indirection, or is excluded via .lintrunner.toml.
4. If this file is a port of an upstream pytorch test (check filename against
   pytorch/test/), compare the corresponding upstream config scope.

Return: a single summary with (a) adapter honors noqa yes/no,
(b) count of existing legitimate occurrences and their handling strategy,
(c) upstream scope comparison if applicable. Do NOT edit any files.
"""
)
```

Thoroughness hint: use `"quick"` for one-file / one-rule checks, `"medium"` for port-vs-upstream comparisons, `"very thorough"` only when triaging a lint report with many rules or when the fix will touch many files.

## Step 5: Fix Strategy Matrix

Pick based on the Step 4.5 classification and whether the linter honors noqa:

| Situation | Strategy |
|---|---|
| Real violation, noqa honored | Fix the code |
| Real violation, noqa **not** honored, small edit possible | Rewrite to avoid the pattern (e.g. `getattr(obj, "create_unbacked_symint")()`) |
| Incidental, noqa honored | Add `# noqa: <CODE>` with a one-line justification comment |
| Incidental, noqa **not** honored | Add file to `exclude_patterns` in `.lintrunner.toml` with a rationale comment |
| Scope mismatch (rule aimed at different subtree) | Add file to `exclude_patterns`; if many files, consider narrowing `include_patterns` (asks user) |
| Formatter error (BLACK/RUFF/CLANGFORMAT) | `lintrunner -a` if available; otherwise apply the suggested diff from the log |
| Expected-output string blocks triggering B950 | `# noqa: B950` on the line with the terminating `"""` (per `AGENTS.md`) |
| Triton `grid = lambda meta: ...` (E731) | `# noqa: E731` on the line with `lambda`, not the closing paren |

When multiple strategies are viable (commonly: exclude file vs. narrow include_patterns vs. rewrite code), **ask the user** with a `question` tool call, recommending the most targeted, least-invasive option first.

## Step 6: Verify the Fix

Re-run the same linter that was failing, scoped to the modified file(s):

- **With lintrunner**: `lintrunner <file>` or `lintrunner --take <RULE_CODE> <file>` to target one rule.
- **Adapter-only** (see Step 3B for the general recipe). Example for a grep-based rule:

  ```bash
  python3 tools/linter/adapters/grep_linter.py \
    --pattern='create_unbacked' \
    --linter-name=META_NO_CREATE_UNBACKED \
    --error-name=test --error-description=test \
    -- <file>
  ```

  A successful fix produces no JSON objects on stdout. Match the error count against the original report before and after your edit.

- **Standalone fallbacks** when the failure is a well-known generic rule: `python3 -m ruff check <file>`, `python3 -m flake8 <file>`.

For `exclude_patterns` changes in `.lintrunner.toml`: verify by running `lintrunner <file>` (it reads the config); the adapter itself won't reflect the change because `lintrunner` is what applies `exclude_patterns`.

## Step 7: Apply the Fix

- Mode A: edit inside `agent_space/<repo>-pr<PR>/`, not the main repo or its submodules.
- Mode B: edit inside the repo root identified in Step 1B.
- Keep the diff minimal. Don't bundle unrelated reformatting with the lint fix.
- If you add an `exclude_patterns` entry, include an inline comment explaining the rationale (why the file is legitimately exempt).
- If you remove dead `# noqa` comments because the linter doesn't honor them, mention it in the commit message — otherwise it looks like a drive-by change.

### Only touch files that appear in the lint report

The set of files you may modify is exactly:

- Files listed in the lint report from Step 2A / Step 3B, **or**
- `.lintrunner.toml` (only when the Fix Strategy Matrix calls for an `exclude_patterns` / `include_patterns` change).

Nothing else. Do not fix unrelated linter advice you notice along the way, do not reformat surrounding code, do not "while I'm here" rename things. If a bulk formatter (`lintrunner -a`, `ruff format`, `black`) would touch files outside this set, run it **only on the in-scope paths**, never with `--all-files`.

Before staging, confirm the diff is scoped:

```bash
git status -s
git diff --stat
# Every path shown must be either (a) a file from the lint report, or
# (b) .lintrunner.toml. If not, revert the stray change with
# `git checkout -- <path>` before continuing.
```

If you believe an out-of-scope edit is genuinely required (e.g. the only fix for the reported rule is to adjust an import in a sibling file), stop and ask the user with `question` before making it — don't silently expand scope.

## Step 8: Commit and Push

Follow `AGENTS.md`:
- Do not commit unless the user asked. A user answering "yes" to a "commit and push?" question *is* explicit permission.
- Commit message: short subject, a paragraph explaining the *why*, no bullet list of individual file changes for small fixes. Preserve `ghstack-source-id` / `Pull-Request` trailers if present on prior commits. End with "Authored with Claude."
- **Mode A**: push to the PR head branch on the head fork, never to `main`:

  ```bash
  cd agent_space/<repo>-pr<PR>
  git push origin <head_ref>
  ```

- **Mode B**: there is no PR branch to push to. After committing locally, ask the user whether they want the commit pushed and to which remote/branch. Default to no push.

- Don't force-push unless the user asked or the remote has a dangling prior attempt by you in this session.

## Constraints

1. **Never install tools unprompted** — ask the user per `AGENTS.md`.
2. **Never edit the submodule pin** when fixing a torch-xpu-ops PR from inside a pytorch checkout.
3. **Always start from a fresh clone** (Mode A). Never reuse an existing `agent_space/<repo>-pr<PR>/` directory from a previous session — pick a new suffix or remove it first. See Step 3A "Always use a clean clone".
4. **Never modify files outside the lint report's scope.** The only permissible edits are (a) the files flagged by the linter and (b) `.lintrunner.toml` when the Fix Strategy Matrix calls for it. See Step 7 "Only touch files that appear in the lint report".
5. **Never assume `# noqa` works** — check the adapter. Dead noqa comments are a code smell, not a fix.
6. **Do not broaden `exclude_patterns`** to globs like `test/**/*.py` to make a single file pass; use the specific path.
7. **Do not squash / rewrite history** on the PR branch without user approval — the PR has reviewers tracking commits.
8. **Only commit when explicitly asked.** When offering fix strategies via `question`, the user's selection is permission to edit but *not* to commit; ask again before `git commit`.
9. **One fix, one commit.** If you fix two unrelated lint rules, consider two commits unless the user prefers otherwise.

## Tools Used

- `bash` — `gh`, `git`, adapter invocations, `lintrunner`
- `gh api` / `gh pr view` — PR metadata and job logs (Mode A)
- `read`, `grep`, `glob` — inspect `.lintrunner.toml`, adapter source, offending files; small/targeted lookups
- `task` with `subagent_type: explore` — Step 4.5 cross-file / cross-repo / multi-rule investigation. See triggers in Step 4.5.
- `edit`, `write` — apply fixes
- `question` — when fix strategies have meaningful trade-offs, or before committing
- `todowrite` — only for ≥3-step multi-file lint fixes

## Known Patterns

### `grep_linter.py`-backed rules with over-broad scope

Symptom: a rule that in pytorch only scans one file triggers on a ported test in torch-xpu-ops, and `# noqa: <CODE>` comments do nothing.

Example: `META_NO_CREATE_UNBACKED` fires on `test/xpu/dynamo/test_misc_xpu.py` (a port of `test/dynamo/test_misc.py`) because torch-xpu-ops sets `include_patterns = ["**/*.py"]`. Fix used in PR #3383: add the specific test file to `exclude_patterns` and drop the dead `# noqa` comments so the port matches upstream verbatim.

### `flake8`/`ruff`-backed rules (E731, B950, F401, ...)

`# noqa: <CODE>` works. Place it on the line where the pattern actually matches:
- E731 (`grid = lambda meta: ...`): on the `lambda` line, not the closing paren
- B950 inside a triple-quoted expected output: on the line with the closing `"""`, not inside the string

### Formatter conflicts

If `lintrunner -a` is available and the failure is purely formatting, a single `lintrunner -a <files>` + commit is usually enough. Review the diff first to confirm it's non-semantic.

### CUDA-specific tests failing XPU-specific lints

If the file needs substantive refactoring for XPU (device strings, decorators, imports), that's outside this skill — see `port-cuda-tests-xpu` and `fix-issues-identified-by-comments`. This skill only covers the lint layer.

## Worked Example: PR intel/torch-xpu-ops#3383

1. `gh pr view 3383 --repo intel/torch-xpu-ops ...` showed `preci-lint-check` FAILURE, job id in `detailsUrl`.
2. `gh api repos/intel/torch-xpu-ops/actions/jobs/72814124123/logs` returned six `META_NO_CREATE_UNBACKED` errors on `test/xpu/dynamo/test_misc_xpu.py`.
3. PR head was `daisyden/daisyden/dynamo_xpu` on `github.com/daisyden/torch-xpu-ops` — a fork of a different repo than the current checkout (pytorch). Cloned into `agent_space/torch-xpu-ops-pr3383/`.
4. `.lintrunner.toml` in that clone defined the rule with `include_patterns = ["**/*.py"]` and the `grep_linter.py` adapter with no `--allowlist-pattern`. The existing `# noqa: META_NO_CREATE_UNBACKED` comments in the file were therefore dead.
5. Fix strategies offered to user: (a) add file to `exclude_patterns` [recommended], (b) scope `include_patterns` to meta registrations, (c) rewrite calls via `getattr`. User chose (a).
6. Applied the exclude_patterns entry; separately, user asked whether to also remove the dead `# noqa` comments. Confirmed by diffing upstream `pytorch/test/dynamo/test_misc.py` (which has no such comments) and removed them to match.
7. Reproduced the original 6 errors by running the adapter directly (`python3 tools/linter/adapters/grep_linter.py --pattern=create_unbacked ...`) against the unfixed file, confirming the fix surface matches CI.
8. Committed (after user asked "ready to commit and push?") and pushed to `origin daisyden/dynamo_xpu`.

Key lessons from this example are baked into the Fix Strategy Matrix and Constraints above.
