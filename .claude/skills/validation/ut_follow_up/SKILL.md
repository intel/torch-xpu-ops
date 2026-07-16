---
name: ut-follow-up
description: Analyze XPU unit test failures, classify root causes, fix test code issues, and submit well-documented issues to intel/torch-xpu-ops using structured issue templates, proper labels, and Context sections cross-linking porting PRs. Use when analyzing XPU test failures, filing issues on intel/torch-xpu-ops, or when user mentions UT follow-up, submitting UT issues, XPU test failures, or filing XPU bug reports.
---

# XPU UT Follow-up for intel/torch-xpu-ops

## Overview

Orchestrates the end-to-end flow from failing XPU unit tests to filed issues:
run and analyze failures, fix the ones that are test-code bugs, and submit
well-structured issues for the rest. Each stage is delegated to a focused
subskill; this skill coordinates them and enforces the decision hierarchy.

When a failure is observed during a porting/enablement PR on
`intel/torch-xpu-ops`, every submitted issue MUST include a **Context** section
cross-linking the PR (enforced by `create-xpu-issue`).

## Subskills

| Stage | Subskill | Responsibility |
|---|---|---|
| Analyze | `analyze-ut-results` | Run tests, group failures, cross-ref known issues, verdict per group |
| Fix | `fix-ut-test-code` | Apply allowed test-code fixes and rerun, or escalate |
| Submit | `create-xpu-issue` | Build + submit the structured issue, return cross-link actions |

## Input Modes

This skill handles three entry points:

1. **From scratch** — given test file(s). Run Phase 1 (analyze) -> Phase 2
   (fix) -> Phase 3 (submit).
2. **From classify-ut** — given a list of pre-classified `Submit Issue` rows
   (each with `name_xpu`, `classname_xpu`, `testfile_xpu`, `message_xpu`,
   `status_xpu`). **Skip Phase 1's pytest run**: treat each row's
   `message_xpu` as the failure signature, group rows that share a signature,
   run only the known-issue cross-reference step of `analyze-ut-results` to
    avoid duplicates. Each row then resolves to a **PR** (if it is a test-code
    bug fixed via Phase 2) or an **issue** (Phase 3), and is reported back per
    the Return Contract.
3. **Targeted tuple mode** — given explicit `test_file`, `test_class`, and
   `test_cases` (list of test method names). Convert this input to the
   `analyze-ut-results` target list format:

   ```json
   [
     {
       "test_file": "<test_file>",
       "test_class": "<test_class>",
       "test_name": "<one entry from test_cases>"
     }
   ]
   ```

   then run the same pipeline (Phase 1 -> Phase 2 -> Phase 3). Keep
   `test_file`/`test_class`/`test_name` identity through all phases so outputs
   map back 1:1 to the requested `test_cases`.

## Confirm Before Filing (MANDATORY)

Never create a PR, POST a GitHub issue, or `git push` without explicit user
confirmation, regardless of input mode. The flow is always
**draft -> confirm -> file**:

1. Prepare the full draft — an issue body via `create-xpu-issue`, or the
   test-code fix diff + PR title/body (do NOT submit yet).
2. Present each draft to the user and ask for explicit per-item approval
   (use the `question` tool: approve / edit / skip).
3. Only submit the approved drafts. Skipped drafts are not filed.

This keeps a human in the loop even when `classify-ut` routes work here
automatically.

## Preconditions

1. **Environment** — canonical local XPU env, bootstrapped once:
   ```bash
   bash .opencode/skills/validation/scripts/setup_env.sh nightly pytorch_opencode_env
   source ~/miniforge3/bin/activate pytorch_opencode_env
   python3 -c "import torch; print('XPU:', torch.xpu.is_available())"
   ```
   `setup_env.sh` installs PyTorch nightly XPU + triton, `junitparser`, and `gh`.
2. **GitHub auth** — `GITHUB_TOKEN` with `repo` scope exported (`gh auth status`).
3. **Git remotes** — `intel` (upstream) and `daisyden` (fork) configured in
   `third_party/torch-xpu-ops`.

## Decision Hierarchy

```
For each failing test group, in order:
1. Fixable in TEST CODE? (import path / CUDA->XPU API / missing skip guard / syntax)
   -> fix-ut-test-code -> rerun -> if passes/skips cleanly, done.
2. Matches a KNOWN issue (pytorch/pytorch or intel/torch-xpu-ops)?
   -> attach the verified issue; submit only if a new tracking issue is warranted.
3. NEW backend/infrastructure/pytorch-codebase bug?
   -> create-xpu-issue (group by error pattern, one issue per pattern).
```

## Workflow

### Phase 1 — Analyze

Delegate analysis of the target test file(s). The subskill runs tests, groups
failures, cross-references known issues, and returns a verdict per group.

For targeted tuple mode (`test_file`, `test_class`, `test_cases`), expand into a
list of `{test_file, test_class, test_name}` entries (one per `test_cases`
item), then pass that list to `analyze-ut-results`.

```python
task(
    subagent_type="explore",
    load_skills=["analyze-ut-results"],
    description=f"Analyze failures: {test_file}",
    prompt=f"Run and analyze the provided target(s) on XPU. "
           f"If targeted tuple mode was provided, use this explicit list of "
           f"{{test_file,test_class,test_name}} entries: {targets_json}. "
           f"Group failures by error pattern, cross-reference pytorch/pytorch "
           f"and intel/torch-xpu-ops known issues, and return JSON groups with "
           f"category, root_cause, known_issues, and verdict."
)
```

### Phase 2 — Fix (for groups with `verdict == fix-ut-test-code`)

```python
task(
    subagent_type="explore",
    load_skills=["fix-ut-test-code"],
    description=f"Fix test code: {test_names}",
    prompt=f"Apply allowed test-code fixes for {test_names} in {test_file}. "
           f"Root cause: {root_cause}. Rerun to confirm. Return outcome JSON; "
           f"if the root cause is not test code, revert and escalate to issue submission."
)
```

- If `outcome` is `fixed-passing` / `fixed-skipping`, the fix is real:
  prepare a **PR draft** (diff + title/body), present it for approval, and on
  approval submit it via `gh pr create`. The group's result is
  `outcome = "pr"` with the PR URL.
- If `outcome == escalate-to-issue`, route that group to Phase 3 (issue).

### Phase 3 — Submit issue (for groups with `verdict == create-xpu-issue` or escalated)

Build the draft, present it, and file only after approval (see
"Confirm Before Filing").

```python
task(
    subagent_type="explore",
    load_skills=["create-xpu-issue"],
    description=f"Draft issue: {signature}",
    prompt=f"Build a DRAFT intel/torch-xpu-ops issue for failure group '{signature}' "
           f"covering tests {test_names}. Root cause: {root_cause}. "
           f"Related issues: {known_issues}. PR context: {pr_number_or_none}. "
           f"Return the full draft body and labels; do NOT POST it."
)
```

Present each returned draft to the user, get per-issue approval via the
`question` tool, then POST only the approved drafts and collect their URLs and
after-filing cross-link actions.

### Phase 4 — Cross-link

For every filed issue that used a Context section, perform the returned
after-filing actions: add the issue URL to the porting PR commit message, add
an inline `# Tracked in intel/torch-xpu-ops#NNNN` comment next to the skip, and
append the URL to the PR's tracking-issues section.

## Return Contract

Return one entry per input row so the caller can record results. Key each
entry by the test's CUDA identity:

```json
[
  {
    "name_cuda": "...",
    "classname_cuda": "...",
    "testfile_cuda": "...",
    "outcome": "pr|issue|skipped",
    "url": "PR or issue link, or null when skipped",
    "summary": "one-line description"
  }
]
```

- `outcome == "pr"` — a test-code fix was submitted as a PR; `url` is the PR link.
- `outcome == "issue"` — an issue was filed; `url` is the issue link.
- `outcome == "skipped"` — user declined or it duplicates an existing item;
  `url` is null (cite the existing link in `summary`).

Include every handed-off row, even skipped ones.

For targeted tuple mode input, map identities as:
- `testfile_cuda` <- input `test_file`
- `classname_cuda` <- input `test_class`
- `name_cuda` <- each entry from input `test_cases`

## Constraints

- **Tools**: `bash` (`pytest`, `curl`, `gh`, `git`), `read`, `edit`, `grep`,
  `task`. No web tools; `gh search issues` must use `is:issue`.
- One issue per error pattern; group related tests. Apply <=5 labels including
  the required `skipped` and `module: ut`.
- Never invent issue numbers; cite only `gh issue view`-verified issues.
- **Never create a PR, POST an issue, or `git push` without explicit user
  approval** (draft -> confirm -> file), even when invoked automatically by
  `classify-ut`.
- Test-code fixes are limited to the allowlist in `fix-ut-test-code`; never
  modify backend/infra code or test assertions.
- Do not commit unless explicitly asked. ASCII only in authored content.
