---
name: quick-workflow
description: Fast-path XPU test enablement for one or more test classes, skipping the review-refactoring gate and issue-filing follow-up. Runs develop-xpu-test (source edits), then pytest plus analyze-ut-results (verification/failure triage), then submit-xpu-test-pr (confirm-gated draft PR), grouped by test_file, with every stage logged under agent_space/quick_workflow/ (backed up and recreated fresh at the start of each run). Never edits test method body logic (routes latent bugs to followup instead) and never cites a closed GitHub issue to justify a skip/xfail. Input is a list of test_file/test_class pairs plus conda_env and pytorch_folder; supports a single class or many across multiple files. Use when a test file is already known/trusted to be device-generic and you just want XPU turned on and a PR opened quickly, without the full enable-xpu-test orchestrator's review gate, environment provisioning, or known-issue/create-issue phases.
---

# Quick Workflow: Develop -> Analyze -> Submit

A minimal, fast-path XPU-enablement pipeline. Three stages only:

```
develop-xpu-test  ->  pytest + analyze-ut-results  ->  submit-xpu-test-pr
```

This is a **stripped-down sibling of `enable-xpu-test`**, not a replacement for
it. It deliberately skips:

- The `review-test-refactoring` gate (Phase 1 of `enable-xpu-test`) — assume the
  caller has already vetted that each target file/class is device-generic.
- Environment provisioning (Phase 0.5) — `conda_env` and `pytorch_folder` must
  already exist and work.
- The known-issue-check / create-issue follow-up (Phase 5B) — failures are
  reported and left as `@expectedFailureXPU`/`skipOps` markers or flagged in
  the final report; filing GitHub issues is out of scope here.

Use `enable-xpu-test` instead when you need the review gate, provisioning, or
issue-filing automation. Use `quick_workflow` when you already know the target
classes are clean and you just want XPU on, verified, and a PR opened.

## Inputs

| Field | Required | Description |
|---|---|---|
| `test_targets` | Yes | List of `{test_file, test_class}` (paths relative to `pytorch_folder`). A single `{test_file, test_class}` object is accepted as a one-element list — this skill supports both a single class and multiple classes across one or more files. |
| `conda_env` | Yes | Existing conda env with XPU-enabled PyTorch installed. Not created here — if missing, stop and tell the caller to provision it first (e.g. via `enable-xpu-test`'s Phase 0.5 / `setup_env.sh`). |
| `pytorch_folder` | Yes | Existing local pytorch/pytorch checkout. Not cloned here — if missing, stop and tell the caller to provision it first. |

Example multi-class, multi-file input:

```json
{
  "test_targets": [
    {"test_file": "test/distributions/test_distributions.py", "test_class": "TestDistributions"},
    {"test_file": "test/export/opinfo_schema.py", "test_class": "TestOpInfo"},
    {"test_file": "test/functorch/test_aotdispatch.py", "test_class": "TestPythonKey"},
    {"test_file": "test/functorch/test_aotdispatch.py", "test_class": "TestEagerFusionOpInfo"}
  ],
  "conda_env": "pytorch_opencode_env",
  "pytorch_folder": "/home/daisyden/daisy_pytorch"
}
```

## Session Setup (MANDATORY, first action of every run)

All logging for this workflow lives under `agent_space/quick_workflow/` in the
current working directory (the repo the orchestrating session is running in —
NOT `pytorch_folder`). Every run starts clean:

```bash
# 1. Back up any previous run's logs (never silently overwrite them).
if [ -d agent_space/quick_workflow ]; then
  ts=$(date -u +%Y%m%dT%H%M%SZ)
  mv agent_space/quick_workflow "agent_space/quick_workflow.bak.${ts}"
fi

# 2. Create a fresh log directory for this run.
mkdir -p agent_space/quick_workflow/logs

# 3. Start the session log.
date -u +"[%Y-%m-%d %H:%M:%S] session_start | test_targets=<N> | conda_env=${conda_env} | pytorch_folder=${pytorch_folder}" \
  >> agent_space/quick_workflow/session_log.txt
```

Do this **before** the precondition check below, so even a precondition
failure is captured in a fresh `session_log.txt` rather than appended to a
stale one from a prior run.

## Precondition Check (before Stage 1)

Verify both inputs exist; do NOT provision them yourself:

```bash
conda env list | awk '{print $1}' | grep -qx "${conda_env}" || echo "MISSING conda_env: ${conda_env}"
test -d "${pytorch_folder}/.git" || echo "MISSING pytorch_folder: ${pytorch_folder}"
conda run -n "${conda_env}" python3 -c "import torch; assert torch.xpu.is_available()" \
  || echo "BROKEN env: torch.xpu.is_available() is False in ${conda_env}"
```

Log the outcome either way:

```bash
date -u +"[%Y-%m-%d %H:%M:%S] precondition_check | conda_env=<ok|missing> | pytorch_folder=<ok|missing> | xpu_available=<true|false>" \
  >> agent_space/quick_workflow/session_log.txt
```

If any check fails: **stop**, append `[FATAL] precondition_check: <reason> — halting session`
to `session_log.txt`, report the missing/broken precondition, and tell the
caller to provision `conda_env`/`pytorch_folder` (e.g. via `enable-xpu-test`
Phase 0.5) before retrying this skill. Do not attempt to fix it here.

## Execution Model (group by file)

Group `test_targets` by `test_file`. For each file group, run Stage 1
(develop) once for all its classes, then Stage 2 (verify + analyze) once for
the group. This mirrors `enable-xpu-test`'s file-grouping so a shared file is
read/edited once regardless of how many classes in it are targeted.

Accumulators, carried across all groups into Stage 3:
- `passed_classes` — edits verified clean (0 unexplained failures after any
  `analyze-ut-results`-recommended markers are applied); staged with `git add`.
- `followup_classes` — edits that could not be made clean (see Stage 2 Step 3);
  reverted, reported to the user, NOT staged.

## Logging (MANDATORY, every stage, under `agent_space/quick_workflow/`)

| File | Written by | Content |
|---|---|---|
| `session_log.txt` | Session Setup, every stage | One-line-per-event append log: `[timestamp] <step> \| file_group: <test_file> \| classes: [<c...>] \| detail: <brief>` |
| `stage1_develop.json` | Stage 1 | Keyed by file-group slug (e.g. `test_distributions_py`); each entry holds the per-class diff summary and edit outcome (`enabled` / `skipped-no-device-axis`). |
| `stage2_verify_analyze.json` | Stage 2 | Keyed by file-group slug; each entry holds the pytest summary counts, any `analyze-ut-results` groups, and the final per-class outcome (`clean` / `followup`). |
| `stage3_submit.json` | Stage 3 | The final `submit-xpu-test-pr` result: branch, commit message, diff stat, PR url or `null`, and approval decision. |
| `logs/<file_group_slug>_pytest.log` | Stage 2 | Raw stdout of every pytest invocation for that file group (Step 1's first run, any re-runs after markers are added, and the final confirmation run — append, do not overwrite). |

Write the JSON files with `python3 -c "import json; ..."` or the `write` tool;
append (never truncate) `session_log.txt` and the per-group `*_pytest.log`
files. Update each stage's JSON file as that stage completes for each file
group — do not defer all writes to the very end (a mid-run failure should
still leave a readable partial log).

## Stage 1: Develop (per file group)

For each `file_group = (test_file, [classes])`:

```
task(subagent_type="explore", load_skills=["develop-xpu-test"], run_in_background=False,
     prompt="Enable XPU for classes [<classes>] in <pytorch_folder>/<test_file>. ...")
```

Instruct the subagent to:
- Apply `develop-xpu-test`'s Step 2-4 edits (instantiation enablement, decorator
  parity, op_db `DecorateInfo` widening) for **every class in this group**,
  scoping op_db edits per-class exactly as that skill requires.
- Skip that skill's own Step 1 review gate call — the caller of
  `quick_workflow` is asserting the file/classes are already clean. (If the
  subagent discovers a genuine structural blocker while editing — e.g. a class
  has no device axis at all, as with a plain `unittest.TestCase` — it should
  stop editing that one class, explain why, and continue with the rest of the
  group rather than silently forcing an edit.)
- Honor `develop-xpu-test`'s Constraint 3a: never edit test method body logic
  or add a skip/inline-conditional workaround, even for an obvious latent bug
  discovered while editing. Report any such bug (file, line, code, suggested
  fix) alongside the diff instead — Stage 2 will surface it again via pytest
  and route it to `followup_classes` rather than have Stage 1 paper over it.
- Report the diff, grouped per class.

Keep the subagent's `task_id` to reuse for Stage 2 (file already read/edited
once).

After the subagent returns, write/update `agent_space/quick_workflow/stage1_develop.json`
for this file group and append a line to `session_log.txt`:

```bash
date -u +"[%Y-%m-%d %H:%M:%S] stage1_develop | file_group: ${test_file} | classes: [${classes}] | detail: <N edited, M skipped-no-device-axis>" \
  >> agent_space/quick_workflow/session_log.txt
```

## Stage 2: Verify + Analyze (per file group, reusing Stage 1 session)

### Step 1: Run pytest for the whole group

```bash
source <conda_activate> "${conda_env}"
cd "${pytorch_folder}"
pytest "<test_file>" -k "<ClassA> and xpu or <ClassB> and xpu or ..." \
  --timeout 600 --tb=line -q 2>&1 | tee -a "<agent_space_dir>/quick_workflow/logs/<file_group_slug>_pytest.log"
```

(`<agent_space_dir>` is the absolute path to the working directory's
`agent_space` from Session Setup — use an absolute path here since `cd`
changes into `pytorch_folder` first.)

Use one invocation covering every class in this file group's `xpu` variant.
Record the summary line (passed/failed/skipped/xfailed) and, if any `FAILED`
lines are present, their exact test node ids.

### Step 2: Diff scope sanity check

Quick version of `verify-xpu-test`'s Step 0 (not a full gate, just a sanity
check since the review gate was skipped entirely here): confirm the diff
touches only `<test_file>` and, if applicable, `common_methods_invocations.py`
entries whose `DecorateInfo` class/test-name match one of this group's target
classes.

```bash
git diff --stat
git diff -- torch/testing/_internal/common_methods_invocations.py
```

If an out-of-scope `DecorateInfo` (different class/test name) is found, revert
just that hunk and note it in the report — do not fail the whole group over it.

### Step 3: Analyze failures (only if Step 1 had failures)

If the pytest run in Step 1 was all pass/skip/xfail for every class in the
group, skip this step entirely — adopt `passed` for each class from the pytest
counts directly (no subagent spawn needed).

If any class has failures, dispatch `analyze-ut-results` **once per file
group** covering all its failing classes:

```
task(task_id=<stage1_task_id>, load_skills=["analyze-ut-results"], run_in_background=False,
     prompt="Analyze XPU failures in <test_file> for classes [<failing classes>]. ...")
```

Follow `analyze-ut-results`'s own workflow (group by error signature,
cross-reference known pytorch/pytorch and intel/torch-xpu-ops issues, verdict
per group). Before treating any cited GitHub issue as evidence, verify its
current state — `gh issue view <number> --repo <owner>/<repo> --json
state,stateReason`. Only an `OPEN` issue is valid justification for adding or
keeping a device-conditional skip/xfail; a `CLOSED` issue (even one titled
`[Bug Skip]: ...`) usually means the gap was already resolved through the
project's own skip-list mechanism, not that new source-level gating is
warranted — re-verify the actual current behavior instead of trusting a
closed issue's title.

For each failure group returned:

- **Root cause is proven non-XPU-specific or already tracked by an OPEN
  upstream issue** (matches `analyze-ut-results`' `test-code`/
  `infrastructure`/`backend`/`pytorch-codebase` categories with a verified-open
  known issue, or is independently reproducible on CPU) -> add a targeted
  `@expectedFailureXPU` (single-method classes) or extend the class's
  `skipOps`/xfail set (op_db-driven classes) citing the issue/evidence,
  matching the file's existing decorator/skip convention. Re-run the affected
  subset of Step 1's pytest command (append to the same
  `<file_group_slug>_pytest.log`) to confirm the class now shows 0 unhandled
  failures (failures become `xfailed`/`skipped`). If the added
  `expectedFailure` unexpectedly **passes** (e.g. the bug is driver-version- or
  hardware-specific and doesn't reproduce on this host), **revert it** — do not
  leave a decorator that itself fails "expected test to fail, but it passed."
- **Root cause is a latent bug in the test method's own body logic** (e.g. a
  `torch.randint(...)`/similar call missing a `device=` kwarg, previously
  masked by a device-specific early-return that never fired for XPU) -> do
  **NOT** edit the method body and do **NOT** add a decorator to route around
  it (see `develop-xpu-test` Constraint 3a — this workflow inherits that
  boundary even during Stage 2/3 remediation, not just Stage 1). Revert this
  class's Stage 1 edits, move it to `followup_classes` with the exact bug
  description (file, line, current code, suggested one-line fix labeled "not
  applied by this workflow"), and report it plainly. A body-logic fix is a
  separate, dedicated change (e.g. via the `fix-ut-test-code` skill or
  explicit user request) — never bundled into this workflow's PR.
- **Root cause is a genuine, unresolved, in-scope defect** that a decorator
  cannot responsibly paper over (e.g. the class's own device-plumbing is
  broken, not an op-level gap) -> do not add a blanket xfail. Revert this
  class's edits, move it to `followup_classes`, and report the root-cause
  finding plainly. Filing an issue for it is out of scope for this skill — the
  caller/user decides whether to escalate to `create-xpu-issue` separately.

After resolving all groups for this file, re-run Step 1's pytest command once
more for the whole group (again appended to the same log file) to get the
final, authoritative summary. A class only moves to `passed_classes` when its
final run shows **0 unhandled `FAILED`** (xfailed/skipped is fine).

### Step 4: Stage classes and log

For each class in the group:
- **Clean (0 failed)** -> `git add` the file(s) it touches (if not already
  staged by a sibling class in the same file); append to `passed_classes`.
- **Could not be made clean** (Step 3's second bullet) -> ensure its hunks are
  reverted; append to `followup_classes` with the root-cause note.

Write/update `agent_space/quick_workflow/stage2_verify_analyze.json` for this
file group (pytest summary, analyze groups if any, final per-class outcome),
and append to `session_log.txt`:

```bash
date -u +"[%Y-%m-%d %H:%M:%S] stage2_verify_analyze | file_group: ${test_file} | classes: [${classes}] | detail: <pytest summary, N passed to passed_classes, M to followup_classes>" \
  >> agent_space/quick_workflow/session_log.txt
```

## Stage 3: Submit (once, after all groups processed)

If `passed_classes` is empty, do **not** call `submit-xpu-test-pr` — report
`followup_classes` and stop.

Otherwise, call `submit-xpu-test-pr` once for the whole batch:

```
task(subagent_type="explore", load_skills=["submit-xpu-test-pr"], run_in_background=False,
     prompt="Submit one combined draft PR for the following enabled classes: ...")
```

Follow that skill's workflow exactly as written — it is unmodified here:
inspect the working tree, choose/create a branch, **rebase onto upstream
pytorch/pytorch viable/strict (or main) before committing**, draft the commit
message (root cause + Test Plan with literal pytest commands per class,
AI-assistance disclosure), and **present the diff + commit message + PR target
for explicit user approval before any `git commit`, `git push`, or
`gh pr create`**. This confirm-gate is non-negotiable and applies even when
`quick_workflow` was invoked automatically.

If `followup_classes` is non-empty, list them plainly in the final report
(file, class, root-cause note) alongside the PR result — do not silently drop
them.

Write `agent_space/quick_workflow/stage3_submit.json` (branch, commit message,
diff stat, PR url or `null`, approval decision) and append the final line to
`session_log.txt`:

```bash
date -u +"[%Y-%m-%d %H:%M:%S] stage3_submit | detail: <status, pr_url or none>" \
  >> agent_space/quick_workflow/session_log.txt
```

## Output

```json
{
  "status": "submitted|passed-no-pr-needed|partial|no-passing-classes",
  "pr_url": "... or null",
  "passed_targets": ["<test_file>::<test_class>", "..."],
  "followup_targets": [
    {"test_file": "...", "test_class": "...", "root_cause": "..."}
  ],
  "per_target": [
    {"test_file": "...", "test_class": "...",
     "pytest_summary": "N passed, M xfailed, 0 failed",
     "outcome": "enabled|followup"}
  ],
  "logs": [
    "agent_space/quick_workflow/session_log.txt",
    "agent_space/quick_workflow/stage1_develop.json",
    "agent_space/quick_workflow/stage2_verify_analyze.json",
    "agent_space/quick_workflow/stage3_submit.json"
  ]
}
```

`status`:
- `submitted` — PR opened for `passed_classes`.
- `partial` — PR opened, but `followup_classes` is non-empty.
- `no-passing-classes` — every class ended up in `followup_classes`; no PR.

## Constraints

1. **No review gate.** This skill does not call `review-test-refactoring`. The
   caller is responsible for having already confirmed the target classes are
   device-generic. If `develop-xpu-test` discovers mid-edit that a class has no
   device axis at all, skip editing that class and report it — do not force an
   edit or silently skip the report.
1a. **Never edit test method body logic.** This workflow's Stage 1/2 remediation
   is limited to `develop-xpu-test`'s three edit types plus targeted
   `@expectedFailureXPU`/`skipOps` additions (Constraint 5 below). A bug found
   inside a test method's own body (missing `device=` kwarg, etc.) is never
   patched or routed around with an inline conditional — route the class to
   `followup_classes` with the bug description instead. See `develop-xpu-test`
   Constraint 3a, which this workflow inherits without exception.
1b. **Never cite a closed GitHub issue as justification for a gate.** Before
   using any `analyze-ut-results`-surfaced issue as evidence for an
   `@expectedFailureXPU`/`skipOps` addition, verify it is `OPEN` via
   `gh issue view <number> --repo <owner>/<repo> --json state,stateReason`. A
   closed issue means the gap was likely already resolved elsewhere (e.g. the
   project's own skip-list); re-verify actual behavior instead.
2. **No environment provisioning.** `conda_env`/`pytorch_folder` must already
   exist and work; missing/broken preconditions are a hard stop, not
   auto-provisioned.
3. **No issue filing.** `analyze-ut-results` may surface known-issue URLs as
   evidence for an `@expectedFailureXPU`/`skipOps` decision, but this skill
   never calls `check-known-issue` or `create-xpu-issue` itself.
4. **Group by file.** Run Stage 1 once per `test_file`, not once per class;
   reuse the Stage 1 `task_id` for Stage 2 so the file is read/edited once.
5. **A class is only `passed` when its final pytest run shows 0 unhandled
   `FAILED`.** xfailed/skipped counts as clean; a bare `FAILED` does not. An
   `@expectedFailureXPU`/`skipOps` entry that itself fails with "expected test
   to fail, but it passed" is NOT clean — revert that specific entry rather
   than leaving an incorrect xfail in the diff (see Stage 2 Step 3).
6. **Per-class isolation.** One class needing revert/follow-up never blocks or
   discards sibling classes' edits in the same file or group.
7. **PR submission is confirm-gated.** `submit-xpu-test-pr`'s constraints
   (never commit/push/open a PR without explicit user approval, rebase onto
   viable/strict first, stage explicit paths only) apply unchanged and are
   never bypassed by this skill.
8. **Single call, once at the end.** `submit-xpu-test-pr` is called exactly
   once for the whole batch, never per-class or per-file-group.
9. **ASCII only** in any new code/comments; match each file's existing style.
10. **Logging is mandatory and session-scoped.** Every run starts by backing
    up any existing `agent_space/quick_workflow/` (rename to
    `agent_space/quick_workflow.bak.<UTC-timestamp>/`, never delete) and
    creating a fresh one; every stage writes its JSON log and appends to
    `session_log.txt` as it completes, not only at the very end.

## See Also

- `enable-xpu-test` — the full orchestrator (adds review gate, environment
  provisioning, and known-issue/create-issue follow-up) that this skill is a
  fast-path alternative to.
- `develop-xpu-test`, `analyze-ut-results`, `submit-xpu-test-pr` — the three
  subskills this workflow chains together, unmodified.
