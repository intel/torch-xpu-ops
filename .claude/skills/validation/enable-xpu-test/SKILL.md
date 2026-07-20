---
name: enable-xpu-test
description: End-to-end orchestrator for XPU test enablement. Takes one or more test targets (test_file/test_class list) plus conda_env/pytorch_folder; groups targets by test_file and runs the review-test-refactoring gate, develop-xpu-test, verify-xpu-test, and (only-on-failure) analyze-ut-failures ONCE PER FILE for all its classes, reusing subagent sessions via task_id to save tokens, with per-class isolation on failure, then submits ONE combined draft PR (rebased onto pytorch/pytorch strict/viable) for all passing classes while routing failing classes to known-issue/issue-creation follow-up. Mandatory step/subagent logging to agent_space and hard-stop only on batch-wide critical errors.
---

# Enable-XPU-Test Orchestrator

Coordinate subskills to enable XPU on one or more test targets, then open ONE
combined draft PR. This skill orchestrates; it does not replace the subskills.

Phases: 0 provision -> 1 review -> 2 develop -> 2.5 remove-skips -> 3 verify
-> 4 analyze (only-on-failure) -> 5 submit. Phases 1-4 run once per
`test_file` group; Phase 5 runs once at the end. Phase 2.5 is optional — run
it only when you want to probe and remove stale method-level skips after
enablement.

## Inputs

- `test_targets` (required): list of `{test_file, test_class, test_cases?}`
  (paths relative to `pytorch_folder`). A bare `test_file` + `test_class` pair is
  accepted as a one-element list.
- `conda_env` (required): conda env for local test/verify. Created by
  `setup_env.sh` if missing (Phase 0.5).
- `pytorch_folder` (required): local pytorch checkout root. Cloned/prepared by
  `setup_env.sh` if missing (Phase 0.5).

## Subskills Used

| Phase | Skill | Purpose |
|---|---|---|
| 1 Review | `review-test-refactoring` | Quality gate before enablement |
| 2 Develop | `develop-xpu-test` | Apply XPU-enable source edits (per-class op_db scoping) |
| 2.5 Remove skips | `remove-xpu-skips` | Probe and remove stale method-level XPU skips (issue-gated, one skip at a time) |
| 3 Verify | `verify-xpu-test` | Local XPU verification |
| 4 Analyze | `analyze-ut-failures` | Group failures, return verdict (only on failure) |
| 5B Known issue | `check-known-issue` | Is a failing case already tracked? |
| 5B New issue | `create-xpu-issue` | File a tracking issue when none exists |
| 5A PR | `submit-xpu-test-pr` | Rebase + commit + push + open ONE draft PR |

## Execution Model (group-by-file, batch each phase)

Group `test_targets` by `test_file`. Run Phases 1-4 **once per file group** for
all its classes (each subskill returns **per-class** results). This avoids
re-reading the file and re-grepping `common_methods_invocations.py` per class.

- **Session reuse:** within a file group, chain develop -> remove-skips ->
  verify (-> analyze) via `task_id` so the file is read once.
- **Analyze only on failure:** skip `analyze-ut-failures` for classes that
  verified all-pass (use verify counts as verdict `passed`); spawn it only for
  classes with failures/xpass.
- **Per-class isolation:** a class failing review/verify/analysis does NOT abort
  the group or batch. Surgically revert only that class's hunks and route it to
  Phase 5B; keep siblings. Only batch-wide critical errors hard-stop.
- **One PR at the end:** Phase 5 runs once across all groups. Zero passing
  classes => no PR.

Spawn budget: ~3-4 calls per file + one final PR call (vs 4 per class).

Accumulators: `passed_classes` (edits kept, `git add`ed), `pending_classes`
(edits kept pending Phase 5B resolution — backend-gap failures that may get
`@skipIfXpu`), `followup_classes` (edits reverted, routed to 5B for issue
filing only). Common params below: `E={conda_env}`, `P={pytorch_folder}`.

## Logging (MANDATORY, all under `agent_space/`)

Create/append `session_log.txt`, `logs/background_status.log`, and per-phase
JSON under `enable_xpu_orchestrator/` (key by file-group slug; store a per-class
`results` array). Formats:

```text
[YYYY-MM-DD HH:MM:SS] <step> | subagent: <skill> | task: <brief> | file_refs: <refs>
Delegated: <phase> | subagent_type: <type> | load_skills: [<skills>] | task_count: <N> | batch_key: <test_file> | classes: [<c...>] | task_id: <reused-or-new>
```

## Phase 0.5: Environment Provisioning (conditional)

Resolve inputs before Phase 1. `conda_env` exists if `conda env list` shows it;
`pytorch_folder` exists if it is a dir containing `.git` (a bare unresolved name
= missing).

If either is missing, run `setup_env.sh` ONCE (it creates the env and prepares
the checkout + torch-xpu-ops pin together). For a bare `pytorch_folder` name,
pass a concrete path (default `$HOME/daisy_pytorch`):

```bash
bash .opencode/skills/validation/scripts/setup_env.sh nightly "<conda_env>" "<pytorch_folder_abs>"
# args: [build_type=nightly] [env_name] [pytorch_folder] [torch_version?]
```

Then verify: `conda env list` shows the env; `<pytorch_folder>/.git` and the
target `test_file`(s) exist; `python -c "import torch; assert torch.xpu.is_available()"`
passes in the env. If still broken after `setup_env.sh`, treat as a critical
error (hard-stop). Log to `session_log.txt` and `phase0_5_setup_env.json`.

## Phase Loop (per file group)

Working-tree discipline: after a class PASSES, `git add` its file(s). If a class
has failures, keep its edits in the tree and route to Phase 5B. Revert only
when Phase 5B determines the failure is a test-code bug or has no known issue —
never `git checkout --` a whole file shared with other classes (`test_file` or
`common_methods_invocations.py`); restore then re-apply sibling hunks (`git checkout -p` /
saved per-class patch). Reverting one class must never discard a sibling's
accumulated edits.

For each `file_group = (test_file, [classes])`:

### Phase 1: Review Gate

`task(subagent_type="explore", load_skills=["review-test-refactoring"], run_in_background=False)`
— review `test_file` ONCE; request a PER-CLASS Blockers/Majors/Minors + pass/fail
verdict for `[classes]`.

- Zero Blockers => class eligible for Phase 2.
- >=1 Blocker => record in `followup_classes` (review-blocked), exclude from
  Phase 2. If all classes blocked, skip to next group.

### Phase 2: Develop Enablement

`dev = task(subagent_type="explore", load_skills=["develop-xpu-test"], run_in_background=False)`
— enable XPU for the review-passing (`eligible`) classes in `test_file`, using
`E`/`P`. Instruct: apply the per-class op_db scoping gate to EACH class
independently (only widen `DecorateInfo` entries belonging to that class's own
generic test names; leave op_db untouched for a class with no matching entries);
report edits grouped per class. Keep `dev.task_id` for Phase 3.

### Phase 2.5: Remove Stale Skips (optional)

Run only when you want to clean up method-level `@skipIfXpu`, `@skipXPU`,
`@skipXPUIf`, or inline device-type guards that may no longer be needed.
This phase uses `remove-xpu-skips` to probe each skip individually.

```python
rem = task(
    load_skills=["remove-xpu-skips"],
    run_in_background=False,
    prompt="Run remove-xpu-skips on <test_file> for classes: <eligible_classes>. "
           "Conda env: <E>. Pytorch folder: <P>. For each skip found, check issue "
           "state (if applicable), try removal, run pytest on XPU, and keep or revert. "
           "Return per-skip results."
)
```

Each skip is handled one at a time:
- P1-P4 (`@skipIfXpu`, `@skipXPU`, `@skipXPUIf`): issue-gated — only probed
  when the referenced issue is CLOSED (verified via `gh issue view`).
- P5-P6 (inline guards, `@unittest.skipIf(not TEST_CUDA, ...)`): always probed
  — try widening to include XPU, test, keep/revert.

Removals that pass verification stay in the working tree (stacked on Phase 2's
edits). Removals that fail are reverted individually without discarding sibling
changes. Logs go to `agent_space/remove_xpu_skips/`.

After Phase 2.5 is complete, proceed to Phase 3 with the accumulated edits
(Phase 2 enablement + any kept skip removals).

### Phase 3: Verify Enablement (reuse session)

`ver = task(task_id=dev.task_id, load_skills=["verify-xpu-test"], run_in_background=False)`
— verify all `eligible` classes in ONE pytest run (`-k "ClassA or ClassB"`),
using `E`/`P`. Require a PER-CLASS verdict (`verified` / `out-of-scope changes` /
`needs revert` / `not effective`) with pass/skip/xfail counts and any xpass.

Per class: `verified` -> Phase 4 eval. `out-of-scope changes` / `needs revert` /
`not effective` -> surgically revert that class's hunks, record in
`followup_classes`.

### Phase 4: Analyze (only classes with failures)

For `verified` all-pass classes, adopt verdict `passed` from verify counts (NO
spawn). Only for classes with failures/xpass:

`task(task_id=ver.task_id, load_skills=["analyze-ut-failures"], run_in_background=False)`
— analyze the failing classes (`test_file`, failing classes, `test_cases`,
`E`/`P`); return JSON with per-class verdict + per-test-case failure groups
(including root-cause classification).

Per class: `passed` -> `git add` its files, append to `passed_classes`.
`has-failures` -> **keep edits in the tree**, move to `pending_classes` (to be
resolved in Phase 5B). Do NOT revert yet — the edits may be kept with
`@skipIfXpu` if the failures are known backend gaps.

After all groups processed, go to Phase 5 ONCE.

## Phase 5: Combined Submission (once)

Do 5B first so issue links can go in the PR body. If `passed_classes` is empty,
open NO PR; report per-class outcomes + any 5B issue links.

### Phase 5B: Failure Follow-up (`pending_classes` + `followup_classes`)

For each class in `pending_classes` (edits still in the tree, has failures):

1. **Classify failures**: from `analyze-ut-failures` output, determine if each
   failing test case is a genuine backend gap or a test-code bug.
   - **Test-code bug**: revert the class's hunks, move to `followup_classes`.
     No issue filed (the bug is in test code, not the backend).
   - **Backend gap**: proceed to known-issue check.

2. **check-known-issue**: for each backend-gap failure, run
   `task(load_skills=["check-known-issue"], run_in_background=False)` —
   check `test_file`/`class`/`test_name`/`error`/device=xpu; collect existing
   issue URLs.

3. **Decision per failing test case**:
   - **Known issue EXISTS** → add `@skipIfXpu(msg="See <issue_url>")` on that
     individual test method (not class-wide). Import `skipIfXpu` from
     `torch.testing._internal.common_utils` if not already present.
   - **No known issue** → `task(load_skills=["create-xpu-issue"], run_in_background=False)`
     — file a tracking issue (signature, tests, root_cause, pr=<pending>,
     enablement Context). Do NOT add a `@skipIfXpu` (no issue URL to reference).

4. **Class outcome**:
   - At least one `@skipIfXpu` added → moves to `passed_classes` (enabled
     with per-method skips; passing siblings still run on XPU).
   - No skips added (all failures were new issues with no prior tracker) →
     revert the class's hunks, move to `followup_classes`.

5. **Gather all issue URLs** (both known and newly created) into
   `followup_issue_urls` for the PR body / cross-referencing.

Review-blocked classes (no runnable failure signature) need no issue unless
the user asks.

### Phase 5A: Submit ONE Combined PR (>=1 passing class)

The tree holds accumulated verified edits for every `passed_classes` entry.
Call `submit-xpu-test-pr` ONCE:

`task(subagent_type="explore", load_skills=["submit-xpu-test-pr"], run_in_background=False)`
— submit ONE combined draft PR for `passed_classes`, using `P`, rebased onto
`pytorch/pytorch` viable/strict (or main). PR body must list every enabled class
+ its `test_file` + per-class verification result, and include
`followup_issue_urls`. The skill stages only intended files, rebases, and opens
one draft PR against `pytorch/pytorch`; its diff-scope checks cover the union of
passing classes' files. Never submit per-failing-class.

## Critical Error Handling (HARD STOP)

Hard-stop the whole run only on batch-wide blockers:
- Provider/subagent unavailable (model/rate-limit/quota).
- Broken env (`torch` import fails, `torch.xpu.is_available()==False`, invalid
  env) that persists AFTER Phase 0.5 provisioning.
- Missing required inputs (`test_targets` empty; a target missing
  `test_file`/`test_class`; `conda_env`/`pytorch_folder` absent).
- `pytorch_folder`/`conda_env` still missing after `setup_env.sh` ran.
- Unrecoverable command failure blocking the next phase.

A missing env/folder is NOT itself fatal — provision first (Phase 0.5). A single
class failing review/verify/analysis is NOT fatal — per-class isolation handles
it. On a fatal blocker: append `[FATAL] <phase>: <error> — halting session` to
`session_log.txt`, save details to `logs/<phase>_fatal.log`, and end.

## Required Logs and Artifacts

`session_log.txt`, `logs/background_status.log`, and under
`enable_xpu_orchestrator/`: `phase0_5_setup_env.json` (if provisioning ran),
`phase1_review.json`, `phase2_develop.json`, `phase3_verify.json`,
`phase4_analyze.json`, `phase5_followup.json` (on failures),
`phase5_submit_pr.json`. Key per-phase JSON by file-group slug with a per-class
`results` array.

## Output Contract

```json
{
  "status": "passed|partial|issue-follow-up|failed-hard-stop",
  "pytorch_folder": "...",
  "pr_url": "... or null",
  "passed_targets": ["<test_file>::<test_class>", "..."],
  "followup_targets": ["<test_file>::<test_class>", "..."],
  "known_issue_urls": ["..."],
  "created_issue_urls": ["..."],
  "per_target": [
    {"test_file": "...", "test_class": "...",
     "review": "pass|fail",
     "verify": "verified|out-of-scope|needs-revert|not-effective|n/a",
     "analysis_verdict": "passed|has-failures|n/a",
     "outcome": "enabled|review-blocked|verify-failed|has-failures"}
  ],
  "logs": ["agent_space/session_log.txt", "agent_space/enable_xpu_orchestrator/phase1_review.json"]
}
```

Status: `passed` all enabled + in PR; `partial` some enabled (PR opened), rest
follow-up; `issue-follow-up` none enabled, only issues (no PR);
`failed-hard-stop` batch-wide critical error.

## Constraints

1. Group by file; run Phases 1-4 once per `test_file` group, not per class.
2. Reuse subagent sessions within a group via `task_id` (file read once).
3. Analyze only on failure; adopt verify verdict for all-pass classes.
4. Review gate is per class; a fail isolates that class, never hard-stops the batch.
5. Log every phase/subagent in `agent_space` (per group, per-class results).
6. Never continue past a batch-wide critical error.
7. Sequential dependent phases use explicit `load_skills=[...]`, `run_in_background=False`.
8. Never skip verification for an enabled class.
9. PR submission uses `submit-xpu-test-pr`, called ONCE for the whole batch.
10. Surgical revert only — never discard a passing class's edits when reverting another.

## See Also

`develop-xpu-test`, `remove-xpu-skips`, `verify-xpu-test`,
`submit-xpu-test-pr`, `analyze-ut-failures`, `check-known-issue`,
`create-xpu-issue`.
