---
name: fix-xpu-ci-checker
description: Diagnose and fix `linux-ut ... / summary` (CI checker) failures on intel/torch-xpu-ops PRs. Use when a PR's UT summary check is red, when "Known issues" filter mismatches surface, when failures must be split between PR-touched code (fix in PR) and unrelated cases (dynamic-skip via tracking issues), and when deciding what kind of CI rerun to trigger. Covers fetching CI artifacts via `gh`, deep semantic analysis of `Cases:` filter format, updating tracking issues only with user approval, and choosing between a failed-jobs rerun versus a full rerun.
---

# Fix XPU CI Checker (`linux-ut / summary`)

## When to use

Use this skill when a PR on `intel/torch-xpu-ops` has a red `linux-ut (...) / summary` (a.k.a. "summary check", "ut_result_check") job, or any other UT-checker job whose pass/fail is decided by `.github/scripts/ut_result_check.sh` filtering `failures_*.log` against a `Known_issue.log` produced from skip tracking issues.

Two example PRs this skill was distilled from (do **not** treat them as templates — the analysis must always be re-done):
- intel/torch-xpu-ops#3383 — checker red because a newly-ported test file had a collection-time `ModuleNotFoundError`.
- intel/torch-xpu-ops#3475 — checker red because two tracked issues (`#3473`, `#2444`) had `Cases:` lines whose format did not full-line-match the runner's emitted strings, and one case was a flaky pre-existing failure.

## Required mental model

The summary checker is a **fixed-string, full-line filter**:

```bash
# .github/scripts/ut_result_check.sh
grep -vFxf "$known_file" "$ut_file" > "$output_file"
```

- `$ut_file` — list of failures the test runner emitted, one per line.
- `$known_file` — `Known_issue.log`, assembled by parsing the `Cases:` section of every `skipped`-labeled tracking issue on `intel/torch-xpu-ops`.
- `-Fxf` means **fixed-string, full-line**: a tracking issue only suppresses a failure when the `Cases:` line is byte-for-byte equal to the runner-emitted line.

For the `op_ut` suite the runner emits:

```
op_ut,<full.dotted.module.path>.<TestClassName>,<test_name>
```

Note the **two commas**, with the dot between module path and class name. A common authoring mistake is putting the comma between module and class (`...<module>,<TestClass>.<test>`); that line will never match and the failure will never be filtered.

`Cases:` parsing in `mark_passed_issue` (same script, line ~323) is on the `Cases:` line itself, so the leading `op_ut,` prefix and the exact case string both matter.

## Tools

| Tool | Purpose |
|------|---------|
| **Bash** | `gh run view/download`, `gh issue view/edit/comment`, `gh api ... rerun-failed-jobs`, log parsing |
| **Read / Grep / Glob** | Inspect downloaded `failures_*.log`, `Known_issue.log`, `op_ut*.log`, ported test files |
| **Task (explore)** | Search the PR diff and tracking-issue corpus when the failing case's ownership is non-obvious |
| **Edit** | Apply a code fix to a file the PR already touches (only when ownership is clear) |
| **Question** | **Required** before editing any tracking issue body or any file the PR did not author |
| **WebFetch** | Read GitHub issues / PR pages when `gh` output is not enough (rare) |

## Preconditions

1. `gh` CLI authenticated against `intel/torch-xpu-ops` with `repo` scope (`gh auth status`).
2. Local checkout of `intel/torch-xpu-ops` (typically at `third_party/torch-xpu-ops/`) so `ut_result_check.sh` and the PR's source files can be read.
3. The PR number, the failing workflow run id, and the failing job name are known. If only a PR URL is given, derive these via `gh pr checks <PR> --repo intel/torch-xpu-ops` and `gh run list --repo intel/torch-xpu-ops --branch <head>`.
4. Network access to GitHub API (to download artifacts and read tracking issues).
5. **Do not** require local pytest execution — diagnosis here is log-driven. Test execution is a separate skill (`submit_ut_issues`, `port-to-xpu-ops`).

## Constraints

These constraints encode the user's stated preferences and must not be relaxed without explicit user approval.

| Constraint | Rule |
|-----------|------|
| **PR-scoped code edits** | Only edit a file if the failing case lives in code the PR added or modified. Never silently patch unrelated files in the PR branch. |
| **No skip-condition weakening** | Do not add broad `unittest.skip`, weakened asserts, or remove coverage. Dynamic skip happens via tracking issue + `Cases:`, not by editing the test. |
| **Tracking-issue edits need approval** | Any change to an existing tracking issue body (e.g. adding a `Cases:` line, fixing comma format) MUST be confirmed via `Question` first. Comments on tracking issues that only document what was already done do not require approval, but should still be terse. |
| **Preserve existing `Cases:` lines** | When adding a new line, append; do not rewrite or reorder lines that already match other suites/paths. |
| **No new tracking issue without justification** | Reuse an existing `skipped`-labeled issue if the failure signature, file, and class match. Only file a new issue when no existing one applies — and then follow `submit_ut_issues` (Context section is mandatory when filed during a PR). |
| **Do not push force / amend pushed commits** | If a PR-scoped fix has already been pushed, follow up with a new commit, not `--amend --force`. |
| **Rerun-policy follows code state** | See [Phase 4](#phase-4--rerun-policy). |
| **Context cross-link** | When opening or editing a tracking issue during a PR, ensure the issue carries a Context section linking the PR (per `submit_ut_issues`). |
| **Authored with Claude trailer** | All commits authored from this skill must include the trailer the branch already uses (typically `Authored with Claude.`). |

## Workflow

### Phase 1 — Get the CI failure

1. Identify the failing job from the PR's checks page:
   ```bash
   gh pr checks <PR> --repo intel/torch-xpu-ops | grep -iE "summary|ut_result_check|fail"
   gh run list --repo intel/torch-xpu-ops --branch <head_branch> --limit 5
   ```
2. Locate the run id for the suite that fed into the failing checker (e.g. `linux-ut (op_ut)`), not the checker job itself — the artifacts come from the suite run:
   ```bash
   gh api repos/intel/torch-xpu-ops/actions/runs/<run_id>/jobs \
     --jq '.jobs[] | {name, status, conclusion}'
   ```
3. Download the suite's UT artifact directory (typically named like `Inductor-XPU-UT-Data-<PR>-<suite>-<run_id>-1`):
   ```bash
   mkdir -p /tmp/pr<PR>_logs && cd /tmp/pr<PR>_logs
   gh run download <run_id> --repo intel/torch-xpu-ops
   ```
4. Inside the artifact, the relevant files are:
   - `<suite>/Known_issue.log` — the assembled allow-list from tracking issues.
   - `<suite>/failures_<suite>.log` — runner-emitted failures (left side of `grep -vFxf`).
   - `<suite>/failures_<suite>_filtered.log` — leftover after filtering. **Non-empty here is what makes the checker red.**
   - `<suite>/op_ut*.log` (or equivalent) — full pytest stdout/stderr for traceback extraction.
   - `<suite>/ut_result_check.sh` — exact filter logic for this run (re-read it; do not assume).

### Phase 2 — Classify each unfiltered failure

For every line in `failures_<suite>_filtered.log`, decide ownership before doing anything:

```
Failure line:
  op_ut,third_party.torch-xpu-ops.test.xpu.<module>.<Class>,<test>
            │
            ├── Was the test/file ADDED or MODIFIED by this PR?
            │    └── git diff <base>...HEAD -- <path-of-the-test-file>
            │
   ┌────────┴─────────────────┐
   │                          │
PR-OWNED                  NOT PR-OWNED
   │                          │
   ▼                          ▼
Phase 3a                   Phase 3b
(fix in PR)            (dynamic-skip via issue)
```

Key sub-checks (no shortcuts, no pure pattern-matching):
- Read the failing case's Python source. Confirm whether the failure is at *collection time* (e.g. `ModuleNotFoundError`) or *execution time*.
- Run `git log -p <base>...HEAD -- <test-file>` to see whether the PR introduced/enabled this test.
- For execution-time failures, read the matching pytest traceback in the artifact log (`op_ut*.log`, search for `FAILED <module>::<Class>::<test>` and read upward).
- For "worker crashed" / `node down: Not properly terminated`, the per-test traceback may be absent — record the worker-crash signature itself and treat the case as a flaky/resource failure (still subject to Phase 2 ownership rules).

### Phase 3a — PR-owned failure: fix in the PR

Only proceed when the PR clearly owns the file. Then:

1. Determine the minimal fix consistent with the surrounding repo conventions. Examples encountered:
   - Collection-time `ModuleNotFoundError: No module named 'common_utils'` in a newly ported `test_*_xpu.py` → add the matching `sys.path.append("../../../../test/<dir>")` shim used by sibling tests in the same directory (e.g. `test_vmap_xpu.py`).
   - A test newly enabled by the PR fails with a real bug → either fix the test code or, if the bug is upstream, follow `submit_ut_issues` to file a tracking issue *and* apply the dynamic skip in Phase 3b instead.
2. Apply the edit with `Edit` (never `Write` over the whole file).
3. Commit on the PR's local branch with the project's trailer. Do not amend already-pushed commits.
4. Push fast-forward to the PR's head ref (typically `<fork>:<branch>`).
5. Trigger the appropriate rerun (Phase 4). After a code change, this is a **full rerun**.

### Phase 3b — Not-PR-owned failure: dynamic skip via tracking issue

The CI checker treats every failure as fatal unless suppressed via `Known_issue.log`. To suppress, the failure's exact line must appear under `Cases:` in some `skipped`-labeled tracking issue.

1. **Find a candidate existing issue first.** Search by signature (test name, module, class, error excerpt):
   ```bash
   gh issue list --repo intel/torch-xpu-ops --label skipped --search "<test_name>" --state all
   gh issue list --repo intel/torch-xpu-ops --label skipped --search "<error excerpt>" --state all
   ```
   Read each candidate's body. An issue is a match only if:
   - Same test name AND
   - Same root cause / error signature OR same upstream PyTorch DISABLED tracking AND
   - The PR's case path is plausibly covered (or can be added as another `Cases:` entry).
2. **Verify the `Cases:` line format.** Re-read `ut_result_check.sh` for this run and confirm the exact format the suite emits. For `op_ut`, the canonical line is:
   ```
   op_ut,<full.dotted.module.path>.<TestClassName>,<test_name>
   ```
   If the existing issue has a malformed line (e.g. comma before the class name), that is itself the bug — it explains why filtering missed previously.
3. **Ask the user before editing the issue body.** Use `Question` with a short summary of the proposed change (which lines are added/edited, which issue, why). Even for "obvious" format fixes — the user explicitly required confirmation here.
4. **Edit the issue.** Append the new `Cases:` line; if a malformed line exists, replace only that line. Do not reorder or remove lines for other suites/paths.
   ```bash
   gh issue view <N> --repo intel/torch-xpu-ops --json body -q .body > /tmp/issue<N>_body.txt
   # ... edit /tmp/issue<N>_body.txt ...
   gh issue edit <N> --repo intel/torch-xpu-ops --body-file /tmp/issue<N>_body.txt
   ```
5. **Comment on the issue** documenting:
   - Which line(s) were added/changed and why (full-line `grep -vFxf` semantics).
   - The PR and CI run that surfaced it.
   - The **observed CI error signature on this run**, especially when it differs from the signatures already documented (e.g. pytest-xdist `node down: Not properly terminated` vs. the previously documented `UR_RESULT_ERROR_OUT_OF_RESOURCES`). New signatures must be appended even if the same test is already tracked.
6. **If no existing issue applies**, hand off to the `submit_ut_issues` skill to file a new one. The new issue MUST have a Context section linking the PR (mandatory in that skill when filed during a PR). Then apply step 4–5 above on the newly created issue.

### Phase 4 — Rerun policy

The rerun choice is dictated by whether code changed:

| Situation | Rerun |
|-----------|-------|
| **Phase 3a applied (code change in PR)** | **Full rerun** of the suite, because the new commit must be exercised end-to-end and prior caches/results no longer reflect HEAD. |
| **Phase 3b only (no code change, only tracking-issue body edit)** | **Failed-jobs rerun.** The runner output is unchanged; only the checker step needs to re-evaluate `Known_issue.log` against the same `failures_*.log`. Re-running passing jobs wastes CI. |
| **Mixed (both code change and issue edit)** | Full rerun. The new commit's runner output may differ; do not rely on the prior failure list. |

Commands:

```bash
# Failed-jobs rerun (no code change)
gh api -X POST repos/intel/torch-xpu-ops/actions/runs/<run_id>/rerun-failed-jobs

# Full rerun (after a code change)
gh api -X POST repos/intel/torch-xpu-ops/actions/runs/<run_id>/rerun
# or, after pushing a new commit, a fresh workflow run is triggered automatically.
```

If the local `gh` is too old for `gh run rerun --failed`, use the `gh api` form above (this was needed in practice on the host where this skill was authored).

### Phase 5 — Verify

1. Wait for the rerun to complete:
   ```bash
   gh run view <new_run_id> --repo intel/torch-xpu-ops --json status,conclusion
   gh api repos/intel/torch-xpu-ops/actions/runs/<new_run_id>/jobs \
     --jq '.jobs[] | select(.name | test("summary|op_ut")) | {name, status, conclusion}'
   ```
2. If the summary job is still red, re-download the new artifact and re-enter Phase 2 with the new `failures_<suite>_filtered.log`. Common second-pass causes:
   - `Cases:` line still off by one component (e.g. dotted-path component count mismatch).
   - Suppression added to the wrong issue body (one with a different label set so `Known_issue.log` ignored it).
   - A new failure surfaced that was previously masked by an earlier failure aborting the worker.
3. Once green, leave a final comment on the PR (or rely on the existing comments on the tracking issues) summarizing what was done, so future readers do not re-do the analysis.

## Decision tree (one screen)

```
PR summary check is red
        │
        ▼
[Phase 1] Download suite artifact; read failures_<suite>_filtered.log
        │
        ▼
For each leftover line:
  ┌── Is the test file PR-owned (added/modified by this PR)?
  │
  ├── YES → [Phase 3a] Minimal code fix in the PR
  │            └── commit + push → [Phase 4] FULL rerun
  │
  └── NO  → [Phase 3b] Dynamic skip
               ├── Find existing skipped-labeled issue with matching signature
               │     ├── exists → ASK USER → edit body (append/repair Cases:) + comment
               │     └── none   → hand off to submit_ut_issues (Context-linked) + edit + comment
               └── [Phase 4] FAILED-JOBS rerun (no code change)
        │
        ▼
[Phase 5] Verify; if still red, recurse with fresh artifact
```

## Anti-patterns

- ❌ Pattern-matching the failure signature to "the most likely existing issue" without reading the issue body and the test source.
- ❌ Adding a broad `@unittest.skipIf` to the test file to make CI green when the case is not PR-owned. Use the tracking-issue/Cases mechanism instead.
- ❌ Rewriting the entire `Cases:` section of a tracking issue. Append or surgically replace; preserve other lines.
- ❌ Triggering a full rerun when only a tracking-issue body changed — wastes CI and loses signal on what the failed-jobs rerun would have shown.
- ❌ Editing tracking issues without `Question`-based user confirmation.
- ❌ Filing a new tracking issue without a Context section linking the PR.

## Quick reference

```bash
# 1. List checks for a PR
gh pr checks <PR> --repo intel/torch-xpu-ops

# 2. Inspect a run's jobs
gh api repos/intel/torch-xpu-ops/actions/runs/<run_id>/jobs \
  --jq '.jobs[] | {name, status, conclusion}'

# 3. Download all artifacts of a run
gh run download <run_id> --repo intel/torch-xpu-ops --dir /tmp/pr<PR>_logs

# 4. Re-read the filter rule actually used by this run (do not trust memory)
sed -n '1,80p' <artifact_dir>/<suite>/ut_result_check.sh

# 5. See the lines the checker still considers failing
cat <artifact_dir>/<suite>/failures_<suite>_filtered.log

# 6. View / edit a tracking issue
gh issue view <N> --repo intel/torch-xpu-ops --json body -q .body > /tmp/issue<N>_body.txt
gh issue edit <N> --repo intel/torch-xpu-ops --body-file /tmp/issue<N>_body.txt
gh issue comment <N> --repo intel/torch-xpu-ops --body "..."

# 7. Reruns
gh api -X POST repos/intel/torch-xpu-ops/actions/runs/<run_id>/rerun-failed-jobs   # no code change
gh api -X POST repos/intel/torch-xpu-ops/actions/runs/<run_id>/rerun               # after a code change
```

## Related skills

- `submit_ut_issues` — file new tracking issues with the required Context section when no existing issue applies.
- `port-to-xpu-ops` / `port-cuda-tests-xpu` — when the PR-owned fix is actually a porting-convention gap.
- `fix-issues-identified-by-comments` — when the failing case was flagged by a reviewer rather than CI.
- `agent-guidelines` — load at the start of any coding task that this skill triggers.
