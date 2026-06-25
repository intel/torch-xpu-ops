---
name: xpu-ci-health-check
description: >-
  Check PyTorch ciflow/xpu (xpu.yml) on the main branch, collect the failing
  XPU test cases from the most recent completed run(s), analyze the ROOT CAUSE
  of each failure with AI, and produce a list with a
  prefilled "disable issue" link per case. USE WHEN the user asks to check XPU
  CI health, find failing XPU tests, or generate XPU
  disable-issue drafts.
---

# XPU CI Health Check Skill

This skill automates XPU CI health checks with the following workflow:

1. **Collect** failure evidence automatically by running the bundled script.
2. **Analyze** the root cause of each case yourself (AI), using the traceback
   evidence plus the suspect commit/PR — never copy the raw exception as the
   conclusion.
3. **Render the result** generate a list with one row per failing case.

> Root cause analysis is YOUR job, not the script's. The script only gathers evidence
> (case id, traceback excerpt, commit sha, disable link). You must reason about
> *why* it failed and *which change* likely caused it.

## Prerequisites

- **GitHub token**: Optional. The script uses the `gh` CLI if `--token` is not provided; ensure `gh auth login` is configured.

## Step 1 — Collect evidence (always run this first)

Run the bundled collection script from the PyTorch repository root. The script
is located at `<path/to/skills>/xpu-ci-health-check/scripts/collect_failures.py`
(included in this skill).

```bash
# Run from PyTorch repository root
cd /path/to/pytorch  # or your PyTorch clone location
python </path/to/skills>/xpu-ci-health-check/scripts/collect_failures.py --run-limit 1
```


### Flags

- `--run-limit N` — Inspect the last N completed `main` runs (default: `1`).
- `--token` — GitHub API token. If omitted, the script falls back to `gh` CLI auth.

The script prints a JSON evidence bundle to stdout:

```jsonc
{
  "runs": [{ "run_number": 9275, "head_sha": "5ef6fae…", "html_url": "…" }],
  "cases": [
    {
      "case_id": "test/inductor/test_cutlass_backend.py::TestCutlassBackend::test_xxx",
      "commit_sha": "5ef6fae…",
      "commit_short": "5ef6fae",
      "hud_url": "https://hud.pytorch.org/pytorch/pytorch/commit/5ef6fae…",
      "commit_url": "https://github.com/pytorch/pytorch/commit/5ef6fae…",
      "job_name": "linux-jammy-xpu-… / test (default, …)",
      "failure_line": "… FAILED …",
      "error_excerpt": "…traceback / exception text…",
      "issue_title": "DISABLED test_xxx (__main__.TestCutlassBackend)",
      "issue_body": "Platforms: xpu\n\nThis test was disabled because…\n\ncc …",
      "issue_labels": ["module: xpu", "triaged"],
      "issue_url": "https://github.com/pytorch/pytorch/issues/new?title=DISABLED%20…&body=…&labels=module%3A%20xpu,triaged"
    }
  ]
}
```

> **Disable issue template is frozen in the script.** The `issue_title`,
> `issue_body`, `issue_labels`, and `issue_url` fields are produced by
> `build_disable_issue()` inside `scripts/collect_failures.py`, aligned with the
> reference issue template https://github.com/pytorch/pytorch/issues/185907
> (Platforms line + "recent examples" section + cc mention + labels `module: xpu`,
> `triaged`). **When creating an issue, use these exact fields verbatim — never
> rewrite the title, body, or labels.**


## Step 2 — Analyze the root cause (AI analysis required)

If the number of failed cases > 10, skip this step and let the root case field be empty.
For each case in `cases`, use subagent to do:

1. Fetch latest origin main and git checkout to the `commit_sha` locally.
2. **Identify the real failure**: Read `error_excerpt` and identify the failing frame
   and exception type (e.g., `InductorError: NotImplementedError: `), not just the surface `FAILED` line.

3. Root cause:
    - Determine regression status of the case:
       - **Newly added or updated case** → `No`
       - **Existing old case** (appeared before) → `Yes`
       - **Insufficient evidence** → `Unknown` (explain why)
    - Find the code change that cause the case failure and also the guilty commit/PR.
    - Analysis the root cause.
    - **Write root cause** in a strict 3-point structure (do not omit any point):
        - **Introduced by which PR**: Identify the most likely PR/commit that introduced the failure. If uncertain, state the top suspect and what evidence is missing.
        - **Root cause of the failure**: Explain the concrete failing mechanism on XPU using evidence from traceback/logs.
        - **Is this fail only on XPU**: try to analysis if cuda will fails, if no evidence is available, state that explicitly.
    - Use available history (recent runs, torch-ci failure history, prior reports).

## Step 3 — Render output

Produce a list these details, one row per case:

### List the details for each case as follows:

- **Commit**: `` `<commit_short>` `` linked to HUD (e.g., `[5ef6fae](https://hud.pytorch.org/...)`).
- **Case**: The `case_id` from the evidence bundle.
- **Is regression**: `Yes` / `No` / `Unknown` (see Step 2.4; `No` = newly appeared).
- **Root cause**: Both required points from Step 2 (introduced PR + why XPU fails mechanism).
- **Disable link**: `[Create disable issue](<issue_url>)`.
  - **IMPORTANT**: Use `issue_url` **exactly as produced by the script** — do not truncate.
  - The full URL carries prefilled title, body (Platforms + recent examples + cc), and
    labels (`module: xpu`, `triaged`).
  - A URL with only `?title=...` is **invalid** and will not match the template.

After the list, add a one-line health summary:
- Whether `main` is green or red for `ciflow/xpu`.
- The run number(s) and count of failing cases inspected.



## Important Notes

- **Zero failures**: If the script returns no cases, report that `main` is **green** for
  the inspected run(s). Do not fabricate failures.
- **No auto-creation**: Disable links are drafts only. A human must review and approve
  before creating issues.
- **Template reference**: The frozen template follows
  https://github.com/pytorch/pytorch/issues/185907 exactly (Platforms + recent examples +
  cc + labels).
