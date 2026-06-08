---
name: xpu-alignment
description: Scan pytorch/pytorch issues, PRs, and commits (CUDA, ROCm, or any backend) for bugs that may also affect XPU, adapt upstream reproducers for XPU, and validate on local XPU nightly. Use when aligning upstream backend fixes for XPU parity.
---

# XPU Alignment

Scan `pytorch/pytorch` for issues, PRs, and bug-fix commits across any backend (CUDA, ROCm, CPU, MPS) that may also affect XPU due to shared code paths. Adapt upstream reproducers for XPU, validate locally, route confirmed bugs, produce an auditable report.

## Inputs

The caller (user or orchestrator) provides these. This skill is self-contained; there are no external driver scripts.

1. **Scan window**: a start/end date pair (`YYYY-MM-DD` to `YYYY-MM-DD`). If none is given, default to "yesterday" (a single-day window ending today).
2. **Run directory**: a working directory for all artifacts, conventionally `runs/<date-or-range>/`. Create these subdirectories under it:
   - `artifacts/` (with `artifacts/details/`)
   - `scripts/`
   - `reports/`
   All paths in this skill are relative to the run directory unless noted.
3. **Workspace-local XPU interpreter**: a `.venv/bin/python` or `.conda*/bin/python` inside the workspace running the latest XPU nightly. Never use an interpreter outside the workspace. Preflight (Step 0) ensures the nightly is present and fresh:
   - If no workspace venv exists, create one in-workspace (`python -m venv .venv`).
   - Check the installed torch build date / version. If torch is missing or the nightly is stale (older than the latest available), reinstall:
     `python -m pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/xpu`
   - This replaces what `common.sh` used to do; do it inline with the workspace interpreter.
4. **GitHub access**: via the GitHub MCP server, or the `gh` CLI as a fallback. Never hardcode tokens; rely on the ambient `gh auth` / MCP credentials.

## Rules

1. Workspace-local XPU interpreter: look for `.venv/bin/python` or `.conda*/bin/python` in the workspace. If missing, create a venv in-workspace; never use interpreters outside this workspace.
2. Preflight checks nightly version; reinstall if stale or missing to ensure the latest nightly.
3. Zero pending actionable rows = done. Otherwise write `## Progress checkpoint`.
4. **Batched pipeline**: batch deep-filter -> batch write repros -> serial execution -> batch route -> batch report. Ledger enables resume from any interruption.
5. Never hardcode GitHub tokens.

## Key Tables

### Bucket Vocabulary

| Bucket | Meaning |
|--------|---------|
| `confirmed` | same bug reproduces on XPU |
| `related-failure` | XPU fails differently on same scenario |
| `not-reproduced` | upstream failure does not reproduce |
| `blocked-env` | missing dependency or distributed setup |
| `blocked-platform` | XPU lacks required path |
| `blocked-fetch` | cannot fetch issue details |
| `blocked-commit-context` | commit lacks enough context for repro |
| `blocked-script-error` | repro failed before reaching oracle |
| `needs-performance-harness` | perf-only, needs benchmark |
| `not-applicable` | rejected before validation |

### Confirmation Criteria

Sufficient: crash, segfault, assertion failure, hang, wrong numerical result, wrong shape/stride/dtype, off-by-one beyond atol=1e-4.

Not sufficient: tiny float noise within tolerance, documented unsupported behavior, invalid repro setup.

### CUDA -> XPU Adaptation

When adapting an upstream reproducer, map device APIs:
- `cuda` -> `xpu` for `.to("cuda")`, `device="cuda"`, `torch.cuda.*` -> `torch.xpu.*`.
- `torch.cuda.synchronize()` -> `torch.xpu.synchronize()`.
- `@onlyCUDA` / `requires_cuda` test markers -> run directly on XPU.
- Drop CUDA-only kwargs (e.g., `device_type="cuda"` in autocast) and substitute `"xpu"`.
- Keep the numerical scenario, shapes, dtypes, and oracle identical; only the device changes.

## Workflow

### Step 0: Preflight

Verify XPU torch import and `torch.xpu.is_available()`, verify GitHub access, create output directories (`artifacts/details`, `reports`, `scripts`), and save `collect_env` output to `artifacts/collect_env.txt`.

Run the freshness check from the Inputs section first: ensure the workspace venv exists and holds a current XPU nightly, reinstalling with the `--pre torch --index-url .../nightly/xpu` command if stale or missing.

If any preflight check fails:
- `torch.xpu.is_available()` returns `False` -> ensure Intel oneAPI runtime is loaded (`source /opt/intel/oneapi/setvars.sh`) and the XPU driver is installed.
- `import torch` fails -> reinstall the XPU nightly wheel into the workspace venv.
- GitHub access fails -> verify auth (`gh auth status`) or MCP credentials.
- Output directory creation fails -> check filesystem permissions.

### Step 1: Collect candidates

Search `pytorch/pytorch` using GitHub MCP (fall back to `gh` CLI). Use the caller-specified time window. Paginate with `per_page=100`; split date ranges if hitting the 1000-result cap.

**Principle**: cast a wide net -- prefer over-collecting then filtering, rather than missing candidates at the search stage.

**Source types:**

1. **Issues** -- collect all issues in the specified time window, across all states (open + closed); do not pre-filter by labels or keywords at this stage.
2. **PRs** -- collect all PRs in the specified time window, across all states (open, merged, closed); do not pre-filter by labels or keywords at this stage.
3. **Commits** -- collect all commits in the specified time window; do not require merged-PR linkage or keyword filtering at this stage.

Save to `artifacts/raw_candidates.json` (deduplicated by id, metadata only -- no bodies/diffs yet). Each entry has `kind: "issue"|"pr"|"commit"`.

### Step 1.5: Title triage

Initialize `artifacts/candidate_ledger.jsonl` from raw candidates. Reject by title/message alone when it clearly indicates non-bug or platform-exclusive scope:
- Titles that start with `DISABLED test_` or only disable/enable CI tests (covered by UT and issue-handler scenarios)
- Platform prefixes: `[Windows]`, `[MPS]`, `[Build]`, `[Dependabot]`, `[RFC]`
- Docs/CI/infra/release-only keywords
- Obvious duplicates of already-processed candidates
- For commits: pure refactor/style/typo/doc commits (no functional change)

Each ledger row tracks at least: `id`, `kind`, `title`, `url`, `title_status` (pass/reject), `deep_status` (pending/pass/reject), `local_status` (pending/done), `local_bucket`.

**Principle**: reject only when you're confident the title rules out XPU relevance. When in doubt, pass.

### Step 2: Batched pipeline

#### 2a. Batch deep-filter

Fetch all passed candidates' details -> save to `artifacts/details/<id>.json`:
- **Issues/PRs**: fetch body, linked PRs/commits, test names
- **Commits**: fetch commit message + diff (`gh api repos/pytorch/pytorch/commits/<sha>` or `git show`). Save the diff summary and affected files.

For each, decide reject or pass-to-repro (update `deep_status`).

**Rejection principle**: reject only when the content confirms the bug is in platform-exclusive code with no XPU equivalent. Examples:
- Metal/MPS shaders, HIP driver-level, Windows linker
- CUDA allocator internals (CUDACachingAllocator), hardware-specific (device capability, RTX model)
- vLLM/distributed infrastructure with no standalone PyTorch repro available
- Commits that only touch CUDA-specific codegen templates (`torch/inductor/codegen/cuda/`) with no shared path

**Pass principle**: if the bug touches shared code (Inductor, Dynamo, autograd, dispatcher, ATen, Triton, runtime) or you're unsure, pass it through. Attempt a reproducer -- that's cheaper than a false negative.

**Commit-specific**: if the diff is too small or lacks context to construct a meaningful reproducer (e.g., one-line typo fix in a comment), set `deep_status: "reject"` with reason "insufficient commit context" and move on.

#### 2b. Batch write reproducers

For all `pass-to-repro` candidates, write `scripts/repro_<id>.py` in one pass. Each repro must:
1. Print `torch.__version__` and `torch.xpu.is_available()`
2. Verify the op ran on XPU (not CPU fallback)
3. Print `RESULT: <bucket>` as the final meaningful line

**Repro source by kind:**
- **Issues**: extract the reproducer from the issue body
- **PRs**: extract from the PR description, or from the test added/modified in the PR's commits
- **Commits**: extract the regression test added in the commit (look for new `test_*` functions in `test/` files in the diff). If no test was added, construct a minimal repro from the fix diff -- the "before" state is the bug.

Prefer extracting existing upstream code and adapting it (CUDA -> XPU mapping above). Only write from scratch when no upstream repro exists.

#### 2c. Serial execution

Run each repro script sequentially (for crash/timeout isolation) with the workspace XPU interpreter and a timeout (~120s). Capture stdout/stderr to `artifacts/output_<id>.log`. Parse each `RESULT:` line -> update ledger (`local_status: "done"`, `local_bucket`).

If tensor `.device` is `cpu`, mark `blocked-script-error` (CPU fallback, not valid XPU test).

#### 2d. Batch route

For confirmed/related-failure bugs:
- Shared code (Inductor, autograd, dispatcher, ATen, Triton, runtime) -> `pytorch/pytorch`
- XPU kernel in `aten/src/ATen/native/xpu/` -> `pytorch/pytorch`
- XPU kernel not upstream -> `intel/torch-xpu-ops`
- Bug reveals XPU backend gap (different error, missing feature) -> `intel/torch-xpu-ops`
- CPU-only crashes that affect all backends -> `pytorch/pytorch`
- When in doubt -> `pytorch/pytorch`

#### 2e. Batch write report

Write `reports/full_scan.md` yourself (no external renderer). Include every candidate whose `local_status == done`; exclude rows rejected at deep-filter (`deep_status == reject`).

Each entry must preserve the numbered format and include at least:
- candidate id, title, and kind
- evidence URL
- reproducer script path and output log path
- an exact `Local XPU result: `<bucket>`` line
- route suggestion for `confirmed` and `related-failure`

Requirements:
- The report must remain auditable: keep the numbered entry format and include an exact `Local XPU result: `<bucket>`` line for every tested candidate.
- For confirmed bugs, include enough local evidence and context for issue filing without reopening raw logs.
- Use upstream issue/PR content or commit context to describe the scenario; do not reduce entries to title-only summaries.
- Blocked and not-reproduced entries may be shorter, but must still include the repro path, output log path, and decisive local outcome.

#### 2f. Write issue drafts

Write `reports/issue_drafts.md` yourself for all `confirmed` and `related-failure` candidates, using this exact body structure:

```
## Issue 1

**Suggested title:** [cuda_xpu_alignment] <original bug title>
**Suggested labels:** xpu-alignment, <upstream-issue|upstream-pr>, <confirmed|related-failure>

**Upstream source:** <upstream URL> (upstream-issue | upstream-pr)
**Scan date:** <YYYY-MM-DD> to <YYYY-MM-DD>
**Local XPU result:** confirmed on torch <version>, <GPU model>

---

### 🐛 Describe the bug

<clear description of the bug with root-cause analysis where available>

```python
# minimal self-contained XPU-adapted reproducer (copy-pasteable)
```

```
<actual output / error message>
```

---

### Versions

```
<full contents of artifacts/collect_env.txt>
```

---

## Issue 2
...
```

Label selection rules:
- Always include `xpu-alignment`
- Add `upstream-issue` if sourced from a GitHub issue, `upstream-pr` if sourced from a GitHub PR or commit
- Add `confirmed` if `local_bucket == "confirmed"`, `related-failure` if `local_bucket == "related-failure"`

If no confirmed or related-failure candidates exist, write `reports/issue_drafts.md` with a single line: `No confirmed or related-failure candidates in this scan.`

### Step 3: Audit

Audit the report and ledger yourself by reading them (no external audit script). The scan is auditable and complete only when ALL of these hold:

1. **Zero pending actionable rows** in `artifacts/candidate_ledger.jsonl`: there is no row with `title_status == pass` AND `deep_status != reject` AND `local_status == pending`. Any such row means work remains -- do not finalize.
2. **Every numbered entry** in `reports/full_scan.md` has an exact `Local XPU result: `<bucket>`` line where `<bucket>` is one of the Bucket Vocabulary values.
3. **Report scope matches the ledger**: the report counts only entries with `local_status == done`, and every deep-rejected row (`deep_status == reject`) is excluded from the report.

If any check fails, write `## Progress checkpoint` describing the pending rows or mismatches and continue; do not write the final summary.

Write `## Final Summary` only when all three audit checks pass. Include filter stats (collected / title-rejected / deep-rejected / passed-to-repro), validation stats (per-bucket counts), and routing stats.

## Guardrails

- Do not file issues from this skill.
- `confirmed` requires a local run reproducing the issue.
- Never hardcode GitHub tokens.

## Outputs

Artifacts produced under the run directory:
- `artifacts/raw_candidates.json` -- deduplicated candidate metadata
- `artifacts/candidate_ledger.jsonl` -- per-candidate status ledger (resume point)
- `artifacts/details/<id>.json` -- fetched body/diff per passed candidate
- `scripts/repro_<id>.py` -- XPU-adapted reproducer per pass-to-repro candidate
- `artifacts/output_<id>.log` -- captured stdout/stderr per executed repro
- `artifacts/collect_env.txt` -- `collect_env` output for issue Versions section
- `reports/full_scan.md` -- auditable report of all tested candidates
- `reports/issue_drafts.md` -- issue-ready drafts for confirmed/related-failure candidates
