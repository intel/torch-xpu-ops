---
name: issue-triage
description: >
  Shallow, text-only triage of a GitHub issue on pytorch or torch-xpu-ops.
  Classifies the issue and returns a report (markdown table + JSON) for
  the caller to act on. Read-only; the skill itself does not comment on
  or label the issue.
---

# Issue Triage — Shallow Classification & Handling Decision

Text-only triage. Reads the issue title, body, and labels. Does NOT read
source code, does NOT run tests, does NOT open PRs, does NOT modify the
issue in any way (no comments, no labels, no body edits).

Produces a single **report** returned to the caller, containing:

1. A markdown table with the classification fields.
2. A JSON summary with the same fields as structured data.

The caller (a bot job, an orchestrator skill, or a human) decides what to
do with the report: post it as a comment, apply labels, feed it into a
larger pipeline, or just read it.

`Handling` (agent-fixable vs needs-human) is derived from the other
signals — see "Step 5" below. Deep root-cause analysis and any override
of this decision happens later in a downstream skill (out of scope for
this skill).

## Inputs

- A GitHub issue: URL, number, or raw title+body+labels. If given a
  number or URL, fetch it:

  ```bash
  gh issue view "$N" --repo "$OWNER/$REPO" --json title,body,labels
  ```

- Read-only. This skill never writes to the repository — no comments,
  no labels, no file edits. The only side effects are the `gh issue
  view` read above and printing the report to stdout.

## Shell helpers

Recipes below assume one helper is in scope:

```bash
abort() { echo "ABORT: $*" >&2; exit 1; }
```

## Preflight

`gh` must be authenticated with read access to the target repo:

```bash
gh auth status 2>&1 | grep -q "Logged in to github.com" \
  || abort "gh not authenticated; run: gh auth login"
```

No write scope is required — this skill only reads.

## Step 1: Classify issue_type

- **bug** — test failures, runtime errors, assertion errors, incorrect
  output, crashes.
  Indicators: error tracebacks, failing test names, `RuntimeError`,
  `AssertionError`, "fails with", `### 🐛 Describe the bug`, test logs.

- **skip-list** — a "Bug Skip" tracking issue asking whether a list of
  already-skipped tests should still be skipped.
  Indicators: `Bug Skip` in the title/template, `agent_test: skip-list`
  label, body is a checklist of test node ids (often with
  `~~strike-through~~` for entries already resolved), no fresh
  traceback.

- **nonbug** — feature requests, tasks, performance issues, questions,
  discussions, tracking issues, enhancement proposals, feature gaps.
  Indicators: "Enable", "[Task]", "Consider", "Align", "feature gap",
  "clarification", checklists of work items, `enhancement` label,
  `performance` label, no failing tests.

**Labels are authoritative** — if labels say `agent_test: skip-list`,
`issue_type = skip-list` regardless of body content.

## Step 2: Detect `reproduction_missing`

Report `yes` when the issue lacks all of:

- A reproducer command (pytest node id, `python -c ...`, shell command).
- A test node id reference (e.g. `test_foo.py::TestBar::test_baz`).
- A minimal code snippet that triggers the failure.

Report `no` when at least one of the above is present.

Skip-list issues list already-skipped test node ids in their body, so
they satisfy the second bullet and report `no`.

## Step 3: Estimate `scope`

Based on issue text alone (no source reading):

- **`pytorch`** — issue explicitly points at pytorch code (`torch/`,
  `aten/`, `torch/_inductor/`, `torch/_dynamo/`), a pytorch PR, or a
  framework-level regression.
- **`torch-xpu-ops`** — issue explicitly points at torch-xpu-ops code
  (`src/ATen/native/xpu/`, XPU kernels, SYCL implementations), or a
  ported CUDA test failing on XPU due to a kernel gap.
- **`both`** — issue text names changes needed in BOTH repos (e.g.
  pytorch API addition + XPU implementation of that API).
- **`unclear`** — issue text does not specify. This is the common case
  for most bug reports; a downstream deep-triage skill will decide
  after reading source.

## Step 4: Detect `runtime_dependencies`

Scan the issue body, error log, environment section, and labels for
explicit mentions of external runtime dependencies. Closed set:

| Value | What it means |
|---|---|
| `triton` | Inductor / torch.compile GPU codegen backend. |
| `onednn` | Intel oneDNN library (matmul, conv, etc.). |
| `onemkl` | Intel oneMKL library (BLAS, LAPACK, sparse). |
| `driver` | GPU driver, level-zero, compute-runtime, `libze_intel_gpu.so`. |
| `sycl` | SYCL runtime / DPC++ compiler. |
| `xccl` | XCCL communication library. |

Only report dependencies **explicitly named** in the issue. Do NOT infer
from a stack-trace path alone (e.g. a traceback through
`torch._inductor` does not by itself imply `triton` — the issue must
say so or show a triton-side error).

Empty array `[]` when none are named.

## Step 5: Derive `Handling`

Evaluate in order; first match wins:

1. `issue_type` is `nonbug` or an umbrella tracking task (a "parent"
   issue tracking multiple skip-listed/child test issues, not a single
   bug itself) → **needs-human**
   (reason: `"not a bug / task issue"`).
2. `reproduction_missing == yes` → **needs-human**
   (reason: `"no reproducer or test-name reference"`).
3. `runtime_dependencies` is non-empty → **needs-human**
   (reason: `"runtime dependency requires human triage: <list>"`).
4. Issue explicitly requires hardware or a non-public model/dataset the
   agent cannot access → **needs-human**
   (reason: names the missing resource).
5. Otherwise → **agent-fixable**.

`scope=both` and `scope=unclear` do NOT force needs-human — a
downstream deep-triage skill decides the final target repo.

## Step 6: Return the report

Emit both a markdown block and a JSON block to stdout, in that order,
separated by a blank line. The caller reads one, both, or neither.

### 6a. Markdown block

Assemble this exact structure (values from Steps 1–5). This is what a
caller would paste as a comment:

```markdown
<!-- agent:triage -->

## Issue Triage

| Field | Value |
|-------|-------|
| Reproduction missing | yes / no |
| Scope | pytorch / torch-xpu-ops / both / unclear |
| Dependencies | comma-separated list, or (none) |
| Handling | agent-fixable / needs-human |

**Reason:** <one-line, only when handling=needs-human>

*Automated by issue-triage.*
```

Each `Value` cell above lists the allowed choices; emit exactly one of
them. Never leave a literal `|` inside a cell — it splits the cell and
breaks the rendered table.

Include the `<!-- agent:triage -->` marker on the first line so a
downstream caller can locate its own previous comment (if any) and
update it in place. The marker is part of the report; the skill does
not consume it.

Omit the `**Reason:**` line entirely when `handling == agent-fixable`.

### 6b. JSON block

Immediately after the markdown block (with one blank line between),
emit:

```json
{
  "issue_type": "bug | skip-list | nonbug",
  "reproduction_missing": true | false,
  "scope": "pytorch | torch-xpu-ops | both | unclear",
  "runtime_dependencies": [],
  "handling": "agent-fixable | needs-human",
  "reason": "",
  "suggested_labels": []
}
```

Field notes:

- `reproduction_missing` is a JSON boolean: `true` for the `yes` shown
  in the table, `false` for `no`.
- `runtime_dependencies` is an array from the closed set in Step 4;
  empty `[]` when none named.
- `reason` is required non-empty when `handling == needs-human`, empty
  string when `handling == agent-fixable`.
- `suggested_labels` lists the labels a caller might apply to the
  issue. Advisory only — the skill does not apply them. Populate per
  the rules below.
- Do NOT invent values not derived from the issue text.

### 6c. Suggested labels

`suggested_labels` is populated as follows:

- If `reproduction_missing == true` → include `agent:reproduction-needed`.
- If `handling == "needs-human"` → include `agent:needs-human`.
- Empty array when neither applies (i.e. agent-fixable with a
  reproducer).

Scope and dependency values live in the JSON structure, not as labels.

## HARD RULES

- **Never modify the issue.** No comments, no labels, no body edits.
  The caller applies changes based on the report.
- **Never read source or run tests.** This skill's contract is
  text-only shallow triage. Deep source-reading analysis belongs to a
  separate downstream skill.
- **Emit exactly one markdown block and one JSON block to stdout,** in
  that order. Nothing else. No prose intro, no closing summary. The
  caller parses stdout; extra text corrupts the parse.
