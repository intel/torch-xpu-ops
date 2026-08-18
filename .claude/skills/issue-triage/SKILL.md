---
name: issue-triage
description: >
  Shallow (text-only) triage of a GitHub issue on pytorch or torch-xpu-ops.
  Classify as bug/skip-list/nonbug, detect whether a reproducer is present,
  estimate scope (pytorch/torch-xpu-ops/both/unclear), identify runtime
  dependencies (triton/onednn/onemkl/driver/sycl/xccl), and emit a handling
  decision (agent-fixable / needs-human) based on those signals. Posts the
  result to the issue as a comment, applies matching GitHub labels, and
  returns a JSON summary. Use as a standalone triage entry point, or as
  the shallow first stage of a larger fix pipeline.
---

# Issue Triage — Shallow Classification & Handling Decision

Text-only triage. Reads the issue title, body, and labels. Does NOT read
source code, does NOT run tests, does NOT open PRs. Produces three
outputs:

1. A markdown table posted as a comment on the issue.
2. GitHub labels applied to the issue.
3. A JSON summary returned to the caller.

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

- Read-only with respect to the repository: this skill never reads
  source, runs tests, or edits files. Its only writes are the issue
  comment and the labels in Step 6.

## Shell helpers

Recipes below assume two helpers are in scope. Define them once at the
top of your shell (or source a shared file):

```bash
abort()    { echo "ABORT: $*" >&2; exit 1; }
log_warn() { echo "WARN: $*"  >&2; }
```

## Preflight

`gh` must be authenticated with `repo` or `write:issues` scope (this
skill writes a comment and applies labels):

```bash
gh auth status 2>&1 | grep -q "Logged in to github.com" \
  || abort "gh not authenticated; run: gh auth login"

scopes=$(gh auth status -t 2>&1 | grep -oE "'[a-z:_-]+'" | tr -d "'")
echo "$scopes" | grep -qE '^(repo|write:issues)$' \
  || abort "issue-triage requires 'repo' or 'write:issues' scope; got: $scopes"
```

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

## Step 6: Output

### 6a. Markdown table

Assemble this table (values from Steps 1–5). This is the primary
human-readable output:

```markdown
| Field | Value |
|-------|-------|
| Reproduction missing | yes / no |
| Scope | pytorch / torch-xpu-ops / both / unclear |
| Dependencies | comma-separated list, or (none) |
| Handling | agent-fixable / needs-human |
```

Each `Value` cell above lists the allowed choices; emit exactly one of
them. Never leave a literal `|` inside a cell — it splits the cell and
breaks the rendered table.

When `Handling == needs-human`, append a `**Reason:** <one-line>` line
below the table.

### 6b. Post / update the issue comment

Post the table as a GitHub comment on the issue, wrapped with a marker
so re-runs can find and update it in place:

```
<!-- agent:triage -->

## Issue Triage

<table from 6a>

<optional reason line>

*Automated by issue-triage.*
```

Look up the existing triage comment (if any) by the marker and edit
it in place; otherwise post new. Concrete recipe:

```bash
OWNER=<owner> REPO=<repo> N=<issue_number>

triage_body_file=$(mktemp)
# Write the marker + heading + table + reason to $triage_body_file.

# List existing triage comments, oldest first. Use the REST endpoint: it
# returns numeric comment ids, which the PATCH/DELETE comment endpoints
# require. `gh issue view --json comments` returns GraphQL node ids
# instead, and those endpoints reject them.
mapfile -t triage_ids < <(gh api "/repos/$OWNER/$REPO/issues/$N/comments" \
    --paginate \
    --jq '.[] | select(.body | startswith("<!-- agent:triage -->")) | .id')

if [ "${#triage_ids[@]}" -eq 0 ]; then
    gh issue comment "$N" --repo "$OWNER/$REPO" \
        --body-file "$triage_body_file" \
        || abort "gh issue comment failed"
else
    # `--field body="$(cat file)"` — do NOT use `-f body=@file`; gh
    # does not expand @path for POST/PATCH fields the way curl does.
    gh api "/repos/$OWNER/$REPO/issues/comments/${triage_ids[0]}" \
        --method PATCH \
        --field body="$(cat "$triage_body_file")" \
        || abort "gh api PATCH failed"

    # Best-effort dedup: delete the newer duplicates. Failure here is a
    # warning, not a hard abort.
    for dup_id in "${triage_ids[@]:1}"; do
        gh api "/repos/$OWNER/$REPO/issues/comments/$dup_id" \
            --method DELETE \
            || log_warn "could not delete duplicate triage comment $dup_id"
    done
fi
```

### 6c. Apply GitHub labels

Only apply "needs human attention"-class labels; scope and dependency
values live in the JSON output, not as labels:

- If `reproduction_missing == yes` → add `agent:reproduction-needed`.
- If `Handling == needs-human` → add `agent:needs-human`.

```bash
if [ "$reproduction_missing" = "yes" ]; then
    gh issue edit "$N" --repo "$OWNER/$REPO" \
        --add-label "agent:reproduction-needed" \
        || log_warn "add-label agent:reproduction-needed failed"
fi

if [ "$handling" = "needs-human" ]; then
    gh issue edit "$N" --repo "$OWNER/$REPO" \
        --add-label "agent:needs-human" \
        || log_warn "add-label agent:needs-human failed"
fi
```

`gh` returns non-zero if a label is already applied. That is a no-op,
not a failure. `log_warn` here is intentional: label state on the
issue is still correct even when the API call reports the label was
already there.

### 6d. Return JSON to the caller

Return this JSON as the LAST thing in your response, no markdown fences,
no explanation:

```json
{
  "issue_type": "bug | skip-list | nonbug",
  "reproduction_missing": true | false,
  "scope": "pytorch | torch-xpu-ops | both | unclear",
  "runtime_dependencies": [],
  "handling": "agent-fixable | needs-human",
  "reason": ""
}
```

Field notes:

- `reproduction_missing` is a JSON boolean: `true` for the `yes` shown
  in the table, `false` for `no`.
- `runtime_dependencies` is an array from the closed set in Step 4;
  empty `[]` when none named.
- `reason` is required non-empty when `handling == needs-human`, empty
  string when `handling == agent-fixable`.
- Do NOT invent values not derived from the issue text.

## HARD RULES

- **Never modify the issue body.** The body is user-owned. All
  agent-side state on the issue lives in comments (single
  `<!-- agent:triage -->` comment) and labels.
- **Never read source or run tests.** This skill's contract is
  text-only shallow triage. Deep source-reading analysis belongs to a
  separate downstream skill.
- **`gh api ... -f body=@file` is wrong** for editing comments. `gh`
  does not expand `@path` for POST/PATCH fields the way curl does.
  Use `--field body="$(cat file)"`.
- **Comment ids must come from the REST list endpoint.** PATCH/DELETE
  on `/repos/.../issues/comments/<id>` need the numeric id returned by
  `gh api /repos/.../issues/<n>/comments`, not the GraphQL node id
  returned by `gh issue view --json comments`.
- **Labels are advisory** — repeated `add-label` calls for a label
  already on the issue are no-ops, not failures.
