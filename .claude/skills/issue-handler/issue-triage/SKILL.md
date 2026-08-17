---
name: issue-triage
description: >
  Shallow (text-only) triage of a GitHub issue. Classify as bug,
  skip-list, or nonbug; identify runtime dependencies (triton, onednn,
  onemkl, driver, ...); estimate scope (pytorch / torch-xpu-ops / both /
  unclear); and emit a preliminary verdict (agent-fixable / NEEDS_HUMAN)
  based on issue body/labels alone. No source access. Use as the first
  stage of issue handling; deep root-cause analysis and any override of
  the preliminary verdict happen later in `fix/root-cause`.
---

# Issue Triage — Shallow Classification & Preliminary Verdict

Text-only triage. Reads the issue title/body/labels and emits a
structured JSON classification without touching source code. This is
the shallow layer; the deep counterpart is `fix/root-cause`, which reads
source and may override this skill's `scope` / `verdict` outputs.

## Inputs

You receive a GitHub issue: its title, body, and labels.

If the body is already split into sections (description, reproducer,
error_log, environment, additional_context), use them directly. If not,
identify those portions yourself from the raw body before classifying.

## Step 1: Classify issue_type

**Bug** — test failures, runtime errors, assertion errors, incorrect output, crashes.
Indicators: error tracebacks, failing test names, "RuntimeError", "AssertionError",
"fails with", `### 🐛 Describe the bug`, test logs.

**Skip-list** — a "Bug Skip" tracking issue asking whether a list of already-skipped
tests should still be skipped. Indicators: `Bug Skip` in the title or template,
`agent_test: skip-list` label, body is a checklist of test node ids (often with
`~~strike-through~~` for entries already resolved), no fresh error traceback.
Distinct from a plain bug because the fix path is per-entry re-verification, not
root-cause analysis of a single failure — see `fix/skip-list`.

**Non-bug** — feature requests, tasks, performance issues, questions, discussions,
tracking issues, enhancement proposals, feature gaps.
Indicators: "Enable", "[Task]", "Consider", "Align", "feature gap", "clarification",
checklists of work items, `enhancement` label, `performance` label, no failing tests.

**Labels are authoritative** — if labels say `agent_test: ut`, test_type is `ut`;
if labels say `agent_test: skip-list`, issue_type is `skip-list`.

## Step 2: Identify runtime dependencies

Scan the issue body, error log, environment section, and labels for
mentions of external runtime dependencies. Report each one that is
explicitly named. Common values:

- `triton` — Inductor / torch.compile GPU codegen backend.
- `onednn` — Intel oneDNN library (matmul, conv, etc.).
- `onemkl` — Intel oneMKL library (BLAS, LAPACK, sparse).
- `driver` — GPU driver, level-zero, compute-runtime, `libze_intel_gpu.so`, etc.
- `sycl` — SYCL runtime / DPC++ compiler.
- `ipex` — Intel Extension for PyTorch (if the issue references it).
- `xccl` — XCCL communication library.

Report ONLY dependencies explicitly named in the issue. Do NOT infer
from the failing symbol path alone (e.g. a traceback through
`torch._inductor` does not automatically mean `triton` is a
dependency of the bug — it might be, but only if the issue text says
so or shows a triton-side error).

Output as a JSON array; empty array `[]` if none named.

## Step 3: Estimate scope

Based on issue text only (no source reading), estimate which repo(s)
need to change to fix the bug. Values:

- `"pytorch"` — issue explicitly points at pytorch code (`torch/`,
  `aten/`, `torch/_inductor/`, `torch/_dynamo/`), a pytorch PR, or a
  framework-level regression.
- `"torch-xpu-ops"` — issue explicitly points at torch-xpu-ops code
  (`src/ATen/native/xpu/`, XPU kernels, SYCL implementations) or a
  ported CUDA test failing on XPU due to a kernel gap.
- `"both"` — issue text names changes needed in BOTH repos (e.g.
  pytorch API addition + XPU implementation of that API). This is
  passed down to `fix/root-cause`, which decides whether to attempt or
  fall back to `NEEDS_HUMAN`.
- `"unclear"` — issue text does not specify. This is the common case
  for most bug reports; `fix/root-cause` will decide the final
  `target_repo` after reading source. **`"unclear"` is NOT
  `NEEDS_HUMAN`** — it is the normal handoff to the deep triage stage.

## Step 4: Preliminary verdict

Emit `agent-fixable` or `NEEDS_HUMAN` based on issue text alone.

**Return `NEEDS_HUMAN` when the issue itself indicates the agent
cannot make progress:**

- **Non-bug or umbrella task** — `nonbug` classification (feature
  request, discussion, `[Task]`), or a tracking issue enumerating
  many independent items.
- **No error signal** — bug label but zero traceback, zero error
  message, zero failing test name. Agent has nothing to grep for.
- **No minimal reproducer AND no test-name reference** — no
  reproducer command, no `test_x.py::TestY::test_z` mentioned
  anywhere, no code snippet that triggers the failure.
- **Hardware-only failure** — issue explicitly requires a specific
  hardware setup the agent does not have access to (e.g. multi-node
  distributed, non-public silicon, requires custom firmware).
- **Non-public dependency** — issue requires a private model,
  checkpoint, dataset, or internal-only tool.

**Return `agent-fixable` when** the issue has enough signal for the
downstream pipeline to at least attempt reproduction and analysis.
`fix/root-cause` may still return `NEEDS_HUMAN` later after seeing the
code — that is expected and not a failure of this stage.

Note: `scope=unclear` alone does NOT force `NEEDS_HUMAN`. Most bugs
have unclear scope until source is read.

## Output

Return ONLY this JSON object, no markdown fences, no explanation:

```json
{
  "issue_type": "bug | skip-list | nonbug",
  "test_type": "ut | e2e | \"\"",
  "runtime_dependencies": ["triton", "onednn", "onemkl", "driver", "sycl", "ipex", "xccl"],
  "scope": "pytorch | torch-xpu-ops | both | unclear",
  "verdict": "agent-fixable | NEEDS_HUMAN",
  "reason": "<one-line reason, mandatory for NEEDS_HUMAN, else empty>",
  "platform": "xpu | <specific GPU model>",
  "category": "<category string>",
  "related_components": "<components string>",
  "context": "<one-line summary with upstream links if available>",
  "formatted_body": "<pipeline mode only; empty string in interactive mode>"
}
```

Field notes:

- `test_type`: `"ut"` for unit tests, `"e2e"` for end-to-end, `""` if unclear.
- `runtime_dependencies`: array of strings from the closed set above.
  Empty `[]` when none named. Do NOT invent values not explicitly in
  the issue.
- `scope`: preliminary — `fix/root-cause` may override after reading
  source. `"unclear"` is a valid, common value; do NOT default to
  `NEEDS_HUMAN` just because scope is unclear.
- `verdict`: preliminary — `fix/root-cause` may downgrade
  `agent-fixable` to `NEEDS_HUMAN` after reading source. This skill
  never upgrades `NEEDS_HUMAN` to `agent-fixable` from source; it
  only sees text.
- `reason`: required non-empty when `verdict == "NEEDS_HUMAN"`; empty
  string when `verdict == "agent-fixable"`.
- `platform`: `"xpu"` unless a specific GPU is mentioned (e.g. BMG).
- Do NOT hallucinate — if info isn't in the issue, use `""` (or `[]`
  for the array field).

In pipeline mode, populate `formatted_body` as a string that
`issue-handler` will use as the discovery section of the state
comment (posted to the issue as a GitHub comment, never written into
the issue body).

## Pipeline mode: issue body is read-only

**Do not modify the issue body in either mode.** The
`issue-handler`-owned state comment (marker `<!-- agent:state -->`)
carries all agent-side state (stage, discovery log, etc.). This stage
owns the "Discovery log" section inside that state comment, not the
issue body.

See the pipeline mode comment contract in [`../SKILL.md`](../SKILL.md)
(`issue-handler`).
