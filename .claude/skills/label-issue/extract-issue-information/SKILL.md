---
name: extract-issue-information
description: Extract basic metadata from a single intel/torch-xpu-ops GitHub issue and output JSON. Use when you need issue_id, title, status, labels, classified type/module/test_module/dependency, and PyTorchXPU project fields for ONE issue given its number or URL. Simplified single-issue, gh-only version of the batch Excel generator.
---

# Extract Basic Issue Info

Fetch one GitHub issue and emit its basic metadata plus rule-based classification as JSON.
It works on any GitHub repo's issue when given a full issue URL. A bare issue number
defaults to `intel/torch-xpu-ops` (override with `--repo owner/name`).

## When to use

Use this when you have a single issue number or URL and want structured JSON:
issue identity fields, GitHub labels, a rule-based `type`/`module`/`test_module`/`dependency`
classification, and the issue's PyTorchXPU project fields.

Give it a full issue URL to target any repo. A bare number
defaults to `intel/torch-xpu-ops` unless you pass `--repo owner/name`.

Do NOT use this for batch/multi-issue runs or Excel output. It handles exactly
one issue per invocation. It DOES extract test cases (unit-test and E2E),
traceback, and reproduce steps for that single issue.

## Prerequisites

- Authenticated `gh` CLI on `PATH`. The PyTorchXPU project fields and native issue type
  are fetched through GraphQL. Without that
  the project fields degrade to empty (the run still succeeds).
- Python 3.

The PyTorchXPU project fields (`priority`, `pytorchxpu_*`) and `github_type` are populated
only for issues that belong to the PyTorchXPU project (intel/torch-xpu-ops). For issues in
any other repo, or intel/torch-xpu-ops issues that are not in the project, these fields are
"" (best-effort, graceful degradation; the run still exits 0).

If `python3` or its dependencies are missing, check for a `.venv` in the project root
or a parent directory and activate it, then retry. Do NOT install tools yourself.

## Usage

Run from the repository root.

By issue number (defaults to intel/torch-xpu-ops):

```bash
python3 .claude/skills/label-issue/extract-issue-information/scripts/extract_basic_info.py 4344
```

By issue URL for any repo:

```bash
python3 .claude/skills/label-issue/extract-issue-information/scripts/extract_basic_info.py https://github.com/CuiYifeng/torch-xpu-ops-sandbox/issues/8
```

By intel/torch-xpu-ops issue URL:

```bash
python3 .claude/skills/label-issue/extract-issue-information/scripts/extract_basic_info.py https://github.com/intel/torch-xpu-ops/issues/4344
```

Override the repo for a bare issue number with `--repo owner/name`:

```bash
python3 .claude/skills/label-issue/extract-issue-information/scripts/extract_basic_info.py 8 --repo CuiYifeng/torch-xpu-ops-sandbox
```

The `--repo owner/name` flag sets the repository for a bare issue number. It is
ignored when a full issue URL is given (the URL's own owner/name wins). The
default is `intel/torch-xpu-ops`.

Also write the JSON to a file (still printed to stdout):

```bash
python3 .claude/skills/label-issue/extract-issue-information/scripts/extract_basic_info.py 4344 --output out.json
```

## Output schema

The script prints a single JSON object with these fields.

| Field | Source | Notes |
|-------|--------|-------|
| issue_id | gh REST | Issue number (integer). |
| repo | gh REST (input) | The issue's repository as "owner/name" (from the URL, or --repo/default for a bare number). |
| title | gh REST | Issue title. |
| body | gh REST | Raw issue body. Included so callers can resolve `low_confidence` fields without another remote fetch. |
| status | gh REST | Issue state, "open" or "closed". |
| assignee | gh REST | First assignee login, or "". |
| reporter | gh REST | Issue author login. |
| labels | gh REST | Array of label name strings. |
| created_time | gh REST | ISO 8601 creation timestamp. |
| updated_time | gh REST | ISO 8601 last-update timestamp. |
| milestone | gh REST | Milestone title, or "". |
| summary | classifier | Issue title truncated to 150 chars. |
| type | classifier | See Classification reference. |
| issue_type | classifier | Canonical type: Bug, Task, Feature, or Epic. Derived from github_type > labels > classifier heuristic. |
| github_type | gh GraphQL issueType | Native GitHub issue type name, or "". |
| module | classifier | See Classification reference. |
| test_module | classifier | See Classification reference. |
| dependency | classifier | See Classification reference; "" when none detected. |
| priority | gh GraphQL project | Normalized to P0-P3, or "". |
| pytorchxpu_status | gh GraphQL project | PyTorchXPU project Status field, or "". |
| pytorchxpu_estimate | gh GraphQL project | PyTorchXPU project Estimate field, or "". |
| pytorchxpu_depending | gh GraphQL project | PyTorchXPU project Depending field, or "". |
| pytorchxpu_short_comments | gh GraphQL project | PyTorchXPU project Short Comment field, or "". |
| os | classifier (regex) | "Linux" or "Windows" detected from the issue body; "" if not found. |
| platform | classifier (regex) | Canonical Intel GPU platform code (PVC, BMG, ARC, ARL, LNL, MTL, CRI); "" if not found. |
| platform_specific | classifier (runtime) | `true` if issue platform differs from the local GPU family, `false` otherwise. Empty platform → `false`. Local detection failure → `true` (conservative). |
| traceback | classifier (regex) | Full Python traceback (call stack frames + error/exception message) if present, else "". |
| reproduce_steps | classifier (regex) | Shell command lines (cd/export/git/bash/pytest/python/etc.) extracted from the body, newline-joined; "" if none found. |
| test_file | classifier (regex) | Primary unit-test file (first parsed unit-test case); "" if none. |
| test_class | classifier (regex) | Primary unit-test class; "" if none. |
| test_case | classifier (regex) | Primary unit-test case/method; "" if none. |
| test_cases | classifier (regex) | Array of all test cases found in the issue (de-duplicated). Empty array if none. See Test cases below. |
| pr_context | classifier (regex+LLM) | PR or branch context if the issue is tied to a specific PR/branch. See PR/branch context below. |
| low_confidence | classifier | Array of field names the script could not confidently classify. |

## Classification reference

Enum outputs for the rule-based classifier fields:

- `issue_type`: Bug | Task | Feature | Epic (canonical; priority: github_type > labels > type heuristic)
- `type`: feature request | performance issue | accuracy issue | functionality bug | internal task | unknown
- `module`: distributed | inductor | dynamo | aten_ops | AO | low_precision | profiling | optimizer | fx | export | autograd | unknown
- `test_module`: ut | e2e | build | infrastructure
- `dependency`: oneDNN | oneMKL | Triton | AO | transformers | oneAPI | driver | oneCCL | "" (empty)

## Test cases

`test_cases` is an array of every test case parsed from the issue, de-duplicated
per issue. Elements take one of two shapes.

Unit-test entries (when `test_module` is `ut`, `build`, or `infrastructure`):

- `test_type`: one of the known test types (op_ut, op_extend, e2e, benchmark, ut, test_xpu, ...).
- `test_file`: reconstructed test file path (string-only mapping, no on-disk verification).
- `origin_test_file`: upstream file path derived from `test_file`.
- `test_class`: test class name, or "".
- `test_case`: test method name, or "".
- `source`: `"torch-xpu-ops"` when the test file name ends with `_xpu`
  (e.g. test_masked_xpu.py), otherwise `"pytorch"` (an upstream PyTorch test).

So a unit-test entry has the shape
`{test_type, test_file, origin_test_file, test_class, test_case, source}`.
Module-level entries carry the same keys but may have empty `test_class` and
`test_case` (they record only the file that failed to import).

E2E entries (when `test_module` is `e2e`):

- `reproducer`, `benchmark`, `model`, `phase`, `dtype`, `amp`, `test_type`,
  `backend`, `disable_cudagraphs`.

E2E entries do NOT have a `source` field.

Notes:

- Entries are de-duplicated per issue.
- The `benchmark` field (huggingface | timm | torchbench) and e2e model
  detection use the authoritative model lists in
  `intel/torch-xpu-ops/.ci/benchmarks/{huggingface,timm,torchbench}_models_list.txt`.
  The script loads these from a local checkout at runtime (pass `--pytorch-folder`,
  set `PYTORCH_FOLDER`, or rely on the `~/ai4ee` default; it searches
   `third_party/torch-xpu-ops/.ci/benchmarks` and `.ci/benchmarks`). Hardcoded
   lists in the script are only a fallback when no checkout is found.
- E2E classification (`test_module` = `e2e`) triggers on an `e2e` label, a
  `benchmarks/{dynamo,timm,huggingface,torchbench}/` or `run_benchmark.py` path,
  or any authoritative-list model name (torchbench names like `alexnet` /
  `BERT_pytorch` and huggingface class names, not just `hf_`/`timm_` prefixes)
  mentioned together with an explicit benchmark-framework context.
- For unit-test entries, an empty-case row is dropped when a real case exists
  for the same test file.
- `test_cases` uses string-only path mapping; there is no on-disk verification.

## PR/branch context

`pr_context` captures when an issue is tied to a specific PR or branch (e.g., a
CI failure on a PR, not on main/nightly). Structure:

```json
{
  "has_pr_context": true,
  "repo": "pytorch/pytorch",
  "pr_number": 12345,
  "branch": "feature-branch-name",
  "source": "regex|llm"
}
```

- `has_pr_context`: `true` when a PR or branch reference is detected, `false` otherwise.
- `repo`: repository of the PR/branch (`pytorch/pytorch` or `intel/torch-xpu-ops`).
- `pr_number`: integer PR number, or `null` if only a branch is referenced.
- `branch`: branch name string, or `null` if only a PR number is referenced.
- `source`: `"regex"` when extracted via URL/pattern matching, `"llm"` when
  determined by LLM fallback.

When `has_pr_context` is `false`, the value is simply:
```json
{"has_pr_context": false, "repo": null, "pr_number": null, "branch": null, "source": null}
```

### Regex extraction (first pass)

Look for these patterns in the issue title and body:

- GitHub PR URLs: `https://github.com/<owner>/<repo>/pull/<number>`
- PR references: `#<number>` in context of "PR", "pull request", "merge"
- Branch references: `branch: <name>`, `on branch <name>`, `refs/heads/<name>`
- CI log URLs containing `/pull/<number>/` or `/tree/<branch>`

### LLM fallback

When regex extraction finds nothing but the issue body contains signals that
it occurred on a non-main branch or PR (e.g., mentions "this PR", "my branch",
"cherry-pick", "backport", CI failure context referencing a specific change),
the calling agent MUST read the issue body and determine the PR/branch context.
Add `pr_context` to `low_confidence` when regex finds nothing but signals exist.

## OS and platform

Two best-effort fields describe the reporting environment:

- `os`: `"Linux"` | `"Windows"` | `""`. Detected from OS keywords in the body
  and from a collect_env `OS:` line when present.
- `platform`: canonical Intel GPU code inferred in priority order:
  1. Labels matching `hw: <CODE>` (e.g. `hw: BMG`, `hw: PVC`).
  2. Device names/aliases in the title.
  3. Device names/aliases in the body.
  Mappings: Data Center GPU Max / Ponte Vecchio -> PVC; Battlemage / B580 -> BMG;
  Alchemist / A770 -> ARC; Arrow Lake -> ARL; Lunar Lake -> LNL;
  Meteor Lake -> MTL; Crescent Island -> CRI. A single value is chosen,
  most-specific first; `""` if none matches.

Both fields are BEST-EFFORT and are NEVER added to `low_confidence`.

## Unit-test detection

An issue is treated as a unit test if ANY of the following hold:

- it carries a `module: ut` label;
- a parsed test file lives under `test/` or `test/xpu/`, or its name starts
  with `test_`;
- a parsed test class name starts with `Test`;
- a parsed test case/method name starts with `test_`.

When the issue is a unit test, the top-level `test_file` / `test_class` /
`test_case` mirror the primary parsed case, and `reproduce_steps` is NOT
required: the test id is itself the reproducer, so an empty `reproduce_steps`
is not flagged in `low_confidence`.

## Inline LLM fallback

The script populates `low_confidence` with the names of fields it could not
confidently extract. It contains ONLY these field names:

- `reproduce_steps` - listed when NO shell command was found AND the issue is
  NOT a unit test (a unit test's test id is its own reproducer).
- `test_cases` - listed when no test case parsed but the issue looks
  test-related (`test_module` is `ut` or `e2e`).
- `pr_context` - listed when regex found no PR/branch but the issue body
  contains signals of a non-main context (mentions "this PR", "my branch",
  CI log URLs, etc.).

`dependency` and `traceback` ARE output fields, but they are NEVER flagged in
`low_confidence`. The `os` and `platform` fields are best-effort and are also
never flagged.

When `low_confidence` is non-empty, the calling agent MUST:

1. Read the `body` and `title` fields in the extracted JSON.
2. For `reproduce_steps`, extract the real shell commands that reproduce the
   issue.
3. For `test_cases`, read the body and fill in the real test cases.
4. For `pr_context`, determine the PR number/branch from context and populate
   the `pr_context` object (set `source: "llm"`).
5. Overwrite those fields in the JSON with the determined values.
6. Remove each resolved field name from `low_confidence`.

This fallback is inline: no disk queue, no sub-agent, no batch processing.

## Edge cases / exit codes

- Exit 0: success. The JSON is printed to stdout. If the project / issueType fetch fails
  (missing scope, network error, timeout), those project fields degrade to "" and the run
  still exits 0.
- Exit 1: fetch failure (404 or network error), or the input number refers to a pull
  request. Pull requests are rejected.
- Exit 2: malformed input reference (not a number and not a recognizable issue URL),
  or a `--repo` value without a `/` separator.
- Closed issues are allowed; `status` will be "closed".

## Scope

This script does exactly one thing: emit JSON metadata for a single issue. It does NOT:

- produce Excel output,
- process batches or multiple issues,
- verify test files on disk (test_cases uses string-only path mapping),
- generate a Not-applicable sheet,
- accept a `conda_env` argument. It accepts optional `--pytorch-folder` only
  to load authoritative benchmark model lists; it does not validate or modify
  that checkout.
