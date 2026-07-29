---
name: issue-triage/extract-issue-information/extract-basic-info
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

Give it a full issue URL to target any repo (e.g.
`https://github.com/CuiYifeng/torch-xpu-ops-sandbox/issues/8`). A bare number
defaults to `intel/torch-xpu-ops` unless you pass `--repo owner/name`.

Do NOT use this for batch/multi-issue runs or Excel output. It handles exactly
one issue per invocation. It DOES extract test cases (unit-test and E2E),
traceback, and reproduce steps for that single issue.

## Prerequisites

- Authenticated `gh` CLI on `PATH`. The PyTorchXPU project fields and native issue type
  are fetched through GraphQL, so the token needs the `read:project` scope. Without that
  scope the project fields degrade to empty (the run still succeeds).
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
python3 .claude/skills/issue-triage/extract-issue-information/extract-basic-info/scripts/extract_basic_info.py 4344
```

By issue URL for any repo:

```bash
python3 .claude/skills/issue-triage/extract-issue-information/extract-basic-info/scripts/extract_basic_info.py https://github.com/CuiYifeng/torch-xpu-ops-sandbox/issues/8
```

By intel/torch-xpu-ops issue URL:

```bash
python3 .claude/skills/issue-triage/extract-issue-information/extract-basic-info/scripts/extract_basic_info.py https://github.com/intel/torch-xpu-ops/issues/4344
```

Override the repo for a bare issue number with `--repo owner/name`:

```bash
python3 .claude/skills/issue-triage/extract-issue-information/extract-basic-info/scripts/extract_basic_info.py 8 --repo CuiYifeng/torch-xpu-ops-sandbox
```

The `--repo owner/name` flag sets the repository for a bare issue number. It is
ignored when a full issue URL is given (the URL's own owner/name wins). The
default is `intel/torch-xpu-ops`.

Also write the JSON to a file (still printed to stdout):

```bash
python3 .claude/skills/issue-triage/extract-issue-information/extract-basic-info/scripts/extract_basic_info.py 4344 --output out.json
```

## Output schema

The script prints a single JSON object with these fields.

| Field | Source | Notes |
|-------|--------|-------|
| issue_id | gh REST | Issue number (integer). |
| repo | gh REST (input) | The issue's repository as "owner/name" (from the URL, or --repo/default for a bare number). |
| title | gh REST | Issue title. |
| status | gh REST | Issue state, "open" or "closed". |
| assignee | gh REST | First assignee login, or "". |
| reporter | gh REST | Issue author login. |
| labels | gh REST | Array of label name strings. |
| created_time | gh REST | ISO 8601 creation timestamp. |
| updated_time | gh REST | ISO 8601 last-update timestamp. |
| milestone | gh REST | Milestone title, or "". |
| summary | classifier | Issue title truncated to 150 chars. |
| type | classifier | See Classification reference. |
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
| traceback | classifier (regex) | Full Python traceback (call stack frames + error/exception message) if present, else "". |
| reproduce_steps | classifier (regex) | Shell command lines (cd/export/git/bash/pytest/python/etc.) extracted from the body, newline-joined; "" if none found. |
| test_file | classifier (regex) | Primary unit-test file (first parsed unit-test case); "" if none. |
| test_class | classifier (regex) | Primary unit-test class; "" if none. |
| test_case | classifier (regex) | Primary unit-test case/method; "" if none. |
| test_cases | classifier (regex) | Array of all test cases found in the issue (de-duplicated). Empty array if none. See Test cases below. |
| low_confidence | classifier | Array of field names the script could not confidently classify. |

## Classification reference

Enum outputs for the rule-based classifier fields:

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
- For unit-test entries, an empty-case row is dropped when a real case exists
  for the same test file.
- `test_cases` uses string-only path mapping; there is no on-disk verification.

## OS and platform

Two best-effort fields describe the reporting environment:

- `os`: `"Linux"` | `"Windows"` | `""`. Detected from OS keywords in the body
  and from a collect_env `OS:` line when present.
- `platform`: canonical Intel GPU code inferred from device names/aliases:
  Data Center GPU Max / Ponte Vecchio -> PVC; Battlemage / B580 -> BMG;
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
confidently extract. It contains ONLY these two field names:

- `reproduce_steps` - listed when NO shell command was found AND the issue is
  NOT a unit test (a unit test's test id is its own reproducer).
- `test_cases` - listed when no test case parsed but the issue looks
  test-related (`test_module` is `ut` or `e2e`).

`dependency` and `traceback` ARE output fields, but they are NEVER flagged in
`low_confidence`. The `os` and `platform` fields are best-effort and are also
never flagged.

When `low_confidence` is non-empty, the calling agent MUST:

1. Read the issue body and title.
2. For `reproduce_steps`, extract the real shell commands that reproduce the
   issue.
3. For `test_cases`, read the body and fill in the real test cases.
4. Overwrite those fields in the JSON with the determined values.
5. Remove each resolved field name from `low_confidence`.

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
- accept a `pytorch_folder` or `conda_env` argument.
