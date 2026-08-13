---
name: extract-issue-information
description: Extract basic metadata from a single intel/torch-xpu-ops GitHub issue and output JSON. Use when you need issue_id, title, status, labels, classified type/module/test_module/dependency, and PyTorchXPU project fields for ONE issue given its number or URL. Simplified single-issue, gh-only version of the batch Excel generator.
---

# Extract Basic Issue Info

Fetch ONE GitHub issue and emit its metadata plus rule-based classification as
JSON, including test cases (unit-test and E2E), traceback, and reproduce steps.

Do NOT use this for batch/multi-issue runs or Excel output; it handles exactly
one issue per invocation and never verifies test files on disk.

## Prerequisites

- Authenticated `gh` CLI on `PATH`, with `read:project` scope for the GraphQL
  project fields.
- Python 3. If missing, activate a `.venv` in the repo root or a parent
  directory and retry. Do NOT install tools yourself.

## Usage

Run from the repository root.

```bash
python3 .claude/skills/label-issue/extract-issue-information/scripts/extract_basic_info.py \
  <issue> [--repo owner/name] [--pytorch-folder <path>] [--output out.json]
```

| Argument | Purpose |
|---|---|
| `<issue>` | Bare issue number, or a full GitHub issue URL. |
| `--repo owner/name` | Repo for a bare number. Default `intel/torch-xpu-ops`. Ignored when a full URL is given, since the URL's own owner/name wins. |
| `--pytorch-folder` | Local checkout used only to load authoritative benchmark model lists. Never validated or modified. |
| `--output` | Also write the JSON to this path. It is always printed to stdout. |

Examples:

```bash
# bare number -> intel/torch-xpu-ops
... extract_basic_info.py 4344

# full URL -> any repo
... extract_basic_info.py https://github.com/usernmae/torch-xpu-ops-sandbox/issues/8

# bare number against another repo
... extract_basic_info.py 8 --repo username/torch-xpu-ops-sandbox
```

## Script layout

`scripts/` holds one entry point, five logic modules, and one data file. All
pattern and keyword tables live in `patterns.json`, not in code.

| File | Role |
|---|---|
| `extract_basic_info.py` | CLI entry point: argparse, orchestration, JSON assembly. |
| `patterns.py` | Loads `patterns.json` and exposes the tables as constants. |
| `patterns.json` | All regex/keyword tables. **List order is semantic** for `module.keywords`, `dependency.keywords`, `platform_keywords`, and `e2e_dtype_patterns` (first match wins) - do not sort or dedupe. In `platform_keywords`, an entry starting or ending with `\b` is matched as a regex; every other entry as a lowercase substring. |
| `classifiers.py` | `type`, `issue_type`, `module`, `test_module`, `dependency`, `os`, `platform`. |
| `testcases.py` | Test-case and E2E parsing, file resolution, de-duplication. |
| `benchmarks.py` | Benchmark model lists and model-name detection. |
| `github.py` | `gh` REST/GraphQL fetches and issue-reference parsing. |
| `text.py` | Traceback, reproduce steps, and PR link extraction. |

To change a keyword or pattern, edit `patterns.json` only. Two constraints:
`patterns.json` must sit beside `patterns.py` (a missing or malformed file is a
fatal error), and modules that read a mutable global must import the module
(`import benchmarks`) rather than the value (`from benchmarks import X`), so a
later `set_benchmark_models()` call is visible.

## Output schema

A single JSON object.

| Field | Source | Notes |
|-------|--------|-------|
| issue_id | gh REST | Issue number (integer). |
| repo | input | Resolved `owner/name`. |
| title | gh REST | Issue title. |
| body | gh REST | Raw body. Included so callers can resolve `low_confidence` without another fetch. |
| status | gh REST | `open` or `closed`. |
| assignee | gh REST | First assignee login, or "". |
| reporter | gh REST | Issue author login. |
| labels | gh REST | Array of label name strings. |
| created_time / updated_time | gh REST | ISO 8601 timestamps. |
| milestone | gh REST | Milestone title, or "". |
| summary | classifier | Title truncated to 150 chars. |
| type | classifier | `feature request` \| `performance issue` \| `accuracy issue` \| `functionality bug` \| `internal task` \| `unknown`. |
| issue_type | classifier | Canonical `Bug` \| `Task` \| `Feature` \| `Epic`. Precedence: `github_type` > labels > `type` heuristic. |
| github_type | gh GraphQL | Native GitHub issue type name, or "". |
| module | classifier | `distributed` \| `inductor` \| `dynamo` \| `aten_ops` \| `AO` \| `low_precision` \| `profiling` \| `optimizer` \| `fx` \| `export` \| `autograd` \| `unknown`. |
| test_module | classifier | `ut` \| `e2e` \| `build` \| `infrastructure`. |
| dependency | classifier | `oneDNN` \| `oneMKL` \| `Triton` \| `AO` \| `transformers` \| `oneAPI` \| `driver` \| `oneCCL` \| "". |
| priority | gh GraphQL | Normalized to P0-P3, or "". |
| pytorchxpu_status / _estimate / _depending / _short_comments | gh GraphQL | PyTorchXPU project fields, or "". |
| os | classifier | `Linux` \| `Windows` \| "". From body keywords and a collect_env `OS:` line. |
| platform | classifier | Intel GPU code (PVC, BMG, ARC, ARL, LNL, MTL, CRI), or "". See Platform below. |
| platform_specific | runtime | `true` when the issue platform differs from the local GPU family. Empty platform -> `false`; local detection failure -> `true` (conservative). |
| traceback | classifier | Full Python traceback (frames + error message), else "". |
| reproduce_steps | classifier | Shell command lines from the body, newline-joined; "" if none. |
| test_file / test_class / test_case | classifier | Mirror of the first parsed unit-test case; "" if none. |
| test_cases | classifier | All parsed test cases. See Test cases below. |
| pr_link | classifier | PR URL the issue is tied to; "" when none. See PR link below. |
| low_confidence | classifier | Field names needing LLM resolution. See Inline LLM fallback. |

`github_type`, `priority`, and the `pytorchxpu_*` fields are populated only for
issues in the PyTorchXPU project (intel/torch-xpu-ops). Anywhere else, or when
the GraphQL fetch fails, they degrade to "" and the run still exits 0.

## Test cases

`test_cases` is an array, de-duplicated per issue, using string-only path
mapping. Elements take one of two shapes.

**Unit-test entries** (`test_module` is `ut`, `build`, or `infrastructure`) have
the shape `{test_type, test_file, origin_test_file, test_class, test_case, source}`:

- `test_type`: a known test type (`op_ut`, `op_extend`, `e2e`, `benchmark`, `ut`, `test_xpu`, ...).
- `test_file`: reconstructed test file path; `origin_test_file` is the upstream path derived from it.
- `test_class` / `test_case`: names, or "".
- `source`: `torch-xpu-ops` when the file name ends with `_xpu`
  (e.g. `test_masked_xpu.py`), else `pytorch`.

Module-level entries carry the same keys with empty `test_class`/`test_case`,
recording only the file that failed to import. An empty-case row is dropped when
a real case exists for the same file.

**E2E entries** (`test_module` is `e2e`) carry `reproducer`, `benchmark`,
`model`, `phase`, `dtype`, `amp`, `test_type`, `backend`,
`disable_cudagraphs` - and no `source` field.

E2E classification triggers on an `e2e` label, a
`benchmarks/{dynamo,timm,huggingface,torchbench}/` or `run_benchmark.py` path,
or an authoritative-list model name (torchbench names like `alexnet` /
`BERT_pytorch`, huggingface class names - not just `hf_`/`timm_` prefixes)
mentioned with explicit benchmark-framework context.

The `benchmark` field (huggingface | timm | torchbench) and model detection use
`.ci/benchmarks/{huggingface,timm,torchbench}_models_list.txt`. The script
searches `third_party/torch-xpu-ops/.ci/benchmarks` then `.ci/benchmarks` under,
in order: `--pytorch-folder`, `$PYTORCH_FOLDER`, the current directory.

There is **no hardcoded fallback list**. If a list cannot be found, that bucket
stays empty, a warning names it on stderr, and model-name-based e2e detection is
disabled for it - label- and path-based e2e signals still work. A stale built-in
list would silently mis-classify issues, so an empty list plus a warning is
preferred over a wrong answer.

## PR link

`pr_link` is the URL of the PR the issue is tied to (e.g. a CI failure on a PR
rather than on main/nightly), or `""`.

```json
"pr_link": "https://github.com/pytorch/pytorch/pull/12345"
```

Regex pass, first match wins over the title and body:

- PR URLs `https://github.com/<owner>/<repo>/pull/<number>`
- Cross-repo shorthand `owner/repo#<number>`, normalized to a PR URL

A bare `#<number>` is NOT matched - without an `owner/repo` prefix it is a
same-repo issue reference. A branch-only reference yields `""`, since it has no
PR URL.

## Platform

`platform` is inferred in priority order: a `hw: <CODE>` label (e.g. `hw: BMG`),
then device names/aliases in the title, then in the body. Most-specific match
wins; `""` if none.

Mappings: Data Center GPU Max / Ponte Vecchio -> PVC; Battlemage / B580 -> BMG;
Alchemist / A770 -> ARC; Arrow Lake -> ARL; Lunar Lake -> LNL;
Meteor Lake -> MTL; Crescent Island -> CRI.

## Unit-test detection

An issue is a unit test if ANY hold: it carries a `module: ut` label; a parsed
test file lives under `test/` or `test/xpu/` or its name starts with `test_`; a
parsed class name starts with `Test`; a parsed case name starts with `test_`.

## Inline LLM fallback

`low_confidence` lists ONLY these fields, and only under these conditions:

| Field | Listed when |
|---|---|
| `reproduce_steps` | No shell command found AND the issue is not a unit test (a unit test's id is its own reproducer). |
| `test_cases` | No test case parsed but `test_module` is `ut` or `e2e`. |
| `pr_link` | Regex found no PR, but the body signals a non-main context ("this PR", "my branch", "cherry-pick", "backport", a CI run URL). |

`dependency`, `traceback`, `os`, and `platform` are output fields that are NEVER
flagged.

When `low_confidence` is non-empty, the calling agent MUST read `body`/`title`,
resolve each listed field, overwrite it in the JSON, and remove its name from
`low_confidence`. For `pr_link`, leave `""` if the issue is tied only to a
branch or to no PR. This fallback is inline: no disk queue, no sub-agent, no
batch processing.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success; JSON on stdout. A failed project/issueType fetch still exits 0 with those fields "". |
| 1 | Fetch failure (404, network), or the reference is a pull request. PRs are rejected. |
| 2 | Malformed reference (not a number, not a recognizable issue URL), or `--repo` without a `/`. |

Closed issues are allowed; `status` will be `closed`.
