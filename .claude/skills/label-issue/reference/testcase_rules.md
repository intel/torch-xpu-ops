# Test Surface and Test Case Rules

Decide `test` FIRST. An `e2e` issue uses the E2E case shape and skips
unit-test parsing entirely.

## `test`

The `test` value is one of the `code` values in `categories.test` of
`label_def.json`. Evaluate the codes in the order they are
defined in the JSON, testing each code's `evidence` and `keywords` against
`lowercase(label + " " + title + " " + body)`.

### Precedence and default

Take the first signal that fires, in this order:

1. e2e signal -> end-to-end code
2. oob signal -> out-of-box code
3. ut signal -> unit-test code

If none fires, default to the unit-test code — the surface `categories.test`
marks as the fallback.

## Unit-test entries

Shape: `{test_type, test_file, origin_test_file, test_class, test_case, error_message, source}`.

### Sources to scan, in this order

Scan each source top-to-bottom. Append every case as you find it. **Do not
sort.**

#### 1. A `Cases:` section

Everything after the first literal `Cases:`, stopping at the first `\n###`,
`\nVersions`, or a fence. Skip lines starting with `###` or `...`, and lines
fully wrapped in `~~...~~`.

Split each line on commas; require 3+ fields. Field 1 must be a known test
type: `op_ut`, `op_extend`, `op_extended`, `e2e`, `benchmark`, `ut`,
`test_xpu`.

- `op_ut,<dotted.path>,<case>` - field 2 is the dotted test path, field 3 the
  case. The case must be non-empty, 3+ chars, and contain no space. A
  `Class.method` field 3 splits into `test_class` + `test_case` when the path
  did not already yield a class.
- `op_ut,,<dotted.module>` - empty field 2 means a module-level import failure.
  Field 3 is the module path; `test_class` and `test_case` are `""`.

#### 2. A `test_cases:` section

Same CSV rules, but only lines beginning with `- `.

#### 3. Pytest node IDs, anywhere in the body

Match a `pytest` (or `python -m pytest`) invocation, skip its flags, and capture
the node id.

- `file.py::Class::method` -> `test_class=Class`, `test_case=method`
- `file.py` or `file.py::Class` -> empty-case row. `file.py::Class` does NOT
  make `Class` the `test_case`.

Keep the path literally; do not reconstruct it or derive a different
`origin_test_file`.

#### 4. `test_xpu,<path>,<case>` inside fenced code blocks

`test_type` is `test_xpu`. Build the file path by replacing every `.` with `/`
and appending `.py`. If the dotted path contains `.test_`, the component after
`.test_` becomes `test_class`.

#### 5. `-k <name>` selectors, in two passes

`test_type` is `ut` for both.

1. Inside a fenced block, associate `-k <name>` with a `pytest -v <test
   path>.py` in the SAME block.
2. Outside blocks, match `pytest -v <path>.py -k <name>`.

#### 6. `python benchmarks/dynamo/...` commands

Only when the body contains `benchmarks/dynamo/`. The script path is
`test_file`, the whole command is `test_case`, `test_type` is `e2e`.

#### 7. Body-wide `-k` fallback

Last. If the body mentions `pytest` at all, emit every `-k <name>` paired with
the first `pytest ... <test path>.py` found in the body. `test_type` is `ut`.

This runs after source 6, not alongside source 5, because it is the broadest
match: it pairs a selector with a `pytest` invocation anywhere in the body, so
running it earlier would shadow the block-scoped pairings above.

### Forms that yield NO case

Do not invent cases from:

- `FAILED x::y::z` summary lines
- unittest's `test_x (module.Class)` form
- a bare test name with no `-k` and no node id

These are deliberately unsupported. If the only evidence is one of these, leave
`test_cases` empty.

### Reconstructing `test_file` from a dotted path

For the CSV forms only:

1. Split on `.`.
2. Strip trailing components whose first character is uppercase; those are the
   class chain, joined by `.` into `test_class`.
3. If the remaining components contain `torch-xpu-ops` followed by `test`:
   `torch-xpu-ops/test/<rest joined by />.py`
4. Else if the first component is `test`: drop it, then `test/<rest>.py`
5. Else: `<components joined by />.py`

Examples:

```
test.test_xpu.TestFoo              -> test/test_xpu.py            class TestFoo
test.test_ops.TestFoo.test_bar     -> test/test_ops.py            class TestFoo
torch-xpu-ops.test.test_xpu.TestFoo -> torch-xpu-ops/test/test_xpu.py  class TestFoo
```

### Deriving `origin_test_file`

Applies to reconstructed CSV paths only.

- `test/xpu/<name>_xpu.py` -> `test/<name>.py`
- `test/xpu/<name>.py` -> `test/<name>.py`
- a path containing `benchmarks/` -> unchanged
- anything else -> unchanged

For pytest, code-block, `-k`, and benchmark-command rows,
`origin_test_file` equals `test_file`.

### `source`

Take the **basename** of `test_file`. If it ends in `_xpu.py` (equivalently, its
stem ends in `_xpu`), `source` is `torch-xpu-ops`; otherwise `pytorch`. Empty
`test_file` -> `""`.

The rule is basename-based, not full-path-based: `test/xpu/test_ops.py` is
`pytorch`, while `test/xpu/test_ops_xpu.py` is `torch-xpu-ops`.

### Dropping empty-case rows

An entry with an empty `test_case` is dropped when another entry for the SAME
`test_file` has a non-empty `test_case`. If it is the only entry for that file,
keep it - it records a module-level import failure.

## E2E entries

Shape: `{reproducer, benchmark, model, phase, dtype, amp, test_type, backend,
disable_cudagraphs, error_message}`. No `source` field.

Match against `title + " " + body`, case-insensitively.

| Field | Rule |
|---|---|
| `reproducer` | Up to 3 command lines. Prefer fenced-block lines that are non-empty, do not start with `#`, and either start with `python`/`pytest`/`XPU_`/`./` or contain `python`. Otherwise the first matches of: `pytest ...`, `python test/...`, `python -m pytest ...`, `XPU_QUANT_CONFIG=...python...`, `python benchmarks/dynamo/...`, `python ...run_benchmark...`. Last resort: first 200 chars of `title`. |
| `phase` | `training` if the text contains `training`, else `training` if it contains `train`, else `inference`. |
| `dtype` | First match, in this order: `bfloat16`/`bf16` -> `bfloat16`; `float16`/`fp16` -> `float16`; `float32`/`fp32` -> `float32`; `int8`/`int 8` -> `int8`. Default `float32`. Order matters: `bfloat16` wins when several appear. |
| `amp` | `true` when `--amp` or the word `amp` appears, else `false`. |
| `test_type` | `performance` when the text contains `throughputs`, `performance`, or `latency`; else `accuracy`. |
| `backend` | `--backend=<word>` wins; else `eager` if `eager` appears; else `inductor`. Default `inductor`. |
| `disable_cudagraphs` | `yes` when `disable-cudagraphs` or `disable_cudagraphs` appears, else `no`. |
| `model` | Each authoritative-list model matching as a whole word produces one row. |
| `benchmark` | Which list the model came from: check torchbench, then huggingface, then timm, comparing case-insensitively and also ignoring underscores. |

### No model matched

Emit a single row only when the text contains `benchmark`, `huggingface`,
`timm`, or `torchbench`. Then `model` is `"unknown"` and `benchmark` is:
`huggingface` for `hf_`/`huggingface`, `timm` for `timm_`/`timm.`,
`torchbench` for `torchbench`, else `unknown`.

### E2E de-duplication

The key is exactly these eight fields:

```
benchmark, model, phase, dtype, backend, test_type, amp, disable_cudagraphs
```

`reproducer` is EXCLUDED, so flag-order and whitespace variants of one run
collapse. Two entries differing in any of the eight - including AMP or cudagraph
mode alone - stay distinct. First occurrence wins.

## `error_message` - per-case failure signature

Every entry (unit-test and E2E) carries an `error_message`: the normalized
error/exception text attributed to THAT case, used by the grouping step
(`group_issue.md`) to separate distinct causes. Populate it in this order:

1. **Explicit per-case column.** If a `Cases:` / `test_cases:` CSV row carries a
   4th field after the case (`<type>,<path>,<case>,<error>`), that field is the
   `error_message`, verbatim.
2. **Adjacent error line.** Otherwise, the exception header attributed to the
   case in the body - the last `SomeError: message` / `E   SomeError: message`
   line inside the traceback or `FAILED ...` block for that case.
3. **Sole traceback fallback.** Otherwise, when the issue reports exactly one
   failure signature (one traceback, one exception header) and cannot be
   attributed per case, every entry takes that same normalized header.
4. **None found.** Otherwise `""`.

Normalize before storing: keep the exception class and message text, drop
run-specific noise (memory addresses, device ids, absolute file paths, line
numbers, tolerances/deltas, timestamps). This is the SAME normalization the
grouping step assumes, so two cases with the same true cause yield the same
`error_message` string.

## Ordering - the determinism contract

The parent takes `test_cases[0]` as the analyzed case and requires that choice
to be identical across runs.

**Unit-test issues.** Emit in the numbered source order of **Sources to scan**
above (1 `Cases:` through 7 body-wide `-k`), and within each source from the top
of the body to the bottom. De-duplicate by keeping the FIRST occurrence.

**E2E issues.** Emit in loaded-model-list order: huggingface names, then timm,
then torchbench.

Never re-rank by severity, dtype, alphabet, or "most interesting". Index 0 is
whatever the scan reached first.

## Top-level mirror fields

`test_file`, `test_class`, and `test_case` mirror the first **unit-test-shaped**
entry - the first entry with no `benchmark` key. Since `test` is decided
first and an `e2e` issue never reaches unit-test parsing, every entry in
`test_cases` on a `ut` issue is unit-test-shaped, so this is always
`test_cases[0]`. On an E2E issue no unit row exists, so all three are `""`.

## Benchmark model lists

Needed for E2E model detection. Read them; there is no built-in list.

Search these roots in order: `pytorch_folder`, `$PYTORCH_FOLDER`, the current
directory. Under each root, try `third_party/torch-xpu-ops/.ci/benchmarks` then
`.ci/benchmarks`. Use the first directory that yields any models.

In that directory read the top-level file AND the `p0/`, `p1/`, `p2/` tier
files, unioning them and keeping first occurrence:

```
<dir>/<file>  <dir>/p0/<file>  <dir>/p1/<file>  <dir>/p2/<file>
```

| Benchmark | File |
|---|---|
| huggingface | `huggingface_models_list.txt` |
| timm | `timm_models_list.txt` |
| torchbench | `torchbench_models_list.txt` |

Each line contributes its first token only; the rest is a batch size
(`ModelName,128` or `ModelName 128`).

The tier files matter: wrapped torchbench names such as `hf_Bert` and
`timm_vovnet` exist ONLY in a tier file. Reading just the top-level list misses
them.

Match a model as a whole token, case-insensitively, not adjacent to a word
character or hyphen. Prefer the longest match when names overlap.

If a list cannot be found, that bucket stays empty and model-name E2E detection
is disabled for it; label- and path-based signals still work. A stale built-in
list would silently mis-classify issues, so an empty bucket is preferred over a
wrong answer.
