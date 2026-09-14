# Traceback, Reproduce Steps, and PR Link Rules

## `traceback`

The full Python traceback, or `""`.

### Canonical form

Start at the first literal `Traceback (most recent call last):`. Consume
forward until ANY of:

1. A blank line whose following content is NOT one of:
   - `The above exception was the direct cause of the following exception`
   - `During handling of the above exception, another exception occurred`
   - another `Traceback (most recent call last):`
   - a `File "..."` frame
2. A line starting with `###`
3. A fence

Return the first such match, whitespace-stripped.

### Chained tracebacks are ONE traceback

A chained traceback is split by blank lines but is a single logical traceback.
Keep every linked segment:

```
Traceback (most recent call last):
  File "a.py", line 1, in <module>
    boom()
RuntimeError: first

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "b.py", line 2, in <module>
    again()
ValueError: second
```

Both `RuntimeError: first` and `ValueError: second` belong in the output. The
SECOND exception usually names the owning component, so truncating at the first
blank line loses the very frame the root-cause step needs.

Blank lines BETWEEN frames of one traceback are also not terminators - some
reporters insert them.

But a blank line followed by narrative prose, a `### Versions` heading, or a new
fenced block DOES terminate. The environment dump must never land in
`traceback`.

### Headerless form

When no `Traceback (most recent call last):` exists, accept an exception line
followed - optionally after blank lines - by one or more `File "...", line N`
frames plus their indented source lines.

An exception name qualifies when it ends in `Error`, `Exception`, or `Warning`,
OR is one of: `KeyboardInterrupt`, `SystemExit`, `StopIteration`,
`StopAsyncIteration`, `GeneratorExit`, `MemoryError`, `RecursionError`,
`SystemError`. The trailing `:` is optional.

The explicit list matters: `KeyboardInterrupt` follows none of the three suffix
conventions, and it is exactly the signal a hang or timeout report carries.

Otherwise `""`.

## `reproduce_steps`

Shell command lines from the BODY only, newline-joined, first-occurrence order,
de-duplicated. No cap. `""` when none. The title is not scanned.

### Per line

1. Strip a leading markdown list marker (`- `, `* `, `+ `, `1. `, `> `) plus
   surrounding whitespace and backticks. Strip trailing backticks.
2. Reject empty lines and lines starting with `#`.
3. Accept when the line starts with any of: `pytest`, `python3`, `python`,
   `./`, `pip install`, `cd `, `export `, `bash `, `sh `, `source `, `git `,
   `cmake`, `make`, `ninja`, `wget `, `curl `, `conda `, `pip `.
4. Or accept these env-prefixed forms:
   - `XPU_*=` or `PYTORCH_*=` followed somewhere by `python`
   - `ZE_AFFINITY_MASK=<value> <command>`
   - any `UPPERCASE_VAR=<value> <command>`
5. Then apply the prose rejection below.

### Prose rejection

Reject an otherwise-matching line when, after the first token, it reads as an
English sentence rather than a shell command — i.e. some later token is a
standalone English function word (`and`, `the`, `is`, `for`, `with`, `to`, `if`,
`when`, `manually`, and the like). Compare tokens case-insensitively after
stripping `.,;:()[]"'`.

A real shell command joins with operators - `&&`, `|`, `;` - never with the word
`and`. So a standalone function word means the line is prose that happens to
begin with a command name.

Rejected: `python support is documented for this feature.`,
`cd pytorch and then build it manually`, `git blame shows the regression`,
`make sure you have the driver installed`

Accepted: `cd pytorch && pip install -e .`,
`pip install -e . -v --no-build-isolation`, `pytest test/xpu/test_ops_xpu.py`,
`ZE_AFFINITY_MASK=0 python repro.py`, `make -j32`

**Trailing punctuation is NOT a rejection signal.** `pip install -e .`
legitimately ends in a dot. Do not add a sentence-punctuation rule.

## `pr_link`

The PR the issue is tied to (a CI failure on a PR rather than on main/nightly),
or `""`.

### 1. Trusted explicit URL

Search `title + "\n" + body` for
`https://github.com/<owner>/<repo>/pull/<number>`. Normalize the first match to
that canonical form and return it. Only `/pull/` URLs are trusted directly.

### 2. Ambiguous references

GitHub uses both remaining forms for issues AND PRs, so never assume either is a
PR. Look for the first match, in this order, and resolve it:

| Form | Matches when | Resolves against |
|---|---|---|
| `owner/repo#<number>` | Not preceded by `/` or a word character. | That `owner/repo`. |
| `#<number>` | Not preceded by `/` or a word character, so `foo/bar#12` and `abc#12` do not qualify. | The **current repo** - `intel/torch-xpu-ops` unless the input named another. |

Search `title + "\n" + body`. The two forms are alternatives: use a bare `#N`
only when no `owner/repo#N` was found, so the total cost stays at most ONE
resolution call per run, and only when step 1 found no `/pull/` URL.

Resolve with exactly one call against the repo the form points at:

```bash
gh api repos/<owner>/<repo>/issues/<number>
```

| Response | Result |
|---|---|
| Contains a `pull_request` key | It is a PR; emit `https://github.com/<owner>/<repo>/pull/<number>`. |
| No `pull_request` key | It is an issue; `pr_link` stays `""`. |
| Any failure - no `gh`, timeout, 404, private repo, invalid JSON | `""`. |

### 3. Not matched

A branch-only reference yields `""`, having no PR URL. Never guess a PR number.
