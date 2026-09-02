# Filing procedure

Everything here happens after you have grouped the failures. The order matters:
each step can only reduce what the next one files.

## 1. Decide whether to file anything

`run.json.gates`:

| Gate | Meaning |
|---|---|
| `build_failed` | the build did not succeed, so nothing downstream can be trusted. File nothing. |
| `abort` | more new failures than the run-level threshold. File nothing. |
| `oversized` | too many failures to group meaningfully. File nothing; say what the volume looks like. |

Then your own reading, per leg, from `run.json.legs`:

```
infra_pattern_ratio > limits.infra_leg_share
  and new_failures >= limits.infra_leg_min_cases
    -> that leg is machine breakage. File nothing from it.
```

This is deliberately blunt. A share of denylisted messages that high says the
machine misbehaved, and the ordinary-looking failures around them are not
trustworthy either. Filing them would mute tests on the strength of a bad
night.

## 2. Drop what is already muted

```bash
gh issue list --repo <repo> --state open --label skipped --limit 200 \
  --json number,title,body
```

Read the `Cases:` block of each. Any case line that appears in an open issue is
already muted by it - drop that case from your group. If every case in a group
is already placed, do not file anything for it; the existing issue covers it.

If some of a group's cases are already placed and some are not, file only the
unplaced ones, and comment on the existing issue that the same root cause
produced more failures tonight.

Also fetch `skipped_bmg`, `regression` and `new_case_failure`, since an issue
carrying one of those may not carry `skipped`.

## 3. File

One issue per group, following
`.github/ISSUE_TEMPLATE/agent/ut-auto-issue-body.md`.

### The Cases block

```
<!-- cases:begin -->
Cases:
op_extended,test_ops_xpu.TestFooXPU,test_bar_xpu_float32
op_extended,test_ops_xpu.TestFooXPU,test_bar_xpu_bfloat16
<!-- cases:end -->
```

One `line` value per case, copied verbatim from `cases.json`. No bullets, no
backticks, no indentation, no blank lines, no abbreviation. Write the body to a
file and pass it with `--body-file` rather than `--body`, so nothing is
reshaped by shell quoting.

### Title

```
[Bug Skip]: <prefix><what is failing>
```

**Every title starts with the literal `[Bug Skip]: `.** It is how a human
scanning the issue list tells a machine-filed skip from a hand-written report,
and there is no case in which it is dropped - not for a regression, not for a
collection error, not to make room in a title that is already long.

`<prefix>` is `[Regression] ` when the group's cases carry `cls: regression`,
`[New Case] ` when they carry `cls: new_case_failure`, and empty otherwise -
read off the evidence like the labels, not judged. For a group
of whole-module rows the title names the file instead:

```
[Bug Skip]: [Failed to collect] <prefix><test file>: <the import error>
```

Keep it under about 140 characters and plain ASCII.

### Labels

Copy the `labels` list from any case in the group - `cases.json` carries it
resolved, per case. It is the same list for every case in a uniform group,
because it is a pure function of `cls` and the runner that ran the leg.

What it resolves to, for reading the list rather than for producing one:

- `skipped` on every issue, without exception. It is what mutes the case.
- `skipped_bmg` only when the leg ran on a BMG runner - `fetch_issues.sh:25`
  honours it on those and ignores it everywhere else, so putting it on a
  non-BMG issue claims a scope the issue does not have.
- The classification, and only when it is `regression` or `new_case_failure`.
  `persistent` and `unknown` carry no classification label on purpose, since
  neither "it used to pass" nor "it is a new case" is true of them, and the
  body states the position instead.

Do not derive it, and in particular do not reach for `new_case_failure` because
a failure looks new to you.

`run.json.labels` holds the same lists keyed `<cls>|<leg>`, for a group whose
cases were all already placed.

### Body

Paste, do not compose:

| Section | Source |
|---|---|
| `Cases:` block | `cases.json`, verbatim |
| ErrorLog traceback | `tracebacks.json`, verbatim, or say none was captured |
| baseline table | `blocks.json.baseline_table_header` + `baseline_table_rows[category]` |
| bisect range | `blocks.json.compare_links[category]` |
| stale-baseline note | `blocks.json.baseline_staleness[category]`, when present |
| whole-module table and verdict | `blocks.json.collection_error[line]` |
| reproduce command | `cases.json.reproduce[category]`, with the case substituted for `failed_case` |
| collect_env | `run.json.collect_env[leg]` |

Only the summary, the root cause and the title are yours to write.

A group classified `persistent` or `unknown` has no baseline table to paste.
Say instead that the failure predates the baseline, or that no nightly in the
lookback window completed the category healthily, so onset could not be
determined.

### Size

If a group has more than `limits.max_cases_per_issue` cases, or its body would
exceed `limits.safe_body_chars`, split it into numbered parts:

- Part 1 carries the ErrorLog; later parts link back to it.
- Every part carries its own complete slice of the `Cases:` block.
- Marker `part=<n>/<total>` on each.

Never shorten the `Cases:` block to make a body fit.

### Marker

End the body with `run.json.marker_template`, filled in. It is how a later
night tells a machine-filed issue from a hand-written one.

## 4. Cross-link

When one root cause was split across several issues - by classification, by row
shape, or by size - comment on each with links to the others. Nothing else in
the pipeline will ever say they are related.

## 5. Report

As your final message: how many issues you filed and their numbers, how many
cases they cover, what you deliberately did not file and why, and anything you
were unsure about.
