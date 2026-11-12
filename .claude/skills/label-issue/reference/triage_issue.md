# Root Cause Rules

How to establish the defect and its owner for the representative case in Step 3.1
of the label-issue skill. SKILL.md defers to this file.

## Inputs

Work from `extract.json`: the representative case's `error_message`, the
`traceback`, `test_cases`, `reproduce_steps`, `title`, and `body`.

The `error_message` is the primary failure signature. Read its exception class
and message first, then locate that failure in the `traceback` frame.

## Choose the mode

The mode depends on whether a `pytorch_folder` checkout is available:

| | Mode A — traced | Mode B — evidence-only |
|---|---|---|
| When | `pytorch_folder` given and exists | absent or nonexistent |
| Sources | the checkout, plus Step 1 evidence | Step 1 evidence and `gh` only |
| Never | propose a fix | clone, fetch, or search a checkout |

## Mode A — traced

Delegate the trace to a read-only deep-analysis subagent, and ask it for:

- the call path to the failure, with `file:line`, and
- who owns the defect: the test file, `pytorch/{aten,torch,c10}`,
  `third_party/torch-xpu-ops/`, or a third party.

Wait for the subagent's result and use it directly; do not re-run the trace
yourself.

## Mode B — evidence-only

Conclude a cause ONLY when the evidence is self-sufficient — the `traceback` or
`error_message` names the owning file and states the defect — and cite what you
used.

Otherwise set `root_cause` to exactly:

```
insufficient information for root causing: no pytorch_folder provided and issue evidence is not self-sufficient
```

## Output

Never guess an owner or infer a `file:line` you did not read. Record `trace_mode`
and `root_cause` in at most 2 lines.
