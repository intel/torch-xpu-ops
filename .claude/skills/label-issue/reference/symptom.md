# Symptom Rules

The symptom axis describes the *nature* of the failure, independent of which
component owns it. It is a multi-label axis: zero, one, or more symptom labels may
apply to one issue.

## Authoritative label + keyword source

All symptom labels and their keywords live in `categories.symptom` of
`proposed_labels.json`. Read them from there; never hard-code a symptom name or
keyword spelling in this file. The labels are:

`Accuracy`, `performance`, `regression`, `random`, `inference`, `training`.

Each label carries a `keywords` array in the JSON. Match those keywords against
`lowercase(title + " " + body + " " + traceback)` for the Step 1.5 analyzed case,
excluding the `## Versions` / `Collecting environment` dump.

## Decision rules

- Emit a symptom label only on direct evidence, not on an incidental keyword in
  the environment dump or an unrelated log line.
- The axis is additive. `Accuracy`, `regression`, and `training` can all apply to
  one issue (e.g. a training-phase accuracy regression). Emit every one that the
  evidence supports.
- `Accuracy` vs `performance` are mutually informative but not exclusive: a run
  can regress on both. Judge each independently from its own keywords/evidence.
- `regression` requires an explicit before/after signal (a version or commit where
  it passed and one where it fails), not merely the word "regression" in prose.
- `random` requires an explicit flaky/nondeterministic/intermittent signal.
- `inference` vs `training` come from the workload phase. When the issue is an E2E
  entry, prefer `extract.json`'s E2E `phase` field; otherwise judge from the
  keywords. Emit both only when the issue clearly reports both phases.
- If no symptom keyword matches on real evidence, emit no symptom row. An empty
  symptom axis is a valid, common outcome.

## Emit

Emit each matched label verbatim from `categories.symptom[].name` (e.g.
`Accuracy`, `performance`). These names are case-sensitive; copy them from the
JSON exactly.
