# Dtype Rules

The dtype axis records the data type(s) implicated in the failure. It is a
multi-label axis: emit one label per distinct dtype the failure is tied to, or
none when no dtype is implicated.

## Authoritative label + keyword source

All dtype labels and their keywords live in `categories.dtype` of
`proposed_labels.json`. Read them from there; never hard-code a dtype spelling in
this file. Each label carries a `code` (the bare dtype name) and a `keywords`
array.

Match those keywords against `lowercase(title + " " + body + " " + traceback)`
for the Step 1.5 analyzed case, excluding the `## Versions` / `Collecting
environment` dump.

## Decision rules

- Emit a dtype label only when the dtype is part of the failure signature: it
  appears in the failing test's parametrization, the error message, the
  traceback, or the reproduce command. A dtype mentioned only incidentally (e.g.
  a package name in the env dump, or an unrelated code snippet) does not qualify.
- Prefer the dtype carried by structured fields first: for a unit-test case use
  the dtype parametrization suffix on the test case name when present; for an E2E
  entry use `extract.json`'s E2E `dtype` field. Fall back to keyword matching on
  the text only when no structured dtype is available.
- The axis is additive. When a single analyzed case exercises several dtypes
  (e.g. a promotion test over `float32` and `float64`), emit one row per dtype.
- AMP labels (`dtype: amp_bf16`, `dtype: amp_fp16`) are distinct from the plain
  precision labels. Emit an AMP label only when autocast/AMP is explicit; do not
  also emit the plain `dtype: bfloat16` / `dtype: float16` unless the failure
  independently implicates the non-AMP dtype.
- If no dtype keyword matches on real evidence, emit no dtype row. An empty dtype
  axis is a valid, common outcome (many failures are dtype-agnostic).

## Emit

Emit each matched label verbatim from `categories.dtype[].name` (e.g.
`dtype: bfloat16`, `dtype: float32`). Copy the label column from the JSON; never
emit the bare `code`.
