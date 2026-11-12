# Grouping Rules

How to group an issue's failing cases into distinct cause-groups for Step 2 of
the label-issue skill. This file carries the full grouping logic; SKILL.md Step 2
defers to it.

Group `extract.json`'s failures by **cause**: two cases belong to the SAME group
when they share one underlying defect, and to DIFFERENT groups when their causes
are demonstrably distinct.

## How to group

Choose the grouping basis by what evidence the issue provides:

### A. Error message and traceback are present

When the cases carry an `error_message` (per-case, on each `test_cases[]` entry)
and/or the issue-level `traceback`, group by them FIRST. Use each case's
normalized `error_message` plus the traceback's failing frame (both are already
normalized during extraction (extract_issue.md): addresses, device ids, file paths, line numbers,
tolerances/deltas, and timestamps are dropped). Same `error_message` + same
failing frame -> same group; a clearly different message or frame -> different
group.

When the `error_message` is absent, generic, or identical across cases (e.g. a
bare `AssertionError`), break the tie with the facet keys below, in order — the
FIRST key that shows a demonstrably different cause decides the split, then STOP:

1. **Op / kernel** — the failing operator or kernel named in the traceback
   (e.g. `addmm` vs `layer_norm`). Different originating op -> different group.
2. **Dtype** — one dtype hits an unsupported-dtype/`NotImplementedError` path
   while another fails a tolerance check. Same failing dtype -> same group.
3. **Parameters** — non-shape parametrization (flags, modes, reduction type,
   `keepdim`, backend, memory format) when it changes the code path taken.
4. **Tensor shape** — split only when the shape itself is the cause (e.g. one
   shape overflows/goes out of bounds while another passes or fails differently).

### B. No error message or traceback

When the issue provides no `error_message` on any case and no `traceback`, decide whether a split
is warranted from the case facets directly — the **op / kernel**, **dtype**,
**parameters**, **tensor shape**, or any other distinguishing facet. Cases that
share the same facet signature are ONE group; a facet that points to a
demonstrably distinct cause splits them. If the facets are uniform (or the issue
describes a single concern), it is ONE group.

## Signal vs noise

The facet keys (dtype, op, parameters, shape, and other facets) are grouping
SIGNALS only when they are the distinguishing cause; when the failure is otherwise
identical across them, they are noise and do NOT split the group. So cases with
the same normalized signature across every facet are ONE group even across
different test functions, files, or parametrizations. Conversely, split only when
a facet shows a demonstrably different cause.

Case count alone is NOT the signal, and neither is the number of test functions:
one unimplemented-dtype error reported across many cases is ONE group.

## Outcome

- **One group** -> no split signal. This is common and expected; many issues
  describe a single cause.
- **Two or more groups** -> the issue mixes distinct causes. Emit the
  split-recommendation label from `categories.triage` (match its `evidence`;
  read the name from the JSON). Record a one-line signature per group for the
  output. This skill NEVER splits, files sub-issues, or edits anything.
