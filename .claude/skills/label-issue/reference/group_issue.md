# Grouping Rules

How to group an issue's failing cases into distinct cause-groups for Step 2 of
the label-issue skill. This file carries the full grouping logic; SKILL.md Step 2
defers to it.

Group `extract.json`'s failures by **cause**: two cases belong to the SAME group
when they share one underlying defect, and to DIFFERENT groups when their causes
are demonstrably distinct.

## Ordered ladder of keys

Decide grouping with an **ordered ladder of keys**. Apply the keys in the order
below; the FIRST key that cleanly separates (or unifies) the cases decides the
grouping — once a key resolves the split, STOP and do not consult the lower keys.
Each lower key only breaks ties the keys above it left ambiguous.

1. **Error message** — the normalized error class + message text (drop
   run-specific noise: addresses, device ids, file paths, line numbers,
   tolerances/deltas, timestamps). Same normalized message -> same group; a
   clearly different message -> different group. If this alone resolves it, stop.
2. **Dtype** — when the message is identical/generic, split by the dtype that
   drives the failure (e.g. one dtype hits an unsupported-dtype/`NotImplementedError`
   path while another fails a tolerance check). Same failing dtype -> same group.
3. **Op / kernel** — the failing operator or kernel named in the traceback
   (e.g. `addmm` vs `layer_norm`). Different originating op -> different group.
4. **Parameters** — non-shape parametrization (flags, modes, reduction type,
   `keepdim`, backend, memory format) when it changes the code path taken.
5. **Tensor shape** — split by shape only when the shape itself is the cause
   (e.g. one shape overflows/goes out of bounds while another passes or fails
   differently).

## Signal vs noise

Keys 2–5 are grouping SIGNALS only when they are the distinguishing cause; when
the error is otherwise identical across them, they are noise and do NOT split the
group. So two failures with the same normalized message, dtype, op, params, and
causal shape are ONE group even across different test functions, files, or
parametrizations. Conversely, split a generic signature (e.g. a bare
`AssertionError`) only when a key on the ladder shows a demonstrably different
cause.

Case count alone is NOT the signal, and neither is the number of test functions:
one unimplemented-dtype error reported across many cases is ONE group.

## Outcome

- **One group** -> no split signal.
- **Two or more groups** -> the issue mixes distinct causes. Emit the
  split-recommendation label from `categories.triage` (match its `evidence`;
  read the name from the JSON). Record a one-line signature per group for the
  output. This skill NEVER splits, files sub-issues, or edits anything.
