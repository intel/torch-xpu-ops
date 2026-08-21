# Evidence and Review

Read this reference when collecting a time window, adapting or running a
reproducer, classifying local evidence, or independently reviewing a candidate.

## Candidate set

For `[start, end)` in UTC, cover issues created, PRs created or merged, and commits
reachable from the `pytorch/pytorch` default branch whose committer timestamp is
in the interval. Ordinary comments, labels, and other updates to older objects do
not create candidates. Fetch linked older objects when they provide context.

Use pagination or split queries as needed. Collection is incomplete when an API
cap, quota, authentication error, or ambiguous endpoint prevents exhausting a
source. Record the error instead of silently narrowing the set.

Titles and labels help prioritize reading but do not decide relevance. Inspect
the body, changed code, tests, linked work, or diff whenever it could change the
decision. Clear documentation/infrastructure-only changes, platform-exclusive
behavior, nonfunctional refactors, and duplicate issue/PR/commit chains can be
rejected with a concrete explanation.

## Faithful XPU evidence

Prefer the upstream reproducer or regression test. When adapting another backend,
change only device mechanics unless evidence requires more. Preserve supported
inputs, shapes, strides, dtype, mode, seed, and oracle. Reuse identical initialized
inputs for comparisons and use upstream/dtype-appropriate tolerances.

Before execution, identify:

- the upstream oracle and observation stage;
- the XPU operation or compiler path being exercised;
- evidence that the target path, rather than setup data, ran on XPU;
- how the script distinguishes the same failure, a related failure, and success;
- a bounded timeout and a credential-scrubbed execution command.

Retain the exact script and raw stdout/stderr. A broad exception or substring
match does not establish a result by itself. Treat timeout, crash, or setup
failure as actionable only when target-stage evidence makes the signature
defensible.

Use these local results:

| Result | Meaning |
|---|---|
| `confirmed` | The target XPU path exhibited the upstream oracle/signature. |
| `related-failure` | The target ran and exposed a different actionable defect. |
| `not-reproduced` | The faithful target ran and the upstream oracle passed. |
| `blocked-env` | Runtime, dependency, device, or topology prevented validation. |
| `blocked-platform` | XPU has no corresponding target path. |
| `blocked-fetch` | Required public source material could not be retrieved. |
| `blocked-script-error` | Setup/script failure prevented a defensible result. |
| `needs-performance-harness` | A performance claim lacks a valid comparison. |

## Independent review

The reviewer did not produce the scan. For every `confirmed` or
`related-failure` candidate, independently check:

1. reproducer fidelity and oracle;
2. target-path XPU execution and observed signature;
3. whether the behavior is a defect rather than unsupported or expected behavior;
4. whether an independent XPU change is required;
5. current source, fix PR, and canonical tracker state;
6. whether a claimed fix is present in the tested build and passes the same check.

Search `intel/torch-xpu-ops` for a canonical tracker before allowing a new issue.
Use exactly one verdict:

- `needs-xpu-fix`
- `track-upstream`
- `fixed`
- `non-issue`
- `duplicate`
- `verification-gap`

Prefer `verification-gap` over a forced conclusion. A `needs-xpu-fix` payload has
a `[xpu-alignment]` title, exactly the `ai_generated` label, upstream source and
scan window, observed XPU behavior, target-path evidence, a copy-pasteable
reproducer, relevant output, environment identity, and ownership rationale.
