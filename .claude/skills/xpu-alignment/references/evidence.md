# Evidence and Review

Use this reference only for decisions that depend on evidence quality: semantic
candidate eligibility, faithful XPU reproducer construction or result
classification, and independent review. It prevents weak source signals, setup
failures, and provisional scan results from being promoted into unsupported XPU
trackers.

Do not use it for deterministic enumeration or pagination, artifact schemas,
stage ownership and digests, workflow gating or publishing, or implementation of
the resulting fix. Use the
[automation contract](automation-contract.md) for automation mechanics.

## Candidate set

For `[start, end)` in UTC, cover issues created, PRs created or merged, and commits
reachable from the `pytorch/pytorch` default branch whose committer timestamp is
in the interval. Ordinary comments, labels, and other updates to older objects do
not create candidates. Fetch linked older objects when they provide context.

An observed inventory object is not yet an alignment validation candidate. If
the upstream issue body or reproducer, PR tests or diff, or commit diff shows
that its primary scope is independent XPU work already tracked or implemented
upstream, reject it with `already-xpu-scoped` in the reason. Do not create a
reproducer, review it, or open a parallel `torch-xpu-ops` issue. A title, label,
or XPU mention by itself is only a prioritization signal and is not sufficient
evidence for this rejection.

Alignment validation instead targets generic behavior and work originating in
CPU, CUDA, ROCm, MPS, or any other backend when XPU parity is not already
addressed. Shared or multi-backend work remains eligible even when XPU is named
as one affected backend. For an explicitly linked issue/PR/commit chain, validate
one canonical object at most. Reject the other objects with `duplicate-chain` in
the free-text reason and name the canonical inventory id.

In automation, the deterministic collector owns pagination and the raw inventory.
Treat every observed inventory object as in scope. Its collection is partial when
an API quota, authentication error, timeout, or endpoint failure prevents a source
from reaching the window boundary. Preserve its per-source page count, fetched
count, last cursor, rate-limit state, and error instead of silently narrowing the
set. Partial inventory may be analyzed, and fully covered units may be published
after independent review; the incomplete scope must still be reported and must
not be mistaken for a complete scan.

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
5. current source, fix PR merge state, and canonical tracker state;
6. whether a claimed fix is present in the tested build and passes the same check.

A non-null merge timestamp (`mergedAt` in GraphQL and `gh pr view`, or
`merged_at` in the REST API) establishes that a PR was merged directly. Its
absence does not establish that the change is unmerged: check the frozen
default-branch snapshot for a linked commit or an equivalent source change.
Treat the behavior as landed when that evidence is reachable from the default
branch, even if the PR is closed without a merge timestamp. A closed state alone
establishes neither outcome.

An open or genuinely unlanded PR may justify `needs-xpu-fix` only when evidence
independently establishes a defect in the current XPU implementation whose fix
does not depend on that PR landing. If independent XPU parity work would be
required only after the proposed upstream behavior lands, use `track-upstream`,
set `implementation_repository` to `intel/torch-xpu-ops`, and emit no payload.
Treat a closed, unlanded proposal with no independent current XPU defect as
`non-issue`.

Before allowing a new issue, search `pytorch/pytorch` for an issue or PR that
explicitly owns the independent XPU work and search `intel/torch-xpu-ops` for a
canonical tracker. The current source, a generic related issue, or an XPU mention
alone does not establish upstream ownership. When upstream explicitly owns the
XPU work, use `track-upstream`, set `implementation_repository` to
`pytorch/pytorch`, and emit no payload. When an existing ops tracker covers the
work, record its URL as `canonical_tracker`; do not create a new payload or
automatically comment on the existing tracker.

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
