# Duplicate Rules

A duplicate is an existing issue that already tracks the same failure. Search
open issues in `intel/torch-xpu-ops` and `pytorch/pytorch` for a similar test
case and a similar error message or root cause.

## Search commands

Ask for `state`, `labels`, and `body` in the search itself. One enriched search
returns everything needed to classify a match, so no follow-up `gh issue view`
per candidate is required. Run the queries in parallel and always append
`is:issue` to exclude pull requests.

```bash
# Extract any GitHub issue URL already present in the error message first;
# that is the fastest and most reliable signal.
gh issue view <number> --repo=<repo> --json title,body,state,labels,url

# Otherwise search. Run these concurrently, then wait.
F='number,title,state,labels,url,body'
gh search issues "<full_test_name> is:issue" --repo=intel/torch-xpu-ops --limit=10 --json=$F &
gh search issues "<test_class> is:issue"     --repo=intel/torch-xpu-ops --limit=10 --json=$F &
gh search issues "<error_snippet> is:issue"  --repo=intel/torch-xpu-ops --limit=10 --json=$F &
gh search issues "<full_test_name> xpu is:issue" --repo=pytorch/pytorch --limit=10 --json=$F &
wait
```

Use `gh issue view` only to resolve a known issue number, such as one embedded
in a skip message or referenced by a candidate. Do not call it to enrich a
search hit that already returned `state` and `labels`.

## Search rules

- Search both repositories. A downstream `intel/torch-xpu-ops` issue and an
  upstream `pytorch/pytorch` issue for one failure legitimately coexist.
- Prefer open issues. Include a closed issue only when it carries `wontfix`
  or `not_target`, `not_target` is absorbed by the `wontfix` in new label
  definitions.
- Match on the test case first, then on the error message or the traced root
  cause. Two of the three signals must agree before claiming a duplicate, with
  one exception: a literal body match on the full test name stands alone (see
  below).
- String-match the returned `body` for the literal full test name, and for a
  `,<test_class>,<full_test_name>` line. Skip-tracking issues list their cases
  verbatim in the body, so a body match is authoritative even when the title
  looks unrelated. This is the **single-signal exception** to the two-of-three
  rule: a verbatim full-test-name hit in the body is sufficient on its own,
  because such a list is an explicit enumeration of tracked cases, not a
  coincidental text overlap. It does not require a matching error or root cause.
  The exception is deliberately narrow — it needs the *full* test name, so a
  bare test-class or operator substring never qualifies.
- Run one search set for the representative case selected in Step 3. When the issue
  reports several cases, the other entries are out of scope for this run.
- Never report the source issue as its own duplicate. Drop any match whose
  repository and issue number equal the source issue's.
- The search must actually succeed. If every query fails or returns nothing
  parseable, omit the `duplicate` row and emit the one-line
  `Duplicate search: failed (<reason>)` note per
  [output_format.md](output_format.md) — never silently treat a failed search
  the same as a clean `has_duplicate: false` result.

## Relevance and evidence

| Evidence | `relevance` |
|---|---|
| Same test case and the same error message, traceback, or traced root cause, or a literal body match on the full test name | `HIGH` |
| Same test class, file, or operator with a similar error, or the same root cause reached from a different test | `MEDIUM` |
| Only the file, operator, or a generic error string is shared | `LOW` |

Cite the matching issue URL and state which signals agreed, or cite the
single-signal body-match exception when that is what qualified the match. A
shared operator name or a generic `RuntimeError` alone is never sufficient.

## Recommended action

| Condition | `recommended_action` |
|---|---|
| `HIGH` relevance in the **same** repository as the source issue | `close_as_duplicate` |
| `HIGH` or `MEDIUM` relevance in the **other** repository | `cross_link` |
| `MEDIUM` or `LOW` relevance with overlapping but wider scope, such as a class-level tracker | `merge_context` |

A cross-repository match is always `cross_link`. Never recommend closing an
issue in one repository merely because a counterpart exists in the other.

## Inherited exclusion labels

| Source of `wontfix` | Effect |
|---|---|
| The source issue's own labels | The flag is `true`; the source is `own_labels`. |
| A `HIGH` or `MEDIUM` relevance duplicate's labels | The flag is `true`; the source is `duplicate:<repo>#<number>`. |
| Neither, or only a `LOW` relevance duplicate | The flag is `false`. |

`wontfix` absorbs the old `not_target`; treat a legacy `not_target` label on a
matched issue as `wontfix`. An inherited flag emits the `wontfix` label exactly as
an own label would. Cite the duplicate's URL whenever a flag is inherited.

## Minimum decision checks

1. Confirm that both repositories were searched and that the queries returned
   parseable results.
2. Confirm that every search requested `state` and `labels`, so no per-candidate
   `gh issue view` was needed.
3. Confirm that self-exclusion was applied.
4. Confirm that each reported duplicate cites a URL and the agreeing signals.
5. Confirm that a cross-repository match was not marked `close_as_duplicate`.
6. Confirm that an inherited `wontfix` came from a `HIGH` or
   `MEDIUM` relevance duplicate.
