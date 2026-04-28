# Report Template

Use this template when returning a static triage finding, a `LIKELY_OK`
clearance, or a review-summary decision for an existing finding.

## Required Fields

Every finding should answer these fields explicitly:

- `goal`: 1, 2, or 3
- `sub_goal`: the most specific bucket that explains the claim
- `verdict`: one of the canonical labels from the main skill
- `confidence`: low, medium, or high
- `operator_surface`: exact schema, overload, or variant under review
- `user_visible_impact`: what contract changes for users, or why there is no
  such change
- `xpu_evidence`: the concrete XPU-side code or metadata evidence
- `peer_evidence`: CUDA, upstream, or shared-path evidence used as comparison
- `exclusion_check`: fallback, composite, decomp, delegate, waiver, and test
  metadata checks that were performed
- `next_action`: stop, downgrade, local repro, or human review

## Preferred Markdown Format

```md
## Finding Summary

- Goal: <1 | 2 | 3>
- Sub-goal: <short bucket>
- Verdict: <LIKELY_XPU_BUG | PARITY_GAP | MISSING_XPU_IMPL | LIKELY_OK | NEEDS_HUMAN_REVIEW | WAIVED>
- Confidence: <low | medium | high>
- Operator surface: <exact schema or overload>

## Why This Matters

<one or two sentences describing the user-visible impact, or explicitly stating
that the difference is not user-visible>

## XPU Evidence

- <file or symbol>
- <what the code does or does not do>

## Peer Evidence

- <CUDA, upstream, wrapper, composite, or delegate evidence>
- <how it differs or why it is equivalent>

## Exclusion Check

- Waiver: <applies | not applicable | ambiguous>
- Fallback: <present | absent>
- Composite or decomp: <present | absent>
- Structured or delegate path: <present | absent>
- Test metadata: <supporting only | not used | contradicts>

## Next Action

- <stop with LIKELY_OK>
- <keep as likely bug>
- <downgrade to NEEDS_HUMAN_REVIEW>
- <hand off for local repro>
```

## Short Form For Clear Non-Bugs

Use this shorter form when the evidence clearly clears the finding:

```md
- Verdict: LIKELY_OK
- Operator surface: <schema>
- Reason: <shared path, fallback redirect, composite coverage, or non-semantic difference>
- Key evidence: <one or two file-grounded bullets>
- Next action: stop
```

## Short Form For Ambiguous Cases

Use this when the evidence is mixed and a local repro or deeper review is still
needed:

```md
- Verdict: NEEDS_HUMAN_REVIEW
- Operator surface: <schema>
- Blocking uncertainty: <what remains unresolved>
- Positive evidence: <what still looks concerning>
- Negative evidence: <what already weakens the claim>
- Next action: <local repro | inspect helper | inspect delegate | inspect wrapper>
```

## Writing Rules

- Name the exact schema or overload, not only the base op.
- State the user-visible impact before discussing implementation shape.
- Record negative evidence, not only the bug-looking evidence.
- Do not use test metadata as the only runtime-coverage argument.
- Do not call a finding confirmed unless a real local repro exists.