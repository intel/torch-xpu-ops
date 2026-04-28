# False-Positive Patterns

Use this reference when a finding looks suspicious but the evidence still feels
too indirect or too dependent on file shape, test metadata, or stale wording.

## Quick Rule

If the current claim does not yet identify a user-visible contract difference or
a concrete missing runtime path, treat it as a downgrade candidate first.

## Common False-Positive Families

- No dedicated XPU kernel file:
  Support may come from structured, shared, or generic paths.
  Verify instead: backend YAML, delegate, composite, decomp, or shared helper coverage.
  Typical outcome: `LIKELY_OK` or reroute.
- XPU kernel is shorter than CUDA:
  Conciseness is not a semantic defect.
  Verify instead: helper definitions, wrapper behavior, and validation sites.
  Typical outcome: `LIKELY_OK`.
- oneDNN or oneMKL vs cuDNN or cuBLAS:
  Vendor library choice is not itself a parity bug.
  Verify instead: user-visible behavior and validation.
  Typical outcome: `LIKELY_OK`.
- Helper call sites differ:
  Call-site-only review often misses helper equivalence.
  Verify instead: actual helper definitions.
  Typical outcome: downgrade or stop.
- Structured wrapper lacks local registration:
  The wrapper may inherit delegate coverage.
  Verify instead: the `structured_delegate` target and generated path.
  Typical outcome: `LIKELY_OK`.
- Missing backend-local backward symbol:
  Generic autograd or a wrapper formula may already cover it.
  Verify instead: `derivatives.yaml`, shared helpers, and decomposition coverage.
  Typical outcome: `LIKELY_OK`.
- Skip or xfail is the main evidence:
  Test metadata is only supporting evidence.
  Verify instead: runtime coverage and source-backed behavior.
  Typical outcome: `NEEDS_HUMAN_REVIEW` or `LIKELY_OK`.
- Missing XPU-specific test:
  Absence of a test is not absence of support.
  Verify instead: actual registration and implementation.
  Typical outcome: `LIKELY_OK`.
- CUDA entry exists in YAML:
  CUDA presence alone does not prove missing XPU support.
  Verify instead: fallback, composite, delegate, decomp, and backend YAML.
  Typical outcome: reroute or downgrade.
- Fallback counted as missing implementation:
  Fallback is runtime coverage, not Goal 3 absence.
  Verify instead: whether fallback itself is a Goal 1 problem.
  Typical outcome: reroute to Goal 1.
- Historical detail says missing support:
  Old text may contradict current facts.
  Verify instead: current source-backed dispatch and implementation.
  Typical outcome: downgrade stale record.
- Waived NVIDIA-specific op looks unsupported on XPU:
  Justified exclusions are not bugs.
  Verify instead: waiver categories and examples.
  Typical outcome: `WAIVED`.

## How To Restate Weak Claims Correctly

Prefer these restatements:

- Instead of “XPU implementation is missing,” say “No XPU-local file found yet; must still exclude delegate, composite, fallback, and shared-path coverage.”
- Instead of “Parity gap because CUDA uses cuDNN,” say “Different vendor library observed; no user-visible contract difference established yet.”
- Instead of “Backward missing,” say “No backend-local backward symbol found yet; must still check generic autograd and shared wrapper formulas.”
- Instead of “Goal 3 missing implementation,” say “Fallback exists, so the real question is whether CPU fallback is an XPU defect under Goal 1.”

## Escalate Only When One Of These Becomes True

- the valid input space changes
- a parameter is ignored or behaves differently
- result values, warnings, or error paths diverge
- no callable XPU runtime path exists after all exclusions are checked
- a concrete XPU-side validation, alias, version-bump, numeric, or dispatch
  defect is visible in the code

## Review Habits That Reduce False Positives

- read helper definitions before comparing call sites
- compare exact schemas, not just base op names
- record negative evidence alongside bug-looking evidence
- treat test metadata as supporting evidence only
- prefer current source-backed facts over stale batch wording