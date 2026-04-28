# False-Positive Patterns

Use this reference when a finding looks suspicious but the evidence still feels
too indirect or too dependent on file shape, test metadata, or stale wording.

## Quick Rule

If the current claim does not yet identify a user-visible contract difference or
a concrete missing runtime path, treat it as a downgrade candidate first.

## Common False-Positive Families

| Pattern | Why It Is Weak | Verify Instead | Typical Outcome |
|---|---|---|---|
| No dedicated XPU kernel file | Support may come from structured, shared, or generic paths | backend YAML, delegate, composite, decomp, shared helper | `LIKELY_OK` or reroute |
| XPU kernel is shorter than CUDA | Conciseness is not a semantic defect | helper definitions, wrapper behavior, validation sites | `LIKELY_OK` |
| oneDNN or oneMKL vs cuDNN or cuBLAS | Vendor library choice is not itself a parity bug | user-visible behavior and validation | `LIKELY_OK` |
| Helper call sites differ | call-site-only review often misses helper equivalence | actual helper definitions | downgrade or stop |
| Structured wrapper lacks local registration | wrapper may inherit delegate coverage | `structured_delegate` target and generated path | `LIKELY_OK` |
| Missing backend-local backward symbol | generic autograd or wrapper formula may already cover it | `derivatives.yaml`, shared helper, decomposition | `LIKELY_OK` |
| Skip or xfail is the main evidence | test metadata is only supporting evidence | runtime coverage and source-backed behavior | `NEEDS_HUMAN_REVIEW` or `LIKELY_OK` |
| Missing XPU-specific test | absence of a test is not absence of support | actual registration and implementation | `LIKELY_OK` |
| CUDA entry exists in YAML | CUDA presence alone does not prove missing XPU support | fallback, composite, delegate, decomp, backend YAML | reroute or downgrade |
| Fallback counted as missing implementation | fallback is runtime coverage, not Goal 3 absence | whether fallback itself is a Goal 1 problem | reroute to Goal 1 |
| Historical detail says missing support | old text may contradict current facts | current source-backed dispatch and implementation | downgrade stale record |
| Waived NVIDIA-specific op looks unsupported on XPU | justified exclusions are not bugs | waiver categories and examples | `WAIVED` |

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