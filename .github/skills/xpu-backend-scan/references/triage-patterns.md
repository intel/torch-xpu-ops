# Triage Patterns

Non-obvious patterns where static evidence can mislead. When you see these, add the noted context to your finding.

| Pattern | Why it's misleading | What to check instead |
|---------|--------------------|-----------------------|
| No dedicated XPU kernel file | May use structured delegate, composite, or shared path | Backend YAML, `structured_delegate` target, composite dispatch |
| XPU kernel shorter than CUDA | Conciseness ≠ defect — may use shared helpers | Helper definitions, actual validation sites |
| Helper call sites differ | Helpers may be functionally equivalent | Read the helper definitions, not just call sites |
| Structured wrapper lacks local registration | May inherit delegate coverage | `structured_delegate: foo.out` — judge by target |
| No backend-local backward symbol | Generic autograd may cover | `derivatives.yaml`, shared formulas |
| CUDA entry in YAML without XPU | CUDA presence does not prove XPU absence | Check all 5 coverage signal types |
