# Reproducer and Execution Evidence

## Construct a faithful check

Prefer a reproducer or regression test already present in the upstream issue,
PR, or fix. When adapting another backend to XPU, change only device-specific
mechanics unless evidence requires more:

- map device placement and synchronization to XPU;
- preserve supported inputs, shapes, strides, dtype, mode, seed, and oracle;
- reuse identical initialized inputs for comparisons;
- for compiler cases, identify the eager baseline and compiled target stage;
- explain every scenario change that could affect the result.

Do not invent a generic test when the source lacks enough behavioral context.
Reject it as `insufficient-repro-context` or classify a performance-only claim as
`needs-performance-harness` when appropriate.

## Semantic precheck

Before a script becomes executable, record:

- upstream oracle and the stage at which it is observed;
- target XPU operation/path and inputs;
- the proof that the target, rather than setup data, ran on XPU;
- how the script distinguishes the same failure, a related failure, and success;
- script SHA-256 and `approved`, `rework`, or `reject` status.

Approval requires syntactically valid code, initialized/reproducible data, a
faithful oracle, target-path proof, bounded work, and no embedded credentials or
unrelated external actions. Broad exception or substring matching cannot alone
establish `confirmed`.

An independent pre-execution semantic check is useful when the runtime provides
one, but it is not a security boundary and is not a substitute for isolated
execution. Final review must be independent from the scan producer.

## Execution boundary

Treat every generated script as untrusted even after semantic precheck.

In automation:

- the credential-bearing agent writes an execution plan but never runs a repro;
- a later non-agent step runs only plan entries marked approved and whose bytes
  match the recorded SHA-256;
- that step receives no GitHub, model-provider, cloud, or publishing credential;
- use a fixed caller-provided Python executable, no shell evaluation, a bounded
  timeout, a controlled working directory, a non-root identity that cannot write
  scan-owned files, and one fresh process/scratch home at a time;
- retain stdout, stderr, exit code/signal, timeout state, duration, and script
  digest in machine-readable execution results plus raw logs;
- do not trust a script's self-reported `RESULT:` without reconciling it against
  the planned oracle and target-path evidence.

Interactive execution is allowed only on a disposable XPU environment without
valuable credentials. If that condition is not already established, ask before
running fetched or generated code.

For an abnormal termination claimed as a target bug, reproduce the same
target-stage signature in a fresh process when practical. A timeout or crash that
cannot be distinguished from setup failure is `blocked-script-error`.
