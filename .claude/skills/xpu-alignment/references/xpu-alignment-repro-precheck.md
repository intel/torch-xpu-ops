# Reproducer Pre-Execution Audit

This is the quality gate between repro construction and serial execution. Read it
for Step 2c. It prevents a generated script from manufacturing XPU evidence that
does not exercise the upstream behavior.

## Reviewer and scope

The orchestrating agent starts one fresh subagent that did not write the repros.
Select a runtime-advertised higher capability tier when available; otherwise use
exact parent-model inheritance. Do not infer a tier from a model name. The
reviewer receives the candidate ledger, fetched details, every generated repro,
and no expected bucket or answer key.

Audit every `deep_status: pass` script before it runs. Write
`reports/repro_precheck.md` with the reviewer basis and one section per script:

```md
## <candidate id> - <title>

- **Upstream oracle / target stage:** <what must happen and where>
- **Target XPU operation/path:** <concrete operation, inputs, compiler result, or dispatch>
- **XPU proof planned:** <the in-script assertion or emitted target result>
- **Verdict:** <approved | rework | reject>
- **Reason:** <specific evidence>
```

`rework` scripts must be changed and audited again by a fresh compliant reviewer.
Keep each prior result in the report. `reject` is allowed only after documenting
the consulted source material and why no faithful repro can be constructed; turn
it into a deep rejection with that reason, never a generic template script.

## Approval rules

Approve only when all of the following are true:

1. The upstream repro, test, or fix diff supplies enough concrete behavior to
   recreate the oracle. Missing faithful material is
   `insufficient-repro-context`; it is not permission to invent a generic test.
2. The script preserves supported inputs, shapes, dtypes, mode, and the upstream
   oracle. It seeds random data, avoids uninitialized values, and avoids changing
   the scenario unless the change is explicitly justified.
3. The operation under test, its target inputs, or its compiled target result is
   explicitly XPU. `torch.xpu.is_available()`, creating an unrelated XPU tensor,
   and printing a setup device do not count.
4. The script can distinguish the upstream oracle from a different failure.
   It may emit `RESULT: confirmed` only after observing the upstream failure at
   the expected stage. A different failure needs explicit independent semantics
   before it may emit `RESULT: related-failure`.
5. The script obtains a result from real execution. It must not classify a broad
   exception, title keyword, or generic error string as `confirmed`.
6. Python code is syntactically Python. External commands use
   `subprocess.run` with captured exit status and output; never embed shell
   commands as Python statements.
7. A performance-only claim has a valid benchmark and oracle. Otherwise reject it
   as `needs-performance-harness`.

For compiler cases, identify both the eager baseline and the compiled target
stage. An eager validation failure is not evidence of a compiler failure.

## Execution evidence

An approved script must print its target XPU proof next to the observed oracle,
then print one final `RESULT: <bucket>` line. The serial runner captures that
output verbatim. A crash, abort, or timeout has no self-reported result: the
parent records its exit code, signal, timeout status, and log, then classifies it
only if the target stage and XPU path are otherwise established.

After execution, the main agent cross-checks the precheck with the log. A missing
or contradictory target-path XPU proof is `blocked-script-error`, even if the
script printed `confirmed`.
