# Infra or product

Per group, decide whether the failures describe a bug in the code under test
or a machine that misbehaved. Set `infra.verdict` to `infra`, `product` or
`unsure`.

## The trap

The messages that look most like infrastructure are the ones that say least:

```
UR_RESULT_ERROR_DEVICE_LOST
XPU out of memory. Tried to allocate 2.00 GiB
RuntimeError: Native API failed
```

A device lost or an out-of-memory carries no operator, no shape and no dtype,
so it cannot tell you what caused it. A test allocating far too much memory
produces exactly the same message as a runner whose GPU fell off the bus. So
does a kernel that hangs the device. Reading the message alone gets this wrong
in both directions, and getting it wrong towards `infra` means a real
regression is never filed and repeats silently every night.

## What actually separates them

**Breadth.** A machine that loses its GPU or fills its disk does not stop at
one test file. The same message across many unrelated files in one night is the
machine. The same message confined to one file, or to one operator across a
couple of files, is that code.

**Coincidence with something specific.** Failures that all touch one operator,
one dtype, one kernel or one recently changed area point at that thing, whatever
the message sounds like.

**The machine itself.** `run.json` gives `runners` per leg. One error appearing
on two different machines argues against a machine fault. The same error on one
machine while the other leg is clean argues for it.

**The rest of the run.** `run.json.report.categories` shows whether each
category completed. A run that finished everything else and produced a handful
of device errors is a different situation from one that stopped early.

**What the traceback shows.** A traceback that ends inside a test's own
allocation or a specific kernel is product. One that ends in driver or runtime
teardown with nothing above it is weaker evidence either way.

## What each verdict costs

Nothing checks this decision after you, so weigh the two errors against each
other rather than trying to be right.

| You decide | If you are wrong |
|---|---|
| `infra`, and it was a product bug | no issue is filed, the cases keep running and keep appearing in the nightly report. A human can still see them. Recoverable. |
| `product`, and it was the machine | an issue is filed and the cases are muted, for a fault that will clear itself. The tests stay dark until someone closes it. |

They are not symmetric, and the asymmetry points one way: **when the evidence
does not settle it, do not file.** Say so in your final message instead, so the
night's report still names the failures.

`run.json.limits.infra_max_test_files` is the breadth figure the deterministic
side used to use, and it is still a reasonable anchor: one denylisted message
reaching more test files than that in a single night is far more likely the
machine than a bug. Treat it as a strong prior, not a rule - a shared operator
or a recent change in one area can outweigh it, and a narrow error on a runner
that is failing everything else can go the other way.

## Practical guidance

- Default to `unsure` when you are unsure. It is not a wasted answer: it leaves
  the deterministic rule in charge, which is the right outcome when you have
  nothing to add.
- Use `product` on a wide denylisted error only when you have a specific reason
  the failures belong together as code - a shared operator, a shared kernel, a
  recent change in that area - and put that reason in `infra.reason`.
- Use `infra` on a narrow error when the evidence points at the machine despite
  the small blast radius, for example the same device error appearing across
  categories on one runner while the other runner is clean.
- Never call something `infra` because it is hard to triage. That decision
  mutes nothing, but it does mean nobody looks.

## Withdrawing a whole leg, or the whole run

Some evidence is about the run rather than about any one group: a leg that
produced hundreds of device errors, a category that stopped part way, a machine
that failed everything it touched. `run.json.legs[leg].infra_pattern_ratio`
puts a number on the first of those, and
`run.json.report.categories` on the second.

When a leg looks like that, file nothing from it - including the
ordinary-looking failures, which are not trustworthy either on a night the
machine misbehaved. When only part of a leg looks wrong, judge group by group
instead; withdrawing the leg would also drop the failures that had nothing to
do with it.
