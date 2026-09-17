# Infra or product

Per group, decide whether the failures describe a bug in the code under test or
a machine that misbehaved. Only the first is filed; for anything else set
`file: false` and give the reason, so the night's report still names the
failures.

## The trap

The messages that look most like infrastructure are the ones that say least:

```
UR_RESULT_ERROR_DEVICE_LOST
XPU out of memory. Tried to allocate 2.00 GiB
RuntimeError: Native API failed
```

None of these carries an operator, a shape or a dtype, so none of them can tell
you what caused it. A test allocating far too much memory produces exactly the
same string as a runner whose GPU fell off the bus, and so does a kernel that
hangs the device. Reading the message alone gets this wrong in both directions.

## What actually separates them

**Breadth.** A machine that loses its GPU or fills its disk does not stop at
one test file. The same message across many unrelated files in one night is the
machine; confined to one file, or to one operator across a couple of files, it
is that code. More than about five unrelated test files is already more than a
product bug usually manages.

**Coincidence with something specific.** Failures that all touch one operator,
one dtype, one kernel or one recently changed area point at that thing,
whatever the message sounds like.

**The machine itself.** `run.json.runners` gives the machine per UT job. One
error on two different machines argues against a machine fault; the same error
on one while the other job is clean argues for it.

**What the traceback shows.** A traceback ending inside a test's own allocation
or a specific kernel is product. One ending in driver or runtime teardown with
nothing above it is weak evidence either way.

## Which way to err

Nothing checks this decision after you, so weigh the two mistakes rather than
trying to be right.

Call it infrastructure when it was a product bug and no issue is filed: the
cases keep running and keep appearing in the nightly report, where a human can
still find them. Call it a product bug when it was the machine and an issue is
filed: the cases are muted for a fault that will clear itself, and the tests
stay dark until somebody closes the issue.

The first is recoverable and the second is not, which points one way: **when
the evidence does not settle it, do not file.** Say so in the draft's `reason`
instead.

Two corollaries. Use `file: true` on a wide denylisted error only when you have
a specific reason the failures belong together as code - a shared operator, a
shared kernel, a recent change in that area - and put that reason in the
summary. And never mark something infrastructure because it is hard to triage:
that mutes nothing, but it does mean nobody looks.
