# Triage Governance

This reference is for post-scan review, not default forward scanning.

Use it when:
- reviewing an existing batch of findings
- deciding whether a recurring family is a true bug family or false-positive
  family
- deciding whether a lesson belongs only in skill text or is stable enough to
  promote into deterministic automation

## Governance Boundary

Keep three layers distinct:

### Skill Or KB Layer

Use this layer for:
- reading order
- recurring misread patterns
- negative and positive examples
- cautions that still require judgment

This layer improves future scan quality but does not automatically repair old
historical output.

### Indexing Or Fact-Layer Automation

Promote a lesson into deterministic fact extraction only when it is a stable
recognition problem, for example:
- delegate inheritance
- wrapper inheritance
- structured coverage detection
- fallback detection
- source-backed implementation detection

### Result-Normalization Automation

Promote a lesson into normalization only when:
- the condition can be expressed as a stable boolean combination
- current source-backed facts clearly overturn stale historical text
- the rule applies to a family rather than a one-off manual judgment

## What Should Not Become Automation

Do not promote the following into deterministic rules:

- single-op manual judgments with no family pattern
- conclusions based only on intuition or prose detail text
- anything that still requires runtime validation
- ambiguous cases where positive and negative evidence still conflict

The failure mode is permanent suppression of future real bugs.

## Evidence Priority

When reviewing existing findings, rank evidence from strongest to weakest:

1. authoritative static facts
2. actual helper, kernel, wrapper, and registration definitions
3. fallback, composite, decomposition, structured delegate, or shared-path
   evidence
4. tests, skip, xfail, or OpInfo
5. finding detail text
6. implementation-shape intuition

## Review Workflow

### 1. Review The Tail Before Changing Rules

Inspect non-OK findings first. Do not suppress more noise until the dominant
families are understood.

### 2. Join Historical Findings With Current Static Facts

Useful fields include:
- `has_cuda`
- `has_xpu`
- `verified_xpu_dispatch`
- `in_xpu_source_tree`
- `in_fallback`
- `has_composite`
- `has_decomp_registration`
- `structured_delegate`

The point is to identify where stale text conflicts with current facts.

### 3. Group By Family Before Reading Row By Row

Cluster by:
- goal or sub-goal
- verdict
- repeated keywords or patterns
- shared helper, wrapper, or kernel family

Judge family-level truth first, then repair row-level labeling.

### 4. Use Detail Text As A Pointer, Not As Ground Truth

Let detail text tell you what helper or wrapper to inspect next. Do not trust it
as the final verdict source.

### 5. Look For Negative Evidence First

Before keeping a bug, ask whether one of these already clears or downgrades it:
- fallback exists
- composite or decomposition already covers it
- helper definitions are equivalent
- generic autograd already covers backward behavior
- CUDA itself lacks a meaningful implementation

### 6. Preserve Only User-Visible Contract Differences

Keep parity or missing-support findings only when they affect:
- legal input space
- parameter semantics
- dtype, device, or layout support
- backward availability
- result behavior, warnings, or error paths

Optimization-only or implementation-shape differences should usually be
downgraded first.

## Goal-Specific Review Heuristics

### Goal 1 Usually Real

- CPU fallback or silent device transfer
- source-backed XPU code with broken dispatch connection
- inplace or foreach paths missing version-bump behavior
- missing XPU guards, alias checks, or layout checks
- clear numeric or atomic correctness defects

### Goal 1 Often False Positive

- clearing a row only because a sibling or shared helper exists
- attaching the real bug to the wrong variant
- treating a shared wrapper difference as XPU-only

### Goal 2 Usually Real

- parameter contracts differ
- valid CUDA configurations are rejected on XPU
- backward coverage is genuinely absent on XPU
- a missing branch changes public behavior or exposed capability

### Goal 2 Often Over-Called

- vendor-library or launch-shape differences
- helper call-site review without reading helper definitions
- test or xfail evidence without source-backed semantic divergence

One-sentence standard: keep a Goal 2 finding only when the same user call would
produce a different contract on XPU.

### Goal 3 Usually Real

- CUDA has a usable path
- XPU has no dispatch or usable implementation
- no fallback exists
- no shared, composite, or decomposition path exists
- no waiver applies

### Goal 3 Often Too Weak

- CUDA itself is stub, NYI, or error-only
- XPU already has fallback
- current source-backed facts overturn stale missing-support claims

## Promotion Checklist

Before turning a lesson into automation, ask:

1. Does the conclusion come from authoritative static facts?
2. Will the condition remain stable on the next scan?
3. Is this a family-level root cause rather than a one-off exception?
4. Could codifying it suppress a future real bug?
5. If left only in skill text, will the same family recur expensively?

Promote only when the first four are strong and the fifth shows clear value.