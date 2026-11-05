# Backward Compatibility Guidelines For Torch XPU Ops Reviews

Treat any user-visible behavior change in `torch-xpu-ops` as backward-compatibility-sensitive until verified otherwise.

As a top-level principle, ANYTHING that changes ANY user-visible behavior is potentially BC-breaking. It will then need to be classified as a behavior not exercised in practice, a bug fix, or a voluntary BC-breaking change.

As a reviewer, you MUST be paranoid about any potentially BC-breaking change. One of your key roles is to identify such user-visible changes and ensure the PR author worked through the implications.

## What Counts As User-Visible Behavior

This includes more than public Python signatures. Flag changes in:

- Output values, shapes, strides, or layout behavior
- Error type, error timing, or error messages when tests or callers depend on them
- Dtype promotion, accumulation precision, or autocast behavior
- Fallback behavior versus explicit failure
- Determinism, synchronization side effects, or default execution path
- `functional`, `out=`, `inplace`, view, or backward semantics

## BC-Breaking Change Classification

### API Changes

| Change Type | BC Impact | Action Required |
|-------------|-----------|-----------------|
| Removing a public function/class | Breaking | Deprecation period required |
| Renaming a public API | Breaking | Deprecation period required |
| Changing function signature (removing/reordering args) | Breaking | Deprecation period required |
| Adding required arguments without defaults | Breaking | Add default value instead |
| Changing argument defaults | Potentially breaking | Document in release notes |
| Changing return type | Breaking | Deprecation period required |
| Removing or renaming private API | Potentially breaking | Validate no external usage |

### Behavioral Changes

| Change Type | BC Impact | Action Required |
|-------------|-----------|-----------------|
| Any user-visible behavior change | Potentially breaking | Flag to author for discussion |
| Raising new exceptions | Potentially breaking | Validate and document |
| Changing exception types | Potentially breaking | Document in release notes |
| Changing default device behavior | Breaking | Explicit migration |

## Review Rules

- A change can be BC-sensitive even when it looks like a bug fix
- If behavior changes, ask whether existing callers may rely on the old behavior
- Do not dismiss a semantic difference just because it only affects XPU users
- Treat silent fallback changes as BC-sensitive when they alter observed behavior or performance contracts

## Common BC Questions For XPU Reviews

1. Does the PR change which backend path is used for a public operator?
2. Does it change dtype promotion, accumulation precision, or mixed-precision behavior?
3. Does it alter output layout, contiguity expectations, or channels-last handling?
4. Does it raise a new error or reject inputs that previously ran?
5. Does it change `out=`, `inplace`, backward, or view behavior?
6. Does it make execution more synchronous in a way users can observe?

## Common BC Pitfalls

### Changing Function Signatures

**Bad:**
```python
# Before
def forward(self, x, y):
    ...
# After - breaks callers using positional args
def forward(self, x, z, y):
    ...
```

**Good:**
```python
# After - add new args at end with defaults
def forward(self, x, y, z=None):
    ...
```

### Changing Default Behavior

**Bad:**
```python
# Silently changing default from False to True
def function(x, normalize=True):  # Was normalize=False
    ...
```

**Good:**
```python
def function(x, normalize=None):
    if normalize is None:
        warnings.warn(
            "normalize default is changing from False to True in next release",
            FutureWarning,
        )
        normalize = False  # Keep old default during deprecation
    ...
```

### Changing Exception Types

**Bad:**
```python
# Users catching ValueError will miss the new exception
raise TypeError("...")  # Was ValueError
```

**Good:**
```python
# New type inherits from old
class NewError(ValueError):
    pass
raise NewError("...")
```

## When BC Breaks Are Acceptable

### With Proper Deprecation

1. Deprecation warning added for at least one release
2. Migration path documented
3. Release notes updated
4. Justified benefit outweighs the disruption

### Without Deprecation (Rare)

Immediate BC breaks may be acceptable for:
- Security vulnerabilities
- Serious bugs that make the API unusable
- APIs explicitly marked experimental/beta

## Acceptable Outcomes

- No user-visible behavior change after inspection
- A real bug fix with a clear correctness justification and tests
- An intentional behavior change with explicit rationale and acknowledgement of the compatibility impact

## Reviewer Guidance

- If you cannot verify the blast radius, mark the point as BC-sensitive or `not verified` rather than asserting it is safe
- Ask for release-note or migration discussion only when the change really alters user-visible behavior
- Keep BC findings concrete: say what changed, who could observe it, and what evidence you checked

## Review Checklist For BC

- [ ] No removed public APIs — or proper deprecation path exists
- [ ] No changed signatures — or new args have defaults
- [ ] No changed defaults — or deprecation warning added
- [ ] No changed return types/shapes — or migration path documented
- [ ] No changed exception types — or new types inherit from old
- [ ] Any other user-visible behavior change — give full list to author
