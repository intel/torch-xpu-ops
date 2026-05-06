---
name: pytorch-ci-triage
description: Analyze CI failures on a public pytorch PR and fix if related
---

# CI Failure Triage

## When to Use
When CI checks are failing on a public PR to pytorch/pytorch.

## Process
1. **Categorize each failure:**
   - **Related to your change** — fix it
   - **Pre-existing / flaky** — note it but don't change code
   - **Infrastructure** — ignore (retry will fix)

2. **For related failures:**
   - Read the failure logs
   - Identify the root cause
   - Make the fix
   - Ensure the fix doesn't break other tests

3. **For unrelated failures:**
   - Add a PR comment noting it's pre-existing
   - Don't waste time trying to fix others' issues

## Common Patterns
- Linting failures → fix formatting
- Type check failures → fix type annotations
- Test failures → check if your change broke an assumption
- Build failures → check imports and dependencies
