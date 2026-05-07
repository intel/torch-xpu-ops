---
name: pytorch-review-fix
description: >
  Address code review feedback on a PyTorch XPU fix PR.
  Used by the private_review agent after reviewer requests changes.
---

# Review Fix

## When to use
When a PR reviewer has requested changes and the agent needs to address their feedback.

## Steps
1. Read all review comments carefully — understand what the reviewer wants.
2. For each actionable comment:
   - If it's a code change request, implement it directly.
   - If it's a question, answer it in code (add a comment or docstring).
   - If it suggests a different approach, follow the reviewer's suggestion.
3. Run the failing test(s) to verify nothing regresses.
4. Keep changes minimal — only address what was asked.

## Hard Rules
- NEVER use @skipIfXpu, @skip, unittest.skip, or any skip decorator.
- Do NOT commit submodule pointer changes (third_party/*).
- Do NOT ignore review comments — address every actionable one.
- If a comment is unclear, make the most reasonable interpretation.

## Common Review Patterns
- "tolerance too loose" → tighten atol/rtol to the minimum that passes
- "wrong dtype" → ensure test covers the correct dtypes for XPU
- "unnecessary change" → revert that part of your diff
- "add test for edge case" → add a focused test case
- "refactor suggestion" → follow it unless it contradicts the fix
