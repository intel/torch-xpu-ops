# Public E2E Pipeline — Bug Report

## BUG-001: Triage output parsing failure (markdown format)
- **Status**: FIXED (commit `7b8bbe1d`)
- **Symptom**: Triage agent returns structured markdown (`**Verdict:** \`IMPLEMENTING\``) instead of JSON → parser fails → defaults to NEEDS_HUMAN
- **Fix**: Added `_parse_markdown_triage()` fallback that extracts verdict/root_cause/fix_strategy from markdown when JSON parsing fails
- **Affected**: #2795

## BUG-002: OpenCode orphan processes (idle hang)
- **Status**: FIXED (commit `1d6fd5bc`)
- **Symptom**: OpenCode closes stdout while API hangs → pipeline `_read_lines()` exits loop but process stays alive indefinitely
- **Fix**: 180s idle timeout + explicit `proc.kill()` after stdout EOF
- **Affected**: All issues (systemic)

## BUG-003: Dashboard token mismatch
- **Status**: FIXED (commit in `agent/pipeline-redesign-v2`)
- **Symptom**: `_token_for_repo()` didn't recognize `TRACKING_REPO` (ZhaoqiongZ/torch-xpu-ops-exp) → used wrong token → 403 on dashboard update
- **Fix**: Added `TRACKING_REPO` to `_token_for_repo()` routing → uses `REVIEW_GH_TOKEN`
- **Affected**: Dashboard #1694

## BUG-004: IN_REVIEW not terminal
- **Status**: FIXED (same commit as BUG-003)
- **Symptom**: Pipeline kept cycling on IN_REVIEW issues, logging "not yet implemented"
- **Fix**: Added `IN_REVIEW` to `TERMINAL_STAGES`

## BUG-005: Submodule fixes not supported
- **Status**: KNOWN LIMITATION
- **Symptom**: Code_fix agent runs in `~/pytorch` but many XPU issues need changes in `third_party/torch-xpu-ops` (submodule). Agent completes without error but produces no diff.
- **Affected**: #2795, #3390, #2140
- **Workaround**: Mark as NEEDS_HUMAN; human applies fix in `intel/torch-xpu-ops` directly
- **Future fix**: Add mode to run code_fix in `~/torch-xpu-ops` directly, then update submodule pointer

## BUG-006: Triage timeout on complex/broad issues
- **Status**: KNOWN LIMITATION
- **Symptom**: Issues that are umbrella tasks, feature gaps, or performance investigations exceed 900s triage timeout
- **Affected**: #1856, #1969, #3150, #3080
- **Workaround**: Mark as NEEDS_HUMAN after 2 failed attempts
- **Future fix**: Increase timeout for complex issues, or pre-classify issue complexity and skip triage for known-hard types
