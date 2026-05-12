# Public E2E Run — Bug Report

## Run Info
- **Date**: 2026-05-11
- **Branch**: `agent/pipeline-redesign-v2` @ `56ce94f6`
- **ISSUE_REPO**: `intel/torch-xpu-ops`
- **TRACKING_REPO**: `ZhaoqiongZ/torch-xpu-ops-exp`
- **Dashboard**: https://github.com/ZhaoqiongZ/torch-xpu-ops-exp/issues/1694

---

## Pipeline Issues Encountered

### Bug 1: Triage agent hangs on API calls
- **Affected issues**: #2795 (public), #193, #147 (exp)
- **Symptom**: Opencode reads the skill file (step 1), starts step 2, then the LLM API call hangs indefinitely — no output, no error, no timeout from the API. The process stays alive consuming ~3% CPU.
- **Impact**: 900s timeout wasted per occurrence. Issue stays at DISCOVERED.
- **Root cause**: Unknown — likely Anthropic API intermittent hang. Simple prompts ("Say hello") respond instantly. Only happens with large triage prompts (~3-4K chars).
- **Workaround**: Pipeline retries on next cycle. Eventually succeeds (exp #193 succeeded on retry).
- **Fix needed**: Consider adding a per-step idle timeout (e.g. 120s with no output → kill and retry) instead of only a total timeout.

### Bug 2: Error comments leak full prompt in traceback
- **Status**: ✅ FIXED in commit `d09819ed`
- **Details**: Error comments posted to issues contained the full opencode command including the entire issue body as the prompt argument. Fixed by sanitizing timeout/rc/generic errors.

### Bug 3: Dashboard overwrites instead of merging
- **Status**: ✅ FIXED in commit `d09819ed`
- **Details**: `build_report` replaced the entire dashboard body. Now merges with existing rows.

### Bug 4: Dashboard went to public ISSUE_REPO
- **Status**: ✅ FIXED in commit `56ce94f6`
- **Details**: Added `TRACKING_REPO` config to always send dashboard to private exp repo.

---

## Batch Progress

### Batch 1: #2795 #2560 #3361 #3388

| Issue | Format | Triage | Fix | PR | Notes |
|-------|--------|--------|-----|----|-------|
| #2795 | ✅ | ❌ rc=-15 (API hang, killed) | — | — | Needs retry |
| #2560 | ✅ | 🔄 In progress | — | — | |
| #3361 | ✅ | ⬜ | — | — | |
| #3388 | ✅ | ⬜ | — | — | |

### Batch 2: #1856 #1969 #2715
Not started.

### Batch 3: #3390 #3150 #3080 #2207 #2140
Not started.
