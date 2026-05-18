# Pre-flight Report — Private Repo Pipeline (`intel-sandbox/torch-xpu-ops-exp`)

**Date:** 2026-05-18  
**Branch:** `agent/pipeline-redesign-v2`  
**Prepared by:** Hermes pipeline agent  
**Dashboard:** [intel-sandbox/torch-xpu-ops-exp#1694](https://github.com/intel-sandbox/torch-xpu-ops-exp/issues/1694)

---

## 1. Config Changes (Phase 0)

| File | Change |
|---|---|
| `config/agent_config.yml` | All three repo keys (`issue_repo`, `tracking_repo`, `xpu_ops_upstream`) → `intel-sandbox/torch-xpu-ops-exp` |
| `issue_handler/utils/config.py` | Fallback strings `intel/torch-xpu-ops` → `intel-sandbox/torch-xpu-ops-exp` |
| `scripts/batch_format.py` | Removed live `os.environ["ISSUE_REPO"]` override that bypassed config |
| `issue_handler/fixing_steps/code_fix.py` | Updated stale comments referencing old repo |

**Commits:**
- `ed41f184` — private repo + body template alignment
- `0c1129ba` — remove all stale `intel/torch-xpu-ops` hardcoded refs
- `a54bc15b` — `xpu_ops_upstream` → private repo
- `77db2006` — `sync_labels` fix (pre-fetch labels before remove)

---

## 2. Issue Roster (Phase 1)

**Total:** 42 open issues across 6 category files in `~/torch-xpu-ops/issues-for-torch-ops/`

| Metric | Count |
|---|---|
| Total issues | 42 |
| Stage: NONE (fresh) | 35 |
| Stage: DISCOVERED | 6 |
| Stage: TRIAGED (terminal) | 1 (#198) |
| Stage: NEEDS_HUMAN (terminal) | 1 (#312) |

**Label anomalies found and resolved:**

| Issue | Anomaly | Action |
|---|---|---|
| #312 | `agent:active` left after NEEDS_HUMAN transition | Removed (pipeline bug — see §5) |
| #335 | Stale `agent:skipped` | Removed |
| #342 | Stale `agent:skipped` | Removed |
| #336 | Stale `agent:skipped` | Removed |
| #341 | Stale `agent:skipped` | Removed |
| #338 | Stale `agent:needs-human` (was at DISCOVERED) | Removed |

> **Label policy (hard rule):** Pipeline and Hermes may ONLY add/remove the 6 status labels: `agent:active`, `agent:triaged`, `agent:done`, `agent:needs-human`, `agent:blocked`, `agent:skipped`. All other labels (`agent_category_*`, `agent_test:*`, etc.) are hands-off.

---

## 3. Routing Audit (Phase 2)

**Verdict: ✅ Clean — no stale `intel/torch-xpu-ops` refs remain in pipeline Python code.**

| Check | Result |
|---|---|
| Hardcoded `intel/torch-xpu-ops` in `issue_handler/*.py` | 0 hits |
| `UPSTREAM_ISSUE_REPO` imported anywhere | Never — dead constant only in `config.py` |
| `ISSUE_REPO` constant source | `agent_config.yml` → `intel-sandbox/torch-xpu-ops-exp` |
| `sync_labels` touches non-status labels | No — `ALL_AGENT_LABELS` derived only from `stage_labels` in config |

**Routing table (triage_agent → fix dispatch):**

| `target_repo` value | Fix path |
|---|---|
| `torch-xpu-ops` | Copilot assignment on `intel-sandbox/torch-xpu-ops-exp` |
| `pytorch` | opencode agent on `chuanqi129/pytorch` |

**Dashboard update:** `e2e_report.py` writes to `TRACKING_REPO` = `intel-sandbox/torch-xpu-ops-exp`, issue #1694. Confirmed updating correctly after every pipeline cycle.

---

## 4. Test Run Results (Phase 3)

Two fresh issues run through the full FORMAT → DISCOVERED → TRIAGED/NEEDS_HUMAN cycle.

| Issue | Title | Verification | Triage Result | Label | `target_repo` | `🎯 Triage complete` | Cost |
|---|---|---|---|---|---|---|---|
| [#32](https://github.com/intel-sandbox/torch-xpu-ops-exp/issues/32) | channel last hardswish_ extra copy | Skipped (performance issue, no test cmd) | `NEEDS_HUMAN` — design decision required | `agent:needs-human` ✅ | `torch-xpu-ops` | ✗ (expected) | $0.12 |
| [#94](https://github.com/intel-sandbox/torch-xpu-ops-exp/issues/94) | test_cow failures | ✅ Reproduced (pytest ran, still fails) | `TRIAGED` — skip list addition | `agent:triaged` ✅ | `torch-xpu-ops` | ✅ checked | $0.12 |

**Pipeline behavior verified:**
- ✅ FORMAT → DISCOVERED in cycle 1
- ✅ DISCOVERED → TRIAGED/NEEDS_HUMAN in cycle 2
- ✅ Verification step runs when test command available (#94), skips gracefully when not (#32 — performance issue)
- ✅ `sync_labels` correctly sets status label, no stale `agent:active` left
- ✅ Dashboard #1694 updated after each cycle
- ✅ Non-status labels untouched throughout

---

## 5. Pipeline Bug Fixed

**Bug:** `agent:active` was left on #312 after it transitioned to `NEEDS_HUMAN`.

**Root cause:** `sync_labels` iterated `ALL_AGENT_LABELS` and called `remove_label` for every label not matching the target — including labels not present on the issue. Ghost-remove calls throw `CalledProcessError` and are caught by `except Exception: pass`. If `agent:active` removal hit a transient error immediately after the noise calls, it was silently swallowed.

**Fix (commit `77db2006`):** `sync_labels` now pre-fetches the current label set and skips `remove_label` calls for labels not actually on the issue, eliminating noise and making real failures visible.

---

## 6. Issues to Watch

| Issue | Stage | Reason |
|---|---|---|
| [#32](https://github.com/intel-sandbox/torch-xpu-ops-exp/issues/32) | `NEEDS_HUMAN` | Performance design decision — TensorIterator channel-last handling vs foreach approach |
| [#312](https://github.com/intel-sandbox/torch-xpu-ops-exp/issues/312) | `NEEDS_HUMAN` | Accumulation type decision — `XPU_ACC_TYPE(float, float)` vs double; need to verify CUDA behavior first |

---

## 7. Recommendation

**✅ Pipeline is ready for full batch run.**

No blockers. All config changes are committed and verified. Label policy is enforced. The two known `NEEDS_HUMAN` issues are correct pipeline behavior, not bugs.

**Suggested next steps:**
1. Run full batch: all 35 fresh + 6 DISCOVERED issues through FORMAT + TRIAGE
2. Review `NEEDS_HUMAN` issues with human (design decisions)
3. Enable cron once batch results are reviewed
