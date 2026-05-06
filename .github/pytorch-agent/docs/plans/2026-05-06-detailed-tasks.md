# Detailed Implementation Tasks

## Task 1: `utils/issue_body.py` — Issue body read/write helpers

**Purpose:** Replace `state.py` comment-based state with issue-body-based state. All state lives in the issue body sections.

**Functions:**
```python
def parse_sections(body: str) -> dict[str, str]
    # Parse markdown body into {section_name: content} dict
    # Sections delimited by ## headers

def update_section(body: str, section: str, content: str) -> str
    # Replace content of a named section, preserving other sections

def get_status(body: str) -> dict
    # Parse <!-- agent:status --> block → {"stage": "...", "last_updated": "..."}

def set_status(body: str, stage: str) -> str
    # Update the status block with new stage + timestamp

def check_action_item(body: str, item_substring: str) -> str
    # Mark a checkbox done: - [ ] Foo → - [x] Foo

def append_log(body: str, marker: str, log_text: str) -> str
    # Append text inside a <details> block identified by marker

def render_initial_body(data: dict) -> str
    # Render the full template from structured data dict
```

**No dependencies on other project files** — pure string manipulation.

---

## Task 2: `config.py` updates

**Changes:**
- Add `ISSUE_REPO = os.environ.get("ISSUE_REPO", "ZhaoqiongZ/torch-xpu-ops-exp")`
- Keep `UPSTREAM_ISSUE_REPO` as alias or migrate callers
- `_token_for_repo` in github_client.py must route `ISSUE_REPO` to `REVIEW_GH_TOKEN`

---

## Task 3: `github_client.py` updates

**Add:**
```python
def update_issue_body(repo: str, number: int, body: str) -> None
    # Public method wrapping _gh_api PATCH
```

**Update `_token_for_repo`:**
- Route `ISSUE_REPO` to `REVIEW_GH_TOKEN`

---

## Task 4: `discovery_agent.py` — NEW

**Purpose:** Read raw issue, format into template, overwrite body.

**Flow:**
1. `run(issue_number)` — main entry
2. Read issue body + labels via `gh.get_issue_detail(ISSUE_REPO, ...)`
3. Check for `<!-- agent:status -->` marker → skip if present
4. Call LLM with discovery skill + raw body → get structured extraction
5. Build formatted body via `issue_body.render_initial_body()`
6. `gh.update_issue_body()` to overwrite
7. `gh.add_label()` → `agent:active`
8. CLI: `python -m pytorch_agent.discovery_agent --issue N`

**LLM prompt:** "Given this raw issue, extract: summary, failed tests, error log, reproducer, commit scope. Output as JSON."

**Skill:** `pytorch-issue-discovery` — teaches extraction patterns.

---

## Task 5: `triage_agent.py` — REWRITE of issue_triaging_agent.py

**Purpose:** Read formatted issue, analyze, write root cause + fix strategy.

**Flow:**
1. `run(issue_number)` — main entry
2. Read formatted issue body
3. Check status → must be `TRIAGING`, skip otherwise
4. Select skill based on label: `agent_test: ut` → `pytorch-triage-ut`, `agent_test: e2e` → `pytorch-triage-e2e`
5. Call LLM with triage skill + formatted issue
6. Parse output → root cause + fix strategy + verdict (IMPLEMENTING or NEEDS_HUMAN)
7. Update issue body: fill Root Cause Analysis, Proposed Fix Strategy sections
8. Update status, check action item, append triage log
9. Update labels

---

## Task 6: `fixing_steps/implement.py` — SLIM DOWN

**Keep:** Branch setup, squash, push, PR creation (git + GitHub ops)
**Remove:** Prompt building, LLM dispatch logic
**Add:** Call LLM with fix skill, using issue body as the prompt context

The issue body now contains everything (summary, root cause, fix strategy, reproducer), so the prompt is simple: "Read the issue and implement the fix."

---

## Task 7: `issue_fixing_agent.py` — UPDATE orchestrator

**Add stages:** `DISCOVERED` → discovery_agent, `TRIAGING` → triage_agent
**Keep:** existing IMPLEMENTING, IN_REVIEW, etc. dispatch

---

## Task 8: Skills (4 new SKILL.md files)

Written to `torch-xpu-ops/.github/skills/`:
- `pytorch-issue-discovery/SKILL.md`
- `pytorch-triage-ut/SKILL.md`
- `pytorch-triage-e2e/SKILL.md`
- `pytorch-fix/SKILL.md`
