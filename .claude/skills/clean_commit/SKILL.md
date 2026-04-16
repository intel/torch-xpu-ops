---
name: clean_commit
description: Ensure code updates follow clean git commit practices - one change per commit, clear messages, no history modification
---

# Clean Commit

Enforce clean git commit practices to prevent messy code history.

## Core Rules

### Rule 1: One Change = One Commit
After ANY code change (file write, edit, patch), **create a new git commit immediately**.
- Do NOT batch multiple changes into a single commit
- Each logical change gets its own commit

### Rule 2: Clear Commit Messages
Write clear, short, descriptive commit messages in English.

Format:
```
<type>: <short description>

- <detail 1>
- <detail 2>
```

Types:
- `fix`: bug fix
- `feat`: new feature
- `style`: minor adjustment (only if user approves)
- `refactor`: code refactoring
- `test`: test addition/modification
- `docs`: documentation

Example:
```
fix: parse logic in extractor

- Handle empty input case
- Add validation for edge cases
```

### Rule 3: Do NOT Change Git History
- No git amend
- No git rebase
- No git force push
- Unless user explicitly allows

### Rule 4: No Config Changes
- Do NOT modify .gitignore without permission
- Do NOT change git config without permission

## Workflow

### After Any Code Change:

1. **Check git status:**
```bash
git status
```

2. **Stage the changed files:**
```bash
git add <file_path>
```

3. **Create commit with clear message:**
```bash
git commit -m "<type>: <short description>"
```

4. **Verify commit:**
```bash
git log -1 --oneline
```

## Tools Used

| Tool | Purpose |
|------|---------|
| bash | Run git commands (status, add, commit, log) |
| read | Check current file state before edit |

## Constraints

- Working directory: any git repository
- Always commit after file write/edit operations
- Never batch changes
- Commit message must be descriptive

## Error Handling

| Error | Handling |
|-------|----------|
| No changes to commit | Skip commit, no action needed |
| Multiple files changed | Ask user which files belong to same logical change |
| Untracked files | Ask user if new files should be committed |
| Commit message too short | Require at least 5 characters in description |

## When to Commit

Commit immediately after:
- `write` tool creates/updates a file
- `edit` tool modifies a file
- Any patch application
- Any file creation

## When NOT to Commit

- Initial repo clone (already has history)
- Read-only operations
- Search/grep operations

## Example Workflows

### Example 1: Single File Edit
```
User: Fix bug in parser.py
→ Edit file with fix
→ git status (check changes)
→ git add parser.py
→ git commit -m "fix: handle empty input in parser"
→ git log -1 (verify)
```

### Example 2: Multiple Files
```
User: Add feature with tests
→ Write main feature file
→ Write test file
→ git status (see 2 files)
→ Ask user: "These are separate changes or one feature?"
→ If separate: commit each file separately
→ If one feature: git add both, single commit
```

### Example 3: Ask Before Action
```
User: Make changes to .gitignore
→ Do NOT modify without asking
→ Ask: "Do you want me to modify .gitignore?"
→ If yes: make change + commit
→ If no: skip
```

## Anti-Patterns to Avoid

❌ Batch multiple fixes into one commit
❌ Use vague messages like "fix", "update", "changes"
❌ Amend commit to add more changes
❌ Force push to share work
❌ Modify .gitignore without asking

## Integration with Other Skills

This skill should be loaded alongside other skills to ensure clean commits:
- After **check_pytorch_skip** makes changes → commit
- After **skiplist_pr** makes changes → commit
- After any agent code modification → commit

## Integration with create_skill

This skill was created using the **create_skill** workflow. When modifying this skill:
1. Track all new tools used
2. Track all new constraints encountered
3. Update this SKILL.md file with the create_skill pattern

To create a similar skill, load the create_skill skill first:
```bash
skill load create_skill
```
