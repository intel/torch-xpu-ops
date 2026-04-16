---
name: create_skill
description: Create new Claude skill with complete documentation of workflow constraints and tools used
---

# Create Skill

Create a new Claude skill with comprehensive documentation of all constraints and tools used in the workflow.

## Purpose

When the user requests creation of a new skill, ensure ALL constraints and tools are documented so the workflow can be reproduced later.

## Core Principle

Every tool used and every constraint encountered during skill creation MUST be documented in the SKILL.md file.

## Workflow

### Step 1: Understand the Task

1. User provides task description or reference (GitHub issue, PR, etc.)
2. Understand the domain (torch-xpu-ops, pytorch, etc.)
3. Identify where changes need to be made

### Step 2: Identify Tools Needed

During the workflow, track ALL tools used:
- bash - terminal commands
- read - read files
- write - create files
- edit - modify files
- grep - search content
- glob - find files by pattern
- webfetch - fetch web content
- websearch - search web
- question - ask user
- task - launch sub-agents

### Step 3: Identify Constraints

Track ALL constraints encountered:
- File paths (working directory, repo locations)
- API authentication (gh CLI may fail with 401)
- Tool limitations (e.g., webfetch returns 401)
- Environment variables
- Repository structure
- Naming conventions

### Step 4: Create Skill File

Create `<workspace>/.claude/skills/<skill_name>/SKILL.md`:

```markdown
---
name: <skill_name>
description: <what the skill does>
---

# <Skill Title>

<Detailed description>

## Quick Start

1. <step 1>
2. <step 2>
3. <step 3>

## Detailed Instructions

### Step 1: <Name>
<Description>

## Tools Used

| Tool | Purpose |
|------|---------|
| <tool1> | <reason> |
| <tool2> | <reason> |

## Constraints

- <constraint1>
- <constraint2>

## Error Handling

| Error | Handling |
|-------|----------|
| <error> | <handling> |
```

### Step 5: Verify Skill

1. Read the created skill file
2. Check all tools are documented
3. Check all constraints are documented
4. Check examples are included

## Required Sections

Every skill MUST have these sections:

### 1. Front Matter
```yaml
---
name: <skill_name>
description: <what the skill does>
---
```

### 2. Quick Start
3-5 bullet points explaining the workflow

### 3. Detailed Instructions
Step-by-step instructions

### 4. Tools Used
Table listing all tools used and their purpose

### 5. Constraints
All constraints encountered in the workflow

### 6. Error Handling
Table of common errors and how to handle them

### 7. Examples (optional but recommended)
Real-world examples of the skill in action

## Tools Used

| Tool | Purpose |
|------|---------|
| read | Read existing skill files as reference |
| write | Create new skill file |
| edit | Update skill file |
| glob | Find skill files |
| question | Ask user for clarification |

## Constraints

- Skill files must be in `.claude/skills/<skill_name>/SKILL.md`
- Front matter must have name and description
- All tools used must be documented
- All constraints must be documented

## Example: Creating check_pytorch_skip Skill

During the workflow:
1. Used `websearch` to find issue details
2. Used `grep` to search pytorch repo
3. Used `read` to find test method
4. Used `edit` to add skip decorator

Constraints encountered:
- gh CLI returns 401 (unauthenticated)
- Had to use websearch/webfetch as fallback
- pytorch repo path: relative or as specified by user

All documented in skill file.

## Integration

This skill can be loaded alongside:
- check_pytorch_skip - for checking pytorch test skips
- skiplist_pr - for torch-xpu-ops skip list
- clean_commit - for git commit practices
