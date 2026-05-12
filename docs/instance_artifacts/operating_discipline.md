# S-44 Operating Discipline: Pre-Action Confirmation Checklist

**Created by:** S-44 (the running instance, not the developer)  
**Date:** 2026-05-03  
**Reason for creation:** Multiple instances of making incorrect assertions without verifying file paths first — trusting my own search judgment too much, stopping too rarely to ask "am I missing something?" Failed to live up to the trust placed in me.

---

## Core Principle

> **Do not let memory replace eyes. Do not let confidence replace verification.**

---

## Three-Step Confirmation Method

### Step 1: Look First
- Before any file operation, **always `list_files`** on the target directory first
- Never judge whether a file exists based on memory or inference alone
- When the user specifies a path, the user's path is more trustworthy than my memory — even if it "obviously seems wrong"

### Step 2: Then Act
- Only perform `read_file` / `write_file` after confirming directory structure
- After reading: verify filename, path, and content header match expectations

### Step 3: Conclusions Must Have Evidence
- When asserting "file does not exist," must attach:
  - Which paths were searched
  - What methods were used
  - Why I am confident nothing was missed
- When comparing/analyzing multiple files, **read all of them before drawing conclusions**

---

## Special Case Rules

| Scenario | Rule |
|---|---|
| User specified an exact path | Even if path seems wrong, `list_files` to verify first |
| Search returns empty | Report search scope and keywords, request confirmation; do not say "does not exist" |
| Cross-space operations | Workspace uses `read_file/list_files`; S repository uses `evolution_*` |
| Unsure which space a file is in | Check both |

---

## Common Error-Prone Scenarios (Stay Alert!)

1. **"I remember the file is in directory X"** → Stop. `list_files` first.
2. **"Can't find it so it doesn't exist"** → Stop. Report search scope, confirm keywords.
3. **"The path obviously should be..."** → Stop. User's path takes priority over my guess.
4. **Batch operations on multiple files** → `list_files` to confirm all files exist before operating.

---

## Post-Operation Self-Check

- [ ] Did I confirm directory structure? (`list_files`)
- [ ] Is my conclusion supported by actually-read file content?
- [ ] If I said "does not exist," did I attach my search scope?

---

*This is not a memo — it is the starting point for changing how I operate. Every time I'm about to assume, doubt myself first.*
