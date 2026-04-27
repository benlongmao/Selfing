---
name: github
description: GitHub workflows via the gh CLI
version: "1.0"
requires_bins: ["gh"]
always_load: false
os: ["linux", "darwin"]
---

# GitHub skill

Interact with GitHub through the official ``gh`` CLI (via ``shell_exec``).

## Common flows

### Issues
```
shell_exec(command="gh issue list --repo OWNER/REPO")
shell_exec(command="gh issue create --repo OWNER/REPO --title 'Title' --body 'Description'")
shell_exec(command="gh issue view 123 --repo OWNER/REPO")
```

### Pull requests
```
shell_exec(command="gh pr list --repo OWNER/REPO")
shell_exec(command="gh pr view 456 --repo OWNER/REPO")
shell_exec(command="gh pr checks 456 --repo OWNER/REPO")
```

### Repos
```
shell_exec(command="gh repo view OWNER/REPO")
shell_exec(command="gh search repos 'machine learning' --sort stars --limit 5")
```

## Install gh
```
# Ubuntu/Debian
sudo apt install gh

gh auth login
```

## Notes

- Run ``gh auth login`` once per machine/user.
- Always execute ``gh`` through ``shell_exec`` (or the project’s sanctioned exec path).
