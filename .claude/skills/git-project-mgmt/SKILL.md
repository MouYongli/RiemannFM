---
name: git-project-mgmt
description: >
  GitHub project management with gh CLI and git for a deep learning research team.
  Trigger when user wants to: create/manage issues, branches, PRs, releases, labels,
  milestones, review code, check CI, manage GitHub Projects, or any gh/git workflow.
  Also trigger for: "open a PR", "file an issue", "create a release", "check CI",
  "review code", "push changes", "merge", "tag", "branch", or any reference to the
  team's git workflow conventions.
---

# Git Project Management for RieDFM-G

This skill manages the full GitHub workflow for the RieDFM-G project.
Always follow the conventions defined in the project's CLAUDE.md.

## Prerequisites

Before any operation, verify:

```bash
gh auth status
git config user.name && git config user.email
```

## Quick Reference

| Task | Action |
|------|--------|
| Start new work | Create Issue → Create branch → Develop → PR |
| Branch naming | `<type>/<issue#>-<description>` |
| Commit format | `<type>(<scope>): <subject>` |
| Issue title | `[Type] description` (Type capitalized) |
| PR title | `<type>(<scope>)[#N]: <subject>` (omit [#N] if no Issue) |
| Merge strategy | Squash merge, delete branch |

## Core Conventions

### Types

`feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `ci`, `perf`, `revert`, `experiment`

### Scopes (RieDFM-G)

`manifolds`, `flow`, `models`, `layers`, `data`, `losses`, `utils`, `cli`, `configs`

### Issue Title Format

```
[Type] description
```

Examples:

- `[Feat] SPD manifold with affine-invariant metric`
- `[Bug] NaN in hyperbolic log map at high curvature`
- `[Experiment] Curvature ablation on FB15k-237`

### Branch Naming

```
<type>/<issue-number>-<short-description>
```

Examples:

- `feat/12-spd-manifold`
- `fix/34-nan-log-map`
- `experiment/56-curvature-ablation`

### Commit Messages

```
<type>(<scope>): imperative subject ≤72 chars

Optional body explaining WHY.

Closes #<issue-number>
```

### PR Title

```
<type>(<scope>)[#<issue-number>]: <subject>
```

If no corresponding Issue, omit `[#N]`:

```
<type>(<scope>): <subject>
```

Examples:

- `feat(manifolds)[#12]: add SPD manifold`
- `chore(ci): add ruff to CI pipeline`

## Workflows

For step-by-step command sequences, read `references/workflows.md`.

## Label Policy

Labels vary by operation type. Follow this strictly.

### Issue Labels

| Category | Required? | When |
|----------|-----------|------|
| `type:` × 1 | Required | Always |
| `priority:` × 1 | Required | Always |
| `scope:` × 1~2 | Recommended | When specific modules involved |
| `ml:` × 1 | Recommended | Experiment-type issues only |
| `status:` | Dynamic | Add/remove as state changes |

### PR Labels

| Category | Required? | When |
|----------|-----------|------|
| `type:` × 1 | Required | Always (match Issue/branch) |
| `scope:` × 1~2 | Recommended | Match the Issue |
| `priority:` | Never | Belongs to Issue level |
| `ml:` | Never | Belongs to Issue level |

### Release / Tag

No labels. Use SemVer tags (`v1.2.0`) and Milestones.

## Decision Guide

| User wants to... | Start with... |
|---|---|
| Work on a new feature | `references/workflows.md` § Feature Development |
| Fix a bug | `references/workflows.md` § Feature Development (same flow) |
| Run and track experiment | `references/workflows.md` § Experiment Workflow |
| Do a release | `references/workflows.md` § Release |
| Set up repo settings | `references/workflows.md` § Repo Initialization |
| Review someone's PR | `references/workflows.md` § Code Review |
| Check project progress | `references/workflows.md` § Daily Operations |
