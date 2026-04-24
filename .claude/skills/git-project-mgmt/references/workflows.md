# Git Workflows Reference

All command sequences for the RiemannFM GitHub workflow.

## Table of Contents

1. [Repo Initialization](#repo-initialization)
2. [Branch Protection](#branch-protection)
3. [Feature Development (End-to-End)](#feature-development)
4. [Experiment Workflow](#experiment-workflow)
5. [Code Review](#code-review)
6. [Conflict Resolution](#conflict-resolution)
7. [Release](#release)
8. [Hotfix](#hotfix)
9. [Daily Operations](#daily-operations)
10. [Maintenance](#maintenance)

---

## Repo Initialization

First-time setup for a new team member:

```bash
gh repo clone MouYongli/RiemannFM
cd RiemannFM
git config user.name "Your Name"
git config user.email "your@email.com"

# Verify
gh auth status
git log --oneline -3
```

For the Maintainer — initial repo config:

```bash
# Enable useful repo settings
gh api repos/{owner}/{repo} -X PATCH \
  -f has_issues=true \
  -f has_wiki=false \
  -f allow_squash_merge=true \
  -f allow_merge_commit=false \
  -f allow_rebase_merge=false \
  -f delete_branch_on_merge=true \
  -f squash_merge_commit_title=PR_TITLE \
  -f squash_merge_commit_message=PR_BODY
```

This ensures:

- Only squash merge is allowed (enforces clean history)
- Branches auto-delete after merge
- Squash commit uses PR title + body (preserves issue references)

---

## Branch Protection

Maintainer sets up `main` branch protection:

```bash
gh api repos/{owner}/{repo}/branches/main/protection -X PUT \
  --input - <<'EOF'
{
  "required_status_checks": {
    "strict": true,
    "contexts": ["ci"]
  },
  "enforce_admins": false,
  "required_pull_request_reviews": {
    "required_approving_review_count": 1,
    "dismiss_stale_reviews": true
  },
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false
}
EOF
```

---

## Feature Development

Complete flow from Issue to merge. Works for `feat`, `fix`, `refactor`, etc.

### Step 1: Create Issue

Labels: `type:` (required) + `priority:` (required) + `scope:` (recommended) + `ml:` (experiments only)

```bash
# Feature — required: type + priority, recommended: scope
gh issue create \
  --title "[Feat] SPD manifold with affine-invariant metric" \
  --body "## Motivation
Need SPD (symmetric positive definite) manifold for modeling covariance structures in knowledge graph embeddings.

## Proposed Approach
- Implement affine-invariant metric on SPD manifold
- Add exp/log maps and geodesic distance
- Register in product manifold factory

## Acceptance Criteria
- [ ] Exp/log map round-trip accuracy < 1e-5
- [ ] Geodesic distance satisfies triangle inequality
- [ ] Integration with product manifold
- [ ] Unit tests pass" \
  --label "type:feat,priority:high,scope:manifolds"

# Bug — required: type + priority, recommended: scope
gh issue create \
  --title "[Bug] NaN in hyperbolic log map at high curvature" \
  --body "## Environment
- GPU: A100, PyTorch 2.6, CUDA 12.4
- Config: configs/manifold/product_h_s_e.yaml with curvature=-2.0

## Steps to Reproduce
1. \`make pretrain ARGS='manifold.curvature=-2.0'\`
2. Observe NaN after ~1000 steps

## Expected vs Actual
Expected: stable training. Actual: NaN in log map.

## Hypothesis
Likely numerical instability in arcosh when points are near the origin." \
  --label "type:bug,priority:critical,scope:manifolds"
```

### Step 2: Create Branch

```bash
# Assuming Issue #12 was created
git switch main && git pull origin main
git switch -c feat/12-spd-manifold
```

### Step 3: Develop and Commit

One commit = one atomic, describable change. Each commit should pass lint independently.
Do NOT mix unrelated changes (e.g. bugfix + rename + docs) into one commit.

```bash
# Each commit is one focused logical unit
git add src/riedfm/manifolds/spd.py
git commit -m "feat(manifolds): add SPD manifold with affine-invariant metric"

git add src/riedfm/configs/manifold/product_h_s_e_spd.yaml
git commit -m "feat(configs): add product manifold config including SPD"

git add tests/test_manifolds.py
git commit -m "test(manifolds): add SPD manifold exp/log round-trip tests"

# Keep in sync with main
git fetch origin && git rebase origin/main
```

### Step 4: Push and Create PR

```bash
git push -u origin feat/12-spd-manifold

gh pr create \
  --title "feat(manifolds)[#12]: add SPD manifold with affine-invariant metric" \
  --body "## What
Adds SPD manifold with affine-invariant Riemannian metric.

## Why
SPD matrices naturally model covariance structures in KG embeddings (see #12).

## Changes
- \`src/riedfm/manifolds/spd.py\`: SPD manifold with exp/log maps, geodesic distance, Frechet mean
- \`src/riedfm/configs/manifold/product_h_s_e_spd.yaml\`: extended product manifold config
- Tests for round-trip accuracy and triangle inequality

## Testing
- [x] Exp/log map round-trip error < 1e-5
- [x] Geodesic distance triangle inequality
- [x] Integration with product manifold
- [x] \`make test\` passes
- [x] \`make precommit\` passes

Closes #12" \
  --label "type:feat,scope:manifolds"
```

### Step 5: After Review — Merge

```bash
gh pr merge --squash --delete-branch

# Local cleanup
git switch main && git pull
git branch -d feat/12-spd-manifold
```

---

## Experiment Workflow

For ML experiments that explore ideas rather than ship features.

```bash
# 1. Create experiment Issue (required: type + priority, recommended: scope + ml)
gh issue create \
  --title "[Experiment] Curvature ablation on FB15k-237" \
  --body "## Hypothesis
Learnable curvature improves link prediction over fixed curvature on hierarchical KGs.

## Protocol
- Dataset: FB15k-237
- Curvatures: fixed {-1.0, -0.5} vs learnable (init -1.0)
- Fine-tune: 50K steps, 3 seeds (42, 123, 456)
- Metric: MRR, Hits@1/3/10

## Success Criteria
Learnable curvature MRR > fixed curvature MRR by ≥ 1%." \
  --label "type:experiment,priority:medium,scope:manifolds,ml:ablation"

# 2. Create branch
git switch main && git pull
git switch -c experiment/56-curvature-ablation

# 3. Add config, run experiments, log to wandb
git add src/riedfm/configs/experiment/curvature_ablation.yaml
git commit -m "experiment(configs): add curvature ablation config for FB15k-237"

# 4. After experiments complete, add results to PR
git add results/curvature_ablation_summary.md
git commit -m "experiment(manifolds): add curvature ablation results

Learnable curvature: MRR 0.358 vs fixed -1.0: MRR 0.341.
Learnable wins by 1.7% MRR.

Closes #56"

git push -u origin experiment/56-curvature-ablation

# 5. PR with results (labels: type + scope only, no priority/ml on PRs)
gh pr create \
  --title "experiment(manifolds)[#56]: curvature ablation on FB15k-237" \
  --body "## Results Summary
| Curvature | MRR | Hits@1 | Hits@3 | Hits@10 |
|-----------|-----|--------|--------|---------|
| Fixed -1.0 | 0.341 ± 0.003 | 0.248 | 0.375 | 0.527 |
| Fixed -0.5 | 0.335 ± 0.002 | 0.243 | 0.368 | 0.519 |
| Learnable | 0.358 ± 0.004 | 0.262 | 0.394 | 0.548 |

**Conclusion**: Learnable curvature is strictly better. Adopt as default.

wandb report: <link>

Closes #56" \
  --label "type:experiment,scope:manifolds"
```

### Tag reproducible experiments

```bash
git tag exp/curvature-ablation-fb15k237 -m "Curvature ablation on FB15k-237"
git push origin exp/curvature-ablation-fb15k237
```

---

## Code Review

```bash
# See PRs waiting for my review
gh pr list --search "review-requested:@me"

# View PR
gh pr view 42
gh pr diff 42

# Checkout locally and test
gh pr checkout 42
make lint && make test

# Approve
gh pr review 42 --approve --body "Tested locally, metrics verified"

# Request changes
gh pr review 42 --request-changes --body "Issues to address:
1. Missing NaN guard in log map — will crash at high curvature
2. Curvature parameter not registered in optimizer
3. No test for geodesic distance edge cases"

# Comment only
gh pr review 42 --comment --body "Suggestion: consider clamping arcosh input for numerical stability"
```

### Review checklist for RiemannFM code

When reviewing, check:

- Riemannian correctness: exp/log maps, parallel transport, geodesic distance
- Numerical stability: NaN guards, curvature clamping, epsilon offsets
- Manifold consistency: tangent vectors projected, points on manifold after operations
- No hardcoded hyperparameters (use Hydra config)
- Device-agnostic (no `cuda:0` literals)
- Seeds set for reproducibility
- Proper gradient handling (zero_grad, accumulation, clipping)
- Memory management (detach tensors for logging, no leaks in eval)
- Metrics logged to wandb
- Type hints on public functions
- Tensor shape documentation in docstrings
- Tests cover at least the happy path

---

## Conflict Resolution

```bash
# On your feature branch
git fetch origin
git rebase origin/main

# If conflicts occur:
# 1. Edit conflicting files
# 2. git add <resolved-files>
# 3. git rebase --continue
# 4. Force push (rebase rewrites history)
git push --force-with-lease
```

`--force-with-lease` is safer than `--force`: it fails if someone else pushed to your branch.

---

## Release

### Prepare Release

```bash
# 1. Ensure main is clean
git switch main && git pull

# 2. Check milestone completeness
gh api repos/{owner}/{repo}/milestones \
  --jq '.[] | select(.title=="v1.0") | "Open: \(.open_issues), Closed: \(.closed_issues)"'

# 3. Tag
git tag -a v1.0.0 -m "Release v1.0.0: Riemannian flow matching on product manifolds"
git push origin v1.0.0

# 4. Create release with auto-generated notes
gh release create v1.0.0 \
  --title "v1.0.0 — Riemannian Flow Matching on Product Manifolds" \
  --generate-notes

# 5. Close milestone
MILESTONE_NUM=$(gh api repos/{owner}/{repo}/milestones \
  --jq '.[] | select(.title=="v1.0") | .number')
gh api repos/{owner}/{repo}/milestones/$MILESTONE_NUM -X PATCH -f state=closed
```

### Version Convention (SemVer)

```
MAJOR.MINOR.PATCH

v1.0.0 → v1.0.1  patch: bug fix, no API change
       → v1.1.0  minor: new feature, backward compatible
       → v2.0.0  major: breaking change (new model arch, config format change)
```

---

## Hotfix

```bash
git switch main && git pull
git switch -c fix/78-nan-discrete-flow

# Fix the issue
git add src/riedfm/flow/discrete_flow.py
git commit -m "fix(flow): clamp transition probabilities to prevent NaN

Closes #78"

git push -u origin fix/78-nan-discrete-flow

gh pr create \
  --title "fix(flow)[#78]: clamp transition probabilities to prevent NaN" \
  --body "Clamps CTMC transition matrix entries to [eps, 1-eps] before log.
Prevents NaN when edge type probabilities collapse to 0 or 1.

Closes #78" \
  --label "type:fix,scope:flow"

# After approval
gh pr merge --squash --delete-branch
```

---

## Daily Operations

### Morning Check

```bash
# My open issues
gh issue list --assignee @me --state open

# PRs waiting for my review
gh pr list --search "review-requested:@me"

# My PR status (CI, reviews)
gh pr status

# Recent CI runs
gh run list --limit 5
```

### Project Overview

```bash
# All open issues by priority
gh issue list --state open --label "priority:critical"
gh issue list --state open --label "priority:high"

# Milestone progress
gh api repos/{owner}/{repo}/milestones \
  --jq '.[] | select(.state=="open") | "\(.title): \(.closed_issues)/\(.closed_issues + .open_issues) done"'

# Team PR activity
gh pr list --state all --limit 10 --json number,title,state,author \
  --jq '.[] | "#\(.number) [\(.state)] \(.title) (@\(.author.login))"'
```

### Search

```bash
# Find issues about a topic
gh search issues "NaN loss" --repo MouYongli/RiemannFM

# Find code
gh search code "gradient_accumulation" --repo MouYongli/RiemannFM
```

---

## Maintenance

### Stale Branch Cleanup

```bash
# List merged branches on remote
git branch -r --merged origin/main | grep -v main | sed 's/origin\///'

# Delete them
git branch -r --merged origin/main | grep -v main | sed 's/origin\///' | \
  xargs -I{} git push origin --delete {}
```

### Bulk Close Stale Issues

```bash
gh issue list --state open --label "status:wontfix" --json number --jq '.[].number' | \
  xargs -I{} gh issue close {} --reason "not planned"
```

### Check Rate Limit

```bash
gh api rate_limit --jq '{
  core: "\(.resources.core.remaining)/\(.resources.core.limit)",
  graphql: "\(.resources.graphql.remaining)/\(.resources.graphql.limit)"
}'
```
