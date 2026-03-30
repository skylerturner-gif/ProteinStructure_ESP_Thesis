# GitHub Repository Setup

One-time setup commands to configure labels, milestones, a Kanban board, and branch protection. Run after pushing the repo to GitHub.

## Prerequisites

```bash
gh auth login
gh repo view    # confirm you are in the right repo
```

## Labels

Delete GitHub's default labels first (optional but keeps things clean):

```bash
gh label delete "enhancement" --yes 2>/dev/null || true
gh label delete "good first issue" --yes 2>/dev/null || true
gh label delete "help wanted" --yes 2>/dev/null || true
gh label delete "invalid" --yes 2>/dev/null || true
gh label delete "question" --yes 2>/dev/null || true
gh label delete "wontfix" --yes 2>/dev/null || true
```

Create project labels:

```bash
# Software concerns
gh label create "bug"           --color "d73a4a" --description "Something is broken"
gh label create "feature"       --color "0075ca" --description "New feature or capability"
gh label create "hotfix"        --color "b60205" --description "Urgent fix blocking work"
gh label create "refactor"      --color "e4e669" --description "Code quality improvement, no behavior change"
gh label create "documentation" --color "cfd3d7" --description "Documentation update"
gh label create "needs-triage"  --color "ededed" --description "Needs review and assignment"
gh label create "priority: high" --color "e11d48" --description "Blocking other work"

# Thesis / academic concerns
gh label create "thesis-writing"  --color "0e8a16" --description "Thesis chapter or writing task"
gh label create "thesis-chapter"  --color "2ea44f" --description "Full chapter milestone"
gh label create "experiment"      --color "1d76db" --description "ML experiment or ablation study"
gh label create "literature"      --color "5319e7" --description "Literature review task"
```

Verify:

```bash
gh label list
```

## Milestones

Update the `due_on` dates to match your actual thesis schedule before running.

```bash
gh api repos/:owner/:repo/milestones --method POST \
  -f title="Stage 1: Data Pipeline" \
  -f description="Complete ESP pipeline for all training proteins" \
  -f due_on="2026-05-01T00:00:00Z"

gh api repos/:owner/:repo/milestones --method POST \
  -f title="Stage 2: Baseline Model" \
  -f description="EGNN training loop, initial baseline metrics" \
  -f due_on="2026-07-01T00:00:00Z"

gh api repos/:owner/:repo/milestones --method POST \
  -f title="Stage 3: Experiments" \
  -f description="Ablations, hyperparameter sweeps, comparisons with baselines" \
  -f due_on="2026-09-01T00:00:00Z"

gh api repos/:owner/:repo/milestones --method POST \
  -f title="Stage 4: Thesis Writing" \
  -f description="All chapters drafted and revised with advisor" \
  -f due_on="2026-11-01T00:00:00Z"

gh api repos/:owner/:repo/milestones --method POST \
  -f title="Stage 5: Defense" \
  -f description="Final revisions, submission, and defense preparation" \
  -f due_on="2026-12-15T00:00:00Z"
```

## GitHub Projects (Kanban Board)

Create the project:

```bash
gh project create --owner @me --title "ESP Thesis Board"
# Note the project number returned (e.g. 1)
```

Column setup must be done in the browser (the `gh` CLI does not support column creation for classic Projects):

1. Go to **github.com → Projects → ESP Thesis Board**
2. Rename the default columns (or add new ones) to: **Backlog → In Progress → In Review → Done**
3. Go to **Project Settings → Linked repositories** → add this repo

When creating issues, assign them to the board and drag them across columns as work progresses.

## Branch Protection

Require CI to pass before merging PRs to `main`. The `enforce_admins: false` setting means you (the repo owner) can still push directly in an emergency — useful for a solo thesis project.

```bash
gh api repos/:owner/:repo/branches/main/protection \
  --method PUT \
  -H "Accept: application/vnd.github+json" \
  --input - <<'EOF'
{
  "required_status_checks": {
    "strict": true,
    "contexts": ["lint-and-test"]
  },
  "enforce_admins": false,
  "required_pull_request_reviews": null,
  "restrictions": null
}
EOF
```

The value `"lint-and-test"` must match the `name:` of the job in `.github/workflows/ci.yml` exactly.

To verify protection is active:

```bash
gh api repos/:owner/:repo/branches/main/protection
```
