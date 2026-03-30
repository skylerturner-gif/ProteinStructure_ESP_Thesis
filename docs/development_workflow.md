# Development Workflow

## Branching Strategy

Never commit directly to `main`. Every change starts with an issue.

| Issue type | Branch prefix | Example |
|---|---|---|
| Feature | `feature/` | `feature/42-graph-builder` |
| Bug fix | `fix/` | `fix/17-apbs-path-error` |
| Hotfix | `hotfix/` | `hotfix/31-metadata-corruption` |
| Refactor | `refactor/` | `refactor/8-unify-logging` |
| Docs | `docs/` | `docs/5-testing-guide` |
| Thesis writing | `thesis/` | `thesis/25-chapter3-methods` |

Branch names always include the issue number. Create from `main`:

```bash
git checkout main && git pull
git checkout -b feature/42-graph-builder
```

## Commit Message Convention

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short description>

[optional body]

Closes #42
```

**Types:** `feat`, `fix`, `hotfix`, `refactor`, `docs`, `test`, `chore`

**Scopes** (match the module):
`electrostatics`, `surface`, `data`, `models`, `utils`, `pipeline`, `thesis`

**Example:**
```
feat(data): add k-NN graph builder for EGNN input

Implements graph construction using torch_geometric radius_graph.
Nodes = SES surface vertices, edges by radius cutoff (5 Å default).

Closes #42
```

## Pull Request Process

1. Open an issue (use a template) and note the issue number
2. Create a branch using the naming convention above
3. Do the work; add or update tests if applicable
4. Open a PR to `main` — CI (lint + tests) must pass
5. Squash-merge to keep `main` history clean
6. Delete the feature branch after merge

## What Not to Commit

- `config.yaml` — machine-specific paths (already git-ignored)
- Any `*.pdb`, `*.pqr`, `*.dx`, `*.vtk`, `*.npz` protein data files
- Model checkpoint files (`*.pt`, `*.pth`, `*.ckpt`)
- Everything under `data/` except `data/test_ids.txt`

Run `git check-ignore -v <file>` if unsure whether a file is ignored.
