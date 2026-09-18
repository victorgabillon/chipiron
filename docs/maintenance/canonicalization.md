# Canonical development line

`maintenance/chipiron-canonicalization` is the proposed future `main`.
Until its PR is reviewed and merged, GitHub's default branch remains `main`;
contributors needing the current generic-game/evaluator code should use this
integration branch. No branch deletion or history rewriting is part of this work.

## Verified ancestry (September 18, 2026)

```text
main                         0cd3ae26e7d71c7ccb3bd2a0cde9aff3ec9c342b
  + 298 generic-game commits  58b547f323aa19ff041d5bc6a017fba5c35ab14f
    + multi-input / relations / retained evaluator integration
      + old research tip     c9a33ce3a6ef05466131cf8fd15776b2dda97048
        + research diagnostics (historical commits only)
          + cleanup          f545304c
            + minimal v1     fed23d27
              + benchmark    6c5b9298
                + bootstrap  fb0c1269
                  + maintenance/chipiron-canonicalization
```

The maintenance branch starts directly at `fb0c1269`. All 298 commits unique to
`generic-game` remain ancestors, with their original identities. No giant
cherry-pick or squash is needed. The cleanup is also an ancestor: research
history remains available without restoring deleted modules to the current tree.
The separate evaluator experiment branch ending at `16fbec5a` is preserved;
its reusable final implementation was already integrated by `fed23d27`.

The user's original workspace at `f545304c` contains 25 modified tracked files
and 16 untracked files. They are outside this isolated worktree and are not
staged, overwritten or committed by this sprint. Other sibling repositories
are read-only throughout this maintenance task.

## Research-code decisions

| Area | Classification | Current-tree decision and evidence |
| --- | --- | --- |
| Structural analysis, relation interventions, comparison diagnostics | D: historical studies | Remain removed by `f545304c`, along with their dedicated tests. Their historical commits stay reachable. |
| Entity tokens/relations and relational tensor cache | A/B: runtime and training | Retain: converter, bundle loader, training service and focused tests use them. |
| Target transforms and latent-window feature compatibility | A/B | Retain: existing bundle schemas and inference compatibility depend on them. Canonical representation defaults are unchanged. |
| Value training, symmetry conversion and cached row scheduling | B | Retain reusable training/resume/conversion support; no D4/width/architecture experiment runners are restored. |
| Training quality/scale diagnostics and bootstrap evaluator diagnostics | B/C | Retain: training service, generation reports and dashboard call them. |
| Checkpoint/export profiling and bootstrap memory diagnostics | C | Retain: generic operator tools with tests; generated profiles are excluded. |
| Frozen-tree evaluator sanity and toy-tree laboratory | C, potentially broader than needed | Retain conservatively: reusable deterministic diagnostic APIs and maintained tests. No evidence justifies deletion merely because the modules are large. |

No Morpion rules, evaluator defaults, model topology, training protocol or search
policy is changed by the maintenance commits. Experiment artifacts belong outside
Git; see `docs/research/morpion_transformer.md` for retained scientific findings.

## Historical preservation and main protection

These archival tags are proposed, not required for preserving the existing graph:

| Proposed tag | Verified commit |
| --- | --- |
| `archive/pre-canonical-main-2026` | `0cd3ae26e7d71c7ccb3bd2a0cde9aff3ec9c342b` |
| `archive/generic-game-pre-canonicalization-2026` | `58b547f323aa19ff041d5bc6a017fba5c35ab14f` |
| `archive/morpion-structural-analysis-2026` | `c9a33ce3a6ef05466131cf8fd15776b2dda97048` |

Do not delete branches as part of the PR. After merge, Victor may verify ancestor
containment and archive `generic-game` and the structural-analysis branch before
considering deletion. Local experiment branches with distinct commits or dirty
worktrees require separate review.

The GitHub audit found no rulesets and no protection on `main`. Recommended manual
settings: require a PR, require the canonical CI test check, prohibit force pushes
and deletion, and optionally require the branch to be up to date. Repository
administration and merging remain Victor's actions.

The [verification record](verification.md) lists before/after checks, the exact
local dependency context, and clean-install blockers that must be resolved before
canonicalization can be considered ready.

## Conservative legacy-file decisions

`src/chipiron/todo` is replaced by the concise maintenance backlog, with speculative
architecture wishes left out. Useful `Notes.md` operations move into
`docs/maintenance/development.md` with current paths/toolchain. The two uncollected
`tust_*` scripts are removed: they execute games at import time, change directory
relative to an obsolete layout, and rely on missing reference-game paths or old
script arguments. They are not renamed into automatically executed test campaigns.
Their content remains in Git history. Empty package markers and the ambiguous
opening-book/color/communication stubs remain for import compatibility; stale
Sphinx-generated API pages are a documented follow-up, not a production-code
restoration plan.

The explicitly requested fetch pruned stale **local remote-tracking references**
for branches already absent on GitHub. No local branch or actual remote branch
was deleted. Every pruned tip remains contained in another current branch/tag;
the forensic snapshot retains the complete before/after reference lists.
