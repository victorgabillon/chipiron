# Published dependency and selector reconciliation

This record supersedes the blocked status in the earlier dependency-unblock and
verification reports. Those reports remain as dated evidence. The original
dirty workspaces are preserved; the maintained changes were developed and tested
in separate worktrees.

## Maintained dependency histories

Coral PR #4 was merged with its original history. Coral main
`0feb7b2d50f95f31280f81c34f158f01dd74891d` contains the published `v0.1.15`
source `c3a16707bb5f3255e28f07e53dc769be989e4dd5`. Its relation scale,
additive-bias inference fix and legacy state-dict support remain unchanged.
Original dirty Coral work consists of additional runtime overrides/diagnostics;
it is preserved separately and is not required for evaluator-v1.

Anemone PR #71 brought the published 0.2.22 and 0.2.23 histories into main
`13f67c3293851a842010d10220d7e61c668a4697`. PR #72 then added the optional
alternating policy, followed by release PR #73. Published **Anemone 0.2.24**
comes from main commit `31528014dda463ff3df40bb48bb5c8acce696e73`, tagged
`v0.2.24` only after integration. The release workflow now verifies main
ancestry and version/tag agreement before publication.

The [trusted publication workflow](https://github.com/victorgabillon/anemone/actions/runs/35572740546)
passed. The fresh PyPI wheel matches all 287 package payload files at that commit.
Its SHA-256 is
`7225e0ab537ca2bf3b33d492b9b996549ff38e2246255701c66affe81f878f89`;
the sdist SHA-256 is
`b4ab4e42308a8548fb268cbd2abe7fafcc08e150aa9c8c0b64fa087fe635693e`.
Official PyPI provenance identifies the same source. Artifact/source comparison
and attestation inspection do not claim independent Sigstore trust-chain validation.

## Explicit Morpion behavior

Chipiron pins published Coral 0.1.15 and Anemone 0.2.24. The canonical Morpion
`default_search_args()` explicitly requests `alternating_by_step`. Anemone's
global default remains `inverse_depth`.

Each selector counts successful depth selections starting at zero. Odd choices
use the existing inverse-depth sampler; even choices use the existing minimum
of `(opened_count * (depth + 1), depth)`. Within-depth ranking is unchanged.
Even depth selection consumes no random draw, while subsequent node ranking can.
This counter is selector state, not the global tree-growth step.

Comparing the effective search configurations before and after reconciliation
changes only `node_selector.base.depth_selection_policy`. Opening expansion,
rollouts, recommendation, branch budget, evaluators, workers and seeds retain
their previous configuration.

The two original logging tests construct their own search arguments. Their
shared fixture now also requests alternating explicitly. All three original
blocker test functions and their assertions remain unchanged; no skip or
cosmetic assertion adjustment was needed.

## Checkpoint boundary

Anemone persists the selection count in typed, sharded and streaming restore
paths, including stale-cache rebuilds. It also preserves the composed selector's
existing RNG and reconstructs its JSON state correctly. These changes preserve
the next node choice as well as the next subpolicy.

Chipiron adds eight integration cases: save after selector choice 1 or 2, then
restore through plain JSON, Zstandard JSON, gzip JSON or the actual sharded
streaming path. Each compares two subsequent choices, full selection diagnostics,
tree size and RNG state against the uninterrupted run, using an unrelated
constructor seed for the restored runner.

The additive counter defaults to zero in older payloads, so checkpoint versions
remain unchanged. A legacy file lacking the count remains readable, but cannot
recover unknown historical alternating parity. No earlier published Anemone
release supported alternating. Both original policies retain their behavior.

## Validation and final review

Coral's reconciliation gate passed 78 tests with one CUDA-only skip and no new
static debt. Anemone passed 679 tests for release-history integration and 726
tests for the alternating/release candidate, with strict Ruff, formatting, Mypy,
Pyright, Pylint, build and package checks passing. Deterministic probes matched
the preserved prototype over 80 selections, apart from documented intentional
compatibility/checkpoint corrections.

Chipiron validation uses a fresh environment with a normally installed candidate
wheel and published dependencies, without editable siblings or source-path
overrides. The installed evaluator-v1/atom-codec smoke and all three original
blockers plus eight checkpoint integration cases pass. The complete normal fast
suite passed **1,291 tests**, with **2 CUDA-only skips and 4 deselected**, in
681.21 seconds. All ten evaluator-v1 tests pass. Ruff, formatting, actionlint,
toolchain, build/Twine, dependency consistency and the static ratchet pass.
The static counts remain Mypy 91 / Pyright 73 / Pylint 45, with no additions or
removals; passing the ratchet does not mean zero inherited analyzer debt.

PR #52 remains a final review checkpoint. Require green CI on its current head
before marking it ready. This task does not merge it or enable auto-merge.

## Preserved work and historical exception

All 51 original dirty/untracked files and the five unique commits on
`feat/morpion-transformer-evaluator-v1` remain preserved. No old branch, tag or
worktree was deleted. External Morpion runs, model bundles and research archives
were not modified or duplicated. Three verified Git bundles plus the original
dirty diffs and file hashes provide durable recovery outside the repositories.

Chipiron 0.2.2's published source differs from its current tag. The explicit
[historical provenance exception](historical_release_provenance.md) records the
original source, wheel hash and preserved archive. No historical tag or ancestry
was rewritten.

The detailed report, per-file feature ledger, test logs, artifact hashes and
zero-loss checks are retained under
`/home/pompote/oldata/victor/reconciliation_execution_2026-09-20/`.
