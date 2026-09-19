# PR #52 dependency unblock audit — 2026-09-18

This follow-up addresses published dependency availability, installed-distribution
validation and inherited static-analysis debt. It does not reopen the general
canonicalization cleanup or change runtime/search/model source.

## Release scope

Coral 0.1.15 is published at `c3a16707bb5f3255e28f07e53dc769be989e4dd5`
([PR #4](https://github.com/victorgabillon/coral/pull/4)). It includes only the
reusable relation scaling and additive attention-mask inference correction from
`40e4e76`, plus focused compatibility tests and release quality checks.
The architecture, weight keys, relation definitions and legacy scale-1 default
are preserved. Type-specific/D4/capacity experiments from the dirty original
checkout are excluded. Exact-head CI passes: 77 tests, one CUDA-only skip,
strict Ruff/format/build, Mypy/Pyright zero, and three inherited Pylint messages
with no additions. Tag `v0.1.15` published through the existing trusted workflow.
Published dependency wheels install with Chipiron's declared dependencies in a
fresh environment without editable/source overrides; their packaged source
hashes match the tested release wheels.

Anemone 0.2.22 is published at `8821d0c2b4711a279f1e29850a30e70db931fb90`
([PR #70](https://github.com/victorgabillon/anemone/pull/70), tag `v0.2.22`).
`build_atoms` was introduced in `61410575c167652a86e6a938f218e7d0a6bd6590`
and already exists on current main. The release adds no runtime source to main:
only version metadata, a checkpoint API contract test and reproducible quality
pins. All 676 tests and strict quality/build gates pass locally and on GitHub.

Six dirty Anemone files are excluded: `checkpoints/payloads.py`, the Linoo
`depth_policy.py`, `linoo.py`, `report.py`, `types.py`, and
`tests/test_linoo_selector.py`. Their `alternating_by_step` policy/default,
selection-step checkpoint state and alternating diagnostics are not part of
this checkpoint API release. Previously committed main refactors remain as
required by the instruction to start from current main.

The full package-only run also exposed a public argument-parser incompatibility:
Parsley cannot call the `LinooDepthSelectionPolicy` type-alias object. Published
Anemone 0.2.23 at `53cd6f7e2d8407dbb2b9c11fc1cff5c5e2ad9cff`
([PR #71](https://github.com/victorgabillon/anemone/pull/71), tag `v0.2.23`)
narrows its fix to string annotations at the public argument and
constructor boundary, followed by the same membership validation expressed
as a typed predicate. Both supported policies and the inverse-depth default
remain unchanged; all 44 search methods match 0.2.22. All 679 tests, strict
static checks and build gates pass, including exact-head GitHub CI and the
trusted release workflow. The public wheel passes the parser/default/checkpoint
contracts outside the checkout. Chipiron pins this release and Coral 0.1.15.
The unrelated alternating implementation, default and checkpoint step changes
remain excluded.

The fresh package check also exposed an inherited Anemone metadata gap:
`import anemone` eagerly imports Graphviz, although Anemone declares it only
under its `debug` extra. This also exists in 0.2.21. Chipiron already declares
`graphviz>=0.20.1`; installing that existing dependency from PyPI makes the
public-package contracts pass. The Chipiron installation needs no undeclared
dependency or source override. A bare Anemone-only install without that extra
still has this limitation.

## Compatibility blocker exposed by installed packages

Chipiron inherits the default from `LinooArgs`. Clean Anemone uses
`inverse_depth`; the excluded dirty implementation changes that default to
`alternating_by_step`. Existing Chipiron tests explicitly require the latter.
This affects runtime behavior, not just a version string or logging spelling.

Tests had fabricated package namespaces pointing to adjacent source trees,
which hid this distinction. Those overrides are removed, with every existing
assertion preserved. Canonical tox now builds/installs the wheel, checks package
import origins, and uses `CHIPIRON_TEST_INSTALLED=1` without sibling PYTHONPATH.
Focused unit-test mocks remain ordinary scoped mocks.

The installed test selection also exposed missing one-match input YAML files in
the wheel. Package-data and sdist metadata now include the three existing configs,
including the neutral GUI launcher config. Their contents and the existing
launcher assertions are unchanged; the installation smoke resolves that resource.

Do not change the selector default, relax assertions, or import dirty siblings
merely to turn CI green. Resolving this blocker requires an explicit separate
scope decision. PR #52 remains draft until genuine package-only tests pass.

## Static analysis

The initial counts are Mypy 91, Pyright 73 and Pylint 45; all 209 diagnostic
identities match the earlier maintenance audit with zero additions/removals.
The [ratchet](static_analysis_baseline/README.md) exposes inherited diagnostics
and rejects additions, extra occurrences and baseline growth. Full strict
`tox -e lint,typecheck` reports remain available. A passing ratchet means no new
debt, not zero analyzer diagnostics. Ruff and tests remain strict.

## Preservation and remote scope

The original three working trees retain their original HEADs, status and all
51 dirty/untracked file hashes (Chipiron 41, Coral 4, Anemone 6). No model bundles,
external Morpion runs, datasets, search policy or bootstrap source is modified.
Release branches/tags are pushed through existing publishers. No main branch
is pushed, no PR is merged and no historical branches are deleted. Main
protection and archival tags remain deferred until PR #52 is genuinely green.
