# Canonicalization verification — September 18, 2026

The maintenance branch is reviewable, but **blocked before canonicalization**.
Passing checks with explicit local dependencies do not establish that the
published dependency set is releasable. See the
[dependency gate](reproducibility.md) and [backlog](../maintenance_backlog.md).

## Environment and commands

Validation uses Python 3.13.0 in an external venv with system site packages, plus
the repository's pinned Ruff 0.15.20, Pyright 1.1.411, mypy 2.1.0, Pylint 4.0.6,
tox 4.56.1 and build 1.5.0. Torch is 2.9.0; the existing Docker recipe declares
Torch 2.8.0 CPU. This local environment is not a fresh installation of all package
requirements. Published-package compatibility is checked separately below.

The equivalent commands from the maintenance checkout are:

```bash
python -m ruff check .
python -m ruff format --check .
python -m tox -e tooling
python -m pyright --pythonpath "$(command -v python)" src/chipiron --outputjson
python -m pyright --pythonpath "$(command -v python)" scripts/record_dependency_state.py tests/maintenance --outputjson
python -m mypy --strict src/chipiron
python -m pylint src/chipiron
actionlint
git diff --check fb0c1269
python -m build --no-isolation --outdir "$EXTERNAL_AUDIT_DIR/dist"
```

For local compatibility tests, the three sibling source roots are explicitly
selected and their state recorded; tests no longer discover adjacent checkouts:

```bash
PYTHONPATH="$CORAL_SRC:$ANEMONE_SRC:$ATOMHEART_SRC:src" QT_QPA_PLATFORM=offscreen \
  python -m pytest -m 'not integration and not external_data' --no-cov -q --maxfail=12
```

Here Coral is clean commit `40e4e76088f93764bedd01d19a16563f00ef20cb`, Anemone is
`7950881f436d3b1ca3dba1d3ed733bb3a43bfd14` with six pre-existing dirty files, and
Atomheart is clean commit `9ced0b0d9166806cadc83aa9ca60a79b1093fa1c`.
Anemone's existing patch and dependency/file hashes are archived externally.
These local source overrides are not embedded in CI or production defaults.

## Results

- Baseline fast tests: 1,268 passed, two failed, four deselected. The failures were
  test-harness issues: a procfs read racing with process exit, and observability
  stubs failing when the learning module had already been imported.
- Final full rerun after those fixes, using the declared minimum pytest 9.0.2:
  **1,275 passed**, four deselected, 95 warnings, in **620.14 seconds**. An earlier
  confirmation with inherited pytest 9.0.0 also passed all 1,275 tests.
- Focused maintenance/process-lifetime/learning checks: **10 passed**.
- Ruff check and format pass across **581 Python files**. Tooling drift guard
  and tox's tooling environment pass. Actionlint 1.7.12 and workflow safety
  contract tests pass. `git diff --check` passes.
- Pyright: **73 errors before and after**, with identical diagnostic identities
  after normalizing checkout paths and shifted line numbers. The new maintenance
  script and two test files have **zero Pyright errors**.
- Mypy: **91 errors in 27 of 436 checked files**, identical before/after. Pylint:
  **45 messages**, identical before/after, score **9.98/10**. These are failing
  inherited gates, not green checks; strict tox settings remain enabled.
- Wheel and source distribution build successfully. The artifacts exclude the
  removed runtime files and old research modules, and retain the intentional
  legacy parameter YAML. Installing the wheel separately and importing outside
  the checkout succeeds with the explicit sibling overrides; the canonical
  Transformer constructs with 25 features, relation scale 0.25 and 106,049
  parameters. No external Morpion training datasets or model bundles were added;
  the deliberately retained legacy chess parameters and test fixtures remain.
- The declared Coral 0.1.14 wheel fails canonical model construction with an
  unsupported `relation_bias_scale` argument. With published Anemone 0.2.21,
  the fast test attempt fails collection on missing
  `anemone.checkpoints.build_atoms`. These remain release blockers.
- [GitHub CI on the implemented code](https://github.com/victorgabillon/chipiron/actions/runs/35350405161)
  at `bf12412b` independently confirms the Anemone collection failure. Its Ruff,
  formatting, tooling and package build steps pass. Mypy reports the same 91
  local diagnostics plus one published-Coral `relation_bias_scale` error (92
  total); Pylint likewise fails on inherited diagnostics and published dependency
  incompatibilities. The complete tox run fails in 552.81 seconds. The subsequent
  verification commit changes documentation only; its own CI status should still
  be checked before any later merge.
- Docker base-image availability and workflow syntax/contracts are checked.
  A full Docker build was not run; its existing initialization downloads data
  and the clean dependency set is already known to be incompatible. The release
  workflow must build and smoke-test its image before it can publish.

## Preservation checks

All 298 `generic-game` commits remain ancestors. The 19 Morpion source/test paths
removed by the earlier cleanup remain absent. All three production Python files
touched by maintenance have identical parsed ASTs to `fb0c1269`; their changes
are formatting only. No rules, model defaults, training protocol or search
semantics were changed.

The original workspace remains at `f545304c`: its Git status and SHA-256 values
for all 41 modified/untracked files match the pre-sprint snapshot. Sibling
repositories and external Morpion experiment artifacts were not edited. No long
research experiment, branch deletion, history rewrite or merge to main ran.

Raw logs, wheel artifacts, dependency records, source patch and forensic Git
snapshots stay outside Git in the dated maintenance audit directory. Only this
compact verification record is committed.
