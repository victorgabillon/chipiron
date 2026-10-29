# Chipiron Python 3.14 modernization — 28 October 2026

The migration uses the complete validated October 27 continual-learning tree.
The final tested executable source is `a6987792796e21210315a2d76358aa48d71bce71` on
`migrate/python314-continual`, descended from report commit
`7e5d8e22728ea0421a580fdfd9af00dbba640034` and executable `fb444210`.
Public main remains `50fb67a42577e195bc565f4d03e38d919640d08a`; it was rechecked
through the GitHub API and was not used as the migration foundation.

The first commit on the new branch is the standalone Python 3.13-compatible
workspace safety patch `5e78a117`. Merge `ff36b52a` then incorporates the existing,
validated Python modernization history without rewriting or squashing it.
Commit `a6987792` adds the full-Dataset regression, completes documentation tooling
and records the reproducible CPU constraints. The original scientific branch,
the prior `feat/python314-modernization` branch at `a41efda6`, and the independent
`fix/morpion-legacy-workspace-boundary` branch remain available.

The only production changes since the earlier tested `ab60c856` wheel are the
legacy-entrypoint guard in `bootstrap_loop.py` and `cycle_validation.py`.
Every neural model, candidate trainer, scientific worker, Dataset/Training
acceptance path, activation implementation and search runner is byte-identical
to that previously validated migration. The completion results at the end of
this report supersede the initial migration lane and package identities below.

The [October 27 report](morpion_continual_bootstrap.md) remains the historical
record, including the summer/current capability matrix, candidate architectures,
continuation policy, 40,934-node engineering run and operator commands. This
report adds native Python 3.14 evidence without replacing that history.

Anemone 0.3.1 is published. The repaired Rust binding 0.1.6 is authorized,
merged and tagged after passing CI. Its PyPI upload is blocked by a missing
trusted-publisher configuration; the CI-built release wheel is validated locally.
A fresh PyPI-only installation needs that release before it can resolve the
final dependency metadata. The existing validated private environment is usable.

## Source and environment isolation

The normal checkout, `/home/pompote/oldata/victor/chipiron`, contained overlapping
research edits at `f545304c24a7889f20471de0944d8097a9d192c8`. This task reused the
existing October 27 worktree at
`/home/pompote/oldata/victor/morpion-scratch/continual-learning-20261027/worktree`
and created an ordinary branch there. It created no worktree or clone. The
original `feat/morpion-continual-learning` reference remains at `7e5d8e2`.

Private environments, caches, screenshots, packages, comparisons and experiments
are under `/home/pompote/oldata/victor/python314-chipiron-20261028`, abbreviated
`E` below. `E/env.sh` records the actual environment variables. The previous
Python 3.13 environment and historical experiments were retained.

The native environment is standard CPython **3.14.10**, GIL enabled and JIT
disabled, on Linux x86-64. Native application tests use the installed wheel with
`PYTHONPATH=.` and `CHIPIRON_TEST_INSTALLED=1`; they do not import an editable
Chipiron tree. Scientific CPU comparisons set OMP, OpenBLAS and MKL threads to
one. Mutable MLflow, leaderboard, Qt, application and temporary outputs are in
the private runtime directory.

## Published dependencies and language changes

Published versions were checked against PyPI metadata on October 28, rather
than inferred from sibling source checkouts. The final environment uses these
public foundations:

| Distribution | Version | Verification |
| --- | --- | --- |
| valanga | 0.3.0 | Public wheel and installed package bytes equal |
| parsley-coco | 0.2.0 | Public wheel and installed package bytes equal |
| atomheart | 0.3.0 | Public wheel and installed package bytes equal |
| algorhino-anemone | 0.3.1 | Public release after the runtime annotation fix below |
| algorhino-coral | 0.2.0 | Public wheel and installed package bytes equal |

The direct runtime dependency is `torch>=2.14.1`; the validated installation is
the stable official CPU wheel **2.14.1+cpu**. All **12,151** members of its
`torch/` package, including native libraries and headers, match that wheel.
Its SHA-256 is
`35103e180214793207f95d7e12e100ecba3ace6f5fb37e1fcefbb07e096ed180`.

Other validated versions include PySide6 6.12.0, NumPy 2.5.4, pandas 3.1.0,
Numba 0.68.1, MLflow 3.17.2, PyArrow 25.0.1, Plotly 7.1.0, Streamlit 1.66.1,
Hugging Face Hub 2.3.0 and CustomTkinter 6.0.0. PyArrow 26 is published but
incompatible with MLflow's `<26` constraint; 25.0.1 is the newest compatible
resolution. The metadata requires PyArrow >=25. The resource sampler's actual
direct dependency, psutil >=7.2.2, is now declared, as are chess and pytest-xdist
in their appropriate dependency groups. `pip check` passes.

Development tools are modern stable releases: Ruff 0.17.3, mypy 2.4.0,
Pyright 1.1.414, Pylint 4.1.3, pytest 9.2.0, coverage 7.17.0,
pytest-cov 7.1.0, tox 4.67.4, Black 26.10.1, build 1.6.1,
pre-commit 4.7.0, setuptools 84.2.1 and wheel 0.48.0. Full transitive versions
and discovery metadata are retained in the evidence directory.

`requires-python`, `.python-version`, classifiers, formatters, static tools,
tox, CI, Docker and release workflows now target Python 3.14. There is no maintained
3.13 path. The operator preparer's exact-version guard and entrypoint guard were
updated too. Existing scientific producer checks remain strict.

Native deferred annotations replace 388 `from __future__ import annotations`
imports. Existing generic protocols and queue types use native parameter syntax;
31 obsolete global TypeVars were removed. A legitimate method-generic TypeVar
remains. Variance and bounds are explicit where the contracts require them.
Runtime-evaluated dataclass annotations keep their real imports so Parsley and
dacite can resolve them. Existing configuration owns the checkpoint and eviction
aliases needed to avoid an import cycle, with old import paths re-exported.

Strict static debt fell from 209 diagnostics to **zero**, without new type
suppressions. Read-only batch and optimizer protocols describe actual consumers;
lazy public APIs now expose truthful static exports. The value trainer calls the
typed public `torch.autograd.backward(loss)` with the same defaults. Unused
cache-metrics arguments were removed; training chunk behavior is preserved.

Ruff lint and formatting pass across the configured tree. Strict mypy checks all
482 production Python files; Pyright and Pylint also report zero diagnostics.
CI and release static gates require zero diagnostics instead of permitting the
archived Python 3.13 baseline. FULL remains required. CPU constraints and job
timeouts accommodate the native environment without weakening assertions.

No compiler or AMP mode was enabled, no multiprocessing default was overridden,
and no new worker coordinator, neural architecture, checkpoint format or model
registry was introduced. The state encoding identifier
`atomheart-0.2.11-dump-state-ref` deliberately remains unchanged: it names the
existing encoding contract, rather than the installed distribution version.
The obsolete test-only Valanga API backfill was removed: tests use the actual
pinned public API. The active maintenance guide now describes Python 3.14 and
zero-diagnostic static gates; dated historical reports retain their original
environment descriptions.

Docker now selects `python:3.14-bookworm`, Torch 2.14.1 CPU and NumPy 2.5.4.
The [official image manifest](https://raw.githubusercontent.com/docker-library/official-images/master/library/python)
lists that Python tag. Registry manifest inspection and `docker build --check`
pass without warnings. A complete image build, whose existing initialization
downloads external engines/data, was not performed or published.

## Anemone 0.3.1 repair and authorized publication

Anemone 0.3.0 put names required by runtime `get_type_hints` behind type-only
imports in six selector argument modules. Native Chipiron configuration parsing
therefore failed with `NameError`. The fix restores those runtime imports and
adds seven regression cases; it changes no search algorithm or schema.

The user explicitly authorized: “Publish Anemone 0.3.1 after CI passes.”
The local fix is `de984c1126ac966884151c2c95485545bfc62843`.
Its installed-wheel validation passed **1,604 tests**, strict static checks,
positive and negative generic-contract checks, formatting and strict package
validation. The seven new cases fail against public 0.3.0 and pass with the fix.

[PR #80](https://github.com/victorgabillon/anemone/pull/80) passed the required
`ci` check at the exact proposed head (run 40675539125). It was merged normally
as `673371f8c6d68be53fb066f0b44ef79f79ff9fb1`; the merged tree equals the tested
tree. Main CI 40676748935 and
[release CI 40676836242](https://github.com/victorgabillon/anemone/actions/runs/40676836242)
both passed. Tag `v0.3.1` points to that merge. Existing trusted publishing
published [Anemone 0.3.1](https://pypi.org/project/algorhino-anemone/0.3.1/).
No protection, required check or publication permission was bypassed.

The freshly downloaded public wheel SHA-256 is
`2e06f724c524bafacc4927e15ed86d4c0bd5b414bf98f1821703fa57942c1f7f`;
the sdist SHA-256 is
`65833235d2b0bf26c91723dddc98e3630c06d2db05469f49d64f91d20a31b0b8`.
All 290 source/typing members match the merge, both public artifacts and the
installed copy. The local candidate was replaced by this genuine PyPI release
before final Chipiron validation. No Chipiron push, merge or publication is
part of this authorization.

## Rust chess binding and native DataLoader correction

The final PR sweep exposed two genuine Python 3.14 failures outside Morpion.
`shakmaty_python_binding` 0.1.5 builds from its public sdist with PyO3 0.19.2,
but `MyMove(uci, board)` segfaults even in an isolated interpreter. The native
0.1.6 candidate updates PyO3 to stable 0.30.1 and the module initialization
signature to its supported `Bound<PyModule>` API. It retains the published
0.1.5 chess logic and underlying Shakmaty versions; it uses no ABI-forward-
compatibility flag or replacement move-construction path. Maturin is 1.15.0.

The public GitHub master was still version 0.1.4 and lacked some published 0.1.5
move-modification logic. The candidate therefore preserves the genuine PyPI
0.1.5 source rather than downgrading to that obsolete tree. Special moves,
500 deterministic plies compared with python-chess, and the five-case installed
chess/stalemate matrix pass. Candidate source, wheel, review patch, CI/release
workflow and regression evidence are retained under `E/shakmaty-diagnosis`,
`E/shakmaty-candidate` and `E/shakmaty-final.patch`.

The user separately authorized: “Publish Rust binding 0.1.6 after CI passes.”
[PR #1](https://github.com/victorgabillon/shakmaty_python_binding/pull/1) passed CI
40703193567 at exact head `41e37163505a4779eaff4b9dd7c7c99aaf71a7b2` and merged
normally as `986e6d907ae563afae2695044fc6bc3952016727`. Its merged tree equals
the tested tree. Master CI 40703671112 passed; annotated tag `v0.1.6` points to
that merge. The minimum Rust declaration is 1.85, matching PyO3's metadata;
correcting this declaration preserves the exact native binary used by FULL.

The [tag workflow](https://github.com/victorgabillon/shakmaty_python_binding/actions/runs/40703927007)
builds and tests successfully, but its publication job reports
`invalid-publisher`: a valid GitHub token has no corresponding PyPI publisher.
Release status: **CI-tested, merged and tagged; awaiting PyPI account setup**.
The needed GitHub publisher settings are owner `victorgabillon`, repository
`shakmaty_python_binding`, workflow filename `ci.yml`, environment `pypi`.
The [PyPI publisher setup](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)
requires the project owner to register that identity. The available connector
cannot edit PyPI account settings. No branch protection or publishing check was
bypassed; only the failed upload needs retrying after setup.

`E/shakmaty-release-candidate.json` records the complete source/artifact identities;
the initial local candidate native library SHA-256 is
`80ae7e5c848ed6c21a470938d0e9bf686cb0ab6d584bca5fdf87fc8c2cebffbb`.
All six core source/build/typing files match the candidate sdist exactly.

The actual tag-CI artifacts are retained under `E/shakmaty-ci-release`. Their
six core files also match the immutable merge commit exactly. The release wheel
SHA-256 is `aba03469af3b22655e6a336124729ff850a39b6d2f9b9e1478a043bb0059e4ae`;
the sdist SHA-256 is `38035914bac219e9becbb54b0a871c811d5105c8f294a7c3fc92ee6094117cb7`.
All four installed package members match this wheel. With it installed, the
native constructor/special-move/500-ply checks pass, **14 focused Chipiron tests
pass in 31.39 s**, and the complete Morpion search/27-update training snapshot is
again exactly equal to the Python 3.13 reference. These are CI release artifacts,
not yet public PyPI artifacts. `E/shakmaty-publication.json` records the release
and upload status, and `E/native314-ci-release/snapshot.json` records the final
numerical comparison.

The supervised chess DataLoader also failed to pickle converter functions
defined inside their factory. The same four conversion functions now live at
module scope, preserving tensor construction and parameters while allowing
native forkserver serialization. All five representation choices pass exact
tensor checks after a pickle round-trip, and the real supervised training test
runs its DataLoader worker successfully. The focused group passes seven cases.
The existing process context and worker count remain unchanged. Chipiron now
requires Rust binding >=0.1.6 so a fresh installation cannot silently select the
known crashing implementation.

## Initial migration package identity (ab60c856)

Chipiron wheel and sdist passed strict Twine validation. All **483** production
Python/typing files (482 Python files plus `py.typed`) match Git `ab60c856`, the
wheel, sdist and installed package byte for byte. There are no extra installed
Python files. The frozen Morpion feature baseline is also present and unchanged
in the sdist. Installed smoke checks cover all five foundation origins, packaged
YAML/resources, typing marker and evaluator/checkpoint contracts. Evaluator-v1
retains 106,049 parameters and relation-bias scale 0.25.

| Artifact | SHA-256 |
| --- | --- |
| Initial migration wheel | `9f4bd15829b22d2ff6491f24a353f75bf478df5cf15b11075f58c7a1d2440326` |
| Initial migration sdist | `5dd4945e5b319a95dfc0170db20aa8c14fb1caa208532d7d08885a03ccfb32f0` |

The initial migration manifest is `E/installed-source-verification-final.json`. The earlier
`ed202ad` artifacts used by the scientific comparisons and pilot remain in
`E/dist` with their original `E/installed-source-verification.json`. All Morpion
production bytes remained unchanged through `ab60c856`; the completion adds only
the explicit legacy-entrypoint rejection guard. Public foundational wheel/member identities
are in `E/public-foundations-verification.json`;
`E/torch-installed-verification.json` records the official Torch comparison.

## Initial scientific equivalence and retained continual learning

The independent snapshot harness compares the October 27 installed Python
3.13.5/Torch 2.14.0 implementation with the final installed Python
3.14.10/Torch 2.14.1 implementation and public foundations. The complete final
snapshot is **exactly equal**, not approximately equal.

* Search performs 32 Growth steps, cold-restores, then performs eight more, with
  a sufficiently high node cap and an assertion that continuation advances.
  Final trees have 783 nodes without rollouts and 1,860 with two-move rollouts.
  Complete scientific payload digests match, including continuation.
* All nine original candidates train for their normal five epochs over three
  continuing rounds. Initial/final raw tensors, cumulative Adam steps and
  optimizer tensor hashes match for all 27 updates.
* An AST comparison of eight critical files finds **158 unchanged function
  implementations** after excluding annotations and imports. This is a semantic
  body comparison, not a claim that migrated files are byte-identical.

Evidence: `E/baseline313-v2/snapshot.json`,
`E/native314-ci-release/snapshot.json`, `E/scientific-equivalence-final.json` and
`E/continual-implementation-ast-audit.json`. An earlier exploratory 128-node
fixture did not actually advance after restore; it is retained as diagnostic
evidence and is not used for the equivalence claim.

The default remains `continue_from_previous_accepted_weights`, with
`continue_adam`, separate candidate lineages, constant configured learning rate
and recorded deterministic seeds/RNG provenance. Losing candidates continue
from their own latest accepted weights. Corrupt or incompatible parents fail
closed. Acceptance, stale-worker fences, retry idempotence, incumbent comparison,
atomic activation, complete refresh and exact selector/search/RNG continuation
remain the October 27 mechanisms.

## Normal games and interfaces

The complete existing chess integration module passed **3 tests** in 28.11 s,
including its internal match-configuration matrix, random reproducibility and
Python/Rust implementation equivalence. Actual neural evaluators and tree
search ran. A separate earlier two-test run overlaps this module and is not
added to its count.

Two permanent native-runtime regressions verify nested configuration annotations
and an actual `PlayerProcess` under Python 3.14's Linux default `forkserver`.
The latter starts the real Integer Reduction player, exchanges pickled requests
and moves, and checks clean process shutdown. These enter FAST, PR and FULL.

Real Qt 6.12 widgets ran with xcb under Xvfb for chess, checkers, Integer
Reduction and Morpion. Each rendered valid SVG/current state; the pause button
dispatched the actual GUI command, and Morpion's action-view toggle rendered.
All four CustomTkinter launchers built and processed updates. Chess and Morpion
screenshots were visually inspected. Artifacts are `E/gui/results.json` and
`E/gui/*.png`; `E/gui_smoke.py` records the checks.

Observed external warnings remain visible: python-chess uses a deprecated
asyncio policy API, PyTorch warns about legacy TorchScript on Python 3.14, and
Coral's existing `norm_first` transformer configuration disables a nested-tensor
optimization. Actual chess evaluation/training paths pass. The explicit-fork
process-death tests also produce Python's multithreaded-fork deprecation warning;
the actual default-context player test uses forkserver. No warning suppression
or alternate execution mode was added to conceal these observations.

Streamlit AppTest exercised Overview, Record, Tree, Evaluator and Operations
against the completed native pilot, with **zero exceptions on all five pages**.
Evidence: `E/dashboard-smoke.json` and `.log`. These are automated interface smoke
checks, not an assertion that every interactive human workflow was exercised.

## Native four-worker experiment

The existing prepare/run-continuous launcher ran a fresh Greek-cross tree with
zero evaluator and all nine normal five-epoch candidates. It used 32-step Growth
batches, two-move rollouts, 128 Dataset rows/chunk, and cadence one. Its original
finite limits were 5,000 unique nodes, 900 elapsed seconds, 300 seconds per
worker, 8,192 MiB group RSS, 8,192 MiB minimum RAM/disk, 4,096 MiB workspace,
2,048 MiB scratch and 15-second shutdown grace. Limits were never extended.

This experiment's exact producer is the `ed202ad` installed wheel retained in
`E/dist`. Its Morpion source is byte-identical to `ab60c856`; the later
production correction concerns chess converter serialization. The initial migration lanes
exercise the complete Morpion scientific/recovery matrices on `ab60c856`.

| Observation | Native result |
| --- | --- |
| Final tree | 4,998 unique nodes, generation 11 |
| Scientific publications / accepted Datasets | 11 / 7 |
| Accepted Training rounds | 5, at generations 2, 3, 5, 6, 8 |
| Candidate bundles | 45; every candidate continues its own weights and Adam |
| Actual model activations | 3, at generations 4, 7, 10 |
| Cumulative monitored wall time | 255.855 s across two sessions |
| Peak simultaneous process-group RSS | 1,224.965 MiB |
| Peak workspace / scratch allocation | 64.875 / 7.094 MiB |
| Minimum available RAM / free disk | 69,486.070 / 236,498.590 MiB |
| Independent best terminal replay | 22 moves, 58 points, node 1394 |
| Final reason | `next_complete_expansion_exceeds_unique_node_budget` |

The independent existing scientific audit verifies all 45 bundles, raw parent
and successor weights and Adam ancestry. It independently recomputes the first
two complete activations' neural values, backed-up values, PV/certainty and full
checkpoint payload; all three activation completion receipts are present.
The round without sufficient improvement retains the incumbent.

| Candidate group | Cumulative Adam steps at accepted generations 2, 3, 5, 6, 8 |
| --- | --- |
| Each of the eight linear/MLP candidates | 10, 20, 30, 40, 50 |
| Entity-token transformer | 65, 130, 195, 260, 325 |

Two independent cold restores and offline Growth continuations yield the same
complete scientific payload hash:
`88dc8efaa166acd5b980d31ff9eb792c87c1d9e6f36fe4f8862133354596ac1d`.
Both sessions confirm clean worker shutdown. Restarting the completed experiment
leaves the authoritative state and cumulative resource usage byte-identical.

The first operator stop happened at generation 2, 1,395 nodes, when the genesis
model name was projected from null to `seed`. The helper initially labeled this
“after committed activation”; that label is inaccurate. This is Growth
stop/resume evidence, not interruption during activation. The classification is
recorded in `E/pilot314-stop-classification.json`. Dedicated activation recovery
is covered separately by the real-M1-drain regression in the final suites.

Evidence: `E/pilot314-transcript.json`, `E/pilot314-audit.json`, resource sessions
and the retained `E/runtime/pilot314` workspace. Timings are observed native
measurements, not a matched speedup benchmark. Validation MSE and the terminal
record do not establish a causal playing-strength improvement. No native 50k
run or 200k launch is claimed; the previous 40,934-node run belongs to the
October 27 Python 3.13 report.

## Initial migration validation (ab60c856)

The initial FAST, PR and one FULL invocation ran serially against the installed
`ab60c856` wheel and public foundations. Their selections overlap and counts must
not be added. The normal exclusions remain integration/external-data; the
normal chess integration module was exercised separately above.

Those initial lanes used the explicitly identified local Rust 0.1.6 candidate,
whose native library bytes are identical in PR and FULL. It is not described as
a published distribution. All five foundational libraries are genuine public
releases.

| Lane | Selected | Passed | Skipped | Console elapsed |
| --- | --- | --- | --- | --- |
| FAST | 218 | 218 | 0 | 94.53 s |
| PR | 2,884 | 2,882 | 2 | 949.84 s |
| FULL | 3,048 | 3,046 | 2 | 1,347.21 s |

The two skips in PR and FULL are the existing CUDA-only device-transfer checks in the
relational model and multi-input supervised batch tests; CUDA is unavailable in
the validated CPU environment. The real-M1-drain activation recovery case passes
in PR in 114.32 s and FULL in 110.06 s. CUDA behavior is not claimed as validated.

The exact selected identities, setup/call/teardown outcomes and worker
collections are retained in `E/{fast,pr,full}-metrics.json`; each lane also has
JUnit XML and a complete log. The recorder fails on missing or duplicated
execution evidence. FULL retains line, branch and subprocess coverage.
All lanes exit zero with no execution-identity errors. Full statement coverage
is 83.25%, branch coverage 69.50%, and combined coverage 80.35%; the standalone
artifact is `E/coverage-final.json`. The seven added runtime/serialization cases
account for the increase from the October 27 FULL result of 3,039 passes.

Final static and packaging evidence: `mypy-final.log`, `pyright-final.json`,
`pylint-final.json`, `ruff-final.log`, `format-final.log`,
`toolchain-final.log`, `packaging-final.log`, `twine-final.log`, `pip-check-final.log`
and `installed-contracts-final.log`, all under `E`. The final conftest correction
also passes Ruff lint/format and the toolchain drift guard. Earlier FAST/PR
invocations are retained under `E/pre-final` and
`E/failed-pr-rust-and-pickle` as diagnostic evidence, including the actual
Rust segfault and DataLoader serialization failure. They are excluded from the
final lane results. That diagnostic PR also passed the real-M1-drain activation
recovery case in 108.72 s; the final lanes verify it again.

## Reproduction and lifecycle review

The validated environment can be used directly:

```bash
cd /home/pompote/oldata/victor/morpion-scratch/continual-learning-20261027/worktree
source /home/pompote/oldata/victor/python314-chipiron-20261028/env.sh
export PYTHONPATH=. CHIPIRON_TEST_INSTALLED=1
make test-fast PYTHON="$PYTHON_BIN"
make test-pr PYTHON="$PYTHON_BIN"
make test-full PYTHON="$PYTHON_BIN"
QT_QPA_PLATFORM=xcb "$PYTHON_BIN" -m chipiron.scripts.main_chipiron
```

After Rust binding 0.1.6 is published, a fresh environment can use CPython 3.14
and build this commit's wheel; install
its test/lint/typecheck/dev/dashboard extras with
`.github/ci-cpu-constraints.txt` and the official
`https://download.pytorch.org/whl/cpu` extra index. Do not copy the old 3.13
environment or substitute unpublished sibling packages. The retained freeze and
artifact manifests identify the exact validated resolution.

The existing [operator instructions](morpion_continual_bootstrap.md) and launcher
remain available. A new Python 3.14 workspace must be prepared with the native
environment so its scientific producer identity is correct; do not mutate an
old experiment's provenance to force it through version checks. Inspect the
requested budgets before manually preparing or launching a larger run. This
task did not launch the 200k experiment.

Final preservation checks pass. The normal checkout retains its original HEAD,
branch and status; all 41 modified/untracked file sizes and SHA-256 values match
the pre-task inventory. All existing worktree registrations remain present and
none were added. The October 27 branch still points to `7e5d8e2`. All **487**
recorded protected inputs under `big_run_01` and
`generic_linoo_fresh_with_bigrun_models_v1` retain their original sizes and hashes.
These are the recorded input identities, not a claim that every historical
artifact was exhaustively inventoried. Anemone's original migration branch is
also unchanged, and its authorized fix branch is clean. Evidence is
`E/preservation-final.json` and `E/protected-inputs-final.json`.

The existing worktree is retained because it contains the reviewable migration
branch. The native environment and immutable evidence are retained for
reproduction. The shared CPython installation is its interpreter base and must
remain available. Caches and the diagnostic package-export directory are
regenerable; they may be removed separately after review. Existing checkouts,
the old environment and historical scientific artifacts were not removed.

The migration delta before this report, from the October 27 report commit, is
537 files, 1,324 insertions and 1,817 deletions, largely annotation/import
modernization and obsolete test backfill removal. Production source alone is
926 insertions and 1,171 deletions (net -245 lines). Those figures describe the initial migration; the completion delta is recorded below.
Chipiron remains a local, unpublished branch. Publication would require separate
user authorization. The remaining installation blocker is PyPI's missing
trusted publisher for the authorized Rust binding 0.1.6 release; the source,
merge, tag, CI artifacts and local validation are complete.


## Completion: architecture audit and isolated P0 correction

The complete 20-section architecture audit supplied by the user was read,
including its source references. Its retained copy is
`/home/pompote/Downloads/chipiron_audit.md`, SHA-256
`0bcf2f4bc4f9b0e1138e6ed613cb591bee2037ddb1cded1350c9e6d2dfbf2d63`.
The observed defect matches F0: the public single-process loop accepted default
arguments pointing at scientific authority and could overwrite configuration
before loading run state, including with `max_cycles=0`.

The rejection-only guard checks the actual persisted publication marker,
publication namespaces 1/2/3 (including interrupted genesis), and the caller's
state for the direct cycle API. It runs before directory/configuration writes,
memory diagnostics or any runner call. It neither adopts nor repairs a workspace.
Twelve new regression cases cover real scientific genesis, zero/one loop cycles,
the direct cycle API, stale caller state, interrupted publication, incomplete
markers and a genuine legacy control. They compare all file bytes and file/directory
modification times and assert no runner calls. Before the fix, eleven rejection
cases fail and the legacy control passes; with it, all 66 new/existing loop tests
pass on Python 3.13.5 in 35.63 s and on Python 3.14.10 in 25.35 s.

Commit `5e78a117` starts directly at `7e5d8e2` and changes only two production
files, its regression test and a backport note. It can be independently
cherry-picked onto a development branch descended from the frozen foundation.
See [the backport instructions](morpion_legacy_workspace_boundary.md). No frozen
installed package or active experiment was patched in place. Validation used the
retained 3.13 interpreter/dependencies read-only with bytecode disabled and all
mutable runtime output under `E`.

F1–F9 and the audit's P1/P2/P3 consolidation proposals remain deferred. In
particular, candidate orchestration, eager/streaming kernels, record/frontier
codecs, legacy model formats and public facades remain intact. Dashboard
projection-authority/cache issues remain separate follow-up work; passing GUI
and dashboard smoke tests does not claim those audit findings were corrected.

## Completion: full Dataset and documentation support

The new exhaustive regression trains every one of the existing nine candidates
on all **513 rows**, with `max_rows=None` and decoding chunks bounded to **31**
rows. All nine report exactly 513 total samples and nonempty train/validation
partitions. The fixture uses one epoch to bound test cost and retains each
candidate's architecture; it introduces no sampling or optimizer policy.
The focused test passes in 15.29 s and also enters the final FULL lane.
The production Dataset and Training row limits still default to `None`;
`training_publication` forwards the admitted `training_max_rows` unchanged.
The existing accepted-reference, canonical-split and continuation suites remain
active. This is compatibility validation, not a full-Dataset cadence study or
a claim about the cost of training at 200k nodes.

Read the Docs now selects Python 3.14, which is supported by its
[official configuration](https://docs.readthedocs.com/platform/stable/config-file/v2.html).
Sphinx source-path resolution is independent of the caller's working directory.
The complete declared `docs/requirements.txt` installation passes under the
216 runtime/development constraints. It adds only `myst-parser==5.1.0`,
`numpydoc==1.11.0` and `mdit-py-plugins==0.6.1`; no previously validated
package is replaced. `E/completion/documentation-constraints.txt` pins those
additional documentation packages, and `packages-final.txt`/`pip-inspect-final.json`
record the complete final environment. `pip check` remains clean.
The active Sphinx 9.2 HTML build succeeds with **198 warnings**; these concern
existing generated API references, missing documentation targets and static assets.
The [maintenance backlog](../maintenance_backlog.md) already identifies the
obsolete `chess_env`/`treevalue` API pages for a dedicated documentation pass.
This migration does not claim a warning-free documentation build or silently
suppress those warnings. Evidence: `E/completion/sphinx.log` and `sphinx/`.

## Completion: installed package and regression results

All lanes ran serially against the installed `a6987792` wheel in the single
retained CPython 3.14.10 environment. No editable Chipiron or sibling source
substitution was used. All five foundations are public PyPI distributions;
Rust 0.1.6 remains the exact CI artifact identified above.

| Lane | Selected | Passed | Skipped | Recorded pytest time |
| --- | --- | --- | --- | --- |
| FAST | 230 | 230 | 0 | 95.02 s |
| PR | 2,896 | 2,894 | 2 | 1031.70 s |
| FULL | 3,061 | 3,059 | 2 | 1503.90 s |

Selections overlap and must not be added. All lane exit codes are zero, with no
execution-identity errors. The two existing CUDA-only skips remain explicit.
The twelve P0 cases account for the increase in FAST/PR, and those twelve plus
the one full-Dataset case account for FULL's increase. No previous test was
removed. The real M1-drain activation-recovery test passes in both PR and FULL.
Detailed node IDs, worker collection evidence, setup/call/teardown outcomes,
JUnit and coverage are in `E/completion/{fast,pr,full}*` and `coverage.json`.

Final FULL statement coverage is **83.25%**, branch coverage **69.51%** and
combined coverage **80.36%**. The counts are 28,806 covered statements out of
34,600 and 6,412 covered branches out of 9,224. These are measured coverage,
not a claim of 100% coverage.

Strict mypy, Pyright and Pylint all report zero diagnostics across 482 production
Python files. Ruff lint and formatting pass across 886 configured Python files;
the toolchain coherence guard and `git diff --check` pass. No new suppressions
were introduced. The inherited scientific baseline and GUI/process results above
remain applicable: the completion changes only rejection at the legacy boundary.

All 483 production source/typing members match committed Git, wheel, sdist and
installed bytes; no extra Python files are installed. Frozen Morpion reference
features remain in the sdist. Strict Twine, `pip check`, installed resources and
all six package/evaluator/checkpoint contract checks pass.

| Final artifact | SHA-256 |
| --- | --- |
| Wheel | `1febf69f91763a6c116a22fc46eb2e2a155c538490c46e2371b338b8338ebf6e` |
| Sdist | `143bc9b1d54779800783ac33d0bcf525f012ef8da0c7b17c787b6853a2805a92` |

`E/completion/installed-source-verification.json` is the final package manifest.
The original artifacts/evidence under `E/dist-final` remain available and are not
overwritten. The sdist's existing omission of `tests/conftest.py` remains explicit;
the test lanes use committed repository tests against the installed wheel.

The tiny upstream python-chess Syzygy fixtures total 14,080 bytes and live only
under `E/completion/syzygy`. Their Git blob IDs and SHA-256 hashes are verified.
Python and native Rust adapters agree on WDL, DTZ and White's value for four
KQvK/KRvK positions, both sides to move; both return legal best moves. This
closes a blind spot in the existing test, which can return without probing if
large external tablebases are absent. Evidence: `syzygy-verification.json`.
It is not a validation of every five/seven-piece tablebase or another platform.

## Complete direct dependency comparison

The old column is the installed frozen Python 3.13 reference, not obsolete public-main metadata. Exact old/new requirements, latest public metadata, wheel filenames and hashes are recorded in `E/completion/dependency-review.json`. All 50 current direct runtime, optional and build requirements were checked against PyPI on October 28.

| Distribution | Frozen reference | Validated Python 3.14 | Latest published stable | Groups |
| --- | --- | --- | --- | --- |
| [algorhino-anemone](https://pypi.org/project/algorhino-anemone/) | 0.2.26 | 0.3.1 | 0.3.1 | runtime |
| [algorhino-coral](https://pypi.org/project/algorhino-coral/) | 0.1.15 | 0.2.0 | 0.2.0 | runtime |
| [atomheart](https://pypi.org/project/atomheart/) | 0.2.11 | 0.3.0 | 0.3.0 | runtime |
| [black](https://pypi.org/project/black/) | not installed | 26.10.1 | 26.10.1 | dev |
| [build](https://pypi.org/project/build/) | 1.5.0 | 1.6.1 | 1.6.1 | dev |
| [chess](https://pypi.org/project/chess/) | 1.11.2 | 1.11.2 | 1.11.2 | runtime |
| [coverage](https://pypi.org/project/coverage/) | 7.16.2 | 7.17.0 | 7.17.0 | test |
| [customtkinter](https://pypi.org/project/customtkinter/) | 6.0.0 | 6.0.0 | 6.0.0 | runtime |
| [dacite](https://pypi.org/project/dacite/) | 1.9.2 | 1.9.2 | 1.9.2 | runtime |
| [gdown](https://pypi.org/project/gdown/) | 6.4.2 | 6.4.2 | 6.4.2 | runtime |
| [gprof2dot](https://pypi.org/project/gprof2dot/) | 2025.4.14 | 2025.4.14 | 2025.4.14 | runtime |
| [graphviz](https://pypi.org/project/graphviz/) | 0.21 | 0.21 | 0.21 | runtime |
| [huggingface-hub](https://pypi.org/project/huggingface-hub/) | 2.2.0 | 2.3.0 | 2.3.0 | runtime |
| [ijson](https://pypi.org/project/ijson/) | 3.6.0 | 3.6.0 | 3.6.0 | runtime |
| [matplotlib](https://pypi.org/project/matplotlib/) | 3.11.2 | 3.11.2 | 3.11.2 | runtime |
| [mlflow](https://pypi.org/project/mlflow/) | 3.17.1 | 3.17.2 | 3.17.2 | runtime |
| [mypy](https://pypi.org/project/mypy/) | 2.1.0 | 2.4.0 | 2.4.0 | typecheck |
| [numba](https://pypi.org/project/numba/) | 0.68.0 | 0.68.1 | 0.68.1 | runtime |
| [numpy](https://pypi.org/project/numpy/) | 2.2.6 | 2.5.4 | 2.5.4 | runtime |
| [orjson](https://pypi.org/project/orjson/) | 3.13.0 | 3.13.0 | 3.13.0 | runtime |
| [pandas](https://pypi.org/project/pandas/) | 2.3.2 | 3.1.0 | 3.1.0 | runtime |
| [parsley-coco](https://pypi.org/project/parsley-coco/) | 0.1.36 | 0.2.0 | 0.2.0 | runtime |
| [platformdirs](https://pypi.org/project/platformdirs/) | 4.13.5 | 4.13.7 | 4.13.7 | runtime |
| [plotly](https://pypi.org/project/plotly/) | 7.1.0 | 7.1.0 | 7.1.0 | runtime |
| [pre-commit](https://pypi.org/project/pre-commit/) | not installed | 4.7.0 | 4.7.0 | dev |
| [psutil](https://pypi.org/project/psutil/) | 7.2.2 | 7.2.2 | 7.2.2 | runtime |
| [pyarrow](https://pypi.org/project/pyarrow/) | 25.0.1 | 25.0.1 | 26.0.0 | runtime |
| [pylint](https://pypi.org/project/pylint/) | 4.0.6 | 4.1.3 | 4.1.3 | lint |
| [pyright](https://pypi.org/project/pyright/) | 1.1.411 | 1.1.414 | 1.1.414 | typecheck |
| [pyside6](https://pypi.org/project/pyside6/) | 6.9.2 | 6.12.0 | 6.12.0 | runtime |
| [pytest](https://pypi.org/project/pytest/) | 9.1.1 | 9.2.0 | 9.2.0 | test |
| [pytest-cov](https://pypi.org/project/pytest-cov/) | 7.1.0 | 7.1.0 | 7.1.0 | test |
| [pytest-xdist](https://pypi.org/project/pytest-xdist/) | 3.8.0 | 3.8.0 | 3.8.0 | test |
| [pyyaml](https://pypi.org/project/pyyaml/) | 6.0.3 | 6.0.3 | 6.0.3 | runtime |
| [rich](https://pypi.org/project/rich/) | 15.0.0 | 15.0.0 | 15.0.0 | runtime |
| [ruff](https://pypi.org/project/ruff/) | 0.15.20 | 0.17.3 | 0.17.3 | lint |
| [setuptools](https://pypi.org/project/setuptools/) | 84.2.0 | 84.2.1 | 84.2.1 | build |
| [shakmaty-python-binding](https://pypi.org/project/shakmaty-python-binding/) | 0.1.5 | 0.1.6 | 0.1.5 | runtime |
| [sortedcollections](https://pypi.org/project/sortedcollections/) | 2.1.0 | 2.1.0 | 2.1.0 | runtime |
| [sphinx](https://pypi.org/project/sphinx/) | not installed | 9.2.0 | 9.2.0 | dev |
| [sphinx-autodoc-typehints](https://pypi.org/project/sphinx-autodoc-typehints/) | not installed | 3.17.4 | 3.17.4 | dev |
| [sphinx-rtd-theme](https://pypi.org/project/sphinx-rtd-theme/) | not installed | 3.1.0 | 3.1.0 | dev |
| [streamlit](https://pypi.org/project/streamlit/) | 1.64.0 | 1.66.1 | 1.66.1 | dashboard |
| [torch](https://pypi.org/project/torch/) | 2.14.0+cpu | 2.14.1+cpu | 2.14.1 | runtime |
| [torchinfo](https://pypi.org/project/torchinfo/) | 1.8.0 | 1.8.0 | 1.8.0 | runtime |
| [tox](https://pypi.org/project/tox/) | not installed | 4.67.4 | 4.67.4 | dev |
| [twine](https://pypi.org/project/twine/) | not installed | 7.0.0 | 7.0.0 | dev |
| [types-pyyaml](https://pypi.org/project/types-pyyaml/) | 6.0.12.20260906 | 6.0.12.20260906 | 6.0.12.20260906 | typecheck |
| [valanga](https://pypi.org/project/valanga/) | 0.2.5 | 0.3.0 | 0.3.0 | runtime |
| [zstandard](https://pypi.org/project/zstandard/) | 0.25.0 | 0.25.0 | 0.25.0 | runtime |

PyArrow 25.0.1 is the only validated direct dependency deliberately below the latest stable version: MLflow 3.17.2 requires `<26`. The Rust binding is a separately identified CI candidate at 0.1.6 while PyPI still serves 0.1.5; it must not be described as public. Dependencies with unchanged installed versions already match the latest compatible stable release. Anemone 0.3.1 supersedes the requested 0.3.0 because the latter has the reproduced runtime-annotation defect described above.

The existing `.github/ci-cpu-constraints.txt` now records 216 exact distribution versions for the validated Linux x86-64 environment. It contains no local wheel URLs, and the toolchain drift check compares its tool pins with `pyproject.toml`. The pending Rust release remains an explicit resolver blocker; the constraints do not hide it. The retained PyPI artifact hashes and official CPU Torch wheel identify the actual validated binaries.

Numba 0.68.1 requires NumPy `<2.6,>=1.22` and llvmlite `<0.51,>=0.50.0dev0`; the installed versions are NumPy 2.5.4 and llvmlite 0.50.0. pandas 3.1 requires NumPy >=2.3.3 on Python 3.14. The native smoke check verifies these requirements, actually compiles and executes a Numba nopython kernel, and round-trips nullable string/float/bool data through pandas and PyArrow exactly. Evidence: `E/completion/runtime-stack.json`. Chipiron currently contains no Numba-decorated production function, so this is a dependency capability check, not a claim about a nonexistent application kernel.

## Published foundation identities

| Distribution | Version | Public wheel SHA-256 |
| --- | --- | --- |
| valanga | 0.3.0 | `94b4b9e2b5b45890378416d936d956afda38f7402b726d14b264dc714e475e93` |
| parsley-coco | 0.2.0 | `b7e5e40a5815f381d4c84cca9c733655bcebfba1deca2e659112bee2a8e5fd98` |
| atomheart | 0.3.0 | `e4472f20ebdaad80cb65dafc0a09ed1fb9f03b6870eda363112c403d945244b1` |
| algorhino-anemone | 0.3.1 | `2e06f724c524bafacc4927e15ed86d4c0bd5b414bf98f1821703fa57942c1f7f` |
| algorhino-coral | 0.2.0 | `67048523ba4029818f789f405a364cbbbaef2d8418d4fab6d82a0cbb8fc1f73d` |

All six required Anemone scientific commits are ancestors of the verified public 0.3.1 merge: `74bbd12` (certificates/effective values), `5f43821` (bounded restore), `c015194` (shared score transform), `90c7ef4` (ordered continuation), `37e48c4` (streamed checkpoint construction), and `82b169f` (state-free max comparison). `E/completion/anemone-scientific-ancestry.json` records exact identities. Public wheel/source/installed equality was already verified for that immutable merge.

## Release-note review and compatibility focus

The [NumPy 2.5 release notes](https://numpy.org/doc/stable/release/2.5.0-notes.html) describe expired APIs and removal of distutils. No production/test use of `numpy.distutils` or the removed `np.int`/`np.float`/`np.str` aliases was found. The exact neural/reference comparisons and compiled Numba probe check the installed numerical stack.

The [pandas 3.1 notes](https://pandas.pydata.org/docs/whatsnew/v3.1.0.html) describe string/Arrow fixes and changed categorical inference; [Copy on Write](https://pandas.pydata.org/docs/development/copy_on_write.html) is another behavior boundary. Explicit nullable-string/float/bool Arrow round-trip checks and the existing dashboard/learning tests exercise the actual integration. No global pandas compatibility mode was introduced.

[Numba 0.68.1](https://numba.readthedocs.io/en/stable/release/0.68.1-notes.html) fixes reduction and indexing regressions and aligns accumulator dtypes with NumPy. It is selected with its published NumPy/llvmlite bounds, and the nopython signature is verified after actual execution.

[Qt for Python release notes](https://doc.qt.io/qtforpython-6/release_notes/pyside6_release_notes.html) were reviewed alongside the published 6.12 wheel metadata. The actual GUI event/render tests remain the application evidence; free-threaded or alternative Qt/process execution modes were not enabled. Minor dependency updates retain their existing application APIs and are covered by the installed game, model-loading, training, logging and dashboard regressions. No downstream release-note performance claim is presented as a Chipiron speedup.

## Completion: matched scientific traces and measured runtime

Three fresh executions of the same retained oracle finish successfully. The
original Python 3.13 environment is read-only; all output goes under
`E/completion`. The intermediate cohort imports a read-only source archive of
`7e5d8e2`; the final cohort imports the installed `a6987792` wheel.

| Requested comparison | Source / stack | Result |
| --- | --- | --- |
| Frozen reference | Original source, Python 3.13.5, Torch 2.14.0 CPU and frozen dependencies | Pass |
| Interpreter alone | Original source plus all original dependencies on 3.14 | Not isolated: the cp313 native wheels cannot be reused, and old Rust 0.1.5 crashes on 3.14 |
| New dependencies | Original source, Python 3.14.10, public foundations, Torch 2.14.1 CPU, Rust 0.1.6 CI wheel | Search/training pass; original scientific export rejects changed dependency identities |
| Final modernization | Installed final source, same Python 3.14 dependency set | Search, training and atomic activation pass |

The reference, new-dependency and final cohorts have **exactly identical**
scientific snapshots. They also match the previously frozen
`E/baseline313-v2/snapshot.json`; no fixture was regenerated to accommodate a
disagreement. Search performs 32 steps and eight genuine continuation steps,
warm and cold, both with and without rollouts. The resulting 783/1,860-node
trees retain their complete checkpoint/selector/order/RNG state. All six
original compressed checkpoint files and the seventh checkpoint after complete
model refresh are byte-for-byte equal across the three cohorts.

All nine original architectures train for their normal five epochs in each of
three continuing rounds. Initial/final weights, state dictionaries, cumulative
Adam counters and optimizer moments match exactly for all 27 candidate-rounds.
The instrumentation additionally compares **165 loss-forward
calls** (predictions, targets and scalar loss) and **165 Adam steps**
(all parameter gradients before the update), including dtype, shape and raw
tensor-byte hashes. Every captured tensor matches exactly; no tolerance or
reference replacement is used. Each second/third round starts from that same
candidate's previous final weights. The accepted-parent tests and prior
45-bundle pilot separately validate admission and optimizer inheritance across
real scientific publication boundaries.

| Instrumented phase (total seconds; calls) | Frozen 3.13 | Original source / modern stack | Final 3.14 |
| --- | --- | --- | --- |
| Growth | 12.7475; 6 | 11.8178; 6 | 10.4367; 6 |
| Model loading | 0.0947; 27 | 0.0781; 27 | 0.0643; 27 |
| Tree snapshot extraction | 1.4944; 1 | 1.1016; 1 | 1.0544; 1 |
| Dataset row extraction | 0.0211; 1 | 0.0160; 1 | 0.0154; 1 |
| Training | 5.7719; 27 | 5.0807; 27 | 4.4130; 27 |
| Checkpoint restoration | 1.1040; 3 | 0.9249; 3 | 0.7839; 3 |
| Checkpoint writing | 3.7963; 7 | 2.9111; 7 | 2.6914; 7 |
| Refresh certificate validation | 0.5775; 1 | 0.4516; 1 | 0.4267; 1 |
| Complete model refresh primitive | 4.2711; 1 | 3.5235; 1 | 3.1385; 1 |
| Whole instrumented execution | 31.607 s | 27.461 s | 24.278 s |
| Peak single-process RSS | 445.770 MiB | 475.348 MiB | 476.434 MiB |
| Separate atomic activation transaction, 787 nodes | 5.3838 s | Rejected before transaction | 5.1562 s |

These are one sequential instrumented pass per cohort, with fixed CPU thread
counts and identical work. Instrumentation and filesystem/cache effects are
included; phases are not asserted to partition total runtime. The recorded
times do not establish a Python-only or statistical speedup. The middle/final
cohorts separate source modernization from dependency compatibility for these
fixtures; interpreter and dependency effects cannot be disentangled from the
old/new comparison without additional environments and a separate study.

The complete refresh primitive above is distinct from an atomic activation
transaction. A separate fresh invocation of the original committed activation
fixture times the real 787-node transaction on the frozen 3.13 and final 3.14
stacks. Both commit the requested candidate, preserve Torch RNG, make an exact
retry inert and produce the same complete checkpoint semantic digest. Its
one-candidate/one-epoch setup bounds cost; the nine-candidate acceptance and
continuation evidence remains the real four-worker pilot above. This is one
matched activation fixture per stack, not evidence of a statistical speedup.

The original source over the modern dependencies fails during setup, before
activation, because its scientific export explicitly admits only the frozen
Atomheart/Valanga/Anemone/Coral versions. This is retained as a genuine
compatibility boundary, not patched out in the reference. The final migration
updates that declared contract to the verified public versions while keeping
the producer-fingerprint and acceptance checks strict. Search/training trace
success alone must not be read as complete compatibility of the original source.
Evidence: `activation314-original-source.log` and the two successful
`activation*/timing.json` records.

The bounded four-worker pilot additionally recorded three real atomic
activations and five accepted training rounds, as detailed above. Its resource
ledger records 255.855 seconds across two sessions. Final PR/FULL suites cover
activation acceptance and crash recovery. No larger experiment was launched
to manufacture performance evidence.

Reproduce the evidence comparison with `E/completion/compare_science.py`.
`science-comparison.json` contains the exact checkpoint hashes, counts, source
origins, interpreter strings, timings and RSS. `science_trace.py` and
`after_lanes.sh` record how each cohort was executed. Full tensor hash streams
are retained in each cohort's `trace.json` and `numerical-trace.json`.

## Integration and release readiness

The final executable delta from `7e5d8e2` is 958 production lines added and
1171 removed (net -213). The source boundary and old numerical
implementation are preserved; most initial modernization changes remove deferred
annotation scaffolding and obsolete typing code. The P0 correction adds only
32 production lines. The final report commit is documentation only.

Proposed integration sequence, subject to review and separate push/merge
instructions:

1. Preserve/publish the complete scientific branch as the review foundation;
   do not recreate the migration on obsolete public main.
2. Review `5e78a117` as an independent safety patch, with its Python 3.13 evidence
   and explicit backport instructions. Keep frozen experiments pinned.
3. Review `migrate/python314-continual` against the scientifically validated
   foundation (or its reviewed safety-patch descendant), retaining the existing
   scientific and modernization commits. Run the existing required CI lanes.
4. Resolve the external Rust publisher/release dependency, then verify fresh
   public artifacts and a clean public-only installation before any release.

No Chipiron branch was pushed, merged remotely or published in this completion.
The newest assignment explicitly excludes PyPI publication, so the prior paused
Rust upload was not retried. The existing merged/tagged 0.1.6 candidate needs a
PyPI trusted publisher; until its public release exists, a fresh public-only
resolver cannot install the final dependency set. **Release readiness is blocked
by that public dependency**, even though local migration validation passes.
PyPI currently lists Chipiron 0.2.4, and 0.3.0 has no published artifacts.
Chipiron 0.3.0 remains an unreleased local development candidate; no existing
PyPI version was overwritten. Recheck publication/version state when a release
is separately authorized.

No new worktree or repository clone was created. A small read-only Git source
archive under the private evidence root is used only for the cross-version
comparison, with no `.git` directory or independent environment. The existing
worktree, single native environment, shared interpreter and evidence are retained
for review and reproduction. The original 3.13 environment, original branches,
all existing worktrees and historical experiments are retained. The fresh
preservation audit confirms all 41 existing dirty files and 487 recorded
historical inputs unchanged. No 200k run was launched.

Storage before this completion had approximately 229 GiB free; the retained
native environment was 2.9 GiB, its shared task cache 3.1 GiB and private runtime
1.6 GiB. The bounded four-worker pilot's measured process-group peak remains
1,224.965 MiB RSS, with 64.875 MiB workspace and 7.094 MiB scratch. Current disk
and final preservation records are retained in `E/completion/delivery.json`.
