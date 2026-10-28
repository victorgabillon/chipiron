# Chipiron Python 3.14 modernization — 28 October 2026

The migration uses the complete validated October 27 continual-learning tree.
Its final tested source commit is
`ab60c85600cf7a40fc07c7f6f0b7e019e84aab5f`, on
`feat/python314-modernization`, descended from
`7e5d8e22728ea0421a580fdfd9af00dbba640034` (October 27 report and executable
`fb444210`). The earlier performance foundation `f21fc096` is already in this
ancestry. Morpion production source is byte-identical to migration commit
`ed202ad246d87bbffb577dc92091d955ff7c3743`. The descendants align Docker,
remove the obsolete Valanga test backfill and make chess input converters
serializable under the native forkserver context. Public `main` was not used as
the migration source.

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
the validated native library SHA-256 is
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

## Package identity

Chipiron wheel and sdist passed strict Twine validation. All **483** production
Python/typing files (482 Python files plus `py.typed`) match Git `ab60c856`, the
wheel, sdist and installed package byte for byte. There are no extra installed
Python files. The frozen Morpion feature baseline is also present and unchanged
in the sdist. Installed smoke checks cover all five foundation origins, packaged
YAML/resources, typing marker and evaluator/checkpoint contracts. Evaluator-v1
retains 106,049 parameters and relation-bias scale 0.25.

| Artifact | SHA-256 |
| --- | --- |
| Final Chipiron wheel | `9f4bd15829b22d2ff6491f24a353f75bf478df5cf15b11075f58c7a1d2440326` |
| Final Chipiron sdist | `5dd4945e5b319a95dfc0170db20aa8c14fb1caa208532d7d08885a03ccfb32f0` |

The final manifest is `E/installed-source-verification-final.json`. The earlier
`ed202ad` artifacts used by the scientific comparisons and pilot remain in
`E/dist` with their original `E/installed-source-verification.json`. All Morpion
production bytes remain unchanged. Public foundational wheel/member identities
are in `E/public-foundations-verification.json`;
`E/torch-installed-verification.json` records the official Torch comparison.

## Scientific equivalence and retained continual learning

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
`E/dist`. Its Morpion source is byte-identical to final `ab60c856`; the later
production correction concerns chess converter serialization. The final lanes
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

## Final validation

Final FAST, PR and one FULL invocation ran serially against the installed
`ab60c856` wheel and public foundations. Their selections overlap and counts must
not be added. The normal exclusions remain integration/external-data; the
normal chess integration module was exercised separately above.

These final lanes use the explicitly identified local Rust 0.1.6 candidate,
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
926 insertions and 1,171 deletions (net -245 lines). The final report commit is
documentation only; the exact tested source identity remains `ab60c856`.
Chipiron remains a local, unpublished branch. Publication would require separate
user authorization. The remaining installation blocker is PyPI's missing
trusted publisher for the authorized Rust binding 0.1.6 release; the source,
merge, tag, CI artifacts and local validation are complete.
