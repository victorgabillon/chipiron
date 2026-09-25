# CI performance with equivalent validation

CI retains the required check named `test`. It is an always-running aggregator of
`quality` (tooling/Ruff), `pytest` (installed distribution), and `static` (all three
analyzers plus debt ratchet). Any failure, skip or cancellation in a dependency
fails the aggregate. The original component timeouts remain 15 minutes and job
limits remain 45 minutes. Tests never run conditionally based on changed files.

## Test-level work came first

Twenty profiling unit/event cases previously rescanned the entire unrelated
Torch/Qt/pytest heap on each call. A function-scoped fixture now supplies fresh
ordinary containers, a frozenset and a real Value object as the ambient GC input.
Only the two profiling modules' GC references are replaced, not global GC. The
same explicit runner graphs, traversal, size/type summaries, event hooks, caps,
logging and assertions execute. A real unpatched process-heap smoke and specialized
ownership-count tests remain. Before and after these cases prove the same graph,
anatomy and event contracts. Lost assertions and intended behavior coverage: zero.

The match matrix caches successful immutable Hugging Face download paths within
its module. The first resolution still downloads real files; every game still
loads fresh model instances. All configurations, seeds, board implementations,
search budgets, move comparisons and assertions remain. Dedicated model-resolution
and failure tests are unchanged. Lost assertions and intended behavior coverage:
zero; no mutable model/player state is shared.

Two direct regressions now test pawn/quiet-move progress with actual Python/Rust
boards and unsupported-state fallbacks. Previously some helper paths were covered
only incidentally by particular game trajectories. No game case is removed.
Four additional cases explicitly verify checkpoint payload cache invalidation
when a real file's path, size or mtime changes, or it disappears. This coverage
previously depended on incidental cache state left by another test. The test
uses `os.utime` for deterministic timestamp changes instead of sleeping.

## Installation and parallel execution

Hosted CPU jobs constrain Torch to 2.14.0+cpu before dependency resolution, retaining
the baseline's Torch API version. Runtime/project dependency semantics are unchanged.
Pip caching is enabled and uses a separate CPU namespace so an obsolete CUDA cache
cannot be restored by fallback. The installed-wheel guard, fresh wheel/sdist build,
wheel installation and all canonical marker exclusions remain.

After the intrinsic changes, pytest uses two workers with `--dist=loadfile`.
Application outputs and SQLite tracking databases are isolated by run UUID and
worker; ordinary serial runs retain their existing environment behavior. The
observer plugin is loaded in every process. It records actual selected/deselected
collections, rejects worker disagreements, and preserves pytest's failure status.
All 1,293 original selected node IDs remain, plus 18 reviewed regressions: two real
board helpers, 12 analyzer-failure cases, and four checkpoint cache identity cases. Both expected skips and all four
marker deselections remain. Before/after evidence checks exact covered production
lines and branches separately for installed and explicitly imported source modules.

Mypy, Pyright and Pylint run concurrently with the exact original command lines,
600-second per-tool timeouts, raw reports, normalization and immutable baseline
comparison. Results are processed in deterministic tool order. Twelve regressions
prove that a crash, empty failing report, malformed JSON or timeout in any one
analyzer fails even if both sibling analyzers succeed. `--workers 1` is available for sequential diagnostics.

## Evidence and release safety

`ci-validation-evidence` contains test identities/durations, JSON line/branch
coverage, JUnit, installed dependencies and the process list after tests.
`ci-static-evidence` retains all raw analyzer reports and ratchet summaries.
Quality logs retain the toolchain/Ruff results. Compare complete artifacts rather
than rounded coverage percentages or pytest's abbreviated distributed summary.

Runtime source, search/evaluator/checkpoint semantics, seeds and release workflows
are unchanged. Releases still validate the tagged checkout fully, including
existing build/Twine/metadata and Docker checks. No latest-main result substitutes
for exact tagged-checkout validation. No long Morpion experiment belongs to this CI
work, and no prepared bootstrap workspace is accessed.
