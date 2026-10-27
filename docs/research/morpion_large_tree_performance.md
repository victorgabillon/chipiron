# Current-format large-tree performance

This investigation starts from Chipiron `f290fd7ea8746f3fedef70853d716b1ce06a0e36`
(production identical to FULL-tested `7b7a5b91d523e0bf54064e2d4a44b372b0d57ae3`)
and installed Anemone `37e48c40ceff430e373fd07bd00fcf41359bafda`.
No subsequently completed source-lifetime implementation was available when the
isolated branches were created. The bounded-history implementation is retained.

Worktrees are `/home/pompote/oldata/victor/morpion-scratch/s28/w` and `s28/n`;
branches are `perf/morpion-large-tree-current` and
`perf/morpion-recover-state-free`. Original dirty development worktrees are
excluded from implementation and execution. Historical experiments, including
gen38, are evidence sources only. No historical checkpoint is executed or migrated.

## Historical mechanism inventory

Symbols below are relative to their package; abbreviated commits are resolvable
in the local Chipiron/Anemone repositories. Historical numbers are evidence of
their original workloads, never denominators for current-format speedups.

| Mechanism / original problem | Original source | Current equivalent | Classification at foundation | Evidence and action |
| --- | --- | --- | --- | --- |
| Incremental anchor/delta checkpoint and lazy handles; full history duplication and eager replay | Anemone `57ec5d5`, `22c3618`; split restore `675c013`, module extraction `1e8879a`, `0680c9f` | `checkpoints/sharded_restore.load_search_from_sharded_checkpoint`, `state_handles.CheckpointBackedStateHandle`, dense payload store | Preserved but bypassed downstream | Shell creation explicitly sets `build_state_representation=False`. Selector comparisons subsequently resolve states. Keep codec/store; recover comparison fast path. |
| Reuse serialized state payloads instead of reconstructing unchanged states on save | Anemone `22c3618`, `6141057` | `build_state_payloads._try_reuse_checkpoint_state_payload`, `_delta_reuse_matches_parent` | Preserved and active | Reuse checks codec/graph-parent compatibility; alternate graph and codec parents remain distinct. Keep exact payload checks. |
| State-independent max comparison; selector terminality/effective-value checks materialized most states | Anemone PR77 `ec9fec7e47ca661eb1dd9330d115d010d9e5e64d` | Missing dispatch in `NodeTreeEvaluationState._decision_semantic_compare`; missing objective hook | Removed accidentally in selected dependency ancestry | `git merge-base --is-ancestor ec9fec7e 37e48c40` returns 1. Historical 242,680-node restore comparison: 2,333.004→202.394 seconds; 230,446→0 state decodes; selector 2,135.641→6.298 seconds. Foundation 1k profile reproduces selector-triggered cost. Recover exact old change and tests. |
| Lazy evaluation substructures; eager default ordering/PV/frontier allocations | Anemone `d580d73e72c09c13b9343b3338df00fcfec2617c` | `NodeTreeEvaluationState` lazy runtime properties and checkpoint restoration | Preserved; benefit limited by persisted nonempty state | Earlier 242k investigation reports most persisted structures nonempty. Measure owned memory; do not claim every node has zero auxiliary state. |
| Sparse Linoo node cache; duplicate default records | Anemone `b2606a9753119b42501d87b971ba7f144812c03b` | `node_selector/linoo/runtime_state.py`, `checkpoint_state.py`, `Linoo._node_state_by_id` | Preserved and active | Separate durable sparse state, candidate heaps and O(N) node map are legitimate owners. Old memory review confirms sparse behavior. No speculative replacement. |
| Fast canonical scientific export; repeated unchanged-state replay | Earlier sharded export; Chipiron duplicate-normalization fix `d7590c0f204ff5ff39339d9e62a2a20557bc7cd1`; current reuse `bdad537a`, `ae35addb`, execution fence `c1c737ca` | `closed_scientific_snapshot.save_closed_scientific_snapshot`, `scientific_validation_evidence` scopes | Legacy path removed intentionally; validated equivalent preserved and active | Old 5k three-export total: flat 81.111 seconds vs sharded 1.017. Old paths differed in canonical ordering/target/provenance and cannot replace closed science. Exact committed records are copied; mutable updates/topology freshly validated. |
| Ordinary evaluator restore avoids full neural refresh | Current bound-head restoration in Chipiron runner | `_load_runtime_from_sharded_checkpoint(model_bundle_path=...)`, `load_or_create` | Preserved and active | Committed evaluator loads before runtime construction; `evaluator_restored_from_committed_head`. Explicit activation still requires full scientific refresh. |
| Evict inactive states and rematerialize from bounded recipes | Chipiron C4/C5/C6 policies; Anemone bounded resolver `5f43821` | `growth_state_eviction_*`; `CheckpointStateResolver` bounded LRU | Preserved; experimental growth policies not enabled | Earlier lifecycle bound-8 churn had exact parity but no demonstrated throughput win. Existing default resolver bounds decoded retention independently of Growth eviction. No new eviction policy. |
| Reuse live runtime across durable publication | Chipiron PR74 `aeece2c1694a90bb79880db1733966d22c36c6a1`, cold gate `5e5c8b45`; native adoption Anemone `e29b584`, combination `866560f` | Production reconstructs each invocation; native adoption absent from pinned ancestry | Experimental and never adopted | Original 10k mismatch: representative-parent order. Native adoption fixed old-format tested parity, but flat-export whole-call warm 189.127 vs cold 157.727 seconds. Investigate only after recovering major simpler costs. |
| Bounded runtime serialization instead of complete typed checkpoint | Anemone `37e48c40`, Chipiron `d8e26e6e` | Existing streaming split writer, 1,024 construction/physical-shard size | Preserved and active | Prior 5k auxiliary peak 12.67→6.12 MB, same continuation. More files (7→19) is an observed cost, not proof of a large-tree inode bottleneck. |

Original comparison is retained in
`/home/pompote/oldata/victor/bootstrap_runtime_memory_pr77_ab_2026-10-20/gen38-restore-profile-pr77/comparison.md`.
It names exact Chipiron `29d0dbf772dd16186268f0c7656d9fd02fd33480` and PR77;
the old Anemone baseline Git identity was not recorded, which limits attribution.
This task reads the report without rerunning that experiment.

Other historical sources: Anemone `fbd984a:docs/research/checkpoint_lifecycle/investigation.md`,
Chipiron `morpion_training_export_performance_2026-10-20.md`, the seven foundation
reports named in the assignment, and the October 8 runtime-memory review.
The requested approximate 226k/250k/500k/721k numbers remain historical leads
until their original timing evidence and identities are located. Directory
labels alone are not measurements: the retained `profile_*250k` metadata examined
here actually reports 205,980 nodes.

## Measurement discipline

Baseline environment `s28/v` is frozen; candidate environment `s28/c` owns separate
copies of scientific packages and shares read-only heavy tooling. Baseline
Git/wheel/installed verification compares all 478 Chipiron and 289 Anemone Python
files, retaining per-file SHA-256 in `s28/e/identities-baseline.json`.
Current-format legal fixtures use the existing scientific initializer, explicit
constant-42 M1, default Linoo/alternation, bounded two-step rollouts, closed
compressed science and identical numerical policy. Checkpoints are format 3.

Processes are fresh, OS page cache uncontrolled/warm; no physical cold-disk claim.
Profiled inclusive times are not added to nested phases. Unprofiled matched
measurements determine throughput. CI does not overlap controlled benchmarks.
The existing monitored recipe enforces finite time/step, RAM, disk and scratch
budgets; raw failed/abandoned attempts are retained and not counted as commits.

The initial disposable profile used an incorrectly named control file and began
the frozen configuration's 4,000-step limit. It was interrupted after about
20 seconds, before publication, and retained as `s28/p1` / `e/p1.*`. The corrected
profile uses `MorpionBootstrapPaths.control_path`, 32 steps, and a separate copy.
It commits 1,167 nodes, from 1,009 to 2,176. Profiled useful search is 37.841 s,
including 27.912 evaluation, with feature candidate scanning 20.784 s inclusive;
scientific export 14.158 s inclusive; runtime restore 2.909 s, of which selector
restore is 2.539 s. These instrumented times are diagnostic, not throughput claims.

Before any workload, the protection inventory records 1,408,446 files,
161,327,273,399 logical bytes and 50 worktrees. It covers historical Morpion runs,
the default Chipiron data root and prior scratch roots, excluding only this new
task root. Final comparison is required before readiness is stated.

Implementation, matched profiles, lifecycle decision, learning validation,
scale results and final frozen CI will be appended as their evidence completes.
