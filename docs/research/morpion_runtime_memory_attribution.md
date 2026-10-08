# Morpion runtime memory attribution before asynchronous scaling

This is a diagnostic change. It does not alter search, training, cache eviction,
checkpoint formats, scientific defaults or worker scheduling. The four-worker
asynchronous artifact pipeline remains the intended production architecture.

## Evidence and limits

The operator's gen38 attempt restored 242,680 nodes / 244,584 branches:
523.9 MiB RSS before restore, 9,515.5 MiB after rebuild, 9,516.0 MiB after temporary
payload release / GC, 1,119 MiB available against the unchanged 1,200 MiB guard,
and approximately 2,333 seconds of restore time. Growth correctly did not start.
These are **operator measurements**, not a repeat performed by this PR.
The reported RSS delta is 8,992.1 MiB, approximately 38,853 bytes per tree node;
it includes runtime objects, allocator retention and other process allocations.
It is not an owned-object measurement.

We inspected the real checkpoint **on disk only** and executed synthetic
nine-node checkpoints. The full gen38 restore must be run explicitly by the
operator after review. Its phase breakdown, decoded-state count, category
bytes/node and allocator/native residual remain unmeasured. This PR does not
claim to have explained all 9.5 GiB from a nine-node example.

## Strongest findings

1. **Selector restore defeats part of lazy decoding.** In the synthetic checkpoint,
   anchor/delta decode counts stay zero through nodes, edges and latest expansions.
   Selector restore then decodes one anchor and four deltas, retained as five
   entries in `CheckpointStateResolver._resolved_states`. Instrumentation captures
   at most three scalar call stacks per decode kind:

   ```text
   Linoo.restore_from_checkpoint_payload
     -> restore_node_states_from_payload -> _classify_node -> _is_terminal_node
     -> NodeTreeEvaluation.is_terminal -> get_effective_value_candidate
     -> compare_candidate_values -> _decision_semantic_compare
     -> tree_node.state -> CheckpointBackedStateHandle.get -> resolver.resolve
   ```

   Anemone 0.2.26's generic value comparison passes `tree_node.state` to the
   objective. Partial nodes with both direct and backed-up values reach this
   comparison. The real disk audit finds **228,320 such nodes (94.08%)**. This is
   an exposure count, not a measured full-restore decode count or proof of how
   much RSS those states occupy. The checkpoint resolver cache is an unbounded
   dictionary, separate from Chipiron's bounded rematerialization cache.

2. **Warm Growth reuse is ineffective.** `pipeline/stages.py` calls
   `runner.load_or_create()` per cycle. `runtime/runner.py` resets eviction
   bookkeeping, restores another runtime and then assigns `self._runtime`.
   The old runtime remains referenced while the new one is built, permitting
   overlap. The focused real two-cycle test observes two different restores.
   Commit `e74fd3a38b5737de8f4adfc9200e9853067c8985` (2026-07-08,
   "Keep Morpion growth runtime warm across cycles") retained the runner / changed
   launcher cycle limits, but its runner already restored unconditionally.
   This was not established as a subsequent regression: the unconditional restore
   predates that commit (`1c2db3ad`, 2026-06-29). Reuse needs explicit checkpoint,
   reevaluation patch, evaluator and run-state consistency tests before changing it.

3. **JSONL training is not fully memory bounded.** The row reader streams chunks,
   and evaluator training is sequential, but `training/service.py` chooses whole
   flat/entity/relational tensor caches. The flat and entity builders retain lists
   of per-row tensors and then stack/concatenate them; cache load uses `torch.load`
   without memory mapping. Packed legacy graph tokens are included in this path.
   Cache tensors, construction overlap, optimizer state, batches, and optional
   previous-model diagnostics can dominate Training RAM. This requires attribution
   before treating Training as a lightweight neighbor of Growth.

## Existing optimization audit

The live column below refers to the **synthetic restore**, not a gen38 restore.

| Mechanism | Evidence / current qualification |
| --- | --- |
| Split sharded runtime checkpoint | Real gen38 has `node_shells`, `state_payloads`, `node_runtime`; no flat `node_records` fallback. |
| Direct sharded restore | Directory dispatch in `runner._load_runtime_from_checkpoint`; focused test forbids both monolithic typed-payload loader paths. |
| Incremental runtime reconstruction | Existing Anemone restore callbacks reused; real `node_runtime` path uses batches of 2,048 records. |
| Bounded shard memory | Runtime records stream, but state-payload and node-shell readers each build a list for a shard (real checkpoint: up to 100,000 records). |
| Checkpoint-backed handles | All sampled synthetic handles retain `CheckpointBackedStateHandle`. This does not prove their resolver cache is empty. |
| Shared resolver | One resolver in the synthetic sample, with five decoded states retained. Real cache count awaits the operator run. |
| Dense payload store | All 242,680 disk IDs are unique, dense and zero-based. Synthetic restore selects `DenseCheckpointPayloadStore`; live gen38 selection remains to observe. |
| Lazy state decoding / summaries | All real payloads contain summaries with tags. Selector restore nonetheless materializes states through value comparison. |
| Sparse Linoo state | Synthetic table: eight entries for nine nodes, zero default `opened` entries in the sample. Sparse defaults work; nearly every node may still have nondefault state. Real count pending. |
| Lazy evaluation substates | Released restore clears empty/default payloads. Of 242,680 real records, only 5,361 ordering/PV/backup payloads qualify as empty, and 16,725 frontier payloads qualify. Thus most substates are legitimately nonempty in this checkpoint. Synthetic sample has five allocated substates of each kind among eight nodes. |
| Growth eviction | Persisted `frontier_cold` / `delta_when_safe`, scan interval 10, limit 5,000, recent window 100, delta depth 32. Growth-only machinery does not bound checkpoint decoding during restore. |
| Bounded rematerialization cache | Persisted cap 10,000; immediately after synthetic restore its count is zero. It is a different cache from `_resolved_states`. |
| RAM admission / load forecast | Existing worker mechanisms preserved; the failed real run stopped before Growth. Diagnostic adds an admission floor without reducing the persisted floor. |
| Growth recursive / tracemalloc / GC / referrer tools | Existing infrastructure retained. Diagnostic reuses its concrete-field size walker and evaluation histograms; no unrestricted tracemalloc/referrer walk is enabled. |
| Sharded training exports / JSONL | Present; readers still build full training snapshots or whole tensor caches in important paths. |
| Per-stage observability | Main includes PR #70; this adds only missing restore boundaries and read-only reporting. |

Disk sizes are serialization sizes, **not RAM estimates**:

| Shard kind | Shards | Uncompressed bytes |
| --- | ---: | ---: |
| State payloads | 3 | 85,567,991 |
| Node shells | 3 | 49,883,964 |
| Node runtime | 3 | 526,370,170 |
| Selector state | 1 | 23,125,273 |
| Latest expansions | 1 | 310,598 |
| Metadata | 1 | 14,805 |

State payloads comprise 58,949 anchors and 183,731 deltas. Manifest SHA-256:
`6ecfff57c9e715cd6ad147fb732ba2b27aba01a02fd3e81890f476d3b1847b79`.
The disk audit streams individual records; its maximum line allocation is 4 MiB
and each ID bitmap is at most 5 MiB (hard limit five million nodes).

## Restore-only profiler

Entry point:
`python -m chipiron.environments.morpion.bootstrap.profiling.runtime_attribution.cli`.
Require either `--inspect-only` (disk audit) or `--restore-only` (one restore).
`--attach-evaluator` attaches the selected persisted bundle without reevaluation.
The diagnostic runner rejects fresh creation, reevaluation, Growth and checkpoint
save; the CLI never calls Dataset, Training or Reevaluation. No export fallback.

Reports must go to a **new directory outside both experiment roots**. Artifact
size/mtime inventories are compared before and after; focused tests additionally
compare all SHA-256 hashes. Avoid running concurrent writers during diagnosis.
`--code-root` verifies a clean committed checkout against the loaded Python source.
No Anemone installation is modified: observation wrappers exist only inside the
single-threaded diagnostic context and are removed even on exceptions.

Outputs:

- `checkpoint-audit.json`: disk facts, lazy-payload candidates and prefix sample.
- `restore-phases.jsonl`: immediate terminal/file progress, elapsed time, RSS,
  available RAM, process lifetime peak RSS, decode counts and existing callback
  metadata. Decoding progress is also emitted every 2,000 decodes, including
  during a long selector restore. Includes all 13 requested boundaries, plus evaluator attachment and
  post-profile/runtime-release GC. A phase-sampled peak is not an exact temporary
  peak; process lifetime high-water marks also include imports / previous phases.
- `cheap-profile.json`: unique shallow roots / dictionaries, bounded sampled node
  structures, container counts, model tensor storage and optimization observations.
- `bounded-deep-profile.json` with `--deep`: one identity set shared across all
  components; a global object budget apportioned among categories. First-reach
  attribution is a lower bound, **not proof of exclusive ownership**.
- `summary.json`: code identity, package versions, counts and immutability result;
  the profiler itself must add **zero state decodes**.

The profiler reads concrete fields and never requests `node.state` / `state.tag`.
It samples directly from depth buckets without creating a full node list. Defaults:
128 sampled nodes, 20,000 additional recursively visited objects, depth 6. Hard
ceilings: 512 / 100,000 / 12; no unlimited setting. It does not clone the tree.
References/visited identities are bounded. Traversal may scan existing containers;
these are memory caps, not a hard wall-time guarantee. Cheap output is written
first; deeper traversal is skipped if RAM is unknown or below the persisted floor.

Categories cover algorithm/tree/evaluation nodes, parent/child and unopened branch
containers, exploration indices, handles/resolvers, payload store, anchor/delta
payloads, summaries, decoded states/cache, rematerialization cache, selector,
descendants, latest expansions, and model/evaluator. Each category reports shallow
bytes/count, additional reachable bytes/count, observed bytes per tree node and
RSS fraction. Per-node shallow sample projections are explicitly separate; do not
sum them with measured bytes. Shared roots are reserved and counted once. Native
model storage is deduplicated separately; CUDA bytes are not subtracted from RSS.
Residual RSS includes imports, unsampled objects, other native memory and allocator
retention. It cannot be labeled "allocator waste" from RSS alone.

Synthetic measurements (Python 3.13 / Torch 2.9 environment; nine nodes, sample eight):
no decoder calls before selector; five afterward; attribution adds zero.
Representative shallow sample means, **not gen38 estimates**: AlgorithmNode 208 B,
structural TreeNode 88 B, evaluation 112 B, state handle 48 B, parent/child
containers 182 B, unopened containers 940 B. One shallow resolver is 64 B;
its decoded-cache dictionary is 224 B, excluding states. These small shells cannot
explain a many-GiB RSS without their referenced graphs. The tiny restore changes
RSS by only about 0.024 MiB on an already imported ~540 MiB test process; allocator
reuse makes that unsuitable for extrapolation. Full gen38 category estimates and
phase deltas are deliberately pending the explicit operator measurement.

## Four-worker memory model and next sequence

| Worker | Current memory behavior / useful next boundary |
| --- | --- |
| Growth | Owns full live runtime; restore creates payload/shell shard lists, per-node structures and decoded states. Repeated cycles rebuild, possibly overlapping old/new runtime. Save/export has additional payloads and node/update lists. Preserve a validated warm runtime, then measure save/export peaks. |
| Dataset | `cycle_dataset.load_training_snapshot_for_generation` calls `load_morpion_sharded_training_tree_snapshot`, which reconstructs the full `TrainingTreeSnapshot` before streaming row output (`pipeline/stages.py`). Separate process means another O(N) node/metadata/state-reference graph; it is not a second full search runtime, but is not bounded by output chunk size. |
| Training | Sequential loops in `cycle_training.py`; `del previous_model`, `del trained_model`, then `log_after_cycle_gc` explicitly calls GC. No explicit CUDA cache release there; reserved CUDA memory can persist without live tensors. Whole CPU tensor caches and optional diagnostic rows/models are the larger qualification to streaming claims. |
| Reevaluation | Loads the same full training snapshot, a sorted ID tuple, and `nodes_by_id` maps. Its window bounds inference/patch work, not input snapshot memory. At gen38, ID tuple slots alone are ~1.85 MiB on 64-bit Python, excluding IDs, dict tables, and the much larger snapshot. Full snapshot RSS remains unmeasured. |

The focused test proves Dataset and Reevaluation construct distinct full node
collections from the same tiny immutable export. Do not multiply the 9.5 GiB
Growth figure by four: their object graphs differ. Nevertheless, Growth alone
leaves less than its reserve, while two snapshots, training tensors and temporary
save/load allocations add overlapping peaks. Independent preflight free-RAM
checks also cannot reserve memory atomically across processes.

Prioritize separate, measured changes:

1. Run the single approved gen38 diagnostic. Quantify selector decode/cache cost,
   lazy substate occupancy and phase peaks before choosing numeric savings targets.
2. Preserve objective/search semantics while avoiding needless state materialization
   in terminality/value comparisons. Then evaluate a bounded checkpoint resolver
   cache / safe handle eviction. The existing rematerialization cache is not enough.
3. Implement warm Growth reuse with explicit artifact/model identity and patch
   consistency; prove continuation parity, including evaluator updates. Avoid the
   old/new runtime overlap. Benchmark restore and checkpoint/export separately.
4. Stream training-export shards to Dataset rows and Reevaluation windows/patches.
   Avoid full snapshots and repeated global ID maps; preserve generation consistency.
5. Bound or memory-map training caches while keeping sequential evaluators and
   explicit cleanup. Measure CPU RSS and CUDA allocated/reserved separately.
6. Keep four independent asynchronous workers. Use memory-aware admission/reservations
   and backpressure for heavy load/save/cache phases, permitting lightweight and GPU
   overlap within measured headroom. Serialize only unsafe phases when necessary.

Targets are directional until attribution: materially below current Growth RSS,
reserve plus measured lightweight-worker peak after restore, substantially less
than ~39 minutes restore, and **one restore per warm Growth lifetime**, subject to
explicit recovery/model update rules. Permanent serial orchestration is not the
production recommendation.

## Operator command template

After the diagnostic PR and focused tests are green, use a clean validated checkout
and the already validated Python environment. Run once, with no other experiment
writer active. Budget against the previous ~39-minute restore; profiling is capped,
but no new runtime prediction is claimed.

```bash
PYTHONPATH=/path/to/validated-checkout/src /path/to/validated-python \
  -m chipiron.environments.morpion.bootstrap.profiling.runtime_attribution.cli \
  --code-root /path/to/validated-checkout \
  --work-dir /path/to/derived-workspace \
  --output-dir /path/outside/experiments/new-gen38-profile \
  --restore-only --attach-evaluator --deep \
  --sample-nodes 128 --recursive-max-objects 20000 --recursive-max-depth 6 \
  --minimum-free-before-restore-mib 10240
```

10,240 MiB is conservative admission headroom derived from the previously observed
~8,992 MiB RSS increase plus the existing 1,200 MiB reserve; it is not a promised
maximum or a new scientific target. Unknown / insufficient RAM refuses restore.
The persisted reserve is always honored. Close unrelated memory-heavy applications
if admission refuses; do not lower the guard to force it through. This command
exits after profiling and cannot launch the continuation.
