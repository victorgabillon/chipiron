# Morpion training export: reproducibility, cost and allocation investigation

Date: 2026-10-20. Production warm reuse remains disabled. All new runs are bounded synthetic CPU fixtures in new `/tmp/morpion-export-*` directories. No real gen38 restore/continuation, distributed worker, historical-workspace mutation, dependency change or merge was performed.

## Findings

The retained 10k observation **used flat export**, despite the new request describing it as sharded. PR74's benchmark explicitly constructs `MorpionBootstrapArgs(training_export_mode="flat")`; retained observations have `components=None`, `profile=None` and flat `tree_exports/generation_*.json` artifacts. It did not exercise the sharded writer or collect its phase counters. The old evidence and PR78 are unchanged.

The approximately 20% warm export penalty is **not reproduced** in the new 1k/5k trials. Flat export is expensive in both modes because it resolves nearly all checkpoint-backed states and asks Atomheart to recover a deterministic replay order again for every node. The exact cause of the old single-trial cold/warm difference remains unestablished. These results do not prove that all larger or asynchronous workloads have identical performance.

A separate small allocation issue is established in the sharded writer: new node records already own a normalized JSON state-ref copy, but `_node_record_to_dict` normalized/copied that private payload again. Removing that second copy preserves every serialized byte while lowering the first 5k export's traced Python allocation peak by approximately 10.5%. There is no consistent throughput improvement across both cold and warm samples, and no claim that this change fixes the old 20% observation.

The cross-format audit also finds pre-existing state-ref ordering, target/provenance and incoming-parent differences. The existing fast sharded path cannot be recommended as an exact replacement for flat export without a separate correctness review.

## Controlled comparisons

Each measured trial is a fresh process. All cases use the same prepared synthetic checkpoint, three saves/exports, two bounded Growth steps per save, no model training and no eviction/rollout. Runtime cold reference restores three times; the experimental warm subclass restores once, adopts only after publication and retains one runtime. The ordinary runner remains cold.

At 1k, each export format runs in order `C,C,W,W,C,W,W,C` (four trials per runtime mode). At 5k the order is `W,C,C,W` (two per mode), reversing the first mode. After-change sharded trials use `W,C,C,W` at both sizes. Baseline and after pools are sequential; allocation diagnostics additionally reverse implementation order. Profiling/tracemalloc/ablation runs are excluded from throughput pools. No new 10k run was justified by the smaller results.

Numbers below are **median sums of the three export stages**, excluding load, Growth, checkpoint save and adoption. They are not the old 157.73/189.13-second whole-pipeline measurements.

| Requested fixture | Export | Trials per mode | Cold wall s | Warm wall s | Cold CPU s | Warm CPU s | Cold/warm median process peak RSS MiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1k | flat | 4 | 10.862 | 10.867 | 10.860 | 10.865 | 313.9 / 314.0 |
| 1k | sharded | 4 | 0.520 | 0.526 | 0.520 | 0.525 | 307.4 / 306.3 |
| 5k | flat | 2 | 81.111 | 79.508 | 81.085 | 79.483 | 424.7 / 422.0 |
| 5k | sharded | 2 | 1.017 | 0.981 | 1.016 | 0.977 | 338.5 / 338.3 |

The 1k flat total ranges overlap: cold 10.52–11.10 seconds, warm 10.50–12.06 seconds. Warm/cold median differences are approximately +0.04% at 1k and −1.98% at 5k. Sample sizes are small; this is evidence against attributing the retained 20% observation to warm adoption, rather than a universal equivalence/performance proof. Host scheduling, allocator and CPU-frequency effects on the old run cannot be reconstructed from its wall-only measurements.

Per-cycle median wall times at 5k (cycles export generations 2, 3 and 4):

| Export | Mode | Cycle 1 s | Cycle 2 s | Cycle 3 s |
| --- | --- | --- | --- | --- |
| flat | cold | 27.316 | 27.018 | 26.776 |
| flat | warm | 25.397 | 26.764 | 27.347 |
| sharded | cold | 0.515 | 0.263 | 0.238 |
| sharded | warm | 0.497 | 0.254 | 0.229 |

Process peaks are lifetime `ru_maxrss` observations, including preparation before export in that process; they are not stage-local peaks. Full per-trial profiles, RSS boundaries, CPU time, raw artifact/result fingerprints and source identities are in `benchmarks/morpion_training_export_20261020/results.json`.

## Save/export/adoption ordering and ownership

The existing pipeline and instrumented event traces establish this order:

1. Finish bounded Growth and build the complete typed checkpoint payload.
2. Warm instrumentation prepares native adoption while that payload exists. Cold prepares no adoption plan. Preparation holds a compact native resolver/store and weak runtime identity, not the full typed node/runtime graph; it does not install handles or decode states.
3. The ordinary checkpoint writer completes. Returning from `save_checkpoint` drops its full typed payload/local write objects; the warm plan still owns the compact saved refs.
4. Training export runs against the current live handles. Thus **the first export precedes the first adoption**, although the warm adoption plan has already been prepared.
5. Generation publication and run-state save complete; the caller acknowledges publication.
6. Warm adopts the checkpoint's native lazy handles/order, drops the plan and rotates its existing caller-owned eviction resolver. The replacement native resolver is initially empty. Old resolver caches are released when their last owner disappears; no owned cache is forcibly cleared. Cold replaces its runtime at the next load instead.

Flat export fills the current native resolver's `_resolved_states` cache through actual state access. Its temporary snapshots/JSON objects are released after saving, while resolved states remain owned by handles until the following boundary. Sharded export reads checkpoint refs without resolving them and only calls the materialized dumper for new materialized nodes.

A first-cycle warm diagnostic with preparation enabled takes 4.029 seconds; skipping preparation takes 4.061 seconds. Both perform exactly 264 anchor and 698 delta decodes, with GC approximately 0.017 seconds. This single ablation is not a precision timing comparison, but supplies no evidence of the large preparation-related export penalty. Its scientific export/checkpoint output is unchanged.

## Phase attribution and existing optimizations

Existing `MorpionTrainingExportProfile` and `MorpionShardedTrainingExportStats` are used first. Additional diagnostic tools are standard `cProfile`, `time.process_time`, GC callbacks and a separate intrusive `tracemalloc` pass. No product profiling framework, payload representation or resolver was added. Nested/cumulative durations below overlap and must not be summed as independent phases.

For the first 5k export, baseline medians are:

| Existing measurement | Flat cold s | Flat warm s | Sharded cold s | Sharded warm s |
| --- | --- | --- | --- | --- |
| State access | 11.198 | 10.362 | 0.000019 | 0.000022 |
| State-ref conversion | 15.742 | 14.686 | 0.214 | 0.209 |
| Node snapshot construction, inclusive | 27.263 | 25.348 | 0.190 | 0.183 |
| Parent/child extraction | 0.054 | 0.049 | 0.038 | 0.036 |
| Value extraction | 0.075 | 0.071 | 0.037 | 0.036 |
| Node metadata extraction | 0.126 | 0.116 | 0.091 | 0.088 |
| Sharded row build | — | — | 0.456 | 0.441 |
| Sharded JSON encoding | — | — | 0.0323 | 0.0315 |
| Sharded file writes | — | — | 0.0025 | 0.0028 |

All eleven requested areas are accounted for:

- **Live traversal:** `_all_nodes_in_tree_order` is small in cProfile (about 0.00006–0.00010 seconds at 1k). The sharded aggregate's `node_traversal_s=0` is an unrecorded counter, not a claim of zero traversal work.
- **TrainingNodeSnapshot construction:** every live node still receives a current snapshot/update. At 5k repeated sharded exports, inclusive snapshot work is approximately 0.179 seconds cold / 0.174 warm; metadata is the largest component. Flat inclusive construction is dominated by its state-ref work.
- **State references:** flat `_dump_state_ref` classifies reusable handles for profiling, then explicitly accesses `node.state` and calls `dump_state_ref` anyway. Its `reusable_checkpoint_payloads` count does not mean flat actually reuses those refs. The sharded path uses the existing `state_ref_payload_without_resolving` helper.
- **Delta flattening:** the existing compact chain converter is active. The first 1k sharded cProfile has 732 flatten calls, totaling 0.0157 seconds under profiling. It reads payload ancestry and appends compact moves without checkpoint decodes.
- **New-node records:** first sharded export records every node because this synthetic preparation has no prior training index. Subsequent 5k generations create only 39 and 26 immutable records. Their state-ref conversion costs approximately 0.0393 and 0.0165 seconds cold; pre-existing refs are reused.
- **Mutable updates:** all 5,039 / 5,078 / 5,104 current nodes are written, in current order, at the three generations. Current children, direct/backed-up values, exactness, terminality, visit data and metadata are never skipped for throughput. `_node_update_to_dict` totals about 0.0007 seconds at 1k under cProfile; snapshot/metadata construction is substantially larger.
- **Global index:** the existing source-generation node index is loaded and used. At 1k its repeated loading totals approximately 0.0011 seconds under cProfile. Rebuilding/reloading this mapping is not the observed bottleneck.
- **JSON object construction:** the writer holds typed update/record lists and builds dictionaries before encoding. The allocation probe sees 5,039 typed records, 5,039 typed updates and 5,039 immutable-row dictionaries simultaneously. The second normalized state-ref copy is demonstrably redundant.
- **JSON encoding:** existing counters show approximately 0.032 seconds initially and 0.014 on repeated 5k generations. Flat serialization is small relative to replay/snapshot work. Streaming JSON could reduce buffers but would not remove the dominant flat state work.
- **Writing:** sharded file-writing time is approximately 0.001–0.003 seconds. Slow storage is not supported as the cause in these fixtures.
- **GC/allocator:** 5k flat GC totals approximately 0.87 seconds cold and 0.93 warm over three exports, about 1% of wall time. Sharded totals are about 0.014 / 0.013 seconds. Allocation costs inside normal functions remain part of their measured CPU time; the allocation probe separately identifies the redundant owned-ref graph.

PR77 state-free evaluation is active in the pinned combined Anemone source; value/metadata reading does not account for the flat state-ref cost. PR78 compact adoption is active at the committed boundary. The expensive compatibility serialization comes from Atomheart 0.2.11: `dump_state_ref` aliases `dump_anchor_ref`, and `_ordered_played_moves` reconstructs a deterministic legal replay order. In the first 1k flat cProfile it is called 1,038 times and takes 3.44 of 7.48 profiled seconds; state reconstruction accounts for much of the rest. The corresponding sharded profile makes only 32 such calls for new materialized states. These cProfile times include profiler overhead and are not throughput samples.

Actual **checkpoint codec decode counts**, identical cold and warm:

| Fixture | Export | Cycle 1 anchor/delta | Cycle 2 anchor/delta | Cycle 3 anchor/delta |
| --- | --- | --- | --- | --- |
| 1k | flat | 264 / 698 | 264 / 728 | 301 / 730 |
| 1k | sharded | 0 / 0 | 0 / 0 | 0 / 0 |
| 5k | flat | 1,344 / 3,648 | 1,354 / 3,661 | 1,364 / 3,680 |
| 5k | sharded | 0 / 0 | 0 / 0 | 0 / 0 |

Zero checkpoint decode calls do not mean zero temporary Morpion states: the new-node canonical dumper itself replays moves internally. The important distinction is that sharded export does not rematerialize all old runtime states.

## Focused correction and before/after evidence

Native implementation commit: `d7590c0f204ff5ff39339d9e62a2a20557bc7cd1`, directly against protected Chipiron main `50fb67a42577e195bc565f4d03e38d919640d08a`. The source change replaces the second `checkpoint_payload_to_jsonable(record.state_ref_payload)` call with the record's already-normalized, detached private payload. The earlier normalization at node-record creation remains. No fields, format, cache policy, generation handling or producer/consumer contract changes.

The new semantic regression uses a dumper that returns and mutates the same nested mapping on successive calls. Each node must retain its own state-ref snapshot. This protects the important initial ownership copy; it does not assert implementation call counts.

The isolated actual-fixed-source 5k allocation diagnostic produces:

| Metric | Before | After |
| --- | --- | --- |
| First export traced peak, bytes | 15,161,168 | 13,563,821 |
| Reverse-order baseline traced peak, bytes | 15,161,593 | 13,563,821 |
| Repeated generation-3 traced peak, bytes | 7,504,063 | 7,501,860 |
| Repeated generation-4 traced peak, bytes | 7,523,347 | 7,520,225 |

The first reduction is approximately **1.52 MiB / 10.5%** in traced Python allocations. At the immutable-shard write boundary, the reduction is already visible before encoding, tying it to duplicate nested ref objects. Both forward and reverse baseline comparisons show the same effect. Later generations change little because only 39/26 refs are new. Tracemalloc covers allocations after export entry, not the complete process or native allocations. Observed process peaks in these intrusive runs are 342.4 MiB before / 340.5 after; normal before/after sharded RSS medians differ by less than 1 MiB, so there is no strong whole-process RSS improvement claim.

Normal three-export wall medians before/after:

| Fixture/mode | Before s | After s |
| --- | --- | --- |
| 1k cold | 0.520 | 0.501 |
| 1k warm | 0.526 | 0.521 |
| 5k cold | 1.017 | 0.956 |
| 5k warm | 0.981 | 0.990 |

The mixed timing result supports an allocation optimization, not a consistent warm-throughput speedup. Bytes written are exactly unchanged: the three 5k sharded writes total 5,626,294, 1,750,964 and 1,751,749 bytes per generation respectively.

Across **32 normal trials / 96 export stages**, assertions compare every normalized JSON field and every retained export file's **raw bytes** within each format, across cold/warm and before/after. All match. Final checkpoint components, selections and every decompressed record in retained generations 3/4 also match across formats and implementations. No scientific fields or parent references are dropped, reordered or normalized to achieve equality.

## Cross-format correctness blockers discovered

The optimization preserves existing output exactly. It does **not** establish flat/sharded equivalence. The separate unchanged-baseline audit finds, at 1k generation 4:

- 511 nodes have different replay move order in `state_ref_payload`. Node 6 flat ref is `[0,[9222105397312061435,9222668330085744635]]`; sharded reverses those moves. Thus directly substituting compact chain flattening into flat export would violate the requested exact state-ref preservation.
- 1,031 nodes have different `target_value_scalar`; node 6 is `null` flat versus `2.2500000450000006` sharded. The sharded reader supplies a direct-value fallback where the flat builder's tree-target field is absent.
- All 1,108 nodes differ in `effective_value_source`; for root 0, flat emits `tree_child`, while the sharded reader reconstructs `tree_value`. Metadata also has explicit format/source-generation differences.
- Node 769 gains parent 125 after its first training export. Flat reflects parents `["106","125"]`; sharded still returns `["106"]`. Parent IDs are stored in the supposedly immutable record while mutable children are updated. A separate four-node generic fixture reproduces this: an added `b -> join` edge appears in `b.child_ids`, but `join.parent_ids` stays `["a"]` instead of `["a","b"]`.

These are pre-existing reader/schema/immutability assumptions, not changes introduced by the allocation patch. They are recorded rather than normalized away. The full audit is retained at `/tmp/morpion-export-format-audit.json`, with a fingerprint and compact examples in `results.json`. Repairing them requires separate scientific/format review. No format change is made here.

## Long-term writer costs and remaining bottlenecks

Let N be current nodes, E current edges, K newly exported nodes and H replay-history/ancestry work.

| Area | Flat | Existing sharded writer |
| --- | --- | --- |
| State-ref work | Every node, including reconstruction/canonical replay | Initial N refs, mostly compact; repeated K materialized refs |
| Mutable node work | All N current snapshots and edges | All N current snapshots/updates and edges |
| Immutable persistence | Entire export rewritten | New K records; old records reused through creation index |
| Index work | None | Load/write O(N) creation index |
| Bytes written | All node fields, refs and topology | O(N mutable fields + E + K refs + N index) |
| Temporary memory | Decoded state cache + snapshots + dictionaries + JSON/UTF-8 buffers | Index + typed records/updates + dictionaries + JSON buffers |

The small patch removes the second O(K) normalized-ref object graph; it does not remove O(N) updates or output buffers. The remaining repeated sharded cost is predominantly current snapshot/metadata construction. The existing incremental writer already avoids re-encoding old immutable states, but its incoming-parent assumption is invalid for growing DAGs. Future streaming/index caching must retain every changed value, incoming/outgoing link, ordering field and target/provenance field. Changed scientific data cannot be omitted to reduce write volume. Initial construction, repeated generation time, byte volume and peak RAM should continue to be measured separately.

## Validation, exact sources and reproduction

- Eight sharded-writer tests pass, including the new nested-mutable ownership regression.
- **172 relevant writer/runtime/pipeline/performance tests pass** with a test-only override directing default leaderboard writes to each test's temporary directory. All record extraction/persistence logic and assertions still run.
- Full repository Ruff checks/format and toolchain-coherence checks pass. The existing static ratchet passes with **zero new diagnostics** (209 baseline / 206 current); the reviewed debt file is unchanged.
- An optional broad local run was stopped after approximately nine minutes at 98%: 1,364 passed, 22 failed and 2 skipped. It includes unmarked legacy integration/data-download tests and tests that try to write the default home-directory leaderboard. That write is blocked by the sandbox and reproduces on untouched main (`test_first_run_writes_bootstrap_config`, after 165 passes). This run is **not** reported as green. The retained `validation_safety.py` supplies only temporary default leaderboard paths for the successful relevant suite. HOME is unchanged. Remote CI status is reported on the draft PR.

Exact identities:

- Protected Chipiron main: `50fb67a42577e195bc565f4d03e38d919640d08a`.
- Unchanged PR74 benchmark base: `aeece2c1694a90bb79880db1733966d22c36c6a1`, new baseline source `/tmp/chipiron-export-profile-before/src/chipiron`.
- Independent writer fix: `d7590c0f204ff5ff39339d9e62a2a20557bc7cd1`, new worktree `/home/pompote/oldata/victor/chipiron-training-export-profile`, branch `perf/morpion-training-export`.
- Clearly labeled temporary PR74 + writer-fix combination: `65b00a77968e3c4120b10865be56b22846838e99`, import `/tmp/chipiron-export-profile-integration/src/chipiron`.
- Anemone integration source: `866560f7606eea095981ec25090c5f2c5908a643`, the previous explicit PR77 + PR78-implementation combination; import `/tmp/anemone-checkpoint-lifecycle-pr77/src/anemone`. PR77 and PR78 themselves are unchanged.
- Python 3.13.5; Atomheart 0.2.11; Valanga 0.2.5; Torch 2.14.0+cpu. Distribution versions and Atomheart codec source fingerprint are retained in `environment.json`. No installed benchmark package was modified. Test/static tools use a separate new validation environment; dashboard compatibility selects its own Matplotlib/Pillow versions, which are unrelated to measured export timing.

`results.json` records all executed driver/raw result/source hashes. Retained research scripts add annotations, docstrings, formatting and safe output/relative-import controls to the executed diagnostic text; they do not change production data operations. Raw executed drivers and profiler files remain under `/tmp` at their fingerprinted paths. The retained allocation probe imports its sibling driver; the audit accepts an explicit fixture root and a new output file.

Reproduce only in new temporary fixture roots and explicitly pinned clean sources:

```sh
export CHIPIRON_EXPORT_SOURCE=/tmp/NEW_ISOLATED_PINNED_SOURCE
export PYTHONPATH=/tmp/PINNED_ANEMONE/src:$CHIPIRON_EXPORT_SOURCE/src
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=/tmp/new-export-mpl
python profile_morpion_export.py --root /tmp/NEW_FIXTURE --action prepare --nodes 1000
python profile_morpion_export.py --root /tmp/NEW_FIXTURE --action measure \
  --work /tmp/NEW_FIXTURE/flat-cold --mode cold --export-mode flat
# Repeat with a new work path for warm and sharded; alternate execution order.
# --cprofile is diagnostic only. profile_export_internals.py adds intrusive
# allocation snapshots; keep these runs outside throughput pools.
```

The driver rejects existing preparation/output directories and non-temporary measurement paths, and checks the synthetic fixture marker before measuring. It accepts only 1–10k requested nodes and 1–3 cycles. Fingerprint/verify all source SHAs before execution. Existing retained evidence is never used as a writable measurement workspace.

## Recommendation

Keep production warm reuse **disabled**. The earlier warm regression is unconfirmed, and the concrete sharded allocation patch is suitable for a separate draft review. Production activation still needs the flat/sharded target, provenance and DAG-parent issues resolved under the required review, a separately approved lifecycle/caller integration, and larger/longer asynchronous validation. There is no four-worker throughput or production memory-efficiency claim from these fixtures.
