# Artifact-pipeline compute and memory observations

Observations are always on, taken at stage boundaries, and do not change search,
training, evaluator selection, checkpoint formats or worker scheduling. They do
not deserialize a checkpoint for the dashboard. No background sampler is used.

## Persistence and units

Small atomic JSON records live in
`pipeline/performance/generation_NNNNNN/<stage>-<pid>-<time_ns>.json`.
Each stage writer owns its records. Passive runtime records described below use
bounded slots; they do not update scientific state or a global summary file.
Retries and repeated reevaluation patches remain distinct. Metrics write errors
are logged without failing or retrying scientific work. Temporary files are
ignored by readers. This is rename-atomic publication, with the same filesystem
crash-durability limitations as existing status artifacts (no additional fsync).

Every completed observation has `schema_version=1`, `generation`, `stage`, `pid`,
`started_unix_s`, `finished_unix_s`, monotonic `elapsed_s`, `rss_before_mb`,
`rss_after_mb`, `process_peak_rss_mb`, `available_ram_before_mb`,
`available_ram_after_mb`, `cuda_before`, and `cuda_after`.

RAM units are **MiB**, despite the existing helpers' `_mb` names. The RSS peak is
**process lifetime**, including previous generations, imports and restoration;
it is not a stage-local maximum. Available RAM is system-wide. No inference of
peak RAM is made for historical runs that only saved boundary RSS.

CUDA observations are optional. CPU-only workers never invoke allocator APIs.
Non-ML stages do not import Torch. Individual training/inference windows reset
peak allocator counters and synchronize only at their boundaries. They record
`device`, `device_name`, `device_total_bytes`, `allocated_bytes`, `reserved_bytes`,
`max_allocated_bytes`, and `max_reserved_bytes` before/after. These describe the
worker's PyTorch allocator, not other processes, driver overhead, or total device
utilization. Boundary elapsed time includes completion of CUDA work; no batch
synchronization is added. The pre-window synchronization drains earlier work
before the timed interval. CUDA is unavailable rather than fabricated on CPU.

| Stage | Additional fields |
| --- | --- |
| `growth` | cycle_index; node_count_before/after; nodes_added; branch_count_before/after; growth_steps_requested; actual growth_steps (unknown for third-party runners without the optional counter); nodes_added_per_second |
| `checkpoint` | node_count; existing `components`: bytes, payload_build_s, serialization/compression/write/total timing, anchor/delta counts where available |
| `export` | node_count; existing `components`: bytes_written, rows_written, new/reused_node_count, row_build_s, json_encode_s, write_s; `profile` including state-ref conversion; `state_eviction` including rematerialization counters |
| `dataset` | input_tree_generation; rows_considered (snapshot node count), rows_emitted/output_row_count, bytes_written, snapshot_load_s, record_scan_s, frontier_scan_s, rows_extract_write_s, leaderboard_s |
| `training` | dataset_rows, selected_evaluator, selection_policy; interval includes row loading, diagnostics, selection and publication |
| `evaluator` | evaluator_name; metrics: train/validation loss and MAE, epochs, train/validation sample counts, final_loss; interval covers training service including model construction and save, excludes separate diagnostics |
| `reevaluation` | model_generation, evaluator_name, patch_id, rows_requested/reevaluated, node_count, inference_window, model_inference_s, model_load_s, patch_rows_build_s, patch_build_write_s, start/end/next cursor, completed_full_pass_count |
| `patch_apply` | patch_id, model_generation, rows_requested/applied, existing counts including missing/recomputed/selector_invalidated |
| `cycle` | Growth cycle_index, node_count, nodes_added; includes restore, patch consumption, growth, checkpoint/export |

Per-evaluator boundary observations are also stored in the optional `performance`
field of each evaluator result in `training_status.json`; legacy results default
to an empty mapping. Existing scientific metrics remain unchanged.

Dataset extraction and streaming write share one iterator, so their combined
elapsed time is intentionally labelled as such. Reevaluation's
`model_inference_s` measures host evaluate calls only; the enclosing
`inference_window.elapsed_s` includes state reconstruction, model load, row
construction and CUDA completion. These nested times must not be added together.

## Dashboard

**Operations → Performance & scaling → Load performance measurements** reads only
small JSON metrics/manifests/status files, on request. It shows per-generation
stage times, nodes/+nodes, dataset rows, nodes/sec, RSS/GPU peaks,
boundary RSS and minimum observed available RAM; duration trends
against generation, nodes or dataset size; per-evaluator time/memory/loss/MAE and
selection; and a stage timeline with process, model-generation and patch IDs.
The Evaluator page also plots all saved evaluator loss curves together.
Overview's refresh path is unchanged.

Readers bound ordinary artifacts to 256 KiB and authority metadata to 4 MiB,
show the latest 256 generations, and cap stage files at 2,048 per directory and
normalized observations at 16,384, with explicit warnings if truncated. Missing old
fields remain N/A. Existing growth/checkpoint/export/cycle metrics are read from
manifest metadata. Raw records remain available for external analysis.

Stages overlap across processes. Stage totals include completed retries;
`cycle` is **Growth's cycle**, not the sum of pipeline stages. Timeline span is
first observed start to last observed finish, including idle gaps. The table is
sufficient to inspect overlap and patch dependencies, but does not claim to
compute a causal critical path. Historical interrupted stages may have no completion record; absence is not
proof of failure. Retained Training admissions and new passive invocation starts
expose incomplete attempts without inventing their end times. Unsaved growth
cycles carry their cycle_index and observed generation, so multiple observations
can share a generation. Peak memory comparisons must account for worker restarts.

## Read-only post-run summary

`python -m chipiron.environments.morpion.bootstrap.performance_report --work-dir PATH`
prints JSON to stdout using the same `load_performance_summary` as Operations.
Redirect stdout to a separate runtime directory. The reader never repairs
projections, acquires ownership, loads tensors, or opens scientific snapshot
payloads. This is reporting, not a scientific artifact audit.

Accepted Training references come from bounded `run_state.json` metadata. The
reader verifies the small proposal's exact SHA-256 and joins its
`status.metadata.round_performance`, nested evaluator `performance` fields and
private `pipeline/performance` records. Copies of the same interval count once;
their source paths remain visible. A round covers the recorded pre-admission to
status-construction interval; the separately labelled Training core is not added
to it. Unaccepted, failed (only when explicit) and incomplete attempts remain
separate. Multiple attempts indicate retry evidence without inventing attempt
order or failure causes. Legacy completed projections do not prove acceptance.

Committed activation descriptors supply refresh measurements and training/model
identity; the outer activation measurement remains its own stage total. Tree
publication generation can differ from Training/model generation. Refresh and
evaluator intervals are children, not additional parent-stage wall time.

The report includes observed endpoint span, union coverage, gaps within that
span, distinct-process interval overlap and residual time within measured
parents. These are not a causal critical path, CPU utilization, or a complete
run budget. Incomplete attempts can extend outside recorded bounds. Parent
residuals and gaps are **unattributed**, not diagnosed waits. Clock inversions or
wall/elapsed discrepancies above max(50 ms, 0.1% elapsed) are flagged and excluded
from interval arithmetic; raw fields remain inspectable. Missing duration is N/A.

## Passive runtime boundaries

`pipeline/performance/runtime/` contains small best-effort slot records (at most
16 KiB each, 256 slots per worker/event family). Collisions replace older evidence;
reports explicitly state that runtime totals cover retained evidence only. This
is diagnostic retention, not a claim journal. It adds no scientific locks,
per-node/per-batch synchronization, changed polls, training settings or budgets.

Growth's existing learning-boundary decision and Dataset/Training's existing
pending/claimable diagnostics coalesce sampled waits by shell stream and reason:
`learning_boundary`, `no_input`, or `claim_contention`. Repeated polls update a
single open interval; a changed decision closes it. Open intervals expose
`observed_elapsed_s` through the last observation, while completed duration stays
unknown. These sampled spans include polling and restart overhead; they do not
prove the predicate was continuously true between observations. Linux boot ID
and parent creation ticks prevent joining unrelated restart streams.

Ownership observations use existing acquire/timeout/release boundaries, recording
wait and hold separately after timeout or release. They never affect the lease
or propagated exception. A raised owned body emits no ownership record, preserving
rejection paths that require a byte-identical workspace; absence is not a zero
duration or proof of success. Launcher dispatch stores an incomplete invocation and
replaces it with completion/failure on exit. Linux startup is process birth to
dispatch, including interpreter/imports/configuration; its Unix placement is
explicitly anchored from process start ticks and CLOCK_BOOTTIME at dispatch.
Unavailable startup clocks remain missing. Repeated process identities make
restart costs visible without claiming why a supervisor restarted a process.

`cpu_s` measures only the current process. `children_cpu_s` is separate waited-for
child CPU and excludes live children; it is not complete process-tree CPU.
`process_peak_rss_mb` remains a process-lifetime peak. Device timing scope says
whether host wall time has successful device synchronization at both existing
boundaries; it is never relabelled GPU-kernel elapsed time. Historical host
inference timings without synchronization evidence remain explicitly unknown.

## Historical gen38 continuation audit (2026-10-05)

Preparation inspected
`generic_linoo_fresh_with_bigrun_models_v1` read-only. No experiment was launched.
The desired temporary change is 10 → **2 growth steps per cycle**, with at most
**10 Growth cycles**, aiming approximately at generations 39–48. The persisted
configuration remains authoritative. The first 1–2 generated observations should
report actual nodes per step; parameters must not be adjusted automatically.

**Preparation is blocked; this is not a launch-ready continuation.**

- Run state is generation 38 / cycle 42; checkpoint and training export each
  contain 242,680 nodes. Restore must use the runtime checkpoint. The checkpoint
  manifest's embedded generation is null; generation provenance comes from run
  state, paths, evaluator_version=38 and export generation=38.
- Checkpoint metadata has `rng_state: null` and no `rollout_rng_state`. Current
  Anemone supports restoring these fields but cannot recover states never saved
  by the historical producer. Identical stochastic continuation is unprovable.
- Current main rejects the historical evaluator mapping, which uses `graph_*`
  settings. Silently rebuilding it from current defaults is unacceptable.
- The active bundle is `mlp_41` from model generation **430**, restored manually;
  gen38 training selected **mlp_20**. The training scheduler correctly infers an
  external seed and keeps the local lower bound at 38, so training 39–48 is
  eligible. A separate active-model publication guard compares against raw
  generation 430 and would leave that bundle active after local training.
  This blocks promotion of the automatically selected local model, not training
  itself. No forced evaluator is configured. The existing regression
  `test_training_stage_trains_local_generation_after_external_seed` explicitly
  verifies this distinction.
- Existing workers choose the latest pending generation, so an asynchronous run
  does not guarantee training of every intermediate generation. Changing that
  scheduling is outside observational instrumentation.
- Reevaluation publishes a singleton patch consumed by Growth. Once bounded
  Growth ends, a final patch cannot be drained/applied without an additional
  explicitly designed consumption phase. Repeated worker polling cannot solve it.
- Gen38 records about 19,081 seconds (5.3 hours) of summed evaluator training,
  including an anomalous 11,479 seconds for linear_10. Ten generations cannot be
  promised within the preferred 12-hour budget; these historical numbers need
  calibration, not a confident linear extrapolation.
- In-place continuation also invokes retention pruning. A separate derived
  workspace is needed to preserve every historical artifact.

The external review/preparation directory contains a read-only audit command,
exact persisted config with the proposed single override, checksums of restore
artifacts, and a launcher guard that refuses execution. A derived experiment,
explicit configuration migration/RNG reset, active-model provenance handling and
bounded drain design require Victor's decision before a real launcher can be
made correct. None are silently introduced in this instrumentation PR.
