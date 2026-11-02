# K1 current-pipeline dashboard

The existing five pages remain Overview, Record, Tree, Evaluator and Operations.
Overview and artifact-pipeline Operations are read-only. Legacy single-process
controls remain available for old runs. No monitoring action launches scientific
work, loads weights, probes GPU hardware or initializes Torch.

To associate a B3 runtime explicitly, pass both paths to the Streamlit entrypoint:

```bash
python -m streamlit run \
  src/chipiron/environments/morpion/bootstrap/dashboard_streamlit_entry.py \
  --server.address=127.0.0.1 --browser.gatherUsageStats=false \
  -- --work-dir /path/to/runtime/work --runtime-root /path/to/runtime
```

The installed entrypoint has the same script arguments. Without `--runtime-root`,
only workspace evidence is used; a parent's `status.json` is never discovered.
The B3 legacy adapter verifies `LAUNCH_READY.json`'s runtime, scientific work-dir
(`scientific_work_dir` or older `work_dir`) and exact run command. Its receipt's
run command must agree. Unknown explicit schema versions are refused.

Completion requires agreement between the declared early goal, status, current
scientific head, chronological receipt and lease release, and confirmed child
cleanup. Normal SIGINT/143 worker shutdown does not turn that goal into a stop.
Manual cancellation, failure, unconfirmed cleanup and contradictory evidence are
reported separately. PID identity includes boot/start ticks and, where recorded,
command/session/namespace; timestamps alone cannot establish liveness. Missing
historical process identity is deliberately unknown.

## Observation contracts and bounds

The dashboard reuses J1's `EvidenceReader` for bounded metadata and workspace-path
containment. Current stage paths come from the saved publication index, limited
to 64 recent publication generations. Legacy stage discovery is capped at 256
artifacts before retaining 64; if the cap is exceeded, current stage status is
withheld rather than treating an arbitrary sample as the latest generation. Oversized/corrupt/missing artifacts produce unknown values
or visible errors. History reads retain at most 4 MiB / 2,000 rows. Stage artifacts
use J1's 256 KiB ceiling, active bundle metadata 1 MiB, run authority 4 MiB.
No overview read walks a full workspace or reads science/result reports, tree
payloads or weights. Full history, storage, J1 performance and detailed evaluator
diagnostics are explicit/lazy. Evaluator reads at most 32 candidate manifests.

I6.1's optional v1 `pipeline/worker_processes/<stage>.json` is capped at 16 KiB.
Growth, Dataset, Training and Reevaluation validate worker and monitor identity.
Idle needs a heartbeat within five seconds; starting/running/cleanup use the
active job clock, since active heartbeats occur only at job boundaries. Process
lifetime is not job age. Missing/unknown versions never become live workers.
Existing J1 lifetime/wait records are bounded by their 256-slot retention and
use verified identities; sampled wait decisions require freshness. Saved waits
are translated into input/learning-gate/ownership/activation context without
claiming scientific ownership. The current persistent status does **not** record
admitted source/model or a precise wait reason; those remain unknown. A job
boundary timestamp does not prove continuous progress.

## Scientific and resource meanings

Lookahead and window progress count ordinary Growth cycles. Dataset/model lag
counts publication generations. Accepted Training references, selected candidate,
active pointer and committed activation outcomes are distinct. Pending activation
uses the latest Training's recorded request and its canonical digest in publication
authority; absent request evidence is unknown, not a guessed queue state. Candidate
lineages retain each candidate's own parent bundle and optimizer continuation.

Active-model metrics prefer flat current bundle metadata (`num_epochs`,
`num_train_samples`, `num_validation_samples`, `validation_loss`, `learning_rate`,
`batch_size`) over legacy `training_config` / `validation_metrics`. Missing flat
keys fall back; explicit nulls remain null. Latest Training never supplies missing
active-bundle metrics. Validation loss is labeled MSE only when the recorded loss
name is `mse`; legacy explicitly named MSE remains supported.

Configured device/profile/threads are separate from reported active-bundle values
and historical worker environment receipts. `cuda:0` alone does not identify a
vendor; saved HIP/device-name evidence can identify AMD. Preparation opt-ins and
cache observations are shown only when saved. Missing GPU utilization, unique
physical memory or profile evidence remains unknown. Summed process RSS can count
shared pages repeatedly; Torch allocator peaks are not total GPU memory. Owner
elapsed time includes startup/cleanup and is distinct from canonical monitor time.
Resource session evidence is the session selected by `resource_monitor/usage.json`;
it is historical, not a new hardware sample or a safety guarantee.

The prominent score plot contains best observed exact/terminal **total points**
versus elapsed seconds. Greek cross starts with 36 points; moves since start are
separate. Elapsed starts at the declared owner start or, before a receipt exists,
the saved owner's process creation time. Dataset publication and Growth observation
times are observations, not proven discovery times. Coverage is recent inspected
saved data, not an exhaustive final-tree record. Loss curves span changing training
and validation datasets and do not establish playing strength.

## Known heavy-view limitation

A small current closed-scientific fixture is valid through the current scientific
reader, but the existing full-history materializer only understands legacy flat
and sharded training snapshots and returns no snapshot for this format. Thus its
whole-tree classifications/current record reconstruction are unavailable for that
format. K1 retains this separately scoped limitation; it does not route overview
refreshes through a full scientific oracle. Existing checkpoint-based Tree
inspection and legacy sharded inspection are covered by their tests; no full-tree
inspection of the real L1/L4 experiments was performed.
