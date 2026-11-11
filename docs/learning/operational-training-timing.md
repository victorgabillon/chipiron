# Operational training timing

Legacy training retains synchronized fine phase timing by default. For an explicit
Morpion bootstrap opt-in, set `CHIPIRON_TRAINING_TIMING_MODE=host_enqueue` in the
Training worker's launch environment. Unknown values fail before fresh training.
This is operational instrumentation, outside bootstrap configuration hashes and
numerical execution profiles. A persistent worker captures the request for each
fresh job; spawned candidates carry their captured `MorpionStreamingTrainingArgs`.
Accepted retries return their stored telemetry, even if the environment has since
changed; they do not train or rewrite it.

Direct callers pass `timing_mode="host_enqueue"` to `train_morpion_regressor`, or
set that field on `MorpionStreamingTrainingArgs`. Both default to `synchronized`.
CPU controls and CUDA/ROCm support the opt-in; other devices reject it. Only the
retained AMD workload has timing-variant numerical evidence, not universal GPU
qualification. The installed production replay remains an explicit validation gate.

The setting reaches in-memory, raw streaming, flat-cache, entity-token-cache and
relational-cache training and full train/validation evaluation. The existing shared
batch `synchronize_cuda` parameter controls detailed timers. No computation, scalar
metric extraction, validation checks, precision, execution profile, batching, shuffle,
seed, optimizer or accepted parent changes. There is no global timing override.

Service, epoch and evaluation elapsed boundaries explicitly complete device work.
Fine opt-in keys are `host_enqueue_forward`, `host_enqueue_backward`, etc.; flattened
metrics become `timing_train_host_enqueue_forward_s`, etc. These are host/enqueue
spans, which can wait for earlier work. They are **not GPU operation durations** and
must not be compared by phase with synchronized spans. Old keys/default semantics
remain unchanged. Coarse durations remain device-complete host elapsed time; nested
scopes must not be added together. `timing.mode`, `timing.fine_scope`, and
`timing.coarse_scope` annotate model telemetry; `timing_mode` annotates metrics and
`performance.training_timing_mode` survives bootstrap result serialization/retries.
The dashboard shows that actual mode; old records are `legacy/unrecorded`.

I7.1's private same-backend exact replay observed about12.47% lower candidate-service
mean with outside CPU contention. This does not establish whole-bootstrap speedup.
The production change is opt-in; no batch/profile/kernel or dependency change follows.
