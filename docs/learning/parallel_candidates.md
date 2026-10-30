# Bounded CPU candidates (I1, opt-in)

The continual streaming Training path can run independent candidates in spawned
processes. The default is still inline serial training. Other paths remain serial;
parallel streaming execution requires a continual policy and `training_device: cpu`.
An explicit `auto` device is rejected for this mode, even on a CPU-only host.

Set operational controls in the Training worker's inherited environment:

| Variable (`CHIPIRON_CANDIDATE_` prefix) | Default | Meaning |
| --- | --- | --- |
| `WORKERS` | `1` | Requested concurrent candidates; `1` preserves legacy execution |
| `THREADS` | `1` | Torch intra-op and native BLAS/OpenMP threads per spawned child |
| `CPU_BUDGET` | `4` | Upper limit for candidate workers × threads |
| `RESERVED_CPUS` | `2` | Capacity left for coordinator and other pipeline workers |
| `MEMORY_MB` | continual `max_group_rss_mb` | Aggregate coordinator/owned-candidate RSS stop threshold |
| `WORKER_MEMORY_MB` | `1024` | Per-candidate RSS reservation and stop threshold |

All values must be positive integers. Increase reservations explicitly for larger
models. The resolved worker count is bounded by requested count, candidate count,
CPU budget, affinity/visible ancestor cgroup quotas minus the CPU reserve, and
available host/cgroup RAM minus the continual RAM reserve and coordinator RSS. No capacity
means refusal, not an increased budget. Thread count and scientific settings are
never automatically tuned. These controls do not enter historical configuration
or admission hashes. With `WORKERS=1`, existing process thread settings apply;
use `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` for serial comparisons.

The coordinator reports its plan and persists it in the attempt's
`candidate_execution/execution.json`; successful Training status includes that
observation. It records cache preparation time, wall time, sampled aggregate peak
RSS, completed candidate names and the child-reaping barrier. Each candidate has
its own CPU/wall/resource observation. These are execution provenance, not new
hyperparameters. Default scientific configurations and the pinned 200k recipe are
unchanged.

## Ownership and failure

D1 admission already makes a private, byte-identical copy of accepted rows. The
coordinator prepares caches there before spawning. Flat, ordinary entity-token
and relational caches have explicit read-only loading: missing or stale data in a
child fails instead of rebuilding. State-split memoization is process-local. No
child writes shared cache manifests. Existing chunking, `max_rows=None`, tensor
cache formats, sample schedule and training primitives are retained. Existing
cache builders and stable split bookkeeping still scale with row count; this
change does not claim constant-memory training. Large inputs must fit the declared
memory budgets or stop.

`CandidateJob` carries the unchanged streaming recipe, seed, accepted parent and
private output path. Both executors call the same training service. Children own
model/Adam/RNG state, output bundles and temporary directories. The coordinator
alone assembles ordered results, diagnostics, selection, status and acceptance.
Results are consumed in configured candidate order only after every child exits;
a child failure aborts the whole round. Accepted retries retain existing receipt
recovery and do not launch jobs.

The 50 ms resource monitor covers preparation and execution, including elapsed
and RSS limits. These are sampled stop thresholds, not kernel-enforced memory
reservations; transient allocation can overshoot. The existing continual operator
still supervises the entire pipeline group and disk/workspace budgets. Configure
reserves for the other workers; candidate limits do not replace that supervisor.
SIGTERM/SIGINT, worker failure and resource stops terminate, kill if necessary,
and join every owned child before control returns to input cleanup or acceptance.
Cleanup ignores repeated termination signals. Every child also watches a private
pipe whose only writer is the coordinator: coordinator SIGKILL closes it and
terminates the child without allowing later bundle writes. The dead coordinator
cannot reap; its existing outer supervisor/OS reaps adopted children. There is no
acceptance or input deletion from that dead attempt.

## D2 and G1 boundaries

The D2 proposal keeps recipes and candidate ordering in Morpion and puts the
admitted-attempt lifetime above this executor. `CandidateJob` and `run_candidate`
are the narrow seam for that work; this PR adds no reusable bootstrap engine or
second tensor training loop.

[G1](https://github.com/victorgabillon/chipiron/pull/92) owns shared device dispatch.
I1 leaves it unchanged. Spawned CPU processes inherit hidden CUDA/ROCm devices and
thread limits before imports, and assert CUDA was not initialized. CPU continuation seeds only the CPU generator; `torch.manual_seed` would also
dispatch to accelerator generators. CPU RNG evidence avoids `get_rng_state_all()` even on GPU-capable installations;
CPU evidence retains an empty `torch_cuda_hex`. CUDA training still captures CUDA
RNG. This deliberately changes irrelevant GPU evidence for CPU runs on GPU hosts,
not their CPU tensors or RNG. Multi-candidate GPU execution is excluded.

## Evidence interpretation

Exact comparisons cover tensors, Adam state/counters, CPU/Python/NumPy RNG,
validation populations, losses and final selection over three rounds with 2/4
workers. Real process tests cover worker failure, RSS stops, graceful termination
and coordinator SIGKILL, checking writer PIDs and output stability. Accepted-round
and retry tests run with requested counts 1/2/4. Cache rebuild flags, preparation
and training times, PIDs, paths, archive hashes and parent bundle hashes derived
from those metadata may legitimately differ. Scientific comparisons must not
strip actual tensor, optimizer, parent-lineage or validation-population evidence.

Raw benchmark artifacts stay outside Git. Report all nine candidates together,
including the transformer, with unchanged architecture, precision, optimizer,
batch sizes and five epochs. A cold tensor cache means the attempt cache was
absent; it does not mean the OS page cache was flushed. Tiny equivalence fixtures
are not performance evidence.
