# Amortize existing publication with bounded batches

> Historical generated outputs are available through the [evidence archive](../evidence-archive.md); original paths in reproduction commands refer to restored or newly generated outputs.

Three 360-second real four-worker sessions copied the same fresh legal G1:
10,005 nodes / 11,014 edges. Model, scientific configuration, executable, 2-second
pause, 240-second worker ownership wait and monitoring limits were fixed. No CI or
other benchmark overlapped these serial sessions. The actual installed baseline
is Git `1946c601`, wheel SHA-256
`bfe5f0350992a796e4a1a864d8d28e26bc238669c0ac52754395807a1d90bf90`;
worktree implementation edits were not installed or imported by those workers.
The original research recipe bytes are retained in `s27/baseline-reproduce.py`.
Every case starts from an independent copy, with private control/cache/authority.

| Steps / invocation | Committed nodes / edges added | Committed steps | Nodes/s | Accepted Dataset / Training | Completed ownership s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 699 / 813 | 20 | 1.937 | 3 / 2 | 289.52 |
| 16 | 1,616 / 1,894 | 48 | 4.478 | 2 / 1 | 271.78 |
| 32 | 3,324 / 3,857 | 96 | 9.189 | 1 / 1 | 318.99 |

The selected batch is **32**, an observed **4.74×** durable-node rate versus 4.
Whole-session CPU per new node also decreases. Each accepted Training has real
positive samples; initialized model bundles are not counted. This is an observed
single matched session per case, not a statistical confidence interval. The first
16-step restore took 56.4 seconds versus 22–34 seconds in most other invocations;
the full native measurements are retained rather than normalized away.

32-step ownership windows were 99.90–113.19 seconds, versus 50.29–69.97 for 4.
Its Dataset accepted at 214.6 seconds and Training at 333.2 seconds, following
Dataset by 118.6 seconds. This trades learning latency for amortized search; it
does not establish FIFO scheduling or guarantee the same latency at larger sizes.
The longer real run must allow sufficient acceptance time and keep waiting bounded.
The existing recipe now defaults its existing `--growth-steps` option to 32; all
other batch/cadence controls remain explicit. No scheduler or production API is added.

32 steps attempted 119.06 seconds of useful Growth; completed ownership was
318.99 seconds. Three completed cold restores cost about 89.8 seconds. Warm reuse
could remove at most that measured portion, before accounting for authority or
pending-work invalidation; it would still pay scientific export/publication.
The 4-step control attempted six restores and 23.71 seconds of useful Growth.
Attempt logs can include work abandoned at shutdown; committed selector/checkpoint
counts above determine actual progress.

Warm reuse is not implemented. A safe cache would have to bind the entire authority
(including Dataset/Training metadata that can change without a head advance),
model/config/epoch, producer cursor, pending patches and exact checkpoint identity,
yield ownership, and discard on stale bindings. Commit currently deliberately
releases its source pins and discards the runtime. The measured batching gain is
large enough to retain that simple cold-recovery contract instead of introducing
another cross-invocation transaction path. Future restore/export work should first
be measured at the larger graph size.

Exact acceptance identities, per-role contention, native CPU, sampled process-group
RSS, staging/scratch peaks, checkpoint/scientific allocation and acceptance latency
are preserved in [the comparison](https://github.com/victorgabillon/chipiron/blob/d8c94d6c07dd072e6e25620ea194861644fa42e5/docs/research/bounded_history/evidence/batching-comparison.json) and each actual
case's artifact report. No candidate was activated during the matched comparison.
