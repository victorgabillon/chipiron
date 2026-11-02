# Persistent Python worker jobs

`MORPION_PERSISTENT_WORKERS=1` opts the existing monitored, headless scientific
four-worker launcher into persistent Python processes. Absence or `0` retains
fresh Python invocations. This is an operational setting, excluded from scientific
configuration identity. Initial support requires explicit CPU or `cuda:0`,
`strict_math` with one intra/inter-op thread and serial continual candidates.

Each process repeatedly calls the existing bounded launcher/stage API. Growth
still performs at most one cycle per job and retains its cold restore and receipt
runtime discard. Every job re-resolves configuration, run state and permitted
source through existing admission. J2 reservations/gates, candidate seeds and
own accepted parents, publication, activation/refresh and source pins are unchanged.
No tree, reader, model or optimizer is retained by the dispatcher between jobs.
Existing same-source Reevaluation disk cache scope remains; Dataset scope rotates
per job. Split LRUs are cleared, garbage collected and initialized GPU allocator
slack released at job boundaries. No GPU context is initialized just for cleanup.

The existing process execution-profile owner configures Torch once. Incompatible
policy/device changes require restart. There are no new RNG seeds or training
loops; existing per-candidate continuation resets remain authoritative. Interpreter,
imports, libraries and possibly driver memory remain resident, so idle RAM can be
higher than with fresh workers. Existing aggregate memory/disk guards still apply.

Python is the session/process-group leader, with output appended directly to its
worker log. Unexpected worker exit stops every owned group, including orphans,
before supervisor cache cleanup. Any job or cleanup failure exits the process;
there is no same-session retry. Existing conservative claim quiescence checks
remain unchanged: idle status does not authorize replacing a claim from a live
worker or surviving old session. Explicit restart is allowed only through the
existing admission and recovery protocol. SIGINT/TERM unwind normal ownership;
existing bounded group termination remains the fallback. No lock protecting
Growth, publication or source inputs is held during idle polls.

`pipeline/worker_processes/<stage>.json` records current process/monitor identity,
job number, active-job clock, boundary/idle heartbeat, cleanup and dispatch
configuration/state hashes. A separate operational stage lock prevents two live
registrations. Dispatch hashes are observations, not the admitted source/model;
existing claim/publication evidence provides that authority. Monitor timeout uses
matching active-job age and excludes fresh idle polls. Missing/stale/mismatched
status falls back conservatively to process age. Heartbeats occur at boundaries
and idle polls, not from an independent progress thread during computation.

Startup is recorded once per process. Bounded runtime slots distinguish each job,
cleanup and poll interval; lifetime RSS peaks remain lifetime measurements.
`--persistent-worker`, `--worker-poll-seconds` and `--worker-max-jobs` are launcher
controls for supervised bounded probes; the cluster supplies each stage's existing
poll delay and the monitor bounds total lifetime. This is not a new scheduler.

Reader evidence/index optimization (I6.2), live mutable tree reuse (I6.3), J3
latest-source scheduling, parallel candidates and new experiment recipes are
outside this change. Exact equivalence and performance claims must state the
validated device/job sequence and distinguish startup from restore, reader,
preparation, training and activation costs. Raw validation evidence stays outside
Git; final qualification and review remain with integration.
