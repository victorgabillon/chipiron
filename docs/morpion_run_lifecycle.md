# Normal continual start and resume

`chipiron-bootstrap start` resumes a compatible current continual workspace by
default. Pass an explicit path for a new run:

```bash
chipiron-bootstrap start --work-dir /path/new-run --cluster-script /path/morpion_cluster.sh --dry-run -- --training-device cpu --max-tree-nodes 1000 --max-elapsed-seconds 60
```

Review the JSON plan, then omit `--dry-run` to execute. The first start requires
an explicit supervisor script. Later starts inherit its recorded path and hash.
Canonical recipe options follow `--`. Omitted options on resume inherit the
persisted recipe; explicit conflicts refuse. Preparation fixes its existing
artifact/sharded publication settings; omit overrides for those fixed fields.

`--mode resume` requires committed state. `--mode fresh` requires an absent or
empty destination. The default `auto` selects between these two actions. Partial,
corrupt, foreign, legacy, sealed experiment or incompatible source/runtime/profile
state refuses without initialization. Existing experiment wrappers retain their
own entrypoints. An old-build workspace cannot resume merely because a new release
is installed.

A read-only preview reports the committed head/incumbent, each candidate's accepted
weights and Adam parent (including losers), accepted Dataset references, native
consumed/remaining budget and last decisive endpoint. A null parent declares a
candidate's first training lineage. Preview checks bounded metadata and dependency
presence; native restore/admission still validate payload bytes at use. It does
not load trees or tensors or initialize a GPU context. Ownership is checked again
before native execution; preview is not a reservation.

New normal fresh runs enroll in managed time sessions at genesis. Existing
unmanaged/old-build/sealed runs are never enrolled automatically. On a clean,
confirmed time end, authorize another bounded period explicitly:

```bash
chipiron-bootstrap start --work-dir /path/run --add-time 1800 --dry-run
chipiron-bootstrap start --work-dir /path/run --add-time 1800
# Automated retries should keep their original client identity:
chipiron-bootstrap start --work-dir /path/run --add-time 1800 --request-id renewal-01
```

Seconds must be decimal integers in 1..43200. Grants add to the original allowance;
actual monitored time, including launch overhead, cleanup and overshoot, stays
charged. An increment must cover existing overshoot plus one second. For example,
1202.75 consumed against1200 plus1800 leaves1797.25 seconds. Recipe, optimizer,
architecture, seeds, precision, node/goal and resource caps remain unchanged.

The default request ID binds the preceding closed native session. Repeating a
pending request recovers its admission; conflicting ID content refuses. Once its
successor has ended, a new no-ID --add-time means another intentional grant. Stable
explicit IDs are required for delayed automated retries across endpoints. Reusing
an already spent request adds nothing and launches no duplicate work. Plain start
can launch an admitted-but-unstarted request or use remaining credit.

Status/dry-run distinguish original allowance, total authorization, finalized
consumption, remaining credit, last native decision and cleanup, and managed state.
While running, remaining credit based on finalized consumption is an upper bound;
ongoing work and cleanup are charged at close, never refunded.
Native maximum_elapsed_time and an earlier supervisor_deadline remain distinct.
An early clean supervisor stop with credit left needs ordinary start, not a grant.
The native invocation ceiling remains unchanged. Successful preparation alone does
not establish a running or completed managed session.

Node/goal termination remains terminal. Cancelled, failed, forced, resource-limited
or uncertain stops cannot be reopened by --add-time. A launch committed without a
positively validated final close requires explicit recovery; dead processes or old
usage samples alone cannot supply free credit. An exact validated close can finish
an interrupted accounting commit without repeating scientific work. All native
receipts and scientific terminal evidence remain intact.

This first managed slice supports the same local host, boot and PID namespace with
fresh coordinator processes (the normal CLI uses exec). Cross-boot/remote recovery,
per-invocation --session-seconds, cap extensions, resets, forks, imports and migration
remain unsupported. Native restore/admission still verifies scientific dependencies;
renewal itself does not regenerate data, train, activate or refresh a tree. The
existing claim/source/quiescence protocol owns pending work and every candidate
continues its own accepted weights/Adam, including losers. Accepted weights are
separate from the active incumbent.

No-argument dashboard and run prepared-script delegation keep their behavior.
Existing advanced unmanaged APIs retain their policy/receipt contracts. Frozen
experiment wrappers/reporters are not managed-run completion authorities; use the
new normal status/accounting view. This interface adds no scheduler or training loop.
