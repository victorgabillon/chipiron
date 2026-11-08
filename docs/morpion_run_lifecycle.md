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

Terminal, node-limited, cancelled or spent work returns `no-work`; failed or
uncertain shutdown/accounting requires explicit recovery. Usage and termination
receipts remain intact. New time grants, reopening, model/tree/data resets,
independent forks, external imports and code migration are future work and have
no supported flags here. This interface adds no scheduler or training loop.

No-argument dashboard, `run` prepared-script delegation and scientific worker
entrypoints keep their existing behavior.
