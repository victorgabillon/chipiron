# Independent fork of a quiescent managed run

`chipiron-bootstrap fork` prepares a new folder from one positively closed,
fully settled, compatible managed parent. It does not start workers.

```sh
chipiron-bootstrap fork --from runs/parent --to runs/child \
  --max-elapsed-seconds 1800 --max-tree-nodes 5000 --dry-run
chipiron-bootstrap fork --from runs/parent --to runs/child \
  --max-elapsed-seconds 1800 --max-tree-nodes 5000 --expect-plan DIGEST
```

The preview reads bounded metadata and file identities. It loads no checkpoint or
tensors, writes no parent files and reserves no future cut. Execution recaptures
the cut under existing monitor/Growth owners and lifetime pins. The parent must
use this same fork-capable build, dependency identities and saved execution profile,
publication-v3/evaluator-contract, atomic Reevaluation, fenced Dataset and durable
Training, with continue-weights/continue-Adam. There is no historical migration or
cross-boot recovery. Active, failed, uncertain, unlaunched and unsettled parents
refuse. New v3 publications durably prepare their stable lifetime pins before
visibility, including terminal heads without a first reader. Fork still requires
existing pins and never repairs missing ones in a stopped source. Even an expired claim, unresolved publication stage, reserved learning
window, pending patch or activation request requires ordinary recovery first.

The child retains the entire committed search state, search/rollout RNG and
certificates, the separately committed incumbent, and each name's latest accepted
weights and Adam. If the parent is a fork, an otherwise unresolved name uses its
verified local imported seed. Never-trained names remain explicit `None`. Ancestor
paths survive only as opaque origin evidence; operational paths resolve locally.
All copies use independent inodes. No Dataset row archive or old input cache is
imported: ordinary destination Dataset work regenerates current local data.

Generation numbering, workspace/producer/epoch, learning windows, patch receipts
and native authority start afresh. The explicit finite time budget starts with zero
consumption; the absolute node goal must exceed the retained node count. Other
scientific settings and resource caps inherit. The parent stays unchanged, including
its terminal node cap. A native node stop can mean either `maximum_unique_nodes`
(the cap is reached) or `next_complete_expansion_exceeds_unique_node_budget`
(the next whole expansion would cross it). Both require bound committed counts and
an explicitly larger destination goal; their original reasons remain distinct. New local generation seeds use the existing rule; an identical
future suffix to parent resume is not promised.

Training admissions use schema3 to bind `fork_origin_v1`; schemas1/2 remain unchanged
for existing runs. The normal continuation primitive loads imported weights/Adam.
A real local acceptance supersedes an imported parent per name, including losing
candidates. The resolver preserves other names if resolving a subset; this feature
does not enable restricted continual rounds, which remain forbidden by the existing
configuration contract. All origin evidence and seed bundles are permanent retention
roots in this first slice, even after supersession. The rolling report and offline
continual audit validate and label the origin separately from local acceptances.

Preparation uses a private sibling stage bound to the intended final canonical root.
The stage cannot execute as a normal managed workspace. A fresh owned process uses
the existing cold loader, checkpoint writer and genesis publisher; it performs no
growth or training. Its inherited destination lease and parent-death signal protect
against an orphan writer. The coordinator kills/reaps its private group before
releasing source owners/pins. A positive verified inventory marker precedes fsync and
atomic no-replace publication. Normal `start --work-dir runs/child ...` is a separate
operator action after publication.

Preparation has file/metadata limits and finite read/output estimates, with inherited
wall/RSS/scratch/workspace/free-space/RAM limits checked during copying and helper
execution. Live owned-output size sampling tolerates disappearing temporary names
during atomic export renames. Links, special files, missing roots, IO errors and
file-count overflow still fail; final closure verification stays strict after the
helper is reaped. Estimates are admission reservations, not a proof every size will fit.
Unsupported oversized closures refuse. A hard failure preserves bounded labeled
stage evidence and never repairs the parent or replaces a foreign target. One exact
retry may quarantine an owned incomplete attempt; its original monotonic deadline
and conservative previous read reservation carry forward. A second incomplete retry
refuses. A validated stage can be reverified and published without repeating genesis, within
the original deadline and remaining one-retry read reservation.
An exact expected plan can acknowledge a published fork and finish directory durability
even after child progress or parent removal. Directory existence alone is insufficient.

CPU `strict_math` with Torch1/1 is the numerical acceptance fixture. AMD installation
identity checks do not establish AMD numerical replay. Broader validation and independent
review remain required before integration.
