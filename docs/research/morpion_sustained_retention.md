# Sustained operation: retention admission boundary

The inherited read-only auditor already fails closed. No collector or retirement
permission is added. Keeping a current capsule plus two predecessors is insufficient
under publication protocol 3: it still declares every committed capsule a root.
The target requires a protocol/lifetime change, rather than count-based pruning.

A read-only audit of the original operational G1–G8 workspace, under existing
Growth ownership and stable capsule pins, validates with zero conditional
candidates and zero authorized reclaimable bytes. Exact report:
`/home/pompote/oldata/victor/morpion-scratch/s26/e/historical-5k-reachability.json`.

| Boundary | Additional complete-tree root |
| --- | --- |
| G1 | Dataset claim `0957aeeb-dc43-4ab0-8781-daab4d05c6f1`, with null owner |
| G2 | Exact receipt source, accepted Dataset/Training and projected stage statuses |
| G3 | Exact receipt checkpoint, accepted Dataset and projected statuses |
| G4 | Exact receipt source, authoritative accepted cursor, accepted Dataset/status |
| G5 | Accepted Dataset/Training, every candidate and projected statuses |
| G6 | Previous boundary and exact receipt checkpoint |
| G7 | Previous boundary |
| G8 | Complete current crash recovery and pending proposal |

The G1 claim expired on October 25. Its empty metadata and null owner contain no
PID, process-group identity, attempt completion or quiescence evidence. Neither age
nor a currently empty `ps` listing proves the original operation's lifetime. The
claim remains untouched. A physical copy's absence of original readers cannot
authorize deleting original dependencies.

Live Dataset/Training contexts already know the exact admitted claim ID. Their
existing cleanup now participates in the bounded ownership wait, then compares
that ID again under Growth ownership before release. This avoids leaving a
conservative claim merely because Growth temporarily held the lock at cleanup.
It neither expires another attempt nor changes crash recovery. The focused
`test_live_owner_releases_its_claim_after_contended_cleanup` exercises this boundary.
An owner killed before cleanup still leaves retained evidence.

The existing tests in `test_morpion_rolling_recovery_report.py` establish that
expired claims, actual reader pins, pending exact proposals, explicit research
retention, unknown owners, external hardlinks, missing pins and corrupted receipts
suppress retirement candidates. `test_morpion_activation_completion.py` separates
historical model/descriptor acknowledgement from unrelated tree bytes on disposable
copies. It also verifies that accepted rows and every candidate remain checked.
The production `prune_generation_files` rejects durable roots independently of
requested count. These mechanisms are retained rather than duplicated.

A future narrow retirement change must explicitly authorize absence of selected
historical payloads in existing publication authority, preserve the authenticated
descriptor ancestry and exact UUID/digest receipts, distinguish completed input
identity from currently executable dependencies, and prevent new readers from
entering a retiring boundary under Growth ownership and existing exclusive pins.
It must define which two complete recovery boundaries survive and how their
producer/model/cursor/receipt context is sealed. Current recovery must still use
only the current head; intentional rollback is a separate admission question.
No generation watermark, shared-block engine, replay checkpoint or second ledger
can substitute for those facts.

Until that contract exists, bounded experiments use explicit time/step, RSS and
disk limits and stop with complete current recovery. Storage is bounded by the
experiment budget, not by automatic historical retirement. Indefinite unattended
search remains blocked on retention and cumulative publication/receipt metadata.
