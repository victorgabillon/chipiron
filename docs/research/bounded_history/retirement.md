# Explicit payload retirement v1

`run_state.json` remains the sole authority. A closed publication record may carry
`retired_payload_v1: <that record's exact descriptor SHA-256>`. This admits absence
only of the sealed `checkpoint/` and `scientific_snapshot/` inventory entries.
Every descriptor, frozen model bundle, config, attempt marker, accepted result,
receipt and cursor survives. Present payload bytes are still checked during a
partial cleanup. Unmarked missing files, changed surviving bytes, unknown files,
symlinks, unsupported markers and missing required models fail closed.

Admission is limited to closed publication 3 with explicit evaluator bindings.
The current and two immediately previous search/scientific boundaries cannot be
marked. Accepted Dataset/Training generations cannot be marked. They retain their
full input contract, including independent Dataset reconstruction and existing
selected-input retries. A marker neither resets a receipt nor advances any model,
cursor, epoch, selector, RNG or scientific generation.

The existing reachability tool has an explicit `--retirement-plan` projection and
an explicit `--retire G` operation. Ordinary inspection is read-only. Planning
separates historical exact receipt checkpoint/source identity from the old tree:
the exact UUID/content/cursor digest, authoritative descriptor binding and original
frozen model bytes remain checked. Current accepted cursor, unacknowledged/pending
proposals, accepted source/results, claims regardless of age, projected unfinished
stage state, unknown controls, real reader pins and caller-specified research roots
remain independent roots. Unknown or external hardlink ownership blocks retirement.
No missing PID, TTL or supervisor status releases a claim.

The transaction holds existing Growth ownership and the analyzer's stable capsule
pins. It validates the current and retained boundaries and every dependency, rejects
unknown staging ownership, and selects only the proven dependency-free generations.
Before any unlink it durably replaces run state and fsyncs the authority and its
parent. Only then does it unlink the exact authorized inventory paths and fsync
each affected directory. All candidate pins remain exclusive through cleanup.
An arriving scientific reader must acquire its pin and re-adopt run state; a retired
boundary is no longer an input, even if all of its bytes happen to remain.

A pre-authorization crash leaves the complete old history. A crash between rename
and directory fsync leaves either admissible authority with all old payloads still
present. A crash after authorization permits any partial physical cleanup. Retrying
validates all remaining required evidence, fsyncs the adopted authority, and finishes
only the already permitted unlinks. Empty payload directories may be removed;
remaining capsule roots and pin inodes never disappear. Repeated cleanup is a no-op.
The generic orphan collector is unchanged and cannot delete a committed capsule.

Normal Growth recovery still selects only the current head. The two prior complete
search checkpoints are retained and can be independently decoded for offline audits;
this does not implement intentional rollback with an older producer/receipt epoch.
An unsupported old executable is not a permitted resume interface. Source upgrades,
unknown workers, claims or unresolved staging require separate reconciliation.

This task executes physical retirement only in generated disposable copies. Its
original fresh runs, historical experiments and research evidence remain intact.
`--retain G` is an explicit per-operation research root, not a new persistent registry;
repeat the required root set on each requested operation. Actual reclaimed allocation
must be measured after cleanup and reported separately from conditional candidates.

NEW Reevaluation admission additionally requires an unretired scientific source,
even if cleanup has not yet removed its bytes. Exact committed acknowledgement
uses the receipt's authenticated source identity and original model bytes instead.
A real retired-source regression distinguishes those paths without re-opening the
old source or weakening cursor/model/UUID checks.
