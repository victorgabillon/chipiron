# Durable create only Reevaluation patch publication

Reevaluation publishes `pipeline/reevaluation_patch.json` as complete immutable
bytes using a hard link from a unique temporary file in the same directory.
Readers see absence or the complete patch from this writer. A competing final
entry always wins. Patch payloads and successful cursor updates retain the
previous serialization and scientific behavior.

This is the narrowly scoped publication improvement from the operator audit of
2026-10-20, boundaries B17 and B18 and test group T01, stacked on Chipiron PR #75
at `79693ce2dc59c6d6ccb951951c9ba75fa69057f3`. The audit's broader common JSON,
claim, cursor recovery, checkpoint and acknowledgement changes remain separate.

## API and publication sequence

`bootstrap.reevaluation_publication.publish_reevaluation_patch_if_absent(patch,
path) -> bool` is the producer API. The existing
`reevaluation_worker._save_reevaluation_patch_exclusive` delegates to it for both
the legacy and opt-in indexed reader paths. The general
`save_reevaluation_patch` replacement API is unchanged and is not this producer's
publication primitive.

1. Serialize `reevaluation_patch_to_dict(patch)` using the previous `indent=2`,
   `sort_keys=True`, default JSON options, UTF-8 and trailing newline.
2. Create a unique `.<final-name>.<random>.tmp` file with `NamedTemporaryFile` in
   the final directory. No shared temporary pathname is reused.
3. Write and flush all bytes. Read back and compare the complete bytes, then
   validate with `reevaluation_patch_from_dict(json.loads(...))`. Keep the
   original bytes; do not serialize the validated object again.
4. Fsync and close the temporary file before publication. No writer modifies
   this inode after linking.
5. Call `os.link(temporary, final)`. This is the atomic visibility point. It
   cannot replace an existing final entry, including a dangling symlink.
6. Fsync the parent directory, unlink only this attempt's temporary name, and
   fsync the parent again before returning `True`.

`False` means `os.link` reported `FileExistsError` and this attempt cleaned up
its temporary file. The worker preserves `pending_patch_exists` and does not
advance its cursor. A cleanup error raises instead of reporting ordinary loss.
Unsupported hard links never fall back to direct final writes or replacement.

## Filesystem assumptions

The supported durability contract is a local Linux/POSIX filesystem implementing
atomic same-filesystem hard links, regular-file fsync and directory fsync, with
storage honoring successful fsync. Parent directories must remain stable and
the existing run-directory ancestry must already be durable. When the helper
creates missing parent directories, it syncs their entries from the existing
ancestor down before writing the patch. A preexisting directory created without
an ancestor durability barrier elsewhere is outside this helper's guarantee.

The temporary file uses mode 0600 and the final hard link retains that mode;
cooperating workers must have read access, normally by running as the same user.
There is no mode/ACL migration of existing patches. Directory-fsync errors,
including unsupported-operation errors, fail explicitly and prevent a successful
publication return. Non-POSIX platforms and network filesystems are not covered
by this durability claim. See the Linux [link contract](https://man7.org/linux/man-pages/man2/link.2.html)
and [fsync contract](https://man7.org/linux/man-pages/man2/fsync.2.html).

## Failure reporting and restart

An I/O failure raises `ReevaluationPatchPublicationError`, an `OSError` retaining
the original errno and exception cause. It exposes:

| Field | Meaning |
| --- | --- |
| `path` | Attempted final patch path. |
| `link_succeeded` | Whether the link syscall returned successfully. False alone does not prove that it had no effect. |
| `final_state` | Point-in-time `lstat` comparison with this attempt's temporary inode: `published`, `absent`, `other`, or `unknown`. |

These fields diagnose publication, not application or ownership. A consumer can
delete or replace the pathname before or after the observation. `published`
does not certify directory durability; `absent` does not prove the patch was
never published or consumed. No final pathname is unlinked as error rollback.
Errors during cleanup can leave both a complete final patch and a private
temporary hard link. An exception is never a reason to delete the final patch
or to advance the cursor.

The current worker updates its cursor only after a successful `True` return.
On restart it inspects the actual pending path using its existing pending-patch
guards. If a final patch remains, it reports `pending_patch_exists` and keeps
the old cursor. Existing loaders can inspect its identity/content. It does not
automatically advance a cursor from that patch, delete it, or infer checkpoint
application. If no final entry remains, the worker follows its existing cursor
behavior; publication history cannot be reconstructed from absence alone.

Orphaned temporary files are ignored. A subsequent attempt uses a new name and
does not reuse or sweep them, whether their contents are complete or partial.
Manual orphan cleanup, if needed, must be separate from active producer work.

## Before and after crash matrix

“After” describes visibility and successful fsync ordering, not tested physical
power-loss behavior. Every failure below leaves the final patch outside cleanup.

| Boundary | Previous final-path exclusive writer | This writer | Cursor and restart behavior |
| --- | --- | --- | --- |
| Before writing | No patch. | No patch; serialization failure creates none. | No advancement. |
| During temporary/final writing | Partial final JSON visible. | Only private temporary bytes are written. | No advancement; restart ignores orphans. |
| After write before file fsync | Final may be complete but unsynced. | Final absent; private file may be unsynced. | No advancement. |
| After file fsync before link | No equivalent separate visibility point. | Complete synced private inode; final absent. | Crash leaves only an orphan. |
| Existing final or simultaneous winner | Existing path blocks exclusive open. | Atomic link loses with EEXIST; existing entry untouched. | `pending_patch_exists`, no cursor update. |
| Link succeeds before directory fsync | No complete atomic publication point. | Complete final JSON visible; namespace persistence not yet confirmed. | Crash/error may leave final; restart inspects it, never assumes failure means absence. |
| Link takes effect but reports error | Not distinguished. | Observed inode can show `published` even with `link_succeeded=False`. | Error prevents advancement; final is retained. |
| After first directory fsync | No durability barrier. | Complete final entry has a successful durability barrier. | Cursor still old until helper returns and worker saves it. |
| During temporary cleanup | No private alias. | Only owned temporary alias is removed; final remains complete. | Cleanup failure/crash can leave aliases and old cursor. |
| After cleanup fsync and successful return | Closed final write, no explicit durability. | File and directory barriers completed; private name removed. | Existing separate cursor update follows. |
| Before cursor update | Patch may already be consumed. | Same application/cursor race remains, with complete patch visibility. | No automatic cursor reconciliation. |
| After cursor update | No checkpoint receipt. | Still no checkpoint receipt; cursor writer itself is unchanged. | Exactly-once recovery is not provided. |
| During Growth acknowledgement | Unlink can precede checkpoint. | Consumer unchanged; same loss and replay risks. | Next protocol must bind acknowledgement to a durable checkpoint. |

## Validation

`test_morpion_reevaluation_publication.py` uses isolated temporary directories,
fault injection and fresh subprocesses. Pipe barriers release two producers at
the link boundary and let an independent reader open the final path during a
partial temporary write, before linking, and immediately afterward. Other cases
cover interrupted/short/invalid/changed temporary writes, file/directory fsync,
unsupported links, ambiguous link errors, cleanup failures, crashes before/after
link and during cleanup, existing entries, and complete/partial orphan files.

`test_morpion_reevaluation_publication_worker.py` compares every patch and cursor
byte against the previous writer with fixed identities/time, through both
legacy and PR #75 indexed paths. It also checks unchanged cursor bytes after
file-fsync, directory-fsync and cleanup failures, and the pending behavior on
retry. The existing PR #75 failure injection now targets the temporary-file
durability step rather than the removed direct final JSON dump.

Subprocess exits establish process-crash behavior. Injected fsync failures and
ordering checks establish the implementation's barriers; they do not certify
hardware persistence or full host/power-loss recovery.

## Remaining risks and next PR

Growth still applies and deletes a patch before committing its effects to a
checkpoint. Blending a repeated patch can change scientific values again. The
producer's patch/cursor writes remain separate, stale concurrent producers can
still race cursor updates after slot consumption, and the cursor writer still
uses its existing shared temporary path without fsync. No acknowledgements,
applied-ID ledger, recovery cursor images, mailbox lock or generation journal
are added here.

The next durability PR should retain a patch until a durable checkpoint commits
its ID/content identity, recover that checkpoint and receipt together, and make
acknowledgement identity-checked and durable, with the ownership and generation
commit prerequisites described in the audit. This change alone does not make
the asynchronous pipeline transactional or exactly once.
