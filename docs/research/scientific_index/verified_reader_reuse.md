# Verified reader canonicality reuse (I6.2 consumer contract)

This candidate extends canonical validation evidence to the bounded reader. It
requires integration's independent contract review before production rollout.
Dataset and Reevaluation use this shared reader; neither caller supplies a trust
flag or independently decides which immutable checks may be skipped.

The reader retains its ordinary generation lease until close. During constructor
initialization, `verified_reader_validation` acquires its own exception-safe pin:
run-state admission, publication ancestry, the committed seal and its complete
inventory are checked by the existing publication machinery. The selected path
must equal that admitted capsule's scientific descriptor. A retained, older
RESERVED source is valid even after Growth has advanced the run-state head.
Neither an unmanaged copy nor a file elsewhere in a capsule gains authority.

Only matching canonical evidence may avoid replay/re-encoding of immutable state
references. It must bind the exact scientific identity, producer, record-shard
descriptors and current validator implementation/dependencies. Missing or stale
evidence follows full canonical validation. Malformed evidence or identity
substitution fails; a nonmatching shard descriptor grants no reuse. Source bytes
are still freshly hashed against both the publication and shard descriptors,
including after building. The constructor scope resets on success and failure;
retaining a reader does not lend its proof to other sources or contexts.

Parsing, immutable row structure, creation membership, current values, certainty,
reciprocal topology, global ordering and node IDs still follow the complete
bounded SQLite construction/validation path. The cache key/schema, completion
rename, integrity check, payload digest, bounded page cache, private temporary
files and pin-based retention remain unchanged. No state, node or index is kept
alive between jobs by this change. Publisher private validation scopes remain
source-bound and keep their existing behavior.

Historical proofs normally become stale because validator identity includes the
reader and evidence source. They are not migrated. Performance qualification of
matching proofs requires newly published fixtures, fully validated under this
implementation, with separately checked identical scientific payloads. Timings
of such reader opens do not establish whole-bootstrap or 200k performance.
