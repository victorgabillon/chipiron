# Historical Chipiron 0.2.2 provenance exception

The wheel published as Chipiron 0.2.2 on 2026-02-23 was built from
`6b8951018d108eafec260d374f706e37b2708af9`. Its original trusted-publishing
record identifies `refs/tags/v0.2.2` and
[GitHub Actions run 22299331498](https://github.com/victorgabillon/chipiron/actions/runs/22299331498).

The current `v0.2.2` tag instead targets
`2ccf1c4ad25cdd5edbf87d2a6084675ffd751b8f`. This is a known historical
exception: the current tag must not be used as the source identity of that wheel.
The date and reason for this history/tag divergence have not been established.
GitHub reports no common ancestor between the original build commit and the
current canonicalization line. No tag was moved and no ancestry was fabricated
by the 2026-09-20 reconciliation.

Published artifact:

- Distribution: `chipiron==0.2.2`.
- Wheel: `chipiron-0.2.2-py3-none-any.whl`.
- SHA-256: `254fb322b444bb49c6fc71ab017397ad38cdd8c1a8d66f6cc4f49a94cc71e2bd`.
- [Official PyPI provenance](https://pypi.org/integrity/chipiron/0.2.2/chipiron-0.2.2-py3-none-any.whl/provenance).
- [Original source commit](https://github.com/victorgabillon/chipiron/commit/6b8951018d108eafec260d374f706e37b2708af9).

The forensic audit compared all 261 packaged payload files with an archive of the
original source commit; all matched. The current tag has only 223 of those files.
The externally preserved original archive is:

```text
/home/pompote/oldata/victor/cross_repo_reconciliation_audit_2026-09-20/chipiron/published-022-source.tar.gz
SHA-256: 0372ee73919d8376eefb4c08378ed616c3b21e66efa76b79c46682faffa3f233
```

The adjacent audit directory retains the official metadata, attestation
statement/certificate, GitHub run and commit responses, comparison results and
full report. Reconciliation recovery bundles and the preservation ledger are
under `/home/pompote/oldata/victor/reconciliation_execution_2026-09-20/`.
These external archives are evidence, not runtime dependencies or files to copy
into the package.

Current maintenance acknowledges this exception without rewriting published
history. It does not block current runtime correctness. Any historical tag or
ancestry reconciliation requires separate approval. New releases must identify
reviewed source already reachable from main and retain artifact hashes and
publishing provenance.
