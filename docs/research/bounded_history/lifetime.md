# Claim and historical payload admission

A verified accepted Dataset/Training descriptor contains its admitted claim UUID.
An exact retry now releases only that matching claim, after validating accepted
rows or every candidate and repairing its projections under Growth ownership.
This handles a crash between acceptance and normal cleanup. A replacement UUID
is preserved. Active readers still hold the existing kernel capsule pin; accepted
source/results remain independent retention roots. Claim removal does not certify
process quiescence and does not itself authorize payload retirement.

| Observation | Claim action | Complete source lifetime |
| --- | --- | --- |
| Surviving worker's fenced cleanup, exact UUID | Release its own claim | Reader pins and other dependencies still apply |
| Exact successful accepted-result retry, exact admitted UUID | Release that completed claim | Accepted source/results stay pinned |
| Successful result with a different current claim | Preserve replacement | Fail closed on its source dependency |
| TERM/KILL or missing PID, no accepted completion | Preserve claim | Retain source regardless of TTL |
| Unknown historical owner/metadata | Preserve | Retirement blocked |

Whole-group shutdown proves no member of that owned group is executing, but the
old claim does not durably bind a retry right and claim attempt to that supervisor
lifetime. It therefore cannot authorize arbitrary revocation. No TTL/PID-based
reaper, lease database or automatic supervisor claim cleanup is introduced.

The original 10k G7 Dataset and G12 Training claims, and the older null-owner G1
claim, remain untouched in their original workspaces. Generated-copy tests must
retain unknown/incomplete claims; completion evidence must never be fabricated to
make a source eligible for deletion.
