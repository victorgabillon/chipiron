# Bounded continuous coordination candidate

This implementation is default off and awaits the affected scientific validation
and independent integration gates. It is not a production-readiness or speedup
claim. Historical default, fixed-window and ready-window policies retain their
contexts and cannot be upgraded in place.

Fresh managed runs may select `--continuous-pipeline-policy PATH` after `--` in
`chipiron-bootstrap start`. The exact JSON contents become immutable configuration:

```json
{
  "kind": "bounded_continuous_v1",
  "max_teacher_epoch_lag": 1,
  "max_source_age_cycles": 8,
  "max_unresolved_growth_cycles": 8,
  "max_promotion_growth_cycles": 2,
  "activation": "owned_cold_full",
  "selection": "newest_complete_once",
  "slots": [1, 1, 1, 1, 1, 1]
}
```

This requires `CHIPIRON_SHARED_PREPARATION=1`, supported serial flat/relational
candidates, automatic cold activation, equal Dataset/Training cadence, closed
sharded Growth publications, and each candidate's own accepted weights and Adam.
No numerical profile, device, model architecture or thread setting is inferred or
changed by selecting the policy. Context4 and the policy digest are persisted at
fresh genesis; older contexts refuse adoption.

The six authoritative slots are pending Source, Dataset job, Ready Dataset,
Training cohort, reserved promotion, and pending promotion. Coalescing changes
only pending work and keeps its oldest debt. Admitted jobs freeze their Source,
Dataset, candidate-specific parents, claims, managed launch and original clock.
Training accepts the entire cohort once, including losing candidates. Acceptance
never activates a model. The serial learner services the oldest reserved
comparison against the incumbent on one immutable validation population. Only
Growth can commit the complete synchronous cold refresh and increment model epoch.
Ordinary cycles and administrative/activation publications have distinct counters.

Two operational Dataset roots remain the hard storage bound. Active consumers,
pending and reserved comparisons pin their inputs. Admissions account retained
history, outstanding output/scratch reservations, native preparation and completion
headroom under the existing production caps. Output quotas check writes to private
Dataset and model attempts. A third root waits for completion/comparison/retirement;
scientific receipts, rows and accepted model history are never deleted to make room.

Stop closes admission before native shutdown acknowledgement. Positive native reap
precedes suspension; unresolved jobs keep their pins and original allowances.
Resume cannot renew a job clock or accept live/unknown claim ownership. A damaged
unconsumed disposable input may rebuild only the same admitted Dataset and exact
recipe/path. Corruption after consumption fails and records mandatory evidence.

A fully settled fork imports exact own candidate lineages and starts a new epoch
zero. Occupied slots, debts, claims, staging and cutover prevent fork. After a clean
closed native session, the explicit `chipiron-bootstrap settled-discard --work-dir
PATH` action records dispositions and retires only unaccepted attempts/disposable
inputs. It cannot discard an in-flight cutover or change accepted science. It is
not an automatic side effect of stop or fork. Selective reset and unrelated manual
activation remain unavailable for context4.

Validation includes real two-round own-Adam training and a same-parent numerical
reference, Dataset/Training crossing activation, native producer deaths, cache
recovery/refusal, atomic cohort acceptance, bounded promotion and two-root pressure,
patch/no-patch cold-cutover faults, eighteen slot/owner combinations, original-clock
refusals, and a native stop/resume/settled-fork fixture. Tests are evidence only when
the installed candidate's exact selected IDs and every test phase pass; prepared
fixtures alone do not qualify acceptance.
