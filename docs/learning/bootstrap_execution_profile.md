# Explicit bootstrap execution profile

Scientific bootstrap experiments can opt into `strict_math` at fresh genesis.
Omitting the option preserves the existing profile and serialized configuration.
Selecting a CUDA device alone does not enable this profile.

Use a verified installed wheel and a qualified Torch environment. The first
AMD recipe uses the CUDA namespace, serial candidates, FP32/highest precision,
Torch intra/inter-op 1/1, and `canonical_8_linear_mlp_transformer_v1` (transformer
batch 8, five epochs; existing Adam/model settings). This is a specific tested
runtime recipe, not a guarantee for every GPU or workload.

```sh
PYTHON=/absolute/path/to/qualified-environment/bin/python
WORK=/absolute/path/to/new-workspace
CHIPIRON_CANDIDATE_WORKERS=1 "$PYTHON" -m chipiron.environments.morpion.bootstrap.launcher \
  --work-dir "$WORK" --scientific --scientific-operation prepare-continuous \
  --execution-profile strict_math --torch-intra-op-threads 1 --torch-inter-op-threads 1 \
  --training-device cuda:0 --evaluator-family canonical_8_linear_mlp_transformer_v1

"$PYTHON" -m chipiron.environments.morpion.bootstrap.launcher \
  --work-dir "$WORK" --scientific --scientific-operation run-continuous \
  --cluster-script /absolute/path/to/scripts/launch_morpion_gnome_cluster.sh
```

Set the experiment's resource and scientific budgets explicitly before preparing
it; the example does not authorize a large experiment. The launcher restores the
persisted policy on cold restart. Every worker independently enters the shared
`chipiron.learning.torch_runtime.torch_execution_profile` scope before model
operations. Training, evaluation, accepted replay, continuation, activation,
refresh and subsequent search remain inside their process owner's scope.
Inspection reads metadata without setting Torch flags, configuring threads or
requiring a visible GPU. CPU inference consumers still use the same compatible
Torch build and strict profile as the producing process.

The process entry configures inter-op threads before parallel work. These thread
settings belong to the process lifetime and are not restored. The shared scoped
helper restores the deterministic, MHA fastpath and SDPA flags it owns on normal
exit, failures and interrupts. Strict mode uses `warn_only=False`; unsupported
operations raise. Process owners must join/reap workers before scope exit.

Programmatic entry points require an explicit lifetime owner. Resolve policy
from the experiment's configuration using `args_with_policy`, configure threads
at process startup using `configure_torch_process_threads(intra_op=1,
inter_op=1)`, and retain `bootstrap_execution(args)` around the entire lifetime
of all model consumers. `bootstrap_execution(args, process_start=True)` combines
those steps only when the caller owns a fresh process. Public launcher calls
establish the profile scope but require the process threads already configured;
the CLI owns both startup and the complete invocation. Direct stage calls reject
an unowned strict workspace before claiming or publishing work. Legacy
single-process bootstrap loops do not support this explicit policy.

The fresh declaration is bound to the experiment config identity; observed flags,
thread counts and Torch/HIP/CUDA versions are recorded in genesis and model
provenance. Resume and accepted-result reuse retain the existing full guards.
Legacy parents cannot be relabelled strict. Changing profile, thread counts or
Torch builds requires a separately qualified fresh lineage; a ROCm strict bundle
is not silently migrated into the CPU-only build. Every parameter, including
redundant attention key bias, remains in checkpoints and hashes.

The separate I3 `transformer_cpu_threads` policy remains unchanged. Combining it
with this profile, or changing the 1/1 process recipe, requires separate
qualification and is rejected here. Numerical claims remain those of the
retained qualification: no CPU/GPU weight equality, changed tolerances, or
resolution of indeterminate validation decisions is implied by this integration.
