# Explicit Torch execution profiles

The default is `existing`: Chipiron leaves the caller's Torch settings alone.
There is no environment-variable opt-in, device change or new training loop.
Use `strict_math` explicitly at the **process ownership boundary**, before model
execution or worker threads start, and keep it active through training, held-out
evaluation, checkpoint/resume, inference and accepted-result reuse:

```python
from chipiron.learning.torch_runtime import torch_execution_profile

with torch_execution_profile("strict_math") as runtime:
    print(runtime.to_metadata())
    # Invoke the existing Chipiron training/evaluation workflow here.
    # Join all model-execution threads before leaving this scope.
```

This sets `torch.use_deterministic_algorithms(True, warn_only=False)`, disables
the MHA fastpath and selects `sdpa_kernel(SDPBackend.MATH)`. Unsupported
deterministic operations raise. There is no retry with weaker settings.
Device selection, model definitions, optimizer parameters, precision and CPU
thread counts are unchanged. The helper restores the preceding deterministic
(including warning mode), MHA and SDPA flags on success and failure.

These are process-global settings. The helper rejects competing scope owners
in different threads; that lock cannot isolate arbitrary Torch execution or
unmanaged flag changes. Worker threads may execute inside an already established
scope. Same-thread nested scopes restore their parent; nested `existing` inherits
the effective strict profile. Spawned interpreters need their own explicit scope.
The existing spawned CPU candidate executor changes thread counts, so it rejects
`strict_math` before launching work. Serial candidate execution is supported.
Do not enter a scope per forward call or change it while another model runs.

`torch_execution_profile_info()` observes requested/effective identity, effective
flags, Torch/Git/HIP/CUDA versions, intra/inter-op threads, default dtype and
FP32 matmul precision without initializing an accelerator. Existing device
metadata records resolved backend and device identity separately.
`torch_execution_profile_metadata()` is empty for `existing`, preserving default
manifest/recipe identities; strict mode records a versioned
`torch_execution_profile` sidecar field. It rejects settings changed inside the
strict scope. All model parameters (including attention key biases) and optimizer
state remain in checkpoints and artifact hashes.

Morpion bundle saves and final training-metadata updates record the active
profile. Loading a strict bundle requires a matching scope; a runtime forward
guard prevents the loaded model from escaping that scope. Historical bundles can
still be loaded for explicit diagnostic inference under a strict scope, but
continuation requires matching provenance: a missing field means `existing`.
Changing profile, backend version, observed threads or precision requires a new
lineage, not a silent continuation. Malformed profile evidence fails closed.

Accepted Training admissions include the explicit profile observation, and
accepted retries must match that configuration and bundle provenance. Keep the
scope active around the whole bootstrap Training operation, including publication
and retry. A default run cannot adopt a strict accepted result or vice versa.

The G3 numerical qualification used FP32, highest matmul precision, one intra-op
and one inter-op thread on Torch 2.14.1+rocm7.14 / HIP 7.14.60850, Radeon 890M.
The helper **does not set those thread counts**. I3 owns CPU thread budgeting and
thread policy; combining its changed thread counts with strict math requires
separate qualification. Recorded settings are observations, not a claim that
every combination, model, batch size or backend is qualified.

G3's bounded functional checks do not establish CPU/GPU bitwise equivalence:
strict all-parameter comparisons still fail for redundant key biases, and the
tiny seed-2718 checkpoint choice remains indeterminate under the original margin
criterion. This numerical opt-in makes no general speedup promise. Raw G0/G2/G3
and installed-wheel G4 GPU evidence belongs outside Git.

The behavior follows the documented PyTorch APIs:
[strict algorithms](https://docs.pytorch.org/docs/2.14/generated/torch.use_deterministic_algorithms.html),
[MHA fastpath](https://docs.pytorch.org/docs/2.14/backends.html#torch.backends.mha.set_fastpath_enabled),
[SDPA math selection and restoration](https://docs.pytorch.org/docs/2.14/generated/torch.nn.attention.sdpa_kernel.html).

Scientific bootstrap's explicit config/CLI and process lifetime owner are
covered in [Bootstrap execution profile](bootstrap_execution_profile.md).
