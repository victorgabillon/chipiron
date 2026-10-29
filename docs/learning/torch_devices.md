# Torch device handling and validation limits

`chipiron.learning.torch_runtime.resolve_torch_device` is the shared resolution
boundary for chess supervised learning and Morpion training (including standalone
`train_entity_value`). C2's tensor primitives and existing training wrappers remain
the only update implementations; there is no accelerator-specific trainer.

## Selection and errors

`auto` preserves the existing policy: use `torch.device("cuda")` when
`torch.cuda.is_available()` is true, otherwise `torch.device("cpu")`. This means
the current CUDA-namespace device, not an explicit index zero. It does not probe
or automatically select MPS or XPU. Requests are stripped and lowercased.

Chess keeps its existing automatic selection and configuration API. Morpion keeps
its configured device and defaults, including standalone value training's `cuda`
default. `cpu`, `cuda`, `cuda:N`, `mps`, `mps:0`, `xpu`, and `xpu:N` resolve through
the same helper. Explicit unavailable accelerators and out-of-range indices raise
`TorchDeviceUnavailableError`; they never silently fall back to CPU. Availability
is a Torch build/host check, not an allocation, operator, memory-capacity or
training qualification test. Other Torch-parsable device types retain their
legacy pass-through behavior; Chipiron provides no validation or synchronization
contract for those types.

Both NVIDIA CUDA and AMD ROCm use `cuda` strings and `torch.cuda` APIs in PyTorch.
Diagnostics identify `backend=nvidia_cuda` or `backend=amd_rocm` from
`torch.version.hip`, and expose both `cuda_version` and `hip_version`. The legacy
`cuda_available`, count, name and resolved-device fields keep their meanings:
they describe the CUDA namespace, not necessarily NVIDIA hardware. A CPU-selected
run reports `backend=cpu` even in a ROCm build. New diagnostic fields have defaults
so existing `TorchDeviceInfo` construction remains valid. Existing persisted
bundle metadata and checkpoint schemas are unchanged.

## Synchronization

`synchronize_torch_device_if_needed` lives in `torch_runtime` and remains exported
from `chipiron.learning.supervised` and its `regression` module. CPU is a no-op;
CUDA/ROCm calls `torch.cuda.synchronize(device)`, MPS calls
`torch.mps.synchronize()` without an argument, and XPU calls
`torch.xpu.synchronize(device)`. These signatures were checked in the pinned
Torch **2.14.1** installation. Backend errors propagate. The legacy
`synchronize_cuda` keyword still controls timing synchronization, now for all
three accelerator namespaces. C2 primitives do not introduce implicit waits;
existing timing/checkpoint boundaries own them.

## Support and evidence

| Backend | Runtime handling | Training validation status |
| --- | --- | --- |
| CPU | Explicit or automatic fallback | Required installed-package CI and CPU regression oracles; includes C1 restoration and C2 numerical/RNG parity and dropout resume. |
| NVIDIA CUDA | Automatic or explicit, indexed availability and synchronization | Existing CUDA-only cases require hardware. G1's CPU dispatch tests do not qualify NVIDIA execution. |
| AMD ROCm | CUDA namespace; HIP build diagnostics and CUDA synchronization | G0's actual Radeon 890M / gfx1150 FP32 smoke, reused for G1, is bounded execution evidence. Its after-Adam parameter comparison remains failed; no general training qualification. |
| Apple MPS | Explicit availability/index check and synchronization | Dispatch tested with mocks only; no MPS hardware execution or model/operator qualification. Exact stochastic resume is unsupported by current checkpoint RNG fields. |
| Intel XPU | Explicit availability/index check and synchronization | Dispatch tested with mocks only; no XPU hardware execution or model/operator qualification. Exact stochastic resume is unsupported by current checkpoint RNG fields. |

G0 used Torch 2.14.1+rocm7.14 and HIP 7.14.60850 with the existing FP32 tiny
entity-token fixture and production regression/service paths. Forward, loss and
gradient tolerance checks passed. Fifteen of 2,961 parameter coordinates failed
the after-Adam bound (`atol=3e-5`, `rtol=2e-4`); maximum difference was
`8.5230754e-4`. Tiny key-bias gradients near Adam epsilon (`1e-8`) explain the
amplification, but do not waive the failed gate. Same-device model/optimizer
roundtrip and next-step equality passed for that fixture. G1 retains those
settings and tolerances. This is separate from G2 experiments, benchmark tuning,
the pinned 200k recipe and any long-run or cross-device equivalence claim.

## RNG and checkpoint audit

- Standalone `value_training.py` saves Torch CPU RNG and `cuda_rng`, restores them
  before the next minibatch, and preserves model/AdamW state and step-based sample,
  augmentation and LR schedules. On ROCm the existing `cuda_rng` field uses HIP
  through `torch.cuda`; it is not a new format. MPS/XPU RNG is neither saved nor
  restored, so dropout resume on those backends cannot be exact. Seeding a backend
  at initialization does not repair missing checkpoint RNG state.
- Accepted-round `continuation.py` resets Python, NumPy and Torch seeds each round
  and records Python/NumPy/CPU/CUDA-namespace RNG provenance. Publication records
  also capture CPU/CUDA state. Neither records MPS/XPU RNG. Accepted-round
  continuation is not arbitrary minibatch replay, regardless of backend.
- CUDA RNG arrays include all visible devices in their existing order. Resuming
  assumes compatible backend/build, device visibility/order, kernels and settings.
  The unindexed `auto` request is not a persisted hardware identity. Keep an
  explicit device and the same environment for reproducible work; changing
  hardware under an unchanged request is not validated by the saved identity.
- Chess C1 reattaches restored SGD momentum and StepLR to the live model. Legacy
  optimizer/scheduler pickle loading has no portable `map_location` contract and
  does not save the full script RNG/batch/decision history. GPU-to-CPU or
  CUDA-to-ROCm migration and whole-script replay are not promised.
- CPU model state/bundle loading and optimizer state restoration do not establish
  cross-backend numerical or RNG equivalence. The G0 same-device smoke has no
  stochastic-dropout resume qualification. CPU dropout tests remain the evidence
  for that stronger contract. G1 changes no seeds, deterministic flags, precision,
  optimizer defaults, checkpoint fields or scientific settings.

References: [PyTorch HIP semantics](https://docs.pytorch.org/docs/2.14/notes/hip.html),
[MPS synchronization](https://docs.pytorch.org/docs/2.14/generated/torch.mps.synchronize.html),
[XPU synchronization](https://docs.pytorch.org/docs/2.14/generated/torch.xpu.synchronize.html).
