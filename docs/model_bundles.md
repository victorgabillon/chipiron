# Model Bundles

Chipiron has historically treated neural-network artifacts as loose file paths:

- a weights file path
- an architecture YAML path
- an implicit `chipiron_nn.yaml` sidecar discovered from the weights folder

This PR introduces a first-class model bundle abstraction so we can reason about the artifact boundary correctly. A model bundle is a folder that contains everything needed to instantiate a model:

- `architecture.yaml`
- `chipiron_nn.yaml`
- one or more weights files, typically `.pt`

The bundle root identifies the folder. The weights selection identifies which file inside that folder should be used when loading the model.

Supported bundle backends in PR1:

- `hf://<namespace>/<repo>/<bundle>@<revision>`
- `package://<path/inside/chipiron>`
- local filesystem folders, absolute or relative

Current PR1 note:

- for local and `package://` bundles, omitting `weights_file` can auto-select a unique `.pt`
- for `hf://` bundles, `weights_file` is currently required because the resolver does not enumerate remote bundle contents

Examples:

```python
from chipiron.models.model_bundle import ModelBundleRef, resolve_model_bundle

hf_ref = ModelBundleRef(
    uri="hf://VictorGabillon/chipiron/prelu_no_bug@main",
    weights_file=(
        "param_multi_layer_perceptron_772_20_1_"
        "parametric_relu_hyperbolic_tangent_player_to_move.pt"
    ),
)
hf_bundle = resolve_model_bundle(hf_ref)

package_ref = ModelBundleRef(
    uri="package://data/players/board_evaluators/nn_pytorch/prelu_no_bug",
    weights_file=(
        "param_multi_layer_perceptron_772_20_1_"
        "parametric_relu_hyperbolic_tangent_player_to_move.pt"
    ),
)
package_bundle = resolve_model_bundle(package_ref)
```

PR1 is intentionally architecture-first. Evaluator creation still uses the legacy path-based plumbing today, and that migration is deferred to PR2.
