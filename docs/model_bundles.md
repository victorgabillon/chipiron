# Model Bundles

Chipiron now treats a neural-network artifact as a model bundle rather than as loose file paths. A model bundle is a folder that contains everything needed to instantiate a model:

- `architecture.yaml`
- `chipiron_nn.yaml`
- one or more weights files, typically `.pt`

The public config points to the bundle folder and, when needed, selects which weights file inside that folder should be used.

Supported bundle backends:

- `hf://<namespace>/<repo>/<bundle>@<revision>`
- `package://<path/inside/chipiron>`
- local filesystem folders, absolute or relative

Neural-network evaluator config is now bundle-first only. The old public fields
`model_weights_file_name` and `nn_architecture_args_path_to_yaml_file` have been removed.

Config example:

```yaml
board_evaluator:
  type: neural_network
  neural_nets_model_and_architecture:
    model_bundle:
      uri: "hf://VictorGabillon/chipiron/prelu_no_bug@main"
      weights_file: "param_multi_layer_perceptron_772_20_1_parametric_relu_hyperbolic_tangent_player_to_move.pt"
```

Runtime flow:

1. Config parsing produces a `ModelBundleRef`.
2. `resolve_model_bundle(...)` resolves that ref to concrete local files.
3. Evaluator construction loads `architecture.yaml`, `chipiron_nn.yaml`, and the selected weights file from the resolved bundle.

Notes:

- For local and `package://` bundles, omitting `weights_file` can auto-select a unique `.pt`.
- For `hf://` bundles, `weights_file` is currently required because the resolver does not enumerate remote bundle contents.

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
