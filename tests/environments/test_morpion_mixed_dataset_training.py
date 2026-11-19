"""Mixed source interpretation, independent target math and unchanged Adam updates.

Unit fixtures stand in for C3's authenticated return; they do not enable a writer
or claim that supplied model identities prove Growth writes. Real C3/C6 capsule
integration remains a separate required test.
"""

import copy
import hashlib
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import torch

from chipiron.environments.morpion.bootstrap.bootstrap_args import MorpionBootstrapArgs
from chipiron.environments.morpion.bootstrap.cycle_dataset import (
    extract_rows_from_training_snapshot,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    artifact_digest,
    file_digest,
)
from chipiron.environments.morpion.bootstrap.prediction_lineage import PredictionOrigin
from chipiron.environments.morpion.bootstrap.sharded_training_v2 import (
    ScientificGeneration,
    load_scientific_training_generation,
    save_scientific_training_generation,
)
from chipiron.environments.morpion.bootstrap.sharded_training_v2_contract import (
    CLOSED_KIND,
    LINEAGE_CONTRACT,
    MIXED_SOURCE_CONTRACT,
)
from chipiron.environments.morpion.learning import (
    load_morpion_supervised_rows,
    save_morpion_supervised_rows_streaming,
)
from chipiron.environments.morpion.learning.mixed_source import (
    SOURCE_POLICY,
    MixedDatasetSource,
    validate_source_policy,
)
from chipiron.environments.morpion.learning.target_confidence import (
    DatasetConfidenceContext,
)
from chipiron.environments.morpion.learning.target_policy import (
    TargetSource,
    training_source_binding,
    training_target_metadata,
    validate_model_target_provenance,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.bundle import (
    load_morpion_model_bundle,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training import (
    MorpionTrainingArgs,
    train_morpion_regressor,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training.continuation import (
    MorpionTrainingContinuation,
)
from tests.environments.test_morpion_sharded_training_v2 import (
    PRODUCER,
    advance,
    nodes_fixture,
)

if TYPE_CHECKING:
    from chipiron.environments.morpion.bootstrap.pv_family_targets import (
        PvFamilyTargetPolicy,
    )


def sha(label: str) -> str:
    """Make visibly synthetic immutable identities for interpretation-only tests."""
    return hashlib.sha256(label.encode()).hexdigest()


def mixed_source(source: ScientificGeneration) -> MixedDatasetSource:
    """Represent a C3 return with A/B writes; never install this fixture in production."""
    origins = (
        PredictionOrigin(
            "genesis",
            sha("A"),
            sha("producer"),
            sha("config"),
            sha("numeric"),
            sha("genesis write"),
        ),
        PredictionOrigin(
            "accepted_patch",
            sha("B"),
            sha("producer"),
            sha("config"),
            sha("numeric"),
            sha("B patch write"),
        ),
    )
    ids = [
        node
        for node, contract in source.value_contracts.items()
        if contract["certainties"]["direct"] == "ESTIMATE"
    ]
    current = {node: origins[index % 2] for index, node in enumerate(ids)}
    assert set(current.values()) == set(origins)
    provenance = {
        **source.provenance,
        "source_format_kind": CLOSED_KIND,
        "storage_layout_version": 1,
        "scientific_contract": MIXED_SOURCE_CONTRACT,
        "source_policy": SOURCE_POLICY,
        "capsule_id": "c" * 32,
        "capsule_descriptor_sha256": sha("capsule"),
        "checkpoint_sha256": sha("checkpoint"),
        "generation_descriptor_sha256": sha("scientific descriptor"),
        "prediction_lineage": {
            "contract": LINEAGE_CONTRACT,
            "path": "prediction_lineage.json",
            "sha256": sha("lineage"),
            "evidence_path": "prediction_evidence.json",
            "evidence_sha256": sha("writes"),
        },
        "prediction_origins": [asdict(origin) for origin in origins],
        "direct_origin_counts": [
            list(current.values()).count(origin) for origin in origins
        ],
    }
    return MixedDatasetSource(
        provenance,
        current,
        origins,
        {node.node_id: 1 for node in source.snapshot.nodes},
    )


def scientific_fixture(tmp_path: Path) -> ScientificGeneration:
    """Reuse the independent native DAG with differing direct/tree/effective slots."""
    nodes = nodes_fixture()
    advance(nodes)
    path, _ = save_scientific_training_generation(
        nodes=nodes,
        root_node_id="0",
        output_dir=tmp_path / "source",
        generation=1,
        producer=PRODUCER,
    )
    return load_scientific_training_generation(path)


@pytest.mark.parametrize("target", ["effective", "tree", "direct"])
@pytest.mark.parametrize("family", ["none", "pv_blend_mean_prediction"])
def test_mixed_targets_preserve_native_slots_certificates_and_dataset_recovery(
    tmp_path: Path,
    target: TargetSource,
    family: PvFamilyTargetPolicy,
) -> None:
    """All values and certainty agree with the existing single-source reference."""
    source = scientific_fixture(tmp_path)
    mixed = mixed_source(source)
    args = replace(
        MorpionBootstrapArgs.new_scientific(tmp_path, target_source=target),
        dataset_family_target_policy=family,
    )
    historical = extract_rows_from_training_snapshot(
        args=args,
        snapshot=source.snapshot,
        generation=1,
        value_contracts=source.value_contracts,
        source_provenance=source.provenance,
    )
    actual = extract_rows_from_training_snapshot(
        args=args,
        snapshot=source.snapshot,
        generation=1,
        value_contracts=source.value_contracts,
        source_provenance=mixed.provenance,
        mixed_source=mixed,
    )
    nodes = {node.node_id: node for node in source.snapshot.nodes}
    assert len(actual.rows) == len(historical.rows)
    kinds = set()
    for row, reference in zip(actual.rows, historical.rows, strict=True):
        assert (
            row.node_id,
            row.target_value,
            row.direct_value,
            row.is_exact,
            row.is_terminal,
        ) == (
            reference.node_id,
            reference.target_value,
            reference.direct_value,
            reference.is_exact,
            reference.is_terminal,
        )
        confidence = row.metadata["target_confidence"]
        assert {
            k: v
            for k, v in confidence.items()
            if k not in {"raw_target_origin", "direct_prediction_origin"}
        } == reference.metadata["target_confidence"]
        kind = confidence["raw_target_origin"]["kind"]
        kinds.add(kind)
        if confidence["raw_target_certainty"] in {"FORCED", "TERMINAL"}:
            assert kind == "certified_value"
        elif target == "direct" or (
            target == "effective"
            and nodes[row.node_id].effective_value_source == "direct_self"
        ):
            assert confidence["raw_target_origin"] == {
                "kind": "direct_prediction",
                "origin_index": mixed.origins.index(mixed.current[row.node_id]),
            }
        else:
            assert confidence["raw_target_origin"] == {"kind": "derived_search_value"}
    if target != "tree":
        assert "certified_value" in kinds
    if target == "tree":
        assert "direct_prediction" not in kinds
    assert (
        actual.metadata["dataset_extraction_identity"]
        != historical.metadata["dataset_extraction_identity"]
    )
    path = tmp_path / "mixed.jsonl"
    save_morpion_supervised_rows_streaming(
        rows=actual.rows, metadata=actual.metadata, path=path
    )
    assert load_morpion_supervised_rows(path) == actual
    target_metadata = training_target_metadata(actual.metadata)
    assert target_metadata["scientific_source_provenance"] == mixed.provenance
    validate_model_target_provenance(
        f"{target}_value" if family == "none" else "family_adjusted_value",
        target_metadata,
    )


@pytest.mark.parametrize(
    "field",
    [
        "prediction_lineage",
        "prediction_origins",
        "direct_origin_counts",
        "checkpoint_sha256",
        "capsule_descriptor_sha256",
        "source_policy",
    ],
)
def test_missing_mixed_identity_refuses(tmp_path: Path, field: str) -> None:
    """A well-formed scalar does not fill missing write/source identity."""
    mixed = mixed_source(scientific_fixture(tmp_path))
    provenance = dict(mixed.provenance)
    del provenance[field]
    with pytest.raises(ValueError, match="B4"):
        validate_source_policy(provenance)


def test_unqualified_origins_and_false_homogeneous_claim_refuse(tmp_path: Path) -> None:
    """Serialized identities do not authorize extraction or legacy target fallback."""
    source = scientific_fixture(tmp_path)
    mixed = mixed_source(source)
    with pytest.raises(ValueError, match="authenticated Growth"):
        DatasetConfidenceContext(source.value_contracts.__getitem__, mixed.provenance)
    with pytest.raises(ValueError, match="homogeneous"):
        validate_source_policy({
            **mixed.provenance,
            "homogeneous_evaluator_sha256": sha("B"),
        })
    with pytest.raises(ValueError, match="explicit"):
        training_target_metadata({"scientific_source_provenance": mixed.provenance})
    with pytest.raises(ValueError, match="explicit"):
        validate_model_target_provenance(
            "backup_value", {"scientific_source_provenance": mixed.provenance}
        )
    wrong = replace(mixed, current={})
    context = DatasetConfidenceContext(
        source.value_contracts.__getitem__, mixed.provenance, wrong
    )
    node = next(node for node in source.snapshot.nodes if node.node_id in mixed.current)
    assert node.direct_value_scalar is not None
    with pytest.raises(ValueError, match="coverage"):
        context.selected(node, node.direct_value_scalar, "direct_value", explicit=True)


def _assert_tensors_equal(actual: Any, expected: Any) -> None:
    """Compare every Adam/model tensor exactly, not just aggregate losses."""
    if isinstance(actual, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_tensors_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            _assert_tensors_equal(left, right)
    else:
        assert actual == expected


def test_training_accepts_mixed_sources_and_continues_exact_adam(
    tmp_path: Path,
) -> None:
    """Changing truthful teacher provenance never changes weights, order or Adam."""
    source = scientific_fixture(tmp_path)
    mixed = mixed_source(source)
    settings = MorpionBootstrapArgs.new_scientific(tmp_path)
    outputs: dict[str, dict[str, Path]] = {}
    metadata: dict[tuple[str, str], Any] = {}
    for name, context in (("historical", None), ("mixed", mixed)):
        rows = extract_rows_from_training_snapshot(
            args=settings,
            snapshot=source.snapshot,
            generation=1,
            value_contracts=source.value_contracts,
            source_provenance=source.provenance
            if context is None
            else context.provenance,
            mixed_source=context,
        )
        path = tmp_path / f"{name}.jsonl"
        save_morpion_supervised_rows_streaming(
            rows=rows.rows, metadata=rows.metadata, path=path
        )
        outputs[name] = {}
        training = MorpionTrainingArgs(
            path,
            tmp_path / f"{name}-first",
            model_kind="mlp",
            hidden_sizes=(8,),
            num_epochs=2,
            batch_size=2,
            device="cpu",
            stable_validation_split=True,
            validation_fraction=0.5,
            validation_seed=17,
            continuation=MorpionTrainingContinuation("candidate", 7),
        )
        for round_name in ("first", "continued"):
            if round_name == "continued":
                if context is not None:
                    # A newly accepted frozen source may carry another origin mix;
                    # continuation constrains output semantics, not teacher age.
                    next_provenance = copy.deepcopy(dict(context.provenance))
                    next_provenance["capsule_id"] = "d" * 32
                    next_provenance["capsule_descriptor_sha256"] = sha("later capsule")
                    next_provenance["prediction_lineage"]["sha256"] = sha(
                        "later lineage"
                    )
                    next_provenance["prediction_lineage"]["evidence_sha256"] = sha(
                        "later writes"
                    )
                    next_origins = (
                        context.origins[0],
                        replace(
                            context.origins[1],
                            evaluator_sha256=sha("C"),
                            evidence_sha256=sha("C equal-score write"),
                        ),
                    )
                    next_provenance["prediction_origins"] = [
                        asdict(origin) for origin in next_origins
                    ]
                    context = replace(
                        context,
                        provenance=next_provenance,
                        origins=next_origins,
                        current={
                            node: next_origins[context.origins.index(origin)]
                            for node, origin in context.current.items()
                        },
                    )
                    next_rows = extract_rows_from_training_snapshot(
                        args=settings,
                        snapshot=source.snapshot,
                        generation=1,
                        value_contracts=source.value_contracts,
                        source_provenance=context.provenance,
                        mixed_source=context,
                    )
                    next_path = tmp_path / "mixed-next.jsonl"
                    save_morpion_supervised_rows_streaming(
                        rows=next_rows.rows, metadata=next_rows.metadata, path=next_path
                    )
                    training = replace(training, dataset_file=next_path)
                parent = Path(training.output_dir)
                training = replace(
                    training,
                    output_dir=tmp_path / f"{name}-continued",
                    continuation=MorpionTrainingContinuation(
                        "candidate",
                        11,
                        str(parent),
                        artifact_digest(parent),
                        file_digest(parent / "param.pt"),
                    ),
                )
            train_morpion_regressor(training)
            outputs[name][round_name] = Path(training.output_dir)
            _, _, manifest = load_morpion_model_bundle(Path(training.output_dir))
            metadata[name, round_name] = manifest.metadata
    for round_name in ("first", "continued"):
        left, _, _ = load_morpion_model_bundle(outputs["mixed"][round_name])
        right, _, _ = load_morpion_model_bundle(outputs["historical"][round_name])
        _assert_tensors_equal(left.state_dict(), right.state_dict())
        _assert_tensors_equal(
            torch.load(
                outputs["mixed"][round_name] / "optimizer.pt", weights_only=True
            ),
            torch.load(
                outputs["historical"][round_name] / "optimizer.pt", weights_only=True
            ),
        )
        actual, expected = (
            metadata["mixed", round_name],
            metadata["historical", round_name],
        )
        for key in (
            "rng_before_training",
            "rng_after_training",
            "optimizer_steps",
            "starting_optimizer_step_count",
            "final_optimizer_step_count",
        ):
            assert actual["continuation"][key] == expected["continuation"][key]
        assert actual["continuation"][
            "dataset_source_binding"
        ] == training_source_binding(actual)
    continued = metadata["mixed", "continued"]
    assert training_source_binding(continued) != training_source_binding(
        metadata["mixed", "first"]
    )
    assert continued["continuation"]["optimizer_state_loaded"]
    assert continued["continuation"][
        "parent_dataset_source_binding"
    ] == training_source_binding(metadata["mixed", "first"])
    changed = copy.deepcopy(continued)
    changed["continuation"]["dataset_source_binding"]["scientific_source_sha256"] = sha(
        "different source"
    )
    with pytest.raises(ValueError, match="source binding"):
        validate_model_target_provenance("effective_value", changed)
