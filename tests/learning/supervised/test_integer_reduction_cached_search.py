"""Real integer-reduction training, admitted replay, and bounded Anemone search.

Keep this explicit composition as one witness: the direct-update reference checks
the shared epoch, resolved replay must avoid training, and real search must retain
exact terminal values. The local admission context exercises run_attempt ordering;
it does not claim durable admission or scientific publication.
"""

import copy
import logging
import math
import random
from contextlib import contextmanager
from typing import TYPE_CHECKING, Literal, cast

import torch
from anemone import TreeAndValuePlayerArgs
from anemone.node_evaluation.direct.protocols import MasterStateValueEvaluator
from anemone.node_evaluation.tree.single_agent.factory import NodeMaxEvaluationFactory
from anemone.node_selector.composed.args import ComposedNodeSelectorArgs
from anemone.node_selector.node_selector_types import NodeSelectorType
from anemone.node_selector.opening_instructions import OpeningType
from anemone.node_selector.priority_check.noop_args import NoPriorityCheckArgs
from anemone.node_selector.uniform.uniform import UniformArgs
from anemone.progress_monitor.progress_monitor import (
    StoppingCriterionTypes,
    TreeBranchLimitArgs,
)
from anemone.recommender_rule.recommender_rule import AlmostEqualLogistic
from anemone.tree_and_value_branch_selector import TreeAndValueBranchSelector
from coral.neural_networks.models.multi_layer_perceptron import (
    MultiLayerPerceptron,
    MultiLayerPerceptronArgs,
)
from coral.neural_networks.nn_model_type import ActivationFunctionType, NNModelType
from valanga.evaluations import Certainty, Value

from chipiron.bootstrap.attempts import Pending, Resolved, run_attempt
from chipiron.environments.integer_reduction.players.evaluators.integer_reduction_state_evaluator import (
    IntegerReductionOverEventDetector,
    IntegerReductionStateEvaluator,
)
from chipiron.environments.integer_reduction.types import (
    IntegerReductionDynamics,
    IntegerReductionState,
)
from chipiron.learning.supervised import (
    TensorSupervisedBatch,
    evaluate_regression_batch,
    train_regression_batch,
)
from chipiron.learning.supervised.cached import (
    StreamingEpochStats,
    evaluate_cached_metrics,
    train_cached_epoch,
)
from chipiron.learning.torch_runtime import torch_execution_profile
from chipiron.players.move_selector.factory import create_tree_and_value_move_selector

if TYPE_CHECKING:
    from collections.abc import Iterator

    from valanga import State


class _LearnedEvaluator(MasterStateValueEvaluator):
    """Test-local feature adapter; terminals retain the game's exact evaluator."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.over = IntegerReductionOverEventDetector()
        self.nonterminal_scores: list[float] = []

    def evaluate(self, state: State) -> Value:
        integer = cast("IntegerReductionState", state)
        if integer.is_game_over():
            return IntegerReductionStateEvaluator().evaluate(integer)
        with torch.no_grad():
            inputs = torch.tensor([[integer.value, integer.steps]], dtype=torch.float32)
            score = float(self.model(inputs).item())
        self.nonterminal_scores.append(score)
        return Value(score=score, certainty=Certainty.ESTIMATE, over_event=None)


def test_integer_reduction_cached_search_attempt() -> None:
    """One partial-batch epoch composes with real rules, admission and search."""
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch_execution_profile("strict_math"), torch.random.fork_rng(devices=[]):
            _exercise_cached_search()
    finally:
        torch.set_num_threads(old_threads)


def _exercise_cached_search() -> None:
    dynamics = IntegerReductionDynamics()
    states = [IntegerReductionState(8)]
    for _ in range(3):
        state = states[-1]
        action = dynamics.action_from_name(state, "half")
        assert action in dynamics.legal_actions(state).get_all()
        states.append(dynamics.step(state, action).next_state)
    assert [(s.value, s.steps) for s in states] == [(8, 0), (4, 1), (2, 2), (1, 3)]
    assert IntegerReductionState(4, 1).tag != IntegerReductionState(4, 2).tag
    values = [IntegerReductionStateEvaluator().evaluate(s) for s in states]
    assert [value.score for value in values] == [0.0, -1.0, -2.0, -3.0]
    assert values[-1].certainty is Certainty.TERMINAL
    inputs = torch.tensor([[s.value, s.steps] for s in states], dtype=torch.float32)
    targets = torch.tensor([[v.score] for v in values], dtype=torch.float32)
    batches: list[tuple[int, ...]] = []

    def build(indices: tuple[int, ...]) -> TensorSupervisedBatch:
        batches.append(indices)
        selection = torch.tensor(indices, dtype=torch.long)
        return TensorSupervisedBatch(
            input_tensor=inputs.index_select(0, selection),
            target_tensor=targets.index_select(0, selection),
            is_batch=True,
        )

    torch.default_generator.manual_seed(72)
    model = MultiLayerPerceptron(
        MultiLayerPerceptronArgs(
            type=NNModelType.MULTI_LAYER_PERCEPTRON,
            number_neurons_per_layer=[2, 4, 1],
            list_of_activation_functions=[
                ActivationFunctionType.RELU,
                ActivationFunctionType.RELU,
            ],
        )
    )
    # Coral constructs all layers; omit its final activation for negative step scores.
    assert isinstance(model.model, torch.nn.Sequential)
    model.model = model.model[:-1]
    reference = copy.deepcopy(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    reference_optimizer = torch.optim.Adam(reference.parameters(), lr=0.003)
    order = [0, 1, 2]
    random.Random(17).shuffle(order)
    expected_batches = [tuple(order[:2]), tuple(order[2:])]
    initial_rng = torch.get_rng_state()
    reference.train()
    reference_sums = [
        train_regression_batch(
            model=reference,
            optimizer=reference_optimizer,
            criterion=torch.nn.MSELoss(),
            batch=build(indices),
            device=torch.device("cpu"),
        )
        for indices in expected_batches
    ]
    reference_rng = torch.get_rng_state()
    batches.clear()
    torch.set_rng_state(initial_rng)
    events: list[str] = []

    @contextmanager
    def admit(
        replay: StreamingEpochStats | None = None,
    ) -> Iterator[Pending[int] | Resolved[StreamingEpochStats]]:
        events.append("enter")
        try:
            yield Pending(4) if replay is None else Resolved(replay)
        finally:
            events.append("exit")

    def compute(row_count: int) -> StreamingEpochStats:
        events.append("compute")
        return train_cached_epoch(
            model=model,
            optimizer=optimizer,
            criterion=torch.nn.MSELoss(),
            batch_size=2,
            validation_fraction=0.0,
            validation_indices=frozenset({3}),
            shuffle=True,
            validation_seed=17,
            num_epochs=1,
            logger=logging.getLogger(__name__),
            explicit_split_policy="integer_fixture_holdout",
            row_count=row_count,
            batch_builder=build,
            cache_mode="integer_tensor_cache",
            row_chunk_size=2,
            max_rows=None,
            epoch_index=0,
            progress_callback=None,
            device=torch.device("cpu"),
        )

    def publish(attempt: int, product: StreamingEpochStats) -> StreamingEpochStats:
        assert attempt == 4
        events.append("publish")
        return product

    def accept(reference: StreamingEpochStats) -> StreamingEpochStats:
        events.append("accept")
        return reference

    result = run_attempt(
        admission=admit(), compute=compute, publish=publish, accept=accept
    )
    assert events == ["enter", "compute", "publish", "accept", "exit"]
    assert batches == expected_batches
    assert (result.train_count, result.validation_count, result.batch_count) == (
        3,
        1,
        2,
    )
    assert result.loss == sum(s.squared_error_sum for s in reference_sums) / 3
    assert torch.equal(torch.get_rng_state(), reference_rng)
    for actual, expected in zip(
        model.parameters(), reference.parameters(), strict=True
    ):
        assert torch.equal(actual, expected)
    actual_state, expected_state = (
        optimizer.state_dict(),
        reference_optimizer.state_dict(),
    )
    assert actual_state["param_groups"] == expected_state["param_groups"]
    for key, entry in actual_state["state"].items():
        for name, value in entry.items():
            assert torch.equal(value, expected_state["state"][key][name])
            if name == "step":
                assert value.item() == 2

    splits: tuple[tuple[Literal["train", "validation"], list[tuple[int, ...]]], ...] = (
        ("train", [(0, 1), (2,)]),
        ("validation", [(3,)]),
    )
    for split, expected_indices in splits:
        batches.clear()
        before_rng = torch.get_rng_state()
        before_gradients = [
            p.grad.clone() for p in model.parameters() if p.grad is not None
        ]
        metrics = evaluate_cached_metrics(
            model=model,
            batch_size=2,
            validation_fraction=0.0,
            validation_indices=frozenset({3}),
            row_count=4,
            batch_builder=build,
            max_rows=None,
            split=split,
            device=torch.device("cpu"),
        )
        assert batches == expected_indices
        assert not model.training
        with torch.no_grad():
            sums = [
                evaluate_regression_batch(
                    model=model, batch=build(indices), device=torch.device("cpu")
                )
                for indices in expected_indices
            ]
        count = sum(s.target_count for s in sums)
        assert metrics.sample_count == metrics.target_count == count
        assert metrics.loss == sum(s.squared_error_sum for s in sums) / count
        assert metrics.mae == sum(s.absolute_error_sum for s in sums) / count
        assert torch.equal(torch.get_rng_state(), before_rng)
        for grad, parameter in zip(before_gradients, model.parameters(), strict=True):
            assert parameter.grad is not None
            assert torch.equal(grad, parameter.grad)

    events.clear()
    before_batches = list(batches)
    assert (
        run_attempt(
            admission=admit(result), compute=compute, publish=publish, accept=accept
        )
        is result
    )
    assert events == ["enter", "exit"]
    assert batches == before_batches
    for key, entry in optimizer.state_dict()["state"].items():
        for name, value in entry.items():
            assert torch.equal(value, expected_state["state"][key][name])

    evaluator = _LearnedEvaluator(model)
    selector = create_tree_and_value_move_selector(
        args=TreeAndValuePlayerArgs(
            node_selector=ComposedNodeSelectorArgs(
                type=NodeSelectorType.COMPOSED,
                priority=NoPriorityCheckArgs(type=NodeSelectorType.PRIORITY_NOOP),
                base=UniformArgs(type=NodeSelectorType.UNIFORM),
            ),
            opening_type=OpeningType.ALL_CHILDREN,
            stopping_criterion=TreeBranchLimitArgs(
                type=StoppingCriterionTypes.TREE_BRANCH_LIMIT, tree_branch_limit=16
            ),
            recommender_rule=AlmostEqualLogistic(
                type="almost_equal_logistic", temperature=1.0
            ),
        ),
        state_type=IntegerReductionState,
        master_state_value_evaluator=evaluator,
        state_representation_factory=None,
        random_generator=random.Random(5),
        dynamics=dynamics,
        node_tree_evaluation_factory=NodeMaxEvaluationFactory[IntegerReductionState](),
    )
    root = IntegerReductionState(6)
    assert isinstance(selector, TreeAndValueBranchSelector)
    exploration = selector.create_tree_exploration(state=root)
    search_result = exploration.explore(random_generator=random.Random(5))
    recommendation = search_result.branch_recommendation
    assert recommendation.recommended_name in {
        dynamics.action_name(root, a) for a in dynamics.legal_actions(root).get_all()
    }
    assert evaluator.nonterminal_scores and all(
        map(math.isfinite, evaluator.nonterminal_scores)
    )
    terminal_nodes = [
        node
        for depth in search_result.tree.descendants
        for node in search_result.tree.descendants[depth].values()
        if node.state.is_game_over()
    ]
    assert terminal_nodes
    for node in terminal_nodes:
        assert isinstance(node.state, IntegerReductionState)
        value = node.tree_evaluation.get_value()
        assert value.certainty is Certainty.TERMINAL
        assert value.score == -node.state.steps
    count = len(evaluator.nonterminal_scores)
    assert evaluator.evaluate(states[-1]) == values[-1]
    assert len(evaluator.nonterminal_scores) == count
