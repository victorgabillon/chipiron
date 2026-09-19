"""Morpion evaluator exports."""

from __future__ import annotations

from typing import Any

__all__ = [
    "MorpionMasterEvaluator",
    "MorpionNeuralEvaluatorArgs",
    "MorpionOverEventDetector",
    "MorpionRegressorMasterEvaluator",
    "MorpionStateEvaluator",
    "build_morpion_master_evaluator",
    "load_morpion_evaluator_from_model_bundle",
]

_EXPORT_MODULES = {
    "MorpionNeuralEvaluatorArgs": ".neural_evaluator_args",
    "MorpionRegressorMasterEvaluator": ".neural_evaluator",
    "load_morpion_evaluator_from_model_bundle": ".neural_evaluator",
    "MorpionMasterEvaluator": ".morpion_state_evaluator",
    "MorpionOverEventDetector": ".morpion_state_evaluator",
    "MorpionStateEvaluator": ".morpion_state_evaluator",
    "build_morpion_master_evaluator": ".morpion_state_evaluator",
}


def __getattr__(name: str) -> Any:
    """Load evaluator implementations only when callers request them."""
    try:
        module_name = _EXPORT_MODULES[name]
    except KeyError as exc:
        raise AttributeError(name) from exc

    from importlib import import_module

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
