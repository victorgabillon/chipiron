"""Scientific discriminators and source-policy fences shared with capsule checks."""

from collections.abc import Mapping
from typing import Any

KIND = "morpion_scientific_training_export"
VERSION = 2
CONTRACT = "anemone-values-canonical-morpion-1"
# Reserved B4 discriminator: no current writer/reader admits this contract.
# Changing the discriminator (not merely adding optional metadata) also makes
# frozen readers fail closed. Never replace producer.identity.evaluator with it.
MIXED_SOURCE_CONTRACT = "anemone-values-mixed-direct-lineage-morpion-1"
LINEAGE_CONTRACT = "morpion-direct-prediction-lineage-1"
# Encoding identity is unchanged by the Python 3.14 releases.
REFERENCE_CONTRACT = "atomheart-0.2.11-dump-state-ref"

# Independent storage envelope; scientific values retain VERSION/CONTRACT above.

CLOSED_KIND = "morpion_closed_scientific_snapshot"
CLOSED_LAYOUT = 1


def closed_envelope(role: str, **fields: object) -> dict[str, Any]:
    """Keep layout identity explicit without reinterpreting historical v2 files."""
    return {
        "format_kind": CLOSED_KIND,
        "format_version": VERSION,
        "scientific_contract": CONTRACT,
        "storage_layout_version": CLOSED_LAYOUT,
        "role": role,
        **fields,
    }


def require_historical_source_policy(provenance: Mapping[str, Any]) -> None:
    """Fence B4 declarations even when attached to a historical source envelope.

    Absence means the unchanged historical contract, not a proven full refresh.
    C3 write evidence and C5 admission must exist before any new policy is enabled.
    """
    nested = provenance.get("scientific_source_provenance")
    sources = (provenance, nested) if isinstance(nested, Mapping) else (provenance,)
    for source in sources:
        if (
            source.get("scientific_contract") == MIXED_SOURCE_CONTRACT
            or "prediction_lineage" in source
            or "source_policy" in source
        ):
            raise ValueError(  # noqa: TRY003
                "B4 source provenance is not admitted: Growth write evidence and "
                "mixed-source admission are not implemented."
            )
