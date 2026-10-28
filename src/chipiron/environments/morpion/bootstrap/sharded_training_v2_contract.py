"""The existing explicit scientific format constants, shared with capsule identity checks."""

from typing import Any

KIND = "morpion_scientific_training_export"
VERSION = 2
CONTRACT = "anemone-values-canonical-morpion-1"
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
