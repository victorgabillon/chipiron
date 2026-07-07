"""Tests documenting Chipiron's Parsley dependency contract."""

import pytest


def test_parsley_import_contract() -> None:
    """Chipiron imports `parsley`; pyproject installs it via `parsley-coco`.

    Parser-backed tests are optional in lightweight local environments, but when
    Parsley is installed the canonical import path must provide the YAML
    dataclass resolver used by Chipiron configuration parsing.
    """
    parsley = pytest.importorskip(
        "parsley",
        reason="parser-backed config tests require the parsley-coco distribution",
    )

    assert hasattr(parsley, "resolve_yaml_file_to_base_dataclass")
