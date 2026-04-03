"""Tests for model bundle resolution."""

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from chipiron.models import model_bundle
from chipiron.models.model_bundle import (
    ARCHITECTURE_FILE_NAME,
    CHIPIRON_NN_FILE_NAME,
    InvalidModelBundleUriError,
    ModelBundleFileNotFoundError,
    ModelBundleRef,
    ModelBundleWeightsSelectionError,
    resolve_model_bundle,
)


def _create_local_bundle(
    bundle_root: Path,
    *,
    weights_file: str = "weights.pt",
    include_architecture: bool = True,
    include_chipiron_nn: bool = True,
) -> Path:
    """Create a minimal local model bundle for tests."""
    bundle_root.mkdir(parents=True, exist_ok=True)

    if include_architecture:
        (bundle_root / ARCHITECTURE_FILE_NAME).write_text(
            "layers: []\n", encoding="utf-8"
        )
    if include_chipiron_nn:
        (bundle_root / CHIPIRON_NN_FILE_NAME).write_text(
            "version: 1\n",
            encoding="utf-8",
        )
    (bundle_root / weights_file).write_text("weights\n", encoding="utf-8")
    return bundle_root


def test_resolve_package_model_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Package bundle resolution should return real local paths."""
    bundle_root = _create_local_bundle(tmp_path / "package-bundle")

    def fake_resolve_package_path(uri: str) -> str:
        assert uri == "package://test/model_bundle"
        return str(bundle_root)

    monkeypatch.setattr(model_bundle, "resolve_package_path", fake_resolve_package_path)

    ref = ModelBundleRef(
        uri="package://test/model_bundle",
        weights_file="weights.pt",
    )

    bundle = resolve_model_bundle(ref)

    assert Path(bundle.bundle_root).is_dir()
    assert Path(bundle.architecture_file_path).is_file()
    assert Path(bundle.chipiron_nn_file_path).is_file()
    assert Path(bundle.weights_file_path).is_file()
    assert Path(bundle.weights_file_path).name == "weights.pt"


def test_resolve_local_model_bundle_from_relative_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local bundle paths should resolve relative to the current working directory."""
    bundle_root = _create_local_bundle(tmp_path / "bundle")
    monkeypatch.chdir(tmp_path)

    bundle = resolve_model_bundle(
        ModelBundleRef(uri="bundle", weights_file="weights.pt")
    )

    assert bundle.bundle_root == str(bundle_root.resolve())
    assert bundle.weights_file_path == str((bundle_root / "weights.pt").resolve())


def test_resolve_hf_model_bundle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """HF bundle resolution should return local cached file paths."""
    downloaded_files: list[tuple[str, str, str]] = []

    def fake_hf_hub_download(*, repo_id: str, filename: str, revision: str) -> str:
        downloaded_files.append((repo_id, filename, revision))
        local_path = tmp_path / "hf-cache" / revision / filename
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text("content\n", encoding="utf-8")
        return str(local_path)

    monkeypatch.setattr(model_bundle, "_download_hf_file", fake_hf_hub_download)

    ref = ModelBundleRef(
        uri="hf://VictorGabillon/chipiron/prelu_no_bug@main",
        weights_file="param_multi_layer_perceptron.pt",
    )

    bundle = resolve_model_bundle(ref)

    assert downloaded_files == [
        ("VictorGabillon/chipiron", "prelu_no_bug/architecture.yaml", "main"),
        ("VictorGabillon/chipiron", "prelu_no_bug/chipiron_nn.yaml", "main"),
        (
            "VictorGabillon/chipiron",
            "prelu_no_bug/param_multi_layer_perceptron.pt",
            "main",
        ),
    ]
    assert Path(bundle.bundle_root).is_dir()
    assert Path(bundle.architecture_file_path).is_file()
    assert Path(bundle.chipiron_nn_file_path).is_file()
    assert Path(bundle.weights_file_path).is_file()


def test_resolve_model_bundle_raises_for_missing_weights_file(tmp_path: Path) -> None:
    """Selecting a missing weights file should fail clearly."""
    bundle_root = _create_local_bundle(tmp_path / "bundle")

    with pytest.raises(ModelBundleFileNotFoundError, match="missing_weights.pt"):
        resolve_model_bundle(
            ModelBundleRef(
                uri=str(bundle_root),
                weights_file="missing_weights.pt",
            )
        )


def test_resolve_model_bundle_raises_for_ambiguous_weights_selection(
    tmp_path: Path,
) -> None:
    """Bundles with multiple .pt files should require an explicit weights file."""
    bundle_root = _create_local_bundle(tmp_path / "bundle")
    (bundle_root / "weights_2.pt").write_text("weights\n", encoding="utf-8")

    with pytest.raises(ModelBundleWeightsSelectionError, match="multiple '.pt' files"):
        resolve_model_bundle(ModelBundleRef(uri=str(bundle_root)))


@pytest.mark.parametrize(
    "missing_file", [ARCHITECTURE_FILE_NAME, CHIPIRON_NN_FILE_NAME]
)
def test_resolve_model_bundle_raises_for_missing_sidecar(
    tmp_path: Path,
    missing_file: str,
) -> None:
    """Missing architecture or chipiron_nn sidecars should fail clearly."""
    bundle_root = _create_local_bundle(
        tmp_path / "bundle",
        include_architecture=missing_file != ARCHITECTURE_FILE_NAME,
        include_chipiron_nn=missing_file != CHIPIRON_NN_FILE_NAME,
    )

    with pytest.raises(ModelBundleFileNotFoundError, match=missing_file):
        resolve_model_bundle(
            ModelBundleRef(
                uri=str(bundle_root),
                weights_file="weights.pt",
            )
        )


@pytest.mark.parametrize(
    "ref",
    [
        ModelBundleRef(
            uri="hf://VictorGabillon/chipiron@main", weights_file="weights.pt"
        ),
        ModelBundleRef(uri="package://", weights_file="weights.pt"),
    ],
)
def test_resolve_model_bundle_raises_for_invalid_uri(ref: ModelBundleRef) -> None:
    """Malformed bundle URIs should fail with a clear error."""
    with pytest.raises(InvalidModelBundleUriError):
        resolve_model_bundle(ref)
