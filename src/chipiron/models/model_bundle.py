"""Model bundle resolution for neural-network artifacts."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from chipiron.utils.small_tools import resolve_package_path

ARCHITECTURE_FILE_NAME = "architecture.yaml"
CHIPIRON_NN_FILE_NAME = "chipiron_nn.yaml"


@dataclass(frozen=True, slots=True)
class ModelBundleRef:
    """Reference a model bundle folder and an optional weights file inside it."""

    uri: str
    weights_file: str | None = None


@dataclass(frozen=True, slots=True)
class ResolvedModelBundle:
    """Resolved local paths for a model bundle."""

    bundle_root: str
    weights_file_path: str
    architecture_file_path: str
    chipiron_nn_file_path: str


class ModelBundleError(ValueError):
    """Base error for model bundle handling."""


class InvalidModelBundleUriError(ModelBundleError):
    """Raised when a bundle URI does not match the expected backend format."""

    @classmethod
    def invalid_hf_prefix(cls, uri: str) -> InvalidModelBundleUriError:
        """Build an error for an hf URI missing the expected prefix."""
        return cls(f"Invalid HF model bundle URI {uri!r}: expected an 'hf://' prefix.")

    @classmethod
    def invalid_hf_format(cls, uri: str) -> InvalidModelBundleUriError:
        """Build an error for a malformed hf bundle URI."""
        return cls(
            f"Invalid HF model bundle URI {uri!r}: expected "
            "'hf://<namespace>/<repo>/<bundle>@<revision>'."
        )

    @classmethod
    def missing_hf_bundle_path(cls, uri: str) -> InvalidModelBundleUriError:
        """Build an error for an hf URI missing its bundle folder path."""
        return cls(f"Invalid HF model bundle URI {uri!r}: missing bundle folder path.")

    @classmethod
    def invalid_package_prefix(cls, uri: str) -> InvalidModelBundleUriError:
        """Build an error for a package URI missing the expected prefix."""
        return cls(
            f"Invalid package model bundle URI {uri!r}: expected a 'package://' prefix."
        )

    @classmethod
    def missing_package_bundle_path(cls, uri: str) -> InvalidModelBundleUriError:
        """Build an error for a package URI missing its bundle folder path."""
        return cls(
            f"Invalid package model bundle URI {uri!r}: missing bundle folder path."
        )


class ModelBundleResolutionError(ModelBundleError):
    """Raised when a model bundle cannot be resolved into local files."""

    @classmethod
    def missing_huggingface_hub(cls) -> ModelBundleResolutionError:
        """Build an error for missing Hugging Face support."""
        return cls(
            "Resolving hf:// model bundles requires the 'huggingface_hub' package."
        )


class ModelBundleRootNotFoundError(ModelBundleResolutionError, FileNotFoundError):
    """Raised when the bundle root directory cannot be located."""

    @classmethod
    def missing_root(
        cls,
        resolved_bundle_root: Path,
        bundle_uri: str,
    ) -> ModelBundleRootNotFoundError:
        """Build an error for an unresolved bundle directory."""
        return cls(
            f"Model bundle root {resolved_bundle_root} could not be resolved for {bundle_uri!r}."
        )

    @classmethod
    def expected_directory(
        cls,
        bundle_uri: str,
        resolved_bundle_root: Path,
    ) -> ModelBundleRootNotFoundError:
        """Build an error for a bundle URI that points to a file."""
        return cls(
            f"Model bundle URI {bundle_uri!r} must point to a directory, not the file "
            f"{resolved_bundle_root}."
        )


class ModelBundleFileNotFoundError(ModelBundleResolutionError, FileNotFoundError):
    """Raised when a required file inside the bundle cannot be located."""

    @classmethod
    def unresolved_hf_file(
        cls,
        file_name: str,
        repo_id: str,
        remote_file_name: str,
        revision: str,
    ) -> ModelBundleFileNotFoundError:
        """Build an error for an HF-hosted bundle file that could not be fetched."""
        return cls(
            f"Could not resolve {file_name!r} from model bundle {repo_id!r} "
            f"at {remote_file_name!r} ({revision})."
        )

    @classmethod
    def missing_bundle_file(
        cls,
        bundle_uri: str,
        file_name: str,
        file_path: Path,
    ) -> ModelBundleFileNotFoundError:
        """Build an error for a missing required bundle file."""
        return cls(
            f"Model bundle {bundle_uri!r} is missing required file {file_name!r} at {file_path}."
        )


class ModelBundleWeightsSelectionError(ModelBundleResolutionError):
    """Raised when the resolver cannot decide which weights file to use."""

    @classmethod
    def no_weights_to_autoselect(
        cls,
        bundle_uri: str,
    ) -> ModelBundleWeightsSelectionError:
        """Build an error for a bundle with no selectable weight files."""
        return cls(
            f"Model bundle {bundle_uri!r} does not specify 'weights_file' and contains no "
            "'.pt' files to auto-select."
        )

    @classmethod
    def ambiguous_weights(
        cls,
        bundle_uri: str,
        pt_files: list[Path],
    ) -> ModelBundleWeightsSelectionError:
        """Build an error for a bundle with multiple possible weight files."""
        return cls(
            f"Model bundle {bundle_uri!r} does not specify 'weights_file' and contains multiple "
            f"'.pt' files: {[path.name for path in pt_files]}."
        )

    @classmethod
    def missing_hf_weights_file(
        cls,
        bundle_uri: str,
    ) -> ModelBundleWeightsSelectionError:
        """Build an error for an HF bundle without an explicit weights file."""
        return cls(
            f"HF model bundle {bundle_uri!r} requires an explicit 'weights_file' because remote "
            "bundle contents are not enumerated during resolution."
        )


@dataclass(frozen=True, slots=True)
class _ParsedHfBundleUri:
    """Parsed structure for an hf:// bundle URI."""

    repo_id: str
    bundle_path: PurePosixPath
    revision: str


def _parse_hf_bundle_uri(uri: str) -> _ParsedHfBundleUri:
    """Parse an hf:// bundle URI into repo id, bundle path, and revision."""
    if not uri.startswith("hf://"):
        raise InvalidModelBundleUriError.invalid_hf_prefix(uri)

    payload = uri[len("hf://") :]
    repo_and_path, revision = (
        payload.rsplit("@", 1) if "@" in payload else (payload, "main")
    )
    parts = [part for part in repo_and_path.split("/") if part]
    if len(parts) < 3:
        raise InvalidModelBundleUriError.invalid_hf_format(uri)

    bundle_path = PurePosixPath(*parts[2:])
    if str(bundle_path) in {"", "."}:
        raise InvalidModelBundleUriError.missing_hf_bundle_path(uri)

    return _ParsedHfBundleUri(
        repo_id=f"{parts[0]}/{parts[1]}",
        bundle_path=bundle_path,
        revision=revision or "main",
    )


def _parse_package_bundle_uri(uri: str) -> str:
    """Extract the package-relative bundle path from a package:// URI."""
    if not uri.startswith("package://"):
        raise InvalidModelBundleUriError.invalid_package_prefix(uri)

    relative_path = uri[len("package://") :].strip("/")
    if not relative_path:
        raise InvalidModelBundleUriError.missing_package_bundle_path(uri)
    return relative_path


def _resolve_package_bundle(uri: str) -> Path:
    """Resolve a package:// bundle URI into a local bundle directory path."""
    _parse_package_bundle_uri(uri)
    return Path(resolve_package_path(uri)).resolve(strict=False)


def _resolve_local_bundle(uri: str) -> Path:
    """Resolve a local bundle path into an absolute bundle directory path."""
    return Path(uri).expanduser().resolve(strict=False)


def _download_hf_file(*, repo_id: str, filename: str, revision: str) -> str:
    """Download one file from the Hugging Face Hub and return its local cache path."""
    try:
        huggingface_hub = importlib.import_module("huggingface_hub")
    except ImportError as error:
        raise ModelBundleResolutionError.missing_huggingface_hub() from error

    return str(
        huggingface_hub.hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
        )
    )


def _download_hf_bundle_file(parsed_uri: _ParsedHfBundleUri, file_name: str) -> Path:
    """Download one file from an HF-hosted bundle."""
    remote_file_name = str(parsed_uri.bundle_path / PurePosixPath(file_name))
    try:
        local_path = _download_hf_file(
            repo_id=parsed_uri.repo_id,
            filename=remote_file_name,
            revision=parsed_uri.revision,
        )
    except ModelBundleResolutionError:
        raise
    except Exception as error:
        raise ModelBundleFileNotFoundError.unresolved_hf_file(
            file_name,
            parsed_uri.repo_id,
            remote_file_name,
            parsed_uri.revision,
        ) from error

    return Path(local_path).resolve(strict=False)


def _validate_bundle_root(bundle_root: Path, bundle_uri: str) -> Path:
    """Validate that the resolved bundle root exists and is a directory."""
    resolved_bundle_root = bundle_root.resolve(strict=False)
    if not resolved_bundle_root.exists():
        raise ModelBundleRootNotFoundError.missing_root(
            resolved_bundle_root,
            bundle_uri,
        )
    if not resolved_bundle_root.is_dir():
        raise ModelBundleRootNotFoundError.expected_directory(
            bundle_uri,
            resolved_bundle_root,
        )
    return resolved_bundle_root


def _validate_bundle_file(bundle_root: Path, bundle_uri: str, file_name: str) -> Path:
    """Validate that a required file exists within the bundle root."""
    file_path = (bundle_root / file_name).resolve(strict=False)
    if not file_path.is_file():
        raise ModelBundleFileNotFoundError.missing_bundle_file(
            bundle_uri,
            file_name,
            file_path,
        )
    return file_path


def _resolve_weights_file(
    bundle_root: Path,
    bundle_uri: str,
    weights_file: str | None,
) -> Path:
    """Resolve the requested weights file, or auto-select a unique .pt file."""
    if weights_file is not None:
        return _validate_bundle_file(bundle_root, bundle_uri, weights_file)

    pt_files = sorted(path.resolve(strict=False) for path in bundle_root.glob("*.pt"))
    if len(pt_files) == 1:
        return pt_files[0]
    if not pt_files:
        raise ModelBundleWeightsSelectionError.no_weights_to_autoselect(bundle_uri)
    raise ModelBundleWeightsSelectionError.ambiguous_weights(bundle_uri, pt_files)


def _build_resolved_model_bundle(
    *,
    bundle_root: Path,
    bundle_uri: str,
    weights_file: str | None,
) -> ResolvedModelBundle:
    """Validate a local bundle directory and return its resolved paths."""
    resolved_bundle_root = _validate_bundle_root(bundle_root, bundle_uri)
    architecture_file_path = _validate_bundle_file(
        resolved_bundle_root,
        bundle_uri,
        ARCHITECTURE_FILE_NAME,
    )
    chipiron_nn_file_path = _validate_bundle_file(
        resolved_bundle_root,
        bundle_uri,
        CHIPIRON_NN_FILE_NAME,
    )
    weights_file_path = _resolve_weights_file(
        resolved_bundle_root,
        bundle_uri,
        weights_file,
    )

    return ResolvedModelBundle(
        bundle_root=str(resolved_bundle_root),
        weights_file_path=str(weights_file_path),
        architecture_file_path=str(architecture_file_path),
        chipiron_nn_file_path=str(chipiron_nn_file_path),
    )


def _resolve_hf_bundle(ref: ModelBundleRef) -> ResolvedModelBundle:
    """Resolve an HF-hosted bundle into concrete local cached files."""
    parsed_uri = _parse_hf_bundle_uri(ref.uri)
    if ref.weights_file is None:
        raise ModelBundleWeightsSelectionError.missing_hf_weights_file(ref.uri)

    architecture_file_path = _download_hf_bundle_file(
        parsed_uri,
        ARCHITECTURE_FILE_NAME,
    )
    chipiron_nn_file_path = _download_hf_bundle_file(
        parsed_uri,
        CHIPIRON_NN_FILE_NAME,
    )
    weights_file_path = _download_hf_bundle_file(parsed_uri, ref.weights_file)

    bundle_root = architecture_file_path.parent.resolve(strict=False)
    return ResolvedModelBundle(
        bundle_root=str(bundle_root),
        weights_file_path=str(weights_file_path),
        architecture_file_path=str(architecture_file_path),
        chipiron_nn_file_path=str(chipiron_nn_file_path),
    )


def resolve_model_bundle(ref: ModelBundleRef) -> ResolvedModelBundle:
    """Resolve a model bundle reference into concrete local filesystem paths."""
    if ref.uri.startswith("hf://"):
        return _resolve_hf_bundle(ref)
    if ref.uri.startswith("package://"):
        bundle_root = _resolve_package_bundle(ref.uri)
    else:
        bundle_root = _resolve_local_bundle(ref.uri)

    return _build_resolved_model_bundle(
        bundle_root=bundle_root,
        bundle_uri=ref.uri,
        weights_file=ref.weights_file,
    )


__all__ = [
    "ARCHITECTURE_FILE_NAME",
    "CHIPIRON_NN_FILE_NAME",
    "InvalidModelBundleUriError",
    "ModelBundleError",
    "ModelBundleFileNotFoundError",
    "ModelBundleRef",
    "ModelBundleResolutionError",
    "ModelBundleRootNotFoundError",
    "ModelBundleWeightsSelectionError",
    "ResolvedModelBundle",
    "resolve_model_bundle",
]
