"""Lifecycle contracts for the opt-in, process-owned Torch execution profile."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from unittest.mock import Mock

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from chipiron.learning.torch_runtime import (
    TorchExecutionProfileError,
    torch_execution_profile,
    torch_execution_profile_info,
    torch_execution_profile_metadata,
    validate_torch_execution_profile,
)


@pytest.mark.fast
def test_existing_preserves_custom_settings_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default scope neither resets user flags nor adds historical recipe fields."""
    with sdpa_kernel(SDPBackend.MATH):
        before = torch_execution_profile_info()
        setter = Mock(side_effect=AssertionError("default must not call a setter"))
        monkeypatch.setattr(torch, "use_deterministic_algorithms", setter)
        monkeypatch.setattr(torch.backends.mha, "set_fastpath_enabled", setter)
        monkeypatch.setattr(torch, "set_num_threads", setter)
        with torch_execution_profile() as info:
            assert info == before
            assert torch_execution_profile_metadata() == {}
            validate_torch_execution_profile({})
        assert torch_execution_profile_info() == before


@pytest.mark.fast
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("warn_only", [False, True])
def test_strict_settings_restore_on_success_and_base_exception(
    fail: bool, warn_only: bool
) -> None:
    """Restore even a pre-existing warning policy and a nonstandard SDPA selection."""
    original = torch_execution_profile_info()
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    try:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            before = torch_execution_profile_info()
            try:
                with torch_execution_profile("strict_math") as info:
                    assert info.deterministic and not info.warn_only
                    assert not info.mha_fastpath
                    assert info.math_sdp
                    assert not (
                        info.flash_sdp or info.mem_efficient_sdp or info.cudnn_sdp
                    )
                    assert info.intra_op_threads == before.intra_op_threads
                    assert info.inter_op_threads == before.inter_op_threads
                    assert info.default_dtype == before.default_dtype
                    assert (
                        info.float32_matmul_precision == before.float32_matmul_precision
                    )
                    metadata = torch_execution_profile_metadata()
                    assert json.loads(json.dumps(metadata)) == metadata
                    validate_torch_execution_profile(metadata)
                    if fail:
                        raise KeyboardInterrupt  # noqa: TRY301
            except KeyboardInterrupt:
                assert fail
            assert torch_execution_profile_info() == before
    finally:
        torch.use_deterministic_algorithms(
            original.deterministic, warn_only=original.warn_only
        )


@pytest.mark.fast
def test_nested_existing_inherits_and_nested_strict_restores() -> None:
    """A nested consumer cannot downgrade the owner's process-wide policy."""
    before = torch_execution_profile_info()
    with torch_execution_profile("strict_math") as outer:
        with torch_execution_profile() as inner:
            assert inner == replace(outer, requested="existing")
            assert (
                torch_execution_profile_metadata()["torch_execution_profile"]
                == outer.to_metadata()
            )
        with torch_execution_profile("strict_math") as nested:
            assert nested == outer
        assert torch_execution_profile_info() == outer
    assert torch_execution_profile_info() == before


@pytest.mark.fast
def test_concurrent_scope_entry_rejected_but_workers_can_observe() -> None:
    """Establish once before workers; independent runtime owners cannot race."""

    def competing_owner() -> None:
        with torch_execution_profile("strict_math"):
            pytest.fail("another thread cannot acquire ownership")

    with torch_execution_profile("strict_math"):
        metadata = torch_execution_profile_metadata()
        with ThreadPoolExecutor(max_workers=1) as workers:
            assert workers.submit(torch_execution_profile_metadata).result() == metadata
            with pytest.raises(TorchExecutionProfileError, match="Another thread"):
                workers.submit(competing_owner).result()


@pytest.mark.fast
def test_drift_raises_and_restores() -> None:
    """External mutation must not be recorded as a successful strict profile."""
    before = torch_execution_profile_info()
    with (
        pytest.raises(TorchExecutionProfileError, match="changed"),
        torch_execution_profile("strict_math"),
    ):
        torch.use_deterministic_algorithms(True, warn_only=True)
    assert torch_execution_profile_info() == before


@pytest.mark.fast
def test_setup_failure_restores(monkeypatch: pytest.MonkeyPatch) -> None:
    """A partial setup cannot leak deterministic flags or leave ownership locked."""
    before = torch_execution_profile_info()
    setter = torch.backends.mha.set_fastpath_enabled

    def fail_disabled(value: bool) -> None:
        setter(value)
        if not value:
            message = "setup failed"
            raise RuntimeError(message)

    with monkeypatch.context() as patch:
        patch.setattr(torch.backends.mha, "set_fastpath_enabled", fail_disabled)
        with (
            pytest.raises(RuntimeError, match="setup failed"),
            torch_execution_profile("strict_math"),
        ):
            pytest.fail("setup should fail")
    assert torch_execution_profile_info() == before
    with torch_execution_profile("strict_math"):
        assert torch.are_deterministic_algorithms_enabled()


@pytest.mark.fast
def test_unsupported_error_propagates_without_retry() -> None:
    """An operation's deterministic rejection is never converted to warn-only."""
    before = torch_execution_profile_info()
    with (
        pytest.raises(
            RuntimeError, match="does not have a deterministic implementation"
        ),
        torch_execution_profile("strict_math"),
    ):
        torch.zeros(2).put_(torch.tensor([0, 0]), torch.ones(2), accumulate=False)
    assert torch_execution_profile_info() == before


@pytest.mark.fast
def test_continuation_identity_is_explicit_and_fail_closed() -> None:
    """Absent, malformed, downgraded and altered backend evidence is rejected."""
    with torch_execution_profile("strict_math"):
        saved = torch_execution_profile_metadata()
        for invalid in (
            {},
            {"torch_execution_profile": None},
            {"torch_execution_profile": {}},
        ):
            with pytest.raises(TorchExecutionProfileError):
                validate_torch_execution_profile(invalid)
        record = dict(saved["torch_execution_profile"])
        record["intra_op_threads"] += 1
        with pytest.raises(TorchExecutionProfileError):
            validate_torch_execution_profile({"torch_execution_profile": record})
        record = dict(saved["torch_execution_profile"])
        record["schema_version"] = True
        with pytest.raises(TorchExecutionProfileError):
            validate_torch_execution_profile({"torch_execution_profile": record})
    with pytest.raises(TorchExecutionProfileError):
        validate_torch_execution_profile(saved)
    with pytest.raises(TorchExecutionProfileError):
        validate_torch_execution_profile({"torch_execution_profile": None})
    validate_torch_execution_profile({})


@pytest.mark.fast
def test_invalid_profile_fails_before_mutation() -> None:
    """Reject misspellings instead of silently falling back to existing."""
    before = torch_execution_profile_info()
    with (
        pytest.raises(TorchExecutionProfileError, match="Unknown"),
        torch_execution_profile("strict"),
    ):  # type: ignore[arg-type]
        pytest.fail("invalid profile should fail")
    assert torch_execution_profile_info() == before


@pytest.mark.fast
def test_cleanup_attempts_all_flags_and_preserves_original_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing restoration is attached as evidence, never substituted for failure."""
    before = torch_execution_profile_info()
    setter = torch.backends.mha.set_fastpath_enabled
    original = RuntimeError("model failed")

    def restore_error(value: bool) -> None:
        setter(value)
        if value == before.mha_fastpath:
            message = "restore failed"
            raise OSError(message)

    with monkeypatch.context() as patch:
        patch.setattr(torch.backends.mha, "set_fastpath_enabled", restore_error)
        with (
            pytest.raises(RuntimeError) as caught,
            torch_execution_profile("strict_math"),
        ):
            raise original
        assert caught.value is original
        assert any("restore failed" in note for note in original.__notes__)
    assert torch_execution_profile_info() == before
