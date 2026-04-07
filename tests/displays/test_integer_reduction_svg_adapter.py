"""Focused tests for integer-reduction SVG layout behavior."""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

SVG_NS = {"svg": "http://www.w3.org/2000/svg"}
ADAPTER_PATH = (
    Path(__file__).resolve().parents[2]
    / "src/chipiron/displays/integer_reduction_svg_adapter.py"
)


def load_adapter_module() -> types.ModuleType:
    """Load the adapter module with minimal stubs for its direct dependencies."""

    @dataclass(frozen=True, slots=True)
    class ClickResult:
        action_name: str | None
        interaction_continues: bool

    @dataclass(frozen=True, slots=True)
    class RenderResult:
        svg_bytes: bytes
        info: dict[str, str]

    @dataclass(frozen=True, slots=True)
    class SvgPosition:
        state_tag: object
        payload: object

    class SvgGameAdapter:
        pass

    class InvalidSvgAdapterPayloadTypeError(TypeError):
        def __init__(
            self,
            *,
            adapter_name: str,
            expected_type: type[object],
            actual_value: object,
        ) -> None:
            super().__init__(
                f"{adapter_name} expected {expected_type.__name__}, got {actual_value!r}"
            )

    @dataclass(frozen=True, slots=True)
    class IntegerReductionDisplayPayload:
        value: int
        steps: int
        legal_actions: tuple[str, ...]
        is_terminal: bool

    stub_modules = {
        "chipiron": types.ModuleType("chipiron"),
        "chipiron.displays": types.ModuleType("chipiron.displays"),
        "chipiron.displays.svg_adapter_errors": types.ModuleType(
            "chipiron.displays.svg_adapter_errors"
        ),
        "chipiron.displays.svg_adapter_protocol": types.ModuleType(
            "chipiron.displays.svg_adapter_protocol"
        ),
        "chipiron.environments": types.ModuleType("chipiron.environments"),
        "chipiron.environments.integer_reduction": types.ModuleType(
            "chipiron.environments.integer_reduction"
        ),
        "chipiron.environments.integer_reduction.integer_reduction_gui_encoder": (
            types.ModuleType(
                "chipiron.environments.integer_reduction.integer_reduction_gui_encoder"
            )
        ),
    }

    stub_modules["chipiron.displays.svg_adapter_errors"].InvalidSvgAdapterPayloadTypeError = (
        InvalidSvgAdapterPayloadTypeError
    )
    stub_modules["chipiron.displays.svg_adapter_protocol"].ClickResult = ClickResult
    stub_modules["chipiron.displays.svg_adapter_protocol"].RenderResult = RenderResult
    stub_modules["chipiron.displays.svg_adapter_protocol"].SvgGameAdapter = (
        SvgGameAdapter
    )
    stub_modules["chipiron.displays.svg_adapter_protocol"].SvgPosition = SvgPosition
    stub_modules[
        "chipiron.environments.integer_reduction.integer_reduction_gui_encoder"
    ].IntegerReductionDisplayPayload = IntegerReductionDisplayPayload

    for name, module in stub_modules.items():
        sys.modules[name] = module

    spec = importlib.util.spec_from_file_location(
        "test_integer_reduction_svg_adapter_module",
        ADAPTER_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ADAPTER_MODULE = load_adapter_module()
IntegerReductionSvgAdapter = ADAPTER_MODULE.IntegerReductionSvgAdapter
IntegerReductionDisplayPayload = ADAPTER_MODULE.IntegerReductionDisplayPayload


def make_payload(
    *,
    value: int,
    steps: int = 0,
    legal_actions: tuple[str, ...],
    is_terminal: bool = False,
) -> IntegerReductionDisplayPayload:
    """Build a minimal display payload for adapter-only tests."""
    return IntegerReductionDisplayPayload(
        value=value,
        steps=steps,
        legal_actions=legal_actions,
        is_terminal=is_terminal,
    )


def test_integer_reduction_layout_keeps_content_inside_viewport() -> None:
    """The internal layout helper should keep text and buttons inside bounds."""
    adapter = IntegerReductionSvgAdapter()
    payload = make_payload(value=15, legal_actions=("dec1", "half"))

    layout = adapter._build_layout(size=180, margin=9, payload=payload)

    assert 9.0 <= layout.title.x <= 171.0
    assert 9.0 <= layout.title.y <= 171.0
    assert 9.0 <= layout.value.y <= 171.0
    assert 9.0 <= layout.steps.y <= 171.0
    assert 9.0 <= layout.instruction.y <= 171.0
    assert layout.value.y > layout.title.y
    assert layout.steps.y > layout.value.y
    assert layout.instruction.y > layout.steps.y
    assert len(layout.buttons) == 2

    for button in layout.buttons:
        assert button.x >= 9.0
        assert button.y >= layout.instruction.y
        assert button.width > 0.0
        assert button.height > 0.0
        assert button.x + button.width <= 171.0
        assert button.y + button.height <= 171.0


def test_integer_reduction_render_svg_emits_in_bounds_text_and_buttons() -> None:
    """Rendered SVG should expose centered text and button geometry in bounds."""
    adapter = IntegerReductionSvgAdapter()
    payload = make_payload(
        value=12345678901234567890,
        legal_actions=("subtract 1", "halve it"),
    )
    pos = adapter.position_from_update(state_tag=payload.value, adapter_payload=payload)

    render = adapter.render_svg(pos, size=180, margin=9)
    root = ET.fromstring(render.svg_bytes)
    text_nodes = root.findall("svg:text", SVG_NS)
    rect_nodes = root.findall("svg:rect", SVG_NS)

    assert [node.text for node in text_nodes[:3]] == [
        "Integer Reduction",
        "n = 12345678901234567890",
        "steps = 0",
    ]
    assert text_nodes[3].text == "Choose a reduction"
    assert len(rect_nodes) == 3

    for node in text_nodes:
        x = float(node.attrib["x"])
        y = float(node.attrib["y"])
        assert 9.0 <= x <= 171.0
        assert 9.0 <= y <= 171.0

    for node in rect_nodes[1:]:
        x = float(node.attrib["x"])
        y = float(node.attrib["y"])
        width = float(node.attrib["width"])
        height = float(node.attrib["height"])
        assert x >= 9.0
        assert y >= 9.0
        assert width > 0.0
        assert height > 0.0
        assert x + width <= 171.0
        assert y + height <= 171.0

    assert any("textLength" in node.attrib for node in text_nodes)


def test_integer_reduction_terminal_render_has_no_action_buttons() -> None:
    """Terminal positions should keep the message inside the viewport without buttons."""
    adapter = IntegerReductionSvgAdapter()
    payload = make_payload(value=1, legal_actions=(), is_terminal=True)
    pos = adapter.position_from_update(state_tag=payload.value, adapter_payload=payload)

    render = adapter.render_svg(pos, size=180, margin=9)
    root = ET.fromstring(render.svg_bytes)
    text_values = [node.text for node in root.findall("svg:text", SVG_NS)]
    rect_nodes = root.findall("svg:rect", SVG_NS)

    assert text_values == ["Integer Reduction", "n = 1", "steps = 0", "Reached 1"]
    assert len(rect_nodes) == 1
