"""Shared text and number formatting helpers for SVG adapters."""


def fit_font_size(
    *,
    text: str,
    max_width: float,
    min_size: float,
    max_size: float,
    width_factor: float = 0.56,
) -> float:
    """Return a width-aware font size bounded by readable min/max values."""
    if not text:
        return min_size
    width_limited_size = max_width / max(len(text) * width_factor, 1.0)
    return max(1.0, min(max_size, max(min_size, width_limited_size)))


def fmt_svg_number(value: float) -> str:
    """Format SVG coordinates compactly while keeping stable precision."""
    return f"{value:.2f}"
