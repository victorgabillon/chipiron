"""Small centralized visual system, using owned classes and stable test IDs."""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING, Any

from .view_model import age_label, display

if TYPE_CHECKING:
    from .view_model import BootstrapDashboardSnapshot

PALETTE = {
    "green": "#245b52",
    "amber": "#b48035",
    "blue": "#4d719a",
    "muted": "#697873",
}
CSS = """
<style>
:root { --operator-ink:#263631; --operator-muted:#697873; --operator-border:#e1e6df; }
[data-testid="stAppViewContainer"] { background:#f7f8f5; font-family:Arial, sans-serif; }
[data-testid="stMainBlockContainer"] { padding-top:4.5rem; padding-bottom:2rem; max-width:1600px; }
[data-testid="stSidebar"] { background:#eef1ed; border-right:1px solid var(--operator-border); }
[data-testid="stMetric"] { background:white; border:1px solid var(--operator-border); border-radius:10px; padding:14px 16px; min-height:108px; }
[data-testid="stMetricValue"] { font-size:1.45rem; line-height:1.4; }
[data-testid="stMetricValue"] * { white-space:normal !important; overflow-wrap:anywhere; text-overflow:clip !important; }
[data-testid="stMetricLabel"] * { white-space:normal !important; text-overflow:clip !important; }
[data-testid="stMetricLabel"] { color:var(--operator-muted); font-size:.8rem; }
[data-testid="stExpander"] { background:#fff; border-radius:10px; }
h1,h2,h3 { letter-spacing:-.025em; color:var(--operator-ink); }
h2 { font-size:1.35rem !important; } h3 { font-size:1.1rem !important; }
.operator-eyebrow { font-size:11px; letter-spacing:.16em; font-weight:650; color:#698278; text-transform:uppercase; margin:0 0 5px; }
.operator-header { display:flex; align-items:center; justify-content:space-between; gap:24px; margin-bottom:18px; }
.operator-header h1 { font-size:30px; line-height:1.15; font-weight:650; margin:0 0 8px; }
.operator-subtitle { color:#697873; font-size:13px; overflow-wrap:anywhere; }
.operator-pill { display:inline-block; padding:5px 11px; border-radius:20px; background:#e9ede9; color:#52645c; font-size:11px; font-weight:650; letter-spacing:.08em; text-transform:uppercase; white-space:nowrap; }
.operator-pill.running { background:#e2eee8; color:#245b52; }
.operator-pill.prepared { background:#e7edf2; color:#4d719a; }
.operator-pill.failed { background:#f3e5e2; color:#963f34; }
.operator-workers { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:8px 0 12px; }
.operator-worker { border-left:2px solid #c4cfc7; padding:4px 12px; }
.operator-worker strong { font-size:12px; font-weight:600; }
.operator-worker span { color:#697873; font-size:12px; }
.operator-worker small { display:block; color:#697873; font-size:11px; margin-top:3px; }
.operator-note { color:#697873; font-size:13px; padding:14px 0; }
@media(max-width:1150px) { .operator-header h1 {font-size:26px;} .operator-workers {grid-template-columns:repeat(2,1fr);} [data-testid="stMetric"] {padding:10px;} }
</style>
"""


def apply_theme(st: Any) -> None:
    """Apply one maintained stylesheet without remote resources."""
    st.html(CSS)


def page_header(
    st: Any, snapshot: BootstrapDashboardSnapshot, title: str, subtitle: str
) -> None:
    """Keep identity, lifecycle evidence and freshness visible on every page."""
    st.html(
        '<div class="operator-header"><div><div class="operator-eyebrow">Morpion · Research console</div>'
        f'<h1>{escape(title)}</h1><div class="operator-subtitle">{escape(subtitle)}</div></div>'
        f'<div><span class="operator-pill {escape(snapshot.status)}">{escape(snapshot.status)}</span></div></div>'
    )
    st.caption(
        f"{snapshot.work_dir.name} · Artifact updated {age_label(snapshot.updated_at)}",
        help=snapshot.status_detail,
    )
    for error in snapshot.errors:
        st.warning(error)


def metric_row(st: Any, values: list[tuple[str, object]]) -> None:
    """Render exact, consistently formatted values with native metric semantics."""
    widths = [1.8 if label == "Active evaluator" else 1 for label, _ in values]
    for column, (label, value) in zip(st.columns(widths), values, strict=True):
        column.metric(label, display(value), help=f"{label}: {display(value)}")


def worker_strip(st: Any, snapshot: BootstrapDashboardSnapshot) -> None:
    """Never use an optimistic color as a substitute for liveness evidence."""
    rows = "".join(
        f'<div class="operator-worker"><strong>{escape(worker.name)}</strong> '
        f"<span>· {escape(worker.state)}</span><small>{escape(worker.detail)}</small></div>"
        for worker in snapshot.workers
    )
    st.html(f'<div class="operator-workers">{rows}</div>')
