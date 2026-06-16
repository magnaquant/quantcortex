"""Shared plotting conventions for research reports and paper figures.

The categorical colors follow Paul Tol's color-blind-safe schemes documented at
https://personal.sron.nl/~pault/. Plotting dependencies remain lazy so importing
the scientific core does not require a Matplotlib backend.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Literal

PlotContext = Literal["notebook", "paper", "report"]

BRIGHT_COLORS = (
    "#4477AA",
    "#EE6677",
    "#228833",
    "#CCBB44",
    "#66CCEE",
    "#AA3377",
    "#BBBBBB",
)
HIGH_CONTRAST_COLORS = ("#004488", "#DDAA33", "#BB5566")
RETURN_DIVERGING_COLORS = (
    "#B2182B",
    "#D6604D",
    "#F4A582",
    "#FDDBC7",
    "#F7F7F7",
    "#D1E5F0",
    "#92C5DE",
    "#4393C3",
    "#2166AC",
)

INK = "#24272B"
MUTED_INK = "#5D636B"
GRID = "#D9DEE5"
SPINE = "#AEB4BA"
CASH = "#C9CDD1"
REFERENCE_BLUE = HIGH_CONTRAST_COLORS[0]
COUNTERFACTUAL_AMBER = HIGH_CONTRAST_COLORS[1]
NEGATIVE_RED = HIGH_CONTRAST_COLORS[2]
POSITIVE_GREEN = BRIGHT_COLORS[2]


def _style(context: PlotContext) -> dict[str, object]:
    from cycler import cycler

    if context == "paper":
        font_size = 7.2
        title_size = 8.0
        label_size = 7.2
        tick_size = 6.8
        legend_size = 6.5
        line_width = 1.15
    elif context == "report":
        font_size = 10.5
        title_size = 11.5
        label_size = 10.5
        tick_size = 9.5
        legend_size = 9.0
        line_width = 1.5
    elif context == "notebook":
        font_size = 10.0
        title_size = 11.0
        label_size = 10.0
        tick_size = 9.0
        legend_size = 8.5
        line_width = 1.4
    else:  # pragma: no cover - protected by the public type and validation
        raise ValueError(f"unknown plotting context: {context}")

    return {
        "axes.prop_cycle": cycler(color=BRIGHT_COLORS),
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
        "font.size": font_size,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": title_size,
        "axes.titleweight": "normal",
        "axes.labelsize": label_size,
        "xtick.labelsize": tick_size,
        "ytick.labelsize": tick_size,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.edgecolor": SPINE,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.65,
        "grid.alpha": 0.9,
        "lines.linewidth": line_width,
        "lines.markersize": 4.5,
        "legend.fontsize": legend_size,
        "legend.frameon": False,
        "legend.handlelength": 2.2,
        "legend.borderaxespad": 0.3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


def apply_plot_style(context: PlotContext = "report") -> None:
    """Apply the repository plotting style to subsequent Matplotlib figures."""
    import matplotlib as mpl

    mpl.rcParams.update(_style(context))


@contextmanager
def plot_style_context(context: PlotContext = "notebook") -> Iterator[None]:
    """Temporarily apply the repository plotting style."""
    import matplotlib as mpl

    with mpl.rc_context(_style(context)):
        yield


def style_axis(ax, *, grid: Literal["both", "x", "y", None] = "y") -> None:
    """Use restrained spines and optional major-grid guidance on one axis."""
    ax.grid(False)
    if grid is not None:
        ax.grid(True, axis=grid, which="major", color=GRID, linewidth=0.65)
    ax.spines["left"].set_color(SPINE)
    ax.spines["bottom"].set_color(SPINE)
    ax.tick_params(which="both", length=3.0, width=0.65, color=SPINE)


def add_panel_label(ax, label: str) -> None:
    """Place a compact panel label outside the upper-left plot boundary."""
    ax.text(
        -0.11,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize="large",
        fontweight="bold",
        color=INK,
        clip_on=False,
    )


def return_diverging_colormap():
    """Return a red-to-blue map centered on a near-white zero value."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "quantcortex_return_diverging",
        RETURN_DIVERGING_COLORS,
    )


def contrasting_text_color(rgba: tuple[float, float, float, float]) -> str:
    """Choose black or white text using WCAG relative luminance contrast."""

    def linearize(channel: float) -> float:
        return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4

    red, green, blue, _ = rgba
    luminance = (
        0.2126 * linearize(red)
        + 0.7152 * linearize(green)
        + 0.0722 * linearize(blue)
    )
    contrast_with_black = (luminance + 0.05) / 0.05
    contrast_with_white = 1.05 / (luminance + 0.05)
    return "black" if contrast_with_black >= contrast_with_white else "white"
