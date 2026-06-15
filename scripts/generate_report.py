"""Generate a publication-quality tearsheet for a strategy backtest.

Runs a reference strategy on real data through the mandatory-cost engine and
renders a professional multi-panel report - cumulative growth vs benchmarks,
underwater drawdown, rolling Sharpe, monthly-returns heatmap, and a metrics
table - saved as a PNG (for the README) and a self-contained HTML page.

    python scripts/generate_report.py                      # multi_asset_rotation, 2018-2025
    python scripts/generate_report.py --out docs/img/x.png --start 2015

Honest by construction: it reports the measured numbers (net of costs), it does
not tune toward the design targets. Needs network + yfinance + matplotlib.
"""

from __future__ import annotations

import argparse
import base64
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.getLogger("hmmlearn").setLevel(logging.ERROR)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from backtest.costs.transaction_costs import TransactionCostModel
from backtest.engines.vectorized import VectorizedBacktest
from backtest.metrics.tearsheet import Tearsheet

ROTATION_UNIVERSE = ["QQQ", "VGT", "GLD", "TLT", "SPY", "VIG"]


def _growth(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).cumprod()


def build_report(start: str, end: str):
    from data.providers.yfinance_provider import YFinanceProvider
    from strategies.multi_asset_rotation import MultiAssetRotation

    px = YFinanceProvider().get_prices(ROTATION_UNIVERSE, start=start, end=end)
    if px is None or px.empty:
        raise RuntimeError("could not fetch prices (need network + yfinance)")
    px = px.dropna(how="all").ffill().dropna()

    weekly = px.index[px.index.weekday == 0]
    weights = MultiAssetRotation().generate_weights(px, weekly)
    result = VectorizedBacktest(TransactionCostModel(), capital=1.0).run(weights, px)
    rets = result.returns.dropna()

    ts = Tearsheet(rets)
    m = ts.compute()
    m["dsr"] = __import__(
        "backtest.validation.deflated_sharpe", fromlist=["compute_dsr"]
    ).compute_dsr(rets, n_trials=10)

    spy = px["SPY"].pct_change().reindex(rets.index)
    ew = px.pct_change().mean(axis=1).reindex(rets.index)

    strat_g, spy_g, ew_g = _growth(rets), _growth(spy), _growth(ew)

    plt.style.use("seaborn-v0_8-darkgrid")
    fig = plt.figure(figsize=(14, 11))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.1, 0.9, 1.0], hspace=0.45, wspace=0.22)
    fig.suptitle(
        "quantcortex - Multi-Asset Rotation tearsheet\n"
        f"{rets.index[0].date()} to {rets.index[-1].date()} | weekly rebalance | "
        "3 bps commission + 10 bps slippage (measured, net of costs)",
        fontsize=13, fontweight="bold",
    )

    # Row 0: growth of $1 vs benchmarks.
    ax = fig.add_subplot(gs[0, :])
    ax.plot(strat_g.index, strat_g.to_numpy(), label="Multi-Asset Rotation", color="C0", lw=1.6)
    ax.plot(spy_g.index, spy_g.to_numpy(), label="SPY (buy & hold)", color="C7", lw=1.1, alpha=0.8)
    ax.plot(ew_g.index, ew_g.to_numpy(), label="Equal-weight 6-ETF", color="C2", lw=1.1, alpha=0.8)
    ax.set_title("Growth of $1 (vs benchmarks)")
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="upper left", framealpha=0.9)

    # Row 1: drawdown + rolling Sharpe.
    ax_dd = fig.add_subplot(gs[1, 0])
    dd = ts.drawdown_series()
    ax_dd.fill_between(dd.index, dd.to_numpy(), 0.0, color="C3", alpha=0.45)
    ax_dd.set_title("Underwater (drawdown)")
    ax_dd.set_ylabel("Drawdown")

    ax_rs = fig.add_subplot(gs[1, 1])
    rs = ts.rolling_sharpe(126)
    ax_rs.plot(rs.index, rs.to_numpy(), color="C4", lw=1.2)
    ax_rs.axhline(0.0, color="k", lw=0.8)
    ax_rs.set_title("Rolling Sharpe (126d)")

    # Row 2: monthly heatmap + metrics table.
    ax_hm = fig.add_subplot(gs[2, 0])
    table = ts.monthly_returns_table()
    data = table.drop(columns=["YTD"], errors="ignore")
    mat = data.to_numpy(dtype=float)
    vlim = float(np.nanmax(np.abs(mat))) if np.isfinite(mat).any() else 0.05
    im = ax_hm.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=-vlim, vmax=vlim)
    ax_hm.set_xticks(range(len(data.columns)))
    ax_hm.set_xticklabels(data.columns, rotation=45, ha="right", fontsize=7)
    ax_hm.set_yticks(range(len(data.index)))
    ax_hm.set_yticklabels(data.index, fontsize=7)
    ax_hm.set_title("Monthly returns")
    fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)

    ax_tbl = fig.add_subplot(gs[2, 1])
    ax_tbl.set_axis_off()
    rows = [
        ("CAGR", f"{m['cagr']:+.2%}"), ("Ann. volatility", f"{m['ann_vol']:.2%}"),
        ("Sharpe", f"{m['sharpe']:+.2f}"), ("Sortino", f"{m['sortino']:+.2f}"),
        ("Calmar", f"{m['calmar']:+.2f}"), ("Max drawdown", f"{m['max_drawdown']:+.2%}"),
        ("VaR 95%", f"{m['var_95']:.2%}"), ("CVaR 95%", f"{m['cvar_95']:.2%}"),
        ("Deflated Sharpe", f"{m['dsr']:.3f}"), ("Design target", "Sharpe > 1.10"),
    ]
    tbl = ax_tbl.table(cellText=rows, colLabels=["Metric", "Value"],
                       loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.5)
    ax_tbl.set_title("Performance summary", pad=12)

    return fig, m


def write_html(html_path: Path, png_path: Path, m: dict) -> None:
    b64 = base64.b64encode(png_path.read_bytes()).decode("ascii")
    rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>"
        for k, v in [
            ("CAGR", f"{m['cagr']:+.2%}"), ("Sharpe", f"{m['sharpe']:+.2f}"),
            ("Sortino", f"{m['sortino']:+.2f}"), ("Calmar", f"{m['calmar']:+.2f}"),
            ("Max drawdown", f"{m['max_drawdown']:+.2%}"), ("Deflated Sharpe", f"{m['dsr']:.3f}"),
        ]
    )
    html_path.write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<title>quantcortex report</title>"
        "<body style='font-family:system-ui;max-width:1100px;margin:2rem auto'>"
        "<h1>quantcortex - Multi-Asset Rotation</h1>"
        "<p>Measured, net of costs. Targets are aspirational design goals.</p>"
        f"<table border=1 cellpadding=6 style='border-collapse:collapse'>{rows}</table>"
        f"<p><img style='max-width:100%' src='data:image/png;base64,{b64}'></p>"
        "</body>"
    )


def main(argv) -> int:
    ap = argparse.ArgumentParser(description="generate a strategy tearsheet report")
    ap.add_argument("--start", default="2018")
    ap.add_argument("--end", default="2025")
    ap.add_argument("--out", default="docs/img/multi_asset_rotation_tearsheet.png")
    ap.add_argument("--html", default="reports/multi_asset_rotation.html")
    args = ap.parse_args(argv[1:])

    try:
        fig, m = build_report(f"{args.start}-01-01", f"{args.end}-12-31")
    except Exception as exc:
        print(f"report generation failed: {exc}")
        return 1

    png = Path(args.out)
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png}  (Sharpe {m['sharpe']:+.2f}, CAGR {m['cagr']:+.2%}, "
          f"maxDD {m['max_drawdown']:+.1%}, DSR {m['dsr']:.3f})")

    html = Path(args.html)
    html.parent.mkdir(parents=True, exist_ok=True)
    write_html(html, png, m)
    print(f"wrote {html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
