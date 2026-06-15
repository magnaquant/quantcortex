"""Quantify S&P 500 survivorship bias using the point-in-time universe.

Ties together the PIT membership reconstruction (data/universe/sp500_wikipedia)
and real price data: it shows how many names that were in the index on a past
date are (a) no longer in the index today and (b) no longer priceable on a
survivor-only feed like yfinance - i.e. exactly the rows a survivorship-biased
backtest silently omits, inflating returns.

    python scripts/survivorship_demo.py            # as of 2018-06-01
    python scripts/survivorship_demo.py 2015-06-01

Requires network + lxml (Wikipedia) and yfinance (pricing the dropped names).
"""

from __future__ import annotations

import logging
import sys
import warnings

warnings.filterwarnings("ignore")
# yfinance reports failed (delisted) downloads via logging + stderr; quiet it so
# the survivorship summary is readable (the failures are the point, counted below).
logging.getLogger("yfinance").setLevel(logging.CRITICAL)


def main(argv) -> int:
    as_of = argv[1] if len(argv) > 1 else "2018-06-01"
    try:
        from data.universe.sp500_universe import SP500Universe
    except Exception as exc:  # pragma: no cover
        print(f"import error: {exc}")
        return 1

    try:
        uni = SP500Universe.from_wikipedia()
    except Exception as exc:
        print(f"could not build the point-in-time universe (need network + lxml): {exc}")
        return 1

    past = set(uni.constituents(as_of))
    today = set(uni.constituents())
    dropped = sorted(past - today)
    added = sorted(today - past)

    print("S&P 500 point-in-time membership (Wikipedia reconstruction)")
    print("=" * 70)
    print(f"  members as of {as_of}: {len(past)}")
    print(f"  members today:          {len(today)}")
    print(f"  in {as_of[:4]} but gone today (dropped): {len(dropped)}")
    print(f"  added since {as_of[:4]}:                  {len(added)}")

    # A naive backtest that uses *today's* members for all history would simply
    # never see the `dropped` names. Show how many are now un-priceable on a
    # survivor-only feed - those are the silently-deleted bankruptcies/delistings.
    try:
        from data.providers.yfinance_provider import YFinanceProvider

        provider = YFinanceProvider()
        px = provider.get_prices(dropped, start=as_of, end="2024-12-31")
        cols = list(getattr(px, "columns", []))
        # A name is "priceable" only if a column came back AND it carries real
        # (non-NaN) data; a failed/delisted download yields no column or all-NaN.
        priceable = [s for s in dropped if s in cols and px[s].notna().any()]
        unpriceable = sorted(set(dropped) - set(priceable))
        print(f"\n  of the {len(dropped)} dropped names, {len(priceable)} are still")
        print(f"  priceable on yfinance and {len(unpriceable)} are NOT (delisted/")
        print("  merged/renamed) - the rows a survivor-only backtest omits.")
        print(f"  examples no longer priceable: {unpriceable[:12]}")
    except Exception as exc:
        print(f"\n  (pricing step skipped: {type(exc).__name__})")
        print(f"  dropped examples: {dropped[:12]}")

    print("\n" + "=" * 70)
    print("Takeaway: building a backtest universe from today's constituents")
    print(f"silently discards ~{len(dropped)} names that were tradable as of {as_of}.")
    print("Use SP500Universe.from_wikipedia() (or a licensed PIT feed) so the")
    print("universe is queried as-of each rebalance date instead.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
