"""Execution models - the abstract base and the optimistic ``IdealFill``.

An *execution model* maps a desired trade (``target_qty`` shares of a symbol on
a given bar) to a realised **fill price**.  It is the bridge between the
"paper" decisions of a strategy and the prices a real broker would give you,
and is therefore where slippage, participation, and market-impact assumptions
live.

This module defines:

* :class:`ExecutionModel` -- the abstract base every model implements.
* :class:`IdealFill` -- the zero-slippage benchmark that fills at the bar
  close.  It is *optimistic*: no real order fills at the unperturbed close, so
  ``IdealFill`` results should be read as an upper bound on achievable
  performance, useful for sanity checks and for isolating strategy alpha from
  execution drag.
"""

from __future__ import annotations

import abc

import numpy as np
import pandas as pd

__all__ = ["ExecutionModel", "IdealFill"]


class ExecutionModel(abc.ABC):
    """Abstract base class for execution / fill models.

    Subclasses translate a signed order quantity into the price at which that
    order is assumed to fill on a particular bar.  Implementations must be
    **causal**: they may only use information contained in ``bar`` (the data of
    the bar on which the trade is executed) and never future bars.
    """

    @abc.abstractmethod
    def fill(
        self,
        symbol: str,
        target_qty: float,
        bar: "pd.Series",
        **kw,
    ) -> float:
        """Return the per-share fill price for an order.

        Parameters
        ----------
        symbol:
            The asset being traded.  Supplied so a model may key off
            per-symbol calibration (volatility, ADV, ...).
        target_qty:
            Signed order size in shares.  Positive is a buy, negative a sell.
            The sign determines the direction in which slippage / impact pushes
            the fill price (buys fill higher, sells fill lower).
        bar:
            The OHLCV (and optionally ``vwap``/``volume``) row for the bar on
            which the trade executes.  Only this bar's data may be used.
        **kw:
            Optional extra context (e.g. ``adv``, ``sigma``) used by richer
            models.  Plain models ignore it.

        Returns
        -------
        float
            The fill price per share.
        """
        raise NotImplementedError


class IdealFill(ExecutionModel):
    """Frictionless benchmark: every order fills at the bar close.

    ``IdealFill`` applies **zero slippage and zero market impact** -- the fill
    price equals ``bar['close']`` regardless of order size or direction.  This
    is intentionally optimistic and exists as a benchmark: comparing a strategy
    under :class:`IdealFill` against a realistic model (VWAP participation,
    Almgren-Chriss) quantifies the execution drag the strategy is exposed to.
    """

    def fill(
        self,
        symbol: str,
        target_qty: float,
        bar: "pd.Series",
        **kw,
    ) -> float:
        """Return ``bar['close']`` unchanged (no slippage)."""
        quantity = float(target_qty)
        close = float(bar["close"])
        if not np.isfinite(quantity):
            raise ValueError("target_qty must be finite")
        if not np.isfinite(close) or close <= 0.0:
            raise ValueError("bar close must be finite and positive")
        return close

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "IdealFill()"
