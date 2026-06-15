"""News-headline sentiment scoring and causal daily aggregation.

:class:`NewsScorer` turns a stream of timestamped news headlines into a
per-(date, symbol) sentiment signal suitable for use as an alpha factor. It
delegates the text -> sentiment step to a
:class:`~alpha.factors.nlp.finbert_sentiment.FinBERTSentiment` instance
(FinBERT when available, an offline finance lexicon otherwise), then aggregates
those headline-level scores to a daily panel.

Causality
---------
The daily aggregation is causal at *daily* granularity: the signal reported on
date ``t`` uses only headlines dated on or before ``t`` (timestamps are
normalized to calendar dates). An optional exponential time-decay weighting
lets recent headlines dominate while older news fades, with a configurable
half-life. One caveat: intraday cutoffs are not modeled, so a headline
published after the close of ``t`` still lands in day ``t``'s signal. A
close-of-day trading rule should treat day ``t``'s score as actionable at the
close of ``t + 1`` (or pre-filter the news table to a publication-time
cutoff) to be strictly conservative.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from quantcortex.alpha.factors.nlp.finbert_sentiment import FinBERTSentiment


class NewsScorer:
    """Score news headlines and aggregate them into a causal daily panel.

    Parameters
    ----------
    sentiment:
        A :class:`FinBERTSentiment` instance used to score text. If ``None``
        (default), one is constructed with default settings (FinBERT if the
        transformers stack is installed, otherwise the offline lexicon).
    half_life:
        Half-life, in days, of the exponential recency weighting used by the
        time-decayed aggregation helpers. With a half-life of ``h``, a headline
        ``h`` days older than the evaluation date receives half the weight of a
        same-day headline. Must be positive.
    """

    def __init__(
        self,
        sentiment: Optional[FinBERTSentiment] = None,
        half_life: float = 3.0,
    ) -> None:
        if half_life <= 0:
            raise ValueError("half_life must be positive")
        self.sentiment = sentiment if sentiment is not None else FinBERTSentiment()
        self.half_life = float(half_life)

    # ------------------------------------------------------------------
    # Headline scoring
    # ------------------------------------------------------------------
    def score_headlines(self, headlines: Sequence[str]) -> np.ndarray:
        """Score a list of headlines to sentiment in ``[-1, 1]``.

        Parameters
        ----------
        headlines:
            Iterable of headline strings.

        Returns
        -------
        numpy.ndarray
            One sentiment score per headline in ``[-1, 1]``.
        """
        if isinstance(headlines, str):
            headlines = [headlines]
        headlines = list(headlines)
        if not headlines:
            return np.empty(0, dtype=float)
        return self.sentiment.score(headlines)

    # ------------------------------------------------------------------
    # Recency weighting helper
    # ------------------------------------------------------------------
    def recency_weight(
        self,
        age_days: np.ndarray,
        half_life: Optional[float] = None,
    ) -> np.ndarray:
        """Exponential time-decay weight for a headline of a given age.

        The weight for a headline published ``age_days`` before the evaluation
        date is ``0.5 ** (age_days / half_life)`` -- i.e. 1.0 for same-day news
        and halving every ``half_life`` days. Negative ages (future-dated
        headlines) receive zero weight, enforcing causality.

        Parameters
        ----------
        age_days:
            Non-negative age (in days) of each headline relative to the
            evaluation date. May be a scalar or array-like.
        half_life:
            Override for the instance ``half_life``.

        Returns
        -------
        numpy.ndarray
            Weights in ``[0, 1]`` aligned to ``age_days``.
        """
        hl = self.half_life if half_life is None else float(half_life)
        if hl <= 0:
            raise ValueError("half_life must be positive")
        ages = np.asarray(age_days, dtype=float)
        weights = np.power(0.5, ages / hl)
        # Future-dated (negative age) news must not leak: zero its weight.
        weights = np.where(ages < 0, 0.0, weights)
        return weights

    # ------------------------------------------------------------------
    # Daily aggregation
    # ------------------------------------------------------------------
    def aggregate_daily(
        self,
        news_df: pd.DataFrame,
        *,
        date_col: str = "date",
        symbol_col: str = "symbol",
        headline_col: str = "headline",
        time_decay: bool = False,
        lookback_days: Optional[int] = None,
        as_of: Optional[object] = None,
    ) -> pd.DataFrame:
        """Aggregate headline sentiment into a causal ``date x symbol`` panel.

        Each headline is scored, then for every (date, symbol) cell the score
        is the mean (or recency-weighted mean) of sentiment from headlines
        published on that date. The output panel is reindexed over the full set
        of observed dates and symbols.

        Causality is guaranteed two ways:

        * The cell at ``(t, symbol)`` aggregates only same-day headlines, so by
          construction it never uses news published after ``t``.
        * When ``time_decay`` is enabled and ``as_of`` is supplied, weights are
          computed relative to ``as_of`` and any future-dated headlines are
          dropped before weighting -- only same-day-or-prior news contributes.

        Parameters
        ----------
        news_df:
            Long-format news table with at least date, symbol and headline
            columns (names configurable below).
        date_col, symbol_col, headline_col:
            Column names in ``news_df``.
        time_decay:
            If ``True``, weight each headline by its exponential recency weight
            (see :meth:`recency_weight`) before averaging. Without ``as_of``,
            weighting is computed within each date relative to that date (i.e.
            same-day weight 1.0), which is a no-op for same-day-only cells but
            becomes meaningful together with ``lookback_days``.
        lookback_days:
            If set, each output date ``t`` aggregates all headlines in the
            window ``[t - lookback_days, t]`` (inclusive, causal), with weights
            decaying by age relative to ``t``. If ``None`` (default), only
            same-day headlines are aggregated per date.
        as_of:
            Optional single evaluation date. If provided, returns a single-row
            panel for ``as_of`` aggregating all headlines with date ``<= as_of``
            within ``lookback_days`` (or all prior news if ``lookback_days`` is
            ``None``), time-decay weighted relative to ``as_of``.

        Returns
        -------
        pandas.DataFrame
            Panel indexed by date with one column per symbol of mean (or
            decay-weighted mean) sentiment. Cells with no contributing news are
            ``NaN``.
        """
        self._validate_news(news_df, date_col, symbol_col, headline_col)

        df = news_df[[date_col, symbol_col, headline_col]].copy()
        df.columns = ["date", "symbol", "headline"]
        # Normalize to calendar dates: intraday timestamps would otherwise
        # produce one "daily" row per distinct timestamp instead of per day.
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.dropna(subset=["symbol", "headline"])
        if df.empty:
            return pd.DataFrame(dtype=float)

        # Score every unique headline once for efficiency, then map back.
        unique_headlines = pd.unique(df["headline"].astype(str))
        scores = self.sentiment.score(list(unique_headlines))
        score_map = dict(zip(unique_headlines, scores))
        df["sentiment"] = df["headline"].astype(str).map(score_map).astype(float)

        symbols = pd.Index(sorted(pd.unique(df["symbol"])))

        # --- Single as-of evaluation date ---------------------------------
        if as_of is not None:
            asof_ts = pd.to_datetime(as_of)
            window = df[df["date"] <= asof_ts]  # strictly causal
            if lookback_days is not None:
                lower = asof_ts - pd.Timedelta(days=int(lookback_days))
                window = window[window["date"] >= lower]
            row = self._aggregate_window(window, asof_ts, symbols, time_decay)
            return pd.DataFrame([row], index=pd.Index([asof_ts], name="date"))

        # --- Per-date panel ----------------------------------------------
        eval_dates = pd.Index(sorted(pd.unique(df["date"])), name="date")

        if lookback_days is None and not time_decay:
            # Fast path: plain same-day mean via pivot.
            panel = (
                df.groupby(["date", "symbol"])["sentiment"]
                .mean()
                .unstack("symbol")
            )
            return panel.reindex(index=eval_dates, columns=symbols)

        out = pd.DataFrame(np.nan, index=eval_dates, columns=symbols, dtype=float)
        for t in eval_dates:
            if lookback_days is None:
                window = df[df["date"] == t]
            else:
                lower = t - pd.Timedelta(days=int(lookback_days))
                window = df[(df["date"] <= t) & (df["date"] >= lower)]
            row = self._aggregate_window(window, t, symbols, time_decay)
            out.loc[t] = pd.Series(row)
        return out

    # ------------------------------------------------------------------
    # Internal aggregation of a causal window into one row
    # ------------------------------------------------------------------
    def _aggregate_window(
        self,
        window: pd.DataFrame,
        eval_date: pd.Timestamp,
        symbols: pd.Index,
        time_decay: bool,
    ) -> dict:
        """Aggregate a (already causal) window of headlines into a row dict."""
        row = {sym: np.nan for sym in symbols}
        if window.empty:
            return row

        if time_decay:
            age = (eval_date - window["date"]).dt.days.to_numpy(dtype=float)
            weights = self.recency_weight(age)
        else:
            weights = np.ones(len(window), dtype=float)

        tmp = window[["symbol", "sentiment"]].copy()
        tmp["w"] = weights
        tmp["ws"] = tmp["w"] * tmp["sentiment"]

        grouped = tmp.groupby("symbol")
        wsum = grouped["w"].sum()
        wssum = grouped["ws"].sum()
        for sym in wsum.index:
            denom = wsum.loc[sym]
            row[sym] = float(wssum.loc[sym] / denom) if denom > 0 else np.nan
        return row

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_news(
        news_df: pd.DataFrame,
        date_col: str,
        symbol_col: str,
        headline_col: str,
    ) -> None:
        if not isinstance(news_df, pd.DataFrame):
            raise TypeError("news_df must be a pandas DataFrame")
        missing = [
            c for c in (date_col, symbol_col, headline_col) if c not in news_df.columns
        ]
        if missing:
            raise ValueError(f"news_df is missing required columns: {missing}")
