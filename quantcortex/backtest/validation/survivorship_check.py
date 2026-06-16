"""Survivorship-bias validation against point-in-time universe membership.

Survivorship bias arises when a backtest is run on the set of symbols that
*survived* to today (e.g. today's S&P 500) rather than the symbols that were
actually investable on each historical date.  Companies that went bankrupt,
were delisted, or fell out of the index simply vanish from the data, so the
backtest never holds the losers - systematically inflating returns and
understating risk.

The defence is **point-in-time (PIT) membership**: every symbol traded as of a
date must have been a genuine member of the investable universe *on that date*.
This module cross-checks the symbols a backtest actually used against the
universe's PIT membership and flags any anachronistic holdings.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping

from quantcortex.data.universe.base import Universe

__all__ = [
    "SurvivorshipBiasError",
    "survivorship_check",
    "SurvivorshipValidator",
]


class SurvivorshipBiasError(RuntimeError):
    """Raised when a backtest used symbols that were not PIT members."""


def _normalize_symbols(symbols: Iterable[str], name: str) -> set[str]:
    if isinstance(symbols, (str, bytes)):
        raise TypeError(f"{name} must be an iterable of symbols, not a string")
    normalized: set[str] = set()
    for symbol in symbols:
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError(f"{name} must contain non-empty strings")
        normalized.add(symbol.strip().upper().replace(".", "-"))
    return normalized


def survivorship_check(
    universe: Universe, used_symbols: Iterable[str], as_of
) -> Dict[str, object]:
    """Verify that every used symbol was a PIT member of ``universe`` at ``as_of``.

    Parameters
    ----------
    universe:
        The investable universe, providing PIT membership via
        :meth:`Universe.constituents`.
    used_symbols:
        Symbols the backtest held / traded as of ``as_of``.
    as_of:
        The date the symbols were used (anything accepted by the universe,
        e.g. a ``str`` or :class:`pandas.Timestamp`).

    Returns
    -------
    dict
        ``{ok, non_members, n_used, n_members, as_of}`` where ``ok`` is
        ``True`` iff every used symbol was a member, ``non_members`` is the
        sorted list of offending symbols, ``n_used`` the count of distinct used
        symbols, and ``n_members`` the size of the PIT universe at ``as_of``.
    """
    if not isinstance(universe, Universe):
        raise TypeError("universe must implement the Universe interface")
    members = _normalize_symbols(universe.constituents(as_of), "universe members")
    used = _normalize_symbols(used_symbols, "used_symbols")
    non_members = sorted(used - members)
    return {
        "ok": len(non_members) == 0,
        "non_members": non_members,
        "n_used": len(used),
        "n_members": len(members),
        "as_of": as_of,
    }


class SurvivorshipValidator:
    """Validate a whole backtest's holdings against PIT universe membership.

    Parameters
    ----------
    universe:
        The investable universe to validate against.
    strict:
        If ``True``, :meth:`validate_backtest` raises
        :class:`SurvivorshipBiasError` on the first date with a non-member
        holding instead of returning a report.
    """

    def __init__(self, universe: Universe, strict: bool = False) -> None:
        if not isinstance(universe, Universe):
            raise TypeError("universe must implement the Universe interface")
        if not isinstance(strict, bool):
            raise TypeError("strict must be a boolean")
        self.universe = universe
        self.strict = strict

    def validate_backtest(
        self, symbols_by_date: Mapping[object, Iterable[str]]
    ) -> Dict[object, object]:
        """Check every date's holdings against PIT membership.

        Parameters
        ----------
        symbols_by_date:
            Mapping of ``date -> iterable of symbols used on that date``.

        Returns
        -------
        dict
            ``{ok, n_dates, n_flagged, violations}`` where ``violations`` maps
            each offending date to its per-date :func:`survivorship_check`
            result (only dates with non-members are included).

        Raises
        ------
        SurvivorshipBiasError
            If ``strict`` is ``True`` and any date holds non-members.
        """
        if not isinstance(symbols_by_date, Mapping):
            raise TypeError("symbols_by_date must be a mapping")
        violations: Dict[object, object] = {}
        for as_of, symbols in symbols_by_date.items():
            check = survivorship_check(self.universe, symbols, as_of)
            if not check["ok"]:
                if self.strict:
                    raise SurvivorshipBiasError(
                        f"Survivorship bias on {as_of!r}: "
                        f"{check['non_members']} were not PIT members of "
                        f"universe {getattr(self.universe, 'name', '?')!r}"
                    )
                violations[as_of] = check

        n_dates = len(symbols_by_date)
        return {
            "ok": len(violations) == 0,
            "n_dates": n_dates,
            "n_flagged": len(violations),
            "violations": violations,
        }

    def flagged_symbols(
        self, symbols_by_date: Mapping[object, Iterable[str]]
    ) -> List[str]:
        """Return the sorted set of all symbols ever flagged as non-members."""
        report = self.validate_backtest(symbols_by_date)
        flagged: set[str] = set()
        for check in report["violations"].values():  # type: ignore[union-attr]
            flagged.update(check["non_members"])  # type: ignore[index]
        return sorted(flagged)
