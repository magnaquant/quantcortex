"""FinBERT financial-text sentiment scoring with an offline lexicon fallback.

:class:`FinBERTSentiment` scores financial text (headlines, filings, earnings
transcripts) on a ``[-1, 1]`` scale where ``+1`` is maximally bullish and
``-1`` is maximally bearish. The score is defined as the model's *positive*
class probability minus its *negative* class probability.

Primary backend
---------------
HuggingFace **FinBERT** (default ``ProsusAI/finbert``) -- a BERT model
fine-tuned on financial communications for three-way sentiment
(positive / negative / neutral). It is loaded lazily via
``transformers.pipeline`` (or :class:`AutoModelForSequenceClassification` +
:class:`AutoTokenizer`). Usage::

    from transformers import pipeline
    clf = pipeline("text-classification", model="ProsusAI/finbert",
                   top_k=None)
    clf(["Profits beat expectations", "Guidance was cut sharply"])

Offline fallback
----------------
If ``transformers``/``torch`` are not installed, the class transparently falls
back to a lightweight, dependency-free **finance lexicon** sentiment scorer: it
counts occurrences of curated bullish vs. bearish finance terms and returns
``(pos - neg) / (pos + neg + eps)``. This keeps the class fully usable offline
(notebooks/tests with no optional libraries) while preserving the same
``[-1, 1]`` interface.

The transformers import is performed lazily inside the scoring methods, so
importing this module needs only numpy/pandas.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence

import numpy as np

_EPS = 1e-9

# ---------------------------------------------------------------------------
# Lightweight finance sentiment lexicon (fallback backend).
# Curated from the spirit of the Loughran-McDonald financial sentiment word
# lists. Not exhaustive -- a pragmatic offline stand-in for FinBERT.
# ---------------------------------------------------------------------------
_POSITIVE_TERMS: frozenset[str] = frozenset(
    {
        "beat", "beats", "beating", "exceed", "exceeds", "exceeded",
        "outperform", "outperformed", "outperforms", "upgrade", "upgraded",
        "growth", "grow", "grew", "growing", "profit", "profits", "profitable",
        "gain", "gains", "gained", "surge", "surged", "surges", "rally",
        "rallied", "rebound", "rebounded", "strong", "strength", "robust",
        "record", "bullish", "buy", "accelerate", "accelerated", "improve",
        "improved", "improving", "improvement", "raise", "raised", "raises",
        "boost", "boosted", "expansion", "expand", "expanded", "upside",
        "optimistic", "positive", "win", "wins", "won", "soar", "soared",
        "soars", "jump", "jumped", "jumps", "higher", "rise", "rose", "rising",
        "dividend", "buyback", "momentum", "tailwind", "tailwinds", "upbeat",
    }
)

_NEGATIVE_TERMS: frozenset[str] = frozenset(
    {
        "miss", "missed", "misses", "missing", "below", "downgrade",
        "downgraded", "loss", "losses", "lose", "lost", "decline", "declined",
        "declines", "declining", "fall", "fell", "falling", "falls", "drop",
        "dropped", "drops", "plunge", "plunged", "plunges", "slump", "slumped",
        "weak", "weakness", "weaker", "bearish", "sell", "selloff", "warn",
        "warned", "warning", "cut", "cuts", "slash", "slashed", "slashes",
        "concern", "concerns", "risk", "risks", "risky", "lawsuit", "probe",
        "investigation", "fraud", "default", "bankruptcy", "bankrupt", "debt",
        "downturn", "recession", "negative", "pessimistic", "lower", "shrink",
        "shrank", "contraction", "headwind", "headwinds", "disappoint",
        "disappointing", "disappointed", "deficit", "writedown", "impairment",
        "downside", "tumble", "tumbled", "tumbles", "slowdown", "layoff",
        "layoffs",
    }
)

_TOKEN_RE = re.compile(r"[a-z][a-z'-]*")


class FinBERTSentiment:
    """Sentiment scorer for financial text, backed by FinBERT or a lexicon.

    Parameters
    ----------
    model_name:
        HuggingFace model id for the transformer backend. Defaults to
        ``"ProsusAI/finbert"``.
    device:
        Optional device hint for the transformers pipeline (e.g. ``-1`` for
        CPU, ``0`` for the first GPU). ``None`` lets transformers decide.
    max_chars_per_chunk:
        Character budget per chunk when scoring long transcripts. Long text is
        split on sentence boundaries into chunks no larger than this, scored
        independently, and averaged (FinBERT truncates at ~512 tokens, so very
        long documents must be chunked).
    force_lexicon:
        If ``True``, skip the transformer backend entirely and always use the
        offline lexicon. Useful for deterministic tests.

    Attributes
    ----------
    backend_:
        ``"transformers"`` or ``"lexicon"`` once a scoring call has resolved
        the backend.
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        device: Optional[int] = None,
        max_chars_per_chunk: int = 1500,
        force_lexicon: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_chars_per_chunk = int(max_chars_per_chunk)
        self.force_lexicon = bool(force_lexicon)

        self.backend_: Optional[str] = None
        self._pipeline = None  # cached transformers pipeline

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def score(self, texts: Sequence[str]) -> np.ndarray:
        """Score a batch of texts to sentiment in ``[-1, 1]``.

        Parameters
        ----------
        texts:
            Iterable of text strings (e.g. headlines or filing snippets).

        Returns
        -------
        numpy.ndarray
            One float per input text in ``[-1, 1]`` (positive minus negative
            probability). Empty/whitespace-only texts score ``0.0`` (neutral).
        """
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        if not texts:
            return np.empty(0, dtype=float)

        if not self.force_lexicon:
            try:
                scores = self._score_transformers(texts)
                self.backend_ = "transformers"
                return scores
            except ImportError:
                pass  # fall through to lexicon
        self.backend_ = "lexicon"
        return self._score_lexicon(texts)

    def score_transcript(self, text: str) -> float:
        """Score a long document (e.g. an earnings-call transcript).

        The text is chunked on sentence boundaries into pieces no larger than
        ``max_chars_per_chunk`` (FinBERT truncates very long inputs), each
        chunk is scored, and the chunk scores are averaged into a single
        document-level sentiment in ``[-1, 1]``.

        Parameters
        ----------
        text:
            The full document text.

        Returns
        -------
        float
            Mean sentiment across chunks, or ``0.0`` for empty input.
        """
        if text is None or not str(text).strip():
            return 0.0
        chunks = self._chunk_text(str(text), self.max_chars_per_chunk)
        if not chunks:
            return 0.0
        scores = self.score(chunks)
        if scores.size == 0:
            return 0.0
        return float(np.nanmean(scores))

    # ------------------------------------------------------------------
    # Transformer backend (lazy import)
    # ------------------------------------------------------------------
    def _score_transformers(self, texts: List[str]) -> np.ndarray:
        """Score texts with HuggingFace FinBERT (lazy import)."""
        pipe = self._get_pipeline()
        # Clean inputs: the pipeline rejects empty strings on some versions.
        cleaned = [t if (t and str(t).strip()) else " " for t in texts]
        raw = pipe(cleaned, truncation=True)

        out = np.zeros(len(texts), dtype=float)
        for i, (orig, result) in enumerate(zip(texts, raw)):
            if not orig or not str(orig).strip():
                out[i] = 0.0
                continue
            # result is a list of {"label","score"} dicts (top_k=None).
            probs = {
                str(d["label"]).lower(): float(d["score"]) for d in result
            }
            pos = probs.get("positive", 0.0)
            neg = probs.get("negative", 0.0)
            out[i] = pos - neg
        return np.clip(out, -1.0, 1.0)

    def _get_pipeline(self):
        """Build and cache the transformers text-classification pipeline."""
        if self._pipeline is not None:
            return self._pipeline
        from transformers import pipeline  # lazy

        kwargs = {"model": self.model_name, "top_k": None}
        if self.device is not None:
            kwargs["device"] = self.device
        self._pipeline = pipeline("text-classification", **kwargs)
        return self._pipeline

    # ------------------------------------------------------------------
    # Lexicon fallback backend
    # ------------------------------------------------------------------
    def _score_lexicon(self, texts: List[str]) -> np.ndarray:
        """Offline finance-lexicon sentiment: (pos - neg) / (pos + neg + eps)."""
        out = np.zeros(len(texts), dtype=float)
        for i, text in enumerate(texts):
            out[i] = self._lexicon_score_one(text)
        return out

    @staticmethod
    def _lexicon_score_one(text: Optional[str]) -> float:
        if not text or not str(text).strip():
            return 0.0
        tokens = _TOKEN_RE.findall(str(text).lower())
        if not tokens:
            return 0.0
        pos = sum(1 for tok in tokens if tok in _POSITIVE_TERMS)
        neg = sum(1 for tok in tokens if tok in _NEGATIVE_TERMS)
        if pos == 0 and neg == 0:
            return 0.0
        return float((pos - neg) / (pos + neg + _EPS))

    # ------------------------------------------------------------------
    # Chunking helper
    # ------------------------------------------------------------------
    @staticmethod
    def _chunk_text(text: str, max_chars: int) -> List[str]:
        """Split text into chunks <= ``max_chars`` on sentence boundaries."""
        if max_chars <= 0:
            return [text]
        # Split into sentence-ish units; keep delimiters attached.
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        chunks: List[str] = []
        current = ""
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if len(sent) > max_chars:
                # Hard-wrap an overly long sentence.
                if current:
                    chunks.append(current)
                    current = ""
                for start in range(0, len(sent), max_chars):
                    chunks.append(sent[start : start + max_chars])
                continue
            if not current:
                current = sent
            elif len(current) + 1 + len(sent) <= max_chars:
                current = f"{current} {sent}"
            else:
                chunks.append(current)
                current = sent
        if current:
            chunks.append(current)
        return chunks
