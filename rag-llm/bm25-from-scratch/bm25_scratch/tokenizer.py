"""Tokenizer with accent folding (French-friendly) and small optional stop-word lists."""

from __future__ import annotations

import re
import unicodedata

_WORD = re.compile(r"[^\W_]+", re.UNICODE)

STOPWORDS_EN = frozenset("a an the of and or to in on for with is are was were be by at as it this that from".split())
STOPWORDS_FR = frozenset(
    "le la les un une des du de d l et ou a au aux en dans sur pour par avec est sont etait ce cet cette ces "
    "il elle ils elles qui que qu se s sa son ses leur leurs ne pas plus mais".split()
)


def fold_accents(text: str) -> str:
    """'Élève à Béjaïa' -> 'Eleve a Bejaia'. Also maps oe/ae ligatures."""
    text = text.replace("œ", "oe").replace("Œ", "OE").replace("æ", "ae").replace("Æ", "AE")
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


class Tokenizer:
    """lowercase -> (optional) accent folding -> split on non-alphanumerics -> drop stop words.

    French elisions fall out naturally: "l'été" -> ["l", "ete"], and single letters are dropped by
    ``min_length=2``.
    """

    def __init__(self, fold: bool = True, stopwords: frozenset[str] | set[str] | None = None, min_length: int = 2):
        self.fold = fold
        self.min_length = min_length
        sw = set(stopwords or ())
        self.stopwords = {fold_accents(w.lower()) if fold else w.lower() for w in sw}

    def __call__(self, text: str) -> list[str]:
        text = text.lower()
        if self.fold:
            text = fold_accents(text)
        return [t for t in _WORD.findall(text) if len(t) >= self.min_length and t not in self.stopwords]

    def config(self) -> dict:
        return {"fold": self.fold, "stopwords": sorted(self.stopwords), "min_length": self.min_length}

    @classmethod
    def from_config(cls, cfg: dict) -> "Tokenizer":
        return cls(cfg["fold"], set(cfg["stopwords"]), cfg["min_length"])
