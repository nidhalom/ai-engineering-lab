"""BM25 and BM25+ ranking over an inverted index."""

from __future__ import annotations

import heapq
import json
import math
from pathlib import Path

from .index import InvertedIndex
from .tokenizer import Tokenizer


class BM25:
    """Okapi BM25 (``variant="bm25"``) or BM25+ (``variant="bm25+"``, Lv & Zhai 2011).

    score(q, d) = sum_t idf(t) * ( tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl)) + delta )

    with idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5)) (always positive) and delta = 0 for plain
    BM25. BM25+ adds a constant delta (default 1) for every query term present in the document,
    which stops very long documents from being over-penalised by length normalisation.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        variant: str = "bm25",
        delta: float = 1.0,
        tokenizer: Tokenizer | None = None,
    ):
        if variant not in ("bm25", "bm25+"):
            raise ValueError("variant must be 'bm25' or 'bm25+'")
        self.k1, self.b, self.variant, self.delta = k1, b, variant, delta
        self.tokenizer = tokenizer or Tokenizer()
        self.index = InvertedIndex()
        self.docs: list[str] = []

    # ---- building -------------------------------------------------------
    def add(self, documents: list[str]) -> "BM25":
        for d in documents:
            self.docs.append(d)
            self.index.add(self.tokenizer(d))
        return self

    @classmethod
    def from_documents(cls, documents: list[str], **kwargs) -> "BM25":
        return cls(**kwargs).add(documents)

    # ---- scoring --------------------------------------------------------
    def idf(self, term: str) -> float:
        n, df = self.index.n_docs, self.index.df(term)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> dict[int, float]:
        """Scores for every document that contains at least one query term."""
        out: dict[int, float] = {}
        avg = self.index.avg_len or 1.0
        delta = self.delta if self.variant == "bm25+" else 0.0
        for term in set(self.tokenizer(query)):
            postings = self.index.postings.get(term)
            if not postings:
                continue
            idf = self.idf(term)
            for doc_id, tf in postings.items():
                norm = tf + self.k1 * (1 - self.b + self.b * self.index.doc_len[doc_id] / avg)
                out[doc_id] = out.get(doc_id, 0.0) + idf * (tf * (self.k1 + 1) / norm + delta)
        return out

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Top-k (doc_id, score), best first. Ties are broken by lower doc_id."""
        return heapq.nsmallest(top_k, self.scores(query).items(), key=lambda kv: (-kv[1], kv[0]))

    # ---- persistence ----------------------------------------------------
    def save(self, path: str | Path) -> None:
        payload = {
            "params": {"k1": self.k1, "b": self.b, "variant": self.variant, "delta": self.delta},
            "tokenizer": self.tokenizer.config(),
            "index": self.index.to_dict(),
            "docs": self.docs,
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BM25":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls(tokenizer=Tokenizer.from_config(payload["tokenizer"]), **payload["params"])
        obj.index = InvertedIndex.from_dict(payload["index"])
        obj.docs = payload["docs"]
        return obj
