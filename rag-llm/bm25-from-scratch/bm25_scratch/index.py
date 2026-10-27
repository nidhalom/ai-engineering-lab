"""Inverted index: term -> {doc_id: term frequency}."""

from __future__ import annotations

from collections import Counter


class InvertedIndex:
    def __init__(self) -> None:
        self.postings: dict[str, dict[int, int]] = {}
        self.doc_len: list[int] = []

    def add(self, tokens: list[str]) -> int:
        """Index one tokenised document and return its integer id."""
        doc_id = len(self.doc_len)
        self.doc_len.append(len(tokens))
        for term, tf in Counter(tokens).items():
            self.postings.setdefault(term, {})[doc_id] = tf
        return doc_id

    @property
    def n_docs(self) -> int:
        return len(self.doc_len)

    @property
    def avg_len(self) -> float:
        return sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0

    def df(self, term: str) -> int:
        return len(self.postings.get(term, ()))

    def to_dict(self) -> dict:
        return {
            "doc_len": self.doc_len,
            # JSON keys must be strings
            "postings": {t: {str(d): tf for d, tf in p.items()} for t, p in self.postings.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "InvertedIndex":
        idx = cls()
        idx.doc_len = list(data["doc_len"])
        idx.postings = {t: {int(d): tf for d, tf in p.items()} for t, p in data["postings"].items()}
        return idx
