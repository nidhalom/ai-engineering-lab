# bm25-from-scratch

BM25 and BM25+ ranking in pure Python (standard library only), with an accent-folding tokenizer that is friendly to French text, an inverted index, JSON save/load, and a demo comparing it with scikit-learn's TF-IDF on a tiny bilingual corpus. Written to understand exactly what sits inside the search step of a RAG pipeline.

## Features

- `Tokenizer`: lowercasing, accent folding (`Béjaïa` -> `bejaia`, `œ` -> `oe`), optional EN/FR stop words, French elisions handled (`l'été` -> `ete`)
- `InvertedIndex`: term -> {doc id: term frequency}, document lengths
- `BM25(variant="bm25" | "bm25+", k1, b, delta)` with `search`, `scores`, `idf`
- `save(path)` / `BM25.load(path)` as JSON (tokenizer config included, so queries tokenise the same way)
- No runtime dependencies; scikit-learn is only needed for the comparison demo and tests

## Quick start

```bash
pip install -e ".[test]"
pytest -q
python -m bm25_scratch.demo
```

```python
from bm25_scratch import BM25, Tokenizer
from bm25_scratch.tokenizer import STOPWORDS_FR

docs = ["Béjaïa est une ville côtière.", "Alger est la capitale de l'Algérie."]
bm = BM25.from_documents(docs, variant="bm25+", tokenizer=Tokenizer(stopwords=STOPWORDS_FR))
bm.search("capitale algerie", top_k=1)      # [(1, 2.9...)]
bm.save("index.json"); bm = BM25.load("index.json")
```

## Example output

Real output of `python -m bm25_scratch.demo` (16 documents, 16 queries; documents have accents, queries do not):

```
16 docs, 16 queries (EN/FR, accents in docs, none in queries)
system            hit@1    MRR
BM25              0.938  0.969
BM25+             0.938  0.969
sklearn TF-IDF    0.938  0.969

Top 3 for 'capitale algerie':
   4.938  Alger est la capitale de l'Algérie, située sur la côte méditerranéenne.
   2.051  Béjaïa est une ville côtière d'Algérie réputée pour ses plages et ses 
```

All three systems miss the same single query (`fonction d'appartenance gaussienne`, where a different document also contains the very common word "fonction"). On a corpus this tiny the systems are indistinguishable; the point of the demo is that the implementation behaves sensibly, not that BM25 beats TF-IDF. Do not read it as a benchmark.

## Project structure

```
bm25_scratch/
  tokenizer.py   fold_accents, Tokenizer
  index.py       InvertedIndex
  ranker.py      BM25 (BM25 / BM25+), save / load
  demo.py        comparison with scikit-learn TF-IDF
tests/test_bm25.py
```

## How it works

For a query term t and document d:

`idf(t) * ( tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl)) + delta )`, with `idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5))`. Plain BM25 has `delta = 0`; BM25+ (Lv and Zhai, 2011) adds `delta = 1` for every query term present in the document so long documents are not over-penalised by length normalisation. Only documents sharing at least one term with the query are scored, by walking the postings lists. A test checks the score against a hand-computed value.

## Limitations

- No stemming or lemmatisation; French plurals and conjugations do not match each other.
- The whole index lives in memory; there is no incremental delete or update.
- The demo corpus is tiny and hand-made, so the numbers above are illustrative only.

## License

MIT, see [LICENSE](LICENSE).
