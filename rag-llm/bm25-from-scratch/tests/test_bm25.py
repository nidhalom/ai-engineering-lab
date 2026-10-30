import math

import pytest

from bm25_scratch import BM25, InvertedIndex, Tokenizer, fold_accents

DOCS = [
    "the cat sat on the mat",
    "the dog chased the cat",
    "dogs and cats are pets",
    "a long document about birds " + "filler " * 40 + "cat",
]


def test_fold_accents():
    assert fold_accents("Élève à Béjaïa, cœur") == "Eleve a Bejaia, coeur"


def test_tokenizer_french_and_stopwords():
    tok = Tokenizer(stopwords={"le", "de", "l"})
    assert tok("Le chat de l'été!") == ["chat", "ete"]
    assert Tokenizer(fold=False)("été") == ["été"]


def test_inverted_index():
    idx = InvertedIndex()
    idx.add(["a", "b", "a"])
    idx.add(["b"])
    assert idx.postings["a"] == {0: 2}
    assert idx.df("b") == 2 and idx.n_docs == 2 and idx.avg_len == 2.0
    again = InvertedIndex.from_dict(idx.to_dict())
    assert again.postings == idx.postings


def test_matches_hand_computed_score():
    m = BM25.from_documents(["apple banana", "banana", "cherry"], k1=1.2, b=0.75)
    # N=3, df(apple)=1 ; doc0 length 2, avgdl = 4/3
    idf = math.log(1 + (3 - 1 + 0.5) / (1 + 0.5))
    norm = 1 + 1.2 * (1 - 0.75 + 0.75 * 2 / (4 / 3))
    expected = idf * (1 * 2.2 / norm)
    assert m.scores("apple")[0] == pytest.approx(expected)


def test_ranking_prefers_relevant_doc():
    m = BM25.from_documents(DOCS)
    top = m.search("dog chased", 2)
    assert top[0][0] == 1


def test_unknown_terms_and_empty_query():
    m = BM25.from_documents(DOCS)
    assert m.search("zzzz") == []
    assert m.search("") == []


def test_bm25_plus_helps_long_documents():
    plain = BM25.from_documents(DOCS, variant="bm25")
    plus = BM25.from_documents(DOCS, variant="bm25+")
    # long doc (3) contains "cat" once; BM25+ must lift it relative to plain BM25
    gap_plain = plain.scores("cat")[3] / plain.scores("cat")[0]
    gap_plus = plus.scores("cat")[3] / plus.scores("cat")[0]
    assert gap_plus > gap_plain
    assert plus.scores("cat")[3] > plain.scores("cat")[3]


def test_accent_insensitive_search():
    m = BM25.from_documents(["Une ville côtière d'Algérie", "autre texte"])
    assert m.search("cotiere algerie")[0][0] == 0


def test_save_load_roundtrip(tmp_path):
    m = BM25.from_documents(DOCS + ["Élève à Béjaïa"], variant="bm25+", k1=1.1)
    path = tmp_path / "idx.json"
    m.save(path)
    loaded = BM25.load(path)
    assert loaded.variant == "bm25+" and loaded.k1 == 1.1
    for q in ("cat", "dog chased", "eleve bejaia"):
        assert loaded.search(q) == m.search(q)


def test_bad_variant():
    with pytest.raises(ValueError):
        BM25(variant="nope")


def test_demo_runs(capsys):
    from bm25_scratch import demo

    demo.main()
    out = capsys.readouterr().out
    assert "BM25+" in out and "sklearn TF-IDF" in out
