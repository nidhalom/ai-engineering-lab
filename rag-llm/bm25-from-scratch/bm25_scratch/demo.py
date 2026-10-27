"""Compare BM25, BM25+ and scikit-learn TF-IDF on a tiny bilingual corpus.

Run:  python -m bm25_scratch.demo
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from . import BM25, Tokenizer
from .tokenizer import STOPWORDS_EN, STOPWORDS_FR

CORPUS = [
    "Les électrodes du casque EEG enregistrent l'activité cérébrale pendant l'imagerie motrice.",
    "Motor imagery BCI systems decode imagined hand movements from EEG signals.",
    "Le filtre spatial CSP maximise la variance entre deux classes de signaux.",
    "Common spatial patterns are a standard feature extraction method for EEG classification.",
    "La logique floue utilise des fonctions d'appartenance gaussiennes pour classer les données.",
    "K-means clustering finds prototypes that can initialise a fuzzy classifier.",
    "Alger est la capitale de l'Algérie, située sur la côte méditerranéenne.",
    "Béjaïa est une ville côtière d'Algérie réputée pour ses plages et ses montagnes.",
    "Le couscous est un plat traditionnel d'Afrique du Nord préparé avec de la semoule.",
    "Inverted indexes map each term to the list of documents that contain it.",
    "BM25 est une fonction de classement probabiliste utilisée par les moteurs de recherche.",
    "Retrieval augmented generation combines a search step with a language model answer.",
    "Le chunking découpe les documents longs en passages avant l'indexation pour la recherche.",
    "Football fans in Algiers celebrated the national team victory last night.",
    "Les étudiants de l'université USTHB préparent leur master en intelligence artificielle.",
    "Deep convolutional networks such as EEGNet classify brain signals with few parameters.",
]

# (query, index of the relevant document)
QUERIES = [
    ("electrodes activite cerebrale", 0),
    ("imagined hand movement decoding", 1),
    ("filtre spatial variance classes", 2),
    ("feature extraction for EEG", 3),
    ("fonction d'appartenance gaussienne", 4),
    ("prototypes kmeans fuzzy", 5),
    ("capitale algerie", 6),
    ("ville cotiere plages", 7),
    ("plat traditionnel semoule", 8),
    ("term to document list index", 9),
    ("moteurs de recherche classement probabiliste", 10),
    ("language model with search step", 11),
    ("decouper documents longs passages", 12),
    ("national team victory", 13),
    ("master intelligence artificielle universite", 14),
    ("convolutional networks brain signals", 15),
]


def evaluate(rank_fn) -> tuple[float, float]:
    hits, rr = 0, 0.0
    for q, gold in QUERIES:
        ranked = rank_fn(q)
        if ranked and ranked[0] == gold:
            hits += 1
        if gold in ranked:
            rr += 1.0 / (ranked.index(gold) + 1)
    return hits / len(QUERIES), rr / len(QUERIES)


def main() -> None:
    tok = Tokenizer(stopwords=STOPWORDS_EN | STOPWORDS_FR)
    systems = {}
    for name, variant in (("BM25", "bm25"), ("BM25+", "bm25+")):
        m = BM25.from_documents(CORPUS, variant=variant, tokenizer=tok)
        systems[name] = lambda q, m=m: [d for d, _ in m.search(q, 5)]

    vec = TfidfVectorizer(tokenizer=tok, lowercase=False, token_pattern=None)
    mat = vec.fit_transform(CORPUS)
    systems["sklearn TF-IDF"] = lambda q: list(linear_kernel(vec.transform([q]), mat).ravel().argsort()[::-1][:5])

    print(f"{len(CORPUS)} docs, {len(QUERIES)} queries (EN/FR, accents in docs, none in queries)")
    print(f"{'system':16s} {'hit@1':>6s} {'MRR':>6s}")
    for name, fn in systems.items():
        h, mrr = evaluate(fn)
        print(f"{name:16s} {h:6.3f} {mrr:6.3f}")

    m = BM25.from_documents(CORPUS, tokenizer=tok)
    print("\nTop 3 for 'capitale algerie':")
    for d, s in m.search("capitale algerie", 3):
        print(f"  {s:6.3f}  {CORPUS[d][:70]}")


if __name__ == "__main__":
    main()
