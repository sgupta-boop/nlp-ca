"""Phase 4: text representations compared under one cosine-similarity retrieval task.

Representations (all L2-normalized, so the inner product is the cosine similarity):
  tfidf_word   TF-IDF over words, 1-2 grams                     (lexical)
  tfidf_char   TF-IDF over characters, char_wb 3-5 grams         (lexical, robust to spelling/format)
  fasttext     gensim FastText trained on our product names; mean of word vectors (subword)
  minilm       Sentence-BERT all-MiniLM-L6-v2                    (semantic)
  bge          BAAI/bge-small-en-v1.5                            (semantic, stronger)

Task: for every Abt (Amazon) record that has a gold match, rank all Buy (Google) records by cosine.
Metrics: Recall@1, Recall@5, MRR.

Dense vectors are indexed with FAISS (IndexFlatIP = exact inner-product search). TF-IDF vectors are
sparse with tens of thousands of dimensions, so their cosine is computed exactly with a sparse
matrix product instead of densifying them for FAISS.

Run:  python -m src.embed
"""
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src import data_io
from src.preprocess import normalize_text

ROOT = data_io.ROOT
SEED = 42
REPRESENTATIONS = ["tfidf_word", "tfidf_char", "fasttext", "minilm", "bge"]
SBERT_MODELS = {"minilm": "sentence-transformers/all-MiniLM-L6-v2", "bge": "BAAI/bge-small-en-v1.5"}


# ---------------------------------------------------------------- data
def load_benchmark(name: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """left/right tables with columns id, raw, text (normalized) and gold: left id -> set of right ids."""
    if name == "abt_buy":
        d = data_io.load_abt_buy()
        left, right = d["left"][["id", "name"]], d["right"][["id", "name"]]
        pairs = d["matches"].rename(columns={"idAbt": "l", "idBuy": "r"})
    else:
        d = data_io.load_amazon_google()
        left = d["left"][["id", "title"]].rename(columns={"title": "name"})
        right = d["right"][["id", "name"]]
        pairs = d["matches"].rename(columns={"idAmazon": "l", "idGoogleBase": "r"})
    out = []
    for df in (left, right):
        df = df.rename(columns={"name": "raw"}).reset_index(drop=True)
        df["raw"] = df["raw"].fillna("").astype(str)
        df["text"] = df["raw"].map(normalize_text)
        out.append(df)
    gold = pairs.groupby("l")["r"].apply(set).to_dict()
    return out[0], out[1], gold


# ---------------------------------------------------------------- representations
class Encoder:
    """fit(corpus) learns what is needed (vocabulary / word vectors); encode(texts) -> normalized matrix."""

    def __init__(self, kind: str):
        self.kind = kind
        self.model = None

    def fit(self, corpus: list[str]) -> "Encoder":
        if self.kind == "tfidf_word":
            self.model = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True,
                                         token_pattern=r"(?u)\b\w+\b").fit(corpus)
        elif self.kind == "tfidf_char":
            self.model = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(corpus)
        elif self.kind == "fasttext":
            self.model = get_fasttext()
        elif self.kind in SBERT_MODELS:
            self.model = get_sbert(SBERT_MODELS[self.kind])
        return self

    def encode(self, texts: list[str]):
        if self.kind.startswith("tfidf"):
            return normalize(self.model.transform(texts))          # sparse, rows have unit length
        if self.kind == "fasttext":
            return fasttext_vectors(self.model, texts)
        return self.model.encode(texts, batch_size=64, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False).astype("float32")


_sbert = {}


def get_sbert(name: str):
    if name not in _sbert:
        from sentence_transformers import SentenceTransformer
        _sbert[name] = SentenceTransformer(name)
    return _sbert[name]


FASTTEXT_PATH = ROOT / "models" / "fasttext_products.model"


def fasttext_corpus() -> list[list[str]]:
    """Tokenized names: BigBasket, Flipkart, OFF India, and the names (no labels) of both benchmarks."""
    names = []
    for n in ("bigbasket", "flipkart", "off_india"):
        names += pd.read_parquet(ROOT / "data" / "processed" / f"{n}_pre.parquet")["name_clean"].dropna().tolist()
    for b in ("abt_buy", "amazon_google"):
        l, r, _ = load_benchmark(b)
        names += l["text"].tolist() + r["text"].tolist()
    return [n.split() for n in names if n]


def get_fasttext():
    from gensim.models import FastText
    if FASTTEXT_PATH.exists():
        return FastText.load(str(FASTTEXT_PATH))
    corpus = fasttext_corpus()
    # workers=1 makes training deterministic with the fixed seed
    model = FastText(sentences=corpus, vector_size=100, window=5, min_count=2, sg=1, min_n=3, max_n=5,
                     epochs=20, seed=SEED, workers=1)
    FASTTEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(FASTTEXT_PATH))
    return model


def fasttext_vectors(model, texts: list[str]) -> np.ndarray:
    """Mean of the word vectors (subword n-grams make unseen words work too)."""
    out = np.zeros((len(texts), model.vector_size), dtype="float32")
    for i, t in enumerate(texts):
        toks = t.split()
        if toks:
            out[i] = np.mean([model.wv[w] for w in toks], axis=0)
    return normalize(out).astype("float32")


# ---------------------------------------------------------------- search
def cosine_topk(queries, index, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k most cosine-similar index rows for every query row. Inputs are L2-normalized."""
    k = min(k, index.shape[0])
    if sparse.issparse(queries):
        sims = (queries @ index.T).toarray()
        idx = np.argsort(-sims, axis=1)[:, :k]
        return idx, np.take_along_axis(sims, idx, axis=1)
    import faiss
    faiss_index = faiss.IndexFlatIP(index.shape[1])
    faiss_index.add(np.ascontiguousarray(index, dtype="float32"))
    scores, idx = faiss_index.search(np.ascontiguousarray(queries, dtype="float32"), k)
    return idx, scores


def retrieval_metrics(ranked_ids: list[list], gold: list[set]) -> dict:
    r1 = r5 = rr = 0.0
    for ranked, g in zip(ranked_ids, gold):
        rank = next((i + 1 for i, rid in enumerate(ranked) if rid in g), None)
        r1 += rank == 1
        r5 += rank is not None and rank <= 5
        rr += 1 / rank if rank else 0
    n = len(gold)
    return {"recall@1": round(r1 / n, 3), "recall@5": round(r5 / n, 3), "mrr": round(rr / n, 3)}


def evaluate_retrieval(dataset: str, kind: str, field: str = "text") -> dict:
    left, right, gold = load_benchmark(dataset)
    q = left[left["id"].isin(gold)].reset_index(drop=True)
    t0 = time.perf_counter()
    enc = Encoder(kind).fit(left[field].tolist() + right[field].tolist())
    qv, iv = enc.encode(q[field].tolist()), enc.encode(right[field].tolist())
    idx, _ = cosine_topk(qv, iv, k=right.shape[0])   # full ranking, so MRR is exact
    ranked = [right["id"].values[row].tolist() for row in idx]
    m = retrieval_metrics(ranked, [gold[i] for i in q["id"]])
    m.update({"dataset": dataset, "representation": kind, "input": field, "queries": len(q),
              "seconds": round(time.perf_counter() - t0, 1)})
    return m


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    rows = []
    for dataset in ("abt_buy", "amazon_google"):
        for kind in REPRESENTATIONS:
            for field in ("text", "raw"):
                rows.append(evaluate_retrieval(dataset, kind, field))
                print(rows[-1], flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "reports" / "phase4_retrieval.csv", index=False)
    main = res[res["input"] == "text"].pivot(index="representation", columns="dataset",
                                              values=["recall@1", "recall@5", "mrr"])
    print(main.reindex(REPRESENTATIONS))
