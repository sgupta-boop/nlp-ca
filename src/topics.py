"""Phase 9a: category discovery with BERTopic on BigBasket + Flipkart names.

Topics are found without labels (SBERT embeddings -> UMAP -> HDBSCAN -> c-TF-IDF keywords), the LLM writes a
short label for each topic, and the topics are compared with the catalogues' own categories (NMI).

Run:  python -m src.topics fit   (no LLM)   then   python -m src.topics label   (LLM)
"""
import sys

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src import data_io, llm

ROOT = data_io.ROOT
SEED = 42
OUT = ROOT / "reports"
P9 = ROOT / "data" / "processed" / "phase9"


def documents() -> pd.DataFrame:
    bb = pd.read_parquet(ROOT / "data" / "processed" / "bigbasket_pre.parquet")
    fk = pd.read_parquet(ROOT / "data" / "processed" / "flipkart_pre.parquet")
    d = pd.concat([
        pd.DataFrame({"source": "bigbasket", "text": bb["name_clean"], "category": bb["category"]}),
        pd.DataFrame({"source": "flipkart", "text": fk["name_clean"], "category": fk["category"]}),
    ], ignore_index=True)
    return d[d["text"].str.len() > 0].reset_index(drop=True)


def fit(min_cluster_size: int = 60) -> None:
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics import normalized_mutual_info_score as nmi
    from umap import UMAP
    d = documents()
    emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").encode(
        d["text"].tolist(), batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    model = BERTopic(
        umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=SEED),
        hdbscan_model=HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean", prediction_data=True),
        vectorizer_model=CountVectorizer(stop_words="english", min_df=5, ngram_range=(1, 2)),
        calculate_probabilities=False)
    topics, _ = model.fit_transform(d["text"].tolist(), emb)
    d["topic"] = topics
    P9.mkdir(parents=True, exist_ok=True)
    d.to_parquet(P9 / "topics_docs.parquet")
    info = model.get_topic_info()
    info["Representative_Docs"] = info["Representative_Docs"].map(lambda x: list(x)[:5] if x is not None else [])
    info.to_json(P9 / "topic_info.json", orient="records", indent=1, force_ascii=False)
    rows = []
    for src in ("bigbasket", "flipkart"):
        s = d[d["source"] == src]
        inl = s[s["topic"] != -1]
        rows.append({"source": src, "documents": len(s), "outliers_%": round((s["topic"] == -1).mean() * 100, 1),
                     "topics_used": int(inl["topic"].nunique()), "gold_categories": int(s["category"].nunique()),
                     "NMI (outliers as one topic)": round(nmi(s["category"], s["topic"]), 3),
                     "NMI (topic documents only)": round(nmi(inl["category"], inl["topic"]), 3)})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "phase9_topics_nmi.csv", index=False)
    print("topics:", len(info) - 1)
    print(res.to_string(index=False))


class TopicLabel(BaseModel):
    label: str = Field(max_length=60)


def label() -> None:
    info = pd.read_json(P9 / "topic_info.json")
    labels = []
    for _, r in info.iterrows():
        if r["Topic"] == -1:
            labels.append("(outliers)")
            continue
        words = r["Name"].split("_", 1)[-1].replace("_", ", ")
        docs = "\n".join(f"- {x}" for x in r["Representative_Docs"])
        out = llm.ask_json(f"Keywords: {words}\nExample products:\n{docs}\n\nGive this product group a short "
                           "category name (2-4 words).", TopicLabel,
                           system="You name groups of retail products. Answer only with JSON.")
        labels.append(out.label if out else None)
    info["llm_label"] = labels
    info[["Topic", "Count", "Name", "llm_label", "Representative_Docs"]].to_json(
        OUT / "phase9_topics.json", orient="records", indent=1, force_ascii=False)
    print(info[["Topic", "Count", "Name", "llm_label"]].head(30).to_string(index=False))
    print(llm.report())
    llm.unload()


if __name__ == "__main__":
    {"fit": fit, "label": label}[sys.argv[1]]()
