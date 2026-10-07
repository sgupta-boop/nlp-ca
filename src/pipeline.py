"""The whole system as one function, used by the Streamlit app (Phase 9).

raw table -> column mapping (Phase 1) -> preprocessing (Phase 2) -> attributes (Phase 3)
          -> embeddings (Phase 4/7) -> blocking + hybrid matching + clustering (Phase 5)
          -> canonical names (medoid, or the LLM in Phase 6) -> semantic search
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import columns, data_io, preprocess
from src.embed import cosine_topk
from src.match import bigbasket_features, bigbasket_score, cluster_bigbasket

ROOT = data_io.ROOT
FT_MODEL = ROOT / "models" / "minilm-products"
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NER_MODEL = ROOT / "models" / "ner_distilled"


def matching_params() -> dict:
    """Weight/threshold tuned on the BigBasket validation pairs (Phase 5); fallback if not available."""
    path = ROOT / "reports" / "phase5_bigbasket_pairs.csv"
    if path.exists():
        res = pd.read_csv(path)
        best = res.iloc[res["val_f1"].idxmax()]
        return {"w": float(best["w_sbert"]), "threshold": float(best["threshold"]), "use_mrp": "MRP" in best["config"]}
    return {"w": 0.5, "threshold": 0.8, "use_mrp": False}


_model = None


def embedder():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(str(FT_MODEL) if FT_MODEL.exists() else BASE_MODEL)
    return _model


def map_table(raw: pd.DataFrame, use_llm: bool) -> pd.DataFrame:
    headers = list(raw.columns)
    samples = [columns.sample_values(raw[h]) for h in headers]
    if use_llm:
        mapping = columns.map_columns(headers, samples)
    else:
        mapping = pd.DataFrame({"header": headers, "predicted": [columns.map_by_rules(h) for h in headers],
                                "decided_by": "rules"})
    return columns.resolve_price_conflict(mapping, raw)


def extract_attributes(pre: pd.DataFrame) -> pd.DataFrame:
    from src.extract import extract_rules, extract_spacy
    texts = (pre["brand"].fillna("").astype(str) + " " + pre["product_name"].fillna("").astype(str)).str.strip().tolist()
    rules = pd.DataFrame(extract_rules(texts))
    out = pd.DataFrame({"product_name": pre["product_name"], "brand (column)": pre["brand"],
                        "brand (rules)": rules["brand"], "product type (rules)": rules["product_type"],
                        "colour": rules["colour"], "quantity": pre["qty_value"], "unit": pre["qty_unit"],
                        "pack count": pre["pack_count"]})
    if NER_MODEL.exists():
        import spacy
        ner = pd.DataFrame(extract_spacy(spacy.load(NER_MODEL), texts))
        out["product type (distilled NER)"] = ner["product_type"]
        out["variant (distilled NER)"] = ner["variant"]
    return out


def run(raw: pd.DataFrame, use_llm: bool = False, llm_names: int = 15) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    raw = raw.reset_index(drop=True)
    mapping = map_table(raw, use_llm)
    canon = columns.to_canonical(raw, mapping, "upload")
    selection = columns.select_columns(raw)
    if canon["product_name"].isna().all():                      # no name column was recognised:
        primary = selection.loc[selection["role"] == "primary (name)", "column"]   # use the selected one
        if len(primary):
            canon["product_name"] = raw[primary.iat[0]].values
    canon["product_name"] = canon["product_name"].fillna("").astype(str)
    pre = preprocess.preprocess_table(canon).reset_index(drop=True)
    attrs = extract_attributes(pre)

    pre["text"] = (pre["brand"].fillna("").astype(str).map(preprocess.normalize_text) + " " + pre["name_clean"]).str.strip()
    E = embedder().encode(pre["text"].tolist(), batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    T = normalize(TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit_transform(pre["text"]))
    k = min(11, len(pre))
    idx, _ = cosine_topk(E, E, k)
    pairs = np.array(sorted({(min(a, int(b)), max(a, int(b))) for a, row in enumerate(idx) for b in row if a != b}))
    p = matching_params()
    if len(pairs):
        f = bigbasket_features(pre, pairs, E, T)
        use_mrp = p["use_mrp"] and pre["mrp"].notna().any()
        pre["cluster_id"] = cluster_bigbasket(pre, f, p["w"], p["threshold"], use_mrp).values
    else:
        pre["cluster_id"] = np.arange(len(pre))

    names = (pre["brand"].fillna("").astype(str) + " " + pre["product_name"]).str.strip()
    rows = []
    for cid, g in pre.groupby("cluster_id"):
        members = names.loc[g.index].tolist()
        e = E[g.index]
        rows.append({"cluster_id": int(cid), "size": len(g), "canonical_name": members[int((e @ e.T).sum(1).argmax())],
                     "members": members})
    clusters = pd.DataFrame(rows).sort_values("size", ascending=False).reset_index(drop=True)
    if use_llm:
        from src.llm_layer import canonical_name
        top = clusters.index[clusters["size"] > 1][:llm_names]
        clusters.loc[top, "canonical_name"] = [canonical_name(m) or c for m, c in
                                               zip(clusters.loc[top, "members"], clusters.loc[top, "canonical_name"])]
    return {"mapping": mapping, "canonical": canon, "selection": selection, "preprocessed": pre,
            "attributes": attrs, "clusters": clusters, "embeddings": E, "params": p}


def search(result: dict, query: str, k: int = 10) -> pd.DataFrame:
    q = embedder().encode([preprocess.normalize_text(query)], normalize_embeddings=True)
    idx, scores = cosine_topk(q, result["embeddings"], min(k, len(result["embeddings"])))
    pre = result["preprocessed"]
    out = pre.loc[idx[0], ["brand", "product_name", "qty_value", "qty_unit", "price", "cluster_id"]].copy()
    out.insert(0, "cosine", np.round(scores[0], 3))
    return out.reset_index(drop=True)
