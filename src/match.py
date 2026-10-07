"""Phase 5: blocking, hybrid matching and clustering.

1. Blocking     candidate pairs = same brand  OR  in each other's FAISS top-k (SBERT)
2. Hybrid score s = w * cos_SBERT + (1 - w) * cos_charTFIDF
   then attribute rules (Phase 2/3):
     - quantity differs (same unit, > 2 % apart)  -> veto (s = 0)
     - pack count differs                          -> veto
     - brand differs (both known)                  -> s - brand_penalty
     - model numbers differ (both have one)        -> s - model_penalty   (added for electronics)
3. Decision     match if s >= threshold; w, penalties and threshold tuned on a validation split
4. Clustering   graph connected components / agglomerative (average linkage) / HDBSCAN

Run:  python -m src.match
"""
import itertools
import json
import re
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, v_measure_score
from sklearn.preprocessing import normalize

from src import data_io
from src.embed import cosine_topk, get_sbert, load_benchmark
from src.preprocess import normalize_text, parse_quantity

ROOT = data_io.ROOT
SEED = 42
OUT = ROOT / "reports"
SBERT = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------- attributes used by the rules
MODEL_RE = re.compile(r"\b(?=[a-z0-9]*\d)(?=[a-z0-9]*[a-z])[a-z0-9]{4,}\b")


def model_codes(text: str) -> set[str]:
    """Alphanumeric codes such as 'PS-LX350H' -> {'pslx350h'}. Hyphens/slashes inside codes are removed;
    canonical quantities ('1000 g') are not codes because they contain a space."""
    t = re.sub(r"(?<=[a-z0-9])[-/.](?=[a-z0-9])", "", str(text).lower())
    return {c for c in MODEL_RE.findall(t) if not re.fullmatch(r"\d+(gb|tb|mb|mm|cm|in|mp|hz|w|v)", c)}


def codes_conflict(a: set[str], b: set[str]) -> bool:
    """Both have codes and none agree (one code may be a prefix of the other: 'am53bk' ~ 'am53')."""
    if not a or not b:
        return False
    return not any(x.startswith(y) or y.startswith(x) for x in a for y in b)


def attributes(df: pd.DataFrame, text_col: str = "raw", brand_col: str | None = None) -> pd.DataFrame:
    """Per record: brand (column if given, else the rule-based gazetteer from Phase 3), quantity, pack, codes."""
    from src.extract import extract_rules
    out = pd.DataFrame(index=df.index)
    if brand_col is not None and brand_col in df:
        out["brand"] = df[brand_col].fillna("").astype(str).str.lower().str.strip().replace("", None)
    else:
        out["brand"] = [(r["brand"] or "").lower() or None for r in extract_rules(df[text_col].tolist())]
    q = [parse_quantity(t) for t in df[text_col]]
    out["qty_value"] = [x["qty_value"] for x in q]
    out["qty_unit"] = [x["qty_unit"] for x in q]
    out["pack_count"] = [x["pack_count"] for x in q]
    out["codes"] = [model_codes(t) for t in df[text_col]]
    return out


# ---------------------------------------------------------------- features for candidate pairs
class PairScorer:
    """Holds embeddings + TF-IDF vectors for two tables and scores (i, j) pairs."""

    def __init__(self, left_text: list[str], right_text: list[str], left_attr: pd.DataFrame,
                 right_attr: pd.DataFrame, sbert_name: str = SBERT, sbert_model=None):
        model = sbert_model or get_sbert(sbert_name)
        self.le = model.encode(left_text, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        self.re = model.encode(right_text, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        tf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(left_text + right_text)
        self.lt, self.rt = normalize(tf.transform(left_text)), normalize(tf.transform(right_text))
        self.la, self.ra = left_attr.reset_index(drop=True), right_attr.reset_index(drop=True)

    def features(self, pairs: np.ndarray) -> pd.DataFrame:
        i, j = pairs[:, 0], pairs[:, 1]
        f = pd.DataFrame({"i": i, "j": j})
        f["sbert"] = np.einsum("ij,ij->i", self.le[i], self.re[j])
        f["char"] = np.asarray(self.lt[i].multiply(self.rt[j]).sum(axis=1)).ravel()
        la, ra = self.la.iloc[i].reset_index(drop=True), self.ra.iloc[j].reset_index(drop=True)
        both_q = la["qty_value"].notna() & ra["qty_value"].notna() & (la["qty_unit"] == ra["qty_unit"])
        rel = (la["qty_value"] - ra["qty_value"]).abs() / np.maximum(la["qty_value"].abs(), 1e-9)
        f["qty_conflict"] = (both_q & (rel > 0.02)).values
        both_p = la["pack_count"].notna() & ra["pack_count"].notna()
        f["pack_conflict"] = (both_p & (la["pack_count"] != ra["pack_count"])).values
        both_b = la["brand"].notna() & ra["brand"].notna()
        f["brand_conflict"] = (both_b & (la["brand"] != ra["brand"])).values
        f["code_conflict"] = [codes_conflict(a, b) for a, b in zip(la["codes"], ra["codes"])]
        return f


def hybrid_score(f: pd.DataFrame, w: float, brand_penalty: float = 0.0, model_penalty: float = 0.0,
                 use_vetoes: bool = True) -> np.ndarray:
    s = w * f["sbert"].values + (1 - w) * f["char"].values
    s = s - brand_penalty * f["brand_conflict"].values - model_penalty * f["code_conflict"].values
    if use_vetoes:
        s = np.where(f["qty_conflict"].values | f["pack_conflict"].values, 0.0, s)
    return s


# ---------------------------------------------------------------- blocking
def block(scorer: PairScorer, k: int = 20, use_brand: bool = True) -> np.ndarray:
    """Candidate (i, j) pairs: j in i's SBERT top-k, or i in j's top-k, or same known brand."""
    idx_lr, _ = cosine_topk(scorer.le, scorer.re, k)
    idx_rl, _ = cosine_topk(scorer.re, scorer.le, k)
    pairs = {(i, int(j)) for i, row in enumerate(idx_lr) for j in row}
    pairs |= {(int(i), j) for j, row in enumerate(idx_rl) for i in row}
    if use_brand:
        lb, rb = scorer.la["brand"], scorer.ra["brand"]
        right_by_brand = rb.dropna().groupby(rb.dropna()).groups
        for i, b in lb.dropna().items():
            pairs |= {(i, int(j)) for j in right_by_brand.get(b, [])}
    return np.array(sorted(pairs))


# ---------------------------------------------------------------- metrics
def prf(pred: set, gold: set) -> dict:
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / (p + r) if p + r else 0.0, 3)}


def tune(f: pd.DataFrame, labels: np.ndarray, grid_w=(0.0, 0.25, 0.5, 0.75, 1.0),
         grid_brand=(0.0, 0.1, 0.2), grid_model=(0.0, 0.1, 0.2), extra_gold_missing: int = 0,
         use_rules: bool = True) -> dict:
    """Grid search on a validation split. extra_gold_missing = gold pairs lost by blocking (always FN)."""
    best = {"f1": -1}
    n_pos = labels.sum() + extra_gold_missing
    for w in grid_w:
        for bp in (grid_brand if use_rules else (0.0,)):
            for mp in (grid_model if use_rules else (0.0,)):
                s = hybrid_score(f, w, bp, mp, use_vetoes=use_rules)
                order = np.argsort(-s)
                tp = np.cumsum(labels[order])
                k = np.arange(1, len(s) + 1)
                prec, rec = tp / k, tp / max(n_pos, 1)
                f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0)
                b = int(f1.argmax())
                if f1[b] > best["f1"]:
                    best = {"f1": float(f1[b]), "w": w, "brand_penalty": bp, "model_penalty": mp,
                            "threshold": float(s[order][b])}
    return best


def split_ids(ids, frac: float = 0.5) -> tuple[set, set]:
    ids = sorted(ids)
    rng = np.random.default_rng(SEED)
    rng.shuffle(ids)
    cut = int(len(ids) * frac)
    return set(ids[:cut]), set(ids[cut:])


# ---------------------------------------------------------------- benchmark matching (Abt-Buy, Amazon-Google)
def run_benchmark(name: str, sbert_name: str = SBERT, configs=None) -> tuple[pd.DataFrame, dict]:
    left, right, gold = load_benchmark(name)
    la, ra = attributes(left), attributes(right)
    scorer = PairScorer(left["text"].tolist(), right["text"].tolist(), la, ra, sbert_name)
    t0 = time.perf_counter()
    cand = block(scorer)
    block_s = time.perf_counter() - t0
    lid, rid = left["id"].values, right["id"].values
    gold_pairs = {(l, r) for l, rs in gold.items() for r in rs}
    cand_ids = [(lid[i], rid[j]) for i, j in cand]
    labels = np.array([p in gold_pairs for p in cand_ids])
    blocking = {"candidates": len(cand), "all_pairs": len(left) * len(right),
                "reduction_ratio": round(1 - len(cand) / (len(left) * len(right)), 4),
                "pair_completeness": round(labels.sum() / len(gold_pairs), 3), "seconds": round(block_s, 1)}

    f = scorer.features(cand)
    val_ids, test_ids = split_ids(list(gold))   # split the left records that have a match
    others = set(lid) - set(gold)
    v_o, t_o = split_ids(list(others))
    val_ids |= v_o
    test_ids |= t_o
    is_val = np.array([lid[i] in val_ids for i in cand[:, 0]])
    gold_val = {p for p in gold_pairs if p[0] in val_ids}
    gold_test = {p for p in gold_pairs if p[0] in test_ids}
    missing_val = len(gold_val) - labels[is_val].sum()

    rows = []
    configs = configs or {
        "char TF-IDF only": dict(grid_w=(0.0,), use_rules=False),
        "SBERT only": dict(grid_w=(1.0,), use_rules=False),
        "hybrid (no rules)": dict(use_rules=False),
        "hybrid + rules (no model-number rule)": dict(grid_model=(0.0,)),
        "hybrid + all rules": dict(),
    }
    params = {}
    for cname, kw in configs.items():
        use_rules = kw.get("use_rules", True)
        p = tune(f[is_val], labels[is_val], extra_gold_missing=missing_val, **kw)
        s = hybrid_score(f, p["w"], p["brand_penalty"], p["model_penalty"], use_vetoes=use_rules)
        pred_test = {cand_ids[k] for k in np.where((~is_val) & (s >= p["threshold"]))[0]}
        rows.append({"dataset": name, "config": cname, **prf(pred_test, gold_test),
                     "val_f1": round(p["f1"], 3), "w_sbert": p["w"], "brand_penalty": p["brand_penalty"],
                     "model_penalty": p["model_penalty"], "threshold": round(p["threshold"], 3)})
        params[cname] = p
    feats = f.assign(label=labels, is_val=is_val, left_id=[c[0] for c in cand_ids],
                     right_id=[c[1] for c in cand_ids])
    feats.drop(columns=["i", "j"]).to_parquet(ROOT / "data" / "processed" / f"phase5_{name}_candidates.parquet")
    return pd.DataFrame(rows), {"blocking": blocking, "params": params}


# ---------------------------------------------------------------- WDC Products (pairs and clustering)
def wdc_pair_features(df: pd.DataFrame, model) -> pd.DataFrame:
    lt = df["title_left"].fillna("").map(normalize_text).tolist()
    rt = df["title_right"].fillna("").map(normalize_text).tolist()
    la = attributes(pd.DataFrame({"raw": df["title_left"].fillna(""), "b": df["brand_left"]}), brand_col=None)
    ra = attributes(pd.DataFrame({"raw": df["title_right"].fillna(""), "b": df["brand_right"]}), brand_col=None)
    scorer = PairScorer(lt, rt, la, ra, sbert_model=model)
    idx = np.arange(len(df))
    return scorer.features(np.stack([idx, idx], axis=1))


def run_wdc_pairs(model) -> tuple[pd.DataFrame, dict]:
    val = data_io.load_wdc_products("pair", 80, "valid", "large")
    rows, params = [], {}
    fv = wdc_pair_features(val, model)
    lv = val["label"].values.astype(bool)
    tests = {f"test ({u}% unseen)": data_io.load_wdc_products("pair", 80, "gs", unseen=u) for u in (0, 50, 100)}
    feats = {k: wdc_pair_features(v, model) for k, v in tests.items()}
    for cname, kw in {"char TF-IDF only": dict(grid_w=(0.0,), use_rules=False),
                      "SBERT only": dict(grid_w=(1.0,), use_rules=False),
                      "hybrid (no rules)": dict(use_rules=False),
                      "hybrid + all rules": dict()}.items():
        p = tune(fv, lv, **kw)
        params[cname] = p
        for tname, t in tests.items():
            s = hybrid_score(feats[tname], p["w"], p["brand_penalty"], p["model_penalty"],
                             use_vetoes=kw.get("use_rules", True))
            pred = set(np.where(s >= p["threshold"])[0])
            gold = set(np.where(t["label"].values == 1)[0])
            rows.append({"dataset": f"WDC 80% cc {tname}", "config": cname, **prf(pred, gold),
                         "val_f1": round(p["f1"], 3), "w_sbert": p["w"], "brand_penalty": p["brand_penalty"],
                         "model_penalty": p["model_penalty"], "threshold": round(p["threshold"], 3)})
    return pd.DataFrame(rows), params


def similarity_matrix(texts_raw: list[str], model, w: float, brand_penalty: float, model_penalty: float,
                      use_rules: bool = True) -> np.ndarray:
    """Full n x n hybrid similarity for clustering, computed with matrix products:
    SBERT cosine = E E^T, char TF-IDF cosine = T T^T, rules as boolean matrices."""
    texts = [normalize_text(t) for t in texts_raw]
    attr = attributes(pd.DataFrame({"raw": texts_raw}))
    E = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    T = normalize(TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit_transform(texts))
    S = w * (E @ E.T) + (1 - w) * (T @ T.T).toarray()
    if use_rules:
        b = attr["brand"].values
        known = pd.notna(b)
        brand_conf = known[:, None] & known[None, :] & (b[:, None] != b[None, :])
        codes = attr["codes"].tolist()
        n = len(texts)
        code_conf = np.zeros((n, n), dtype=bool)
        has = [k for k in range(n) if codes[k]]
        for a, c in itertools.combinations(has, 2):
            if codes_conflict(codes[a], codes[c]):
                code_conf[a, c] = code_conf[c, a] = True
        q, u, pk = attr["qty_value"].values.astype(float), attr["qty_unit"].values, attr["pack_count"].values.astype(float)
        hq = ~np.isnan(q)
        rel = np.abs(q[:, None] - q[None, :]) / np.maximum(np.abs(q[:, None]), 1e-9)
        qty_conf = hq[:, None] & hq[None, :] & (u[:, None] == u[None, :]) & (rel > 0.02)
        hp = ~np.isnan(pk)
        pack_conf = hp[:, None] & hp[None, :] & (pk[:, None] != pk[None, :])
        S = S - brand_penalty * brand_conf - model_penalty * code_conf
        S = np.where(qty_conf | pack_conf, 0.0, S)
    np.fill_diagonal(S, 1.0)
    return S


def cluster_components(S: np.ndarray, threshold: float) -> np.ndarray:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    A = csr_matrix(S >= threshold)
    return connected_components(A, directed=False)[1]


def cluster_agglomerative(S: np.ndarray, threshold: float) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering
    D = np.clip(1 - S, 0, None)
    np.fill_diagonal(D, 0)
    return AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average",
                                   distance_threshold=1 - threshold).fit_predict(D)


def cluster_hdbscan(emb: np.ndarray, min_cluster_size: int = 2) -> np.ndarray:
    """HDBSCAN on normalized SBERT vectors; noise points (-1) become singleton clusters."""
    from sklearn.cluster import HDBSCAN
    lab = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=1).fit_predict(emb)
    nxt = lab.max() + 1
    out = lab.copy()
    for k in np.where(lab == -1)[0]:
        out[k] = nxt
        nxt += 1
    return out


def cluster_scores(gold, pred) -> dict:
    return {"ARI": round(adjusted_rand_score(gold, pred), 3), "NMI": round(normalized_mutual_info_score(gold, pred), 3),
            "V-measure": round(v_measure_score(gold, pred), 3), "clusters": int(len(set(pred)))}


def run_wdc_clustering(model, params: dict) -> tuple[pd.DataFrame, dict]:
    """Thresholds for components/agglomerative are tuned (by ARI) on the multi-class validation set."""
    val = data_io.load_wdc_products("multi", 80, "valid", "large")
    test = data_io.load_wdc_products("multi", 80, "gs")
    w, bp, mp = params["w"], params["brand_penalty"], params["model_penalty"]
    Sv = similarity_matrix(val["title"].fillna("").tolist(), model, w, bp, mp)
    St = similarity_matrix(test["title"].fillna("").tolist(), model, w, bp, mp)
    rows, chosen = [], {}
    grid = np.round(np.arange(0.40, 0.96, 0.025), 3)
    for meth, fn in (("connected components", cluster_components), ("agglomerative (average)", cluster_agglomerative)):
        best_t = max(grid, key=lambda t: adjusted_rand_score(val["label"], fn(Sv, t)))
        chosen[meth] = float(best_t)
        rows.append({"method": meth, "threshold": float(best_t), **cluster_scores(test["label"], fn(St, best_t))})
    emb = model.encode([normalize_text(t) for t in test["title"].fillna("")], normalize_embeddings=True,
                       show_progress_bar=False)
    rows.append({"method": "HDBSCAN (SBERT vectors)", "threshold": None,
                 **cluster_scores(test["label"], cluster_hdbscan(emb))})
    rows.append({"method": "baseline: every offer alone", "threshold": None,
                 **cluster_scores(test["label"], np.arange(len(test)))})
    return pd.DataFrame(rows), {"thresholds": chosen, "test_offers": len(test), "test_products": int(test["label"].nunique())}


def umap_plot(model, path) -> None:
    import matplotlib.pyplot as plt
    import umap
    test = data_io.load_wdc_products("multi", 80, "gs")
    top = test["label"].value_counts().head(15).index
    sub = test[test["label"].isin(top)]
    emb = model.encode([normalize_text(t) for t in test["title"].fillna("")], normalize_embeddings=True,
                       show_progress_bar=False)
    xy = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=SEED).fit_transform(emb)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(xy[:, 0], xy[:, 1], s=6, c="#cccccc", label="other products")
    cmap = plt.get_cmap("tab20")
    for k, lab in enumerate(top):
        m = (test["label"] == lab).values
        ax.scatter(xy[m, 0], xy[m, 1], s=18, color=cmap(k), label=str(test.loc[m, "title"].iloc[0])[:30])
    ax.set_title("WDC Products (80% corner cases, test): SBERT embeddings, UMAP 2-D\n15 largest products coloured")
    ax.legend(fontsize=6, loc="best", markerscale=1.5)
    ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=130)


# ---------------------------------------------------------------- BigBasket (working data)
BB_PROCESSED = ROOT / "data" / "processed"
BB_GOLD = ROOT / "data" / "gold" / "bigbasket_pairs.csv"


def bigbasket_table() -> pd.DataFrame:
    bb = pd.read_parquet(BB_PROCESSED / "bigbasket_pre.parquet").reset_index(drop=True)
    bb["text"] = (bb["brand"].fillna("").map(normalize_text) + " " + bb["name_clean"]).str.strip()
    return bb


def bigbasket_features(bb: pd.DataFrame, pairs: np.ndarray, E: np.ndarray, T) -> pd.DataFrame:
    """Features for BigBasket pairs (i, j). Brand comes from the brand column (reliable here);
    quantity and pack count from Phase 2; MRP from Phase 1."""
    i, j = pairs[:, 0], pairs[:, 1]
    f = pd.DataFrame({"i": i, "j": j})
    f["sbert"] = np.einsum("ij,ij->i", E[i], E[j])
    char = np.empty(len(pairs))
    for s in range(0, len(pairs), 50_000):          # chunks keep the sparse products small
        a, b = T[i[s:s + 50_000]], T[j[s:s + 50_000]]
        char[s:s + 50_000] = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    f["char"] = char
    q, u, pk = bb["qty_value"].values, bb["qty_unit"].values, bb["pack_count"].values
    both_q = ~pd.isna(q[i]) & ~pd.isna(q[j]) & (u[i] == u[j])
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.abs(q[i].astype(float) - q[j].astype(float)) / np.maximum(np.abs(q[i].astype(float)), 1e-9)
    f["qty_conflict"] = both_q & (rel > 0.02)
    f["pack_conflict"] = ~pd.isna(pk[i]) & ~pd.isna(pk[j]) & (pk[i] != pk[j])
    b = bb["brand"].fillna("").str.lower().values
    f["brand_conflict"] = (b[i] != "") & (b[j] != "") & (b[i] != b[j])
    f["code_conflict"] = False
    mrp = bb["mrp"].values.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        f["mrp_conflict"] = np.abs(mrp[i] - mrp[j]) / np.maximum(mrp[i], mrp[j]) > 0.05
    return f


def bigbasket_score(f: pd.DataFrame, w: float, use_rules: bool = True, use_mrp: bool = True) -> np.ndarray:
    """Brand is a veto on BigBasket (the brand column is reliable); MRP differing by > 5 % is a veto
    when use_mrp (same name at a different MRP is usually a different pack size, see Phase 0)."""
    s = w * f["sbert"].values + (1 - w) * f["char"].values
    if use_rules:
        veto = f["qty_conflict"].values | f["pack_conflict"].values | f["brand_conflict"].values
        if use_mrp:
            veto = veto | f["mrp_conflict"].values
        s = np.where(veto, 0.0, s)
    return s


def bigbasket_candidates(k: int = 20) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    bb = bigbasket_table()
    model = get_sbert(SBERT)
    E = model.encode(bb["text"].tolist(), batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    T = normalize(TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
                  .fit_transform(bb["text"]))
    idx, _ = cosine_topk(E, E, k + 1)
    pairs = {(min(a, int(b)), max(a, int(b))) for a, row in enumerate(idx) for b in row if a != b}
    pairs = np.array(sorted(pairs))
    np.save(BB_PROCESSED / "bigbasket_embeddings.npy", E)
    return bb, pairs, bigbasket_features(bb, pairs, E, T)


def sample_gold_pairs(bb: pd.DataFrame, f: pd.DataFrame, n: int = 300) -> pd.DataFrame:
    """300 same-brand candidate pairs, stratified over 5 hybrid-score bands (60 each), for hand labelling."""
    s = 0.5 * f["sbert"] + 0.5 * f["char"]
    same_brand = ~f["brand_conflict"]
    bands = [(0.95, 1.01), (0.85, 0.95), (0.75, 0.85), (0.65, 0.75), (0.0, 0.65)]
    picks = []
    for lo, hi in bands:
        pool = f[same_brand & (s >= lo) & (s < hi)]
        picks.append(pool.sample(min(n // len(bands), len(pool)), random_state=SEED))
    p = pd.concat(picks)
    cols = ["brand", "product_name", "mrp", "price", "sub_category"]
    a = bb.loc[p["i"].values, cols].reset_index(drop=True).add_suffix("_a")
    b = bb.loc[p["j"].values, cols].reset_index(drop=True).add_suffix("_b")
    out = pd.concat([p[["i", "j"]].reset_index(drop=True), a, b], axis=1)
    out["desc_a"] = bb.loc[p["i"].values, "description"].fillna("").str[:120].values
    out["desc_b"] = bb.loc[p["j"].values, "description"].fillna("").str[:120].values
    return out.sample(frac=1, random_state=SEED).reset_index(drop=True)   # shuffle so bands are hidden


def evaluate_bigbasket(f: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    """Tune on half of the labelled pairs, report on the other half."""
    key = pd.MultiIndex.from_arrays([f["i"], f["j"]])
    g = gold.set_index(["i", "j"])
    feats = f.set_index(["i", "j"]).loc[g.index].reset_index()
    y = g["label"].values.astype(bool)
    rng = np.random.default_rng(SEED)
    is_val = rng.random(len(y)) < 0.5
    rows = []
    configs = {"char TF-IDF only": (0.0, False, False), "SBERT only": (1.0, False, False),
               "hybrid (no rules)": (None, False, False), "hybrid + qty/pack/brand rules": (None, True, False),
               "hybrid + rules + MRP rule": (None, True, True)}
    for name, (w_fixed, rules, mrp) in configs.items():
        best = (-1, None, None)
        for w in ([w_fixed] if w_fixed is not None else [0.0, 0.25, 0.5, 0.75, 1.0]):
            s = bigbasket_score(feats, w, rules, mrp)
            for t in np.unique(s[is_val]):
                pred = s[is_val] >= t
                f1 = prf(set(np.where(pred)[0]), set(np.where(y[is_val])[0]))["f1"]
                if f1 > best[0]:
                    best = (f1, w, t)
        _, w, t = best
        s = bigbasket_score(feats, w, rules, mrp)
        test = ~is_val
        m = prf(set(np.where(s[test] >= t)[0]), set(np.where(y[test])[0]))
        rows.append({"config": name, **m, "val_f1": round(best[0], 3), "w_sbert": w, "threshold": round(float(t), 3),
                     "test_pairs": int(test.sum()), "test_positives": int(y[test].sum())})
    return pd.DataFrame(rows)


def cluster_bigbasket(bb: pd.DataFrame, f: pd.DataFrame, w: float, threshold: float, use_mrp: bool,
                      max_size: int = 30) -> pd.Series:
    """Connected components of the match graph; components larger than max_size (chaining) are
    re-split with average-linkage agglomerative clustering on their own pairs."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    s = bigbasket_score(f, w, True, use_mrp)
    keep = s >= threshold
    n = len(bb)
    A = csr_matrix((np.ones(keep.sum()), (f["i"].values[keep], f["j"].values[keep])), shape=(n, n))
    _, comp = connected_components(A, directed=False)
    comp = comp.copy()
    sizes = pd.Series(comp).value_counts()
    nxt = comp.max() + 1
    for c in sizes[sizes > max_size].index:
        members = np.where(comp == c)[0]
        pos = {m: k for k, m in enumerate(members)}
        S = np.zeros((len(members), len(members)))
        sub = f[np.isin(f["i"], members) & np.isin(f["j"], members)]
        ss = bigbasket_score(sub, w, True, use_mrp)
        for (a, b), v in zip(sub[["i", "j"]].values, ss):
            S[pos[a], pos[b]] = S[pos[b], pos[a]] = v
        np.fill_diagonal(S, 1)
        lab = cluster_agglomerative(S, threshold)
        comp[members] = lab + nxt
        nxt += lab.max() + 1
    return pd.Series(pd.factorize(comp)[0], name="cluster_id")


if __name__ == "__main__":
    import sys
    pd.set_option("display.width", 220)
    step = sys.argv[1] if len(sys.argv) > 1 else "benchmarks"
    if step == "bb_sample":
        bb, pairs, f = bigbasket_candidates()
        f.to_parquet(BB_PROCESSED / "bigbasket_candidates.parquet")
        print("BigBasket candidate pairs:", len(pairs))
        sample_gold_pairs(bb, f).to_csv(ROOT / "data" / "gold" / "bigbasket_pairs_to_label.csv", index=False)
        sys.exit()
    if step == "bb_eval":
        bb = bigbasket_table()
        f = pd.read_parquet(BB_PROCESSED / "bigbasket_candidates.parquet")
        gold = pd.read_csv(BB_GOLD)
        res = evaluate_bigbasket(f, gold)
        print(res.to_string(index=False))
        res.to_csv(OUT / "phase5_bigbasket_pairs.csv", index=False)
        best = res.iloc[res["val_f1"].idxmax()]
        use_mrp = "MRP" in best["config"]
        cid = cluster_bigbasket(bb, f, best["w_sbert"], best["threshold"], use_mrp)
        bb["cluster_id"] = cid.values
        bb.to_parquet(BB_PROCESSED / "bigbasket_clusters.parquet", index=False)
        sizes = bb["cluster_id"].value_counts()
        stats = {"config_used": best["config"], "records": len(bb), "clusters": int(len(sizes)),
                 "multi_member_clusters": int((sizes > 1).sum()), "records_in_multi": int(sizes[sizes > 1].sum()),
                 "largest_cluster": int(sizes.max())}
        print(stats)
        (OUT / "phase5_bigbasket_clusters.json").write_text(json.dumps(stats, indent=1))
        sys.exit()

    results, meta = [], {}
    for name in ("abt_buy", "amazon_google"):
        r, m = run_benchmark(name)
        results.append(r)
        meta[name] = m
        print(r.to_string(index=False), "\nblocking:", m["blocking"], flush=True)
    model = get_sbert(SBERT)
    wdc, wdc_params = run_wdc_pairs(model)
    meta["wdc_pairs"] = wdc_params
    print(wdc.to_string(index=False), flush=True)
    pairs_table = pd.concat(results + [wdc], ignore_index=True)
    pairs_table.to_csv(OUT / "phase5_pair_matching.csv", index=False)

    clus, cmeta = run_wdc_clustering(model, wdc_params["hybrid + all rules"])
    print(clus.to_string(index=False), cmeta, flush=True)
    clus.to_csv(OUT / "phase5_clustering.csv", index=False)
    meta["clustering"] = cmeta
    (OUT / "phase5_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    umap_plot(model, OUT / "figures" / "phase5_umap_wdc.png")
