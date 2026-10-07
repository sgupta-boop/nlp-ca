"""Phase 8: evaluation and ablation - one honest comparison of every stage.

Six configurations, the same test sets and metrics for all:
  1 rules + fuzzy          rapidfuzz token_set_ratio + attribute rules (vetoes/penalties)
  2 TF-IDF cosine          char_wb 3-5 grams
  3 fastText cosine        gensim FastText trained on product names (Phase 4)
  4 SBERT cosine           all-MiniLM-L6-v2
  5 SBERT + NER rules      hybrid SBERT + char TF-IDF, attribute rules (Phase 5)
  6 fine-tuned SBERT + NER + LLM   as 5 with the Phase 7 model, plus LLM grey-zone adjudication

Test sets:
  Abt-Buy, Amazon-Google  candidate pairs from the fixed Phase 5 blocking; test half of the left records
                          (gold pairs lost by blocking are false negatives for every configuration)
  WDC pairs               80 % corner cases, 0 % and 100 % unseen products; tuned on WDC validation
  WDC clustering          multi-class test (agglomerative clustering on each configuration's similarity)
  BigBasket               300 labelled pairs (provisional labels), tuned on half, tested on half

Every configuration's weight/threshold is tuned on the same validation data.

Run:  python -m src.evaluate prepare | llm | report | cost
"""
import json
import sys
import time

import numpy as np
import pandas as pd

from src import data_io, llm
from src.embed import get_fasttext, fasttext_vectors, get_sbert, load_benchmark
from src.match import (SBERT, PairScorer, attributes, cluster_agglomerative, cluster_scores, hybrid_score, prf,
                       similarity_matrix, split_ids, tune)
from src.preprocess import normalize_text

ROOT = data_io.ROOT
SEED = 42
OUT = ROOT / "reports"
P8 = ROOT / "data" / "processed" / "phase8"
FT_MODEL = ROOT / "models" / "minilm-products"
CONFIGS = ["1 rules + fuzzy", "2 TF-IDF cosine", "3 fastText cosine", "4 SBERT cosine", "5 SBERT + NER rules",
           "6 fine-tuned SBERT + NER + LLM"]
GREY = 0.10   # grey zone = threshold +/- 0.10


def pair_features(a_raw: list[str], b_raw: list[str], a_brand=None, b_brand=None) -> pd.DataFrame:
    """All similarity columns + rule conflicts for a list of (a, b) pairs."""
    from rapidfuzz import fuzz
    from sentence_transformers import SentenceTransformer
    a_txt = [normalize_text(t) for t in a_raw]
    b_txt = [normalize_text(t) for t in b_raw]
    da = pd.DataFrame({"raw": a_raw, "brand": a_brand}) if a_brand is not None else pd.DataFrame({"raw": a_raw})
    db = pd.DataFrame({"raw": b_raw, "brand": b_brand}) if b_brand is not None else pd.DataFrame({"raw": b_raw})
    la = attributes(da, brand_col="brand" if a_brand is not None else None)
    ra = attributes(db, brand_col="brand" if b_brand is not None else None)
    idx = np.arange(len(a_raw))
    pairs = np.stack([idx, idx], axis=1)
    base = PairScorer(a_txt, b_txt, la, ra, sbert_model=get_sbert(SBERT)).features(pairs)
    ft = PairScorer(a_txt, b_txt, la, ra, sbert_model=SentenceTransformer(str(FT_MODEL))).features(pairs)
    f = base.rename(columns={"sbert": "sbert_base"})
    f["sbert_ft"] = ft["sbert"].values
    f["fuzzy"] = [fuzz.token_set_ratio(x, y) / 100 for x, y in zip(a_txt, b_txt)]
    m = get_fasttext()
    f["fasttext"] = (fasttext_vectors(m, a_txt) * fasttext_vectors(m, b_txt)).sum(1)
    f["name_a"], f["name_b"] = a_raw, b_raw
    return f


def config_score(f: pd.DataFrame, config: str, p: dict) -> np.ndarray:
    def with_sim(col):
        g = f.copy()
        g["sbert"] = f[col]
        return g
    if config.startswith("1"):
        return hybrid_score(with_sim("fuzzy").assign(char=f["fuzzy"]), 1.0, p["brand_penalty"], p["model_penalty"])
    if config.startswith("2"):
        return f["char"].values
    if config.startswith("3"):
        return f["fasttext"].values
    if config.startswith("4"):
        return f["sbert_base"].values
    if config.startswith("5"):
        return hybrid_score(with_sim("sbert_base"), p["w"], p["brand_penalty"], p["model_penalty"])
    return hybrid_score(with_sim("sbert_ft"), p["w"], p["brand_penalty"], p["model_penalty"])


def tune_config(f: pd.DataFrame, y: np.ndarray, config: str, extra_missing: int = 0) -> dict:
    """Same grid search for every configuration (rules-based ones also tune their penalties)."""
    g = f.copy()
    if config.startswith(("2", "3", "4")):
        g["sbert"] = f[{"2": "char", "3": "fasttext", "4": "sbert_base"}[config[0]]]
        return tune(g, y, grid_w=(1.0,), use_rules=False, extra_gold_missing=extra_missing)
    if config.startswith("1"):
        g["sbert"] = f["fuzzy"]
        return tune(g, y, grid_w=(1.0,), extra_gold_missing=extra_missing)
    g["sbert"] = f["sbert_base" if config.startswith("5") else "sbert_ft"]
    return tune(g, y, extra_gold_missing=extra_missing)


# ---------------------------------------------------------------- data preparation (no LLM)
def prepare() -> None:
    P8.mkdir(parents=True, exist_ok=True)
    for name in ("abt_buy", "amazon_google"):
        c = pd.read_parquet(ROOT / "data" / "processed" / f"phase5_{name}_candidates.parquet")
        left, right, gold = load_benchmark(name)
        ln, rn = dict(zip(left["id"], left["raw"])), dict(zip(right["id"], right["raw"]))
        f = pair_features([ln[i] for i in c["left_id"]], [rn[i] for i in c["right_id"]])
        f["label"], f["is_val"] = c["label"].values, c["is_val"].values
        f["left_id"], f["right_id"] = c["left_id"].values, c["right_id"].values
        val_left = set(c.loc[c["is_val"], "left_id"])
        test_left = set(left["id"]) - val_left
        f.attrs = {}
        f.to_parquet(P8 / f"{name}.parquet")
        json.dump({"gold_val": sum(len(v) for k, v in gold.items() if k in val_left),
                   "gold_test": sum(len(v) for k, v in gold.items() if k in test_left)},
                  open(P8 / f"{name}_gold.json", "w"))
        print(name, len(f), flush=True)
    for part, df in (("wdc_val", data_io.load_wdc_products("pair", 80, "valid", "large")),
                     ("wdc_test0", data_io.load_wdc_products("pair", 80, "gs", unseen=0)),
                     ("wdc_test100", data_io.load_wdc_products("pair", 80, "gs", unseen=100))):
        f = pair_features(df["title_left"].fillna("").tolist(), df["title_right"].fillna("").tolist())
        f["label"] = df["label"].values.astype(bool)
        f.to_parquet(P8 / f"{part}.parquet")
        print(part, len(f), flush=True)
    bb = pd.read_parquet(ROOT / "data" / "processed" / "bigbasket_pre.parquet").reset_index(drop=True)
    gold = pd.read_csv(ROOT / "data" / "gold" / "bigbasket_pairs.csv")
    f = pair_features(bb.loc[gold["i"], "product_name"].tolist(), bb.loc[gold["j"], "product_name"].tolist(),
                      bb.loc[gold["i"], "brand"].tolist(), bb.loc[gold["j"], "brand"].tolist())
    f["label"] = gold["label"].values.astype(bool)
    f["is_val"] = np.random.default_rng(SEED).random(len(f)) < 0.5
    f.to_parquet(P8 / "bigbasket.parquet")


def decide(f_val, y_val, f_test, config, extra_missing=0):
    p = tune_config(f_val, y_val, config, extra_missing)
    s = config_score(f_test, config, p)
    return s, p


# ---------------------------------------------------------------- LLM step (grey zone of config 6)
def llm_step() -> None:
    """Ask the LLM about config-6 grey-zone pairs on every test set (cached, resumable)."""
    from src.llm_layer import adjudicate
    for name, val, test in datasets():
        f_val, y_val, f_test, _, extra = val_test(name, val, test)
        s, p = decide(f_val, y_val, f_test, CONFIGS[5], extra)
        grey = np.where(np.abs(s - p["threshold"]) < GREY)[0]
        answers = {}
        for k in grey:
            r = adjudicate(f_test["name_a"].iat[k], f_test["name_b"].iat[k])
            answers[int(k)] = None if r is None else bool(r.same)
        json.dump(answers, open(P8 / f"llm_{name}.json", "w"))
        print(name, "grey pairs", len(grey), llm.report(), flush=True)
    llm.unload()


def datasets():
    return [("Abt-Buy", "abt_buy", None), ("Amazon-Google", "amazon_google", None),
            ("WDC 0% unseen", "wdc_val", "wdc_test0"), ("WDC 100% unseen", "wdc_val", "wdc_test100"),
            ("BigBasket 300 pairs", "bigbasket", None)]


def val_test(name, val, test):
    if test is None:
        f = pd.read_parquet(P8 / f"{val}.parquet").reset_index(drop=True)
        fv, ft = f[f["is_val"]].reset_index(drop=True), f[~f["is_val"]].reset_index(drop=True)
        extra, gold_test = 0, None
        if (P8 / f"{val}_gold.json").exists():
            g = json.load(open(P8 / f"{val}_gold.json"))
            extra = g["gold_val"] - int(fv["label"].sum())
            gold_test = g["gold_test"]
        return fv, fv["label"].values.astype(bool), ft, gold_test, extra
    fv = pd.read_parquet(P8 / f"{val}.parquet")
    ft = pd.read_parquet(P8 / f"{test}.parquet")
    return fv, fv["label"].values.astype(bool), ft, None, 0


# ---------------------------------------------------------------- the ablation table
def report() -> pd.DataFrame:
    rows, errors = [], []
    for name, val, test in datasets():
        f_val, y_val, f_test, gold_test, extra = val_test(name, val, test)
        y = f_test["label"].values.astype(bool)
        n_gold = gold_test if gold_test is not None else int(y.sum())
        answers = json.load(open(P8 / f"llm_{name}.json")) if (P8 / f"llm_{name}.json").exists() else {}
        for config in CONFIGS:
            s, p = decide(f_val, y_val, f_test, config, extra)
            pred = s >= p["threshold"]
            calls = 0
            if config.startswith("6"):
                for k, v in answers.items():
                    calls += 1
                    if v is not None:
                        pred[int(k)] = v
            tp = int((pred & y).sum())
            prec = tp / pred.sum() if pred.sum() else 0.0
            rec = tp / n_gold if n_gold else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            rows.append({"test set": name, "config": config, "precision": round(prec, 3), "recall": round(rec, 3),
                         "f1": round(f1, 3), "llm_calls": calls})
            if config.startswith("6"):
                wrong = np.where(pred != y)[0]
                for k in wrong[:40]:
                    errors.append({"test set": name, "name_a": f_test["name_a"].iat[k], "name_b": f_test["name_b"].iat[k],
                                   "gold": bool(y[k]), "predicted": bool(pred[k]), "score": round(float(s[k]), 3),
                                   "sbert_ft": round(float(f_test["sbert_ft"].iat[k]), 3),
                                   "char": round(float(f_test["char"].iat[k]), 3),
                                   "rules": ",".join(c for c in ("qty_conflict", "pack_conflict", "brand_conflict",
                                                                 "code_conflict") if f_test[c].iat[k]),
                                   "llm_used": str(k) in answers})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "phase8_ablation_long.csv", index=False)
    wide = table.pivot(index="config", columns="test set", values="f1")
    wide.to_csv(OUT / "phase8_ablation.csv")
    pd.DataFrame(errors).to_csv(OUT / "phase8_errors_pool.csv", index=False)
    print(wide.to_string())
    return wide


def clustering_ablation() -> pd.DataFrame:
    """WDC multi-class test: agglomerative clustering on each configuration's pairwise similarity."""
    from rapidfuzz import fuzz, process
    from sentence_transformers import SentenceTransformer
    val = data_io.load_wdc_products("multi", 80, "valid", "large")
    test = data_io.load_wdc_products("multi", 80, "gs")
    meta = json.loads((OUT / "phase5_meta.json").read_text())["wdc_pairs"]
    grid = np.round(np.arange(0.30, 0.96, 0.025), 3)
    rows = []

    def sims(df, config):
        raw = df["title"].fillna("").tolist()
        txt = [normalize_text(t) for t in raw]
        if config.startswith("1"):
            S = process.cdist(txt, txt, scorer=fuzz.token_set_ratio, workers=1) / 100
            return S
        if config.startswith("2"):
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.preprocessing import normalize
            T = normalize(TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit_transform(txt))
            return (T @ T.T).toarray()
        if config.startswith("3"):
            V = fasttext_vectors(get_fasttext(), txt)
            return V @ V.T
        if config.startswith("4"):
            E = get_sbert(SBERT).encode(txt, normalize_embeddings=True, show_progress_bar=False)
            return E @ E.T
        p = meta["hybrid + all rules"]
        model = get_sbert(SBERT) if config.startswith("5") else SentenceTransformer(str(FT_MODEL))
        return similarity_matrix(raw, model, p["w"], p["brand_penalty"], p["model_penalty"])

    for config in CONFIGS:
        Sv, St = sims(val, config), sims(test, config)
        t = max(grid, key=lambda t: cluster_scores(val["label"], cluster_agglomerative(Sv, t))["ARI"])
        rows.append({"config": config, "threshold": float(t), **cluster_scores(test["label"], cluster_agglomerative(St, t))})
        print(rows[-1], flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "phase8_clustering.csv", index=False)
    return res


# ---------------------------------------------------------------- cost: full pipeline on 1,000 BigBasket products
def cost(n: int = 1000) -> dict:
    """Times every stage of the system on n BigBasket products, including the LLM for grey-zone pairs."""
    from src import columns, preprocess
    from src.match import bigbasket_features, bigbasket_score
    from src.embed import cosine_topk
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    times = {}
    raw = data_io.load_bigbasket(keep_index=True).sample(n, random_state=SEED).reset_index(drop=True)
    t = time.perf_counter()
    headers = list(raw.columns)
    mapping = columns.resolve_price_conflict(columns.map_columns(headers, [columns.sample_values(raw[h]) for h in headers]), raw)
    canon = columns.to_canonical(raw, mapping, "bigbasket")
    times["1 column mapping"] = time.perf_counter() - t
    t = time.perf_counter()
    corr, _ = preprocess.build_corrector()
    pre = preprocess.preprocess_table(canon, corr).reset_index(drop=True)
    times["2 preprocessing"] = time.perf_counter() - t
    t = time.perf_counter()
    from src.extract import extract_rules
    extract_rules((pre["brand"].fillna("") + " " + pre["product_name"]).tolist())
    times["3 attribute extraction (rules)"] = time.perf_counter() - t
    t = time.perf_counter()
    pre["text"] = (pre["brand"].fillna("").map(normalize_text) + " " + pre["name_clean"]).str.strip()
    from sentence_transformers import SentenceTransformer
    E = SentenceTransformer(str(FT_MODEL)).encode(pre["text"].tolist(), normalize_embeddings=True, show_progress_bar=False)
    T = normalize(TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit_transform(pre["text"]))
    times["4 embeddings"] = time.perf_counter() - t
    t = time.perf_counter()
    idx, _ = cosine_topk(E, E, 21)
    pairs = np.array(sorted({(min(a, int(b)), max(a, int(b))) for a, row in enumerate(idx) for b in row if a != b}))
    f = bigbasket_features(pre, pairs, E, T)
    res = pd.read_csv(OUT / "phase5_bigbasket_pairs.csv")
    best = res.iloc[res["val_f1"].idxmax()]
    s = bigbasket_score(f, best["w_sbert"], True, "MRP" in best["config"])
    times["5 blocking + hybrid scoring"] = time.perf_counter() - t
    grey = np.where(np.abs(s - best["threshold"]) < GREY)[0]
    del E
    import gc
    gc.collect()
    from src.llm_layer import adjudicate
    names = (pre["brand"].fillna("") + " " + pre["product_name"]).tolist()
    t = time.perf_counter()
    before = llm.STATS["calls"]
    for k in grey:
        adjudicate(names[f["i"].iat[k]], names[f["j"].iat[k]])
    times["6 LLM grey-zone adjudication"] = time.perf_counter() - t
    out = {"products": n, "candidate_pairs": len(pairs), "grey_zone_pairs": int(len(grey)),
           "llm_calls": llm.STATS["calls"] - before, "llm_cache_hits": llm.STATS["cache_hits"],
           "seconds": {k: round(v, 1) for k, v in times.items()},
           "total_seconds": round(sum(times.values()), 1)}
    llm.unload()
    json.dump(out, open(OUT / "phase8_cost.json", "w"), indent=1)
    print(out)
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    {"prepare": prepare, "llm": llm_step, "report": report, "clustering": clustering_ablation,
     "cost": cost}[sys.argv[1]]()
