"""Phase 7: domain adaptation - fine-tuning Sentence-BERT and Hinglish robustness.

1. Fine-tune all-MiniLM-L6-v2 with MultipleNegativesRankingLoss (in-batch negatives) on
     - WDC Products 80 % train-large positive pairs           (anchor, positive)
     - Phase 6 synthetic data                                (anchor, noisy variant, hard negative)
   and a second model that also sees Hinglish augmentation  (English name, Hinglish variant)
2. Evaluate against the base model on WDC test pairs (0 / 50 / 100 % unseen products) and Abt-Buy retrieval.
3. Hinglish robustness: 200 BigBasket names rewritten by the LLM in casual English and in Hinglish
   (data/gold/hinglish_test.csv). Each query must retrieve its original among all 27k BigBasket names.
   English vs multilingual models (LaBSE, multilingual-E5), plus Aksharantar-based
   transliteration: spelling variants (chawal / chaawal) and back-transliteration to Devanagari.

Run:  python -m src.finetune train | evaluate
"""
import json
import random
import re
import sys
import time

import numpy as np
import pandas as pd

from src import data_io
from src.embed import cosine_topk, evaluate_retrieval, retrieval_metrics, SBERT_MODELS
from src.preprocess import normalize_text

ROOT = data_io.ROOT
SEED = 42
MODELS = ROOT / "models"
OUT = ROOT / "reports"
BASE = "sentence-transformers/all-MiniLM-L6-v2"
FT = MODELS / "minilm-products"
FT_HI = MODELS / "minilm-products-hinglish"
MULTILINGUAL = {"LaBSE": "sentence-transformers/LaBSE",
                "multilingual-E5-small": "intfloat/multilingual-e5-small"}

# Small hand-written English -> Hindi (Roman script) grocery dictionary, used ONLY to build training
# augmentation for names that are disjoint from the Hinglish test set.
HINDI_WORDS = {
    "turmeric": "haldi", "rice": "chawal", "lentils": "dal", "lentil": "dal", "sugar": "cheeni",
    "salt": "namak", "flour": "atta", "wheat": "gehun", "oil": "tel", "tea": "chai", "cumin": "jeera",
    "coriander": "dhaniya", "chilli": "mirch", "chili": "mirch", "onion": "pyaz", "potato": "aloo",
    "tomato": "tamatar", "garlic": "lahsun", "ginger": "adrak", "milk": "doodh", "curd": "dahi",
    "butter": "makhan", "cottage cheese": "paneer", "chickpea": "chana", "chickpeas": "chana",
    "gram": "chana", "mustard": "sarson", "fenugreek": "methi", "cardamom": "elaichi", "clove": "laung",
    "cloves": "laung", "cinnamon": "dalchini", "pepper": "kali mirch", "jaggery": "gud", "soap": "sabun",
    "spinach": "palak", "peas": "matar", "semolina": "suji", "pickle": "achaar", "honey": "shahad",
    "almonds": "badam", "almond": "badam", "cashew": "kaju", "raisins": "kishmish", "spices": "masale",
    "biscuits": "biscuit", "water": "pani", "sweet": "mithai", "vegetables": "sabzi", "fish": "machli",
    "egg": "anda", "eggs": "ande", "chicken": "murgi",
}


# ---------------------------------------------------------------- Aksharantar
def aksharantar_maps(vocab: set[str]) -> tuple[dict, dict]:
    """Stream the 1.3 M-pair Hindi file once (low memory).
    roman_to_native: romanized word -> most frequent Devanagari spelling (for back-transliteration)
    variants:        romanized word -> other romanizations of the same Devanagari word (for augmentation)"""
    from collections import Counter, defaultdict
    path = data_io.RAW / "aksharantar_hin" / "hin_train.json"
    roman_native = defaultdict(Counter)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["english word"] in vocab:
                roman_native[r["english word"]][r["native word"]] += 1
    natives = {n for c in roman_native.values() for n in c}
    native_romans = defaultdict(set)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["native word"] in natives:
                native_romans[r["native word"]].add(r["english word"])
    roman_to_native = {w: c.most_common(1)[0][0] for w, c in roman_native.items()}
    variants = {w: sorted(native_romans[roman_to_native[w]] - {w}) for w in roman_to_native}
    return roman_to_native, variants


def back_transliterate(text: str, roman_to_native: dict) -> str:
    return " ".join(roman_to_native.get(t, t) for t in text.split())


def spelling_variant(text: str, variants: dict, rng: random.Random) -> str:
    return " ".join(rng.choice(variants[t]) if variants.get(t) else t for t in text.split())


def to_hinglish(name: str) -> str | None:
    """Replace English grocery words by their Hindi words; None if nothing was replaced."""
    t = name.lower()
    changed = False
    for en, hi in sorted(HINDI_WORDS.items(), key=lambda kv: -len(kv[0])):
        new = re.sub(rf"\b{re.escape(en)}\b", hi, t)
        changed |= new != t
        t = new
    return t if changed else None


# ---------------------------------------------------------------- training data
def training_data(hinglish_aug: bool = False) -> dict:
    from datasets import Dataset
    wdc = data_io.load_wdc_products("pair", 80, "train", "large")
    pos = wdc[wdc["label"] == 1]
    data = {"wdc": Dataset.from_dict({"anchor": pos["title_left"].fillna("").map(normalize_text).tolist(),
                                      "positive": pos["title_right"].fillna("").map(normalize_text).tolist()})}
    syn = pd.read_json(ROOT / "data" / "processed" / "synthetic_pairs.json")
    a, p, n = [], [], []
    for _, r in syn.iterrows():
        for v in r["variants"]:
            a.append(normalize_text(r["anchor"]))
            p.append(normalize_text(v))
            n.append(normalize_text(r["hard_negative"]))
    data["synthetic"] = Dataset.from_dict({"anchor": a, "positive": p, "negative": n})
    if hinglish_aug:
        test_originals = set(pd.read_csv(ROOT / "data" / "gold" / "hinglish_test.csv")["original"])
        bb = data_io.load_bigbasket().drop_duplicates("product")
        names = [n for n in (bb["brand"].fillna("") + " " + bb["product"]).str.strip() if n not in test_originals]
        rng = random.Random(SEED)
        vocab = set(" ".join(HINDI_WORDS.values()).split())
        _, variants = aksharantar_maps(vocab)
        a, p = [], []
        for name in names:
            h = to_hinglish(name)
            if h:
                a.append(normalize_text(name))
                p.append(normalize_text(spelling_variant(h, variants, rng) if rng.random() < 0.5 else h))
        data["hinglish"] = Dataset.from_dict({"anchor": a, "positive": p})
    return data


def train(out_dir, hinglish_aug: bool = False, epochs: int = 1, batch_size: int = 64) -> dict:
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments)
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers, MultiDatasetBatchSamplers
    import torch
    torch.manual_seed(SEED)
    model = SentenceTransformer(BASE)
    model.max_seq_length = 64
    data = training_data(hinglish_aug)
    loss = MultipleNegativesRankingLoss(model)
    args = SentenceTransformerTrainingArguments(
        output_dir=str(MODELS / "checkpoints"), num_train_epochs=epochs, per_device_train_batch_size=batch_size,
        learning_rate=2e-5, warmup_ratio=0.1, fp16=torch.cuda.is_available(), seed=SEED,
        batch_sampler=BatchSamplers.NO_DUPLICATES,                  # in-batch negatives must differ
        multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL,
        save_strategy="no", logging_steps=50, report_to=[])
    t0 = time.perf_counter()
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=data, loss=loss)
    trainer.train()
    model.save(str(out_dir))
    info = {"model": str(out_dir.name), "train_seconds": round(time.perf_counter() - t0, 1),
            **{f"pairs_{k}": len(v) for k, v in data.items()}, "epochs": epochs, "batch_size": batch_size}
    print(info, flush=True)
    return info


# ---------------------------------------------------------------- evaluation
def wdc_sbert_f1(model) -> dict:
    """SBERT-only pair matching on WDC: threshold tuned on the validation set."""
    from src.match import prf
    def cos(df):
        a = model.encode(df["title_left"].fillna("").map(normalize_text).tolist(), normalize_embeddings=True,
                         batch_size=128, show_progress_bar=False)
        b = model.encode(df["title_right"].fillna("").map(normalize_text).tolist(), normalize_embeddings=True,
                         batch_size=128, show_progress_bar=False)
        return (a * b).sum(1)
    val = data_io.load_wdc_products("pair", 80, "valid", "large")
    sv, yv = cos(val), val["label"].values == 1
    best_t = max(np.unique(np.round(sv, 3)), key=lambda t: prf(set(np.where(sv >= t)[0]), set(np.where(yv)[0]))["f1"])
    out = {}
    for u in (0, 50, 100):
        t = data_io.load_wdc_products("pair", 80, "gs", unseen=u)
        s = cos(t)
        out[f"WDC F1 {u}% unseen"] = prf(set(np.where(s >= best_t)[0]), set(np.where(t["label"].values == 1)[0]))["f1"]
    return out


def hinglish_eval(model, name: str, roman_to_native: dict, variants: dict, prefix: tuple[str, str] = ("", "")) -> list[dict]:
    """Retrieve the original BigBasket name for each rewritten query; corpus = all 27k names."""
    test = pd.read_csv(ROOT / "data" / "gold" / "hinglish_test.csv")
    bb = data_io.load_bigbasket()
    corpus = sorted(set((bb["brand"].fillna("") + " " + bb["product"]).str.strip()))
    pos = {c: k for k, c in enumerate(corpus)}
    qp, dp = prefix
    C = model.encode([dp + normalize_text(c) for c in corpus], normalize_embeddings=True, batch_size=256,
                     show_progress_bar=False)
    rng = random.Random(SEED)
    queries = {
        "English (casual)": test["english_noisy"].tolist(),
        "Hinglish": test["hinglish"].tolist(),
        # second Hinglish set from the hand-written dictionary (same dictionary as the training augmentation,
        # different names), because the LLM rewrote only a few words into Hindi
        "Hinglish (dictionary)": [to_hinglish(o) or o.lower() for o in test["original"]],
        "Hinglish, spelling variants": [spelling_variant(normalize_text(h), variants, rng) for h in test["hinglish"]],
        "Hinglish, back-transliterated": [back_transliterate(normalize_text(h), roman_to_native) for h in test["hinglish"]],
    }
    rows = []
    for qname, qs in queries.items():
        Q = model.encode([qp + normalize_text(q) for q in qs], normalize_embeddings=True, batch_size=128,
                         show_progress_bar=False)
        idx, _ = cosine_topk(Q, C, 10)
        ranked = [[corpus[i] for i in row] for row in idx]
        m = retrieval_metrics(ranked, [{o} for o in test["original"]])
        rows.append({"model": name, "query": qname, **m})
    return rows


def evaluate() -> None:
    from sentence_transformers import SentenceTransformer
    test = pd.read_csv(ROOT / "data" / "gold" / "hinglish_test.csv")
    vocab = set(" ".join(test["hinglish"].map(normalize_text)).split()) | set(" ".join(HINDI_WORDS.values()).split())
    roman_to_native, variants = aksharantar_maps(vocab)
    models = {"MiniLM base": BASE, "MiniLM fine-tuned": str(FT), "MiniLM fine-tuned + Hinglish aug": str(FT_HI),
              **MULTILINGUAL}
    english_rows, hing_rows = [], []
    for name, path in models.items():
        model = SentenceTransformer(path)
        prefix = ("query: ", "passage: ") if "E5" in name else ("", "")
        hing_rows += hinglish_eval(model, name, roman_to_native, variants, prefix)
        print(hing_rows[-5:], flush=True)
        if "MiniLM" in name:
            SBERT_MODELS["custom"] = path
            ab = evaluate_retrieval("abt_buy", "custom")
            english_rows.append({"model": name, **wdc_sbert_f1(model), "Abt-Buy R@1": ab["recall@1"],
                                 "Abt-Buy MRR": ab["mrr"]})
            print(english_rows[-1], flush=True)
        del model
    pd.DataFrame(english_rows).to_csv(OUT / "phase7_english.csv", index=False)
    h = pd.DataFrame(hing_rows)
    h.to_csv(OUT / "phase7_hinglish.csv", index=False)
    print(h.pivot(index="model", columns="query", values="recall@1"))
    json.dump({"aksharantar_words_found": len(roman_to_native),
               "words_with_variants": sum(bool(v) for v in variants.values()),
               "examples": {k: variants[k][:4] for k in list(variants)[:15]}},
              open(OUT / "phase7_aksharantar.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    step = sys.argv[1]
    if step == "train":
        infos = [train(FT), train(FT_HI, hinglish_aug=True)]
        pd.DataFrame(infos).to_csv(OUT / "phase7_training.csv", index=False)
    elif step == "evaluate":
        evaluate()
