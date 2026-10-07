"""Phase 3: attribute extraction (NER) from product names.

Attributes: brand, product_type, variant, quantity, pack_count, colour.
Four methods, all evaluated on WDC-PAVE titles:
  1. rules      spaCy EntityRuler (brand + colour gazetteers) + Phase 2 quantity/pack rules
                + a noun-chunk heuristic for the product type
  2. gliner     GLiNER zero-shot NER with our label names, no training
  3. llm        qwen3:8b, few-shot prompt, JSON validated by a Pydantic schema
  4. distilled  a small spaCy NER trained on LLM-labelled BigBasket + Flipkart names

Run:
  python -m src.extract rules_gliner      # methods 1-2 + evaluation
  python -m src.extract llm_pave          # method 3 on the PAVE test titles (LLM, slow)
  python -m src.extract llm_label N       # LLM labels for the first N distillation names (resumable)
  python -m src.extract distill           # train + evaluate method 4, write the F1 table
"""
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from src import data_io, llm
from src.preprocess import find_pack_count, find_quantities, normalize_like_pave, normalize_text

ROOT = data_io.ROOT
SEED = 42
ATTRS = ["brand", "product_type", "variant", "quantity", "pack_count", "colour"]
EVAL_ATTRS = ["brand", "product_type", "colour", "quantity", "pack_count"]   # PAVE has no 'variant'
OUT = ROOT / "reports"
MODEL_DIR = ROOT / "models" / "ner_distilled"

COLOUR_LIST = sorted("""black white red blue green yellow orange purple pink brown grey gray silver gold
golden beige maroon navy violet magenta cyan teal turquoise ivory cream khaki olive mustard peach coral
lavender multicolor multicolour transparent clear chrome bronze copper charcoal tan burgundy""".split()) + [
    "rose gold", "navy blue", "sky blue", "dark blue", "light blue", "dark green", "light green", "off white"]

# brand-column values that are not brands (seen in Phase 0: Flipkart lists colours and fits as brands)
NOT_BRANDS = set(COLOUR_LIST) | {"regular", "slim", "na", "none", "generic", "unbranded", "others", "other"}


# ---------------------------------------------------------------- 1. rules
def brand_gazetteer() -> list[str]:
    """Brands from the catalogues' brand/manufacturer columns (BigBasket, Flipkart, Buy, Amazon, Google).
    WDC-PAVE is not used, so the evaluation stays fair."""
    ab, ag = data_io.load_abt_buy(), data_io.load_amazon_google()
    raw = pd.concat([data_io.load_bigbasket()["brand"], data_io.load_flipkart()["brand"],
                     ab["right"]["manufacturer"], ag["left"]["manufacturer"], ag["right"]["manufacturer"]])
    brands = raw.dropna().astype(str).str.strip()
    brands = brands[(brands.str.len() >= 2) & ~brands.str.lower().isin(NOT_BRANDS)]
    return sorted(set(brands))


_rule_nlp = None


def get_rule_nlp():
    """en_core_web_sm (for noun chunks) + an EntityRuler with BRAND and COLOUR phrase patterns,
    matched on lower-cased text. The statistical NER is removed: only the rules produce entities."""
    global _rule_nlp
    if _rule_nlp is None:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
        ruler = nlp.add_pipe("entity_ruler", config={"phrase_matcher_attr": "LOWER"})
        patterns = [{"label": "BRAND", "pattern": b} for b in brand_gazetteer()]
        patterns += [{"label": "COLOUR", "pattern": c} for c in COLOUR_LIST]
        ruler.add_patterns(patterns)
        _rule_nlp = nlp
    return _rule_nlp


def _product_type_heuristic(doc, taken: set[int]) -> str | None:
    """Last noun chunk of the first segment (before ' - ', ',' or '('), minus brand/colour tokens and numbers."""
    cut = len(doc)
    for t in doc:
        if t.text in {"-", ",", "(", "|", "/"} and t.i > 0:
            cut = t.i
            break
    chunks = [c for c in doc.noun_chunks if c.end <= cut] or list(doc.noun_chunks)
    for chunk in reversed(chunks):
        words = [t.text for t in chunk if t.i not in taken and not t.like_num and t.pos_ != "DET" and t.is_alpha]
        if words:
            return " ".join(words[-3:])
    return None


def extract_rules(texts: list[str]) -> list[dict]:
    out = []
    for doc in get_rule_nlp().pipe(texts, batch_size=256):
        rec = {a: None for a in ATTRS}
        taken = set()
        for ent in doc.ents:
            key = {"BRAND": "brand", "COLOUR": "colour"}[ent.label_]
            if rec[key] is None:
                rec[key] = ent.text
                taken.update(range(ent.start, ent.end))
        qs = [q for q in find_quantities(doc.text) if q.unit != "count"]
        rec["quantity"] = qs[0].text.strip() if qs else None
        rec["pack_count"] = find_pack_count(doc.text)
        rec["product_type"] = _product_type_heuristic(doc, taken)
        out.append(rec)
    return out


# ---------------------------------------------------------------- 2. GLiNER
GLINER_LABELS = {"brand": "brand", "product type": "product_type", "flavour or variant": "variant",
                 "quantity or size": "quantity", "pack count": "pack_count", "colour": "colour"}
_gliner = None


def get_gliner(name: str = "urchade/gliner_medium-v2.1"):
    global _gliner
    if _gliner is None:
        import torch
        from gliner import GLiNER
        _gliner = GLiNER.from_pretrained(name)
        if torch.cuda.is_available():
            _gliner = _gliner.to("cuda")
    return _gliner


def _to_int(text) -> int | None:
    if text is None:
        return None
    n = find_pack_count(str(text))
    if n is None:
        m = re.search(r"\d+", str(text))
        n = int(m.group()) if m else None
    return n


def extract_gliner(texts: list[str], threshold: float = 0.4) -> list[dict]:
    model = get_gliner()
    out = []
    for text in texts:
        ents = model.predict_entities(text, list(GLINER_LABELS), threshold=threshold)
        rec = {a: None for a in ATTRS}
        for e in sorted(ents, key=lambda e: -e["score"]):   # best-scoring span per label
            key = GLINER_LABELS[e["label"]]
            if rec[key] is None:
                rec[key] = e["text"]
        rec["pack_count"] = _to_int(rec["pack_count"])
        out.append(rec)
    return out


# ---------------------------------------------------------------- 3. LLM
class ProductAttributes(BaseModel):
    brand: str | None
    product_type: str | None
    variant: str | None
    quantity: str | None
    pack_count: int | None
    colour: str | None


LLM_SYSTEM = ("You extract attributes from product names. Copy each value exactly as it is written in the "
              "name (same words, do not translate, expand or invent). Use null when an attribute is absent. "
              "Answer only with JSON.")
FEW_SHOT = [
    ("Alisha Solid Women's Cycling Shorts, Pack of 3, Black",
     {"brand": "Alisha", "product_type": "Cycling Shorts", "variant": "Solid Women's", "quantity": None,
      "pack_count": 3, "colour": "Black"}),
    ("Fortune Sunlite Refined Sunflower Oil 1 L Pouch",
     {"brand": "Fortune", "product_type": "Sunflower Oil", "variant": "Sunlite Refined", "quantity": "1 L",
      "pack_count": None, "colour": None}),
    ("SanDisk Ultra 64GB microSDXC Memory Card (Pack of 2)",
     {"brand": "SanDisk", "product_type": "Memory Card", "variant": "Ultra microSDXC", "quantity": "64GB",
      "pack_count": 2, "colour": None}),
]


def llm_prompt(name: str) -> str:
    shots = "\n\n".join(f"Name: {n}\nJSON: {json.dumps(a)}" for n, a in FEW_SHOT)
    return (f"Attributes: brand, product_type (what the item is), variant (flavour, model line or style), "
            f"quantity (size/weight/volume/capacity with its unit), pack_count (number of items), colour.\n\n"
            f"{shots}\n\nName: {name}\nJSON:")


def extract_llm(texts: list[str], progress_every: int = 25) -> list[dict]:
    out = []
    t0 = time.perf_counter()
    for i, text in enumerate(texts, 1):
        res = llm.ask_json(llm_prompt(text), ProductAttributes, system=LLM_SYSTEM)
        out.append(res.model_dump() if res else {a: None for a in ATTRS})
        if i % progress_every == 0:
            print(f"  {i}/{len(texts)}  {time.perf_counter() - t0:.0f}s  {llm.report()}", flush=True)
    return out


# ---------------------------------------------------------------- 4. distilled spaCy NER
SPACY_LABELS = {"brand": "BRAND", "product_type": "TYPE", "variant": "VARIANT", "quantity": "QUANTITY",
                "pack_count": "PACK", "colour": "COLOUR"}


def spans_from_attributes(text: str, attrs: dict) -> list[tuple[int, int, str]]:
    """Turn LLM attribute values back into character spans of the text (case-insensitive search).
    Values that are not literally in the text are skipped; overlapping spans keep the longer one."""
    spans = []
    low = text.lower()
    for key, label in SPACY_LABELS.items():
        v = attrs.get(key)
        if v is None:
            continue
        v = str(v).strip()
        if key == "pack_count":   # the LLM returns a number; find the phrase that holds it
            m = re.search(rf"(pack|set|combo)\s+of\s+{v}\b|\b{v}\s*(pcs|pieces|pack|x)\b", low)
            if m:
                spans.append((m.start(), m.end(), label))
            continue
        i = low.find(v.lower())
        if i >= 0 and v:
            spans.append((i, i + len(v), label))
    spans.sort(key=lambda s: (-(s[1] - s[0]), s[0]))
    kept = []
    for s in spans:
        if all(s[1] <= k[0] or s[0] >= k[1] for k in kept):
            kept.append(s)
    return sorted(kept)


def train_distilled(texts: list[str], labels: list[dict], n_iter: int = 30, dev_frac: float = 0.1):
    """Train a blank English spaCy NER on LLM-labelled names. Returns (nlp, training log)."""
    import random
    import spacy
    from spacy.training import Example

    data = []
    for t, a in zip(texts, labels):
        spans = spans_from_attributes(t, a)
        if spans:
            data.append((t, spans))
    rng = random.Random(SEED)
    rng.shuffle(data)
    n_dev = int(len(data) * dev_frac)
    dev, train = data[:n_dev], data[n_dev:]

    spacy.util.fix_random_seed(SEED)
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")
    for label in SPACY_LABELS.values():
        ner.add_label(label)

    def examples(rows):
        out = []
        for text, spans in rows:
            doc = nlp.make_doc(text)
            ents = [doc.char_span(s, e, label=l, alignment_mode="expand") for s, e, l in spans]
            ents = [e for e in ents if e is not None]
            # spans that expanded into each other are dropped
            ents = [e for i, e in enumerate(ents) if all(e.end <= f.start or e.start >= f.end for f in ents[:i])]
            out.append(Example.from_dict(doc, {"entities": [(e.start_char, e.end_char, e.label_) for e in ents]}))
        return out

    train_ex, dev_ex = examples(train), examples(dev)
    optimizer = nlp.initialize(lambda: train_ex)
    log = []
    for it in range(n_iter):
        rng.shuffle(train_ex)
        losses = {}
        for batch in spacy.util.minibatch(train_ex, size=32):
            nlp.update(batch, sgd=optimizer, drop=0.2, losses=losses)
        if (it + 1) % 5 == 0:
            scores = nlp.evaluate(dev_ex)
            log.append({"iter": it + 1, "loss": round(losses["ner"], 1), "dev_ents_f": round(scores["ents_f"], 3)})
            print("  ", log[-1], flush=True)
    return nlp, pd.DataFrame(log), {"train": len(train), "dev": len(dev), "skipped_no_span": len(texts) - len(data)}


def extract_spacy(nlp, texts: list[str]) -> list[dict]:
    inv = {v: k for k, v in SPACY_LABELS.items()}
    out = []
    for doc in nlp.pipe(texts, batch_size=256):
        rec = {a: None for a in ATTRS}
        for ent in doc.ents:
            key = inv[ent.label_]
            if rec[key] is None:
                rec[key] = ent.text
        rec["pack_count"] = _to_int(rec["pack_count"])
        out.append(rec)
    return out


# ---------------------------------------------------------------- evaluation on WDC-PAVE
PAVE_MAP = {"Brand": "brand", "Manufacturer": "brand", "Product Type": "product_type", "Color": "colour",
            "Color(s)": "colour", "Pack Quantity": "pack_count", "Capacity": "quantity", "Size/Weight": "quantity"}


def pave_test() -> pd.DataFrame:
    """One row per PAVE test offer: title + the set of gold values per attribute that occur in the title."""
    t = data_io.load_wdc_pave("wdc_with_all_attributes", "test")
    rows = []
    for _, r in t.iterrows():
        gold = {a: set() for a in EVAL_ATTRS}
        for attr, values in r["target_scores"].items():
            key = PAVE_MAP.get(attr)
            if key is None:
                continue
            for v, info in values.items():
                if v != "n/a" and 0 in info["pid"]:
                    gold[key].add(v)
        rows.append({"id": r["id"], "category": r["category"], "title": r["input_title"], **gold})
    return pd.DataFrame(rows)


def norm_value(attr: str, v) -> str | None:
    if v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() == "":
        return None
    if attr == "pack_count":
        n = normalize_like_pave("count", str(v), keep_unparsed=False)
        return n
    return normalize_text(str(v)) or None


def score(preds: list[dict], gold: pd.DataFrame, lenient: bool = False) -> pd.DataFrame:
    """Slot-filling P/R/F1 per attribute. A prediction is correct if it matches one of the gold values
    (strict: identical after normalize_text; lenient: one contains the other).
    A wrong prediction counts as a false positive AND, if gold exists, a false negative."""
    rows = []
    for attr in EVAL_ATTRS:
        tp = fp = fn = 0
        for p, g in zip(preds, gold[attr]):
            pv = norm_value(attr, p.get(attr))
            gv = {norm_value(attr, x) for x in g} - {None}
            ok = pv is not None and any(pv == x or (lenient and (f" {pv} " in f" {x} " or f" {x} " in f" {pv} "))
                                        for x in gv)
            if ok:
                tp += 1
            else:
                fp += pv is not None
                fn += bool(gv)
        p_ = tp / (tp + fp) if tp + fp else 0.0
        r_ = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0
        rows.append({"attribute": attr, "precision": round(p_, 3), "recall": round(r_, 3), "f1": round(f1, 3),
                     "gold_n": int(sum(bool(x) for x in gold[attr]))})
    df = pd.DataFrame(rows)
    macro = {"attribute": "MACRO", **df[["precision", "recall", "f1"]].mean().round(3).to_dict(),
             "gold_n": int(df["gold_n"].sum())}
    return pd.concat([df, pd.DataFrame([macro])], ignore_index=True)


def save_preds(name: str, preds: list[dict], gold: pd.DataFrame, seconds: float) -> None:
    df = pd.DataFrame(preds)
    df.insert(0, "title", gold["title"].values)
    df.to_json(OUT / f"phase3_preds_{name}.json", orient="records", indent=1, force_ascii=False)
    timing = json.loads((OUT / "phase3_timing.json").read_text()) if (OUT / "phase3_timing.json").exists() else {}
    timing[name] = {"seconds": round(seconds, 2), "names": len(preds),
                    "names_per_second": round(len(preds) / max(seconds, 1e-9), 2)}
    (OUT / "phase3_timing.json").write_text(json.dumps(timing, indent=1))


def load_preds(name: str) -> list[dict]:
    return json.loads((OUT / f"phase3_preds_{name}.json").read_text(encoding="utf-8"))


def distillation_names(n: int = 2000) -> list[str]:
    """Half BigBasket (brand + name, because BigBasket names omit the brand), half Flipkart names,
    sampled with a fixed seed. Flipkart adds jewellery, electronics and home goods."""
    bb = data_io.load_bigbasket().sample(frac=1, random_state=SEED)
    fk = data_io.load_flipkart().drop_duplicates("product_name").sample(frac=1, random_state=SEED)
    bb_names = (bb["brand"].astype(str) + " " + bb["product"].astype(str)).head(n // 2).tolist()
    fk_names = fk["product_name"].astype(str).head(n - n // 2).tolist()
    out = []
    for a, b in zip(bb_names, fk_names):   # interleave so any prefix is balanced
        out += [a, b]
    return out


if __name__ == "__main__":
    step = sys.argv[1]
    gold = pave_test()
    titles = gold["title"].tolist()

    if step == "rules_gliner":
        for name, fn in (("rules", extract_rules), ("gliner", extract_gliner)):
            fn(titles[:2])                      # warm-up (model loading) is not timed
            t0 = time.perf_counter()
            preds = fn(titles)
            save_preds(name, preds, gold, time.perf_counter() - t0)
            print(name, "\n", score(preds, gold).to_string(index=False))

    elif step == "llm_pave":
        t0 = time.perf_counter()
        preds = extract_llm(titles)
        save_preds("llm", preds, gold, time.perf_counter() - t0)
        print(score(preds, gold).to_string(index=False))
        print(llm.report())
        llm.unload()

    elif step == "llm_label":
        n = int(sys.argv[2])
        names = distillation_names()[:n]
        labels = extract_llm(names)
        pd.DataFrame({"text": names, "labels": labels}).to_json(
            ROOT / "data" / "processed" / "distill_labels.json", orient="records", indent=1, force_ascii=False)
        print(llm.report())
        llm.unload()

    elif step == "gliner_label":
        # Teacher = GLiNER (the 1,000-name qwen3:8b run would take ~3.6 h at 13 s/name on this laptop).
        n = int(sys.argv[2])
        names = distillation_names()[:n]
        t0 = time.perf_counter()
        labels = extract_gliner(names)
        pd.DataFrame({"text": names, "labels": labels}).to_json(
            ROOT / "data" / "processed" / "distill_labels.json", orient="records", indent=1, force_ascii=False)
        (OUT / "phase3_teacher.json").write_text(json.dumps(
            {"teacher": "GLiNER urchade/gliner_medium-v2.1", "names": n,
             "seconds": round(time.perf_counter() - t0, 1)}))
        print(f"labelled {n} names in {time.perf_counter() - t0:.0f}s")

    elif step == "distill":
        lab = pd.read_json(ROOT / "data" / "processed" / "distill_labels.json")
        # learning curve: does more LLM-labelled data still help?
        curve = []
        for n in (250, 500, 1000, len(lab)):
            if n > len(lab):
                continue
            t0 = time.perf_counter()
            nlp, log, sizes = train_distilled(lab["text"].tolist()[:n], lab["labels"].tolist()[:n])
            train_s = time.perf_counter() - t0
            s = score(extract_spacy(nlp, titles), gold).set_index("attribute")["f1"]
            curve.append({"llm_labelled_names": n, **sizes, "train_seconds": round(train_s, 1), **s.to_dict()})
            print("learning curve:", curve[-1], flush=True)
        pd.DataFrame(curve).to_csv(OUT / "phase3_learning_curve.csv", index=False)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        nlp.to_disk(MODEL_DIR)
        log.to_csv(OUT / "phase3_distill_training.csv", index=False)
        t0 = time.perf_counter()
        preds = extract_spacy(nlp, titles)
        save_preds("distilled", preds, gold, time.perf_counter() - t0)
        print("sizes", sizes, "train seconds", round(train_s, 1))

        tables = []
        for method in ("rules", "gliner", "llm", "distilled"):
            if (OUT / f"phase3_preds_{method}.json").exists():
                p = load_preds(method)
                for mode in (False, True):
                    s = score(p, gold, lenient=mode)
                    s.insert(0, "match", "lenient" if mode else "strict")
                    s.insert(0, "method", method)
                    tables.append(s)
        table = pd.concat(tables, ignore_index=True)
        table.to_csv(OUT / "phase3_f1.csv", index=False)
        print(table[table["match"] == "strict"].pivot(index="attribute", columns="method", values="f1"))
        (OUT / "phase3_distill_sizes.json").write_text(json.dumps({**sizes, "train_seconds": round(train_s, 1)}))
