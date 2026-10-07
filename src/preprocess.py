"""Phase 2: text preprocessing and normalization.

Pipeline for one product name:
  1. normalize_text   Unicode NFKC, lowercase, '&' -> 'and', split "200gm" -> "200 gm",
                      rewrite every quantity in one canonical form ("1kg" and "1000 gm" -> "1000 g"),
                      drop punctuation (except inside numbers), collapse spaces
  2. parse_quantity   regex for number+unit (with mixed fractions), spaCy Matcher for pack counts
                      ("pack of 6", "6 x 200 g", "12 pcs", "24/box"); values normalized to g / ml / count / cm
  3. expand abbreviations (pkt -> packet, choc -> chocolate, ...), chosen from mined candidates
  4. spelling correction with SymSpell, dictionary built from our own product vocabulary
  5. spaCy lemmatization ("biscuits" -> "biscuit"); lowercasing first, see Phase 0 finding

Run:  python -m src.preprocess   (processes the canonical tables, evaluates against WDC-PAVE)
"""
import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction

import pandas as pd

from src import data_io

ROOT = data_io.ROOT
SEED = 42

# ---------------------------------------------------------------- units
# spelling -> (canonical unit, factor to convert into it)
UNITS = {
    # mass -> g
    "mg": ("g", 0.001), "g": ("g", 1), "gm": ("g", 1), "gms": ("g", 1), "gr": ("g", 1), "grm": ("g", 1),
    "gram": ("g", 1), "grams": ("g", 1), "gramme": ("g", 1), "kg": ("g", 1000), "kgs": ("g", 1000),
    "kilo": ("g", 1000), "kilogram": ("g", 1000), "kilograms": ("g", 1000),
    "oz": ("g", 28.349523125), "ounce": ("g", 28.349523125), "ounces": ("g", 28.349523125),
    "lb": ("g", 453.59237), "lbs": ("g", 453.59237), "pound": ("g", 453.59237), "pounds": ("g", 453.59237),
    # volume -> ml
    "ml": ("ml", 1), "millilitre": ("ml", 1), "milliliter": ("ml", 1), "cl": ("ml", 10),
    "l": ("ml", 1000), "lt": ("ml", 1000), "ltr": ("ml", 1000), "ltrs": ("ml", 1000),
    "litre": ("ml", 1000), "litres": ("ml", 1000), "liter": ("ml", 1000), "liters": ("ml", 1000),
    # length -> cm
    "mm": ("cm", 0.1), "cm": ("cm", 1), "inch": ("cm", 2.54), "inches": ("cm", 2.54), '"': ("cm", 2.54),
    "ft": ("cm", 30.48), "feet": ("cm", 30.48), "foot": ("cm", 30.48), "yd": ("cm", 91.44),
    "yds": ("cm", 91.44), "yard": ("cm", 91.44), "yards": ("cm", 91.44),
    # (no "in": "2 in 1 shampoo" is not 2 inches)
    # count
    "pc": ("count", 1), "pcs": ("count", 1), "piece": ("count", 1), "pieces": ("count", 1),
    "nos": ("count", 1), "units": ("count", 1), "sachets": ("count", 1), "tablets": ("count", 1),
    "capsules": ("count", 1), "count": ("count", 1), "ct": ("count", 1),
}
# units kept as they are (no conversion), with the full name used by WDC-PAVE
NAMED_UNITS = {"kb": "Kilobytes", "k": "Kilobytes", "mb": "Megabytes", "m": "Megabytes",
               "gb": "Gigabytes", "g": "Gigabytes", "tb": "Terabytes", "t": "Terabytes", "w": "Watts"}

# a number: 1,000 | 1.5 | .5 | 3-1/2 | 8 1/2 | 4/5
NUMBER = r"(?:\d+[\s-]\d+/\d+|\d+/\d+|\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d*\.\d+|\d+)"
_unit_alternation = "|".join(sorted((re.escape(u) for u in UNITS if u != '"'), key=len, reverse=True))
QTY_RE = re.compile(rf"(?<![\w.])({NUMBER})[\s-]*(?:({_unit_alternation})\b|(\"))", re.IGNORECASE)


def parse_number(text: str) -> float:
    """'3-1/2' -> 3.5, '8 1/2' -> 8.5, '4/5' -> 0.8, '1,000' -> 1000.0"""
    t = text.strip().replace(",", "")
    m = re.fullmatch(r"(\d+)[\s-](\d+/\d+)", t)
    if m:
        return float(int(m.group(1)) + Fraction(m.group(2)))
    if "/" in t:
        return float(Fraction(t))
    return float(t)


@dataclass
class Quantity:
    value: float          # in the canonical unit
    unit: str             # g | ml | cm | count
    text: str             # the matched text


def find_quantities(text: str) -> list[Quantity]:
    out = []
    for m in QTY_RE.finditer(text):
        number, unit = m.group(1), (m.group(2) or m.group(3)).lower()
        canon, factor = UNITS[unit]
        try:
            out.append(Quantity(parse_number(number) * factor, canon, m.group(0)))
        except (ValueError, ZeroDivisionError):
            continue
    return out


# ---------------------------------------------------------------- pack counts (spaCy Matcher)
_nlp_blank = None
_matcher = None
COUNT_WORDS = ["pcs", "pc", "pieces", "piece", "nos", "units", "pack", "packs", "pkt", "pkts", "packets",
               "sachets", "tablets", "capsules", "count", "ct", "each", "bags", "rolls", "forms", "sheets"]


def _get_matcher():
    """spaCy rule-based Matcher for pack-count patterns, on a blank English tokenizer."""
    global _nlp_blank, _matcher
    if _matcher is None:
        import spacy
        from spacy.matcher import Matcher
        _nlp_blank = spacy.blank("en")
        _matcher = Matcher(_nlp_blank.vocab)
        num = {"LIKE_NUM": True}
        _matcher.add("PACK_OF", [[{"LOWER": {"IN": ["pack", "set", "combo", "box", "case", "pk"]}},
                                  {"LOWER": "of"}, num]])
        _matcher.add("N_X", [[num, {"LOWER": {"IN": ["x", "*"]}}]])                 # 6 x 200 g
        _matcher.add("N_UNITS", [[num, {"LOWER": {"IN": COUNT_WORDS}}]])              # 12 pcs
        _matcher.add("N_PER", [[num, {"LOWER": {"IN": ["per", "/"]}},
                                {"LOWER": {"IN": ["box", "carton", "pack", "case", "bag"]}}]])  # 24/box
    return _nlp_blank, _matcher


def find_pack_count(text: str) -> int | None:
    nlp, matcher = _get_matcher()
    doc = nlp(re.sub(r"(\d)\s*([x*/])\s*", r"\1 \2 ", text.lower()))  # "6x200g" -> "6 x 200g"
    for match_id, start, end in sorted(matcher(doc), key=lambda m: m[1]):
        label = nlp.vocab.strings[match_id]
        tok = doc[end - 1] if label == "PACK_OF" else doc[start]
        try:
            n = int(float(tok.text.replace(",", "")))
        except ValueError:
            continue
        if n > 0:
            return n
    return None


def parse_quantity(text: str) -> dict:
    """Main entry: first mass/volume/length quantity + pack count.
    'Tata Salt 1kg' -> {value 1000, unit g, pack_count None}
    'Maggi Noodles 70 g (pack of 4)' -> {value 70, unit g, pack_count 4}"""
    text = unicodedata.normalize("NFKC", str(text))
    qs = [q for q in find_quantities(text) if q.unit != "count"]
    counts = [q for q in find_quantities(text) if q.unit == "count"]
    pack = find_pack_count(text)
    if pack is None and counts:
        pack = int(counts[0].value)
    q = qs[0] if qs else None
    return {"qty_value": round(q.value, 3) if q else None, "qty_unit": q.unit if q else None,
            "pack_count": pack}


def format_quantity(value: float, unit: str) -> str:
    return f"{value:g} {unit}"


# ---------------------------------------------------------------- text normalization
def normalize_text(text: str) -> str:
    # drop trademark signs first: NFKC would turn "™" into the letters "TM" ("Good Day™" -> "daytm")
    t = re.sub(r"[™®©]", " ", str(text))
    t = unicodedata.normalize("NFKC", t).lower()
    t = t.replace("&", " and ")
    t = re.sub(r"(\d)([a-z])", r"\1 \2", t)          # 200gm -> 200 gm
    t = re.sub(r"([a-z])(\d)", r"\1 \2", t)          # pack6 -> pack 6
    # canonical quantities: "1 kg", "1000 gm", "1 kgs" all become "1000 g"
    t = QTY_RE.sub(lambda m: f" {format_quantity(*_canon(m))} ", t)
    t = re.sub(r"(?<!\d)[.,](?!\d)|[^\w\s.,%]", " ", t)  # keep . , only inside numbers, keep %
    t = re.sub(r"(?<=\D)[.,]|[.,](?=\D)", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _canon(m: re.Match) -> tuple[float, str]:
    unit = (m.group(2) or m.group(3)).lower()
    canon, factor = UNITS[unit]
    try:
        return round(parse_number(m.group(1)) * factor, 3), canon
    except (ValueError, ZeroDivisionError):
        return 0.0, canon


# ---------------------------------------------------------------- abbreviations
# Chosen by hand from the mined candidates in reports/phase2_abbrev_candidates.csv
# Mining (top 25) found only 2 real abbreviations (choco, deo); the rest are everyday prefixes
# such as print -> printed. The dictionary = those 2 + common retail short forms seen in the data.
ABBREVIATIONS = {
    "pkt": "packet", "pkts": "packets", "pcs": "pieces", "pc": "piece", "pk": "pack",
    "choc": "chocolate", "choco": "chocolate", "deo": "deodorant",
}


def mine_abbreviation_candidates(token_counts: Counter, min_count: int = 3, top: int = 60) -> pd.DataFrame:
    """Short tokens that are a prefix of a much more frequent long token: likely abbreviations.
    e.g. 'choc' (prefix of 'chocolate')."""
    long_words = [(w, c) for w, c in token_counts.items() if len(w) >= 6 and w.isalpha()]
    rows = []
    for t, c in token_counts.items():
        if not (2 <= len(t) <= 5 and t.isalpha() and c >= min_count):
            continue
        best = max(((w, wc) for w, wc in long_words if w.startswith(t) and len(w) >= len(t) + 2),
                   key=lambda x: x[1], default=None)
        if best and best[1] >= 2 * c:
            rows.append({"short": t, "short_count": c, "expansion": best[0], "expansion_count": best[1]})
    df = pd.DataFrame(rows)
    return df.sort_values("short_count", ascending=False).head(top) if len(df) else df


def expand_abbreviations(tokens: list[str]) -> list[str]:
    return [ABBREVIATIONS.get(t, t) for t in tokens]


# ---------------------------------------------------------------- spelling correction (SymSpell)
class SpellCorrector:
    """Dictionary = tokens seen at least `min_count` times in our own product names.
    Only rare/unknown alphabetic tokens of length >= 5 are corrected, and only towards a word
    that is much more frequent, so brand names and real words are rarely touched."""

    def __init__(self, token_counts: Counter, min_count: int = 3, max_edit: int = 2,
                 correct_rare_seen: bool = False):
        """correct_rare_seen=False: only words never seen in the catalogue are corrected (default,
        because correcting rare-but-seen words damaged real words such as 'toaster' -> 'toasted')."""
        from symspellpy import SymSpell
        self.counts = token_counts
        self.min_count = min_count
        self.correct_rare_seen = correct_rare_seen
        self.sym = SymSpell(max_dictionary_edit_distance=max_edit)
        for w, c in token_counts.items():
            if c >= min_count and w.isalpha():
                self.sym.create_dictionary_entry(w, c)

    def correct_token(self, tok: str) -> str:
        from symspellpy import Verbosity
        seen = self.counts.get(tok, 0)
        if len(tok) < 5 or not tok.isalpha() or seen >= self.min_count:
            return tok
        if seen > 0 and not self.correct_rare_seen:
            return tok
        max_d = 1 if len(tok) < 8 else 2
        sugg = self.sym.lookup(tok, Verbosity.TOP, max_edit_distance=max_d)
        if sugg and sugg[0].distance > 0 and sugg[0].count >= 10 * max(self.counts.get(tok, 0), 1):
            return sugg[0].term
        return tok

    def correct(self, tokens: list[str]) -> list[str]:
        return [self.correct_token(t) for t in tokens]


def add_typo(word: str, rng: random.Random) -> str:
    """One random edit: delete, insert, substitute or swap two neighbours."""
    i = rng.randrange(len(word))
    op = rng.choice(["delete", "insert", "substitute", "swap"])
    letters = "abcdefghijklmnopqrstuvwxyz"
    if op == "delete":
        return word[:i] + word[i + 1:]
    if op == "insert":
        return word[:i] + rng.choice(letters) + word[i:]
    if op == "substitute":
        return word[:i] + rng.choice(letters.replace(word[i], "")) + word[i + 1:]
    j = min(i + 1, len(word) - 1)
    return word[:i] + word[j] + word[i] + word[j + 1:] if i != j else word[:-2] + word[-1] + word[-2]


def evaluate_spelling(corrector: SpellCorrector, token_counts: Counter, n: int = 500) -> dict:
    """Corrupt n frequent words with one random typo; how many are restored exactly?"""
    rng = random.Random(SEED)
    pool = sorted(w for w, c in token_counts.items() if c >= 20 and len(w) >= 5 and w.isalpha())
    words = rng.sample(pool, min(n, len(pool)))
    pairs = [(w, add_typo(w, rng)) for w in words]
    pairs = [(w, t) for w, t in pairs if t != w and token_counts.get(t, 0) < corrector.min_count]
    fixed = sum(corrector.correct_token(t) == w for w, t in pairs)
    changed_wrong = sum(corrector.correct_token(t) not in (w, t) for w, t in pairs)
    rare = [w for w, c in token_counts.items() if 1 <= c < corrector.min_count and len(w) >= 5 and w.isalpha()]
    rare_changed = [w for w in rare if corrector.correct_token(w) != w]
    return {"typo_words": len(pairs), "restored": fixed, "restored_rate": round(fixed / len(pairs), 3),
            "wrong_correction_rate": round(changed_wrong / len(pairs), 3),
            "rare_tokens": len(rare), "rare_tokens_changed": len(rare_changed),
            "rare_changed_rate": round(len(rare_changed) / max(len(rare), 1), 3),
            "examples_rare_changed": [(w, corrector.correct_token(w)) for w in rare_changed[:15]]}


# ---------------------------------------------------------------- lemmatization (spaCy)
_nlp = None


def get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    return _nlp


def lemmatize_many(texts: list[str], batch_size: int = 1000) -> list[str]:
    """Lemmatize already-lowercased texts; numbers and units are left as they are."""
    out = []
    for doc in get_nlp().pipe(texts, batch_size=batch_size):
        out.append(" ".join(t.lemma_ if t.is_alpha else t.text for t in doc))
    return out


# ---------------------------------------------------------------- full pipeline
def corpus_token_counts(names: list[str]) -> Counter:
    c = Counter()
    for n in names:
        c.update(normalize_text(n).split())
    return c


def preprocess_names(names: pd.Series, corrector: SpellCorrector | None = None,
                     lemmatize: bool = True) -> pd.DataFrame:
    names = names.fillna("").astype(str)
    clean = [normalize_text(n) for n in names]
    toks = [expand_abbreviations(c.split()) for c in clean]
    if corrector is not None:
        toks = [corrector.correct(t) for t in toks]
    joined = [" ".join(t) for t in toks]
    lemma = lemmatize_many(joined) if lemmatize else joined
    q = pd.DataFrame([parse_quantity(n) for n in names], index=names.index)
    return pd.DataFrame({"name_clean": clean, "name_norm": lemma}, index=names.index).join(q)


def preprocess_table(df: pd.DataFrame, corrector: SpellCorrector | None = None) -> pd.DataFrame:
    """Works on any canonical table (Phase 1 output)."""
    return df.join(preprocess_names(df["product_name"], corrector))


# ---------------------------------------------------------------- evaluation on WDC-PAVE
# PAVE attribute -> how its normalized value is written
PAVE_KINDS = {"Width": "length", "Height": "length", "Length": "length", "Depth": "length",
              "Size/Weight": "weight_g", "Paper Weight": "weight_kg", "Pack Quantity": "count",
              "Rotational Speed": "speed", "Capacity": "named", "Cache": "named"}


def normalize_like_pave(kind: str, raw: str, keep_unparsed: bool = True) -> str | None:
    """Normalize a PAVE attribute value; values we cannot parse are returned unchanged
    (keep_unparsed=True), which is also what WDC-PAVE does with them."""
    raw = unicodedata.normalize("NFKC", str(raw)).strip()
    out = _normalize_like_pave(kind, raw)
    return raw if (out is None and keep_unparsed) else out


def _normalize_like_pave(kind: str, raw: str) -> str | None:
    if kind == "length":
        qs = [q for q in find_quantities(raw) if q.unit == "cm"]
        if qs:
            return f"{qs[0].value:.1f}"
        m = re.match(rf"\s*({NUMBER})", raw)   # bare number: inches (PAVE convention)
        return f"{parse_number(m.group(1)) * 2.54:.1f}" if m else None
    if kind in ("weight_g", "weight_kg"):
        qs = [q for q in find_quantities(raw) if q.unit == "g"]
        if not qs:
            return None
        return f"{round(qs[0].value)}" if kind == "weight_g" else f"{qs[0].value / 1000:.1f}"
    if kind == "count":
        n = find_pack_count(raw)
        if n is None:
            m = re.match(r"\s*(\d+)", raw)
            n = int(m.group(1)) if m else None
        return str(n) if n is not None else None
    if kind == "speed":
        m = re.match(r"\s*([\d.,]+)\s*(k)?", raw, re.IGNORECASE)
        if not m:
            return None
        v = float(m.group(1).replace(",", "")) * (1000 if m.group(2) else 1)
        return f"{v:g}"
    if kind == "named":
        m = re.match(rf"\s*({NUMBER})\s*([a-zA-Z]+)", raw)
        if not m or m.group(2).lower() not in NAMED_UNITS:
            return None
        return f"{m.group(1)} {NAMED_UNITS[m.group(2).lower()]}"
    return None


def pave_unit_pairs() -> pd.DataFrame:
    """(raw value, gold normalized value) for the unit-bearing PAVE attributes."""
    load = lambda v: pd.concat([data_io.load_wdc_pave(v, s) for s in ("train", "test")])
    raw = data_io.pave_attribute_pairs(load("wdc_with_all_attributes"))
    nrm = data_io.pave_attribute_pairs(load("wdc_with_all_attributes_normalized"))
    r = raw.groupby(["id", "attribute"])["value"].apply(list).reset_index()
    n = nrm.groupby(["id", "attribute"])["value"].apply(list).reset_index()
    m = r.merge(n, on=["id", "attribute"], suffixes=("_raw", "_norm"))
    m = m[(m["value_raw"].str.len() == 1) & (m["value_norm"].str.len() == 1)]
    m = m[m["attribute"].isin(PAVE_KINDS)].copy()
    m["raw"], m["gold"] = m["value_raw"].str[0], m["value_norm"].str[0]
    return m[["id", "attribute", "raw", "gold"]].reset_index(drop=True)


def evaluate_units() -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = pave_unit_pairs()
    pairs["kind"] = pairs["attribute"].map(PAVE_KINDS)
    pairs["ours"] = [normalize_like_pave(k, r) for k, r in zip(pairs["kind"], pairs["raw"])]
    pairs["ours_strict"] = [normalize_like_pave(k, r, keep_unparsed=False) for k, r in zip(pairs["kind"], pairs["raw"])]
    pairs["ok_strict"] = pairs["ours_strict"] == pairs["gold"]
    first_num = lambda s: (re.search(r"\d+(?:\.\d+)?", s) or [None])[0]
    pairs["baseline_number"] = pairs["raw"].map(first_num)
    pairs["ok_ours"] = pairs["ours"] == pairs["gold"]
    pairs["ok_raw"] = pairs["raw"] == pairs["gold"]
    pairs["ok_baseline"] = pairs["baseline_number"] == pairs["gold"]
    by_attr = pairs.groupby("attribute").agg(n=("ok_ours", "size"), no_normalization=("ok_raw", "mean"),
                                             first_number_only=("ok_baseline", "mean"),
                                             ours_parsed_only=("ok_strict", "mean"),
                                             ours=("ok_ours", "mean")).round(3)
    total = pd.DataFrame({"n": [len(pairs)], "no_normalization": [pairs["ok_raw"].mean()],
                          "first_number_only": [pairs["ok_baseline"].mean()],
                          "ours_parsed_only": [pairs["ok_strict"].mean()],
                          "ours": [pairs["ok_ours"].mean()]}, index=["ALL"]).round(3)
    return pd.concat([by_attr, total]), pairs


# ---------------------------------------------------------------- main
def build_corrector() -> tuple[SpellCorrector, Counter]:
    names = pd.concat([pd.read_parquet(ROOT / "data" / "processed" / f"{n}_canonical.parquet")["product_name"]
                       for n in ("bigbasket", "flipkart")]).dropna().tolist()
    counts = corpus_token_counts(names)
    return SpellCorrector(counts), counts


if __name__ == "__main__":
    import time
    pd.set_option("display.width", 220)
    rep = ROOT / "reports"

    acc, pairs = evaluate_units()
    print("== unit normalization vs WDC-PAVE (exact match)\n", acc.to_string())
    acc.to_csv(rep / "phase2_unit_accuracy.csv")
    pairs.to_csv(rep / "phase2_unit_pairs.csv", index=False)

    corrector, counts = build_corrector()
    mined = mine_abbreviation_candidates(counts)
    mined.to_csv(rep / "phase2_abbrev_candidates.csv", index=False)
    print("\n== mined abbreviation candidates (top 25)\n", mined.head(25).to_string(index=False))

    spell_rows = []
    for setting, corr in (("unseen words only (system)", corrector),
                          ("also rare seen words", SpellCorrector(counts, correct_rare_seen=True))):
        sp = evaluate_spelling(corr, counts)
        sp["setting"] = setting
        spell_rows.append(sp)
        print(f"\n== spelling correction: {setting}\n",
              {k: v for k, v in sp.items() if k != "examples_rare_changed"})
    pd.DataFrame(spell_rows).to_json(rep / "phase2_spelling.json", orient="records", indent=1)
    # 40 random changes made by the aggressive setting, for manual judgement
    rng = random.Random(SEED)
    aggressive = SpellCorrector(counts, correct_rare_seen=True)
    rare = sorted(w for w, c in counts.items() if 1 <= c < 3 and len(w) >= 5 and w.isalpha())
    changed = [(w, aggressive.correct_token(w)) for w in rare]
    changed = [(w, c) for w, c in changed if c != w]
    pd.DataFrame(rng.sample(changed, 40), columns=["token", "changed_to"]).to_csv(
        rep / "phase2_rare_changes_sample.csv", index=False)

    timings = {}
    for name in ("bigbasket", "flipkart", "off_india", "abt", "buy"):
        t0 = time.perf_counter()
        df = pd.read_parquet(ROOT / "data" / "processed" / f"{name}_canonical.parquet")
        out = preprocess_table(df, corrector)
        out.to_parquet(ROOT / "data" / "processed" / f"{name}_pre.parquet", index=False)
        timings[name] = round(time.perf_counter() - t0, 1)
        print(f"{name}: {len(out)} rows, quantity found {out['qty_value'].notna().mean():.1%}, "
              f"pack count found {out['pack_count'].notna().mean():.1%}, {timings[name]} s")
    pd.Series(timings).to_csv(rep / "phase2_timings.csv")
