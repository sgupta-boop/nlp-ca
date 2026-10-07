"""Loaders for every dataset in data/raw/.

Each function returns a pandas DataFrame (or a dict of DataFrames) with the
columns exactly as the source provides them. No cleaning happens here: that is
Phase 1 (column mapping) and Phase 2 (text preprocessing).
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def _single_csv(folder: Path) -> Path:
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV in {folder}. Run: python scripts/download_data.py")
    return files[0]


# ---------- working data (Indian retail) ----------

def load_bigbasket(keep_index: bool = False) -> pd.DataFrame:
    df = pd.read_csv(_single_csv(RAW / "bigbasket"))
    # the Kaggle file has an `index` column that is just the row number (1..n)
    if keep_index:
        return df
    return df.drop(columns=[c for c in df.columns if c.lower() in ("index", "unnamed: 0")])


def load_flipkart() -> pd.DataFrame:
    return pd.read_csv(_single_csv(RAW / "flipkart"))


def load_off_india() -> pd.DataFrame:
    """Open Food Facts, India subset (written by download_data.py)."""
    df = pd.read_parquet(RAW / "off_india.parquet")
    # product_name is a list of {lang, text}; keep the English one, else the first
    def pick_name(entries):
        if entries is None or len(entries) == 0:
            return None
        for e in entries:
            if e.get("lang") in ("en", "main"):
                return e.get("text")
        return entries[0].get("text")
    if len(df) and not isinstance(df["product_name"].dropna().iloc[0], str):
        df["product_name"] = df["product_name"].apply(pick_name)
    return df


def load_off_sample() -> pd.DataFrame:
    """First ~12.7k rows of the full Open Food Facts export with all 211 original columns
    (only used to test column mapping on OFF's real, messy header)."""
    return pd.read_csv(RAW / "off_sample.tsv", sep="\t", quoting=3, on_bad_lines="skip",
                       low_memory=False)


# ---------- benchmarks with gold labels ----------

def load_abt_buy() -> dict:
    d = RAW / "abt_buy"
    return {
        "left": pd.read_csv(d / "Abt.csv", encoding="latin-1"),
        "right": pd.read_csv(d / "Buy.csv", encoding="latin-1"),
        "matches": pd.read_csv(d / "abt_buy_perfectMapping.csv"),
    }


def load_amazon_google() -> dict:
    d = RAW / "amazon_google"
    return {
        "left": pd.read_csv(d / "Amazon.csv", encoding="latin-1"),
        "right": pd.read_csv(d / "GoogleProducts.csv", encoding="latin-1"),
        "matches": pd.read_csv(d / "Amzon_GoogleProducts_perfectMapping.csv"),
    }


def load_wdc_products(kind: str = "pair", corner_cases: int = 80,
                      split: str = "gs", size: str = "large", unseen: int = 0) -> pd.DataFrame:
    """kind: 'pair' or 'multi'; corner_cases: 80/50/20; split: 'train', 'valid' or 'gs' (test);
    size: small/medium/large (train/valid only); unseen: 0/50/100 (% unseen products, test only)."""
    rnd = 100 - corner_cases
    prefix = "wdcproducts" if kind == "pair" else "wdcproductsmulti"
    stem = f"{prefix}{corner_cases}cc{rnd}rnd{unseen:03d}un"
    name = f"{stem}_gs" if split == "gs" else f"{stem}_{split}_{size}"
    return pd.read_json(RAW / "wdc_products" / f"{name}.json.gz", compression="gzip", lines=True)


def load_wdc_pave(variant: str = "wdc_normalized", split: str = "test") -> pd.DataFrame:
    path = RAW / "wdc_pave_repo" / "data" / "processed_datasets" / variant / f"{split}.jsonl"
    return pd.read_json(path, lines=True)


# ---------- helper resources ----------

def load_aksharantar_hindi(split: str = "test") -> pd.DataFrame:
    return pd.read_json(RAW / "aksharantar_hin" / f"hin_{split}.json", lines=True)


def load_google_taxonomy() -> pd.DataFrame:
    """One row per taxonomy path, with the top-level category split out."""
    lines = (RAW / "google_taxonomy.en-US.txt").read_text(encoding="utf-8").splitlines()
    paths = [l for l in lines if l and not l.startswith("#")]
    df = pd.DataFrame({"path": paths})
    df["top_level"] = df["path"].str.split(" > ").str[0]
    df["depth"] = df["path"].str.count(" > ") + 1
    return df


def pave_attribute_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten PAVE's nested target_scores into one row per (offer, attribute, value).
    'n/a' means the attribute is not present in the offer."""
    rows = []
    for _, r in df.iterrows():
        for attr, values in r["target_scores"].items():
            for value in values:
                if value != "n/a":
                    rows.append({"id": r["id"], "category": r["category"],
                                 "attribute": attr, "value": value})
    return pd.DataFrame(rows)
