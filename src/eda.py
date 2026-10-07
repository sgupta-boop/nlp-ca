"""Phase 0: profiling helpers used by notebooks/00_eda.ipynb."""
import re
from collections import Counter

import pandas as pd

# number followed by a unit, e.g. "500 g", "1kg", "2 x 200ml", "pack of 6" is handled separately
UNIT_PATTERN = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*"
    r"(kgs?|gms?|grams?|gr|g|mg|ltrs?|litres?|liters?|lt|l|ml|pcs|pc|pieces?|pkts?|packs?|nos|"
    r"inch(?:es)?|cm|mm|gb|tb|mb)\b",
    flags=re.IGNORECASE,
)


def unit_mentions(names: pd.Series) -> Counter:
    """Count how often each unit *spelling* appears after a number ("g", "gm", "gms" are kept apart
    on purpose: the variety of spellings is what Phase 2 has to normalise)."""
    counts = Counter()
    for name in names.dropna().astype(str):
        for _, unit in UNIT_PATTERN.findall(name.lower()):
            counts[unit] += 1
    return counts


def profile_table(df: pd.DataFrame, name: str, name_col: str, brand_col: str | None = None) -> dict:
    """One summary row describing a product table."""
    names = df[name_col].dropna().astype(str)
    tokens = names.str.split().str.len()
    units = unit_mentions(names)
    return {
        "dataset": name,
        "rows": len(df),
        "columns": df.shape[1],
        "name column": name_col,
        "mean null % (all cols)": round(df.isna().mean().mean() * 100, 1),
        "null % in name": round(df[name_col].isna().mean() * 100, 2),
        "exact duplicate rows": int(df.duplicated().sum()),
        "duplicate names": int(names.duplicated().sum()),
        "duplicate names (lowercased)": int(names.str.lower().str.strip().duplicated().sum()),
        "name length chars (median)": int(names.str.len().median()),
        "name length tokens (mean)": round(tokens.mean(), 1),
        "name length tokens (max)": int(tokens.max()),
        "names with a quantity %": round(names.apply(lambda s: bool(UNIT_PATTERN.search(s))).mean() * 100, 1),
        "distinct unit spellings": len(units),
        "distinct brands": int(df[brand_col].nunique()) if brand_col else None,
    }


def null_percent(df: pd.DataFrame) -> pd.Series:
    return (df.isna().mean() * 100).round(1).sort_values(ascending=False)


def top_values(series: pd.Series, n: int = 15) -> pd.DataFrame:
    vc = series.dropna().astype(str).str.strip().value_counts().head(n)
    return vc.rename_axis("value").reset_index(name="count")
