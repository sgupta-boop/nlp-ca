"""Phase 0 tests: every dataset loads with the expected shape, and the EDA unit regex works.

Expected sizes come from the roadmap's dataset table, so a failing test means
a download is incomplete or the source changed.
"""
import pandas as pd
import pytest

from src import data_io
from src.eda import profile_table, unit_mentions


def _has(folder):
    return (data_io.RAW / folder).exists()


# ---------- benchmarks ----------

def test_abt_buy_sizes():
    d = data_io.load_abt_buy()
    assert len(d["left"]) == 1081
    assert len(d["right"]) == 1092
    assert len(d["matches"]) == 1097
    assert list(d["matches"].columns) == ["idAbt", "idBuy"]


def test_amazon_google_sizes():
    d = data_io.load_amazon_google()
    assert len(d["left"]) == 1363
    assert len(d["right"]) == 3226
    assert len(d["matches"]) == 1300


def test_abt_buy_matches_point_to_real_records():
    d = data_io.load_abt_buy()
    assert d["matches"]["idAbt"].isin(d["left"]["id"]).all()
    assert d["matches"]["idBuy"].isin(d["right"]["id"]).all()


def test_wdc_products_pair_and_multi_columns():
    pairs = data_io.load_wdc_products("pair", 80, "gs")
    assert {"title_left", "title_right", "label"} <= set(pairs.columns)
    assert set(pairs["label"].unique()) <= {0, 1}
    multi = data_io.load_wdc_products("multi", 80, "gs")
    assert {"title", "label"} <= set(multi.columns)


def test_wdc_pave_size():
    test = data_io.load_wdc_pave(split="test")
    train = data_io.load_wdc_pave(split="train")
    assert len(test) + len(train) == 565          # roadmap: 565 offers
    pairs = data_io.pave_attribute_pairs(test)
    assert {"attribute", "value"} <= set(pairs.columns) and len(pairs) > 0


def test_google_taxonomy_top_level():
    tax = data_io.load_google_taxonomy()
    assert tax["top_level"].nunique() == 21       # Google taxonomy has 21 root categories
    assert "Food, Beverages & Tobacco" in set(tax["top_level"])


def test_aksharantar_columns():
    df = data_io.load_aksharantar_hindi("test")
    assert {"native word", "english word"} <= set(df.columns)


# ---------- working data (needs Kaggle download) ----------

@pytest.mark.skipif(not _has("bigbasket"), reason="BigBasket not downloaded")
def test_bigbasket_shape():
    df = data_io.load_bigbasket()
    assert len(df) == 27555
    assert {"product", "category", "sub_category", "brand", "sale_price", "market_price"} <= set(df.columns)


@pytest.mark.skipif(not _has("flipkart"), reason="Flipkart not downloaded")
def test_flipkart_shape():
    df = data_io.load_flipkart()
    assert len(df) == 20000
    assert {"product_name", "product_category_tree", "brand"} <= set(df.columns)


@pytest.mark.skipif(not (data_io.RAW / "off_india.parquet").exists(), reason="OFF India not downloaded")
def test_off_india_only_india_rows():
    df = data_io.load_off_india()
    assert len(df) > 1000
    assert df["countries_tags"].str.contains("en:india").all()
    assert df["product_name"].dropna().map(type).eq(str).all()   # names are plain text


# ---------- EDA helpers ----------

def test_unit_mentions_keeps_spellings_apart():
    names = pd.Series(["Atta 5 kg", "Sugar 1kg", "Dal 500 gm", "Oil 1 ltr", "Milk 500ml", "Rice 2 KG"])
    c = unit_mentions(names)
    assert c["kg"] == 3 and c["gm"] == 1 and c["ltr"] == 1 and c["ml"] == 1


def test_unit_mentions_ignores_numbers_without_units():
    assert sum(unit_mentions(pd.Series(["Model 2020 Edition", "Pack 3"])).values()) == 0


def test_profile_table_counts_duplicates():
    df = pd.DataFrame({"name": ["Tea 250 g", "tea 250 g", "Coffee"], "brand": ["A", "A", "B"]})
    p = profile_table(df, "toy", "name", "brand")
    assert p["rows"] == 3
    assert p["duplicate names"] == 0
    assert p["duplicate names (lowercased)"] == 1
    assert p["distinct brands"] == 2
