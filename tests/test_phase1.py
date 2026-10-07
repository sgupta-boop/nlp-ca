"""Phase 1 tests: header rules, embeddings, column selection, canonical tables."""
import pandas as pd
import pytest

from src import columns as C


@pytest.mark.parametrize("header,field", [
    ("sale_price", "price"), ("discounted_price", "price"), ("brand", "brand"),
    ("manufacturer", "brand"), ("product_category_tree", "category"), ("sub_category", "sub_category"),
    ("product_url", "other"), ("crawl_timestamp", "other"), ("uniq_id", "id"), ("title", "product_name"),
    ("description", "description"), ("overall_rating", "rating"), ("quantity", "quantity"),
])
def test_rules_known_headers(header, field):
    assert C.map_by_rules(header) == field


def test_header_tokens_splits_snake_and_camel_case():
    assert C.header_tokens("product_category_tree") == ["product", "category", "tree"]
    assert C.header_tokens("isFKAdvantage") == ["is", "fkadvantage"]


def test_rule_hit_only_when_a_keyword_fires():
    assert C.rule_hit("sale_price") and not C.rule_hit("pid") and not C.rule_hit("index")


def test_embeddings_map_obvious_headers():
    headers = ["product_name", "brand", "price"]
    samples = [["Amul Butter 500 g", "Tata Salt 1 kg"], ["Amul", "Tata"], ["49.0", "120.5"]]
    s = C.embedding_scores(headers, samples, use_samples=False)
    assert [C.FIELDS[i] for i in s.argmax(1)] == ["product_name", "brand", "price"]


def test_price_conflict_higher_median_is_mrp():
    df = pd.DataFrame({"sale_price": [90, 180], "market_price": [100, 200]})
    m = pd.DataFrame({"header": ["sale_price", "market_price"], "predicted": ["price", "price"],
                      "decided_by": ["rules", "rules"]})
    out = C.resolve_price_conflict(m, df)
    assert out["predicted"].tolist() == ["price", "mrp"]


def test_select_columns_keeps_names_drops_ids_prices_urls():
    df = pd.DataFrame({
        "name": [f"Brand{i} Green Tea {i * 10} g pack" for i in range(50)],
        "sku": [f"SKU{i:05d}" for i in range(50)],
        "price": [str(10.5 + i) for i in range(50)],
        "url": [f'["http://img.example.com/{i}.jpg"]' for i in range(50)],
        "category": ["Beverages", "Snacks"] * 25,
    })
    sel = C.select_columns(df).set_index("column")
    assert sel.loc["name", "selected"] and sel.loc["name", "role"] == "primary (name)"
    assert not sel.loc[["sku", "price", "url", "category"], "selected"].any()


def test_entropy_of_uniform_two_values_is_one_bit():
    assert C.entropy(pd.Series(["a", "b"] * 10)) == pytest.approx(1.0)


def test_to_canonical_renames_splits_path_and_fills_missing():
    df = pd.DataFrame({"title": ["Shorts A"], "product_category_tree": ['["Clothing >> Women\'s >> X"]'],
                       "retail_price": ["Rs. 999"], "url": ["http://x"]})
    m = pd.DataFrame({"header": df.columns, "predicted": ["product_name", "category", "mrp", "other"]})
    out = C.to_canonical(df, m, "toy")
    assert list(out.columns) == ["source"] + C.CANONICAL
    assert out.loc[0, "category"] == "Clothing" and out.loc[0, "sub_category"] == "Women's"
    assert out.loc[0, "mrp"] == 999.0 and pd.isna(out.loc[0, "brand"])


def test_canonical_files_have_identical_columns():
    cols = {name: list(pd.read_parquet(C.ROOT / "data" / "processed" / f"{name}_canonical.parquet").columns)
            for name in ["bigbasket", "flipkart", "off_india", "abt"]}
    assert len({tuple(c) for c in cols.values()}) == 1
