"""Phase 6-9 tests: LLM-layer helpers (no LLM calls), Hinglish/transliteration helpers, pipeline on a toy table."""
import random

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src import finetune as F
from src import llm_layer as L


def test_grey_zone_band():
    s = np.array([0.50, 0.62, 0.70, 0.79, 0.81, 0.95])
    assert L.grey_zone(s, 0.70, 0.10, 0.10).tolist() == [False, True, True, True, False, False]


def test_category_schema_only_accepts_known_labels():
    L.CategoryBatch(categories=["Beverages", "Baby Care"])
    with pytest.raises(ValidationError):
        L.CategoryBatch(categories=["Cars"])


def test_variants_schema_needs_exactly_three():
    with pytest.raises(ValidationError):
        L.Variants(noisy_variants=["a", "b"], hard_negative="c")


def test_medoid_name_picks_most_central():
    emb = np.array([[1, 0], [0.9, 0.436], [0, 1]], dtype=float)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    assert L.medoid_name(["a", "b", "c"], emb) == "b"


def test_bb_to_google_mapping_covers_all_categories():
    assert set(L.BB_TO_GOOGLE) == set(L.BB_CATEGORIES)


def test_to_hinglish_replaces_words_and_returns_none_otherwise():
    assert F.to_hinglish("Tata Salt 1 kg") == "tata namak 1 kg"
    assert F.to_hinglish("Organic Turmeric Powder") == "organic haldi powder"
    assert F.to_hinglish("Dove Shampoo") is None


def test_back_transliterate_and_spelling_variant():
    r2n = {"chawal": "चावल", "haldi": "हल्दी"}
    assert F.back_transliterate("basmati chawal 5 kg", r2n) == "basmati चावल 5 kg"
    rng = random.Random(0)
    assert F.spelling_variant("chawal 1 kg", {"chawal": ["chaawal"]}, rng) == "chaawal 1 kg"


def test_pipeline_runs_on_a_toy_table():
    from src import pipeline
    raw = pd.DataFrame({
        "Item Title": ["Tata Salt 1kg", "TATA salt 1 kg", "Amul Butter 500 g", "Amul Butter 100 g",
                       "Fortune Sunflower Oil 1 ltr", "Fortune Sunflower Oil 1 L"],
        "Brand": ["Tata", "Tata", "Amul", "Amul", "Fortune", "Fortune"],
        "Selling Price": [28, 28, 275, 56, 150, 150], "MRP": [30, 30, 290, 60, 160, 160]})
    res = pipeline.run(raw, use_llm=False)
    pre = res["preprocessed"]
    assert pre.loc[0, "cluster_id"] == pre.loc[1, "cluster_id"]          # same product
    assert pre.loc[2, "cluster_id"] != pre.loc[3, "cluster_id"]          # 500 g vs 100 g
    hits = pipeline.search({"preprocessed": pre, "embeddings": res["embeddings"]}, "sunflower oil 1 litre", k=2)
    assert hits["product_name"].str.contains("Sunflower").all()


def test_grey_zone_budget_takes_pairs_closest_to_threshold():
    s = np.array([0.10, 0.69, 0.72, 0.95, 0.70])
    assert L.grey_zone_budget(s, 0.70, budget=3).tolist() == [False, True, True, False, True]


def test_synthetic_rule_keeps_product_and_changes_size_for_negative():
    out = L.synthetic_rule("Tata Salt Lite 1 kg", random.Random(0))
    assert len(out["variants"]) == 3 and out["hard_negative"] == "Tata Salt Lite 2 kg"
    assert out["variants"][1] == "tata salt lite 1 kg"
