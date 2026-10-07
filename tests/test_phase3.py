"""Phase 3 tests: span alignment, scoring, rule extraction, schema validation."""
import pandas as pd
import pytest
from pydantic import ValidationError

from src import extract as X


def test_spans_from_attributes_finds_values_case_insensitively():
    text = "Alisha Solid Women's Cycling Shorts, Pack of 3, Black"
    attrs = {"brand": "alisha", "product_type": "Cycling Shorts", "variant": None, "quantity": None,
             "pack_count": 3, "colour": "Black"}
    spans = X.spans_from_attributes(text, attrs)
    got = {label: text[s:e] for s, e, label in spans}
    assert got == {"BRAND": "Alisha", "TYPE": "Cycling Shorts", "PACK": "Pack of 3", "COLOUR": "Black"}


def test_spans_skip_values_not_in_text_and_overlaps():
    text = "Tata Salt Lite 1 kg"
    spans = X.spans_from_attributes(text, {"brand": "Tata Chemicals", "product_type": "Salt Lite",
                                           "variant": "Lite", "quantity": "1 kg"})
    labels = [l for _, _, l in spans]
    assert "BRAND" not in labels                      # "Tata Chemicals" is not in the text
    assert "TYPE" in labels and "VARIANT" not in labels  # "Lite" overlaps the longer "Salt Lite"


def test_norm_value_pack_and_text():
    assert X.norm_value("pack_count", "24/Box") == "24"
    assert X.norm_value("quantity", "1kg") == X.norm_value("quantity", "1000 GM") == "1000 g"
    assert X.norm_value("brand", None) is None


def test_score_counts_wrong_prediction_as_fp_and_fn():
    gold = pd.DataFrame({a: [set()] * 2 for a in X.EVAL_ATTRS})
    gold["brand"] = [{"Sony"}, {"HP"}]
    preds = [{"brand": "sony"}, {"brand": "Dell"}]
    s = X.score(preds, gold).set_index("attribute")
    assert s.loc["brand", "precision"] == 0.5 and s.loc["brand", "recall"] == 0.5


def test_score_lenient_accepts_containment():
    gold = pd.DataFrame({a: [set()] for a in X.EVAL_ATTRS})
    gold["product_type"] = [{"Stamp Dispenser"}]
    preds = [{"product_type": "Dispenser"}]
    assert X.score(preds, gold).set_index("attribute").loc["product_type", "f1"] == 0.0
    assert X.score(preds, gold, lenient=True).set_index("attribute").loc["product_type", "f1"] == 1.0


def test_rules_extract_brand_colour_quantity_pack():
    rec = X.extract_rules(["Sony Black Headphones 250 g Pack of 2"])[0]
    assert rec["brand"] == "Sony" and rec["colour"] == "Black"
    assert rec["pack_count"] == 2 and rec["quantity"] == "250 g"


def test_pydantic_schema_rejects_bad_types():
    X.ProductAttributes(brand="Amul", product_type="Butter", variant=None, quantity="500 g",
                        pack_count=None, colour=None)
    with pytest.raises(ValidationError):
        X.ProductAttributes(brand="Amul", product_type="Butter", variant=None, quantity="500 g",
                            pack_count="many", colour=None)


def test_distillation_names_are_balanced_and_deterministic():
    a, b = X.distillation_names(10), X.distillation_names(10)
    assert a == b and len(a) == 10
