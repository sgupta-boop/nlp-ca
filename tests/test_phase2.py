"""Phase 2 tests: quantity parser, number parsing, text normalization, spelling, lemmatization."""
import random
from collections import Counter

import pandas as pd
import pytest

from src import preprocess as P


@pytest.mark.parametrize("text,value,unit", [
    ("Tata Salt 1kg", 1000, "g"),
    ("Aashirvaad Atta 1000 gm", 1000, "g"),
    ("Fortune Oil 1 ltr", 1000, "ml"),
    ("Pepsi 2.25L", 2250, "ml"),
    ("Garlic Capsule 500 mg", 0.5, "g"),
    ("Amul Butter 100 GMS", 100, "g"),
    ("Chips 1.5 oz", 42.524, "g"),
])
def test_quantity_value_and_unit(text, value, unit):
    q = P.parse_quantity(text)
    assert q["qty_unit"] == unit and q["qty_value"] == pytest.approx(value, rel=1e-3)


@pytest.mark.parametrize("text,count", [
    ("Maggi Noodles 70 g (Pack of 6)", 6), ("Dove Soap 3 x 100g", 3), ("Cotton Buds 100 pcs", 100),
    ("Pens 24/Box", 24), ("Set of 2 Bowls", 2), ("Tata Salt 1 kg", None),
])
def test_pack_count(text, count):
    assert P.parse_quantity(text)["pack_count"] == count


def test_same_quantity_different_spelling_gives_same_text():
    assert P.normalize_text("Atta 1kg") == P.normalize_text("ATTA 1000 gm") == "atta 1000 g"


def test_two_in_one_is_not_inches():
    assert P.parse_quantity("Clinic Plus 2 in 1 Shampoo")["qty_value"] is None


@pytest.mark.parametrize("text,value", [("3-1/2", 3.5), ("8 1/2", 8.5), ("4/5", 0.8), ("1,000", 1000.0), (".5", 0.5)])
def test_parse_number_fractions(text, value):
    assert P.parse_number(text) == pytest.approx(value)


def test_normalize_text_unicode_case_punctuation():
    assert P.normalize_text("Ｂritannia  Good-Day™ Cookies, 200gm!") == "britannia good day cookies 200 g"
    assert P.normalize_text("Rice & Dal 1.5kg") == "rice and dal 1500 g"


@pytest.mark.parametrize("kind,raw,gold", [
    ("length", '3-1/2"', "8.9"), ("length", "48", "121.9"), ("weight_g", "2.5 oz", "71"),
    ("weight_kg", "20-lb.", "9.1"), ("count", "24/Box", "24"), ("speed", "15K", "15000"),
    ("named", "128MB", "128 Megabytes"), ("named", "16 lbs.", "16 lbs."),
])
def test_normalize_like_pave(kind, raw, gold):
    assert P.normalize_like_pave(kind, raw) == gold


def test_abbreviations_expand():
    assert P.expand_abbreviations(["choco", "bar", "pkt"]) == ["chocolate", "bar", "packet"]


def test_spell_corrector_fixes_unseen_typo_but_not_seen_words():
    counts = Counter({"chocolate": 50, "biscuit": 40, "toaster": 1, "toasted": 30})
    sc = P.SpellCorrector(counts)
    assert sc.correct_token("chocolat") == "chocolate"     # unseen typo -> fixed
    assert sc.correct_token("toaster") == "toaster"         # seen once -> trusted
    assert sc.correct_token("tea") == "tea"                 # too short to touch


def test_add_typo_changes_word_by_one_edit():
    rng = random.Random(0)
    w = "chocolate"
    assert all(abs(len(P.add_typo(w, rng)) - len(w)) <= 1 for _ in range(50))


def test_lemmatize_lowercased_plurals():
    assert P.lemmatize_many(["britannia good day cashew biscuits 200 g"]) == ["britannia good day cashew biscuit 200 g"]


def test_preprocess_table_adds_columns():
    df = pd.DataFrame({"product_name": ["Tata Salt 1kg", "Maggi Noodles 70 g (Pack of 4)"]})
    out = P.preprocess_table(df)
    assert {"name_clean", "name_norm", "qty_value", "qty_unit", "pack_count"} <= set(out.columns)
    assert out.loc[1, "pack_count"] == 4 and out.loc[0, "qty_value"] == 1000
