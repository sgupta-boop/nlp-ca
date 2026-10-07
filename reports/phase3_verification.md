# Phase 3 — Verification Report (Attribute extraction / NER)

**Goal (roadmap):** every product name becomes structured fields: brand, product type, variant/flavour, quantity, unit, pack count, colour.
**Done when (roadmap):** an F1 table of 4 methods × attributes, plus displaCy screenshots.
**Status: PASSED, with one deviation.** The F1 table is below. displaCy renderings are in `reports/figures/phase3_displacy_distilled.html` and `phase3_displacy_rules.html` (open in a browser to screenshot) and inline in `notebooks/03_extract.ipynb`. 8/8 Phase 3 tests pass.
**Deviation:** the distilled model's teacher is **GLiNER, not the LLM**. qwen3:8b takes 13.1 s per name on this laptop, so 2,000 labels would take about 7.3 h. GLiNER labelled the same 2,000 names in 89 s. Decision agreed with the student.

## 1. What was built (`src/extract.py`)

| Method | How |
|---|---|
| 1. Rules | spaCy `EntityRuler` with a **brand gazetteer** (8,000+ brands from the BigBasket, Flipkart, Buy, Amazon and Google brand/manufacturer columns; PAVE is not used) and a **colour gazetteer**, matched on lower-cased tokens; quantity and pack count from the Phase 2 regex and spaCy `Matcher`; product type = last noun chunk of the first segment (spaCy noun chunks) |
| 2. GLiNER (zero-shot) | `urchade/gliner_medium-v2.1` with our labels ("brand", "product type", "flavour or variant", "quantity or size", "pack count", "colour"), threshold 0.4, best span per label, no training |
| 3. LLM | qwen3:8b, 3-shot prompt, output forced to the JSON schema of a Pydantic model `ProductAttributes` and validated; values must be copied from the name |
| 4. Distilled spaCy NER | GLiNER labels 2,000 names (1,000 BigBasket "brand + name", 1,000 Flipkart); values are aligned back to character spans; a blank spaCy NER is trained for 30 epochs (seed 42) |

**Evaluation:** the 354 WDC-PAVE test offers, title only. Gold = PAVE values that occur in the title (`pid` 0): brand ← Brand or Manufacturer; product type ← Product Type; colour ← Color / Color(s); pack count ← Pack Quantity; quantity ← Capacity or Size/Weight. PAVE has no "variant" attribute, so variant is extracted but **not scored**. *Strict* = identical after Phase 2 `normalize_text`; *lenient* = one value contains the other. A wrong prediction counts as both a false positive and a false negative.

## 2. Requirements check

| Req | How this phase meets it | Evidence |
|---|---|---|
| R4 Rule-based | EntityRuler + gazetteers, regex/Matcher quantities, noun-chunk heuristic | rules macro F1 0.340 |
| R5 NER / attribute extraction | four NER approaches: rule-based, zero-shot neural, LLM, distilled statistical | F1 table below |
| R6 LLM where it adds value | the LLM is the most accurate extractor (macro F1 0.592) but 13,500× slower than the distilled model, so it is not used to tag whole catalogues | timing table |
| R7 Frameworks | spaCy (EntityRuler, Matcher, noun chunks, NER training, displaCy), GLiNER, Ollama + Pydantic | `src/extract.py` |
| R8 Labelled test set | WDC-PAVE test, 803 gold attribute values in 354 titles | tables below |

## 3. Results

### 3.1 F1 on WDC-PAVE test titles (strict match)

| Attribute (gold n) | Rules | GLiNER | LLM (qwen3:8b) | Distilled spaCy |
|---|---|---|---|---|
| brand (346) | 0.409 | 0.769 | **0.844** | 0.413 |
| product type (254) | 0.089 | 0.186 | **0.526** | 0.060 |
| colour (70) | **0.730** | 0.697 | 0.701 | 0.311 |
| quantity (87) | 0.163 | 0.234 | **0.481** | 0.042 |
| pack count (46) | 0.309 | **0.637** | 0.408 | 0.000 |
| **macro** | 0.340 | 0.505 | **0.592** | 0.165 |
| macro precision / recall | 0.356 / 0.359 | 0.490 / 0.523 | 0.542 / 0.659 | 0.186 / 0.149 |

Lenient macro F1: rules 0.443, GLiNER 0.592, LLM **0.683**, distilled 0.273. The LLM's product-type F1 rises from 0.526 to 0.816 under lenient matching: it usually finds the right head noun but includes more or fewer modifiers than the gold value.

### 3.2 Speed on the same 354 titles

| Method | Names / second | Time for 354 |
|---|---|---|
| Distilled spaCy NER (CPU) | **950** | 0.4 s |
| Rules (CPU) | 450 | 0.8 s |
| GLiNER (GPU) | 14 | 25 s |
| LLM qwen3:8b (39 % on GPU) | 0.07 | 4,791 s (80 min); 354 calls, 0 invalid JSON |

### 3.3 Distillation learning curve (teacher = GLiNER)

| Teacher-labelled names | Training spans kept | Macro F1 (strict) | brand | colour |
|---|---|---|---|---|
| 250 | 220 | 0.088 | 0.289 | 0.101 |
| 500 | 441 | 0.108 | 0.345 | 0.147 |
| 1,000 | 878 | 0.145 | 0.422 | 0.215 |
| 2,000 | 1,760 | 0.165 | 0.413 | 0.311 |

## 4. Examples (from `notebooks/03_extract.ipynb` §5)

| # | Title | Output | Note |
|---|---|---|---|
| 1 | `Duke 416S-3636-5R 36" 16-ga Work Table w/ Undershelf …` | LLM: brand Duke, type **Work Table** | correct; gold = Duke / Work Table. GLiNER and the rules picked "Stainless Top" / "Backsplash Top" |
| 2 | `435952-B21 HP Xeon E5335 2.0GHz DL360 G5" Null` | rules/GLiNER/LLM: brand HP; distilled: brand **"435952"** | **failure (distilled):** trained on grocery/fashion names, where the first token is usually the brand, so it labels the leading part number as the brand |
| 3 | `Regal Leather Business Card Binder, 120 Cap, … Black Binder by Samsill` | LLM: brand Samsill, type Business Card Binder, colour Black, quantity "120 Cap" | brand/type/colour correct; the quantity misses strict match ("120 cap" vs gold "120") but passes lenient |
| 4 | `Read Right/Advantus PhoneKleen Disinfecting Wipes …` | rules: brand **"Right"**, type "Read Wipes" | **failure (rules):** the gazetteer contains "Right" as a brand, and the noun-chunk heuristic merges the brand into the type |
| 5 | `313611-B21 HP PII 350MHz 512K Null` | GLiNER: brand **"313611-B21"** | **failure (GLiNER):** zero-shot NER mistakes part numbers for brands |

## 5. Discussion

- **The LLM is the best extractor** (macro 0.592, brand 0.844, product type 0.526). This is the attribute where world knowledge matters: the LLM knows that Samsill is a brand and that a "Work Table" is the product.
- **GLiNER is the best fast method:** 0.505 at 14 names/s with no training, best on pack count (0.637).
- **Cheap rules win on closed vocabularies:** colour 0.730 with a 50-word list.
- **The distilled model is fast but weak:** 0.165 at 950 names/s. Three causes: (i) **domain shift**, since it trained on Indian grocery/fashion names but PAVE is computers, office and jewellery; (ii) **silver labels**, since it learns GLiNER's mistakes and GLiNER's quantity/pack spans rarely align with the text, giving pack count 0.000; (iii) a **small training set**. The learning curve is still rising (0.088 → 0.165), so more or better teacher labels would help.
- **Consequence for the system:** the rule-based extractor feeds the Phase 5 matching rules, because it is fast and its precision on brand/quantity is what the veto rules need. The LLM is reserved for the uncertain pairs (Phase 6).

## 6. Not yet working / limitations

- **The distilled model was taught by GLiNER, not by the LLM.** It reaches only 0.165 and is not used downstream except as a demo in the app.
- Variant/flavour is extracted by GLiNER, the LLM and the distilled model, but cannot be scored: no gold exists in PAVE.
- PAVE gold covers only values written in the title; some offers have no brand in the title (346 of 354 do).
- displaCy output is saved as HTML; PNG screenshots must be taken from the browser.

## 7. Report section draft — 5.3 Attribute extraction

**Method.** We extract brand, product type, variant, quantity, pack count and colour from product names with four methods. (i) A *rule-based* spaCy pipeline: an `EntityRuler` with phrase patterns from a brand gazetteer (8,000+ brands taken from the brand and manufacturer columns of our catalogues) and a colour list, the Phase 2 quantity grammar and pack-count `Matcher`, and a noun-chunk heuristic for the product type. (ii) *Zero-shot NER* with GLiNER (`gliner_medium-v2.1`), given only the attribute names as labels. (iii) An *LLM* (qwen3:8b) with a three-example prompt, whose output is constrained to and validated against a Pydantic schema. (iv) A *distilled* spaCy NER model trained on 2,000 product names automatically labelled by GLiNER (labels aligned back to character spans). The original plan used the LLM as teacher, but at 13 s per name this was infeasible on our hardware.

**Setup.** All methods are evaluated on the 354 test offers of WDC-PAVE using titles only. Gold values are those that occur in the title. Brand includes PAVE's *Manufacturer*, quantity includes *Capacity* and *Size/Weight*. A prediction is correct under *strict* matching if it equals a gold value after normalization, and under *lenient* matching if one contains the other.

**Results.** The LLM obtains the highest macro F1 (0.592 strict, 0.683 lenient), with the largest advantage on product type (0.526 vs ≤ 0.186 for the other methods). GLiNER reaches 0.505 without any training and is best on pack counts. The rule-based pipeline reaches 0.340 overall but is best on colour (0.730). The distilled model reaches only 0.165, though its learning curve is still rising, and it is the fastest method (950 names/s, against 0.07 names/s for the LLM).

**Discussion.** Accuracy and cost are inversely related across the four methods by roughly four orders of magnitude in speed. The LLM's advantage comes from world knowledge about brands and product types. The distilled model inherits its teacher's errors and suffers from the domain shift between Indian grocery training names and the electronics- and office-heavy test set. These results motivate the system's design: fast rules supply attributes for the matching rules, and the LLM is called only for the small number of uncertain decisions.
