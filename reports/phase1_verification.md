# Phase 1 — Verification Report (Column standardization and selection)

**Goal (roadmap):** any product table is mapped automatically to one canonical schema, and the text columns worth clustering are picked.
**Done when (roadmap):** four differently shaped files come out with identical column names, plus an accuracy table.
**Status: PASSED.** BigBasket (9+1 cols), Flipkart (15), Open Food Facts India (6) and Abt (4) all come out with the same 11 columns (`data/processed/*_canonical.parquet`). The accuracy table is below. 34/34 tests pass.

## 1. What was built

| File | Purpose |
|---|---|
| `src/columns.py` | rules, embedding and LLM mappers; `map_columns()` (system cascade); `resolve_price_conflict()`; `profile_column()` / `select_columns()`; `to_canonical()`; evaluation |
| `data/gold/header_mapping.csv` | 59 headers from 7 files (BigBasket 10, Flipkart 15, Abt 4, Buy 5, Amazon 5, Google 5, Open Food Facts 15) with the gold field and a gold `cluster_text` flag. **Labelled by Claude; must be checked by the student** |
| `data/raw/off_sample.tsv` | first 12,673 rows of the full OFF export with all 211 original columns (for OFF's real header) |
| `notebooks/01_columns.ipynb` | tables + `reports/figures/phase1_accuracy_vs_calls.png` |
| `tests/test_phase1.py` | 21 tests |
| outputs | `reports/phase1_mapping_accuracy.csv`, `phase1_mapping_predictions.csv`, `phase1_column_selection.csv`, `phase1_mappings_used.csv` |

Canonical schema: `product_name, brand, category, sub_category, price, mrp, quantity, description, rating, id` (+ `source`); columns with no slot are labelled `other` and dropped.

## 2. Requirements check

| Req | How this phase meets it | Evidence |
|---|---|---|
| R1 Standardize columns | Every source column is mapped to the canonical schema; 4 differently shaped files → identical 11 columns | mapping accuracy **0.966** (57/59 headers); `test_canonical_files_have_identical_columns` |
| R2 Select columns with justification | Each column is profiled (avg tokens, unique ratio, letter ratio, entropy, URL ratio). Free-text, high-uniqueness columns are selected; every decision comes with a written reason; the shortest selected column becomes the primary name column | selection P = 0.929, R = 1.000, F1 = **0.963** (13 TP, 1 FP, 0 FN); `phase1_column_selection.csv` |
| R4 Rule-based | keyword rules on header tokens; a value-based price/MRP rule | rules alone 0.898 |
| R5 NLP | header tokenization (snake/camel case); Sentence-BERT sentence embeddings with cosine similarity | embedding rows of the table |
| R6 LLM only where useful | The LLM is asked only about headers no rule recognises: 13 of 59 headers (22 %), JSON output validated by a Pydantic `Literal` schema; 0 invalid replies | LLM-only reaches 0.949 with 59 calls; the system reaches 0.966 with 13 |
| R7 Frameworks | sentence-transformers (all-MiniLM-L6-v2), Ollama + Pydantic | `src/columns.py` |
| R8 Measured on a labelled set | every method scored on the 59-header gold set | table below |

## 3. Results

### 3.1 Header-mapping accuracy (59 gold headers)

| Method | Accuracy | LLM calls |
|---|---|---|
| Rules only (header keywords) — *baseline* | 0.898 | 0 |
| Embeddings, header only | 0.797 | 0 |
| Embeddings, header + 5 sample values (*roadmap method*) | 0.644 | 0 |
| LLM only (qwen3:8b) | 0.949 | 59 |
| Embeddings (header + samples) + LLM if cosine < 0.3 / 0.4 / 0.5 / 0.6 | 0.729 / 0.864 / 0.949 / 0.949 | 8 / 24 / 46 / 57 |
| Embeddings (header only) + LLM if cosine < 0.3 / 0.4 / 0.5 / 0.6 | 0.915 / 0.966 / 0.949 / 0.949 | 13 / 29 / 45 / 59 |
| Cascade: rules if a keyword fires, else LLM | 0.932 | 13 |
| **System: cascade + price/MRP value rule** | **0.966** | **13** |

LLM time for the 59 calls: 215 s (≈ 3.6 s per call; short outputs).

### 3.2 Column selection (gold: 14 free-text columns among 59)

| Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|
| 0.929 | 1.000 | 0.963 | 13 | 1 | 0 |

The primary (name) column chosen per file was correct in all 7 files: `product`, `product_name`, `name`, `title`, `name`, `name`, `product_name`.

## 4. Examples

| # | Input (header + samples) | Output | Note |
|---|---|---|---|
| 1 | Flipkart `pid` (`SWSEBWYCGGNCRXJH`, `NKCDZHF7GVHHGZRJ`, …) | no rule fires → LLM → `id` | correct; the LLM adds value on headers that carry no keyword |
| 2 | Flipkart `retail_price` (999, 32157, …) and `discounted_price` (379, 22646, …) | rules: both `price` → value rule: higher median → `retail_price` = `mrp` | correct after the rule; the rules alone gave both `price` |
| 3 | Buy `description` (e.g. "5 x 10/100Base-TX LAN") | embeddings with samples → `quantity` (cos 0.368) | **failure of the roadmap method:** the sample values dominate the embedded text, and "5 x 10/100" looks like a quantity. Header-only embeddings get it right |
| 4 | BigBasket `type` ("Nappies & Rash Cream", "Hair Removal", …) | LLM → `sub_category`; gold `other` | **system error**, but a defensible one: `type` is a third-level category, and the schema has only two category levels |
| 5 | OFF `serving_size` ("84g", "1 portion (405 g)", …) | rule keyword "size" → `quantity`; gold `other` | **system error:** keyword rules cannot tell serving size from pack size |
| 6 | Flipkart `product_category_tree` | selected as clustering text (14 tokens, 32 % unique, 69 % letters) | **selection false positive:** a long category path passes the free-text thresholds |

## 5. Discussion

- **The roadmap's method was the weakest variant (0.644).** Concatenating 5 sample values to the header lets the values dominate the sentence embedding, so long description samples are mistaken for names, categories or quantities. Header-only embeddings reach 0.797. Simple keyword rules (0.898) beat both, because product-table headers are short and conventional.
- **MiniLM cosine scores are low for this task** (median 0.43, max 0.62), so the roadmap's 0.6 threshold sends 57 of 59 headers to the LLM. That is not a fallback in any useful sense.
- **The final cascade** (rules when a keyword fires, otherwise the LLM, plus the MRP ≥ price rule) matches the best accuracy (0.966) while calling the LLM on only 22 % of headers.
- **Caveat on tuning:** with only 59 gold headers there is no separate validation split. The cascade design and the price/MRP rule were chosen after looking at the errors on this set, so 0.966 is optimistically biased. The roadmap configuration (0.644 / 0.949) and the rules baseline (0.898) were not tuned. The 0.6 threshold is the roadmap's a-priori value.

## 6. Not yet working / limitations

- The gold header set and its `cluster_text` labels were made by Claude, not by hand. **The student must check all 59 rows** (`data/gold/header_mapping.csv`) before submission; the numbers above are provisional until then.
- `quantity` is empty for BigBasket, Flipkart and Abt because those files have no quantity column; it is filled from product names by Phase 2/3 parsing.
- OFF India categories are multilingual ("Boissons et préparations de boissons"); they are left as they are.
- When two columns map to one field (e.g. Flipkart `product_rating` and `overall_rating`), the first is kept and the other dropped.

## 7. Report section draft — 5.1 Column standardization and selection

**Method.** Each input table is mapped to a ten-field canonical schema (`product_name, brand, category, sub_category, price, mrp, quantity, description, rating, id`); columns that fit none of these fields are labelled *other* and dropped. We compare four mappers. (i) *Keyword rules* tokenize the header (snake case and camel case) and assign the first field whose keyword list it matches. (ii) *Semantic matching* embeds a text built from the header and five sample values with Sentence-BERT (all-MiniLM-L6-v2) and assigns the field whose natural-language description has the highest cosine similarity; a header-only variant is also tested. (iii) An *LLM* (qwen3:8b via Ollama) receives the header, the samples and the field definitions, and must answer with JSON validated against a Pydantic enumeration. (iv) The *system cascade* applies the keyword rules when one fires and otherwise asks the LLM; a value-based rule then resolves the common case of two price columns by assigning the one with the higher median to *mrp*, since a maximum retail price cannot be below the selling price. For column selection, each column is profiled by mean token count, unique-value ratio, alphabetic-character ratio, entropy and URL ratio. Columns that are mostly filled, mostly letters, at least two tokens long, at least 30 % unique and not URLs are selected as clustering text, and the shortest selected column is used as the product name.

**Setup.** We built a gold set of 59 headers from seven files (BigBasket, Flipkart, Abt, Buy, Amazon, Google and the original 211-column Open Food Facts export), each labelled with its canonical field and with whether it is free text worth clustering.

**Results.** Table X reports mapping accuracy and the number of LLM calls. Keyword rules reach 0.898. Semantic matching with sample values reaches only 0.644, because the sample values dominate the embedding: description columns are mistaken for names, categories or quantities. Header-only embeddings reach 0.797. The LLM alone reaches 0.949 but needs one call per column. The cascade reaches 0.966 while calling the LLM for only 13 of 59 headers (22 %), with no invalid outputs. Column selection achieves precision 0.929 and recall 1.000; its only false positive is Flipkart's category-path column.

**Discussion.** For short, conventional headers, cheap lexical rules are both accurate and free. The LLM is valuable precisely on headers without informative keywords (`pid`, `index`, `product`), which is where the cascade uses it. Because the gold set is small, the cascade and the price/MRP rule were designed on the same data they were evaluated on, so their scores should be read as optimistic.
