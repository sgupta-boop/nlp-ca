# Phase 0 — Verification Report (Setup and data exploration)

**Goal (roadmap):** every dataset downloaded, loaded and understood.
**Done when (roadmap):** one notebook shows a profile of BigBasket, Flipkart and Abt-Buy side by side.
**Status: PASSED.** All 9 datasets are downloaded and loaded, 13/13 tests pass, and `notebooks/00_eda.ipynb` executes end to end and produces the side-by-side profile (`reports/phase0_profile.csv`).

## 1. What was built

| File | Purpose |
|---|---|
| `scripts/download_data.py` | Downloads all 9 datasets into `data/raw/`; resumable, atomic (`.part` then rename), skips finished steps |
| `scripts/check_env.py` | Library versions, GPU check, spaCy smoke test, qwen3:8b speed benchmark |
| `src/data_io.py` | One loader per dataset; `pave_attribute_pairs()` flattens WDC-PAVE labels |
| `src/eda.py` | `profile_table()`, `unit_mentions()`, `null_percent()`, `top_values()` |
| `src/llm.py` | The only gateway to the LLM: JSON-schema output, Pydantic validation and retries, cache in `cache/llm_responses.json`, call/time/token counters, GPU out-of-memory fallback, `unload()` |
| `notebooks/00_eda.ipynb` | Inventory, side-by-side profile, schemas, null %, samples, name-length plot, unit spellings, brands, categories, duplicates, WDC and PAVE summaries |
| `tests/test_phase0.py` | 13 tests |

## 2. Requirements check

| Req | How this phase meets it | Evidence |
|---|---|---|
| R1 Standardize columns | Not solved yet (Phase 1). This phase measures the problem: 6 sources, 6 different schemas, e.g. the product name is called `product`, `product_name`, `name` or `title`, and the brand `brand`, `manufacturer` or `brands` | notebook §3; profile row "name column" |
| R2 Select columns | Not yet (Phase 1). Null-% and uniqueness profiles that Phase 1 will use are computed | notebook §3 null-% table |
| R3 Group similar names | Not yet (Phase 5). Size of the problem measured: BigBasket has 4,043 duplicate lower-cased names (7,034 rows in 2,991 groups) | notebook §8 |
| R4 Rules + string similarity + cosine | Not yet (Phases 2–5) | – |
| R5 NLP-oriented | spaCy tokenizer and lemmatizer verified; unit-spelling variety measured (18 spellings in BigBasket, 20 in Flipkart) to motivate Phase 2 | `reports/phase0_check_env.txt`; notebook §6 |
| R6 LLM only where useful | `src/llm.py` built as the single, counted, cached LLM gateway; qwen3:8b speed measured so later LLM use can be sized | 5/5 valid JSON, 0 failures, 9.33 s/call |
| R7 Modern frameworks | spaCy 3.8.16, sentence-transformers 6.1.0, GLiNER 0.2.29, BERTopic 0.17.4, FAISS 1.15.1, UMAP 0.5.12, gensim 4.4.0, PyTorch 2.6.0+cu124 (CUDA on GTX 1650 Ti), Ollama | `reports/phase0_check_env.txt` |
| R8 Numbers on labelled sets | All gold/benchmark sets are downloaded and their sizes verified against the published numbers by tests | `tests/test_phase0.py` (13 passed) |

## 3. Measured results

### 3.1 Dataset inventory (all loaded)

| Dataset | Rows | Cols | Matches roadmap? |
|---|---|---|---|
| BigBasket | 27,555 | 9 (+ dropped row-index column) | yes (roadmap counts the index column: 10) |
| Flipkart | 20,000 | 15 | yes |
| Abt / Buy / matches | 1,081 / 1,092 / 1,097 | 4 / 5 / 2 | yes |
| Amazon / Google / matches | 1,363 / 3,226 / 1,300 | 5 / 5 / 2 | yes |
| WDC Products 80% pairs: test / train-large | 4,500 / 19,835 | 17 | – |
| WDC Products 80% multi-class test | 1,000 offers, 500 products | 9 | – |
| WDC-PAVE (train + test) | 565 offers | 7 | yes (565) |
| Open Food Facts, India rows | 21,189 | 6 | – (roadmap gives no number) |
| Aksharantar Hindi train | 1,299,155 pairs | 5 | yes (~1.3 M) |
| Google Product Taxonomy | 5,595 paths, 21 top-level | 3 | yes (version 2021-09-21) |

### 3.2 Side-by-side profile (the "done when" deliverable)

| | BigBasket | Flipkart | Abt | Buy | Amazon | Google | OFF India |
|---|---|---|---|---|---|---|---|
| rows | 27,555 | 20,000 | 1,081 | 1,092 | 1,363 | 3,226 | 21,189 |
| name column | product | product_name | name | name | title | name | product_name |
| mean null % (all columns) | 3.5 | 2.0 | 15.3 | 17.4 | 1.7 | 19.7 | 27.8 |
| null % in name | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 16.32 |
| exact duplicate rows | 354 | 0 | 0 | 0 | 0 | 0 | 0 |
| duplicate names (lower-cased) | 4,043 | 7,377 | 0 | 13 | 18 | 205 | 3,372 |
| median name length (chars) | 30 | 39 | 55 | 48 | 33 | 55 | 17 |
| mean name length (tokens) | 5.7 | 6.4 | 9.1 | 8.3 | 5.7 | 9.3 | 3.0 |
| names containing a quantity (%) | 4.1 | 5.5 | 9.1 | 8.2 | 0.6 | 0.9 | 8.9 |
| distinct unit spellings | 18 | 20 | 10 | 7 | 4 | 7 | 22 |
| distinct brands | 2,313 | 3,499 | – | 116 | 350 | 67 | 5,016 |

### 3.3 Environment and LLM benchmark (`reports/phase0_check_env.txt`)

| Measure | Value |
|---|---|
| qwen3:8b load + first call | 69.0 s |
| Placement | 5.6 GiB, of which 2.1 GiB (39 %) on the GPU, the rest on the CPU |
| Speed | 3.7 output tokens/s, **9.33 s per extraction call** |
| Validity | 5/5 replies valid against the Pydantic schema, 0 retries, 0 failures |
| Projection | 2,000 LLM labels (Phase 3 distillation) ≈ 5.2 h |
| Free RAM while the model is loaded | 0.23 GB (3.6 GB after unloading) |

## 4. Examples (input → output)

| # | Input | Output | Note |
|---|---|---|---|
| 1 | `Tata Salt Lite Low Sodium 1 kg` → qwen3:8b | `{brand: Tata, product_type: Salt, quantity: 1.0, unit: kg}` | correct |
| 2 | `Maggi 2-Minute Masala Noodles pack of 4` → qwen3:8b | `{brand: Maggi, product_type: Noodles, quantity: 4.0, unit: pack}` | correct; pack count was put in the quantity field (the Phase 3 schema needs a separate `pack_count`) |
| 3 | `Britannia Good Day Cashew Biscuits 200 gm` → spaCy lemmas | `Britannia good Day Cashew Biscuits 200 gm` | **Failure:** "Biscuits" is not lemmatized. The small English model tags capitalised words as proper nouns, and proper nouns are not lemmatized |
| 4 | same text, lower-cased first | `britannia good day cashew biscuit 200 gm` | Fix confirmed: Phase 2 must lowercase *before* lemmatizing |
| 5 | BigBasket `100% Green Tea` (Sprig Tea) | appears twice, ₹141.55 and ₹62.10 | **Hard case:** identical name, probably different pack sizes. A name-only matcher must call them the same product |

## 5. Findings that change later phases

1. **BigBasket names rarely contain the brand or the quantity.** Brand is a separate column, only 4.1 % of names contain a quantity, and typical names are "White Quinoa" or "Drink Mix - Badam". So (a) in Phases 4–5 the text to match on should be `brand + product_name`, not the name alone; (b) the Phase 3 brand gazetteer and the Phase 5 quantity veto will rarely fire on BigBasket, and will be evaluated on PAVE / WDC instead, where names carry those attributes; (c) the 300 BigBasket gold pairs must include same-name/different-price cases like example 5.
2. **WDC-PAVE attribute labels differ from our NER labels**, as anticipated. The `wdc_with_all_attributes` variant has 4,722 attribute values (roadmap: 4,687; we count each value of a multi-valued attribute separately). Its labels are Product Type 591, Brand 135 + Manufacturer 227, Color(s)/Color 167, Pack Quantity 99, Capacity 97, Size/Weight 28 and others. Phase 3 will map these onto brand, product type, colour, pack count and quantity, and score only those. Only 32 of the 565 offers are grocery; most are computers, home and office.
3. **Flipkart's brand column is noisy:** colours and fits appear among its top brands ("Black" 167, "White" 155, "Regular" 313, "Slim" 288), and 29.3 % of values are missing. This motivates Phase 1 column checks and Phase 3 extraction from the name.
4. **WDC 80 % pair test set is imbalanced:** 500 matches and 4,000 non-matches. Use F1 on the positive class, not accuracy.
5. **Hardware limits:** the 4 GB GPU cannot hold a PyTorch CUDA context and Ollama's share of qwen3:8b at once (reproduced: `cudaMalloc failed: out of memory`), and with the model loaded only 0.23 GB of RAM is free (an intermittent UMAP import crash). Rules adopted: LLM stages run separately from embedding stages; `llm.unload()` runs after every LLM stage; `llm.py` retries with fewer GPU layers on out-of-memory.

## 6. Not yet working / limitations, stated plainly

- No metric for our own method exists yet. Phase 0 has nothing to score; the first accuracy table comes in Phase 1.
- The three hand-labelled gold sets (header mapping, 300 BigBasket pairs, 200 Hinglish names) do not exist yet. They are built in Phases 1, 5 and 7, and **you** must check every label.
- The LLM benchmark has only 5 names. It measures speed, not accuracy; accuracy is measured on WDC-PAVE in Phase 3.
- 16.3 % of OFF India names are empty, and some "names" are barcodes (e.g. `8906080603761`).
- Download problems solved along the way: Hugging Face rate-limited the OFF Parquet (HTTP 429), so we use the official OFF CSV export; OFF rejects the default Python User-Agent (403); the network dropped large downloads, so downloads now resume.

## 7. Report section draft — 3. Datasets (partial) and 4.1 Experimental setup

**3. Datasets.** We use two Indian retail catalogues as working data and four public benchmarks with gold labels for evaluation. *BigBasket* (27,555 grocery products, 11 categories, 90 sub-categories, 2,313 brands) and *Flipkart* (20,000 general e-commerce listings) have different schemas: the product name is stored as `product` and `product_name` respectively, and Flipkart encodes its category as a path (`Clothing >> Women's Clothing >> …`). A third Indian source, the India subset of *Open Food Facts* (21,189 products), is the noisiest: 16.3 % of products have no name and the mean name has only 3.0 tokens. For matching we use *Abt-Buy* (1,081 × 1,092 records, 1,097 gold matches) and *Amazon-Google* (1,363 × 3,226, 1,300 matches), and *WDC Products* (80 % corner cases; 4,500 test pairs of which 500 are matches, and a 1,000-offer multi-class test set over 500 products) for both pair-wise matching and clustering. For attribute extraction we use *WDC-PAVE* (565 offers, 4,722 attribute values). *Aksharantar* (1.3 M Hindi–Roman word pairs) supports transliteration experiments, and the *Google Product Taxonomy* (5,595 categories, 21 top-level) supplies category labels.

Profiling shows why entity resolution is needed and where it is hard. BigBasket contains 4,043 duplicate names (case-insensitive), forming 2,991 groups, but identical names do not always mean the same product: "100% Green Tea" by the same brand is listed at ₹141.55 and ₹62.10. Quantities are written in at least 18 different unit spellings (e.g. "g", "gm", "gr"; "l", "lt", "ltr"), yet only 4.1 % of BigBasket names contain a quantity at all, because the catalogue stores brand and pack size outside the name. Flipkart's brand column is 29.3 % empty and contains colours and fits ("Black", "Slim") as brands.

**4.1 Experimental setup.** All experiments ran on a laptop (Windows 11, Python 3.11.9, 7.4 GB RAM, NVIDIA GTX 1650 Ti with 4 GB VRAM). The LLM is qwen3:8b served locally by Ollama, with reasoning mode disabled, temperature 0 and seed 42. Every reply is constrained to a JSON schema, validated with Pydantic and cached, so all reported numbers are reproducible without re-querying the model. On this hardware the model runs 39 % on the GPU and generates 3.7 tokens/s (9.3 s per extraction), which is why the system sends to the LLM only the cases the cheaper methods cannot decide.
