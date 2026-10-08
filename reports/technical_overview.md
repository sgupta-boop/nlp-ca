# Technical overview

**Title:** LLM-Assisted Product Entity Resolution: Attribute Extraction, Semantic Embeddings and Hybrid Matching for Product Name Standardization

## 1. Pipeline at a glance

```
raw CSV (any column names)
  │
  ├─ 1. Column mapping          keyword rules → LLM if no rule fires → price/MRP value rule      src/columns.py
  ├─ 2. Column selection        profile each column (tokens, uniqueness, letters, entropy, URLs)  src/columns.py
  ├─ 3. Text normalization      NFKC, lowercase, units "1kg"→"1000 g", abbreviations,             src/preprocess.py
  │                              spelling (SymSpell), lemmas (spaCy)
  ├─ 4. Attribute extraction    brand, type, colour, quantity, pack count                          src/extract.py
  │                              (spaCy EntityRuler, GLiNER, LLM, distilled spaCy NER)
  ├─ 5. Representations         TF-IDF (word, char), fastText, Sentence-BERT; FAISS search        src/embed.py
  ├─ 6. Matching                blocking (FAISS top-20 / same brand) → hybrid score               src/match.py
  │                              (SBERT cosine + char TF-IDF cosine) → rules (size, pack, brand,
  │                              model number, MRP) → threshold
  ├─ 7. Clustering              connected components / agglomerative / HDBSCAN                   src/match.py
  ├─ 8. LLM layer               grey-zone pairs, canonical names, categories                      src/llm_layer.py
  └─ 9. App                     upload → mapping → attributes → clusters → search                app/streamlit_app.py
```

## 2. NLP tools and models

| Tool / model | Type | Used for | Phase |
|---|---|---|---|
| **spaCy 3.8** `en_core_web_sm` | NLP library + small English pipeline | tokenization, lemmatization, noun chunks (product type), part-of-speech tags | 2, 3 |
| spaCy **`Matcher`** | rule-based token patterns | pack counts: "pack of 6", "3 x 100 g", "12 pcs", "24/box" | 2 |
| spaCy **`EntityRuler`** | rule-based NER | brand gazetteer (8,000+ brands), colour list | 3 |
| spaCy **NER training** | statistical NER | small, fast model distilled from GLiNER labels | 3 |
| spaCy **displaCy** | visualization | entity highlighting (`reports/figures/phase3_displacy_*.html`) | 3 |
| Regular expressions | rules | quantities and units, fractions ("3-1/2"), model codes | 2, 5 |
| **SymSpell** (symspellpy) | spelling correction | dictionary built from our own product vocabulary | 2 |
| Unicode **NFKC** | text normalization | full-width letters, ligatures, symbols | 2 |
| **GLiNER** `urchade/gliner_medium-v2.1` | zero-shot NER (transformer) | attribute extraction with label names only, no training | 3 |
| **scikit-learn TF-IDF** | lexical representation | word 1–2-grams and character 3–5-grams; string similarity | 4, 5 |
| **gensim FastText** | sub-word word embeddings | trained on 76k product names | 4 |
| **Sentence-BERT** `all-MiniLM-L6-v2` | sentence embeddings | semantic similarity, blocking, search, column-mapping baseline | 1, 4, 5 |
| **BGE** `BAAI/bge-small-en-v1.5` | sentence embeddings (stronger) | comparison in Phase 4, categorization baseline | 4, 6 |
| **Fine-tuned MiniLM** `models/minilm-products` | sentence embeddings, trained by us | MultipleNegativesRankingLoss on 8,471 WDC pairs + 6,000 synthetic pairs; used by the app | 7 |
| **FAISS** `IndexFlatIP` | vector search | exact cosine nearest-neighbour search, blocking | 4, 5 |
| **RapidFuzz** | fuzzy string matching | token-set ratio (baseline configuration) | 8 (code) |
| **HDBSCAN**, **agglomerative clustering**, connected components | clustering | grouping matched products | 5 |
| **UMAP** | dimensionality reduction | 2-D plot of embeddings (`reports/figures/phase5_umap_wdc.png`) | 5 |
| **qwen3:8b** via **Ollama** | local LLM (8 B parameters) | unknown headers, attribute extraction, grey-zone pairs, canonical names, zero-shot categories, Hinglish rewrites | 1, 3, 6 |
| **Pydantic** | schema validation | forces every LLM reply into a typed JSON schema | 1, 3, 6 |
| **Aksharantar** (AI4Bharat) | Hindi ↔ Roman word pairs | transliteration variants (chawal / chaawal) | 7 (code) |
| **LaBSE**, **multilingual-E5-small** | multilingual sentence embeddings | Hinglish robustness comparison | 7 (code) |
| **BERTopic** | topic modelling | category discovery | 9 (code) |
| **Streamlit** | web app | demo | 9 |
| PyTorch 2.6 + CUDA | deep-learning runtime | GPU for SBERT, GLiNER, fine-tuning | all |
| DuckDB | SQL on files | filtering the 1.3 GB Open Food Facts export | 0 |

## 3. Key results (measured on labelled test sets)

| Component | Test set | Result |
|---|---|---|
| Column mapping | 59 hand-labelled headers | accuracy **0.966** (rules 0.898, LLM alone 0.949); LLM called for 13 of 59 |
| Column selection | same 59 headers | F1 **0.963** |
| Unit normalization | 597 WDC-PAVE values | exact match **0.988** (no normalization 0.101) |
| Spelling correction | 487 synthetic typos | 91.2 % restored |
| Attribute extraction | WDC-PAVE, 354 titles | macro F1: rules 0.340, GLiNER 0.505, **LLM 0.592**, distilled spaCy 0.165 |
| Retrieval | Abt-Buy, 1,076 queries | Recall@1: word TF-IDF **0.831**, BGE 0.803, MiniLM 0.746, fastText 0.625 |
| Blocking | Abt-Buy | keeps 99.9 % of true pairs, removes 93.6 % of comparisons |
| Pair matching | Abt-Buy test | F1 0.529 (TF-IDF) → **0.753** (hybrid + rules) → 0.760 (+ LLM) |
| LLM on uncertain pairs | 65 grey-zone pairs | **0.846** correct vs 0.662 for the threshold |
| Clustering | WDC multi-class, 1,000 offers | ARI **0.296** (agglomerative) |
| BigBasket matching | 500 labelled pairs | F1 **0.906** with the MRP rule (labels provisional) |

## 4. Demo questionnaire: inputs that show every feature

Use these during the demo, in the Streamlit app (`streamlit run app/streamlit_app.py`). Each row says what to do, what should happen, and which feature it proves.

### A. Column standardization (Tab 1)

| # | Do this | Expected result | Shows |
|---|---|---|---|
| A1 | Load `bigbasket_demo_renamed_columns.csv` | "Item Name" → product_name, "Brand Name" → brand, "Selling Price" → price, "MRP (Rs)" → mrp, "Stars" → rating, "Details" → description | rule-based header mapping (R1) |
| A2 | Point at the two price columns | the one with higher values becomes **mrp** | price/MRP value rule |
| A3 | Load `unseen_open_food_facts_world.csv` | `code` → id, `brands` → brand, `url`/`created_t` → other | generalization to an unseen file |
| A4 | Tick **Use the LLM** (Ollama running), reload | headers without keywords (e.g. "Aisle", "Kind") show `decided_by = llm` | LLM used only where rules are unsure (R6) |

### B. Column selection (Tab 2)

| # | Look at | Expected | Shows |
|---|---|---|---|
| B1 | "Item Name" row | selected, role = primary (name) | automatic choice of the clustering column (R2) |
| B2 | "Details" row | selected, role = secondary | |
| B3 | "SKU", "Selling Price" rows | not selected, reason "mostly non-letters" | justification for each decision |
| B4 | "Department" row | not selected, reason "low uniqueness" | |

### C. Normalization and attribute extraction (Tab 3)

| # | Find a product named… | Expected | Shows |
|---|---|---|---|
| C1 | "… 1 kg" / "… 1kg" | quantity 1000, unit g | unit normalization (R5) |
| C2 | "… 500 ml" | 500, ml | |
| C3 | "… Pack of 4" / "3 x 100 g" | pack count 4 / 3 | spaCy Matcher |
| C4 | a name with a colour ("… - Red") | colour = red | EntityRuler gazetteer (NER) |
| C5 | any name | product type filled | noun-chunk rule + distilled NER |

### D. Matching and clustering (Tab 4, "Only clusters with more than one listing" ticked)

| # | Look for | Expected | Shows |
|---|---|---|---|
| D1 | a cluster with 2–4 identical listings | grouped together, one canonical name | entity resolution (R3) |
| D2 | same name, different MRP | **not** grouped | MRP/size rule (R4 rules) |
| D3 | the caption under the table | SBERT weight, threshold, MRP rule shown | hybrid score: cosine + string similarity + rules (R4) |
| D4 | untick the box | all products, most in their own cluster | |

### E. Semantic search (Tab 5)

| # | Type | Expected top results | Shows |
|---|---|---|---|
| E1 | `2 litre cola` | soft drinks / cola | semantic search with sentence embeddings |
| E2 | `green tea` | green tea products | |
| E3 | `basmati rice 5 kg` | basmati rice | quantities in queries |
| E4 | `chocolate biscuits` | biscuits/cookies with chocolate | synonyms (biscuit ≈ cookie) |
| E5 | `shampoo for dandruff` | anti-dandruff shampoos | paraphrase, not exact words |
| E6 | `haldi powder` | turmeric (may fail) | **honest limitation:** English-only model, Hinglish weak |

### F. Questions the teacher may ask, and where to show the answer

| Question | Show |
|---|---|
| "Where is the NLP?" | Section 2 of this file; Tab 3 |
| "Why use an LLM at all?" | `reports/phase6_verification.md` §3.1 (0.846 vs 0.662 on uncertain pairs) |
| "How do you know it works?" | `requirements_traceability.md`; Section 3 of this file |
| "What does each requirement map to?" | `requirements_traceability.md` |
| "Show me a failure" | any `reports/phaseN_verification.md` §4 (failure examples are listed) |
| "Can it handle new data?" | A3: the unseen Open Food Facts file |

## 5. What is not finished

- Phase 7 evaluation (fine-tuned vs base model, Hinglish comparison): code written, not run. The fine-tuned model is trained and used by the app.
- Phase 8 ablation table and Phase 9 BERTopic: code written (`src/evaluate.py`, `src/topics.py`), not run.
- The three gold sets in `data/gold/` were labelled by the assistant and must be checked by the student.
