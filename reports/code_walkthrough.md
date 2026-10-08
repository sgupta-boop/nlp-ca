# Code walkthrough: which file to open and what to say

Follow the data from start to finish. About 10 minutes.

## 0. Start with the big picture: `src/pipeline.py`, `run()` (line 71)

**Point at:** the lines that call each step in order.

**Say:** "This one function is the whole system. It maps the columns, selects the text column, cleans the names, extracts attributes, builds embeddings, matches pairs and clusters them. The app just calls this. I'll now open each step."

## 1. Column mapping: `src/columns.py`

| Line | Point at | Say |
|---|---|---|
| 49 `KEYWORD_RULES` | the list of keywords per field | "Our rules: if a header contains 'price' it's price, 'brand' or 'manufacturer' is brand, and so on. Specific rules are checked first, so 'sub_category' isn't caught by 'category'." |
| 70 `map_by_rules()` | the loop | "We split the header into words and return the first field whose keyword matches." |
| 139 `map_columns()` | the `if rule_hit … else map_by_llm` branch | "If a rule fires we trust it; only if no rule fires do we ask the LLM. That's how the LLM is used on 13 of 59 headers, not all of them. 96.6 % accuracy." |
| 222 `resolve_price_conflict()` | the median comparison | "If two columns both look like price, the one with the higher median is the MRP, because MRP is never below the selling price." |

## 2. Column selection: `src/columns.py`

| Line | Point at | Say |
|---|---|---|
| 169 `profile_column()` | the returned dictionary | "For each column we measure the fill rate, average words, share of unique values, share of letters and URLs." |
| 187 `SELECT_RULES` | the thresholds | "A column is kept only if it passes all five checks; otherwise we write the reason. F1 0.963." |

## 3. NLP cleaning: `src/preprocess.py`

| Line | Point at | Say |
|---|---|---|
| 31 `UNITS` | the dictionary | "Every unit spelling maps to a standard unit and a conversion factor: kg → g × 1000, ltr → ml × 1000." |
| 59 `QTY_RE` | the regular expression | "This regex finds a number followed by a unit, including fractions like 3-1/2." |
| 152 `normalize_text()` | each line in order | "Unicode normalization, lowercase, split '200gm' into '200 gm', and rewrite every quantity in one form, so '1kg' and '1000 gm' become the same text." |
| 117 `find_pack_count()` | the `Matcher` patterns | "spaCy's rule-based Matcher finds 'pack of 6', '3 x 100 g', '12 pcs'." |
| 206 `SpellCorrector` | the condition `seen > 0` | "SymSpell corrects spelling using our own vocabulary, but only unseen words. Correcting rare real words damaged them, so we measured it and turned it off." |
| 286 `lemmatize_many()` | `nlp.pipe` | "spaCy lemmatization: biscuits → biscuit. We lowercase first, because spaCy doesn't lemmatize capitalized words." |

## 4. Attribute extraction (NER): `src/extract.py`

| Line | Point at | Say |
|---|---|---|
| 90 `extract_rules()` | `doc.ents` | "spaCy EntityRuler with a list of 8,000 brands and colours: rule-based NER." |
| 135 `extract_gliner()` | `predict_entities` | "GLiNER, a zero-shot NER model: we only give it the label names." |
| 151 `ProductAttributes` | the Pydantic class | "For the LLM we define the exact output shape; the answer must fit this or it's rejected." |
| 183 `extract_llm()` | the `ask_json` call | "The LLM version. It's the most accurate (0.59 F1), but 13,500 times slower than spaCy." |

## 5. The LLM gateway: `src/llm.py`

| Line | Point at | Say |
|---|---|---|
| 73 `ask_json()` | the cache check, then the validation | "Every LLM call in the project goes through here. First we check the cache; then we call the model with a JSON schema; then Pydantic validates the reply; invalid replies are retried. Every answer is saved, so results are reproducible." |
| 49 `_chat()` | the `options` line | "Temperature 0 and a fixed seed for repeatable answers, a 300-token cap so the model can't loop, and thinking mode off." |

## 6. Text representations: `src/embed.py`

| Line | Point at | Say |
|---|---|---|
| 59 `Encoder` | the `if self.kind == …` branches | "Five ways to turn text into vectors: word TF-IDF, character TF-IDF, fastText, and two Sentence-BERT models." |
| 135 `cosine_topk()` | `faiss.IndexFlatIP` | "Vectors are normalized, so the inner product equals cosine similarity; FAISS finds the nearest ones." |
| 149 `retrieval_metrics()` | the rank loop | "Recall@1, Recall@5 and MRR: is the correct product ranked first, in the top five, and at what rank?" |

## 7. Matching and clustering: `src/match.py`

| Line | Point at | Say |
|---|---|---|
| 110 `block()` | top-k + same brand | "We only compare likely pairs: nearest neighbours or same brand. This removes 94 % of comparisons but keeps 99.9 % of true pairs." |
| 71 `PairScorer.features()` | `sbert`, `char`, the `*_conflict` columns | "For each pair: embedding cosine, character similarity, and whether quantity, pack, brand or model number conflict." |
| 100 `hybrid_score()` | the formula and `np.where(... 0.0 ...)` | "The hybrid score: weighted cosine plus string similarity, minus penalties, and set to zero if the size differs. Rules took F1 from 0.53 to 0.75." |
| 40 `model_codes()` | the regex | "This finds model numbers like PS-LX350H; different codes mean different products." |
| 132 `tune()` | the grid loops | "The weights and threshold are chosen on validation data only; we report on the test data." |
| 289 `cluster_agglomerative()` | `linkage="average"` | "Matches are grouped into clusters; average linkage avoids chaining." |

## 8. The LLM for uncertain pairs: `src/llm_layer.py`

| Line | Point at | Say |
|---|---|---|
| 42 `grey_zone_budget()` | `argsort(abs(scores - threshold))` | "We send only the pairs closest to the threshold, where the system is unsure." |
| 22 `SameProduct`, 32 `adjudicate()` | the prompt and schema | "The LLM answers 'same: true/false' with a reason, as JSON. On those hard pairs it was right 85 % of the time against 66 % for the threshold." |

## 9. The app: `app/streamlit_app.py`

| Line | Point at | Say |
|---|---|---|
| 41 `run_cached()` | `pipeline.run(...)` | "The app reads the CSV and calls the same pipeline; the result is cached so the tabs are instant." |
| 54 `st.tabs(...)` | the five tab names | "Each tab shows one stage: mapping, selection, attributes, clusters, search." |

Then switch to the browser and run the demo.

## 10. Proof it works: `tests/` and `reports/`

**Open:** `tests/test_phase2.py` and point at `test_quantity_value_and_unit`.

**Say:** "Every core function has tests. For example, '1kg', '1000 gm' and '1 ltr' must give the right value and unit. All tests pass. Every number we quote is in `reports/` and mapped to the professor's requirements in `requirements_traceability.md`."

## If you are short on time (3 minutes)

1. `src/pipeline.py` `run()`: the whole system in order.
2. `src/preprocess.py` `normalize_text()`: the NLP cleaning.
3. `src/match.py` `hybrid_score()`: cosine + string similarity + rules.
4. `src/llm.py` `ask_json()`: how the LLM is controlled.
