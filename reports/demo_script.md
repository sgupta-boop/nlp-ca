# 5-minute demo script

**Setup before the demo:** `streamlit run app/streamlit_app.py`, Ollama running only if you will tick "Use the LLM".

| Time | Show | Say |
|---|---|---|
| 0:00–0:30 | The app's title page | "Shops describe the same product differently. Our system takes any product CSV, maps its columns to one schema, understands each product name, and groups listings of the same product. Rules and embeddings do most of the work; a local LLM is used only where they are unsure." |
| 0:30–1:15 | Choose the demo file `unseen_open_food_facts_world.csv` (500 rows, 25 messy columns, never used in development). Tab **1. Column mapping** | "The headers are messy: `code`, `brands`, `product_name`, `energy-kcal_100g`. Keyword rules map most of them; when no rule fires, the LLM decides. On 59 hand-labelled headers this cascade is 96.6 % accurate while calling the LLM for only 13 of them." |
| 1:15–1:45 | Tab **2. Column selection** | "Each column is profiled: tokens, uniqueness, share of letters, entropy. Free-text, unique columns are selected for clustering, and every decision has a written reason. F1 0.963 on our gold set." |
| 1:45–2:30 | Tab **3. Attributes** | "Names are normalized with spaCy (Unicode, case, lemmas), quantities are parsed and converted ('1 kg' and '1000 gm' become '1000 g'; 98.8 % exact match against WDC-PAVE), and brand, colour and product type are extracted. We compared four extractors on WDC-PAVE: the LLM is the most accurate (F1 0.59) but 13,500 times slower than a small spaCy model, so it is not used for every name." |
| 2:30–3:30 | Tab **4. Clusters** (only multi-listing clusters) | "Matching: FAISS finds candidates, then a hybrid score of sentence-embedding cosine and character n-gram TF-IDF cosine, then rules: different quantity, pack count or model number means a different product. This took Abt-Buy F1 from 0.53 to 0.75. Each cluster gets a canonical name." |
| 3:30–4:15 | Tab **5. Search**, type `2 litre cola`, then `olive oil 1 l` | "Semantic search with our fine-tuned embedding model: the query does not need the exact words." |
| 4:15–5:00 | Back to the slides / `reports/` | "Every claim has a number on a labelled test set: header mapping, unit normalization, NER F1, retrieval Recall@1, pair F1, clustering ARI, and the ablation table, where each component is switched on in turn. Limitations: LLM cost on a laptop (about 15 s per call), weak Hinglish generation, and three gold sets that must be checked by hand." |

**If something fails live:** run without the LLM box ticked; the pipeline then uses rules only and still finishes in seconds.
