# LLM-Assisted Product Entity Resolution

**Attribute Extraction, Semantic Embeddings and Hybrid Matching for Product Name Standardization**

NLP Lab · Continuous Assessment project

## What the project does

Different shops describe the same product differently: the columns have different names (`product`, `title`, `Item Name`), and the names are written differently (`Tata Salt 1kg` vs `TATA salt 1000 gm`). This project takes any product table and:

1. **Maps its columns** to one standard schema (product name, brand, category, price, MRP, quantity…).
2. **Selects the columns** worth comparing, with a written reason for each.
3. **Cleans each product name** with NLP: case, symbols, spelling, lemmas, and units (`1kg` → `1000 g`).
4. **Extracts attributes** such as brand, product type, colour, quantity and pack count.
5. **Groups listings of the same product** using rules, string similarity and cosine similarity of embeddings.
6. **Uses a local LLM (qwen3:8b)** only where the cheaper methods are unsure.

A Streamlit app shows all of this on any uploaded CSV, plus semantic search.

## Key results

Every number is measured on a labelled test set.

| Component | Test set | Result |
|---|---|---|
| Column mapping | 59 labelled headers | **96.6 %** accuracy (rules alone 89.8 %), LLM used for 13 of 59 |
| Column selection | same 59 headers | F1 **0.963** |
| Unit normalization | 597 WDC-PAVE values | **98.8 %** exact match (baseline 10.1 %) |
| Attribute extraction | 354 WDC-PAVE titles | macro F1: rules 0.340, GLiNER 0.505, **LLM 0.592**, distilled spaCy 0.165 |
| Retrieval (Recall@1) | Abt-Buy | word TF-IDF **0.831**, BGE 0.803, MiniLM 0.746, fastText 0.625 |
| Pair matching (F1) | Abt-Buy test | 0.529 (TF-IDF) → **0.753** (hybrid + rules) → 0.760 (+ LLM) |
| LLM on uncertain pairs | 65 grey-zone pairs | **84.6 %** correct vs 66.2 % for the threshold |
| Clustering (ARI) | WDC Products, 1,000 offers | **0.296** (agglomerative) |
| BigBasket duplicates | 27,555 products | 996 groups of the same product |

The full report with every table is in [reports/project_report.html](reports/project_report.html).

## How it works

| Stage | Main technique | File |
|---|---|---|
| Column mapping | keyword rules; LLM only when no rule fires; MRP = the higher of two price columns | `src/columns.py` |
| Column selection | profile each column (fill rate, words, uniqueness, letters, URLs) and keep free text | `src/columns.py` |
| Normalization | Unicode NFKC, lowercase, quantity regex, spaCy `Matcher`, SymSpell, spaCy lemmas | `src/preprocess.py` |
| Attribute extraction | spaCy `EntityRuler` + brand list, GLiNER, LLM (Pydantic JSON), distilled spaCy NER | `src/extract.py` |
| Representations | TF-IDF (words, characters), fastText, Sentence-BERT; FAISS cosine search | `src/embed.py` |
| Matching | blocking → hybrid score (SBERT cosine + character TF-IDF cosine) → rules (size, pack, brand, model number) → threshold | `src/match.py` |
| Clustering | connected components, agglomerative, HDBSCAN | `src/match.py` |
| LLM layer | grey-zone pairs, canonical names, categories, test data | `src/llm_layer.py` |
| LLM gateway | one function: JSON schema, validation, retries, cache, call counter | `src/llm.py` |
| Fine-tuning | MiniLM + MultipleNegativesRankingLoss on product pairs | `src/finetune.py` |
| Whole system | used by the app | `src/pipeline.py`, `app/streamlit_app.py` |

## Tools and models

spaCy · GLiNER · Sentence-BERT (all-MiniLM-L6-v2, BGE-small, our fine-tuned MiniLM) · scikit-learn TF-IDF · gensim fastText · FAISS · RapidFuzz · SymSpell · UMAP · HDBSCAN · qwen3:8b via Ollama · Pydantic · Streamlit · PyTorch (CUDA) · DuckDB

## Datasets

| Dataset | Use |
|---|---|
| BigBasket (27,555) and Flipkart (20,000) | working data (Indian retail) |
| Open Food Facts, India (21,189) | messy third source |
| Abt-Buy, Amazon-Google | matching benchmarks with gold pairs |
| WDC Products | hard matching and clustering benchmark |
| WDC-PAVE | attribute extraction and unit normalization gold |
| Aksharantar (Hindi), Google Product Taxonomy | Hinglish experiments, category labels |

`python scripts/download_data.py` downloads all of them into `data/raw/`. BigBasket and Flipkart need a Kaggle token saved in `C:\Users\<you>\.kaggle\access_token`.

Our own small labelled sets are in `data/gold/`: 59 column headers, 500 BigBasket product pairs and 15 Hinglish names.

## Setup

Built on Windows 11, Python 3.11, 7.4 GB RAM, NVIDIA GTX 1650 Ti (4 GB).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# local LLM (only needed for the LLM features)
winget install Ollama.Ollama
ollama pull qwen3:8b

python scripts/download_data.py
python scripts/check_env.py          # checks libraries, GPU and LLM speed
```

## Run the demo

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

The app opens at http://localhost:8501. Choose a file under **"…or use a demo file"**:
- `bigbasket_demo_renamed_columns.csv`: 800 rows with renamed headers and many duplicates
- `unseen_open_food_facts_world.csv`: 500 rows with 25 messy columns, never used during development

Then go through the tabs: **Column mapping → Column selection → Attributes → Clusters → Search**. Leave "Use the LLM" unticked unless Ollama is running.

## Run each phase

| Phase | Command |
|---|---|
| 1 Column mapping and selection | `python -m src.columns` |
| 2 Preprocessing | `python -m src.preprocess` |
| 3 Attribute extraction | `python -m src.extract rules_gliner`, `llm_pave`, `gliner_label 2000`, `distill` |
| 4 Representations | `python -m src.embed` |
| 5 Matching and clustering | `python -m src.match`, then `bb_sample` and `bb_eval` |
| 6 LLM layer | `python -m src.llm_layer prepare`, then `adjudicate`, `names`, `categories`, `synthetic_rules`, `hinglish` |
| 7 Fine-tuning | `python -m src.finetune train` |
| Tests | `python -m pytest` |

All LLM answers are cached in `cache/llm_responses.json`, so re-running a phase repeats no LLM calls.

## Folder structure

```
app/              Streamlit app
src/              one module per stage (see "How it works")
scripts/          data download, environment check, background job runners
notebooks/        one notebook per phase (tables and plots)
tests/            pytest tests for every phase
data/raw/         downloaded datasets (not in git)
data/processed/   cleaned tables, clusters, embeddings (not in git)
data/gold/        our labelled test sets
data/demo/        demo files for the app
models/           fine-tuned MiniLM, distilled NER, fastText (not in git)
cache/            every LLM reply
reports/          project report, per-phase verification reports, figures,
                  presentation script, demo script, viva questions
```

## Project status

| Part | Status |
|---|---|
| Phases 0–5 | complete, each with code, tests, notebook and verification report |
| Phase 6 LLM layer | complete, with a reduced LLM budget (about 15 s per call on this laptop) |
| Phase 7 Fine-tuning | model trained and used by the app; comparison with base and multilingual models not run |
| Phase 8 Ablation, Phase 9 BERTopic | code written (`src/evaluate.py`, `src/topics.py`), not run |
| Streamlit app | complete, tested end to end on both demo files |

## Limitations

- The LLM is slow on a laptop, so it is used on few items (65 grey-zone pairs, 55 categorizations).
- The three sets in `data/gold/` were labelled during the project and still need a manual check.
- Hinglish names (`haldi`, `chawal`) are handled poorly by the English model.
- Colour or shade variants with identical names cannot be told apart from the text.

## Further reading

| File | Contents |
|---|---|
| `reports/project_report.html` | full report with all metrics |
| `reports/phase0–6_verification.md` | per-phase method, results, examples and failures |
| `reports/presentation_script.md` | spoken explanation of the project |
| `reports/demo_script.md`, `reports/technical_overview.md` | demo walkthrough and test inputs |
| `reports/viva_questions.md` | likely questions with answers |
| `requirements_traceability.md` | each requirement (R1–R8) mapped to code and evidence |
