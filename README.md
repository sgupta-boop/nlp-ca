# LLM-Assisted Product Entity Resolution

**Attribute Extraction, Semantic Embeddings and Hybrid Matching for Product Name Standardization**

Given messy product tables from different sources, the system maps their columns to one schema,
understands each product name (attributes such as brand, quantity, unit), and groups names that
refer to the same product. Rules, string similarity and embeddings do most of the work; a local
LLM (qwen3:8b via Ollama) is used only where they are unsure.

## Machine this was built on

Windows 11 · Python 3.11.9 · 7.4 GB RAM · NVIDIA GTX 1650 Ti (4 GB VRAM) · Ryzen 5 4600H

## Setup

```powershell
# 1. Python environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124   # GPU build of PyTorch
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Local LLM
winget install Ollama.Ollama
ollama pull qwen3:8b

# 3. Kaggle credentials (for BigBasket and Flipkart)
#    Kaggle -> Settings -> API -> "Generate New Token", then save the token text to
#    C:\Users\<you>\.kaggle\access_token   (older tokens: kaggle.json in the same folder)

# 4. Data
python scripts/download_data.py

# 5. Check everything works (versions, GPU, LLM speed)
python scripts/check_env.py
```

## Datasets

`scripts/download_data.py` downloads all of them into `data/raw/` (never edited by hand).
To fetch a single one: `python scripts/download_data.py <name>`.

| Name in script | Dataset | Source | Notes |
|---|---|---|---|
| `bigbasket` | BigBasket Entire Product List | Kaggle `surajjha101/bigbasket-entire-product-list-28k-datapoints` | needs Kaggle token |
| `flipkart` | Flipkart Products | Kaggle `PromptCloudHQ/flipkart-products` | needs Kaggle token |
| `abt_buy` | Abt-Buy | dbs.uni-leipzig.de/files/datasets/Abt-Buy.zip | CSVs are Latin-1 encoded |
| `amazon_google` | Amazon-GoogleProducts | dbs.uni-leipzig.de/files/datasets/Amazon-GoogleProducts.zip | Latin-1 |
| `wdc_products` | WDC Products (80/50/20 % corner cases, pair + multi-class, validation sets) | data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/wdc-products/ | gzip JSON lines |
| `wdc_pave` | WDC-PAVE | git clone github.com/wbsg-uni-mannheim/wdc-pave | use `data/processed_datasets/` |
| `aksharantar_hindi` | Aksharantar Hindi | huggingface.co/datasets/ai4bharat/Aksharantar `hin.zip` | `load_dataset()` is broken on the Hub, so the zip is downloaded directly |
| `google_taxonomy` | Google Product Taxonomy 2021-09-21 | google.com/basepages/producttype/taxonomy.en-US.txt | |
| `open_food_facts_india` | Open Food Facts, India rows only (21,189) | static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz (1.3 GB) | filtered to `en:india` with DuckDB, then the big file is deleted. (The Hugging Face Parquet copy returned HTTP 429 when queried remotely.) |

Loaders for all of them are in `src/data_io.py`.

## Running each phase

| Phase | Command | Output |
|---|---|---|
| 0 Setup and EDA | `python scripts/check_env.py`, then `jupyter nbconvert --to notebook --execute --inplace notebooks/00_eda.ipynb` | `reports/phase0_profile.csv`, `reports/figures/`, `reports/phase0_verification.md` |

| 1 Column mapping + selection | `python -m src.columns` | `data/processed/*_canonical.parquet`, `reports/phase1_*.csv`, `notebooks/01_columns.ipynb` |
| 2 Preprocessing | `python -m src.preprocess` | `data/processed/*_pre.parquet`, `reports/phase2_*`, `notebooks/02_preprocess.ipynb` |
| 3 Attribute extraction | `python -m src.extract rules_gliner`, `... llm_pave` (LLM, ~80 min), `... gliner_label 2000`, `... distill` | `reports/phase3_*`, `models/ner_distilled`, `notebooks/03_extract.ipynb` |
| 4 Representations | `python -m src.embed` | `reports/phase4_retrieval.csv`, `models/fasttext_products.model`, `notebooks/04_embed.ipynb` |

Tests: `python -m pytest -v`

## Folder structure

```
data/raw/         downloaded datasets, never edited
data/processed/   canonical tables, embeddings
data/gold/        the 3 hand-labelled sets
src/              one module per phase (+ data_io.py, llm.py shared)
scripts/          download + environment check
notebooks/        one per phase, for plots and tables
tests/            pytest tests
cache/            llm_responses.json (every LLM reply, for reproducibility)
app/              Streamlit app (Phase 9)
reports/          verification reports, figures, report draft
```

## Implementation notes (Windows)

- Clustering uses `sklearn.cluster.HDBSCAN` (scikit-learn ≥ 1.3) instead of the standalone
  `hdbscan` package, which needs a C compiler on Windows.
- fastText vectors are trained with `gensim.models.FastText`; the `fasttext` pip package does
  not build on Windows.
- RAM is the tightest resource: a loaded qwen3:8b leaves ~0.2 GB free. LLM stages therefore
  run separately from embedding stages and end with `llm.unload()`; `llm.py` retries with fewer
  GPU layers if Ollama reports CUDA out-of-memory.
- Downloads resume after dropped connections (`.part` files) and use a named User-Agent
  (Open Food Facts rejects the default one).
- qwen3's "thinking" mode is switched off (`think=False`) for every call: the tasks are
  short structured extractions, and thinking multiplies the generated tokens.
