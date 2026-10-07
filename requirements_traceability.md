# Requirements traceability

Maps each of the professor's requirements to the modules that implement it and the measured
evidence. Updated at the end of every phase. Status: ☐ not started · ◐ partial · ☑ satisfied with evidence.

| ID | Requirement | Status | Implemented in | Evidence (measured on a labelled set) |
|---|---|---|---|---|
| R1 | Standardize columns from differently structured tables into one schema | ◐ | Phase 0: `src/data_io.py` loads 6 differently shaped tables | Problem measured: the name field has 4 different headers (`product`, `product_name`, `name`, `title`) across 6 sources (`reports/phase0_profile.csv`). Mapping + accuracy: Phase 1 |
| R2 | Select columns relevant for clustering automatically, with justification | ☐ | Phase 1 | – |
| R3 | Group similar product names (entity resolution / clustering) | ☐ | Phase 5 | Problem measured: 4,043 duplicate names in BigBasket (Phase 0) |
| R4 | Rule-based AND string-similarity AND cosine similarity | ☐ | Phases 2, 3, 4, 5 | – |
| R5 | Substantially NLP: tokenization, normalization, lemmatization, NER, representations | ◐ | Phase 0: spaCy pipeline verified; unit-spelling variety measured | 18 unit spellings in BigBasket, 20 in Flipkart; lowercasing must precede lemmatization (`reports/phase0_check_env.txt`) |
| R6 | LLMs only where they add value | ◐ | `src/llm.py` (single entry point: JSON schema, Pydantic validation, cache, call counter, `unload()`) | qwen3:8b: 9.33 s/call, 5/5 valid JSON (Phase 0) — cost justifies grey-zone-only use |
| R7 | Modern NLP frameworks | ◐ | spaCy 3.8, sentence-transformers 6.1, GLiNER 0.2, BERTopic 0.17, FAISS 1.15, UMAP 0.5, gensim 4.4, PyTorch 2.6 CUDA, Ollama | all import and run (`reports/phase0_check_env.txt`); used for real from Phase 1 |
| R8 | Every claim backed by a measured number on a labelled test set | ◐ | Benchmarks ready | Abt-Buy, Amazon-Google, WDC Products, WDC-PAVE loaded and size-checked (`tests/test_phase0.py`, 13 passed) |

## Phase log

| Phase | Requirements touched | Key evidence |
|---|---|---|
| 0 | R1 (problem), R3 (problem), R5, R6, R7, R8 (test sets ready) | `reports/phase0_verification.md` |
