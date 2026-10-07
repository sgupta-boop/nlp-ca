# Requirements traceability

Maps each of the professor's requirements to the modules that implement it and the measured
evidence. Updated at the end of every phase. Status: ☐ not started · ◐ partial · ☑ satisfied with evidence.

| ID | Requirement | Status | Implemented in | Evidence (measured on a labelled set) |
|---|---|---|---|---|
| R1 | Standardize columns from differently structured tables into one schema | ☑ | `src/columns.py`: keyword rules → LLM cascade + price/MRP value rule; `to_canonical()` | Header-mapping accuracy **0.966** on 59 gold headers (rules 0.898, roadmap embeddings 0.644, LLM-only 0.949); 4 files → identical 11 columns (`reports/phase1_verification.md`) |
| R2 | Select columns relevant for clustering automatically, with justification | ☑ | `src/columns.py`: `profile_column()`, `select_columns()` (written reason per column) | Selection P 0.929 / R 1.000 / F1 **0.963** on 59 gold headers; primary name column correct in 7/7 files |
| R3 | Group similar product names (entity resolution / clustering) | ☐ | Phase 5 | Problem measured: 4,043 duplicate names in BigBasket (Phase 0) |
| R4 | Rule-based AND string-similarity AND cosine similarity | ◐ | Phase 1: header keyword rules, price/MRP rule, SBERT cosine | rules 0.898, SBERT cosine 0.797 (Phase 1). String similarity: Phases 4–5 |
| R5 | Substantially NLP: tokenization, normalization, lemmatization, NER, representations | ◐ | Phase 0: spaCy pipeline verified; unit-spelling variety measured | 18 unit spellings in BigBasket, 20 in Flipkart; lowercasing must precede lemmatization (`reports/phase0_check_env.txt`) |
| R6 | LLMs only where they add value | ◐ | `src/llm.py` (single entry point); Phase 1: LLM only for headers no rule recognises | Phase 1: 13/59 headers sent to the LLM, accuracy 0.966 vs 0.949 for LLM on all 59; 0 invalid replies |
| R7 | Modern NLP frameworks | ◐ | spaCy 3.8, sentence-transformers 6.1, GLiNER 0.2, BERTopic 0.17, FAISS 1.15, UMAP 0.5, gensim 4.4, PyTorch 2.6 CUDA, Ollama | all import and run (`reports/phase0_check_env.txt`); used for real from Phase 1 |
| R8 | Every claim backed by a measured number on a labelled test set | ◐ | Benchmarks ready | Abt-Buy, Amazon-Google, WDC Products, WDC-PAVE loaded and size-checked (`tests/test_phase0.py`, 13 passed) |

## Phase log

| Phase | Requirements touched | Key evidence |
|---|---|---|
| 0 | R1 (problem), R3 (problem), R5, R6, R7, R8 (test sets ready) | `reports/phase0_verification.md` |
| 1 | R1 ☑, R2 ☑, R4, R5, R6, R7, R8 | `reports/phase1_verification.md` |
