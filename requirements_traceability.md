# Requirements traceability

Maps each of the professor's requirements to the modules that implement it and the measured
evidence. Updated at the end of every phase. Status: ☐ not started · ◐ partial · ☑ satisfied with evidence.

| ID | Requirement | Status | Implemented in | Evidence (measured on a labelled set) |
|---|---|---|---|---|
| R1 | Standardize columns from differently structured tables into one schema | ☑ | `src/columns.py`: keyword rules → LLM cascade + price/MRP value rule; `to_canonical()` | Header-mapping accuracy **0.966** on 59 gold headers (rules 0.898, roadmap embeddings 0.644, LLM-only 0.949); 4 files → identical 11 columns (`reports/phase1_verification.md`) |
| R2 | Select columns relevant for clustering automatically, with justification | ☑ | `src/columns.py`: `profile_column()`, `select_columns()` (written reason per column) | Selection P 0.929 / R 1.000 / F1 **0.963** on 59 gold headers; primary name column correct in 7/7 files |
| R3 | Group similar product names (entity resolution / clustering) | ☑ | `src/match.py`: blocking, hybrid score, rules, connected components / agglomerative / HDBSCAN | Pair F1 Abt-Buy **0.753**, Amazon-Google 0.510, WDC 0.38–0.44, BigBasket 0.906 (provisional labels); WDC clustering ARI **0.296**; BigBasket `cluster_id`: 996 multi-listing clusters |
| R4 | Rule-based AND string-similarity AND cosine similarity | ☑ | rules: Phases 1–3, 5 (vetoes/penalties); string similarity: char n-gram TF-IDF (Phases 4–5), rapidfuzz (Phase 8); cosine: TF-IDF/fastText/SBERT (Phases 4–5) | Abt-Buy F1: char TF-IDF 0.529, SBERT cosine 0.455, + rules **0.753** (Phase 5) |
| R5 | Substantially NLP: tokenization, normalization, lemmatization, NER, representations | ◐ | Phase 2 `src/preprocess.py`: NFKC/case/punctuation normalization, spaCy tokenizer + Matcher + lemmatizer, SymSpell, abbreviations | Unit normalization **0.988** exact match on 597 WDC-PAVE values (baseline 0.101); typo restoration 0.912. Phase 3 NER on WDC-PAVE: macro F1 rules 0.340, GLiNER 0.505, LLM **0.592**, distilled 0.165. Phase 4 representations: Abt-Buy R@1 word TF-IDF 0.831, char TF-IDF 0.786, fastText 0.625, MiniLM 0.746, BGE 0.803 |
| R6 | LLMs only where they add value | ◐ | `src/llm.py` (single entry point); Phase 1: LLM only for headers no rule recognises | Phase 1: 13/59 headers sent to the LLM, accuracy 0.966 vs 0.949 for LLM on all 59; 0 invalid replies |
| R7 | Modern NLP frameworks | ◐ | spaCy 3.8, sentence-transformers 6.1, GLiNER 0.2, BERTopic 0.17, FAISS 1.15, UMAP 0.5, gensim 4.4, PyTorch 2.6 CUDA, Ollama | all import and run (`reports/phase0_check_env.txt`); used for real from Phase 1 |
| R8 | Every claim backed by a measured number on a labelled test set | ◐ | Benchmarks ready | Abt-Buy, Amazon-Google, WDC Products, WDC-PAVE loaded and size-checked (`tests/test_phase0.py`, 13 passed) |

## Phase log

| Phase | Requirements touched | Key evidence |
|---|---|---|
| 0 | R1 (problem), R3 (problem), R5, R6, R7, R8 (test sets ready) | `reports/phase0_verification.md` |
| 1 | R1 ☑, R2 ☑, R4, R5, R6, R7, R8 | `reports/phase1_verification.md` |
| 2 | R4, R5, R7, R8 | `reports/phase2_verification.md` |
| 3 | R4, R5, R6, R7, R8 | `reports/phase3_verification.md` |
| 4 | R4 (cosine), R5, R7, R8 | `reports/phase4_verification.md` |
| 5 | R3 ☑, R4 ☑, R5, R7, R8 | `reports/phase5_verification.md` |
