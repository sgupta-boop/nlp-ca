# Phase 4 — Verification Report (Text representations and cosine similarity)

**Goal (roadmap):** compare lexical, subword and semantic representations under the same cosine-similarity retrieval task.
**Done when (roadmap):** one retrieval table with 5 representations × 3 metrics.
**Status: PASSED.** The table is below (both benchmarks, normalized and raw input). 9/9 Phase 4–5 helper tests pass.

## 1. What was built (`src/embed.py`)

| Representation | Type | Details |
|---|---|---|
| `tfidf_word` | lexical | TF-IDF, word 1–2 grams, sublinear tf, fitted on both tables' names (no labels) |
| `tfidf_char` | lexical, sub-word | TF-IDF, `char_wb` 3–5 grams |
| `fasttext` | sub-word embeddings | gensim FastText (skip-gram, 100 d, char n-grams 3–5, 20 epochs, seed 42, 1 worker for determinism) trained on 76k names: BigBasket, Flipkart, OFF India + the unlabelled benchmark names; a name = mean of its word vectors |
| `minilm` | sentence embedding | `sentence-transformers/all-MiniLM-L6-v2` |
| `bge` | sentence embedding (stronger) | `BAAI/bge-small-en-v1.5` |

All vectors are L2-normalized, so the inner product equals the cosine similarity. Dense vectors are searched with **FAISS `IndexFlatIP`** (exact). TF-IDF vectors are sparse with tens of thousands of dimensions, so their cosine is computed exactly with a sparse matrix product rather than densified into FAISS; the metric is identical.

**Task:** every Abt record with a gold match (1,076 queries) ranks all 1,092 Buy records; every Amazon record with a match (1,113 queries) ranks all 3,226 Google records. Recall@k = gold match in the top k; MRR = mean of 1/rank of the first gold match. The full ranking is used, so MRR is exact.

## 2. Requirements check

| Req | How this phase meets it | Evidence |
|---|---|---|
| R4 Cosine similarity | every representation is compared by cosine similarity | table 3.1 |
| R5 Text representations | lexical (word TF-IDF), sub-word (char TF-IDF, fastText), sentence embeddings (two SBERT models) | table 3.1 |
| R7 Frameworks | scikit-learn, gensim, sentence-transformers, FAISS | `src/embed.py` |
| R8 Labelled test sets | Abt-Buy (1,097 gold matches), Amazon-Google (1,300) | table 3.1 |

## 3. Results

### 3.1 Retrieval with normalized names (Phase 2 `normalize_text`)

| Representation | Abt-Buy R@1 | R@5 | MRR | Amazon-Google R@1 | R@5 | MRR |
|---|---|---|---|---|---|---|
| TF-IDF word 1–2 | **0.831** | **0.966** | **0.891** | **0.752** | **0.954** | **0.844** |
| TF-IDF char 3–5 | 0.786 | 0.943 | 0.856 | 0.699 | 0.948 | 0.810 |
| fastText (mean) | 0.625 | 0.857 | 0.730 | 0.539 | 0.772 | 0.641 |
| SBERT MiniLM-L6 | 0.746 | 0.932 | 0.828 | 0.647 | 0.904 | 0.762 |
| SBERT BGE-small | 0.803 | 0.955 | 0.872 | 0.686 | 0.927 | 0.791 |

### 3.2 Effect of Phase 2 normalization (Recall@1, raw → normalized)

| Representation | Abt-Buy | Amazon-Google |
|---|---|---|
| TF-IDF word | 0.645 → **0.831** | 0.750 → 0.752 |
| TF-IDF char | 0.829 → 0.786 | 0.721 → 0.699 |
| fastText | 0.393 → **0.625** | 0.552 → 0.539 |
| MiniLM | 0.722 → 0.746 | 0.649 → 0.647 |
| BGE-small | 0.763 → 0.803 | 0.694 → 0.686 |
| mean over all 10 | 0.672 → **0.711** | (MRR 0.769 → 0.802) |

Normalization helps most where tokenization matters (word TF-IDF and fastText on Abt-Buy, where "PSLX350H" and "PS-LX350H" become comparable once punctuation is handled). It slightly hurts char TF-IDF, which already captures those overlaps and loses some useful punctuation-based n-grams.

## 4. Examples (Abt-Buy, `notebooks/04_embed.ipynb`)

| # | Query | Word TF-IDF top-1 | MiniLM top-1 | Note |
|---|---|---|---|---|
| 1 | `Panasonic Integrated Telephone System - KXTS108W` | `Panasonic KX-TS108W Corded Phone` ✓ | `Panasonic KX-TS600W Basic Telephone` ✗ | lexical wins: the model code decides, and SBERT does not weight it |
| 2 | `GE GSD4000NWW White Built-In Dishwasher` | `GE GSD4000NWW Dishwasher …` ✓ | `GE GSD2400NWW Dishwasher …` ✗ | **SBERT failure:** near-identical codes look semantically equal |
| 3 | `Whirlpool Duet Sport Front Loading White Washer - WFW8300SWH` | `Whirlpool Duet Sport 27'' Electric Dryer - WED8300SWH` ✗ | `Whirlpool WFW8300SW 27' Front-Load Washer` ✓ | semantic wins: washer ≠ dryer although the codes share characters |
| 4 | `Garmin Deluxe Carrying Case - Black Finish - 0101023101` | `Sony LCS Soft Carrying Case` ✗ | `Garmin Canvas Deluxe Carry Case - 010-10231-01` ✓ | **TF-IDF failure:** "carrying" ≠ "carry" and the code is formatted differently |

## 5. Discussion

Product names are dominated by brand names and model codes, which lexical TF-IDF matches exactly and weights by rarity. Word TF-IDF is therefore the best single representation on both benchmarks, ahead of both sentence encoders. Sentence embeddings are better at semantics (washer vs dryer, carrying vs carry) but blur codes that differ by one character. fastText's averaged word vectors are weakest, because averaging dilutes the rare code tokens. The two families make *different* errors (examples 1–4), which is the motivation for the hybrid lexical + semantic score in Phase 5. MiniLM is kept as the SBERT component in Phases 5–7 so that Phase 7 compares the same model before and after fine-tuning; BGE-small would be a stronger drop-in replacement.

## 6. Limitations

- TF-IDF and fastText are fitted on the benchmark names themselves (unsupervised, no labels): standard transductive practice, stated for transparency.
- Retrieval is one-directional (Abt→Buy, Amazon→Google), as in the roadmap.

## 7. Report section draft — 5.4 Text representations

**Method.** We compare five representations of a (normalized) product name: TF-IDF over word unigrams and bigrams, TF-IDF over character 3–5-grams within word boundaries, the mean of fastText word vectors trained on 76k product names, and two Sentence-BERT encoders (all-MiniLM-L6-v2 and BGE-small-en-v1.5). Vectors are L2-normalized so that inner products are cosine similarities. Dense vectors are indexed with an exact FAISS inner-product index; sparse TF-IDF vectors are compared by sparse matrix multiplication.

**Setup.** For each source record with at least one gold match (1,076 in Abt-Buy, 1,113 in Amazon-Google), all target records are ranked by cosine similarity. We report Recall@1, Recall@5 and mean reciprocal rank.

**Results.** Word-level TF-IDF performs best on both datasets (Recall@1 0.831 and 0.752), followed by character TF-IDF and BGE-small; MiniLM is lower and averaged fastText vectors are lowest (Table Z). Our normalization step raises mean Recall@1 across representations from 0.672 to 0.711, with the largest gains for word TF-IDF (+0.186) and fastText (+0.232) on Abt-Buy.

**Discussion.** Lexical and semantic representations fail on different pairs: TF-IDF misses paraphrases and differently formatted codes, while sentence embeddings confuse products whose model numbers differ by a single character. This complementarity motivates combining them in the hybrid matcher of Phase 5.
