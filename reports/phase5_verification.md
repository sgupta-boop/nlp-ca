# Phase 5 — Verification Report (Blocking, hybrid matching and clustering)

**Goal (roadmap):** turn similarity into final groups of same-product records.
**Done when (roadmap):** BigBasket comes out with a `cluster_id` column and the metrics table is filled.
**Status: PASSED.** `data/processed/bigbasket_clusters.parquet` has `cluster_id` for all 27,555 products. All metrics are below. 9/9 Phase 4–5 tests pass.

## 1. What was built (`src/match.py`)

| Step | How |
|---|---|
| Blocking | candidate pairs = SBERT (MiniLM) FAISS top-20 in either direction ∪ same brand (brand from the Phase 3 rule extractor, or the brand column for BigBasket) |
| Hybrid score | s = w · cos_SBERT + (1 − w) · cos_charTF-IDF(3–5) |
| Attribute rules (Phase 2/3) | quantity differs (same unit, > 2 %) → **veto**; pack count differs → **veto**; brand differs → **penalty** (veto on BigBasket, where the brand column is reliable); **model numbers differ → penalty** (added for electronics: alphanumeric codes, hyphens removed, prefix counts as agreement); **MRP differs > 5 % → veto** (BigBasket only, from the Phase 0 finding) |
| Tuning | w ∈ {0, .25, .5, .75, 1}, penalties ∈ {0, .1, .2} and the threshold chosen by F1 on a **validation split**; results on the **test split** only. Abt-Buy / Amazon-Google: seeded 50/50 split of the left records; WDC: official validation / test files; BigBasket: seeded 50/50 split of the labelled pairs |
| Clustering | graph connected components; agglomerative (average linkage on 1 − s); HDBSCAN on SBERT vectors (noise = singletons). Thresholds tuned by ARI on the WDC multi-class *validation* set. On BigBasket: connected components, with components larger than 30 re-split by agglomerative clustering |
| UMAP | 2-D UMAP of WDC test embeddings, 15 largest products coloured (`reports/figures/phase5_umap_wdc.png`) |

**New gold set:** `data/gold/bigbasket_pairs.csv`, 500 same-brand candidate pairs (300 stratified over five score bands + 200 from score ≥ 0.9), **59 same-product pairs (11.8 %)**. **Labelled by Claude** from name, brand, MRP and description; **the student must check them.** The roadmap asked for half positives, but same-brand BigBasket pairs are mostly different variants or sizes, so the extra 200 were drawn from the high-similarity region to get enough positives.

## 2. Requirements check

| Req | How this phase meets it | Evidence |
|---|---|---|
| R3 Group similar product names | blocking → hybrid pair scoring → clustering; BigBasket `cluster_id` | WDC clustering ARI 0.296; BigBasket 996 multi-listing clusters |
| R4 Rules + string similarity + cosine | char n-gram TF-IDF (string similarity), SBERT cosine, attribute rules | ablation rows in 3.2–3.4 |
| R5 NLP | uses Phase 2 normalization and Phase 3 attributes | rule rows |
| R7 Frameworks | FAISS, scikit-learn (agglomerative, HDBSCAN), UMAP, sentence-transformers | `src/match.py` |
| R8 Labelled test sets | Abt-Buy, Amazon-Google, WDC pairs, WDC multi-class, BigBasket pairs | tables below |

## 3. Results

### 3.1 Blocking

| Dataset | All pairs | Candidates | Reduction ratio | Pair completeness (gold kept) |
|---|---|---|---|---|
| Abt-Buy | 1,180,452 | 76,075 | 0.936 | **0.999** |
| Amazon-Google | 4,397,038 | 102,932 | 0.977 | **0.991** |
| BigBasket | 379.6 M | 384,255 | 0.999 | – |

### 3.2 Pair matching, test split (F1)

| Configuration | Abt-Buy | Amazon-Google | WDC 0 % unseen | WDC 50 % | WDC 100 % |
|---|---|---|---|---|---|
| char TF-IDF only | 0.529 | 0.511 | 0.313 | 0.376 | 0.374 |
| SBERT only | 0.455 | 0.415 | 0.291 | 0.341 | 0.343 |
| hybrid, no rules (tuned w) | 0.529 | 0.511 | 0.307 | 0.361 | 0.370 |
| hybrid + rules, no model-number rule | 0.530 | 0.510 | – | – | – |
| **hybrid + all rules** | **0.753** | 0.510 | **0.379** | **0.441** | **0.432** |

Tuned values for "hybrid + all rules": Abt-Buy w = 0.75, model-number penalty 0.2, threshold 0.685; WDC w = 0.5, model-number penalty 0.2, threshold 0.513.

### 3.3 Clustering, WDC Products 80 % corner cases, multi-class test (1,000 offers, 500 products)

| Method | Threshold (tuned on validation) | **ARI** | NMI | V-measure | Clusters |
|---|---|---|---|---|---|
| connected components | 0.700 | 0.192 | 0.932 | 0.932 | 749 |
| **agglomerative (average)** | 0.525 | **0.296** | 0.924 | 0.924 | 530 |
| HDBSCAN (SBERT vectors) | – | 0.174 | 0.887 | 0.887 | 374 |
| baseline: every offer alone | – | 0.000 | **0.947** | 0.947 | 1,000 |

**NMI and V-measure are misleading here:** with 2 offers per product, putting every offer in its own cluster already scores 0.947. ARI corrects for chance and is the metric to read.

### 3.4 BigBasket, 500 labelled pairs (test half: 251 pairs, 33 positives; provisional labels)

| Configuration | Precision | Recall | F1 |
|---|---|---|---|
| char TF-IDF only | 0.408 | 0.879 | 0.558 |
| SBERT only | 0.450 | 0.818 | 0.581 |
| hybrid (no rules) | 0.450 | 0.818 | 0.581 |
| hybrid + qty/pack/brand rules | 0.450 | 0.818 | 0.581 |
| **hybrid + rules + MRP rule** | **0.935** | **0.879** | **0.906** |

**BigBasket clusters** (configuration chosen on validation: hybrid + rules + MRP rule): 27,555 products → 26,487 clusters; **996 clusters with more than one listing, containing 2,064 products**; largest cluster 12.

## 4. Examples (`notebooks/05_match.ipynb` §4–5)

| # | Pair / cluster | Outcome | Why |
|---|---|---|---|
| 1 | All Abt-Buy test candidates | F1 0.530 → **0.753** when the model-number rule is switched on (table 3.2) | pairs with high cosine but different model codes are penalised below the threshold |
| 2 | Abt `Sony DVP-FX820 Black 8' Portable DVD Player` vs Buy `Sony DVP-FX820/P Portable DVD Player` | **false positive** (s = 0.953) | colour variants differ only by a suffix (/P, /L, /R); "dvpfx820" is a prefix of "dvpfx820p", so the rule treats them as agreeing |
| 3 | Abt `Canon Color Ink Tank - CL41CL` vs Buy `Canon Ink Cartridge … - 0617B002` | **false negative** (s = 0.22) | the same cartridge under two numbering schemes (retail code vs part number); the code rule penalizes it and char overlap is low |
| 4 | BigBasket cluster 2440: 12 × `Revlon Colorsilk Hair Colour With Keratin` (MRP 435) | **over-merged** | different shades share one name and MRP; no text signal separates them |
| 5 | BigBasket cluster 37: 4 × `Engage Bodylicious Deodorant Spray - Mate (For Men)` (MRP 195) | correct | true relistings of one product |

## 5. Discussion

- **Lexical beats semantic for matching decisions**, as in Phase 4 (char TF-IDF 0.529 vs SBERT 0.455 on Abt-Buy). Combining them without rules adds nothing: the tuned weight goes to 0.
- **Attribute rules are what make the hybrid work** on electronics. The model-number rule raises Abt-Buy F1 by 0.22 and makes SBERT useful again (tuned weight 0.75). On Amazon-Google (software) the rules do nothing, because titles differ by version and edition rather than codes.
- **WDC Products (80 % corner cases) is hard by design:** our best configuration reaches 0.38–0.44 F1 without any training. Phase 7 fine-tuning targets it.
- **On BigBasket**, real duplicates are relistings with identical normalized names, so the tuned threshold is 1.0, and the main difficulty is identical names with different pack sizes. The MRP rule solves most of these (F1 0.58 → 0.91). **Caveat:** the labeller used MRP as evidence, so this gain is partly circular and optimistic.
- **Average-linkage agglomerative clustering beats connected components** (ARI 0.296 vs 0.192), because a single wrong edge chains two products together in connected components.

## 6. Not yet working / limitations

- **BigBasket labels are Claude's, not the student's**, and the MRP rule result is optimistic for the reason above.
- Colour/shade variants with identical names (hair colours, BigBasket bottles listed per colour without colour in the name) cannot be separated by text.
- Brand blocking on BigBasket is replaced by brand-aware text (brand prepended), because brand blocks would contain millions of pairs.
- One-to-one matching constraints (useful for clean-clean benchmarks like Abt-Buy) are not used; the decision is a pure threshold.

## 7. Report section draft — 5.5 Blocking, hybrid matching and clustering

**Method.** To avoid comparing all pairs, candidates are generated by *blocking*: two records are compared only if one is among the other's 20 nearest neighbours in Sentence-BERT space (FAISS) or both carry the same brand. Each candidate pair receives a *hybrid score*, a weighted sum of SBERT cosine similarity and character n-gram TF-IDF cosine similarity, which is then adjusted by attribute rules derived from Phases 2 and 3: a quantity or pack-count mismatch vetoes the match, a brand mismatch and a model-number mismatch subtract a penalty, and, for BigBasket, an MRP difference above 5 % vetoes it. The weight, penalties and decision threshold are tuned on a validation split. Matched pairs are grouped by connected components, average-linkage agglomerative clustering, or HDBSCAN over the embeddings.

**Setup.** Pair-level precision, recall and F1 are measured on Abt-Buy and Amazon-Google (test half of the source records; gold pairs lost by blocking count as false negatives), on WDC Products (80 % corner cases) with 0, 50 and 100 % unseen products, and on 500 labelled BigBasket pairs. Clustering is evaluated on the WDC multi-class test set with the adjusted Rand index (ARI), NMI and V-measure.

**Results.** Blocking keeps 99.9 % and 99.1 % of true pairs while removing 93.6 % and 97.7 % of comparisons. The full hybrid with attribute rules reaches F1 0.753 on Abt-Buy (char TF-IDF alone 0.529, SBERT alone 0.455), 0.510 on Amazon-Google and 0.38–0.44 on WDC. On BigBasket the MRP rule raises F1 from 0.58 to 0.91. For clustering, agglomerative clustering obtains the best ARI (0.296); NMI and V-measure are uninformative on this dataset because a trivial all-singletons clustering already scores 0.947.

**Discussion.** Similarity alone cannot separate near-identical products such as model variants or pack sizes, but cheap attribute rules can, provided the attribute is present in the text. Where it is not (hair-colour shades, colour variants named identically), errors remain. These residual cases, where the score is close to the threshold, are the target of the LLM layer in Phase 6.
