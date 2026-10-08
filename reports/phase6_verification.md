# Phase 6 — Verification Report (LLM layer)

**Goal (roadmap):** use the LLM only where rules and embeddings are unsure, and to make the output readable.
**Done when (roadmap):** F1 before vs after adjudication, plus a table of clusters with canonical names.
**Status: PASSED, at a reduced scale.** Both deliverables exist: `reports/phase6_adjudication.csv` and `reports/phase6_canonical_names.json`. **Scale was cut at the student's request** (finish within minutes instead of ~4.5 h of LLM time at ~15 s per call). The reductions are listed in §6 and every number below states its sample size.

## 1. What was built (`src/llm_layer.py`, all calls through `src/llm.py`)

| Task | How | Output schema (Pydantic) |
|---|---|---|
| Grey-zone adjudication | Phase 5 "hybrid + all rules" scores on the **test** split; the pairs closest to the tuned threshold (a fixed LLM budget) are sent to qwen3:8b; its answer replaces the threshold decision | `SameProduct {same: bool, reason: str ≤ 200}` |
| Canonical naming | for a BigBasket cluster, the LLM writes one standard name from up to 10 member names; baseline = the **medoid** member (closest to all others in SBERT space, no LLM) | `CanonicalName {name: str ≤ 120}` |
| Zero-shot categorization | BigBasket's 11 categories as labels; 10 products per call; baseline = cosine between the product and "<category> products" with MiniLM and BGE-small | `CategoryBatch {categories: list[Literal[11 categories]]}` (wrong count = invalid) |
| Synthetic data (for Phase 7) | **made by rules, not by the LLM** (time budget): 2,000 BigBasket names × 3 noisy variants (random typos, retail abbreviations, word reordering) + 1 hard negative (number doubled, or pack/size/variant suffix added) | `data/processed/synthetic_pairs.json` |
| Hinglish test set (for Phase 7) | the LLM rewrites a name casually in English and in Hinglish (Hindi words in Roman script); names are chosen to contain a grocery word with a common Hindi equivalent | `HinglishRewrite {english_noisy, hinglish}` → `data/gold/hinglish_test.csv` (15 items) |

Also added in this phase: `num_predict = 300` caps every LLM reply. Without it, one Hinglish reply looped until the context window was full (≈ 8 min for a single call); the cap turns such a loop into an invalid reply, which the retry and failure logic handles.

## 2. Requirements check

| Req | How this phase meets it | Evidence |
|---|---|---|
| R6 LLM only where it adds value | adjudication only for the most uncertain pairs (65 of 37,101 Abt-Buy test candidates = 0.18 %); every task compared with a non-LLM baseline; LLM calls logged | `reports/phase6_llm_usage.json`; tables below |
| R7 Frameworks | Ollama + Pydantic structured output, sentence-transformers baselines | `src/llm_layer.py` |
| R8 Measured on labelled data | adjudication on Abt-Buy gold; categorization on BigBasket category labels | tables below |

## 3. Results

### 3.1 Grey-zone adjudication, Abt-Buy test split

| | Precision | Recall | F1 |
|---|---|---|---|
| Phase 5 hybrid + rules (before) | 0.737 | 0.769 | 0.753 |
| + LLM on 65 grey-zone pairs (after) | **0.754** | 0.767 | **0.760** |

On the 65 grey-zone pairs themselves (the pairs closest to the threshold, |s − t| ≤ 0.029):

| | Accuracy on the 65 pairs |
|---|---|
| threshold decision | 0.662 |
| **LLM decision** | **0.846** |

Breakdown: the LLM **fixed 19** threshold errors (15 false positives, 4 false negatives) and **broke 7** correct decisions (5 true matches rejected, 2 non-matches accepted). The 65 decisions took about 15 s each (54 calls between 11:51 and 12:04), all valid JSON. They were made during the full run that was later stopped and are cached, so the reported run made 0 new calls.

### 3.2 Canonical names (10 BigBasket clusters, `reports/phase6_canonical_names.json`)

| Cluster (size) | Medoid name (no LLM) | LLM canonical name |
|---|---|---|
| 2 | `MAGGI  Cuppa Noodles - Chilli Chow` | `Maggi Cuppa Noodles Chilli Chow` |
| 2 | `bb Royal Organic - Quinoa` | `Bb Royal Organic Quinoa` |
| 2 | `bb Combo BB Popular Channa Dal 1 kg + Urad Dal - Split 1 kg+BB Royal Rice - Dosa 2 kg` | `BB Combo Popular Channa Dal 1 Kg Urad Dal Split 1 Kg BB Royal Rice Dosa 2 Kg` |
| 2 | `Asian Smart Stackable Cont Set` | `Asian Smart Stackable Cont Set` |
| 2 | `Open Secret Choco Almond Nutty Cookies - Box of 2, 25 g + …` | `Open Secret Choco Almond Nutty Cookies Box of 2 25g Choco Almond Butter Spread 175g` |

Brand kept in the name: LLM 10/10, medoid 10/10. LLM time: 2 new calls (8 cached), 101 s.

### 3.3 Zero-shot categorization (BigBasket, 11 categories)

| Method | Products | Accuracy | Macro-F1 | LLM calls |
|---|---|---|---|---|
| embedding cosine, MiniLM | 550 | 0.456 | 0.423 | 0 |
| embedding cosine, BGE-small | 550 | **0.471** | **0.449** | 0 |
| embedding cosine, MiniLM | 55 | 0.418 | 0.405 | 0 |
| embedding cosine, BGE-small | 55 | 0.382 | 0.336 | 0 |
| LLM zero-shot (qwen3:8b), 10 per call | 55 | 0.418 | 0.365 | 6 (260 s), 0 invalid |

### 3.4 Data for Phase 7

- Synthetic: 2,000 anchors → 6,000 positive pairs + 2,000 hard negatives (rules).
- Hinglish: 15 LLM rewrites. Genuine Hindi substitutions appear in only a few (`oil → tel`, `rice → chawal`, `tea → chai`); the LLM usually left words such as *almond* (badam) unchanged. Phase 7 therefore adds a second, dictionary-based Hinglish rewrite of the same 15 names.

## 4. Examples

| # | Input | LLM output | Note |
|---|---|---|---|
| 1 | `Escort Passport Radar And Laser Detector - … - 8500` vs `Escort 9500CI Escort GPS Radar Detector` (threshold: match) | `same: false` — "Different model numbers (8500 vs 9500CI)" | **fixed** a false positive |
| 2 | `Sony Soft Cyber-Shot Carrying Case - LCSCST` vs `Sony LCS-CSH Soft Camera Case - LCSCSH` | `same: false` — "model numbers LCSCST and LCSCSH are different" | **fixed** a false positive the prefix-based code rule missed |
| 3 | `Netgear ProSafe … - FS116P` vs `Netgear ProSafe FS116P Ethernet Switch - FS116PNA` (gold: same) | `same: false` — "Different model numbers (FS116P vs FS116PNA)" | **LLM error:** "NA" is a region suffix, not a different model |
| 4 | `Nikon 18-200mm Nikkor Zoom Lens - … - 2159` vs `Nikon 18-200mm 3.5-5.6 G ED DX AFS VR … Lens - Niko_215930348` (gold: same) | `same: false` | **LLM error:** extra specifications read as a different product |
| 5 | `MAGGI  Cuppa Noodles - Chilli Chow` cluster | `Maggi Cuppa Noodles Chilli Chow` | clean, but only a cosmetic improvement over the medoid |

## 5. Discussion

- **Adjudication is where the LLM earns its cost.** On the hardest pairs it is right 85 % of the time against 66 % for the threshold, mostly by reasoning about model numbers. Its errors are the mirror image: it treats suffixes and extra specs as different models. Because only 65 of 37,101 pairs were sent (a 15-minute budget), the end-to-end F1 gain is small (0.753 → 0.760); a larger budget would apply the same 85 % accuracy to more of the ~750 pairs within ±0.10 of the threshold.
- **Canonical naming adds little on BigBasket:** clusters are relistings with near-identical names, so the medoid is already a good name and the LLM mostly fixes case and punctuation.
- **Zero-shot categorization does not justify the LLM:** on the same 55 products it ties MiniLM on accuracy (0.418) at 4.3 minutes of LLM time, and BGE embeddings on all 550 products are better (0.471). BigBasket's categories overlap ("Gourmet & World Food" vs "Snacks & Branded Foods"), which limits every method.
- **qwen3:8b is a weak Hinglish writer:** it rarely translates words into Hindi unprompted, which is why Phase 7 also uses a dictionary-based rewrite.

## 6. Reductions made for time (stated plainly)

| Planned | Done | Reason |
|---|---|---|
| 200 grey-zone pairs each on Abt-Buy and WDC | 65 on Abt-Buy, 0 on WDC | ~15 s per call; the student asked to finish within minutes |
| 100 canonical names | 10 | same |
| LLM categorization of 550 products + Google taxonomy | 55 products, Google taxonomy skipped | same |
| 200 LLM-written synthetic anchors | 2,000 rule-made anchors | same; rules are instant |
| 200 Hinglish test items | 15 | same |

All numbers on 55, 65 or 15 items have wide uncertainty; they show the direction of each effect, not a precise size.

## 7. Report section draft — 5.6 LLM layer

**Method.** The LLM (qwen3:8b) is used in four places, always through one gateway that constrains the output to a Pydantic schema, validates it, retries invalid replies, caps the reply length and caches every answer. (i) *Grey-zone adjudication*: the candidate pairs whose hybrid score lies closest to the decision threshold are shown to the LLM, which answers whether the two listings are the same product, with a reason; its answer replaces the threshold decision. (ii) *Canonical naming*: for each cluster the LLM writes one standard name, compared with the medoid member as a no-LLM baseline. (iii) *Zero-shot categorization* into BigBasket's eleven categories, compared with assigning the category whose name embedding is most similar. (iv) *Test-data generation*: Hinglish rewrites of product names for Phase 7.

**Setup.** Adjudication is evaluated on the Abt-Buy test split; categorization against BigBasket's own category labels. Because each call takes about 15 s on our hardware, the LLM was given a fixed budget: 65 adjudications, 10 names, 55 categorizations and 15 rewrites.

**Results.** On the 65 most uncertain pairs, the LLM's decisions are 84.6 % correct against 66.2 % for the threshold, raising Abt-Buy F1 from 0.753 to 0.760. For categorization the LLM (accuracy 0.418 on 55 products) does not outperform embedding similarity (0.418 MiniLM on the same products; 0.471 BGE on 550). Canonical names produced by the LLM are mostly cosmetic improvements over the medoid name.

**Discussion.** The LLM adds value where reasoning about product identifiers is needed and the cheaper signals disagree. It adds little where similarity already works, and its cost (four orders of magnitude slower than the matcher) means it must be restricted to a small fraction of decisions.
