# Likely viva questions, with short answers

**1. What problem does the project solve?**
Product entity resolution: different shops list the same product with different column names and differently written names. We map columns to one schema, normalize and parse names, and group listings that are the same product.

**2. Where exactly is NLP used?**
Header tokenization; Unicode/case/punctuation normalization; spaCy tokenizer, rule-based `Matcher` and lemmatizer; spelling correction (SymSpell); NER/attribute extraction (EntityRuler, GLiNER, LLM, trained spaCy NER); text representations (word and character TF-IDF, fastText, Sentence-BERT); fine-tuning; topic modelling (BERTopic).

**3. Why not use the LLM for everything?**
It is slow on our laptop (≈15 s per call vs about 950 names/s for the distilled spaCy model) and not always better: for categorization it tied simple embeddings (0.418). We use it only where cheaper methods are unsure: unknown headers (13/59) and grey-zone pairs, where it was right 85 % of the time against 66 % for the threshold.

**4. How do you force the LLM to give structured output?**
Ollama receives the JSON schema of a Pydantic model (`format=`); the reply is validated with Pydantic; invalid replies are retried and then counted as failures. Replies are capped at 300 tokens and cached in `cache/llm_responses.json` so results are reproducible.

**5. Why did lexical TF-IDF beat Sentence-BERT in retrieval?**
Product names hinge on brand names and model codes. TF-IDF matches these exactly and weights rare tokens highly; sentence embeddings blur codes that differ by one character (GSD4000 vs GSD2400). Embeddings win on paraphrases (washer vs dryer, carrying vs carry), so we combine both.

**6. What is the hybrid score?**
w · cos(SBERT) + (1 − w) · cos(char 3–5-gram TF-IDF), then rules: a quantity or pack-count mismatch vetoes the match; a brand or model-number mismatch subtracts a penalty. w, penalties and threshold are tuned on a validation split and reported on the test split.

**7. What is blocking and why do you need it?**
Comparing all pairs is quadratic (1.2 M pairs for Abt-Buy, 380 M for BigBasket). We only compare records in each other's FAISS top-20 or with the same brand. This keeps 99.9 % of true pairs while removing 93.6 % of comparisons.

**8. Why is ARI your main clustering metric and not NMI?**
On WDC with about 2 offers per product, putting every offer in its own cluster already gets NMI 0.947. ARI is corrected for chance and gives that baseline 0.

**9. What is cosine similarity, and why normalize vectors?**
cos(a, b) = a·b / (‖a‖‖b‖). After L2-normalization the inner product equals the cosine, so FAISS's inner-product index does exact cosine search.

**10. Why lowercase before lemmatizing?**
spaCy's small English model tags capitalized words as proper nouns, which are not lemmatized ("Biscuits" stays "Biscuits"); after lowercasing, "biscuits" → "biscuit".

**11. Why did you not correct the spelling of every rare word?**
We measured it: allowing corrections of rare words that do appear in the catalogue changed 17.8 % of them and only 7 of 40 judged changes were real fixes (e.g. "timeless" → "wireless"). So only unseen words are corrected (91 % of synthetic typos restored).

**12. What is GLiNER?**
A zero-shot NER model: you give it label names ("brand", "product type") at prediction time and it finds spans, with no training. It reached macro F1 0.505 on WDC-PAVE.

**13. What is distillation here, and why did it do poorly?**
A large model (teacher, GLiNER) labels 2,000 names; a small spaCy NER model learns from those labels and runs much faster. It reached only 0.165 because it learns the teacher's mistakes, and it was trained on Indian grocery/fashion names but tested on electronics and office products (domain shift). Its learning curve was still rising.

**14. What is MultipleNegativesRankingLoss?**
A contrastive loss for sentence embeddings: in each batch, an anchor's positive must score higher than every other example in the batch (in-batch negatives). It only needs positive pairs.

**15. What does the price/MRP rule do, and is it fair?**
When two columns both look like "price", the one with the higher median becomes MRP, since MRP is never below the selling price. On BigBasket, products with the same name but an MRP more than 5 % apart are treated as different sizes. Caveat: the BigBasket gold labels were made using MRP as evidence, so that F1 (0.906) is optimistic.

**16. How do you know your numbers are not tuned on the test set?**
Thresholds and weights are tuned on validation splits (official WDC validation; seeded 50/50 splits elsewhere) and reported on the test parts. Where a design was changed after looking at test errors (Phase 1 cascade, Phase 2 unit fixes), the report says so and also gives the untuned number.

**17. What are the main limitations?**
LLM cost on a laptop (we had to cut LLM budgets); weak Hinglish generation by qwen3:8b; gold sets labelled by the assistant that the student must verify; colour/shade variants with identical names cannot be separated by text.

**18. How is the system evaluated end to end?**
The ablation table runs six configurations (rules + fuzzy → TF-IDF → fastText → SBERT → SBERT + rules → fine-tuned SBERT + rules + LLM) on the same test sets with the same tuning procedure.

**19. What is Hinglish and why does it matter?**
Hindi written in Roman script mixed with English ("chawal basmati 5 kg"). Indian shoppers search this way; English-only models do not know that "chawal" means rice. We test English vs multilingual models (LaBSE, multilingual-E5) and transliteration to Devanagari with Aksharantar.

**20. What would you do with more time?**
A larger LLM budget for grey-zone pairs (its accuracy there was 85 %), LLM-generated training data instead of rule-based, a hand-checked Hinglish set of 200 items, BGE instead of MiniLM as the base model, and one-to-one matching constraints for clean–clean datasets.
