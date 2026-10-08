# Project explanation script (about 8–10 minutes)

*Read the parts in quotes aloud. Lines in italics are actions.*

---

## 1. The problem (1 minute)

"Good morning. Our project is called **LLM-Assisted Product Entity Resolution**.

Here is the problem. Different online shops sell the same products, but they describe them differently. One shop's column is called *product*, another's is *title*, another's is *Item Name*. One shop writes *Tata Salt 1kg*, another writes *TATA salt 1000 gm*. Some products are even listed twice in the same shop.

If you want to compare prices, merge catalogues or remove duplicates, a computer has to understand that these are the same product. That task is called **entity resolution**, and that is what our system does."

*Open `data/demo/bigbasket_demo_renamed_columns.csv` and point at the column names and a few repeated products.*

---

## 2. What the system does (1 minute)

"The system works in five steps:

1. It **standardizes the columns**: whatever the columns are called, it maps them to one standard set: product name, brand, category, price, MRP, quantity and so on.
2. It **selects the columns** worth comparing, for example the product name, and explains why.
3. It **cleans and understands each product name** using NLP: it fixes case and spelling, converts units, and pulls out the brand, quantity, pack size and colour.
4. It **finds and groups** listings of the same product, using rules, string similarity and the cosine similarity of embeddings.
5. It uses a **local large language model** only where the cheaper methods are unsure.

The professor asked us to make the project much more NLP-focused and to use LLMs where they genuinely help. We'll show where each of those happens."

---

## 3. Data (30 seconds)

"We used nine public datasets. Our working data is two Indian catalogues: **BigBasket** with 27,555 grocery products, and **Flipkart** with 20,000 listings. To measure accuracy we used benchmarks that come with correct answers: **Abt-Buy** and **Amazon-Google** for matching, **WDC Products** for hard matching and clustering, and **WDC-PAVE** for attribute extraction. Every number we show is measured on one of these labelled sets."

---

## 4. Step by step, with results (4 minutes)

### Step 1: Column standardization
"First, column names. We tried three approaches. Simple **keyword rules** on the header got 90 % right. **Sentence embeddings** got only 64 %, because the sample values confused them. The **LLM** got 95 %, but it is slow, about 10 seconds per question on our laptop.

So we combined them: rules first, and the LLM only when no rule recognises the header. That reached **96.6 % accuracy** while asking the LLM about only 13 of 59 columns. This is the pattern for the whole project: **cheap methods first, LLM only where needed.**"

### Step 2: Column selection
"Next, which columns to compare. For each column the system measures how many words it has, how unique its values are, and whether it is text or numbers. IDs, prices and links are rejected; free-text columns like the product name are selected. Every decision comes with a written reason. **F1 0.96** on our test set."

### Step 3: Cleaning the names (NLP)
"Now the names themselves. We use **spaCy** for tokenization and lemmatization, so *biscuits* becomes *biscuit*. One thing we found: spaCy only lemmatizes after lowercasing, because it treats capitalized words as names.

We wrote rules that read quantities in many spellings: *1kg*, *1000 gm* and *1 kilo* all become **1000 g**. Checked against the WDC-PAVE answers, this is **98.8 % correct**. We also correct spelling with **SymSpell**, using a dictionary built from our own product names; it restores 91 % of typos."

### Step 4: Extracting attributes (NER)
"Then we pull out the brand, product type, colour, quantity and pack size, which is named entity recognition. We compared four methods on 354 real product titles:

- **Rules** with a brand list: F1 0.34
- **GLiNER**, a model that finds entities from label names alone: 0.51
- **The LLM**: 0.59, the most accurate
- A **small spaCy model** we trained: 0.17, but 13,500 times faster than the LLM

So the LLM is the most accurate, but far too slow to run on every product. That is exactly the trade-off the professor asked us to think about."

### Step 5: Representing text and measuring similarity
"To compare names, we turn them into numbers in five ways: **TF-IDF** on words, TF-IDF on characters, **fastText** word vectors, and two **Sentence-BERT** models. Then we search with **cosine similarity** using **FAISS**.

The surprising result: plain word TF-IDF was best, finding the right match first **83 %** of the time on Abt-Buy. The reason is that product names depend on exact model numbers, which TF-IDF matches exactly, while sentence embeddings are better at meaning, such as knowing a washer is not a dryer. They make different mistakes, so we combine them."

### Step 6: Matching and grouping
"First we only compare products that are likely matches, which removes **94 %** of comparisons while keeping **99.9 %** of the true pairs. Then each pair gets a **hybrid score**: embedding similarity plus character similarity. Then **rules** apply: a different size, pack count or model number means a different product.

On Abt-Buy, the rules raised the matching score from **0.53 to 0.75**. On BigBasket, a rule using the printed MRP raised it to **0.91**. Finally we group the matches into clusters. BigBasket ends up with a `cluster_id` for every product, and **996 groups** of duplicates."

### Step 7: The LLM where it matters
"Some pairs score right at the decision line: the system is unsure. We send only those to the LLM and ask, 'Are these the same product? Answer in JSON.' On these hard pairs the LLM was right **85 %** of the time, against **66 %** for the plain threshold. It is especially good at noticing that two model numbers differ.

We also tested the LLM for categorizing products. There it did **no better** than simple embeddings, so we report that honestly: the LLM adds value for reasoning about uncertain pairs, not for everything."

---

## 5. Live demo (2–3 minutes)

*Run `streamlit run app/streamlit_app.py`. Choose `bigbasket_demo_renamed_columns.csv`.*

1. *Tab 1, Column mapping:* "Our renamed columns, *Item Name*, *Brand Name*, *MRP (Rs)*, are mapped automatically."
2. *Tab 2, Column selection:* "It picked *Item Name* to compare, and explains why the price and ID columns were rejected."
3. *Tab 3, Attributes:* "Quantities converted to grams and millilitres; pack counts; colours."
4. *Tab 4, Clusters:* "Listings of the same product grouped, each with one clean name."
5. *Tab 5, Search:* *type* `2 litre cola` "Semantic search: the words don't have to match exactly."
6. *Switch to `unseen_open_food_facts_world.csv`:* "And a file the system has never seen, with 25 messy columns. It still works."

---

## 6. Honest limitations (30 seconds)

"Three limitations:
1. **LLM cost.** On a laptop each call takes about 15 seconds, so we had to limit how often we use it.
2. **Hinglish.** Names like *haldi powder* or *chawal* are still hard, because our main model is English-only, and the LLM was weak at writing Hinglish test data.
3. **Labels.** Three small test sets were labelled during the project and still need a careful manual check.

Also, the fine-tuning evaluation, the full comparison table and topic modelling are coded but were not fully run before the deadline."

---

## 7. Conclusion (30 seconds)

"To summarize: we built a complete pipeline from messy tables to clean product groups. It is built on NLP — spaCy, NER, TF-IDF, fastText and Sentence-BERT — and uses a large language model only where it measurably helps. Every claim is backed by a number on a labelled test set, and all of it is listed in our requirements traceability file. Thank you, we're happy to take questions."

---

### Numbers to remember

| | |
|---|---|
| Column mapping | 96.6 % (LLM for 13 of 59 columns) |
| Column selection | F1 0.96 |
| Unit normalization | 98.8 % |
| Attribute extraction | LLM 0.59, GLiNER 0.51, rules 0.34 |
| Best retrieval | word TF-IDF, 83 % top-1 |
| Matching with rules | 0.53 → 0.75 (Abt-Buy) |
| LLM on uncertain pairs | 85 % vs 66 % |
| BigBasket duplicates | 996 groups |
