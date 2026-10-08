"""Product entity resolution demo.

    streamlit run app/streamlit_app.py

Upload a product CSV (any column names) -> detected column mapping -> extracted attributes
-> clusters with canonical names -> semantic search ("2 litre cola").
"""
import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import pipeline  # noqa: E402

st.set_page_config(page_title="Product Entity Resolution", layout="wide")
st.title("LLM-assisted product entity resolution")
st.caption("Column mapping → normalization → attribute extraction → hybrid matching → clusters → search")

demo_files = sorted((ROOT / "data" / "demo").glob("*.csv"))
left, right = st.columns([2, 1])
upload = left.file_uploader("Upload a product table (CSV)", type="csv")
demo = right.selectbox("…or use a demo file", ["(none)"] + [f.name for f in demo_files])
use_llm = right.checkbox("Use the LLM (qwen3:8b) for unknown columns and cluster names", value=False,
                         help="Slower: about 5-15 s per LLM call on this laptop. Needs Ollama running.")
max_rows = right.number_input("Rows to process", min_value=50, max_value=5000, value=1000, step=50)

if upload is not None:
    raw_bytes, name = upload.getvalue(), upload.name
elif demo != "(none)":
    raw_bytes, name = (ROOT / "data" / "demo" / demo).read_bytes(), demo
else:
    st.info("Upload a CSV or choose a demo file to start.")
    st.stop()


@st.cache_data(show_spinner=False)
def run_cached(data: bytes, use_llm: bool, n: int):
    raw = pd.read_csv(io.BytesIO(data), sep=None, engine="python", nrows=n, on_bad_lines="skip")
    res = pipeline.run(raw, use_llm=use_llm)
    res.pop("embeddings")          # not cacheable as data; recomputed for search below
    return raw, res


with st.spinner("Running the pipeline…"):
    raw, res = run_cached(raw_bytes, use_llm, int(max_rows))

st.success(f"{name}: {len(raw)} rows, {raw.shape[1]} columns → {res['clusters'].shape[0]} product clusters "
           f"({int((res['clusters']['size'] > 1).sum())} with more than one listing)")

tab_map, tab_sel, tab_attr, tab_clu, tab_search = st.tabs(
    ["1. Column mapping", "2. Column selection", "3. Attributes", "4. Clusters", "5. Search"])

with tab_map:
    st.subheader("Detected mapping to the canonical schema")
    st.dataframe(res["mapping"], width="stretch", hide_index=True)
    st.subheader("Canonical table")
    st.dataframe(res["canonical"].head(50), width="stretch", hide_index=True)

with tab_sel:
    st.subheader("Which columns are worth clustering, and why")
    st.dataframe(res["selection"][["column", "avg_tokens", "unique_ratio", "alpha_ratio", "entropy",
                                   "selected", "role", "reason"]], width="stretch", hide_index=True)

with tab_attr:
    st.subheader("Attributes extracted from each product name")
    st.dataframe(res["attributes"].head(200), width="stretch", hide_index=True)

with tab_clu:
    st.subheader("Clusters of listings that refer to the same product")
    c = res["clusters"]
    only_multi = st.checkbox("Only clusters with more than one listing", value=True)
    view = c[c["size"] > 1] if only_multi else c
    st.dataframe(view.assign(members=view["members"].map(lambda m: " | ".join(m))),
                 width="stretch", hide_index=True)
    st.caption(f"Matching settings: weight on SBERT = {res['params']['w']}, threshold = {res['params']['threshold']:.3f}, "
               f"MRP rule = {res['params']['use_mrp']}")

with tab_search:
    st.subheader("Semantic search")
    query = st.text_input("Search the uploaded products", value="2 litre cola")
    if query:
        if "search_state" not in st.session_state or st.session_state.get("search_key") != (name, use_llm, max_rows):
            pre = res["preprocessed"]
            st.session_state["search_state"] = {
                "preprocessed": pre,
                "embeddings": pipeline.embedder().encode(pre["text"].tolist(), batch_size=128,
                                                         normalize_embeddings=True, show_progress_bar=False)}
            st.session_state["search_key"] = (name, use_llm, max_rows)
        st.dataframe(pipeline.search(st.session_state["search_state"], query), width="stretch",
                     hide_index=True)
