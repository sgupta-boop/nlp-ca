"""Phase 4 (representations, retrieval) and Phase 5 (rules, scoring, clustering) tests."""
import numpy as np
import pandas as pd
from scipy import sparse

from src import embed as E
from src import match as M


# ---------------- Phase 4
def test_retrieval_metrics_known_ranks():
    ranked = [["a", "b"], ["x", "y", "z"], ["p", "q"]]
    gold = [{"a"}, {"z"}, {"nope"}]
    m = E.retrieval_metrics(ranked, gold)
    assert m["recall@1"] == round(1 / 3, 3) and m["recall@5"] == round(2 / 3, 3)
    assert m["mrr"] == round((1 + 1 / 3 + 0) / 3, 3)


def test_cosine_topk_dense_and_sparse_agree():
    rng = np.random.default_rng(0)
    a = rng.random((5, 8)).astype("float32")
    b = rng.random((7, 8)).astype("float32")
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    i_dense, s_dense = E.cosine_topk(a, b, 3)
    i_sparse, s_sparse = E.cosine_topk(sparse.csr_matrix(a), sparse.csr_matrix(b), 3)
    assert (i_dense == i_sparse).all() and np.allclose(s_dense, s_sparse, atol=1e-5)


def test_tfidf_char_handles_format_differences():
    enc = E.Encoder("tfidf_char").fit(["sony ps lx350h turntable", "sony pslx350h", "bose speaker"])
    v = enc.encode(["sony pslx350h", "sony ps lx350h turntable", "bose speaker"])
    sims = (v @ v.T).toarray()
    assert sims[0, 1] > sims[0, 2]


# ---------------- Phase 5
def test_model_codes_join_hyphens_and_skip_units():
    assert M.model_codes("Sony PS-LX350H Turntable 250GB") == {"pslx350h"}
    assert M.model_codes("Bose AM53BK speakers") == {"am53bk"}


def test_codes_conflict_prefix_is_not_conflict():
    assert not M.codes_conflict({"am53bk"}, {"am53"})
    assert M.codes_conflict({"pslx350h"}, {"pslx250h"})
    assert not M.codes_conflict(set(), {"x100"})


def test_hybrid_score_vetoes_and_penalties():
    f = pd.DataFrame({"sbert": [0.9, 0.9, 0.9], "char": [0.7, 0.7, 0.7],
                      "qty_conflict": [False, True, False], "pack_conflict": [False, False, False],
                      "brand_conflict": [False, False, True], "code_conflict": [False, False, False]})
    s = M.hybrid_score(f, w=0.5, brand_penalty=0.2)
    assert np.allclose(s, [0.8, 0.0, 0.6])


def test_tune_finds_perfect_threshold_on_separable_data():
    f = pd.DataFrame({"sbert": [0.9, 0.8, 0.3, 0.2], "char": [0.9, 0.8, 0.3, 0.2],
                      "qty_conflict": False, "pack_conflict": False, "brand_conflict": False, "code_conflict": False})
    p = M.tune(f, np.array([True, True, False, False]), grid_w=(1.0,), use_rules=False)
    assert p["f1"] == 1.0 and 0.3 < p["threshold"] <= 0.8


def test_prf():
    assert M.prf({1, 2, 3}, {2, 3, 4, 5}) == {"precision": 0.667, "recall": 0.5, "f1": 0.571}


def test_clustering_methods_on_block_matrix():
    S = np.array([[1, .9, .1, .1], [.9, 1, .1, .1], [.1, .1, 1, .95], [.1, .1, .95, 1]])
    gold = [0, 0, 1, 1]
    for fn in (M.cluster_components, M.cluster_agglomerative):
        assert M.cluster_scores(gold, fn(S, 0.5))["ARI"] == 1.0
