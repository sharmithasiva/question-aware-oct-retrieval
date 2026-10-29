"""Tests for phase2/hybrid_retriever.py (saved embeddings only; no torch)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "phase2"))
import hybrid_retriever as H  # noqa: E402
import question_eval as QE  # noqa: E402
from oct_retrieval import intent as I  # noqa: E402
from oct_retrieval import intent_retrieval as R  # noqa: E402

EP = pd.read_csv(ROOT / "data/processed/episodes.csv")
FROZEN = ROOT / "phase2/results/question_eval_v2/dev/rankings_tierB_24a90d2768c6"
EMB = QE.load_visit_embeddings("dev")
IMAGE_Q = QE.TIER_B["overall_appearance"]


def frozen():
    rk = pd.read_csv(FROZEN / "rankings.csv")
    return {k: list(g.sort_values("rank").candidate_visit_id) for k, g in rk.groupby(["query_id", "system"])}, \
        pd.read_csv(FROZEN / "query_set.csv")


@pytest.mark.parametrize("enc", QE.ENCODERS)
def test_reproduces_frozen_tier_b_image_and_hybrid_rankings(enc):
    fr, qs = frozen()
    hyb = H.HybridRetriever(EP, EMB[enc])
    img_only = H.HybridRetriever(EP, EMB[enc], image_proxy=None)
    for q in qs.itertuples():
        r = hyb.retrieve(q.question, q.eye_id, q.query_visit_id)
        assert r.route == "image+proxy_rrf" and r.metadata_status == R.UNSUPPORTED
        assert r.final_ranking == fr[(q.query_id, f"S5_hybrid_rrf_{enc}")]
        assert r.proxy_ranking == fr[(q.query_id, "S2p_metadata_proxy_cst")]
        assert r.image_ranking == fr[(q.query_id, f"S3S4_image_{enc}")]
        assert img_only.retrieve(q.question, q.eye_id, q.query_visit_id).final_ranking == r.image_ranking


def test_metadata_questions_are_not_reordered_by_images_and_abstentions_kept():
    qs = QE.build_query_set(EP, "dev", "A")
    sysA = QE.Systems(EP, "dev")
    hyb = H.HybridRetriever(EP, EMB["resnet50"])
    fused = H.HybridRetriever(EP, EMB["resnet50"], fuse_ops=frozenset({I.CLOSEST_TO_CURRENT}))
    n_abstain = 0
    for q in qs.itertuples():
        ref = sysA.rank_all(q)
        r = hyb.retrieve(q.question, q.eye_id, q.query_visit_id)
        assert r.final_ranking == ref["S2_metadata"]                   # None when the frozen system abstains
        if ref["S2_metadata"] is None:
            n_abstain += 1
            assert r.abstained and r.route == "abstain"
            assert fused.retrieve(q.question, q.eye_id, q.query_visit_id).abstained   # never answered by images
            assert ref["S5_hybrid_rrf_resnet50"] is not None    # the v0.1 prototype S5 falls back to images here
        elif r.intent.split("/")[1] == I.CLOSEST_TO_CURRENT:
            f = fused.retrieve(q.question, q.eye_id, q.query_visit_id)
            assert f.route == "metadata+image_rrf" and f.final_ranking == ref["S5_hybrid_rrf_resnet50"]
    assert n_abstain == 4                                              # 3 last_thicker + 1 largest_rise


def test_eligibility_is_hard_even_with_extra_embeddings():
    emb = dict(EMB["resnet50"])
    other = EP[EP.eye_id == 30].visit_id.iloc[0]
    emb[other] = emb["0210MOD_V20"]                                    # identical vector from another eye
    hyb = H.HybridRetriever(EP, emb, image_proxy=None)
    r = hyb.retrieve(IMAGE_Q, 25, "0210MOD_V20")
    hist = list(EP[EP.eye_id == 25].sort_values("visit_index").visit_id)
    assert r.candidates == hist[:hist.index("0210MOD_V20")]
    assert set(r.final_ranking) == set(r.candidates) and other not in r.final_ranking
    assert "0210MOD_V20" not in r.final_ranking and "0210MOD_V21" not in r.final_ranking   # no current/future visit
    with pytest.raises(ValueError):
        hyb.retrieve(IMAGE_Q, 11, "0210MOD_V20")


def test_missing_images_are_handled_and_abstain_safely():
    emb = dict(EMB["resnet50"])
    del emb["0210MOD_V20"]
    r = H.HybridRetriever(EP, emb).retrieve(IMAGE_Q, 25, "0210MOD_V20")
    assert r.abstained and r.status == R.UNSUPPORTED and r.final_ranking is None and r.image_ranking is None
    emb = dict(EMB["resnet50"])
    for v in ("0210MOD_V3", "0210MOD_V5"):
        del emb[v]
    r = H.HybridRetriever(EP, emb, image_proxy=None).retrieve(IMAGE_Q, 25, "0210MOD_V10")
    assert r.final_ranking[-2:] == ["0210MOD_V5", "0210MOD_V3"] and set(r.no_image) == {"0210MOD_V3", "0210MOD_V5"}
    rp = H.HybridRetriever(EP, emb).retrieve(IMAGE_Q, 25, "0210MOD_V10")
    assert sorted(rp.final_ranking) == sorted(rp.candidates)
    assert H.HybridRetriever(EP, {}).retrieve(IMAGE_Q, 25, "0210MOD_V10").abstained
    # metadata questions do not need images at all
    m = H.HybridRetriever(EP, {}).retrieve("Compare with the previous visit.", 25, "0210MOD_V10")
    assert m.route == "metadata" and m.final_ranking[0] == "0210MOD_V9"


def test_unavailable_variable_and_no_history_abstain():
    hyb = H.HybridRetriever(EP, EMB["resnet50"])
    r = hyb.retrieve("When was the last injection before today?", 25, "0210MOD_V10")
    assert r.abstained and r.status in R.ABSTAIN_STATUSES and r.status != R.UNSUPPORTED
    assert r.image_ranking is not None and r.final_ranking is None      # computed for inspection, never used
    first = EP[EP.eye_id == 25].sort_values("visit_index").visit_id.iloc[0]
    assert hyb.retrieve(IMAGE_Q, 25, first).abstained


def test_table_and_rrf():
    r = H.HybridRetriever(EP, EMB["resnet50"]).retrieve(IMAGE_Q, 25, "0210MOD_V10")
    t = r.table()
    assert len(t) == 9 and t.final_rank.notna().all() and t.image_cosine.notna().all()
    assert H.rrf([["a", "b"], ["b", "a"]], ["a", "b"]) == ["b", "a"]   # tie -> more recent
    assert H.rrf([["a", "b"]], ["a", "b", "c"]) == ["a", "b", "c"]
