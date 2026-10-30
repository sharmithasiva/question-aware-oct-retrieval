"""Unified question-aware routing, evidence fusion, abstention and grounding on a SYNTHETIC eye (offline: no repository
data, API key, network or GPU). Eye 901 has 8 visits; as in OLIVES, finding labels exist only for its first (V1) and
last (V8) visit. V1: IRF 50 %, DRT 40 %; V8: IRF 10 %, DRT 0 %.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "phase2"))
import answer as A  # noqa: E402
import demo as D  # noqa: E402
import question_router as QR  # noqa: E402
from oct_retrieval import intent as I  # noqa: E402
from oct_retrieval import intent_retrieval as R  # noqa: E402

CST = [500, 450, 300, 405, 380, 420, 390, 400]
LAST = "9901AOD_V8"


def history(eye=901, subject="9901AOD", csts=CST, irf=(0.5, 0.1), drt=(0.4, 0.0)):
    rows = []
    for i, c in enumerate(csts):
        lab = i in (0, len(csts) - 1)
        k = 0 if i == 0 else 1
        rows.append(dict(visit_id=f"{subject}_V{i + 1}", patient_id=eye, eye_id=eye, subject=subject, trial="TREX DME",
                         split="train", laterality="OD", visit_label=f"V{i + 1}", visit_number=i + 1, visit_index=i,
                         week=np.nan, n_visits_eye=len(csts), cst=float(c), bcva=70.0 + i, n_images=49,
                         has_biomarkers=lab, bm_n_scans=49.0 if lab else np.nan,
                         bm_frac_fluid_irf=irf[k] if lab else np.nan, bm_any_fluid_irf=float(irf[k] > 0) if lab else np.nan,
                         bm_frac_fluid_srf=0.0 if lab else np.nan, bm_any_fluid_srf=0.0 if lab else np.nan,
                         bm_frac_drt_me=drt[k] if lab else np.nan, bm_any_drt_me=float(drt[k] > 0) if lab else np.nan))
    return rows


@pytest.fixture
def demo(tmp_path):
    ep = pd.DataFrame(history() + history(902, "9902BOS", [300, 320, 350], irf=(0.0, 0.0)))
    rng = np.random.default_rng(1)
    emb, images = {}, {}
    for i, v in enumerate(ep.visit_id):
        x = rng.normal(size=16)
        emb[v] = x / np.linalg.norm(x)
        images[v] = tmp_path / f"{v}.png"
        Image.fromarray(np.full((8, 8), i * 9, np.uint8)).save(images[v])
    return D.Demo.from_parts(ep, emb, images)


class TextClient:
    def __init__(self, text):
        reply = SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=text)])
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: reply))


class NoCallClient:
    beta = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(
        AssertionError("the model must not be called"))))


ROUTING = [
    ("When was CST highest?", "metadata", "retrieved"),
    ("Which earlier OCT looks most similar?", "image_similarity", "retrieved"),
    ("When was edema worst?", "edema_proxy", "retrieved"),
    ("When did intraretinal fluid first appear?", "pathology", "insufficient_labels"),
    ("What was the largest consecutive CST increase?", "metadata", "retrieved"),
    ("Compare anatomical appearance across visits", "image_similarity", "retrieved"),
    ("Which earlier visit looks similar, and how did CST change?", "mixed", "retrieved"),
    ("Which earlier visit looks most like today's, and when was CST highest?", "mixed", "retrieved"),
    ("When was the retina thickest?", "metadata", "retrieved"),
    ("Has the swelling improved since the worst visit?", "edema_proxy", "retrieved"),
    ("What changed in the retinal layers?", QR.UNSUPPORTED_FINDING, QR.UNSUPPORTED_FINDING),
    ("When was the last injection before today?", "abstain", "unavailable_variable"),
    ("hello there", "abstain", "clarification_needed"),
    ("Which visit looks similar and why is the sky blue?", QR.PARTIAL, "clarification_needed"),
]


@pytest.mark.parametrize("q,category,status", ROUTING)
def test_routing_table(demo, q, category, status):
    u = demo.router.retrieve(q, 901, LAST)
    assert (u.category, u.status) == (category, status)
    assert all(v in u.candidates for v in (u.final_ranking or []))          # eligibility: earlier, same eye
    assert LAST not in (u.final_ranking or [])


@pytest.mark.parametrize("q", ["When was CST highest?", "What was the largest consecutive CST increase?",
                               "Most recent prior visit with CST above today's value?", "When was edema worst?",
                               "Compare with the previous visit."])
def test_metadata_rankings_are_the_frozen_rankings(demo, q):
    u = demo.router.retrieve(q, 901, LAST)
    mem = demo.retriever.memory(901)
    frozen = R.retrieve(I.parse_intent(q), mem, 7)
    assert u.final_ranking == [mem.visit_ids[i] for i in frozen.ranked]
    assert [c.name for c in u.components] == ["metadata"] and u.components[0].explanation


def test_edema_is_a_disclosed_cst_proxy_with_images_and_eligible_labels_only(demo):
    out = demo.run("When was edema worst?", 901, "9901AOD_V5")             # V8 (labelled) is in the future
    ev, u = out["evidence"], out["unified"]
    assert ev["category"] == "edema_proxy" and "not a clinical edema grade" in " ".join(ev["disclosures"])
    assert ev["retrieved"][0]["visit_id"] == "9901AOD_V1"                  # highest CST among V1..V4
    assert ev["uses_images"] and [i["visit_id"] for i in ev["images"]] == ["9901AOD_V5"] + [
        r["visit_id"] for r in ev["retrieved"]]
    assert {b["visit_id"] for b in u.biomarkers} == {"9901AOD_V1"}         # future V8 labels never read
    assert A.evidence_inconsistencies(ev) == [] and "Using CST as a proxy for edema" in out["answer"]["answer"]


def test_metadata_route_sends_no_images_and_no_labels(demo):
    ev = demo.run("When was CST highest?", 901)["evidence"]
    assert ev["images"] == [] and not ev["uses_images"] and ev["biomarkers"] == []


@pytest.mark.parametrize("q,visit,status,ranking", [
    ("When did intraretinal fluid first appear?", LAST, "insufficient_labels", None),     # 6 unlabelled visits
    ("When did intraretinal fluid first appear?", "9901AOD_V2", "retrieved", ["9901AOD_V1"]),
    ("When was subretinal fluid present?", "9901AOD_V2", "no_qualifying_visit", None),   # SRF 0 at V1
    ("Has the intraretinal fluid improved?", "9901AOD_V2", "insufficient_labels", None),   # today unlabelled
    ("When was the cystoid fluid worst?", "9901AOD_V2", "retrieved", ["9901AOD_V1"]),
    ("Was there intraretinal fluid before?", "9901AOD_V1", "insufficient_data", None),     # no history
])
def test_pathology_uses_labels_only_and_abstains_without_them(demo, q, visit, status, ranking):
    out = demo.run(q, 901, visit, backend="claude", client=NoCallClient() if status != "retrieved" else None)
    u, ev = out["unified"], out["evidence"]
    assert u.category == "pathology" and u.status == status and u.final_ranking == ranking
    assert [c.name for c in u.components] == ["biomarker_labels"] and not ev["uses_images"] and ev["images"] == []
    assert all(b["visit_id"] in u.candidates + [visit] for b in u.biomarkers)
    if status != "retrieved":
        assert out["answer"]["status"] == "abstained" and ev["retrieved"] == []


def test_mixed_question_fuses_image_ranking_with_measured_change(demo):
    out = demo.run("Which earlier visit looks similar, and how did CST change?", 901)
    u, ev = out["unified"], out["evidence"]
    img = demo.retriever.retrieve("Which earlier visit looks similar", 901, LAST)
    assert u.final_ranking == img.final_ranking and [c.name for c in u.components] == ["image_similarity",
                                                                                       "metadata_change"]
    anchor = img.final_ranking[0]
    a = demo.episodes.set_index("visit_id").loc[anchor, "cst"]
    assert f"({400 - a:+.0f} µm)" in ev["facts"][0] and anchor in ev["facts"][0]
    assert ev["retrieved"][0]["visit_id"] == anchor and "metadata_change #1" in ev["retrieved"][0]["sources"]
    assert [i["visit_id"] for i in ev["images"]] == [LAST] + [r["visit_id"] for r in ev["retrieved"]]
    text = f"{anchor} had CST {a:.0f} µm; today is 400 µm, a change of {abs(400 - a):.0f} µm.\nEvidence used: {anchor}"
    ok = demo.run("Which earlier visit looks similar, and how did CST change?", 901, backend="claude",
                  client=TextClient(text))
    assert ok["answer"]["status"] == "answered" and ok["answer"]["warnings"] == []


def test_mixed_with_metadata_clause_keeps_both_rankings(demo):
    u = demo.router.retrieve("Which earlier visit looks most like today's, and when was CST highest?", 901, LAST)
    meta = next(c for c in u.components if c.name == "metadata")
    assert meta.ranking[0] == "9901AOD_V1" and u.components[0].name == "image_similarity"
    ev = demo.run("Which earlier visit looks most like today's, and when was CST highest?", 901)["evidence"]
    assert any("metadata #1" in r["sources"] for r in ev["retrieved"])


def test_image_abstention_is_never_replaced(demo):
    del demo.retriever.emb[LAST]                                           # no embedding for today's scan
    for q in ("Which earlier OCT looks most similar?", "Which earlier visit looks similar, and how did CST change?"):
        out = demo.run(q, 901, backend="claude", client=NoCallClient())
        assert out["unified"].status == R.UNSUPPORTED and out["answer"]["status"] == "abstained"
    assert demo.run("When was CST highest?", 901)["unified"].status == "retrieved"   # metadata unaffected


@pytest.mark.parametrize("q", [q for q, _, s in ROUTING if s != "retrieved"])
def test_abstentions_never_reach_the_model_and_render(demo, q):
    out = demo.run(q, 901, backend="claude", client=NoCallClient())
    assert out["answer"]["status"] == "abstained" and out["evidence"]["guidance"]
    page = D.to_html(out)
    assert "Abstained: no answer was generated." in page and "How the question was routed" in page


@pytest.mark.parametrize("q", [q for q, _, s in ROUTING if s == "retrieved"])
def test_every_answered_category_has_consistent_evidence_and_request(demo, q):
    out = demo.run(q, 901)
    ev = out["evidence"]
    assert A.evidence_inconsistencies(ev) == []
    req = A.claude_request(ev)["messages"][0]["content"]
    labelled = [b["text"].split()[1] for b in req if b["type"] == "text" and b["text"].startswith("B-scan:")]
    assert labelled == [i["visit_id"] for i in ev["images"]]
    page = D.to_html(out)
    assert "Evidence source" in page and ev["category"] in page and "Not for clinical use." in page
