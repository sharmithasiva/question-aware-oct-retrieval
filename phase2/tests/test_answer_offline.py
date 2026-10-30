"""Answer-generation layer on a SYNTHETIC eye history: no repository data, API key, network, scan subset or GPU.

Eye 901 (8 visits): today's CST is 400; earlier CSTs are chosen so that "CST above today's" qualifies exactly
visits V1, V2, V4 and V6, the most recent of which is V6. Visit V7's embedding is closest to V5's.
"""
import base64
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
from oct_retrieval import intent_retrieval as R  # noqa: E402

FAKE_KEY = "sk-ant-api03-THISISNOTAREALKEY_0123456789"
CST = [500, 450, 300, 405, 380, 420, 390, 400]          # V1..V8; V8 is today
IMAGE_Q = "Which earlier visit's OCT looks most like today's?"
CST_ABOVE_Q = "Most recent prior visit with CST above today's value?"
CST_SIMILAR_Q = "When was the CST most similar to today's measurement?"


def visits(eye, subject, csts):
    return [dict(visit_id=f"{subject}_V{i + 1}", patient_id=eye, eye_id=eye, subject=subject, trial="TREX DME",
                 split="train", laterality="OD", visit_label=f"V{i + 1}", visit_number=i + 1, visit_index=i,
                 week=np.nan, n_visits_eye=len(csts), cst=float(c), bcva=70.0 + i, n_images=49)
            for i, c in enumerate(csts)]


@pytest.fixture
def parts(tmp_path):
    ep = pd.DataFrame(visits(901, "9901AOD", CST) + visits(902, "9902BOS", [300, 320, 350]))
    rng = np.random.default_rng(0)
    emb, images = {}, {}
    for i, vid in enumerate(ep.visit_id):
        v = rng.normal(size=16)
        emb[vid] = v / np.linalg.norm(v)
        images[vid] = tmp_path / f"{vid}.png"
        Image.fromarray(np.full((8, 8), i * 10, np.uint8)).save(images[vid])
    near = emb["9901AOD_V5"] + 0.05 * emb["9901AOD_V8"]
    emb["9901AOD_V8"] = near / np.linalg.norm(near)
    return ep, emb, images


@pytest.fixture
def demo(parts):
    ep, emb, images = parts
    return D.Demo.from_parts(ep, emb, images)


class FakeClient:
    """Stands in for anthropic.Anthropic(); records calls. `raise_exc` simulates an API failure."""

    def __init__(self, text="", stop="end_turn", raise_exc=None):
        self.calls, self.text, self.stop, self.raise_exc = [], text, stop, raise_exc
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        if self.raise_exc:
            raise self.raise_exc
        blocks = [SimpleNamespace(type="thinking", thinking="")]
        if self.text:
            blocks.append(SimpleNamespace(type="text", text=self.text))
        return SimpleNamespace(stop_reason=self.stop, content=blocks)


class NoCallClient(FakeClient):
    def _create(self, **kw):
        raise AssertionError("the model must not be called")


# --- retrieval -> evidence ------------------------------------------------------------------------------------

def test_image_question_uses_image_route_with_labelled_scans_and_basis(demo):
    out = demo.run(IMAGE_Q, 901)
    ev = out["evidence"]
    assert ev["route"] == "image+proxy_rrf" and ev["status"] == "retrieved"
    assert "does not detect fluid" in ev["ranking_basis"]
    assert [i["role"] for i in ev["images"]] == ["current", "retrieved", "retrieved", "retrieved"]
    assert ev["images"][0]["visit_id"] == "9901AOD_V8"
    assert {v["visit_id"] for v in ev["retrieved"]} <= set(out["result"].candidates)
    req = A.claude_request(ev)
    imgs = [b for b in req["messages"][0]["content"] if b["type"] == "image"]
    assert len(imgs) == 4 and all(b["source"]["media_type"] == "image/png" for b in imgs)
    assert base64.b64decode(imgs[0]["source"]["data"])[:4] == b"\x89PNG"


def test_hard_rule_question_passes_only_the_qualifying_selected_visit(demo):
    ev = demo.run(CST_ABOVE_Q, 901)["evidence"]
    assert ev["route"] == "metadata" and ev["operation"] == "compare_to_current"
    assert [(v["visit_id"], v["role"]) for v in ev["retrieved"]] == [("9901AOD_V6", "selected")]
    assert ev["retrieved"][0]["cst"] > ev["current"]["cst"]
    assert ev["images"] == [] and all(b["type"] == "text" for b in A.claude_request(ev)["messages"][0]["content"])


def test_similarity_metadata_question_keeps_runners_up_labelled(demo):
    ev = demo.run(CST_SIMILAR_Q, 901)["evidence"]
    assert ev["route"] == "metadata" and [v["role"] for v in ev["retrieved"]] == ["selected", "runner-up", "runner-up"]
    assert ev["retrieved"][0]["visit_id"] == "9901AOD_V4"          # |405 - 400| is the smallest difference
    assert "OCT images were not used" in ev["ranking_basis"] and ev["images"] == []


@pytest.mark.parametrize("q", [CST_ABOVE_Q, CST_SIMILAR_Q, "Compare with the previous visit.",
                               "How has the retina changed since baseline?"])
def test_images_never_reorder_metadata_rankings(demo, q):
    """Even with an embedding for every visit, metadata questions return the frozen metadata ranking unchanged."""
    res = demo.run(q, 901)["result"]
    mem = demo.retriever.memory(901)
    frozen = R.retrieve(R.I.parse_intent(q), mem, 7)
    assert res.route == "metadata" and res.final_ranking == [mem.visit_ids[i] for i in frozen.ranked]
    assert "9901AOD_V8" not in res.final_ranking and not any(v.startswith("9902") for v in res.final_ranking)


# --- abstention ---------------------------------------------------------------------------------------------------

def _no_image_for_today(demo):
    del demo.retriever.emb["9901AOD_V8"]
    return demo


@pytest.mark.parametrize("q,eye,visit,status,setup", [
    ("hello there", 901, None, "clarification_needed", None),
    ("When was the last injection before today?", 901, None, "unavailable_variable", None),
    (CST_ABOVE_Q, 902, None, "no_qualifying_visit", None),                  # today's CST is the highest
    (CST_SIMILAR_Q, 901, "9901AOD_V1", "insufficient_data", None),         # first visit: no history
    (IMAGE_Q, 901, None, "unsupported_image", _no_image_for_today),
])
def test_abstentions_never_call_the_model_and_explain(demo, q, eye, visit, status, setup):
    if setup:
        setup(demo)
    out = demo.run(q, eye, visit, backend="claude", client=NoCallClient())
    a, ev = out["answer"], out["evidence"]
    assert ev["status"] == status and a["status"] == "abstained" and a["backend"] == "none"
    assert a["answer"].startswith("No answer") and ev["retrieved"] == [] and ev["images"] == []
    assert ev["guidance"] and ev["guidance"] in a["answer"] and a["disclaimer"] == A.DISCLAIMER
    page = D.to_html(out)
    assert "Abstained: no answer was generated." in page and "Not for clinical use." in page


# --- model output and failures --------------------------------------------------------------------------------------

def test_grounded_model_answer_has_no_warnings(demo):
    text = ("The selected visit 9901AOD_V6 had a CST of 420 µm, 20 µm above today's 400 µm.\n"
            "Evidence used: 9901AOD_V6, 9901AOD_V8")
    c = FakeClient(text)
    a = demo.run(CST_ABOVE_Q, 901, backend="claude", client=c)["answer"]
    assert a["status"] == "answered" and a["warnings"] == [] and a["answer"] == text and len(c.calls) == 1
    assert c.calls[0]["model"] == A.MODEL and "diagnosis" in c.calls[0]["system"]


def test_ungrounded_model_answer_is_flagged_not_hidden(demo):
    text = "Visit 9901AOD_V3 had CST 512 µm, so fluid has worsened."           # V3 not passed; 512 not in evidence
    a = demo.run(CST_ABOVE_Q, 901, backend="claude", client=FakeClient(text))["answer"]
    assert a["status"] == "answered" and a["answer"] == text
    w = " | ".join(a["warnings"])
    assert "9901AOD_V3" in w and "512" in w and "Evidence used" in w
    page = D.to_html(demo.run(CST_ABOVE_Q, 901, backend="claude", client=FakeClient(text)))
    assert "Check against the evidence" in page


@pytest.mark.parametrize("client,status", [
    (FakeClient(raise_exc=RuntimeError(f"401 invalid x-api-key: {FAKE_KEY}")), "model_error"),
    (FakeClient(raise_exc=ConnectionError("network unreachable")), "model_error"),
    (FakeClient(text="I can't help with that.", stop="refusal"), "model_refused"),
    (FakeClient(text=""), "model_error"),
])
def test_model_failures_are_reported_without_substitute_answer_or_key(demo, client, status):
    out = demo.run(IMAGE_Q, 901, backend="claude", client=client)
    a = out["answer"]
    assert a["status"] == status and a["answer"].startswith("No generated answer") and "no answer was substituted" in a["answer"]
    for s in (a["answer"], a["error"] or "", D.to_html(out), D.to_text(out)):
        assert FAKE_KEY not in s and "THISISNOTAREALKEY" not in s
    assert out["evidence"]["retrieved"]                                    # evidence still shown, unchanged


def test_truncated_answer_is_marked_incomplete(demo):
    a = demo.run(CST_ABOVE_Q, 901, backend="claude", client=FakeClient("9901AOD_V6 had", stop="max_tokens"))["answer"]
    assert a["status"] == "model_incomplete" and "incomplete" in a["warnings"][0]


def test_missing_sdk_is_reported(demo, monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)                  # makes `import anthropic` fail
    a = demo.run(IMAGE_Q, 901, backend="claude")["answer"]
    assert a["status"] == "model_error" and "not installed" in a["answer"]


def test_redact_and_evidence_only_wording():
    assert "[redacted]" in A.redact(f"bad key {FAKE_KEY}") and FAKE_KEY not in A.redact(FAKE_KEY)
    assert "THISISNOT" not in A.redact("Authorization: Bearer THISISNOTAREALTOKEN")
    assert len(A.redact("x" * 1000)) == 300


def test_evidence_only_backend_and_demo_outputs(demo):
    out = demo.run(IMAGE_Q, 901)
    a = out["answer"]
    assert a["status"] == "answered" and a["backend"] == "evidence" and "fused with CST closeness" in a["answer"]
    page, text = D.to_html(out), D.to_text(out)
    for s in ("Not for clinical use.", A.DISCLAIMER, "Current visit", "Selected evidence", "image+proxy_rrf", "does not detect fluid",
              "data:image/png;base64,"):
        assert s in page
    assert text.startswith("NOTE: " + A.DISCLAIMER)
    with pytest.raises(ValueError):
        demo.run(IMAGE_Q, 902, "9901AOD_V8")                               # visit of another eye
    with pytest.raises(ValueError):
        demo.run(IMAGE_Q, 999)
    with pytest.raises(ValueError):
        demo.run("  ", 901)


# --- evidence / image-payload consistency (regression) ----------------------------------------------------------

def test_text_evidence_image_payload_and_summary_agree(demo):
    ev = demo.run(IMAGE_Q, 901)["evidence"]
    rows = [v["visit_id"] for v in ev["retrieved"]]
    req = A.claude_request(ev)["messages"][0]["content"]
    labelled = [b["text"].split()[1] for b in req if b["type"] == "text" and b["text"].startswith("B-scan:")]
    assert labelled == [i["visit_id"] for i in ev["images"]] == [ev["current"]["visit_id"]] + rows
    assert sum(b["type"] == "image" for b in req) == len(labelled)
    assert A.evidence_inconsistencies(ev) == []


def test_missing_image_is_shown_not_silently_dropped(demo):
    ranked = demo.run(IMAGE_Q, 901)["evidence"]["retrieved"]
    missing = ranked[1]["visit_id"]
    del demo.images[missing]                                   # the embedding exists, the image file does not
    out = demo.run(IMAGE_Q, 901)
    ev, a = out["evidence"], out["answer"]
    assert [v["visit_id"] for v in ev["retrieved"]] == [v["visit_id"] for v in ranked]     # ranking unchanged
    assert missing not in [i["visit_id"] for i in ev["images"]] and A.evidence_inconsistencies(ev) == []
    assert f"no B-scan available for {missing}" in a["evidence_used"]
    assert "no B-scan available" in A.evidence_text(ev) and "<td>no</td>" in D.to_html(out)


def test_inconsistency_check_catches_the_reported_class_of_mismatch(demo):
    ev = demo.run(IMAGE_Q, 901)["evidence"]
    bad = dict(ev, images=ev["images"][:-1])                   # last listed visit's scan not sent (the old bug)
    assert A.evidence_inconsistencies(bad)
    meta = demo.run(CST_SIMILAR_Q, 901)["evidence"]
    assert meta["images"] == [] and not any(v["image_attached"] for v in [meta["current"]] + meta["retrieved"])
    assert "no B-scans (this question type does not use images)" in A.evidence_summary(meta)
    assert A.evidence_inconsistencies(dict(meta, images=ev["images"]))   # images on a metadata route are flagged
