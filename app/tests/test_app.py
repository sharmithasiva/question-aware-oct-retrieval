"""Multimodal OCT assistant: routing -> retrieval -> evidence -> VLM, offline (fake VLM / HTTP; no key, network, GPU).

Synthetic eye 901: 8 visits, CST 500 450 300 405 380 420 390 400 (today = V8); finding labels only at V1 and V8.
"""
import io
import json
import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
import answer as A  # noqa: E402
from oct_retrieval import intent as I  # noqa: E402
from oct_retrieval import intent_retrieval as R  # noqa: E402

from app.assistant import Assistant  # noqa: E402
from app.config import Config, load_config, read_dotenv  # noqa: E402
from app.uploads import UploadError, validate_upload  # noqa: E402
from app.vlm import AnthropicVLM, OpenAICompatibleVLM, VLMReply  # noqa: E402

CST = [500, 450, 300, 405, 380, 420, 390, 400]
EYE, TODAY = 901, "9901AOD_V8"
FAKE_KEY = "sk-ant-api03-THISISNOTAREALKEY_0123456789"


def history(eye=901, subject="9901AOD", csts=CST):
    rows = []
    for i, c in enumerate(csts):
        lab = i in (0, len(csts) - 1)
        rows.append(dict(visit_id=f"{subject}_V{i + 1}", patient_id=eye, eye_id=eye, subject=subject, trial="TREX DME",
                         split="train", laterality="OD", visit_label=f"V{i + 1}", visit_number=i + 1, visit_index=i,
                         week=np.nan, n_visits_eye=len(csts), cst=float(c), bcva=70.0 + i, n_images=49,
                         has_biomarkers=lab, bm_n_scans=49.0 if lab else np.nan,
                         **{f"bm_{k}_{s}": (v if lab else np.nan) for s, f in (("fluid_irf", 0.4), ("fluid_srf", 0.0),
                                                                                 ("drt_me", 0.2))
                            for k, v in (("frac", f), ("any", float(f > 0)))}))
    return rows


class FakeVLM:
    def __init__(self, text="Answer.\nEvidence used: X", finish="stop"):
        self.calls, self.text, self.finish = [], text, finish

    def generate(self, system, parts):
        self.calls.append(dict(system=system, parts=parts))
        return VLMReply(text=self.text, finish=self.finish)


class NeverVLM:
    def generate(self, system, parts):
        raise AssertionError("the model must not be called")


@pytest.fixture
def parts(tmp_path):
    ep = pd.DataFrame(history() + history(902, "9902BOS", [300, 320, 350]))
    rng = np.random.default_rng(3)
    emb, images = {}, {}
    for i, v in enumerate(ep.visit_id):
        x = rng.normal(size=16)
        emb[v] = x / np.linalg.norm(x)
        images[v] = tmp_path / f"{v}.png"
        Image.fromarray((np.arange(64 * 64).reshape(64, 64) * (i + 1) % 251).astype(np.uint8)).save(images[v])
    return ep, emb, images


def make(parts, tmp_path, vlm=None, encoder=None):
    ep, emb, images = parts
    cfg = Config(workdir=tmp_path / "rt")
    return Assistant(cfg, vlm=vlm, upload_encoder=encoder, episodes=ep, visit_emb=emb, images=images,
                     emb_encoder="resnet50")


def png_bytes(arr) -> bytes:
    b = io.BytesIO()
    Image.fromarray(arr).save(b, "PNG")
    return b.getvalue()


def image_labels(call):
    return [p["text"] for p in call["parts"] if p["type"] == "text" and p["text"].startswith("B-scan:")]


def evidence_text_sent(call):
    return call["parts"][-1]["text"]


# --- the ten required scenarios --------------------------------------------------------------------------------------

def test_1_cst_highest_is_metadata_with_no_images(parts, tmp_path):
    vlm = FakeVLM()
    t = make(parts, tmp_path, vlm).ask("When was CST highest for this eye?", EYE, TODAY, backend="vlm")
    assert t.category == "metadata" and t.evidence["retrieved"][0]["visit_id"] == "9901AOD_V1"
    assert not any(p["type"] == "image" for p in vlm.calls[0]["parts"])
    assert "CST 500 µm" in evidence_text_sent(vlm.calls[0])


def test_2_largest_increase_phrasing_uses_the_frozen_rule(parts, tmp_path):
    a = make(parts, tmp_path)
    t = a.ask("Which visit had the largest increase in CST?", EYE, TODAY)
    assert t.evidence["interpreted_as"] == "What was the largest consecutive CST increase?"
    mem = a.router.hybrid.memory(EYE)
    frozen = R.retrieve(I.parse_intent(t.evidence["interpreted_as"]), mem, 7)
    assert t.unified.final_ranking == [mem.visit_ids[i] for i in frozen.ranked]


def test_3_worst_edema_discloses_cst_proxy_to_model_and_user(parts, tmp_path):
    vlm = FakeVLM()
    t = make(parts, tmp_path, vlm).ask("When was edema worst?", EYE, TODAY, backend="vlm")
    assert t.category == "edema_proxy" and "not a clinical edema grade" in evidence_text_sent(vlm.calls[0])
    assert "proxy" in vlm.calls[0]["system"] and t.evidence["images"]


def test_4_similar_oct_sends_exactly_the_evidence_scans(parts, tmp_path):
    vlm = FakeVLM()
    t = make(parts, tmp_path, vlm).ask("Which previous visit has an OCT scan most similar to this one?", EYE, TODAY,
                                       backend="vlm")
    assert t.category == "image_similarity" and t.evidence["route"] == "image+proxy_rrf"
    want = [TODAY] + [r["visit_id"] for r in t.evidence["retrieved"]]
    assert [lbl.split()[1] for lbl in image_labels(vlm.calls[0])] == want
    assert sum(p["type"] == "image" for p in vlm.calls[0]["parts"]) == len(want)


def test_5_best_anatomy_is_a_disclosed_cst_proxy_with_scans(parts, tmp_path):
    t = make(parts, tmp_path).ask("Which visit had the best anatomy?", EYE, TODAY)
    assert t.category == "anatomy_proxy" and t.evidence["retrieved"][0]["visit_id"] == "9901AOD_V3"   # lowest CST
    assert any("no anatomy grade" in d for d in t.evidence["disclosures"]) and t.evidence["images"]


def test_6_mixed_question_uses_image_and_metadata(parts, tmp_path):
    t = make(parts, tmp_path).ask("Which earlier visit looks similar, and how did CST change?", EYE, TODAY)
    assert [c["name"] for c in t.evidence["components"]] == ["image_similarity", "metadata_change"]
    assert t.evidence["facts"] and t.evidence["images"]


def test_7_irf_onset_with_incomplete_labels_abstains_without_model(parts, tmp_path):
    t = make(parts, tmp_path, NeverVLM()).ask("When did IRF first appear?", EYE, TODAY, backend="vlm")
    assert t.evidence["status"] == "insufficient_labels" and t.answer["status"] == "abstained"


def test_8_external_upload_is_described_without_any_history(parts, tmp_path):
    vlm = FakeVLM()
    a = make(parts, tmp_path, vlm)
    scan = validate_upload(png_bytes(np.tile(np.arange(200, dtype=np.uint8), (180, 1))), "ext.png", tmp_path / "up")
    t = a.ask("What abnormalities are visible in this OCT scan?", EYE, TODAY, upload=scan, backend="vlm")
    assert t.route == "scan_visual" and scan.external
    text = evidence_text_sent(vlm.calls[0])
    assert image_labels(vlm.calls[0]) == ["B-scan: UPLOADED_SCAN (uploaded by the user)"]
    assert "external image, not linked to any patient" in text
    assert "Eye:" not in text and "Current visit" not in text and "9901AOD" not in text
    assert t.evidence["biomarkers"] == []


def test_9_upload_similarity_ranks_this_eyes_visits_by_cosine(parts, tmp_path):
    ep, emb, _ = parts
    target = emb["9901AOD_V3"]
    a = make(parts, tmp_path, encoder=lambda x: (target + 0.01)[None])
    scan = validate_upload(png_bytes(np.full((120, 120), 90, np.uint8)), "u.png", tmp_path / "up")
    t = a.ask("Which previous visit has an OCT scan most similar to this one?", EYE, TODAY, upload=scan)
    ev = t.evidence
    assert t.route == "upload_similarity" and ev["retrieved"][0]["visit_id"] == "9901AOD_V3"
    assert set(ev["eligible"]) == {v for v in ep[ep.eye_id == EYE].visit_id}      # same eye only; no other eye
    assert ev["images"][0]["role"] == "uploaded" and A.evidence_inconsistencies(ev) == []
    assert any("does not show that the scan belongs to this eye" in d for d in ev["disclosures"])
    assert "UPLOADED_SCAN" in t.answer["answer"]


@pytest.mark.parametrize("exc,expect", [
    (urllib.error.HTTPError("u", 429, "Too Many Requests", {}, io.BytesIO(b"rate limit")), "rate limited"),
    (urllib.error.HTTPError("u", 401, "Unauthorized", {}, io.BytesIO(f"bad key {FAKE_KEY}".encode())), "authentication"),
    (urllib.error.URLError("connection refused"), "cannot reach"),
])
def test_10_api_failures_are_reported_not_answered(parts, tmp_path, exc, expect):
    def opener(req, timeout):
        raise exc
    vlm = OpenAICompatibleVLM("https://example.invalid/v1", "m", api_key=FAKE_KEY, opener=opener)
    t = make(parts, tmp_path, vlm).ask("When was CST highest for this eye?", EYE, TODAY, backend="vlm")
    a = t.answer
    assert a["status"] == "model_error" and expect in a["answer"] and "no answer was substituted" in a["answer"]
    assert FAKE_KEY not in json.dumps(a) and "THISISNOTAREALKEY" not in repr(vlm)
    assert t.evidence["retrieved"]                                        # evidence still shown


# --- more guarantees -------------------------------------------------------------------------------------------------

def test_vlm_receives_only_the_evidence_package(parts, tmp_path):
    vlm = FakeVLM()
    t = make(parts, tmp_path, vlm).ask("Which visit had the best anatomy?", EYE, TODAY, backend="vlm")
    assert vlm.calls[0]["parts"] == A.request_parts(t.evidence) and vlm.calls[0]["system"] == A.SYSTEM


def test_grounding_flags_invented_visits(parts, tmp_path):
    vlm = FakeVLM("CST peaked at 9901AOD_V5 with 999 µm.\nEvidence used: 9901AOD_V5")
    t = make(parts, tmp_path, vlm).ask("When was CST highest for this eye?", EYE, TODAY, backend="vlm")
    w = " ".join(t.answer["warnings"])
    assert "9901AOD_V5" in w and "999" in w


def test_matched_upload_is_answered_from_its_dataset_visit(tmp_path):
    cfg = load_config(env={}, dotenv=None)
    cfg.workdir = tmp_path / "rt"
    a = Assistant(cfg, upload_encoder=lambda x: np.ones((1, 4)))
    v = "0210MOD_V20"
    gray = np.asarray(Image.open(a.scan_path(v)).convert("L"))         # dataset WebP saved losslessly as PNG
    scan = a.match_upload(validate_upload(png_bytes(gray), "copy.png", tmp_path / "up"))
    assert not scan.external and scan.matched_visit == v and scan.matched_eye == 25
    t = a.ask("Which previous visit has an OCT scan most similar to this one?", 11, "0232GOD_V25", upload=scan)
    assert t.eye == 25 and t.visit == v and t.category == "image_similarity" and "identical" in t.notes[0]
    assert all(r["visit_index"] < 19 for r in t.evidence["retrieved"])  # only visits before V20 of eye 25


def test_followup_explains_previous_turn_without_new_retrieval(parts, tmp_path):
    a = make(parts, tmp_path, NeverVLM())
    t1 = a.ask("When was edema worst?", EYE, TODAY)
    t2 = a.ask("What evidence supports your answer?", EYE, TODAY, previous=t1, backend="vlm")
    assert t2.route == "followup" and "proxy" in t2.answer["answer"] and t2.evidence is t1.evidence
    t3 = a.ask("What evidence supports your answer?", EYE, TODAY, backend="vlm")
    assert t3.answer["status"] == "abstained"


def test_current_visit_scan_question_without_upload(parts, tmp_path):
    vlm = FakeVLM()
    t = make(parts, tmp_path, vlm).ask("Does this scan show signs of IRF or SRF?", EYE, TODAY, backend="vlm")
    assert t.route == "scan_visual" and image_labels(vlm.calls[0]) == [f"B-scan: {TODAY} (current visit)"]
    assert t.evidence["biomarkers"][0]["visit_id"] == TODAY                # V8 is labelled
    t2 = make(parts, tmp_path, FakeVLM()).ask("Does this scan show signs of IRF or SRF?", EYE, "9901AOD_V5")
    assert t2.evidence["biomarkers"] == []                                 # unlabelled visit: no labels claimed


def test_upload_not_used_for_history_questions(parts, tmp_path):
    scan = validate_upload(png_bytes(np.full((100, 100), 50, np.uint8)), "u.png", tmp_path / "up")
    t = make(parts, tmp_path).ask("When was CST highest for this eye?", EYE, TODAY, upload=scan)
    assert t.route == "history" and "not used" in t.notes[0] and not t.evidence.get("upload")


# --- adapters, uploads, config ---------------------------------------------------------------------------------------

def test_openai_compatible_payload_and_reply(tmp_path):
    seen = {}

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def opener(req, timeout):
        seen["url"], seen["headers"], seen["body"] = req.full_url, dict(req.header_items()), json.loads(req.data)
        return Resp(json.dumps({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}).encode())
    img = tmp_path / "i.png"
    Image.fromarray(np.zeros((70, 70), np.uint8)).save(img)
    vlm = OpenAICompatibleVLM("https://openrouter.ai/api/v1/", "qwen/qwen2.5-vl-72b-instruct:free", "k", opener=opener)
    r = vlm.generate("sys", [dict(type="text", text="label"), dict(type="image", path=str(img))])
    assert r.text == "ok" and r.finish == "stop" and seen["url"].endswith("/api/v1/chat/completions")
    body = seen["body"]
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert body["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert seen["headers"]["Authorization"] == "Bearer k"


def test_anthropic_adapter_maps_stop_reasons():
    reply = SimpleNamespace(stop_reason="max_tokens", content=[SimpleNamespace(type="text", text="partial")])
    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: reply)))
    r = AnthropicVLM("claude-opus-5-5", client=client).generate("s", [dict(type="text", text="t")])
    assert (r.text, r.finish) == ("partial", "length")


@pytest.mark.parametrize("data,msg", [(b"", "empty"), (b"not an image", "not a readable image"),
                                      (png_bytes(np.zeros((20, 20), np.uint8)), "sides must be")])
def test_upload_validation_rejects(tmp_path, data, msg):
    with pytest.raises(UploadError, match=msg):
        validate_upload(data, "x.png", tmp_path)


def test_upload_validation_formats_and_warnings(tmp_path):
    webp = io.BytesIO()
    Image.fromarray(np.zeros((100, 100), np.uint8)).save(webp, "WEBP")
    with pytest.raises(UploadError, match="unsupported format"):
        validate_upload(webp.getvalue(), "x.webp", tmp_path)
    rgb = np.zeros((100, 100, 3), np.uint8)
    rgb[..., 0] = 200
    s = validate_upload(png_bytes(rgb), "c.png", tmp_path)
    assert s.external and any("colour" in w for w in s.warnings) and s.path.exists()
    with pytest.raises(UploadError, match="larger than"):
        validate_upload(png_bytes(np.zeros((200, 200), np.uint8)), "x.png", tmp_path, max_mb=1e-4)


def test_config_reads_dotenv_and_hides_key(tmp_path):
    (tmp_path / ".env").write_text("OCT_VLM_PROVIDER=openai_compatible\nOPENROUTER_API_KEY='secret-123'\n")
    assert read_dotenv(tmp_path / ".env")["OPENROUTER_API_KEY"] == "secret-123"
    c = load_config(env={"OCT_VLM_MODEL": "m"}, dotenv=tmp_path / ".env")
    assert c.provider == "openai_compatible" and c.api_key == "secret-123" and c.model == "m"
    assert "secret-123" not in repr(c) and "api_key=set" in repr(c)


# --- API and UI (real repository data, evidence-only) -----------------------------------------------------------------

def test_api_endpoints(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import app.api as API
    monkeypatch.setenv("OCT_VLM_PROVIDER", "evidence")
    API.get_assistant.cache_clear()
    c = TestClient(API.app)
    h = c.get("/health").json()
    assert h["status"] == "ok" and "key" not in json.dumps(h).replace("api_key", "")
    assert any(e["eye_id"] == 25 for e in c.get("/eyes").json())
    assert c.get("/eyes/25/visits").json()[0]["visit_id"] == "0210MOD_V1"
    assert c.get("/scans/0210MOD_V20").status_code == 200
    r = c.post("/ask", data=dict(question="When was edema worst?", eye=25, visit="0210MOD_V27")).json()
    assert r["category"] == "edema_proxy" and r["evidence"]["disclosures"]
    assert all("/" not in im["path"] for im in r["evidence"]["images"])         # no local paths leave the server
    up = c.post("/ask", data=dict(question="What abnormalities are visible in this scan?"),
                files={"file": ("x.png", png_bytes(np.full((120, 120), 80, np.uint8)), "image/png")}).json()
    assert up["route"] == "scan_visual" and up["upload"]["external"] is True
    bad = c.post("/ask", data=dict(question="hi"), files={"file": ("x.png", b"nope", "image/png")})
    assert bad.status_code == 400 and "upload rejected" in bad.json()["detail"]
    API.get_assistant.cache_clear()


def test_streamlit_app_runs_end_to_end(monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("OCT_VLM_PROVIDER", "evidence")
    at = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=120)
    at.run()
    assert not at.exception
    for q in ("When was edema worst?", "When did IRF first appear?", "What abnormalities are visible in this OCT scan?",
              "What evidence supports your answer?"):
        at.chat_input[0].set_value(q).run()
        assert not at.exception, (q, at.exception)
    assert len(at.chat_message) == 8
