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
from app.vlm import AnthropicVLM, OllamaVLM, OpenAICompatibleVLM, VLMReply, make_vlm, prepare_image  # noqa: E402

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

def test_1_cst_highest_is_answered_by_the_frozen_rule_without_the_model(parts, tmp_path):
    t = make(parts, tmp_path, NeverVLM()).ask("When was CST highest for this eye?", EYE, TODAY, backend="vlm")
    assert t.category == "metadata" and t.evidence["retrieved"][0]["visit_id"] == "9901AOD_V1"
    a = t.answer
    assert a["status"] == "answered" and a["backend"] == "deterministic" and a["images_sent"] == []
    assert "9901AOD_V1" in a["answer"] and "CST 500 µm" in a["answer"] and "no vision-language model" in a["answer"]


def test_2_largest_increase_phrasing_uses_the_frozen_rule(parts, tmp_path):
    a = make(parts, tmp_path)
    t = a.ask("Which visit had the largest increase in CST?", EYE, TODAY)
    assert t.evidence["interpreted_as"] == "What was the largest consecutive CST increase?"
    mem = a.router.hybrid.memory(EYE)
    frozen = R.retrieve(I.parse_intent(t.evidence["interpreted_as"]), mem, 7)
    assert t.unified.final_ranking == [mem.visit_ids[i] for i in frozen.ranked]


def test_3_worst_edema_discloses_cst_proxy_without_calling_the_model(parts, tmp_path):
    t = make(parts, tmp_path, NeverVLM()).ask("When was edema worst?", EYE, TODAY, backend="vlm")
    assert t.category == "edema_proxy" and t.answer["backend"] == "deterministic"
    assert t.evidence["retrieved"][0]["visit_id"] == "9901AOD_V1"                     # highest CST (frozen rule)
    assert "proxy for edema (not a clinical edema grade)" in t.answer["answer"]
    assert any("not a clinical edema grade" in d for d in t.evidence["disclosures"])
    assert t.evidence["images"]                                                       # scans still shown in the panel


def test_4_similar_oct_ranking_is_deterministic_and_a_visual_comparison_sends_two_scans(parts, tmp_path):
    a = make(parts, tmp_path, NeverVLM())
    t = a.ask("Which previous visit has an OCT scan most similar to this one?", EYE, TODAY, backend="vlm")
    assert t.category == "image_similarity" and t.evidence["route"] == "image+proxy_rrf"
    assert t.answer["backend"] == "deterministic" and t.evidence["retrieved"][0]["visit_id"] in t.answer["answer"]
    vlm = FakeVLM()
    t2 = make(parts, tmp_path, vlm).ask("Which previous visit has an OCT scan most similar to this one, and how do "
                                        "they differ?", EYE, TODAY, backend="vlm")
    top = t2.evidence["retrieved"][0]["visit_id"]
    assert t2.unified.final_ranking == t.unified.final_ranking                       # ranking unchanged
    assert [lbl.split()[1] for lbl in image_labels(vlm.calls[0])] == [TODAY, top]
    assert sum(p["type"] == "image" for p in vlm.calls[0]["parts"]) == 2 and t2.answer["images_sent"] == [TODAY, top]
    assert "B-scan not sent" in evidence_text_sent(vlm.calls[0])                       # runner-ups: rows kept, no image


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
    t = make(parts, tmp_path, vlm).ask("Compare the current scan with the previous visit.", EYE, TODAY, backend="vlm")
    a = t.answer
    assert a["status"] == "model_error" and expect in a["answer"] and "no model answer was substituted" in a["answer"]
    assert "Retrieved evidence without a model" in a["answer"] and "9901AOD_V7" in a["answer"]
    assert FAKE_KEY not in json.dumps(a) and "THISISNOTAREALKEY" not in repr(vlm)
    assert t.evidence["retrieved"]                                        # evidence still shown


# --- more guarantees -------------------------------------------------------------------------------------------------

def test_vlm_receives_only_the_evidence_package(parts, tmp_path):
    vlm = FakeVLM()
    t = make(parts, tmp_path, vlm).ask("Compare the current scan with the previous visit.", EYE, TODAY, backend="vlm")
    assert vlm.calls[0]["parts"] == A.request_parts(A.vision_view(t.evidence, 3))
    assert vlm.calls[0]["system"] == A.SYSTEM


def test_grounding_flags_invented_visits(parts, tmp_path):
    vlm = FakeVLM("CST peaked at 9901AOD_V5 with 999 µm.\nEvidence used: 9901AOD_V5")
    t = make(parts, tmp_path, vlm).ask("Compare the current scan with the previous visit.", EYE, TODAY, backend="vlm")
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


# --- metadata-only questions never reach a model; visual questions send only the scans they need ----------------------

class NeverClaude:
    beta = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(
        AssertionError("Claude must not be called"))))


@pytest.mark.parametrize("q,category", [
    ("When was CST highest for this eye?", "metadata"),
    ("Which visit had the largest increase in CST?", "metadata"),
    ("When was edema worst?", "edema_proxy"),
    ("Which visit had the best anatomy?", "anatomy_proxy"),
    ("When was the CST most similar to today's measurement?", "metadata"),
    ("Which previous visit has an OCT scan most similar to this one?", "image_similarity"),
    ("Which earlier visit looks similar, and how did CST change?", "mixed"),
])
@pytest.mark.parametrize("backend", ["vlm", "claude"])
def test_metadata_only_questions_do_not_invoke_the_model(parts, tmp_path, q, category, backend):
    a = make(parts, tmp_path, NeverVLM())
    t = a.ask(q, EYE, TODAY, backend=backend, client=NeverClaude())
    ref = make(parts, tmp_path).ask(q, EYE, TODAY)                                  # evidence-only reference
    assert t.category == category and t.answer["status"] == "answered" and t.answer["backend"] == "deterministic"
    assert t.unified.final_ranking == ref.unified.final_ranking                      # retrieval unchanged
    assert t.answer["answer"].split("(")[0] == ref.answer["answer"].split("(")[0]     # same frozen answer
    assert "Evidence used:" in t.answer["answer"] and t.answer["images_sent"] == []


@pytest.mark.parametrize("q,visits", [
    ("Does this scan show signs of IRF or SRF?", [TODAY]),
    ("What abnormalities are visible in this OCT scan?", [TODAY]),
    ("Compare the current scan with the previous visit.", [TODAY, "9901AOD_V7"]),
    ("How has the retina changed over time?", [TODAY, "9901AOD_V1", "9901AOD_V7"]),
])
def test_visual_questions_send_the_required_scans(parts, tmp_path, q, visits):
    vlm = FakeVLM()
    t = make(parts, tmp_path, vlm).ask(q, EYE, TODAY, backend="vlm")
    assert len(vlm.calls) == 1 and t.answer["backend"] == "vlm"
    assert [lbl.split()[1] for lbl in image_labels(vlm.calls[0])] == visits
    assert sum(p["type"] == "image" for p in vlm.calls[0]["parts"]) == len(visits) == len(t.answer["images_sent"])


def test_max_images_and_duplicate_scans_are_not_sent(parts, tmp_path):
    ep, emb, images = parts
    images["9901AOD_V1"].write_bytes(images[TODAY].read_bytes())                  # baseline scan file == today's
    vlm = FakeVLM()
    t = make(parts, tmp_path, vlm).ask("How has the retina changed over time?", EYE, TODAY, backend="vlm")
    assert [lbl.split()[1] for lbl in image_labels(vlm.calls[0])] == [TODAY, "9901AOD_V7"]
    vlm2 = FakeVLM()
    a = make(parts, tmp_path, vlm2)
    a.cfg.max_images = 1
    a.ask("Compare the current scan with the previous visit.", EYE, TODAY, backend="vlm")
    assert [lbl.split()[1] for lbl in image_labels(vlm2.calls[0])] == [TODAY]
    assert "Not sent to the model: the B-scans of 9901AOD_V7" in evidence_text_sent(vlm2.calls[0])


class Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def big_scans(parts):
    """Overwrite the fixture scans with distinct 504x496 images, the size of the dataset's atlas B-scans."""
    rng = np.random.default_rng(0)
    for p in parts[2].values():
        Image.fromarray(rng.integers(0, 255, (496, 504), dtype=np.uint8)).save(p)


def test_ollama_request_fits_the_context_budget_and_sets_num_ctx(parts, tmp_path):
    big_scans(parts)
    seen = []

    def opener(req, timeout):
        seen.append((req.full_url, json.loads(req.data)))
        return Resp(json.dumps({"message": {"content": "Answer.\nEvidence used: 9901AOD_V8"},
                                "done_reason": "stop", "prompt_eval_count": 1}).encode())
    vlm = make_vlm("ollama", "qwen2.5vl:3b", "http://localhost:11434/v1", num_ctx=4096, max_tokens=1500)
    assert isinstance(vlm, OllamaVLM) and vlm.context_tokens == 4096
    vlm._open = opener
    got, generate = [], vlm.generate
    vlm.generate = lambda system, ps: got.append(ps) or generate(system, ps)
    t = make(parts, tmp_path, vlm).ask("How has the retina changed over time?", EYE, TODAY, backend="vlm")
    url, body = seen[0]
    assert url == "http://localhost:11434/api/chat" and body["options"]["num_ctx"] == 4096
    user = body["messages"][1]
    assert t.answer["status"] == "answered" and len(user["images"]) == len(t.answer["images_sent"]) >= 2
    assert "Image 1: B-scan: 9901AOD_V8 (current visit)" in user["content"]
    est = vlm.estimate_tokens(A.SYSTEM, got[0])                                   # the parts actually sent
    assert body["options"]["num_predict"] >= A.MIN_OUTPUT_TOKENS
    assert est + body["options"]["num_predict"] <= 4096                             # prompt + answer within budget
    for b64 in user["images"]:
        w, h = Image.open(io.BytesIO(__import__("base64").b64decode(b64))).size
        assert max(w, h) <= 448 and w % 28 == 0 and h % 28 == 0


def test_request_over_budget_is_not_sent_and_evidence_is_kept(parts, tmp_path):
    big_scans(parts)
    vlm = make_vlm("ollama", "qwen2.5vl:3b", "http://localhost:11434", context_tokens=1200)
    vlm._open = lambda req, timeout: (_ for _ in ()).throw(AssertionError("must not be sent"))
    t = make(parts, tmp_path, vlm).ask("Compare the current scan with the previous visit.", EYE, TODAY, backend="vlm")
    a = t.answer
    assert a["status"] == "model_error" and a["error"] == "context budget exceeded"
    assert "the model was not called" in a["answer"] and "1200 tokens" in a["answer"]
    assert "OCT_VLM_NUM_CTX" in a["answer"] and "Retrieved evidence without a model" in a["answer"]
    assert t.evidence["retrieved"][0]["visit_id"] == "9901AOD_V7" and t.evidence["images"]


def test_ollama_context_error_is_explained(parts, tmp_path):
    body = b'{"error":"the input length exceeds the context length"}'
    vlm = make_vlm("openai_compatible", "qwen2.5vl:3b", "http://localhost:11434/v1", context_tokens=0)
    vlm._open = lambda req, timeout: (_ for _ in ()).throw(
        urllib.error.HTTPError("u", 400, "Bad Request", {}, io.BytesIO(body)))
    t = make(parts, tmp_path, vlm).ask("Does this scan show signs of IRF or SRF?", EYE, TODAY, backend="vlm")
    assert t.answer["status"] == "model_error" and "context window" in t.answer["answer"]
    assert "OCT_VLM_MAX_IMAGES" in t.answer["answer"]


def test_image_preparation_downscales_without_upscaling(tmp_path):
    p = tmp_path / "s.png"
    Image.fromarray(np.zeros((496, 504), np.uint8)).save(p)
    _, w, h = prepare_image(p, 448)
    assert (w, h) == (420, 420) or (max(w, h) <= 448 and w % 28 == 0 and h % 28 == 0)
    assert prepare_image(p, 1024)[1:] == (504, 496)
    assert prepare_image(p, None, "jpeg")[0][:2] == b"\xff\xd8"


def test_config_reads_context_budget_settings():
    c = load_config(env={"OCT_VLM_PROVIDER": "ollama", "OCT_VLM_NUM_CTX": "8192", "OCT_VLM_MAX_IMAGES": "2",
                         "OCT_VLM_IMAGE_MAX_SIDE": "392"}, dotenv=None)
    assert (c.num_ctx, c.max_images, c.image_max_side, c.context_tokens) == (8192, 2, 392, None)
    vlm = make_vlm(c.provider, "m", "", **c.vlm_options())
    assert vlm.context_tokens == 8192 and vlm.num_ctx == 8192 and vlm.base_url == "http://localhost:11434"
    assert make_vlm("openai_compatible", "m", "https://openrouter.ai/api/v1").context_tokens == 0


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
