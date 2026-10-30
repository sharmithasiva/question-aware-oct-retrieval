"""Multimodal OCT assistant: routing -> retrieval -> evidence plan -> VLM answer, offline (fake VLM / HTTP; no key, network, GPU).

Synthetic eye 901: 8 visits, CST 500 450 300 405 380 420 390 400 (today = V8); finding labels only at V1 and V8.
"""
import io
import json
import os
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
import oct_image as OI  # noqa: E402
from oct_retrieval import intent as I  # noqa: E402
from oct_retrieval import intent_retrieval as R  # noqa: E402

from app import reasoning as RS  # noqa: E402
from app.assistant import Assistant  # noqa: E402
from app.config import Config, load_config, read_dotenv  # noqa: E402
from app.uploads import ScanIndex, UploadError, embed_scan, validate_upload  # noqa: E402
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


def sent_visits(call):
    """Visit labels of the B-scans in a model call, in order (UPLOAD for the uploaded scan)."""
    return ["UPLOAD" if lbl.startswith("B-scan: the uploaded scan") else lbl.rsplit("visit ", 1)[1].split(" (")[0]
            for lbl in image_labels(call)]


def prompt_of(call):
    return call["parts"][-1]["text"]


MODEL_TEXT = ("On the V1 scan there appear to be more dark intraretinal spaces than today; this is my visual "
              "reading of single downscaled B-scans and may be uncertain. The OLIVES labels exist only for V1 and V8.")


# --- the model writes the answer from the original question and structured evidence -------------------------------

def test_numeric_question_goes_to_the_model_as_text_with_the_exact_rule_result(parts, tmp_path):
    vlm = FakeVLM("CST was highest at V1 (500 µm), 100 µm above today's 400 µm.")
    t = make(parts, tmp_path, vlm).ask("When was CST highest for this eye?", EYE, TODAY, backend="vlm")
    call = vlm.calls[0]
    assert t.category == "metadata" and t.answer["plan"]["kind"] == "numeric" and sent_visits(call) == []
    assert prompt_of(call).startswith("QUESTION (as asked): When was CST highest for this eye?")
    assert "Result of the retrieval rule for this question: visit V1 (CST 500 µm" in prompt_of(call)
    assert "difference of -100 µm" in prompt_of(call)                          # exact arithmetic, done by the app
    assert t.answer["answer"] == vlm.text and t.answer["warnings"] == []        # the model's own prose, validated
    assert t.answer["provenance"][1]["visit_id"] == "9901AOD_V1"                # provenance outside the prose


def test_largest_increase_phrasing_uses_the_frozen_rule_and_the_original_question(parts, tmp_path):
    vlm = FakeVLM("The largest increase was V4.")
    a = make(parts, tmp_path, vlm)
    t = a.ask("Which visit had the largest increase in CST?", EYE, TODAY, backend="vlm")
    assert t.evidence["interpreted_as"] == "What was the largest consecutive CST increase?"
    mem = a.router.hybrid.memory(EYE)
    frozen = R.retrieve(I.parse_intent(t.evidence["interpreted_as"]), mem, 7)
    assert t.unified.final_ranking == [mem.visit_ids[i] for i in frozen.ranked]
    assert "QUESTION (as asked): Which visit had the largest increase in CST?" in prompt_of(vlm.calls[0])


def test_worst_edema_is_judged_on_the_scans_with_cst_as_a_shortlist(parts, tmp_path):
    vlm = FakeVLM(MODEL_TEXT)
    t = make(parts, tmp_path, vlm).ask("When was edema worst?", EYE, TODAY, backend="vlm")
    call, prompt = vlm.calls[0], prompt_of(vlm.calls[0])
    assert t.category == "edema_proxy" and t.answer["plan"]["kind"] == "visual+metadata"
    assert sent_visits(call) == ["V8", "V1", "V2"]                              # today + highest-CST candidates
    assert "candidate shortlisted by CST" in prompt and "CST is not a measure of edema" in prompt
    assert "The visit was selected by" not in prompt                          # no CST-as-answer framing
    assert "OLIVES finding labels exist for V1, V8 only" in prompt
    assert t.answer["answer"] == MODEL_TEXT and "Using CST as a proxy" not in t.answer["answer"]
    assert t.answer["warnings"] == []


def test_best_anatomy_sends_scans_instead_of_repeating_cst(parts, tmp_path):
    vlm = FakeVLM(MODEL_TEXT)
    t = make(parts, tmp_path, vlm).ask("Which visit had the best anatomy?", EYE, TODAY, backend="vlm")
    assert t.category == "anatomy_proxy" and sent_visits(vlm.calls[0]) == ["V8", "V3", "V5"]   # lowest-CST candidates
    assert "no anatomy grade is recorded" in prompt_of(vlm.calls[0])
    assert "approximated" not in prompt_of(vlm.calls[0])


def test_retina_change_since_baseline_attaches_both_scans(parts, tmp_path):
    vlm = FakeVLM(MODEL_TEXT)
    t = make(parts, tmp_path, vlm).ask("How has the retina changed since baseline?", EYE, TODAY, backend="vlm")
    assert t.category == "metadata" and t.answer["plan"]["kind"] == "visual+metadata"
    assert sent_visits(vlm.calls[0]) == ["V8", "V1"] and "difference of -100 µm" in prompt_of(vlm.calls[0])


def test_trajectory_compares_today_baseline_and_previous(parts, tmp_path):
    vlm = FakeVLM(MODEL_TEXT)
    make(parts, tmp_path, vlm).ask("How has the retina changed over time?", EYE, TODAY, backend="vlm")
    assert sent_visits(vlm.calls[0]) == ["V8", "V1", "V7"]


def test_similar_oct_is_compared_by_the_model_and_ranking_is_unchanged(parts, tmp_path):
    vlm = FakeVLM(MODEL_TEXT)
    t = make(parts, tmp_path, vlm).ask("Which previous visit has an OCT scan most similar to this one?", EYE, TODAY,
                                       backend="vlm")
    ref = make(parts, tmp_path).ask("Which previous visit has an OCT scan most similar to this one?", EYE, TODAY)
    assert t.category == "image_similarity" and t.unified.final_ranking == ref.unified.final_ranking
    top = [r["visit_label"] for r in t.evidence["retrieved"][:2]]
    assert sent_visits(vlm.calls[0]) == ["V8"] + top and "image-embedding cosine" in prompt_of(vlm.calls[0])


def test_mixed_question_combines_scans_and_exact_changes(parts, tmp_path):
    vlm = FakeVLM(MODEL_TEXT)
    t = make(parts, tmp_path, vlm).ask("Which earlier visit looks similar, and how did CST change?", EYE, TODAY,
                                       backend="vlm")
    assert [c["name"] for c in t.evidence["components"]] == ["image_similarity", "metadata_change"]
    top = [r["visit_label"] for r in t.evidence["retrieved"][:2]]
    assert sent_visits(vlm.calls[0]) == ["V8"] + top and "CST changed from" in prompt_of(vlm.calls[0])


def test_current_scan_question_sends_the_scan_and_labels(parts, tmp_path):
    vlm = FakeVLM(MODEL_TEXT)
    t = make(parts, tmp_path, vlm).ask("Does this scan show signs of IRF or SRF?", EYE, TODAY, backend="vlm")
    assert t.route == "scan_visual" and sent_visits(vlm.calls[0]) == ["V8"]
    assert "Dataset finding labels (OLIVES annotations, not your observations)" in prompt_of(vlm.calls[0])
    t2 = make(parts, tmp_path, FakeVLM(MODEL_TEXT)).ask("Does this scan show signs of IRF or SRF?", EYE, "9901AOD_V5")
    assert t2.evidence["biomarkers"] == []                                     # unlabelled visit: no labels claimed


def test_external_upload_is_described_without_any_history(parts, tmp_path):
    vlm = FakeVLM(MODEL_TEXT)
    scan = validate_upload(png_bytes(np.tile(np.arange(200, dtype=np.uint8), (180, 1))), "ext.png", tmp_path / "up")
    t = make(parts, tmp_path, vlm).ask("What abnormalities are visible in this OCT scan?", EYE, TODAY, upload=scan,
                                       backend="vlm")
    prompt = prompt_of(vlm.calls[0])
    assert t.route == "scan_visual" and scan.external and sent_visits(vlm.calls[0]) == ["UPLOAD"]
    assert "an external image, not linked to any patient, eye, visit or date" in prompt
    assert "Today:" not in prompt and "9901AOD" not in prompt and "V8" not in prompt
    assert t.evidence["biomarkers"] == []


def test_upload_compared_with_this_eyes_history(parts, tmp_path):
    ep, emb, _ = parts
    target = emb["9901AOD_V3"]
    vlm = FakeVLM(MODEL_TEXT)
    a = make(parts, tmp_path, vlm, encoder=lambda x: (target + 0.01)[None])
    scan = validate_upload(png_bytes(np.full((120, 120), 90, np.uint8)), "u.png", tmp_path / "up")
    t = a.ask("Which previous visit has an OCT scan most similar to this one?", EYE, TODAY, upload=scan, backend="vlm")
    ev = t.evidence
    assert t.route == "upload_similarity" and ev["retrieved"][0]["visit_id"] == "9901AOD_V3"
    assert set(ev["eligible"]) == {f"9901AOD_V{i}" for i in range(1, 8)}       # same eye, before the selected visit
    assert sent_visits(vlm.calls[0]) == ["UPLOAD", "V3", ev["retrieved"][1]["visit_label"]]
    assert "does not show that it belongs to this eye" in prompt_of(vlm.calls[0])


def test_irf_onset_with_incomplete_labels_abstains_without_model(parts, tmp_path):
    t = make(parts, tmp_path, NeverVLM()).ask("When did IRF first appear?", EYE, TODAY, backend="vlm")
    assert t.evidence["status"] == "insufficient_labels" and t.answer["status"] == "abstained"


@pytest.mark.parametrize("exc,expect", [
    (urllib.error.HTTPError("u", 429, "Too Many Requests", {}, io.BytesIO(b"rate limit")), "rate limited"),
    (urllib.error.HTTPError("u", 401, "Unauthorized", {}, io.BytesIO(f"bad key {FAKE_KEY}".encode())), "authentication"),
    (urllib.error.URLError("connection refused"), "cannot reach"),
])
def test_api_failures_are_reported_and_evidence_is_kept(parts, tmp_path, exc, expect):
    def opener(req, timeout):
        raise exc
    vlm = OpenAICompatibleVLM("https://example.invalid/v1", "m", api_key=FAKE_KEY, opener=opener)
    t = make(parts, tmp_path, vlm).ask("When was edema worst?", EYE, TODAY, backend="vlm")
    a = t.answer
    assert a["status"] == "model_error" and expect in a["answer"] and "no answer was substituted" in a["answer"]
    assert a["computed"] and [p["visit_id"] for p in a["provenance"]][:2] == [TODAY, "9901AOD_V1"]
    assert "CST" not in a["answer"].split("(")[0]                               # no metadata answer substituted
    assert FAKE_KEY not in json.dumps(a) and "THISISNOTAREALKEY" not in repr(vlm)


# --- deterministic operations support, never replace, the model --------------------------------------------------------

@pytest.mark.parametrize("q", ["When was edema worst?", "Which visit had the best anatomy?",
                               "How has the retina changed since baseline?", "Does this scan show signs of IRF?",
                               "Which previous visit has an OCT scan most similar to this one?"])
def test_interpretive_questions_are_not_answered_from_metadata_without_a_model(parts, tmp_path, q):
    t = make(parts, tmp_path).ask(q, EYE, TODAY)                               # evidence-only mode
    assert t.answer["status"] == "needs_model" and "vision-language model" in t.answer["answer"]
    assert "µm" not in t.answer["answer"] and t.answer["computed"] is not None


def test_numeric_question_without_a_model_lists_the_computed_evidence(parts, tmp_path):
    t = make(parts, tmp_path).ask("When was CST highest for this eye?", EYE, TODAY)
    assert t.answer["status"] == "answered" and t.answer["backend"] == "evidence"
    assert "not a written answer" in t.answer["answer"] and "visit V1 (CST 500 µm" in t.answer["answer"]


@pytest.mark.parametrize("q", ["When was CST highest for this eye?", "When was edema worst?",
                               "Which previous visit has an OCT scan most similar to this one?",
                               "How has the retina changed over time?", "Which visit had the best anatomy?"])
def test_prompt_names_visits_by_label_and_keeps_ids_out(parts, tmp_path, q):
    vlm = FakeVLM(MODEL_TEXT)
    make(parts, tmp_path, vlm).ask(q, EYE, TODAY, backend="vlm")
    assert not A.VISIT_ID_RE.search(prompt_of(vlm.calls[0])) and vlm.calls[0]["system"] == RS.SYSTEM
    assert not any(A.VISIT_ID_RE.search(lbl) for lbl in image_labels(vlm.calls[0]))


def test_answer_validation_flags_invented_values_visits_dates_and_cst_as_edema(parts, tmp_path):
    vlm = FakeVLM("Edema was worst at V15 in March 2019 with CST 999 µm; CST measures edema severity. "
                  "See 9901AOD_V1.")
    t = make(parts, tmp_path, vlm).ask("When was edema worst?", EYE, TODAY, backend="vlm")
    w = " | ".join(t.answer["warnings"])
    for part in ("999", "V15", "date", "CST as a measure of edema severity", "visit identifiers",
                 "finding labels exist only", "uncertainty"):
        assert part in w, part


def test_claude_backend_uses_the_same_prompt(parts, tmp_path):
    seen = []
    reply = SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=MODEL_TEXT)])
    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: seen.append(kw) or reply)))
    t = make(parts, tmp_path).ask("When was edema worst?", EYE, TODAY, backend="claude", client=client)
    assert t.answer["status"] == "answered" and seen[0]["system"] == RS.SYSTEM
    assert sum(b["type"] == "image" for b in seen[0]["messages"][0]["content"]) == 3


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


def test_upload_not_used_for_history_questions(parts, tmp_path):
    scan = validate_upload(png_bytes(np.full((100, 100), 50, np.uint8)), "u.png", tmp_path / "up")
    t = make(parts, tmp_path).ask("When was CST highest for this eye?", EYE, TODAY, upload=scan)
    assert t.route == "history" and "not used" in t.notes[0] and not t.evidence.get("upload")


def test_duplicate_scan_files_are_sent_once(parts, tmp_path):
    ep, emb, images = parts
    images["9901AOD_V1"].write_bytes(images[TODAY].read_bytes())                # baseline scan file == today's
    vlm = FakeVLM(MODEL_TEXT)
    make(parts, tmp_path, vlm).ask("How has the retina changed over time?", EYE, TODAY, backend="vlm")
    assert sent_visits(vlm.calls[0]) == ["V8", "V7"]


# --- context budget ------------------------------------------------------------------------------------------------

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
        return Resp(json.dumps({"message": {"content": MODEL_TEXT}, "done_reason": "stop"}).encode())
    vlm = make_vlm("ollama", "qwen2.5vl:3b", "http://localhost:11434/v1", num_ctx=4096, max_tokens=1500)
    assert isinstance(vlm, OllamaVLM) and vlm.context_tokens == 4096
    vlm._open = opener
    got, generate = [], vlm.generate
    vlm.generate = lambda system, ps: got.append(ps) or generate(system, ps)
    t = make(parts, tmp_path, vlm).ask("When was edema worst?", EYE, TODAY, backend="vlm")
    url, body = seen[0]
    assert url == "http://localhost:11434/api/chat" and body["options"]["num_ctx"] == 4096
    user = body["messages"][1]
    assert t.answer["status"] == "answered" and len(user["images"]) == len(t.answer["images_sent"]) >= 2
    assert user["content"].startswith("Image 1: B-scan: today, visit V8")
    assert body["options"]["num_predict"] >= RS.MIN_OUTPUT_TOKENS
    assert vlm.estimate_tokens(RS.SYSTEM, got[0]) + body["options"]["num_predict"] <= 4096
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
    assert "the model was not called" in a["answer"] and "1200 tokens" in a["answer"] and "OCT_VLM_NUM_CTX" in a["answer"]
    assert a["provenance"] and t.evidence["retrieved"][0]["visit_id"] == "9901AOD_V7"


def test_ollama_context_error_is_explained(parts, tmp_path):
    body = b'{"error":"the input length exceeds the context length"}'
    vlm = make_vlm("openai_compatible", "qwen2.5vl:3b", "http://localhost:11434/v1", context_tokens=0)
    vlm._open = lambda req, timeout: (_ for _ in ()).throw(
        urllib.error.HTTPError("u", 400, "Bad Request", {}, io.BytesIO(body)))
    t = make(parts, tmp_path, vlm).ask("Does this scan show signs of IRF or SRF?", EYE, TODAY, backend="vlm")
    assert t.answer["status"] == "model_error" and "context window" in t.answer["answer"]


def ollama_opener(replies, seen):
    """Fake Ollama server: answers each request with the next reply (an exception is raised, a dict is returned)."""
    def opener(req, timeout):
        seen.append(json.loads(req.data))
        r = replies[min(len(seen) - 1, len(replies) - 1)]
        if callable(r):
            raise r()
        return Resp(json.dumps(r).encode())
    return opener


def overflow(msg):
    return lambda: urllib.error.HTTPError("u", 400, "Bad Request", {}, io.BytesIO(json.dumps({"error": msg}).encode()))


OK = {"message": {"content": MODEL_TEXT}, "done_reason": "stop", "prompt_eval_count": 100}


def test_ollama_v1_url_uses_the_native_api_so_num_ctx_is_applied():
    vlm = make_vlm("openai_compatible", "qwen2.5vl:3b", "http://localhost:11434/v1", num_ctx=8192)
    assert isinstance(vlm, OllamaVLM) and vlm.base_url == "http://localhost:11434"
    assert vlm.num_ctx == vlm.context_tokens == 8192                     # the budget is the window actually set
    plain = make_vlm("ollama", "qwen2.5vl:3b", "http://localhost:11434")
    assert plain.num_ctx == plain.context_tokens == 4096                  # never rely on the server's default window
    small = make_vlm("ollama", "m", "", context_tokens=8192, num_ctx=4096)
    assert small.context_tokens == 4096                                   # budget never above the window sent


def test_every_ollama_request_sends_num_ctx(parts, tmp_path):
    big_scans(parts)
    seen = []
    vlm = make_vlm("ollama", "qwen2.5vl:3b", "http://localhost:11434")
    vlm._open = ollama_opener([OK], seen)
    make(parts, tmp_path, vlm).ask("When was edema worst?", EYE, TODAY, backend="vlm")
    assert seen[0]["options"]["num_ctx"] == 4096


def test_context_overflow_is_learned_and_retried_with_a_smaller_request(parts, tmp_path):
    big_scans(parts)
    seen = []
    vlm = make_vlm("ollama", "qwen2.5vl:3b", "http://localhost:11434")
    vlm.token_scale = 0.5                                                # simulate an estimate far below the server's count
    vlm._open = ollama_opener([overflow("input length (5324 tokens) exceeds the context length (4096 tokens)"), OK],
                              seen)
    t = make(parts, tmp_path, vlm).ask("When was edema worst?", EYE, TODAY, backend="vlm")
    a = t.answer
    assert a["status"] == "answered" and a["context_retries"] == 1 and len(seen) == 2
    assert vlm.token_scale > 0.5                                          # learned from the server's count
    first, second = (len(b["messages"][1].get("images", [])) for b in seen)
    sizes = [Image.open(io.BytesIO(__import__("base64").b64decode(b["messages"][1]["images"][0]))).size for b in seen]
    assert second < first or sizes[1][0] < sizes[0][0]                  # fewer or smaller scans the second time


def test_server_limit_below_the_budget_lowers_budget_and_window(parts, tmp_path):
    big_scans(parts)
    seen = []
    vlm = make_vlm("ollama", "qwen2.5vl:3b", "http://localhost:11434", num_ctx=8192)
    vlm._open = ollama_opener([overflow("requested context 8192 tokens exceeds available memory; max 4096 tokens"),
                               OK], seen)
    t = make(parts, tmp_path, vlm).ask("When was edema worst?", EYE, TODAY, backend="vlm")
    assert t.answer["status"] == "answered" and vlm.context_tokens == 4096
    assert seen[1]["options"]["num_ctx"] == 4096


def test_repeated_overflow_ends_with_a_clear_error_and_the_evidence(parts, tmp_path):
    big_scans(parts)
    seen = []
    vlm = make_vlm("ollama", "qwen2.5vl:3b", "http://localhost:11434")
    vlm._open = ollama_opener([overflow("the input length exceeds the context length")], seen)
    t = make(parts, tmp_path, vlm).ask("When was edema worst?", EYE, TODAY, backend="vlm")
    a = t.answer
    assert a["status"] == "model_error" and a["provenance"] and a["computed"]
    assert len(seen) <= 1 + RS.MAX_CONTEXT_RETRIES and ("context window" in a["answer"] or "budget" in a["answer"])


def test_non_context_errors_are_not_retried(parts, tmp_path):
    seen = []
    vlm = make_vlm("ollama", "qwen2.5vl:3b", "http://localhost:11434")
    vlm._open = ollama_opener([lambda: urllib.error.URLError("connection refused")], seen)
    t = make(parts, tmp_path, vlm).ask("When was edema worst?", EYE, TODAY, backend="vlm")
    assert len(seen) == 1 and t.answer["context_retries"] == 0 and "cannot reach" in t.answer["answer"]


def test_reported_prompt_size_calibrates_the_estimate(parts, tmp_path):
    big_scans(parts)
    seen = []
    vlm = make_vlm("ollama", "qwen2.5vl:3b", "http://localhost:11434", num_ctx=8192)
    vlm._open = ollama_opener([dict(OK, prompt_eval_count=7000)], seen)
    make(parts, tmp_path, vlm).ask("When was edema worst?", EYE, TODAY, backend="vlm")
    assert vlm.token_scale > 1.0


def test_image_preparation_downscales_without_upscaling(tmp_path):
    p = tmp_path / "s.png"
    Image.fromarray(np.zeros((496, 504), np.uint8)).save(p)
    _, w, h = prepare_image(p, 448)
    assert max(w, h) <= 448 and w % 28 == 0 and h % 28 == 0
    assert prepare_image(p, 1024)[1:] == (504, 496)
    assert prepare_image(p, None, "jpeg")[0][:2] == b"\xff\xd8"


def test_config_reads_context_budget_settings():
    c = load_config(env={"OCT_VLM_PROVIDER": "ollama", "OCT_VLM_NUM_CTX": "8192", "OCT_VLM_MAX_IMAGES": "2",
                         "OCT_VLM_IMAGE_MAX_SIDE": "392"}, dotenv=None)
    assert (c.num_ctx, c.max_images, c.image_max_side, c.context_tokens) == (8192, 2, 392, None)
    vlm = make_vlm(c.provider, "m", "", **c.vlm_options())
    assert vlm.context_tokens == 8192 and vlm.num_ctx == 8192 and vlm.base_url == "http://localhost:11434"
    assert make_vlm("openai_compatible", "m", "https://openrouter.ai/api/v1").context_tokens == 0



# --- uploaded scan vs individual stored B-scans ------------------------------------------------------------------------

def scan_index(parts, tmp_path, per_visit=3, seed=5):
    """Several stored B-scans per visit for eyes 901 and 902, each with its own image file and unit vector."""
    ep = parts[0]
    rng = np.random.default_rng(seed)
    keys, vids, eyes, idx, paths, emb = [], [], [], [], [], []
    for r in ep.itertuples():
        for k, si in enumerate([12, 24, 36][:per_visit]):
            p = tmp_path / f"{r.visit_id}__scan{si}.png"
            Image.fromarray(rng.integers(0, 255, (64, 64), dtype=np.uint8)).save(p)
            x = rng.normal(size=16)
            keys.append(f"{r.visit_id}__scan{si}"); vids.append(r.visit_id); eyes.append(int(r.eye_id))
            idx.append(si); paths.append(p); emb.append(x / np.linalg.norm(x))
    return ScanIndex(keys, vids, eyes, idx, paths, np.array(emb, np.float32))


def make_scans(parts, tmp_path, scans, vlm=None, encoder=None):
    ep, emb, images = parts
    return Assistant(Config(workdir=tmp_path / "rt", visit_mode="mean"), vlm=vlm, upload_encoder=encoder, episodes=ep,
                     visit_emb=emb, images=images, emb_encoder="resnet50", scans=scans)


def upload(tmp_path, value=90):
    return validate_upload(png_bytes(np.full((120, 120), value, np.uint8)), "u.png", tmp_path / "up")


SIM_Q = "Which previous visit has an OCT scan most similar to this one?"


def test_upload_is_ranked_against_individual_scans_with_the_best_scan_kept(parts, tmp_path):
    scans = scan_index(parts, tmp_path)
    i = scans.keys.index("9901AOD_V3__scan36")                        # a NON-central scan
    q = scans.emb[i] + 0.05 * scans.emb[scans.keys.index("9901AOD_V5__scan12")]
    t = make_scans(parts, tmp_path, scans, encoder=lambda x: q[None]).ask(SIM_Q, EYE, TODAY, upload=upload(tmp_path))
    ev = t.evidence
    assert ev["status"] == "retrieved" and t.route == "upload_similarity"
    top = ev["retrieved"][0]
    assert (top["visit_id"], top["best_scan"], top["best_scan_index"], top["n_scans_compared"]) == (
        "9901AOD_V3", "9901AOD_V3__scan36", 36, 3)
    zq = q / np.linalg.norm(q)
    assert top["image_cosine"] == round(float(scans.emb[i] @ zq), 4)
    assert top["best_scan_path"].endswith("9901AOD_V3__scan36.png") and top["best_scan_file"] == "9901AOD_V3__scan36.png"
    assert "individual stored B-scans" in ev["ranking_basis"] and "21 individual" in ev["ranking_basis"]  # 7 visits x 3


def test_visit_score_is_the_maximum_over_its_scans_and_ties_prefer_recent(tmp_path):
    e = np.eye(4, dtype=np.float32)
    s = ScanIndex(keys=["a1", "a2", "b1", "c1", "x1"], visit_ids=["A", "A", "B", "C", "X"], eye_ids=[1, 1, 1, 1, 2],
                  scan_index=[12, 24, 24, 24, 24], paths=[None] * 5,
                  emb=np.stack([e[0], e[1], e[1], e[2], e[1]]))
    q = (0.6 * e[0] + 0.8 * e[1]).astype(np.float32)
    r = s.rank_visits(q, 1, ["A", "B", "C"])
    assert [(b["visit_id"], b["image_key"], round(b["score"], 3), b["n_scans"]) for b in r] == [
        ("B", "b1", 0.8, 1), ("A", "a2", 0.8, 2), ("C", "c1", 0.0, 1)]          # A ties B at 0.8: B is more recent
    assert all(b["visit_id"] != "X" for b in s.rank_visits(q, 1, ["A", "B", "C", "X"]))   # other eye never ranked
    with pytest.raises(ValueError, match="dimensions"):
        s.rank_visits(np.ones(8, np.float32), 1, ["A"])


def test_upload_never_retrieves_another_eyes_scans(parts, tmp_path):
    scans = scan_index(parts, tmp_path)
    q = scans.emb[scans.keys.index("9902BOS_V2__scan24")]             # identical to a scan of ANOTHER eye
    t = make_scans(parts, tmp_path, scans, encoder=lambda x: q[None]).ask(SIM_Q, EYE, TODAY, upload=upload(tmp_path))
    assert all(r["visit_id"].startswith("9901AOD") for r in t.evidence["retrieved"])
    assert all(v.startswith("9901AOD") for v in t.evidence["ranking"])
    assert all(r["visit_index"] < 7 for r in t.evidence["retrieved"])               # before the selected V8


def test_mean_visit_mode_no_longer_blocks_uploads_and_mean_vectors_are_not_used(parts, tmp_path):
    scans = scan_index(parts, tmp_path)
    ep, emb, images = parts
    bogus = {v: -x for v, x in emb.items()}                             # visit-level vectors that would rank inversely
    a = Assistant(Config(workdir=tmp_path / "rt", visit_mode="mean"), upload_encoder=lambda x: scans.emb[3][None],
                  episodes=ep, visit_emb=bogus, images=images, emb_encoder="resnet50", scans=scans)
    t = a.ask(SIM_Q, EYE, TODAY, upload=upload(tmp_path))
    assert t.evidence["status"] == "retrieved" and t.evidence["retrieved"][0]["best_scan"] == scans.keys[3]


def test_embedding_failure_shows_the_real_reason(parts, tmp_path):
    def broken(x):
        raise OSError("could not download resnet50 weights from download.pytorch.org")
    t = make_scans(parts, tmp_path, scan_index(parts, tmp_path), encoder=broken).ask(SIM_Q, EYE, TODAY,
                                                                                    upload=upload(tmp_path))
    assert t.evidence["status"] == "unsupported_image"
    assert "OSError: could not download resnet50 weights" in t.evidence["reason"]
    assert "central-scan" not in t.evidence["reason"]
    t2 = make_scans(parts, tmp_path, scan_index(parts, tmp_path), encoder=lambda x: np.ones((1, 8))).ask(
        SIM_Q, EYE, TODAY, upload=upload(tmp_path))
    assert "embedding mismatch" in t2.evidence["reason"] and "8 dimensions" in t2.evidence["reason"]


def test_non_resnet_scan_embeddings_are_reported_precisely(parts, tmp_path):
    scans = scan_index(parts, tmp_path)
    scans.encoder = "retfound_oct"
    a = make_scans(parts, tmp_path, scans)
    a.cfg.upload_embedding = True
    t = a.ask(SIM_Q, EYE, TODAY, upload=upload(tmp_path))
    assert "made with retfound_oct" in t.evidence["reason"]


def test_upload_embedding_uses_the_stored_pipeline_preprocessing(tmp_path):
    seen = []
    img = (np.arange(496 * 504).reshape(496, 504) % 251).astype(np.uint8)
    scan = validate_upload(png_bytes(img), "s.png", tmp_path)
    v = embed_scan(scan, lambda x: seen.append(x) or np.ones((1, 2048), np.float32) * 3)
    gray, _ = OI.load_gray(scan.path)
    assert seen[0].shape == (1, 3, 224, 224) and np.array_equal(seen[0][0], OI.preprocess(gray)[1])
    assert v.shape == (2048,) and abs(np.linalg.norm(v) - 1) < 1e-6


def test_real_resnet50_upload_matches_the_stored_embedding_of_the_same_scan():
    """Compatibility on real data: a dataset B-scan uploaded as PNG embeds to (almost) its stored vector."""
    if os.environ.get("OCT_TEST_RESNET") != "1":
        pytest.skip("set OCT_TEST_RESNET=1 (needs torch and the ImageNet ResNet-50 weights)")
    import embed_cached as EC
    subset, embf = ROOT / "phase2/data/atlas_fullhist", ROOT / "phase2/results/fullhist/resnet50/embeddings.npz"
    scans = ScanIndex.load(subset, embf)
    enc, dim = EC.make_encoder("resnet50", "gap", OI.pick_device())
    assert dim == scans.dim == 2048 and (scans.encoder, scans.pooling) == ("resnet50", "gap")
    i = 5
    gray = np.asarray(Image.open(scans.paths[i]).convert("L"))
    q = embed_scan(validate_upload(png_bytes(gray), "x.png", ROOT / "app/.runtime/test"), enc)
    assert float(scans.emb[i] @ q) > 0.999
    assert scans.rank_visits(q, scans.eye_ids[i], [scans.visit_ids[i]])[0]["image_key"] == scans.keys[i]


def test_vlm_compares_the_upload_with_the_best_matching_scans_but_does_not_rank(parts, tmp_path):
    scans = scan_index(parts, tmp_path)
    q = scans.emb[scans.keys.index("9901AOD_V3__scan36")]
    vlm = FakeVLM("Both scans may show a similar foveal contour; this is an uncertain visual reading.")
    a = make_scans(parts, tmp_path, scans, vlm=vlm, encoder=lambda x: q[None])
    t = a.ask(SIM_Q, EYE, TODAY, upload=upload(tmp_path), backend="vlm")
    ref = make_scans(parts, tmp_path, scans, encoder=lambda x: q[None]).ask(SIM_Q, EYE, TODAY, upload=upload(tmp_path))
    call = vlm.calls[0]
    assert t.evidence["ranking"] == ref.evidence["ranking"]                         # the model does not rank
    paths = [p["path"] for p in call["parts"] if p["type"] == "image"]
    top2 = t.evidence["retrieved"][:2]
    assert paths == [str(t.upload.path)] + [r["best_scan_path"] for r in top2]     # best scans, not central ones
    assert image_labels(call)[1] == "B-scan: visit V3 (B-scan 36, its best match to the upload)"
    prompt = prompt_of(call)
    assert "Embedding ranking of this eye's eligible visits by their best-matching stored B-scan (fixed" in prompt
    assert "V3 cosine 1.000 (B-scan 36 of 3 compared)" in prompt and not A.VISIT_ID_RE.search(prompt)
    assert "the upload is not this visit's scan" in prompt and "today it is" not in prompt
    prov = {p["visit_id"]: p for p in t.answer["provenance"]}
    assert prov["9901AOD_V3"]["scan_file"] == "9901AOD_V3__scan36.png" and prov["9901AOD_V3"]["similarity"] == 1.0
    assert prov["UPLOADED_SCAN"]["scan_sent"] and t.answer["answer"] == vlm.text

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
