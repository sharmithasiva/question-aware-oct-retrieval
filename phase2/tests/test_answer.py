"""Answer generation and demo: abstentions never reach a model; images only on image routes (fake client, no network)."""
import base64
import html
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase2"))
import answer as A  # noqa: E402
import demo as D  # noqa: E402

DEMO = D.Demo()


class FakeClient:
    def __init__(self, text="Answer.\nEvidence used: X", stop="end_turn"):
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))
        self.text, self.stop = text, stop

    def _create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(stop_reason=self.stop, content=[SimpleNamespace(type="thinking", thinking=""),
                                                               SimpleNamespace(type="text", text=self.text)])


def evidence(question, eye=25, visit=None):
    return DEMO.run(question, eye, visit)["evidence"]


def test_image_route_sends_labelled_current_and_top_images():
    ev = evidence(D.EXAMPLES[0])
    assert ev["route"] == "image+proxy_rrf" and [i["role"] for i in ev["images"]] == ["current", "retrieved", "retrieved"]
    req = A.claude_request(ev)
    assert req["model"] == "claude-opus-5-5" and req["fallbacks"] == "default" and "Do not invent" in req["system"]
    blocks = req["messages"][0]["content"]
    imgs = [b for b in blocks if b["type"] == "image"]
    assert len(imgs) == 3 and all(b["source"]["media_type"] == "image/webp" for b in imgs)
    assert base64.b64decode(imgs[0]["source"]["data"])[:4] == b"RIFF"
    assert blocks[0]["text"].startswith("B-scan: 0210MOD_V27 (current")    # each image is labelled before it
    c = FakeClient()
    out = A.answer(ev, "claude", c)
    assert out["answer"].startswith("Answer.") and len(c.calls) == 1 and out["backend"] == "claude"


def test_metadata_route_sends_no_images():
    ev = evidence("When was the CST most similar to today's measurement?", 61)
    assert ev["route"] == "metadata" and ev["images"] == []
    assert all(b["type"] == "text" for b in A.claude_request(ev)["messages"][0]["content"])


@pytest.mark.parametrize("q,eye,visit", [("Most recent prior visit with CST above today's value?", 11, "0232GOD_V19"),
                                         ("When was the last injection before today?", 61, None)])
def test_abstentions_never_call_the_model(q, eye, visit):
    ev = evidence(q, eye, visit)
    c = FakeClient()
    out = A.answer(ev, "claude", c)
    assert ev["status"] != "retrieved" and c.calls == [] and out["answer"].startswith("No answer")
    assert out["backend"] == "none" and ev["retrieved"] == []


def test_refusal_and_evidence_only_backend():
    ev = evidence(D.EXAMPLES[0])
    assert "declined" in A.answer(ev, "claude", FakeClient(stop="refusal"))["answer"]
    txt = A.answer(ev, "evidence")["answer"]
    assert "0210MOD_V26" in txt and "no language model" in txt
    with pytest.raises(ValueError):
        A.answer(ev, "gpt")


@pytest.mark.parametrize("q", D.EXAMPLES)
def test_demo_pages_render_for_every_example(q):
    for eye in DEMO.eyes:
        out = DEMO.run(q, eye)
        page = D.to_html(out, DEMO.source)
        assert page.startswith("<!doctype html>") and html.escape(out["evidence"]["question"]) in page
        assert "<td>None</td>" not in page
