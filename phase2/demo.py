"""Question -> hybrid retrieval -> answer demo (research prototype; not clinical advice).

Shows, for one question asked at one visit of one eye: the metadata, image and final hybrid rankings, the OCT
B-scans used (image questions), the generated answer and the evidence it was given.

  python phase2/demo.py --eye 25 --question "Which earlier visit's OCT looks most like today's?"
  python phase2/demo.py --eye 61 --visit 01-013_W52 --question "Most recent prior visit with CST above today's value?"
  python phase2/demo.py --serve [--share]                 # small web form (needs: pip install gradio)
  add --backend claude to generate the answer with Claude (needs: pip install anthropic, ANTHROPIC_API_KEY)

Defaults use the saved 73-image full-history embeddings (dev eyes 25, 11, 61). After the larger subset is embedded:
  --subset phase2/data/scan_subset_v1 --emb phase2/results/scan_subset_v1/resnet50/embeddings.npz
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phase2"))
import answer as A  # noqa: E402
import hybrid_retriever as H  # noqa: E402
import oct_image as OI  # noqa: E402

DEFAULT_SUBSET = ROOT / "phase2/data/atlas_fullhist"
DEFAULT_EMB = ROOT / "phase2/results/fullhist/resnet50/embeddings.npz"
EXAMPLES = [
    "Which earlier visit's OCT looks most like today's?",
    "At which earlier visit was the retinal fluid/thickening most similar to today?",
    "When was the CST most similar to today's measurement?",
    "Compare with the previous visit.",
    "Most recent prior visit with CST above today's value?",
    "When was the last injection before today?",
]


class Demo:
    def __init__(self, subset: Path = DEFAULT_SUBSET, emb: Path = DEFAULT_EMB, visit_mode: str = "central"):
        self.episodes = pd.read_csv(ROOT / "data/processed/episodes.csv")
        man = OI.load_manifest(subset)
        c = man[man.scan_index == 24]
        self.images = {r.visit_id: subset / r.local_path for r in c.itertuples() if (subset / r.local_path).exists()}
        vemb = H.load_visit_embeddings(subset, emb, mode=visit_mode)
        self.retriever = H.HybridRetriever(self.episodes, vemb)          # default policy: no fusion for metadata ops
        self.eyes = sorted(int(e) for e in man.eye_id.unique())
        self.source = dict(subset=str(subset.relative_to(ROOT)) if subset.is_relative_to(ROOT) else str(subset),
                           embeddings=str(emb.relative_to(ROOT)) if emb.is_relative_to(ROOT) else str(emb),
                           visit_mode=visit_mode)

    def latest_visit(self, eye: int) -> str:
        return self.episodes[self.episodes.eye_id == eye].sort_values("visit_index").visit_id.iloc[-1]

    def run(self, question: str, eye: int, visit: str | None = None, backend: str = "evidence") -> dict:
        visit = visit or self.latest_visit(eye)
        res = self.retriever.retrieve(question, eye, visit)
        ev = A.build_evidence(res, self.episodes, self.images, question)
        return dict(result=res, evidence=ev, answer=A.answer(ev, backend))


def to_text(out: dict) -> str:
    r, a = out["result"], out["answer"]
    t = r.table().sort_values("final_rank", na_position="last")
    lines = [A.evidence_text(out["evidence"]), "", f"status={r.status} route={r.route} intent={r.intent}", "",
             t.to_string(index=False), "", f"ANSWER ({a['backend']}): {a['answer']}", f"Evidence: {a['evidence_used']}"]
    return "\n".join(lines)


def to_html(out: dict, source: dict | None = None) -> str:
    r, ev, a = out["result"], out["evidence"], out["answer"]
    e = html.escape
    imgs = "".join(
        f"<figure><img src='data:image/webp;base64,{base64.b64encode(Path(i['path']).read_bytes()).decode()}' "
        f"alt='{e(i['visit_id'])}'><figcaption>{e(i['visit_id'])} ({e(i['role'])})</figcaption></figure>"
        for i in ev["images"])
    table = r.table()
    table = table[table.final_rank.notna() | table.metadata_rank.notna() | table.image_rank.notna()]
    table = table.sort_values("final_rank", na_position="last").astype(object).where(table.notna(), "")
    css = ("body{font:14px/1.45 system-ui,sans-serif;max-width:1100px;margin:16px auto;padding:0 16px;color:#1d2329;"
           "background:#fff}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ccd3da;padding:3px 7px;"
           "text-align:right}th{background:#eef1f4}figure{display:inline-block;margin:6px}img{width:300px}"
           ".ans{background:#f4f7fa;border-left:4px solid #2b6cb0;padding:10px 14px;white-space:pre-wrap}"
           ".warn{color:#8a5a00}")
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>OCT retrieval demo</title><style>{css}</style>"
            f"</head><body><h2>{e(ev['question'])}</h2>"
            f"<p>Eye {ev['eye_id']}, current visit <b>{e(ev['current']['visit_id'])}</b>. Status <b>{e(r.status)}</b>, "
            f"route <b>{e(r.route)}</b>, parsed intent <code>{e(r.intent)}</code>. {e(r.reason)}</p>"
            f"<h3>Answer <small>({e(a['backend'])})</small></h3><div class='ans'>{e(a['answer'])}</div>"
            f"<p><b>Evidence given to the answer generator:</b> {e(a['evidence_used'])}</p>"
            + (f"<h3>OCT B-scans</h3>{imgs}" if imgs else "")
            + "<h3>Rankings (metadata, CST proxy, image, final)</h3>" + table.to_html(index=False, na_rep="")
            + f"<h3>Evidence text</h3><pre>{e(A.evidence_text(ev))}</pre>"
            + f"<p class='warn'>Research prototype on OLIVES data (CC BY 4.0); not clinically validated. "
              f"Source: <code>{e(json.dumps(source or {}))}</code></p></body></html>")


def serve(demo: Demo, share: bool) -> None:
    import gradio as gr

    def fn(question, eye, visit, backend):
        try:
            return to_html(demo.run(question, int(eye), visit.strip() or None, backend), demo.source)
        except Exception as ex:  # shown in the page instead of crashing the server
            return f"<p>Error: {html.escape(str(ex))}</p>"
    gr.Interface(fn, [gr.Dropdown(EXAMPLES, value=EXAMPLES[0], allow_custom_value=True, label="Question"),
                      gr.Dropdown([str(e) for e in demo.eyes], value=str(demo.eyes[0]), label="Eye"),
                      gr.Textbox(label="Current visit id (blank = latest)"),
                      gr.Radio(["evidence", "claude"], value="evidence", label="Answer backend")],
                 gr.HTML(), title="Question-aware OCT retrieval demo", flagging_mode="never").launch(share=share)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--question", default=EXAMPLES[0])
    ap.add_argument("--eye", type=int)
    ap.add_argument("--visit")
    ap.add_argument("--backend", choices=["evidence", "claude"], default="evidence")
    ap.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    ap.add_argument("--emb", type=Path, default=DEFAULT_EMB)
    ap.add_argument("--visit-mode", choices=["central", "mean"], default="central")
    ap.add_argument("--html", type=Path, help="write the result page here")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--share", action="store_true")
    a = ap.parse_args(argv)
    demo = Demo(a.subset, a.emb, a.visit_mode)
    if a.serve:
        serve(demo, a.share)
        return 0
    out = demo.run(a.question, a.eye if a.eye is not None else demo.eyes[0], a.visit, a.backend)
    print(to_text(out))
    if a.html:
        a.html.write_text(to_html(out, demo.source), encoding="utf-8")
        print(f"\nHTML -> {a.html}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
