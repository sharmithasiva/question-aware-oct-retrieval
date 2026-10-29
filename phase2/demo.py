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
        episodes = pd.read_csv(ROOT / "data/processed/episodes.csv")
        man = OI.load_manifest(subset)
        c = man[man.scan_index == 24]
        images = {r.visit_id: subset / r.local_path for r in c.itertuples() if (subset / r.local_path).exists()}
        vemb = H.load_visit_embeddings(subset, emb, mode=visit_mode)
        source = dict(subset=str(subset.relative_to(ROOT)) if subset.is_relative_to(ROOT) else str(subset),
                      embeddings=str(emb.relative_to(ROOT)) if emb.is_relative_to(ROOT) else str(emb),
                      visit_mode=visit_mode)
        self._setup(episodes, vemb, images, sorted(int(e) for e in man.eye_id.unique()), source)

    @classmethod
    def from_parts(cls, episodes: pd.DataFrame, visit_emb: dict, images: dict, eyes=None, source=None) -> "Demo":
        """Build from in-memory parts (tests, notebooks): no files are read."""
        d = cls.__new__(cls)
        d._setup(episodes, visit_emb, images, sorted(eyes or episodes.eye_id.unique()), source or {})
        return d

    def _setup(self, episodes, visit_emb, images, eyes, source):
        self.episodes, self.images, self.eyes, self.source = episodes, images, [int(e) for e in eyes], source
        self.retriever = H.HybridRetriever(episodes, visit_emb)          # default policy: no fusion for metadata ops

    def latest_visit(self, eye: int) -> str:
        h = self.episodes[self.episodes.eye_id == eye]
        if h.empty:
            raise ValueError(f"eye {eye} is not in the visit table")
        return h.sort_values("visit_index").visit_id.iloc[-1]

    def run(self, question: str, eye: int, visit: str | None = None, backend: str = "evidence", client=None) -> dict:
        """ValueError for an unknown eye or a visit of another eye; everything else is reported in the result."""
        if not question or not question.strip():
            raise ValueError("empty question")
        visit = visit or self.latest_visit(eye)
        res = self.retriever.retrieve(question, eye, visit)
        ev = A.build_evidence(res, self.episodes, self.images, question)
        return dict(result=res, evidence=ev, answer=A.answer(ev, backend, client))


def _ranks(r) -> pd.DataFrame:
    """Per-candidate ranks, best final rank first (an empty table when the eye has no earlier visit)."""
    t = r.table()
    return t if t.empty else t.sort_values("final_rank", na_position="last")


def to_text(out: dict) -> str:
    r, ev, a = out["result"], out["evidence"], out["answer"]
    t = _ranks(r)
    lines = ["NOTE: " + A.DISCLAIMER, "", A.evidence_text(ev), "", f"status={r.status} route={r.route} intent={r.intent}",
             "", "All eligible earlier visits (metadata / CST proxy / image / final ranks):", t.to_string(index=False), "",
             f"ANSWER [{a['status']}, backend={a['backend']}]: {a['answer']}", f"Evidence given: {a['evidence_used']}"]
    lines += [f"WARNING: {w}" for w in a["warnings"]]
    return "\n".join(lines)


def _img_tag(i: dict) -> str:
    p = Path(i["path"])
    mime = A.MEDIA_TYPES.get(p.suffix.lower(), "application/octet-stream")
    return (f"<figure><img src='data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}' "
            f"alt='Central B-scan of {html.escape(i['visit_id'])}'><figcaption>{html.escape(i['visit_id'])} "
            f"({html.escape(i['role'])} visit, central B-scan)</figcaption></figure>")


def _fmt_num(x, unit=""):
    return "not recorded" if x is None else f"{x:.0f}{unit}"


def _fmt_cos(x):
    return "" if x is None else f"{x:.3f}"


def to_html(out: dict, source: dict | None = None) -> str:
    r, ev, a = out["result"], out["evidence"], out["answer"]
    e = html.escape
    cur = ev["current"]
    banner = {"abstained": ("abst", "Abstained: no answer was generated."),
              "model_error": ("err", "The language model call failed: no answer was generated."),
              "model_refused": ("err", "The language model declined: no answer was generated."),
              "model_incomplete": ("err", "The generated answer is incomplete.")}.get(a["status"])
    rows = "".join(
        f"<tr><td>{v['rank']}</td><td>{e(v['role'])}</td><td>{e(v['visit_id'])}</td><td>{e(v['visit_label'])}</td>"
        f"<td>{_fmt_num(v['week'])}</td><td>{_fmt_num(v['cst'], ' µm')}</td><td>{_fmt_num(v['bcva'])}</td>"
        f"<td>{_fmt_cos(v['image_cosine'])}</td><td>{'yes' if v['image_attached'] else 'no'}</td></tr>"
        for v in ev["retrieved"])
    evidence_table = ("<table><tr><th>Rank</th><th>Role</th><th>Visit</th><th>Label</th><th>Week</th><th>CST</th>"
                      "<th>BCVA (letters)</th><th>Image-embedding cosine</th><th>B-scan sent</th></tr>" + rows + "</table>") if rows else \
        "<p>No earlier visit was selected.</p>"
    table = _ranks(r)
    table = table.astype(object).where(table.notna(), "")
    imgs = "".join(_img_tag(i) for i in ev["images"])
    css = ("body{font:14px/1.45 system-ui,sans-serif;max-width:1100px;margin:16px auto;padding:0 16px;color:#1d2329;"
           "background:#fff}table{border-collapse:collapse;font-size:13px;margin:4px 0}td,th{border:1px solid #ccd3da;"
           "padding:3px 7px;text-align:left}th{background:#eef1f4}figure{display:inline-block;margin:6px}"
           "img{width:300px;max-width:100%}.ans{background:#f4f7fa;border-left:4px solid #2b6cb0;padding:10px 14px;"
           "white-space:pre-wrap}.note{background:#fff8e6;border:1px solid #e0c27a;padding:8px 12px}"
           ".abst{background:#eef4ff;border:1px solid #8fb0e0;padding:8px 12px}.err{background:#fdecec;"
           "border:1px solid #e09a9a;padding:8px 12px}code{font-size:12px}")
    parts = [
        f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,"
        f"initial-scale=1'><title>OCT retrieval demo</title><style>{css}</style></head><body>",
        f"<p class='note'><b>Not for clinical use.</b> {e(A.DISCLAIMER)}</p>",
        f"<h2>{e(ev['question'])}</h2>",
        "<h3>Current visit</h3><table><tr><th>Eye</th><th>Visit</th><th>Label</th><th>Week</th><th>CST</th>"
        f"<th>BCVA (letters)</th></tr><tr><td>{ev['eye_id']}</td><td>{e(cur['visit_id'])}</td>"
        f"<td>{e(cur['visit_label'])}</td><td>{_fmt_num(cur['week'])}</td><td>{_fmt_num(cur['cst'], ' µm')}</td>"
        f"<td>{_fmt_num(cur['bcva'])}</td></tr></table>",
        f"<h3>Retrieval</h3><p>Route <b>{e(r.route)}</b> (status <b>{e(r.status)}</b>; parsed intent "
        f"<code>{e(r.intent)}</code>). {e(ev['ranking_basis'])}</p>",
    ]
    if a["status"] == "abstained":
        parts.append(f"<p class='abst'><b>{e(banner[1])}</b> Reason: {e(ev['reason'] or 'none given')}. "
                     f"{e(ev['guidance'])}</p>")
    parts += [f"<h3>Selected evidence</h3>{evidence_table}",
              f"<h3>Answer <small>({e(a['backend'])}, {e(a['status'])})</small></h3>"]
    if banner and a["status"] != "abstained":
        parts.append(f"<p class='{banner[0]}'><b>{e(banner[1])}</b></p>")
    parts.append(f"<div class='ans'>{e(a['answer'])}</div>")
    if a["warnings"]:
        parts.append("<p class='err'><b>Check against the evidence:</b></p><ul>"
                     + "".join(f"<li>{e(w)}</li>" for w in a["warnings"]) + "</ul>")
    parts.append(f"<p><b>Evidence given to the answer generator:</b> {e(a['evidence_used'])}</p>")
    if imgs:
        parts.append(f"<h3>OCT B-scans given to the answer generator</h3>{imgs}")
    parts += ["<details><summary>All eligible earlier visits: metadata, CST-proxy, image and final ranks</summary>"
              + (table.to_html(index=False, na_rep="") if len(table) else "<p>No earlier visit.</p>") + "</details>",
              f"<details><summary>Evidence text sent to the answer generator</summary><pre>{e(A.evidence_text(ev))}"
              f"</pre></details>",
              f"<p><small>OLIVES data (CC BY 4.0). Source: <code>{e(json.dumps(source or {}))}</code></small></p>",
              "</body></html>"]
    return "".join(parts)


def serve(demo: Demo, share: bool) -> None:
    import gradio as gr

    def fn(question, eye, visit, backend):
        try:
            return to_html(demo.run(question, int(eye), visit.strip() or None, backend), demo.source)
        except Exception as ex:  # shown in the page instead of crashing the server; credentials redacted
            return f"<p>Error: {html.escape(A.redact(f'{type(ex).__name__}: {ex}'))}</p>"
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
    try:
        out = demo.run(a.question, a.eye if a.eye is not None else demo.eyes[0], a.visit, a.backend)
    except ValueError as ex:
        print(f"error: {ex}", file=sys.stderr)
        return 2
    print(to_text(out))
    if a.html:
        a.html.write_text(to_html(out, demo.source), encoding="utf-8")
        print(f"\nHTML -> {a.html}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
