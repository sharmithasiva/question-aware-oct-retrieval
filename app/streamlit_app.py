"""Question-aware multimodal OCT assistant: Streamlit UI (research prototype, not a clinical tool).

Run from the repository root:  streamlit run app/streamlit_app.py
Every answer is produced by app.assistant.Assistant, i.e. question routing -> phase2 retrieval -> evidence package ->
(optional) vision-language model -> grounding checks. The UI only displays what that pipeline returns.
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import altair as alt  # noqa: E402  (bundled with streamlit)
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.assistant import Assistant, Turn  # noqa: E402
from app.config import PRESETS, load_config  # noqa: E402
from app.uploads import UploadError, validate_upload  # noqa: E402
from app.vlm import make_vlm  # noqa: E402

import answer as A  # noqa: E402  (phase2, on sys.path via app.assistant)

SUGGESTIONS = [
    "When was CST highest for this eye?",
    "When was edema worst?",
    "Which previous visit has an OCT scan most similar to this one?",
    "Which visit had the largest increase in CST?",
    "Compare the current scan with the previous visit.",
    "How has the retinal anatomy changed over time?",
    "Which earlier visit looks similar, and how did CST change?",
    "What abnormalities are visible in this OCT scan?",
    "Does this scan show signs of IRF or SRF?",
    "When did IRF first appear?",
    "What evidence supports your answer?",
]
SOURCE_STYLE = {   # source -> (badge text, css class)
    "metadata": ("metadata", "meta"), "metadata_change": ("measured change", "meta"),
    "image_similarity": ("image retrieval", "img"), "upload_image_similarity": ("image retrieval (upload)", "img"),
    "upload_similarity": ("image retrieval (upload)", "img"), "biomarker_labels": ("dataset labels", "lab"),
    "uploaded_scan": ("uploaded scan", "vlm"), "current_visit_scan": ("dataset scan", "vlm"),
}
CSS = """
<style>
.badge{display:inline-block;padding:1px 8px;margin:0 4px 2px 0;border-radius:10px;font-size:0.78rem;font-weight:600}
.meta{background:#dbeafe;color:#1e3a8a}.img{background:#ede9fe;color:#4c1d95}.lab{background:#dcfce7;color:#14532d}
.vlm{background:#ffedd5;color:#7c2d12}.stat{background:#e5e7eb;color:#111827}.bad{background:#fee2e2;color:#7f1d1d}
.disc{border-left:4px solid #d97706;background:rgba(217,119,6,0.08);padding:6px 10px;margin:4px 0;font-size:0.9rem}
.abst{border-left:4px solid #2563eb;background:rgba(37,99,235,0.08);padding:6px 10px;margin:4px 0}
.note{font-size:0.82rem;opacity:0.8}
</style>"""


def badge(text: str, cls: str) -> str:
    return f"<span class='badge {cls}'>{html.escape(text)}</span>"


@st.cache_resource(show_spinner="Loading dataset, embeddings and retrievers ...")
def get_assistant() -> Assistant:
    return Assistant(load_config())


@st.cache_resource
def get_vlm(provider: str, model: str, base_url: str, key: str, timeout: float, max_tokens: int):
    return make_vlm(provider, model, base_url, key, timeout, max_tokens)


def sidebar(a: Assistant):
    cfg = a.cfg
    st.sidebar.title("OCT Assistant")
    st.sidebar.caption("Question-aware episodic retrieval for longitudinal retinal OCT, with a vision-language model. "
                       "Research prototype on OLIVES data: not a diagnostic tool.")

    eyes = a.eyes()
    labels = {int(r.eye_id): f"Eye {r.eye_id} · {r.subject} {r.laterality} · {r.trial} · {r.n_visits} visits"
                              f" · OCT {r.n_with_oct}/{r.n_visits}" for r in eyes.itertuples()}
    eye = st.sidebar.selectbox("Eye / patient", list(labels), format_func=labels.get, key="eye")
    vis = a.visits(eye)

    def vlabel(i):
        r = vis.iloc[i]
        wk = f" · week {int(r.week)}" if pd.notna(r.week) else ""
        cst = f" · CST {r.cst:.0f} µm" if pd.notna(r.cst) else ""
        return f"{r.visit_label}{wk}{cst}" + (" · OCT" if r.has_oct else "") + (" · labels" if r.has_biomarkers else "")
    i = st.sidebar.selectbox("Current visit (questions look at earlier visits)", list(range(len(vis))),
                             index=len(vis) - 1, format_func=vlabel, key=f"visit_{eye}")
    visit = vis.visit_id.iloc[i]

    with st.sidebar.expander("Dataset information"):
        st.markdown(f"- Image subset: `{cfg.subset.name}`; embeddings: `{cfg.embeddings.name}` ({a.emb_encoder}, "
                    f"{cfg.visit_mode} scan)\n- Eyes shown: splits {', '.join(cfg.splits)} ({len(eyes)} eyes); "
                    f"{int((eyes.n_with_oct > 0).sum())} have OCT images here\n- No calendar dates exist in the "
                    "dataset: visits are identified by visit label, plus study week for Prime eyes\n- Finding labels "
                    "(OLIVES) exist only for the first and last visit of each eye")

    st.sidebar.subheader("Answer model")
    names = list(PRESETS)
    default = next((n for n, p in PRESETS.items() if p[0] == cfg.provider and (not cfg.base_url or p[1] == cfg.base_url)),
                   names[0])
    preset = st.sidebar.selectbox("Provider", names, index=names.index(default))
    provider, url, model = PRESETS[preset]
    if provider == cfg.provider:
        url, model = cfg.base_url or url, cfg.model or model
    if provider != "evidence":
        model = st.sidebar.text_input("Model id", model)
        if provider == "openai_compatible":
            url = st.sidebar.text_input("Base URL", url)
        key = cfg.api_key if provider == cfg.provider else ""
        typed = st.sidebar.text_input("API key (optional; kept in this session only)", type="password")
        key = typed or key
        st.sidebar.caption(f"API key: {'set' if key else 'not set'}"
                           + (" (not needed for a local Ollama server)" if "localhost" in url else ""))
    else:
        key = ""
        st.sidebar.caption("No model: answers are built from the retrieved evidence only; images are not interpreted.")
    vlm, backend = None, "evidence"
    if provider != "evidence":
        try:
            vlm, backend = get_vlm(provider, model, url, key, cfg.timeout, cfg.max_tokens), "vlm"
        except ValueError as ex:
            st.sidebar.error(str(ex))

    st.sidebar.subheader("Upload an OCT B-scan")
    f = st.sidebar.file_uploader("PNG or JPEG", type=["png", "jpg", "jpeg"])
    if f is None:
        st.session_state.upload = None
    elif st.session_state.get("upload_id") != f.file_id:
        try:
            st.session_state.upload = a.match_upload(validate_upload(f.getvalue(), f.name, cfg.workdir / "uploads",
                                                                     cfg.upload_max_mb))
        except UploadError as ex:
            st.session_state.upload = None
            st.sidebar.error(f"Upload rejected: {ex}")
        st.session_state.upload_id = f.file_id
    up = st.session_state.get("upload")
    if up is not None:
        st.sidebar.image(str(up.path), caption=up.filename, width="stretch")
        if up.external:
            st.sidebar.info("External scan: not linked to any patient, eye, visit or date.")
        else:
            st.sidebar.success(f"Identical to the dataset B-scan of visit {up.matched_visit} (eye {up.matched_eye}).")
        for w in up.warnings:
            st.sidebar.warning(w)
    if st.sidebar.button("Clear conversation"):
        st.session_state.turns = []
    return eye, visit, up, backend, vlm, preset


def render_answer(t: Turn):
    a = t.answer
    status_cls = "stat" if a["status"] == "answered" else "bad"
    st.markdown(badge(t.category, "stat") + badge(a["status"], status_cls) + badge(f"model: {a['backend']}", "vlm"),
                unsafe_allow_html=True)
    if a["status"] == "abstained":
        st.markdown(f"<div class='abst'><b>No answer — the retriever abstained.</b></div>", unsafe_allow_html=True)
    st.markdown(a["answer"].replace("\n", "  \n"))
    for w in a.get("warnings", []):
        st.warning(f"Check against the evidence: {w}")
    for n in t.notes:
        st.caption(n)
    st.markdown(f"<span class='note'>Eye {t.eye if t.eye is not None else '–'} · current visit {t.visit or '–'} · "
                f"{html.escape(a['evidence_used'])}</span>", unsafe_allow_html=True)


def cst_chart(a: Assistant, t: Turn):
    if t.eye is None:
        return
    v = a.visits(t.eye)
    if v.empty or v.cst.isna().all():
        return
    ev = t.evidence
    sel = {r["visit_id"]: r.get("role", "") for r in ev.get("retrieved", [])}
    cur = (ev.get("current") or {}).get("visit_id")
    v["marker"] = v.visit_id.map(lambda x: "current" if x == cur else ("retrieved" if x in sel else "other"))
    base = alt.Chart(v).encode(x=alt.X("visit_index:Q", title="visit order"),
                               y=alt.Y("cst:Q", title="CST (µm)", scale=alt.Scale(zero=False)),
                               tooltip=["visit_id", "visit_label", "week", "cst", "bcva"])
    pts = base.mark_circle(size=90).encode(color=alt.Color("marker:N", scale=alt.Scale(
        domain=["current", "retrieved", "other"], range=["#dc2626", "#7c3aed", "#94a3b8"]), title=None))
    st.altair_chart(base.mark_line(color="#94a3b8") + pts, width="stretch")


def render_evidence(a: Assistant, t: Turn):
    ev, u = t.evidence, t.unified
    st.markdown("#### How the question was handled")
    st.markdown(badge(f"route: {t.route}", "stat") + badge(f"category: {t.category}", "stat") +
                badge(f"status: {ev['status']}", "stat" if ev["status"] == "retrieved" else "bad"), unsafe_allow_html=True)
    if ev.get("interpreted_as"):
        st.caption(f"Interpreted as: “{ev['interpreted_as']}”")
    if ev.get("ranking_basis"):
        st.caption(ev["ranking_basis"])
    for c in ev.get("components", []):
        name, cls = SOURCE_STYLE.get(c["name"], (c["name"], "stat"))
        st.markdown(badge(name, cls) + html.escape(f"{c['status']}: {c['explanation'] or ''}"), unsafe_allow_html=True)
    if ev["status"] != "retrieved":
        st.markdown(f"<div class='abst'><b>Abstained ({html.escape(ev['status'])})</b>: "
                    f"{html.escape(ev.get('reason') or '')}<br>{html.escape(ev.get('guidance') or '')}</div>",
                    unsafe_allow_html=True)
    for d in ev.get("disclosures", []):
        st.markdown(f"<div class='disc'>{html.escape(d)}</div>", unsafe_allow_html=True)
    if ev.get("facts"):
        st.markdown(badge("metadata", "meta") + "**Computed from recorded values**", unsafe_allow_html=True)
        for f in ev["facts"]:
            st.markdown(f"- {f}")
    if ev.get("retrieved"):
        st.markdown("#### Retrieved visits")
        st.dataframe(pd.DataFrame([dict(rank=r["rank"], role=r["role"], selected_by=", ".join(r.get("sources", [])),
                                         visit=r["visit_id"], label=r["visit_label"], week=r["week"], cst_um=r["cst"],
                                         bcva=r["bcva"], image_cosine=r["image_cosine"],
                                         scan_sent=r.get("image_attached", False)) for r in ev["retrieved"]]),
                     hide_index=True, width="stretch")
    if t.eye is not None:
        st.markdown("#### CST timeline")
        cst_chart(a, t)
    if ev.get("images"):
        st.markdown("#### OCT scans given to the model")
        cols = st.columns(min(len(ev["images"]), 2))
        cos = {r["visit_id"]: r.get("image_cosine") for r in ev.get("retrieved", [])}
        for k, im in enumerate(ev["images"]):
            c = cos.get(im["visit_id"])
            cols[k % len(cols)].image(im["path"], width="stretch",
                                      caption=f"{A.image_label(im).removeprefix('B-scan: ')}"
                                              + (f" · cosine {c:.3f}" if c is not None else ""))
    if ev.get("biomarkers"):
        st.markdown(badge("dataset labels", "lab") + "**OLIVES finding labels** (first and last visit only)",
                    unsafe_allow_html=True)
        for b in ev["biomarkers"]:
            st.markdown(f"- {A._fmt_labels(b)}")
    if t.answer.get("backend") not in ("none", "evidence") and ev.get("images"):
        st.markdown(badge("model observation", "vlm") + "Statements about what is *visible* are the model's own, "
                    "unvalidated observations of single B-scans.", unsafe_allow_html=True)
    with st.expander("Evidence text sent to the model"):
        st.code(A.evidence_text(ev), language=None)
    if u is not None and u.primary is not None:
        with st.expander("All eligible visits: metadata / image / final ranks"):
            st.dataframe(u.primary.table(), hide_index=True, width="stretch")


def main():
    st.set_page_config(page_title="OCT Assistant", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.session_state.setdefault("turns", [])
    a = get_assistant()
    eye, visit, up, backend, vlm, preset = sidebar(a)
    a.vlm = vlm

    st.markdown(f"### Eye {eye} · current visit **{visit}**")
    st.markdown(f"<div class='disc'>{html.escape(A.DISCLAIMER)}</div>", unsafe_allow_html=True)
    chat, panel = st.columns([3, 2], gap="large")

    with chat:
        for t in st.session_state.turns:
            with st.chat_message("user"):
                st.markdown(t.question)
                if t.upload is not None and t.route in ("scan_visual", "upload_similarity"):
                    st.image(str(t.upload.path), width=220, caption="uploaded scan")
            with st.chat_message("assistant"):
                render_answer(t)
        st.caption("Suggested questions")
        cols = st.columns(2)
        clicked = None
        for k, s in enumerate(SUGGESTIONS):
            if cols[k % 2].button(s, key=f"sug_{k}", width="stretch"):
                clicked = s
        q = st.chat_input("Ask about this eye's visit history or the uploaded scan ...") or clicked
        if q:
            prev = st.session_state.turns[-1] if st.session_state.turns else None
            try:
                with st.spinner("Retrieving evidence" + (" and asking the model ..." if backend != "evidence" else " ...")):
                    t = a.ask(q, eye, visit, up, backend=backend, previous=prev)
                st.session_state.turns.append(t)
                st.rerun()
            except ValueError as ex:
                st.error(str(ex))

    with panel:
        st.markdown("### Evidence")
        turns = st.session_state.turns
        if not turns:
            st.info("Ask a question to see the retrieved evidence, the scans given to the model and the disclosures.")
        else:
            k = st.selectbox("Turn", list(range(len(turns)))[::-1], format_func=lambda i: f"{i + 1}. {turns[i].question}")
            render_evidence(a, turns[k])


if __name__ == "__main__":
    main()
