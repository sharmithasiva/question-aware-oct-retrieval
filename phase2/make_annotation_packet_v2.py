"""Build BLINDED Tier B annotation packets, v2 (protocol: phase2/annotations/ANNOTATION_PROTOCOL_v2.md).

One packet per category (B1 overall_appearance, B2 fluid_similarity), each with its own seeded query order, opaque
query codes, candidate codes and image names. Images are re-encoded (lossless WebP) so their bytes cannot be hashed
back to repository files. Annotators see only: the question, the rubric, the query image and coded candidates.
Nothing about systems, scores, ranks, CST/BCVA, weeks or visit labels is written to the packet.

  phase2/annotations/v2/<set>/<B1|B2>/index.html   self-contained grading page (no network; autosaves in browser)
  phase2/annotations/v2/<set>/<B1|B2>/sheet.csv    the same rows for spreadsheet grading (all labels blank)
  phase2/annotations/v2/<set>/<B1|B2>/images/      re-encoded images under random names
  phase2/annotations/v2/<set>/MANIFEST.json        seeds, versions, SHA-256 of every file
  phase2/annotations/keys/v2/<set>_<B1|B2>_key.csv code -> visit mapping (git-ignored; annotators must not open)

Refuses to overwrite an existing packet. Usage: python phase2/make_annotation_packet_v2.py --set dev
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phase2"))
import oct_image as OI  # noqa: E402
import question_eval as QE  # noqa: E402
from freeze_rankings import git_head, sha256  # noqa: E402

VERSION = "v2"
ANN = ROOT / "phase2" / "annotations"
CATEGORIES = {"B1": "overall_appearance", "B2": "fluid_similarity"}
LABELS = ("2", "1", "0", "U", "X")
SHEET_COLUMNS = ["row_id", "packet", "query_code", "question", "query_image", "candidate_code", "candidate_image",
                 "presentation_order", "label", "rationale", "query_gradable", "annotator_id", "annotator_role",
                 "annotation_date", "notes"]
RUBRIC = {
    "B1": [("2", "Highly similar", "Same overall retinal profile: foveal contour, thickness profile, presence/size/"
                                   "position of dark (hypo-reflective) spaces closely match."),
           ("1", "Partially similar", "Same main pattern but clearly different extent (e.g. smaller cysts, flatter "
                                      "contour)."),
           ("0", "Not similar", "Different overall profile (e.g. cystic vs dry fovea)."),
           ("U", "Uncertain", "You cannot decide between grades."),
           ("X", "Ungradable", "Candidate image cannot be judged (artefact, signal loss, off-centre or cut-off).")],
    "B2": [("2", "Highly similar", "Same fluid status (IRF/SRF present or absent), same location (centre vs "
                                   "off-centre) and similar extent; similar diffuse thickening."),
           ("1", "Partially similar", "Same fluid type and location but clearly different extent, or same "
                                      "thickening without matching fluid."),
           ("0", "Not similar", "Different fluid status or clearly different location."),
           ("U", "Uncertain", "You cannot decide between grades (e.g. borderline small cyst)."),
           ("X", "Ungradable", "Image quality insufficient to assess fluid.")],
}
WHO = {"B1": "Visual-similarity task. Any annotator trained on this rubric.",
       "B2": "Clinical task. A trained OCT grader (ophthalmologist, optometrist or certified grader) is required; "
             "non-expert grades are a workflow pilot only."}
IGNORE = "Ignore brightness, speckle noise, tilt/offset, the '200 µm' scale bar and borders."


def seed(*parts) -> int:
    return int(hashlib.sha256(":".join(map(str, ("tierB-packet", VERSION) + parts)).encode()).hexdigest()[:8], 16)


def build_category(set_name: str, short: str, qs: pd.DataFrame, img_src: dict, out: Path) -> tuple[list, list]:
    cat = CATEGORIES[short]
    q = qs[qs.category == cat].reset_index(drop=True)
    rng = np.random.default_rng(seed(set_name, short, "queries"))
    q = q.iloc[rng.permutation(len(q))].reset_index(drop=True)
    (out / "images").mkdir(parents=True)
    names: dict[str, str] = {}
    name_rng = np.random.default_rng(seed(set_name, short, "image-names"))

    def image_for(visit: str) -> str:
        if visit not in names:
            n = "".join(name_rng.choice(list("abcdefghjkmnpqrstuvwxyz23456789"), 10)) + ".webp"
            Image.open(img_src[visit]).convert("L").save(out / "images" / n, "WEBP", lossless=True)
            names[visit] = n
        return names[visit]

    sheet, key = [], []
    for i, r in enumerate(q.itertuples()):
        qcode = f"{short}-Q{i + 1:02d}"
        cands = r.candidate_visit_ids.split(";")
        order = np.random.default_rng(seed(set_name, short, r.query_visit_id)).permutation(len(cands))
        qimg = image_for(r.query_visit_id)
        for j, k in enumerate(order):
            v, code = cands[k], f"C{j + 1:02d}"
            cimg = image_for(v)
            sheet.append(dict(row_id=f"{qcode}::{code}", packet=f"{set_name}-{short}-{VERSION}", query_code=qcode,
                              question=r.question, query_image=f"images/{qimg}", candidate_code=code,
                              candidate_image=f"images/{cimg}", presentation_order=j + 1, label="", rationale="",
                              query_gradable="", annotator_id="", annotator_role="", annotation_date="", notes=""))
            key.append(dict(packet=f"{set_name}-{short}-{VERSION}", category=cat, query_code=qcode,
                            query_id=r.query_id, query_visit_id=r.query_visit_id, eye_id=r.eye_id,
                            patient_id=r.patient_id, candidate_code=code, candidate_visit_id=v,
                            candidate_position=int(k), query_image=qimg, candidate_image=cimg))
    return sheet, key


def write_html(out: Path, short: str, set_name: str, sheet: list) -> None:
    queries = {}
    for row in sheet:
        qd = queries.setdefault(row["query_code"], dict(code=row["query_code"], image=row["query_image"],
                                                        question=row["question"], cands=[]))
        qd["cands"].append(dict(code=row["candidate_code"], image=row["candidate_image"]))
    data = dict(packet=f"{set_name}-{short}-{VERSION}", short=short, who=WHO[short], ignore=IGNORE,
                labels=list(LABELS), rubric=RUBRIC[short], queries=list(queries.values()), columns=SHEET_COLUMNS)
    page = HTML.replace("__TITLE__", html.escape(f"Annotation {data['packet']}")).replace(
        "__DATA__", json.dumps(data).replace("</", "<\\/"))
    (out / "index.html").write_text(page, encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=list(QE.SETS), required=True)
    ap.add_argument("--confirm-heldout-freeze", action="store_true")
    a = ap.parse_args(argv)
    if a.set == "heldout" and not a.confirm_heldout_freeze:
        ap.error("held-out packets are generated only after the dev choices are logged (protocol v2 §6)")
    base = ANN / VERSION / a.set
    if base.exists():
        raise SystemExit(f"refusing to overwrite existing packet {base}")
    cfg = QE.SETS[a.set]
    episodes = pd.read_csv(ROOT / "data/processed/episodes.csv")
    qs = QE.build_query_set(episodes, a.set, "B")
    QE.assert_no_candidate_leakage(qs, episodes)
    man = OI.load_manifest(cfg["subset"])
    man = man[man.scan_index == 24]
    img_src = {r.visit_id: cfg["subset"] / r.local_path for r in man.itertuples()}
    src_sha = dict(zip(man.visit_id, man.sha256))
    missing = {v for c in qs.candidate_visit_ids for v in c.split(";")} | set(qs.query_visit_id)
    missing -= set(img_src)
    if missing:
        raise SystemExit(f"visits without verified images: {sorted(missing)[:5]} ...")
    keydir = ANN / "keys" / VERSION
    keydir.mkdir(parents=True, exist_ok=True)
    files = {}
    for short in CATEGORIES:
        out = base / short
        sheet, key = build_category(a.set, short, qs, img_src, out)
        pd.DataFrame(sheet, columns=SHEET_COLUMNS).to_csv(out / "sheet.csv", index=False)
        k = pd.DataFrame(key)
        k["query_source_sha256"] = k.query_visit_id.map(src_sha)
        k["candidate_source_sha256"] = k.candidate_visit_id.map(src_sha)
        kp = keydir / f"{a.set}_{short}_key.csv"
        if kp.exists():
            raise SystemExit(f"refusing to overwrite key {kp}")
        k.to_csv(kp, index=False)
        write_html(out, short, a.set, sheet)
        for p in sorted(out.rglob("*")):
            if p.is_file():
                files[str(p.relative_to(ROOT)).replace("\\", "/")] = sha256(p)
        files[str(kp.relative_to(ROOT)).replace("\\", "/")] = sha256(kp)
        print(f"{short}: {k.query_code.nunique()} queries, {len(k)} judgments, "
              f"{len(list((out / 'images').iterdir()))} images -> {out}")
    manifest = dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), version=VERSION,
                    git_head=git_head(), set=a.set, categories=CATEGORIES, labels=list(LABELS),
                    seeds={f"{s}:{p}": seed(a.set, s, p) for s in CATEGORIES for p in ("queries", "image-names")},
                    candidate_seed_rule="seed('tierB-packet','v2',set,category,query_visit_id)",
                    generator_sha256=sha256(Path(__file__)), protocol_sha256=sha256(ANN / "ANNOTATION_PROTOCOL_v2.md"),
                    files_sha256=files)
    (base / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"manifest -> {base / 'MANIFEST.json'}")
    return 0


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--ink:#1e242a;--navy:#233039;--gold:#d7ac54;--muted:#5a6470;--line:#c9cfd6;--bg:#f6f7f9;--card:#fff}
*{box-sizing:border-box}body{margin:0;font:14px/1.45 Calibri,Arial,sans-serif;color:var(--ink);background:var(--bg)}
header{position:sticky;top:0;z-index:5;background:var(--navy);color:#fff;padding:10px 16px;display:flex;gap:16px;
flex-wrap:wrap;align-items:center}header b{color:var(--gold)}header input,header select{padding:4px 6px}
main{max-width:1280px;margin:0 auto;padding:16px}.rubric{background:var(--card);border:1px solid var(--line);
border-radius:8px;padding:10px 14px;margin-bottom:14px}.rubric td{padding:2px 8px;vertical-align:top}
.nav{display:flex;gap:8px;align-items:center;margin:10px 0}.nav button{padding:6px 12px}
.query{display:grid;grid-template-columns:minmax(260px,420px) 1fr;gap:16px;align-items:start}
.qimg{position:sticky;top:64px;background:var(--card);border:2px solid var(--navy);border-radius:8px;padding:8px}
.qimg img{width:100%}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px}
.cand{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px}.cand.done{border-color:#7a9}
.cand img{width:100%;cursor:zoom-in}.opts{display:flex;gap:4px;flex-wrap:wrap;margin:6px 0}
.opts label{border:1px solid var(--line);border-radius:4px;padding:2px 7px;cursor:pointer}
.opts input{margin-right:3px}.cand textarea{width:100%;height:38px}.muted{color:var(--muted)}
#zoom{position:fixed;inset:0;background:rgba(0,0,0,.85);display:none;align-items:center;justify-content:center;z-index:9}
#zoom img{max-width:96vw;max-height:96vh}.warn{color:#a33}
</style></head><body>
<header><span><b id="pk"></b></span><label>Annotator ID <input id="aid" size="8"></label>
<label>Role <select id="role"><option value="">choose…</option><option>ophthalmologist</option><option>optometrist</option>
<option>certified_grader</option><option>trained_non_expert</option><option>researcher_non_expert</option></select></label>
<span id="prog"></span><button id="dl">Download CSV</button></header>
<main><div class="rubric"><div id="who"></div><table id="rub"></table><div class="muted" id="ign"></div>
<div class="muted">Several candidates may get 2, or none. A rationale is required for 2, 1, U and X. Progress is saved
in this browser; download the CSV when finished and send it to the study lead.</div></div>
<div class="nav"><button id="prev">◀ Previous</button><select id="jump"></select><button id="next">Next ▶</button>
<label><input type="checkbox" id="qg"> Query image NOT gradable (skip this query)</label></div>
<div class="query"><div class="qimg"><div><b>QUERY (today)</b> <span id="qc" class="muted"></span></div>
<div id="qq"></div><img id="qi" alt="query OCT"></div><div class="grid" id="grid"></div></div></main>
<div id="zoom"><img alt=""></div>
<script>
const D=__DATA__;const KEY="ann:"+D.packet;let S={};try{S=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){S={}}
S.labels=S.labels||{};S.rat=S.rat||{};S.qng=S.qng||{};let qi=0;
const $=id=>document.getElementById(id);const save=()=>{try{localStorage.setItem(KEY,JSON.stringify(S))}catch(e){}};
$("pk").textContent=D.packet;$("who").innerHTML="<b>"+D.who+"</b>";$("ign").textContent=D.ignore;
$("aid").value=S.aid||"";$("role").value=S.role||"";$("aid").oninput=e=>{S.aid=e.target.value;save()};
$("role").onchange=e=>{S.role=e.target.value;save()};
D.rubric.forEach(r=>{const tr=document.createElement("tr");tr.innerHTML="<td><b>"+r[0]+"</b></td><td><b>"+r[1]+
"</b></td><td>"+r[2]+"</td>";$("rub").appendChild(tr)});
D.queries.forEach((q,i)=>{const o=document.createElement("option");o.value=i;o.textContent=q.code;$("jump").appendChild(o)});
function done(q){if(S.qng[q.code])return true;return q.cands.every(c=>S.labels[q.code+"::"+c.code])}
function prog(){const n=D.queries.filter(done).length;$("prog").textContent=n+" / "+D.queries.length+" queries complete"}
function render(){const q=D.queries[qi];$("jump").value=qi;$("qc").textContent=q.code;$("qq").textContent=q.question;
$("qi").src=q.image;$("qg").checked=!!S.qng[q.code];const g=$("grid");g.innerHTML="";
q.cands.forEach(c=>{const id=q.code+"::"+c.code;const d=document.createElement("div");d.className="cand"+(S.labels[id]?" done":"");
const opts=D.labels.map(l=>'<label><input type="radio" name="'+id+'" value="'+l+'"'+(S.labels[id]===l?" checked":"")+
">"+l+"</label>").join("");d.innerHTML="<b>"+c.code+"</b><img src='"+c.image+"' alt='candidate "+c.code+"'>"+
'<div class="opts">'+opts+'</div><textarea placeholder="rationale (required for 2, 1, U, X)"></textarea>';
d.querySelector("textarea").value=S.rat[id]||"";d.querySelector("textarea").oninput=e=>{S.rat[id]=e.target.value;save()};
d.querySelectorAll("input").forEach(inp=>inp.onchange=e=>{S.labels[id]=e.target.value;d.classList.add("done");save();prog()});
d.querySelector("img").onclick=()=>{$("zoom").querySelector("img").src=c.image;$("zoom").style.display="flex"};
g.appendChild(d)});prog()}
$("qg").onchange=e=>{S.qng[D.queries[qi].code]=e.target.checked;save();prog()};
$("zoom").onclick=()=>$("zoom").style.display="none";
$("prev").onclick=()=>{qi=Math.max(0,qi-1);render();scrollTo(0,0)};$("next").onclick=()=>{qi=Math.min(D.queries.length-1,qi+1);render();scrollTo(0,0)};
$("jump").onchange=e=>{qi=+e.target.value;render()};
function csvq(v){v=String(v==null?"":v);return /[",\n]/.test(v)?'"'+v.replace(/"/g,'""')+'"':v}
$("dl").onclick=()=>{const today=new Date().toISOString().slice(0,10);const rows=[D.columns.join(",")];
D.queries.forEach(q=>q.cands.forEach((c,j)=>{const id=q.code+"::"+c.code;const r={row_id:id,packet:D.packet,
query_code:q.code,question:q.question,query_image:q.image,candidate_code:c.code,candidate_image:c.image,
presentation_order:j+1,label:S.qng[q.code]?"":(S.labels[id]||""),rationale:S.rat[id]||"",query_gradable:S.qng[q.code]?"no":"yes",
annotator_id:S.aid||"",annotator_role:S.role||"",annotation_date:today,notes:""};rows.push(D.columns.map(k=>csvq(r[k])).join(","))}));
const missingRat=D.queries.some(q=>!S.qng[q.code]&&q.cands.some(c=>{const id=q.code+"::"+c.code;
return S.labels[id]&&S.labels[id]!=="0"&&!(S.rat[id]||"").trim()}));
if(!S.aid||!S.role){alert("Enter annotator ID and role first.");return}
if(missingRat&&!confirm("Some 2/1/U/X labels have no rationale. Download anyway?"))return;
const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([rows.join("\n")+"\n"],{type:"text/csv"}));
a.download="completed_"+(S.aid||"annotator")+"_"+D.packet+".csv";a.click()};
render();
</script></body></html>
"""

if __name__ == "__main__":
    sys.exit(main())
