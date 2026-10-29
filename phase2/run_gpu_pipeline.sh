#!/usr/bin/env bash
# ResNet-50 pipeline for scan_subset_v1 on a machine that can reach retinaoct.com and download.pytorch.org.
# Resumable: re-run after any interruption (fetch and embedding cache pick up where they stopped).
# Usage (repository root):  bash phase2/run_gpu_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export NVIDIA_TF32_OVERRIDE=0
SUB=phase2/data/scan_subset_v1
PLAN_SHA=52a181b5a134cc697bafab21b8bf3e225a5c5754c7a2fec652fc985b56548657

echo "== 1/6 plan hash and split"
python - "$SUB/plan.csv" "$PLAN_SHA" <<'EOF'
import hashlib, sys, pandas as pd
p, want = sys.argv[1], sys.argv[2]
got = hashlib.sha256(open(p, "rb").read()).hexdigest()
plan = pd.read_csv(p)
held = {93, 74, 40, 14}
assert got == want, f"plan.csv hash {got} != {want}"
assert set(plan.split) == {"train"} and not held & set(plan.eye_id) and len(plan) == 4685, "plan restrictions violated"
print(f"plan OK: {len(plan)} scans, train only, no held-out eyes")
EOF

echo "== 2/6 fetch (dry run, then resumable download)"
python phase2/fetch_scan_subset.py --dry-run
python phase2/fetch_scan_subset.py
cat "$SUB/fetch_counts.json"

echo "== 3/6 reproduction check on the 73 saved full-history images"
python phase2/embed_cached.py --subset phase2/data/atlas_fullhist --encoder resnet50 \
  --out phase2/results/repro_check/fullhist_resnet50 --check-against phase2/results/fullhist/resnet50/embeddings.npz
python - <<'EOF'
import json, sys
c = json.load(open("phase2/results/repro_check/fullhist_resnet50/run_info.json"))["check_against"]
print(c)
if c["n_common"] != 73 or c["min_cosine"] < 0.999:
    sys.exit("STOP: reproduction check failed (need n_common == 73 and min_cosine >= 0.999)")
print("reproduction OK")
EOF

echo "== 4/6 ResNet-50 embeddings for scan_subset_v1"
python phase2/embed_cached.py --subset "$SUB" --encoder resnet50

echo "== 5/6 evaluation (central scan per visit, then 5-scan visit mean)"
for MODE in central mean; do
  OUT=phase2/results/scan_subset_v1/eval_resnet50_$MODE
  if [ -d "$OUT" ]; then echo "exists, skipping: $OUT"; continue; fi
  B1=""; [ -f phase2/annotations/keys/v2/dev_B1_key.csv ] && B1="--b1"
  python phase2/large_eval.py --subset "$SUB" --emb resnet50=phase2/results/scan_subset_v1/resnet50/embeddings.npz \
    --visit-mode "$MODE" $B1 --out "$OUT" > /dev/null
  echo "report -> $OUT/report.md"
done

echo "== 6/6 demo pages"
mkdir -p phase2/results/scan_subset_v1/demo
for EYE in 25 11 61; do
  python phase2/demo.py --subset "$SUB" --emb phase2/results/scan_subset_v1/resnet50/embeddings.npz --eye $EYE \
    --question "Which earlier visit's OCT looks most like today's?" \
    --html phase2/results/scan_subset_v1/demo/eye${EYE}_image_question.html > /dev/null
done
echo "done. Send back: $SUB/fetch_counts.json, phase2/results/repro_check/fullhist_resnet50/run_info.json,"
echo "      phase2/results/scan_subset_v1/eval_resnet50_{central,mean}/report.md"
