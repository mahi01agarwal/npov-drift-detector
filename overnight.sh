#!/bin/bash
cd /Users/mahimaagarwal/Desktop/npov
H=mahima@10.10.16.12
step(){ echo; echo ">>> [$(date '+%m-%d %H:%M')] $*"; }
echo "=== OVERNIGHT START $(date) ==="

step "1/5 waiting for control expansion to finish"
while pgrep -f build_controls_expand >/dev/null 2>&1; do sleep 60; done
echo "expansion done. controls=$(($(wc -l < npov_data/full_npov_controls_en.csv)-1)) (was 481)"

step "2/5 fetching expanded tag-time text (retry up to 6x; cached/resumable)"
ok=0
for a in 1 2 3 4 5 6; do
  python3 -u build_text_features_expanded.py && { ok=1; break; }
  echo "  attempt $a failed; retry in 45s"; sleep 45
done
echo "text-fetch ok=$ok | expanded articles=$(wc -l < npov_data/nb3_sentences_expanded.jsonl 2>/dev/null || echo 0)"

step "3/5 building expanded clf training data"
python3 -u build_clf_data_expanded.py

step "4/5 retraining RoBERTa (expanded) on dev-4"
scp -o ConnectTimeout=10 npov_data/nb_clf_train_expanded.jsonl $H:/tmp/mahima_npov/npov_data/ && \
ssh -o ConnectTimeout=10 -o ServerAliveInterval=20 $H 'cd /tmp/mahima_npov && . venv/bin/activate && export HF_HOME=/tmp/mahima_npov/hf && python -u train_clf.py --mode expanded --train-file nb_clf_train_expanded.jsonl --epochs 3 > clf_expanded.log 2>&1; echo DEV4_EXIT=$?; tail -18 clf_expanded.log'
scp -o ConnectTimeout=10 $H:/tmp/mahima_npov/npov_data/nb_clf_curve_expanded.csv npov_data/ 2>/dev/null
scp -o ConnectTimeout=10 $H:/tmp/mahima_npov/npov_data/nb_clf_leadtime_preds.csv npov_data/nb_clf_leadtime_preds_expanded.csv 2>/dev/null
scp -o ConnectTimeout=10 $H:/tmp/mahima_npov/clf_expanded.log . 2>/dev/null

step "5/5 expanded interpretation + OVERNIGHT_SUMMARY.md"
python3 -u run_expanded_interp.py

echo; echo "=== OVERNIGHT DONE $(date) ==="
