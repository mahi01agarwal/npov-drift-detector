#!/usr/bin/env python3
"""
Notebook 3, stage B  —  neural subjectivity/bias scoring.  *** GPU stage ***

This is the ONE part of the pipeline that benefits from a GPU. It reads the cleaned
sentences produced by build_text_features.py (stage A) and scores every sentence with
a transformer classifier, then aggregates to per-article features. On an RTX 4060 Ti
(16 GB) the full ~962 articles take a few minutes; on CPU it is an hour-plus.

Default model: a subjective-vs-neutral style classifier fine-tuned on the Wiki
Neutrality Corpus (Pryzant et al. 2020) — exactly the "biased wording an editor later
neutralised" signal we want. Override with --model to try another HF checkpoint.

Read-only on local files. No wiki access here. Transfer nb3_sentences.jsonl to the
box, run this, transfer nb3_neural_features.csv back.

Usage on the GPU box (under tmux):
    pip install torch transformers --quiet          # if not already present
    python3 score_bias_gpu.py \
        --sentences npov_data/nb3_sentences.jsonl \
        --out       npov_data/nb3_neural_features.csv \
        --model     cffl/bert-base-styleclassification-subjective-neutral \
        --batch 128
"""
import argparse, json, time, sys

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sentences", default="npov_data/nb3_sentences.jsonl")
    p.add_argument("--out",       default="npov_data/nb3_neural_features.csv")
    p.add_argument("--model",     default="cffl/bert-base-styleclassification-subjective-neutral")
    p.add_argument("--batch",     type=int, default=128)
    p.add_argument("--max-sents", type=int, default=400,
                   help="cap sentences scored per article (longest-first) to bound cost")
    p.add_argument("--subjective-label", default="SUBJECTIVE",
                   help="label name (or index like 'LABEL_1') treated as the biased class")
    return p.parse_args()

def main():
    args = parse_args()
    import numpy as np, pandas as pd, torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    if torch.cuda.is_available():
        dev = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        dev = "mps"                                   # Apple Silicon GPU
    else:
        dev = "cpu"
        print("WARNING: no GPU visible (no CUDA, no MPS) — this will be slow.",
              file=sys.stderr)
    print("device:", dev, "| model:", args.model)

    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModelForSequenceClassification.from_pretrained(args.model).to(dev).eval()

    # resolve which output column is the "subjective / biased" probability
    id2label = {int(k): v for k, v in mdl.config.id2label.items()}
    want = args.subjective_label
    sub_idx = None
    for i, lab in id2label.items():
        if lab.upper() == want.upper() or ("LABEL_%d" % i) == want.upper():
            sub_idx = i
    if sub_idx is None:                               # fall back to the higher index
        sub_idx = max(id2label)
        print("could not match label %r in %s; using index %d (%s)"
              % (want, id2label, sub_idx, id2label[sub_idx]), file=sys.stderr)
    print("scoring class:", sub_idx, id2label[sub_idx])

    # load articles
    arts = []
    with open(args.sentences) as f:
        for line in f:
            arts.append(json.loads(line))
    print("articles:", len(arts))

    @torch.no_grad()
    def score_batch(batch):
        enc = tok(batch, truncation=True, max_length=128, padding=True,
                  return_tensors="pt").to(dev)
        logits = mdl(**enc).logits
        probs = torch.softmax(logits, dim=-1)[:, sub_idx]
        return probs.float().cpu().numpy()

    rows, t0, done = [], time.time(), 0
    for a in arts:
        sents = a.get("sentences") or []
        if len(sents) > args.max_sents:               # longest sentences first
            sents = sorted(sents, key=len, reverse=True)[:args.max_sents]
        passthru = {k: a[k] for k in ("lag",) if k in a}   # carry lead-time label if present
        if not sents:
            rows.append({"title": a["title"], "y": a["y"], "pair": a["pair"], **passthru,
                         "neural_mean": np.nan, "neural_p75": np.nan,
                         "neural_max": np.nan, "neural_sub_share": np.nan,
                         "neural_top_mean": np.nan, "neural_n_scored": 0})
            continue
        ps = []
        for i in range(0, len(sents), args.batch):
            ps.append(score_batch(sents[i:i + args.batch]))
        ps = np.concatenate(ps)
        top = np.sort(ps)[::-1][:max(1, len(ps) // 10)]     # top decile
        rows.append({
            "title": a["title"], "y": a["y"], "pair": a["pair"], **passthru,
            "neural_mean":      float(ps.mean()),
            "neural_p75":       float(np.percentile(ps, 75)),
            "neural_max":       float(ps.max()),
            "neural_sub_share": float((ps >= 0.5).mean()),  # share of biased sentences
            "neural_top_mean":  float(top.mean()),          # how biased the worst 10% are
            "neural_n_scored":  int(len(ps)),
        })
        done += 1
        if done % 50 == 0 or done == len(arts):
            print("  %4d/%d  %5.0fs" % (done, len(arts), time.time() - t0))

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("wrote", args.out)

if __name__ == "__main__":
    main()
