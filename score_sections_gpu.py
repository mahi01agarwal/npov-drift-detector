#!/usr/bin/env python3
"""
Notebook 4b, stage B  —  PER-SECTION neural subjectivity.  *** GPU stage ***

NB4 found WHOLE-ARTICLE subjectivity is flat (no drift) and the volume check found no
one-section bloat. This is the sharper due-weight test: even if the article overall is
neutral, does ONE section carry the tilt, and does that worst section get worse / the
spread across sections widen as the human tag approaches?

Reads nb4_sections.jsonl (per-section sentence lists at lags {0,90,365}, from
build_sections.py). Scores each section's sentences with the same WNC-trained
subjective/neutral BERT as NB3/NB4, then aggregates PER ARTICLE to section-shape
features (not just a pooled mean):

  sec_worst_share  the single most-subjective section's share of subjective sentences
                   (the "one section is a soapbox" signal — the due-weight worst case)
  sec_dispersion   std of per-section subjective-share across sections
                   (one hot section among neutral ones => high dispersion)
  sec_p90          90th-percentile section subjective-share (robust worst-section)
  sec_range        max - min section subjective-share
  whole_sub_share  pooled over all sentences — reproduces NB4's whole-article number
                   (sanity: section features should move even when this stays flat)

A section must have >= --min-sents sentences to get a per-section estimate (short
sections are noise); dispersion/worst need >= 2 such sections.

Device auto-selects cuda -> mps -> cpu. Carries lag + revid through for the curve.

Output: npov_data/nb4_section_neural.csv   (consumed by run_nb4_sections.py)
"""
import argparse, json, time, sys

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sections", default="npov_data/nb4_sections.jsonl")
    p.add_argument("--out",      default="npov_data/nb4_section_neural.csv")
    p.add_argument("--model",    default="cffl/bert-base-styleclassification-subjective-neutral")
    p.add_argument("--batch",    type=int, default=96)
    p.add_argument("--min-sents", type=int, default=3,
                   help="a section needs this many sentences to get a per-section score")
    p.add_argument("--max-sec-sents", type=int, default=150,
                   help="cap sentences scored per section (longest-first) to bound cost")
    p.add_argument("--subjective-label", default="SUBJECTIVE")
    return p.parse_args()

def main():
    args = parse_args()
    import numpy as np, pandas as pd, torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    if torch.cuda.is_available():
        dev = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"
        print("WARNING: no GPU visible (no CUDA, no MPS) — this will be slow.", file=sys.stderr)
    print("device:", dev, "| model:", args.model)

    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModelForSequenceClassification.from_pretrained(args.model).to(dev).eval()

    id2label = {int(k): v for k, v in mdl.config.id2label.items()}
    want = args.subjective_label
    sub_idx = None
    for i, lab in id2label.items():
        if lab.upper() == want.upper() or ("LABEL_%d" % i) == want.upper():
            sub_idx = i
    if sub_idx is None:
        sub_idx = max(id2label)
        print("could not match label %r; using %d (%s)" % (want, sub_idx, id2label[sub_idx]),
              file=sys.stderr)
    print("scoring class:", sub_idx, id2label[sub_idx])

    arts = []
    with open(args.sections) as f:
        for line in f:
            arts.append(json.loads(line))
    print("article-snapshots:", len(arts))

    # Dynamic per-batch padding — the exact pattern NB4's score_bias_gpu.py ran at
    # proven MPS throughput (4210 articles in 3.4h). (padding="max_length" HANGS on
    # torch 2.7.0 + MPS here, so do NOT use it.)
    @torch.no_grad()
    def score_batch(batch):
        enc = tok(batch, truncation=True, max_length=128, padding=True,
                  return_tensors="pt").to(dev)
        probs = torch.softmax(mdl(**enc).logits, dim=-1)[:, sub_idx]
        return probs.float().cpu().numpy()

    def score_many(sents):
        """Score a flat sentence list. Sort by length so consecutive batches share a
        shape (fewer MPS recompiles), then map results back to the original order."""
        n = len(sents)
        if n == 0:
            return np.array([])
        order = sorted(range(n), key=lambda i: len(sents[i]))
        buf = [sents[i] for i in order]
        res = np.concatenate([score_batch(buf[i:i + args.batch])
                              for i in range(0, n, args.batch)])
        out = np.empty(n, dtype=float)
        out[np.array(order)] = res
        return out

    rows, t0, done = [], time.time(), 0
    for a in arts:
        base = {"title": a["title"], "y": a["y"], "pair": a["pair"],
                "lag": a.get("lag"), "revid": a.get("revid")}
        # flatten every section's (capped) sentences into ONE list, tracking section id,
        # and score the whole article in one grouped pass (NB4's proven throughput).
        flat, sec_ids = [], []
        secs = a.get("sections") or []
        for si, s in enumerate(secs):
            sents = s.get("sentences") or []
            if len(sents) > args.max_sec_sents:
                sents = sorted(sents, key=len, reverse=True)[:args.max_sec_sents]
            flat.extend(sents); sec_ids.extend([si] * len(sents))
        sec_shares = []
        if flat:
            ps_all = score_many(flat)
            sid = np.array(sec_ids)
            for si in range(len(secs)):
                m = sid == si
                if m.sum() >= args.min_sents:                 # per-section estimate only if enough
                    sec_shares.append(float((ps_all[m] >= 0.5).mean()))
            base["whole_sub_share"] = float((ps_all >= 0.5).mean())
            base["n_sents_total"]   = int(len(ps_all))
        else:
            base["whole_sub_share"] = np.nan; base["n_sents_total"] = 0
        ss = np.array(sec_shares)
        base["n_sec_scored"] = int(len(ss))
        if len(ss) >= 2:
            base["sec_worst_share"] = float(ss.max())
            base["sec_dispersion"]  = float(ss.std())           # population std
            base["sec_p90"]         = float(np.percentile(ss, 90))
            base["sec_range"]       = float(ss.max() - ss.min())
            base["sec_mean_share"]  = float(ss.mean())
        else:
            for k in ("sec_worst_share", "sec_dispersion", "sec_p90", "sec_range", "sec_mean_share"):
                base[k] = np.nan
        rows.append(base)
        done += 1
        if done % 50 == 0 or done == len(arts):
            print("  %4d/%d  %5.0fs" % (done, len(arts), time.time() - t0))

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("wrote", args.out)

if __name__ == "__main__":
    main()
