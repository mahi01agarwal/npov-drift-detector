#!/usr/bin/env python3
"""
Direction axis, stage B  —  LLM-judge ONE-SIDEDNESS / due-weight score per snapshot.

Reads npov_data/nb4_direction_inputs.jsonl (from build_direction_input.py), asks an LLM
judge to rate how one-sided each snapshot is in its REPRESENTATION of viewpoints (NOT its
tone — that is the orthogonal NB3/NB4 subjectivity axis), and writes a per-snapshot feature
table. Features then feed run_nb4_direction.py (lead-time + DiD) and the NB6 watchlist.

Backends (pluggable): --backend {anthropic, openai, hf}. Key is read from the environment
(ANTHROPIC_API_KEY / OPENAI_API_KEY); this script NEVER contains a key. hf runs a local
instruct model (cuda->mps->cpu) for the dev-4 path.

Discipline carried from the rest of the project:
  * Per-revid JSON cache (npov_cache/direction/<revid>.json). Resumable; reruns are free.
  * A transient/API failure is NEVER written to the cache or the CSV as a value — only a
    successfully parsed judgment is cached. Failures are retried with backoff, then skipped.

Output: npov_data/nb4_direction.csv   (title,pair,y,lag,revid,one_sidedness,
         missing_other_side,framing_partiality,confidence,lean)
"""
import os, re, json, time, argparse, sys

ROOT = "."; OUT = ROOT + "/npov_data"
CACHE = ROOT + "/npov_cache/direction"
INP = OUT + "/nb4_direction_inputs.jsonl"
CSV = OUT + "/nb4_direction.csv"
os.makedirs(CACHE, exist_ok=True)
LOAD8 = False  # set from --load-8bit (hf backend)
LOAD4 = False  # set from --load-4bit (hf backend, nf4)

SYSTEM = (
 "You are a careful Wikipedia neutrality assessor. You judge exactly ONE property of an "
 "article snapshot: how ONE-SIDED it is in its REPRESENTATION of viewpoints and DUE WEIGHT "
 "-- whether all significant perspectives on contested or evaluative points are covered "
 "proportionally, or whether the article predominantly reflects one side through emphasis, "
 "selection, or OMISSION of relevant criticism/controversy.\n\n"
 "Judge representation and balance ONLY. Do NOT judge writing tone, how emotional, "
 "promotional, or formal the prose sounds, grammar, or citation formatting. A flatly written "
 "article can still be highly one-sided by omission; a lively one can still be balanced. Base "
 "the judgment on which viewpoints and material are present versus conspicuously missing given "
 "THIS subject.\n\n"
 "Respond with ONLY a JSON object (no prose, no code fence) with keys:\n"
 '  "one_sidedness": number 0..1 (0 = all significant viewpoints proportionally represented; '
 '1 = essentially one-sided)\n'
 '  "missing_other_side": number 0..1 (degree to which opposing views, criticism, or '
 'controversy a neutral treatment would include are absent/underweighted)\n'
 '  "framing_partiality": number 0..1 (degree to which contested/evaluative claims are stated '
 'as fact favoring one side rather than attributed or qualified)\n'
 '  "confidence": number 0..1\n'
 '  "lean": short string naming the favored side/subject, or "none"\n'
 '  "rationale": one sentence, <= 30 words'
)

def extract_json(s):
    m = re.search(r"\{.*\}", s, re.S)
    if not m: raise ValueError("no json")
    o = json.loads(m.group(0))
    out = {}
    for k in ("one_sidedness", "missing_other_side", "framing_partiality", "confidence"):
        v = float(o.get(k)); 
        out[k] = min(1.0, max(0.0, v))
    out["lean"] = str(o.get("lean", "none"))[:60]
    return out

# ---------- backends ----------
def make_judge(backend, model):
    if backend == "anthropic":
        import anthropic
        cli = anthropic.Anthropic()  # key from ANTHROPIC_API_KEY
        def ask(text):
            r = cli.messages.create(model=model, max_tokens=300, temperature=0,
                    system=SYSTEM, messages=[{"role":"user","content":"ARTICLE SNAPSHOT:\n"+text}])
            return r.content[0].text
        return ask
    if backend == "openai":
        from openai import OpenAI
        cli = OpenAI()  # key from OPENAI_API_KEY
        def ask(text):
            r = cli.chat.completions.create(model=model, temperature=0, max_tokens=300,
                    messages=[{"role":"system","content":SYSTEM},
                              {"role":"user","content":"ARTICLE SNAPSHOT:\n"+text}])
            return r.choices[0].message.content
        return ask
    if backend == "hf":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model)
        kw = dict(device_map="auto")
        if LOAD4:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        elif LOAD8:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            kw["torch_dtype"] = "auto"
        mdl = AutoModelForCausalLM.from_pretrained(model, **kw).eval()
        print("  hf model loaded on", mdl.device, "| 4bit=", LOAD4, "8bit=", LOAD8, flush=True)
        def ask(text):
            msgs=[{"role":"system","content":SYSTEM},
                  {"role":"user","content":"ARTICLE SNAPSHOT:\n"+text}]
            ids=tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(mdl.device)
            with torch.no_grad():
                out=mdl.generate(ids, max_new_tokens=300, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        return ask
    raise SystemExit("unknown backend "+backend)

def judge_cached(ask, revid, text, retries=4):
    cp = os.path.join(CACHE, "%s.json" % int(revid))
    if os.path.exists(cp):
        try: return json.load(open(cp))
        except Exception: pass
    last=None
    for a in range(retries):
        try:
            rec = extract_json(ask(text))
            json.dump(rec, open(cp, "w"))     # cache ONLY on success
            return rec
        except Exception as e:
            last=e; time.sleep(2*(a+1))
    print("  !! skip revid %s: %s" % (revid, last), flush=True)
    return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--backend", default="anthropic", choices=["anthropic","openai","hf"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--lags", default="365,90,0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--load-8bit", action="store_true", help="hf backend: load in 8-bit (bitsandbytes)")
    ap.add_argument("--load-4bit", action="store_true", help="hf backend: load in 4-bit NF4 (bitsandbytes)")
    ap.add_argument("--dry-run", action="store_true", help="print the prompt for N=limit rows, no calls")
    a=ap.parse_args()
    global LOAD8, LOAD4; LOAD8=a.load_8bit; LOAD4=a.load_4bit
    model = a.model or {"anthropic":"claude-haiku-5-5","openai":"gpt-4o-mini",
                        "hf":"Qwen/Qwen2.5-7B-Instruct"}[a.backend]
    lags=set(int(x) for x in a.lags.split(","))
    rows=[json.loads(l) for l in open(INP)]
    rows=[r for r in rows if r["lag"] in lags]
    if a.limit: rows=rows[:a.limit]
    print("backend=%s model=%s rows=%d lags=%s" % (a.backend, model, len(rows), sorted(lags)), flush=True)

    if a.dry_run:
        for r in rows[:3]:
            print("\n"+"#"*80+"\n", r["title"], "y=%d lag=%d"%(r["y"],r["lag"]))
            print("[SYSTEM]\n"+SYSTEM[:400]+"...\n[USER]\nARTICLE SNAPSHOT:\n"+r["judge_input"][:800]+"...")
        return

    ask=make_judge(a.backend, model)
    import csv
    fields=["title","pair","y","lag","revid","one_sidedness","missing_other_side",
            "framing_partiality","confidence","lean"]
    fcsv=open(CSV,"w",newline=""); w=csv.DictWriter(fcsv, fieldnames=fields); w.writeheader(); fcsv.flush()
    done=0; t0=time.time()
    for i,r in enumerate(rows):
        rec=judge_cached(ask, r["revid"], r["judge_input"])
        if rec is None: continue
        w.writerow({**{k:r[k] for k in ("title","pair","y","lag","revid")}, **rec}); fcsv.flush()
        done+=1
        if (i+1)%25==0 or i==len(rows)-1:
            el=time.time()-t0
            print("  %d/%d  ok=%d  %.2fs/it" % (i+1,len(rows),done, el/(i+1)), flush=True)
    print("wrote %d/%d judged rows -> %s" % (done,len(rows),CSV), flush=True)

if __name__=="__main__":
    main()
