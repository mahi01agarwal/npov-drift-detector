# Overnight run — expanded-data classifier

_Generated 2026-10-25 09:19_

## Dataset
- Controls after expansion: **688** (was 481)
- Expanded tag-time articles fetched: **1376**
- Expanded classifier training rows: **1376** (was 962)

## RoBERTa — expanded model (trained on big non-panel set, drift eval on 421 panel)
- DiD 365→0 = **+0.0022 [-0.0009, 0.0053]** | concurrent AUC 0.677 | n=339  → no drift (CI spans 0)
- per-lag win-rate: 365d=0.688 180d=0.708 90d=0.710 30d=0.712 0d=0.720

## Lexical (TF-IDF+LogReg) — expanded, same held-out panel
- DiD 365→0 = **+0.0095 [0.0031, 0.0161]** | concurrent AUC 0.750 | train 534  → DRIFT

## Baseline (481 pairs, for comparison)
- RoBERTa temporal DiD +0.0081 [0.0023,0.0143] (AUC 0.653); lexical +0.0134 [0.0059,0.0217] (AUC 0.740)
- Verdict so far: small real LEXICAL drift; bag-of-words ≈ RoBERTa at 962 examples.

## What to check on the bigger data
- Did RoBERTa's concurrent AUC overtake lexical with more training data?
- Did RoBERTa's DiD grow relative to lexical (= non-lexical signal emerging)?
- If RoBERTa now clearly beats lexical on drift → there IS semantic drift beyond vocabulary.
