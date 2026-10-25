# How to publish NPOV-Onset (your logins required)

Everything is prepared. These are the only steps that need **your** accounts — I can't run
the login/publish actions for you. Do them in this order.

## 1. Tag a GitHub release (gives Zenodo something to archive)

```bash
cd npov-drift-detector
git tag -a dataset-v1.0 -m "NPOV-Onset dataset v1.0"
git push origin dataset-v1.0
```
Then on GitHub: **Releases → Draft a new release → choose tag `dataset-v1.0` → Publish**.

## 2. Mint a Zenodo DOI

1. Sign in at <https://zenodo.org> with your GitHub account.
2. **Account → GitHub**, flip the switch **ON** for `mahi01agarwal/npov-drift-detector`.
3. Zenodo archives the release from step 1 automatically and issues a DOI.
   (If you already published the release before flipping the switch, just cut a new release,
   e.g. `dataset-v1.0.1` — Zenodo only sees releases created after the switch is on.)
4. `.zenodo.json` at the repo root fills in the title, authors, license, and keywords, so
   the deposit is complete on arrival. Add your ORCID in Zenodo's web form if you have one.
5. Put the **version DOI** badge in the repo README (Zenodo shows the Markdown snippet).

*Manual alternative (no GitHub link):* on Zenodo, **New upload → Dataset**, drag the three
CSVs + `DATASHEET.md`, and the fields are in `.zenodo.json` to copy across.

## 3. Publish the Hugging Face Dataset

The upload-ready bundle is `dataset/huggingface/` (card + the three CSVs).

```bash
pip install -U huggingface_hub
huggingface-cli login            # paste a WRITE token from hf.co/settings/tokens

huggingface-cli upload <your-namespace>/npov-onset dataset/huggingface . \
  --repo-type dataset
```
Replace `<your-namespace>` with your HF username/org. On first upload the repo is created.
Then:
- Open `https://huggingface.co/datasets/<your-namespace>/npov-onset` — the **Dataset
  Viewer** renders all three configs (events / controls / localisation_attempts) in-browser.
- Edit the card's `load_dataset("<your-namespace>/npov-onset", ...)` line to your namespace
  (or I can, once you tell me the namespace).

## 4. Cross-link everything

- Repo README: add the Zenodo DOI badge + an HF dataset link.
- HF card: it already points back to the repo and the datasheet.
- (Later) the Phabricator T428896 update and the paper cite the DOI.

---

### What I prepared for you
- `dataset/DATASHEET.md` — full datasheet (Gebru et al. format), real counts and a per-field dictionary.
- `CITATION.cff` (repo root) — GitHub shows a "Cite this repository" button.
- `.zenodo.json` (repo root) — auto-fills the Zenodo deposit.
- `dataset/huggingface/` — ready-to-upload HF bundle (dataset card + CSVs).
- `dataset/huggingface/load_example.py` — minimal loader + matched-pair reconstruction.
