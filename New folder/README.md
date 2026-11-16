# Medical PDF curation pipeline

PDF papers go in, a cleaned and guarded dataset comes out, ready for AlignTune.

```
pdfs/  ->  pdf_guard.py  ->  CuratorKIT (per paper)  ->  merge + split by paper  ->  output/final/{train,test}.jsonl
              |
              +-> quarantine/   (flagged or unreadable PDFs)
```

## Files

| File | What it does |
|---|---|
| `pdf_guard.py` | Flags hidden or suspicious text in PDFs and moves those files to `quarantine/` |
| `pipeline.py` | Runs the guard, then CuratorKIT on each clean PDF, then merges and splits |
| `tests/make_test_pdfs.py` | Builds test PDFs (one clean, one per trick) to check the guard |

## Setup

```bash
pip install pdfplumber pypdf                     # guard
pip install "curatorkit[pdf,generation,hygiene]" # curation (Python 3.11+)
export COHERE_API_KEY=...                        # or edit LLM_MODEL in pipeline.py
```

CuratorKIT's `pdf` extra installs MinerU, which is AGPL-3.0. Its own license (LSAL)
is free for research and non-commercial use only.

## Run

```bash
python pipeline.py --pdfs pdfs --out output
```

Guard only (no LLM needed): `python pdf_guard.py pdfs --dry-run`

## What the guard catches

White or near-white text, tiny text (< 4 pt), text outside the page, invisible
render mode, instruction-like phrases aimed at AI models (in page text, metadata
and annotations). It is a heuristic filter: it assumes a white page, cannot see
text hidden behind images, and its phrase list is short. "Clean" means "no known
tricks found", not "safe".

## Status

Tested:
- the guard flags all six trick types in the test set, passes the clean PDF, and reports a broken PDF as unreadable
- quarantine moving and the per-paper train/test split (no paper appears on both sides)

Not tested (CuratorKIT was not installed and package installs were unavailable):
- `curate_one` in `pipeline.py` has never been run against the real library
- `HYGIENE_OPTS` is empty: look up the secrets / PII / toxicity options in the
  CuratorKIT configuration reference and fill them in
- the merge assumes the Alpaca export is named `sft_alpaca.jsonl`, as in the README

## Before you submit

1. Only use papers whose license allows reuse (open-access with a permissive license). Keep a `licenses.csv` next to `pdfs/`.
2. Spot-check a sample of generated Q&A by hand and report the error rate.
3. Fine-tune with AlignTune on `output/final/train.jsonl`, evaluate with AuditKIT on
   `test.jsonl`, and compare against the base model. The rules require before/after metrics.
4. A useful ablation: train on guarded data versus unguarded data and report the difference.
