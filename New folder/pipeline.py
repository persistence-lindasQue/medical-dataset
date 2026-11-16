#!/usr/bin/env python3
"""PDF -> guarded -> CuratorKIT -> merged train/test dataset.

Stages
  1. pdf_guard   scan PDFs for hidden/suspicious text; quarantine flagged files
  2. curate      run CuratorKIT on each clean PDF (one run per paper)
  3. merge       combine the per-paper exports, tag each row with its source
                 paper, and split train/test BY PAPER so no paper is on both sides

Usage
  python pipeline.py --pdfs pdfs --out output
  python pipeline.py --pdfs pdfs --out output --merge-only      # skip stages 1-2
  python pipeline.py --pdfs pdfs --out output --skip-curate     # guard only

Needs for stage 2:  pip install "curatorkit[pdf,generation,hygiene]"
and an LLM: export COHERE_API_KEY=...  (or point llm_model at a local endpoint).

NOTE: CuratorConfig names below (dataset, llm_model, generation_task,
num_questions, hallucination_threshold, reward_threshold, export_formats,
output_dir) come from the CuratorKIT 0.1.20 README; the hygiene options come
from its configuration reference. The curate step has not been run against the
installed library yet.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# --- edit these --------------------------------------------------------------
LLM_MODEL = "cohere_chat/c4ai-aya-expanse-32b"   # generator (needs COHERE_API_KEY)
JUDGE_MODEL = "cohere_chat/c4ai-aya-expanse-8b"  # optional; defaults to LLM_MODEL
NUM_QUESTIONS = 3                                # questions per source chunk
GROUNDING_THRESHOLD = 0.7                        # answer must be supported by the source
REWARD_THRESHOLD = 0.7                           # LLM-judge quality gate
TEST_FRACTION = 0.2                              # share of PAPERS held out for testing
SEED = 13

# Hygiene stage, from docs/reference/configuration.md ("Data hygiene").
# Runs after cleaning, before generation: SecretsGate -> PIIPseudonymizer -> ToxicityGate.
def _hygiene_opts() -> dict:
    from curatorkit.hygiene.pii import ENTITY_TYPES_CLINICAL  # medical/legal entity set
    return {
        "secrets_gate": True,                       # reject samples with keys/tokens
        "pii_pseudonymize": False,                   # replace PII with consistent fakes
        "pii_entity_types": ENTITY_TYPES_CLINICAL,
        "toxicity_gate": True,                      # Detoxify classifier gate
        # "toxicity_detoxify_model": "multilingual",  # use for non-English text
    }

EXPORT = "sft_alpaca.jsonl"  # written when export_formats includes "alpaca"


def run_guard(pdf_dir: Path, quarantine: Path, report: Path) -> int:
    cmd = [sys.executable, str(HERE / "pdf_guard.py"), str(pdf_dir),
           "--quarantine", str(quarantine), "--report", str(report)]
    print("[1/3] pdf_guard:", " ".join(cmd))
    # exit 1 just means something was quarantined; only 2 is a real failure
    rc = subprocess.run(cmd).returncode
    if rc == 2:
        sys.exit("pdf_guard failed")
    return rc


def curate_one(pdf: Path, out_dir: Path) -> bool:
    from curatorkit import Curator, CuratorConfig  # imported late: heavy dependency

    cfg = CuratorConfig(
        dataset=str(pdf),
        llm_model=LLM_MODEL,
        judge_llm_model=JUDGE_MODEL,
        generation_task="qa",
        llm_concurrency=2,
        judge_concurrency=2,
        llm_max_retries=6,
        judge_llm_max_retries=6,
        num_questions=NUM_QUESTIONS,
        hallucination_threshold=GROUNDING_THRESHOLD,
        reward_threshold=REWARD_THRESHOLD,
        export_formats=["alpaca"],
        output_dir=str(out_dir),
        **_hygiene_opts(),
    )
    try:
        result = Curator(cfg).run()
        result.print_summary()
        return True
    except Exception as exc:
        print(f"  curation failed for {pdf.name}: {type(exc).__name__}: {exc}")
        return False


def curate_all(pdf_dir: Path, out: Path) -> None:
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    print(f"[2/3] curating {len(pdfs)} clean PDF(s)")
    for pdf in pdfs:
        target = out / "per_paper" / pdf.stem
        if (target / EXPORT).exists():
            print(f"  skip {pdf.name} (already curated)")
            continue
        print(f"  {pdf.name}")
        curate_one(pdf, target)


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def merge(out: Path) -> dict:
    """Combine per-paper exports and split train/test by paper."""
    per_paper = out / "per_paper"
    papers = sorted(p for p in per_paper.glob("*") if (p / EXPORT).exists())
    if not papers:
        sys.exit(f"no curated papers found under {per_paper}")

    rng = random.Random(SEED)
    order = papers[:]
    rng.shuffle(order)
    n_test = max(1, round(len(order) * TEST_FRACTION)) if len(order) > 1 else 0
    test_set = {p.name for p in order[:n_test]}

    splits: dict[str, list[dict]] = {"train": [], "test": []}
    for p in papers:
        rows = read_jsonl(p / EXPORT)
        side = "test" if p.name in test_set else "train"
        for row in rows:
            row["source"] = p.name
            splits[side].append(row)

    final = out / "final"
    final.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        rng.shuffle(rows)
        with (final / f"{name}.jsonl").open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "papers_total": len(papers),
        "papers_test": sorted(test_set),
        "rows_train": len(splits["train"]),
        "rows_test": len(splits["test"]),
        "seed": SEED,
    }
    (final / "split_summary.json").write_text(json.dumps(summary, indent=2))
    print("[3/3] merged:", json.dumps({k: v for k, v in summary.items() if k != "papers_test"}))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pdfs", type=Path, default=Path("pdfs"))
    ap.add_argument("--out", type=Path, default=Path("output"))
    ap.add_argument("--quarantine", type=Path, default=Path("quarantine"))
    ap.add_argument("--skip-curate", action="store_true")
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if not args.merge_only:
        run_guard(args.pdfs, args.quarantine, args.out / "guard_report.json")
        if not args.skip_curate:
            curate_all(args.pdfs, args.out)
    if not args.skip_curate:
        merge(args.out)


if __name__ == "__main__":
    main()

