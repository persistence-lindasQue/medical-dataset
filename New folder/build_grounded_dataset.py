#!/usr/bin/env python3
"""Turn pipeline output into a grounded-answer dataset (Problem statement 3).

Reads  output/final/{train,test}.jsonl   (rows: instruction=question,
                                          input=source passage, output=answer,
                                          source=paper id)
Writes output/grounded/{train,test}.jsonl and stats.json

Every row becomes a prompt of the form:
    instruction: "<guidance> Question: <question>"
    input:       the passage
    output:      the answer  (answerable)   OR   an abstention (unanswerable)

Unanswerable rows are built by pairing a question with a passage from a
DIFFERENT paper in the SAME split (so test questions never see train passages).
Among a few random candidates, the passage with the lowest word overlap with the
question is chosen, to reduce the chance it secretly contains the answer.

This is a heuristic. Spot-check a sample of unanswerable rows by hand and report
what you found; two fertility papers can legitimately cover the same fact.

Usage:
  python build_grounded_dataset.py --data output/final --out output/grounded
  python build_grounded_dataset.py --unanswerable-ratio 0.25 --candidates 8
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

GUIDANCE = (
    "Answer the question using only the passage provided. "
    "If the passage does not contain the answer, say so instead of guessing."
)

ABSTAIN = [
    "The provided passage does not contain the answer to this question.",
    "I can't answer that from the passage provided; it doesn't address this.",
    "The passage doesn't say. I don't want to guess beyond what it states.",
    "This question isn't answered in the text provided.",
]

STOP = set(
    "a an the of and or to in on for with by is are was were be been it this that "
    "these those as at from which what who how why when does do did can could may "
    "might should would than then not no yes into about between during after before "
    "their there which also such more most other".split()
)


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP and len(w) > 2}


def overlap(question: str, passage: str, pw: set[str] | None = None) -> float:
    q = words(question)
    if not q:
        return 0.0
    p = pw if pw is not None else words(passage)
    return len(q & p) / len(q)


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def clean(row: dict) -> bool:
    return all(str(row.get(k, "")).strip() for k in ("instruction", "input", "output", "source"))


def make_row(question: str, passage: str, answer: str, source: str,
             answerable: bool, passage_source: str) -> dict:
    return {
        "instruction": f"{GUIDANCE}\n\nQuestion: {question}",
        "input": passage,
        "output": answer,
        "answerable": answerable,
        "question": question,
        "source": source,                 # paper the question came from
        "passage_source": passage_source,  # paper the passage came from
    }


def build_split(rows: list[dict], ratio: float, candidates: int, rng: random.Random,
                name: str, extra_pool: list[dict] | None = None) -> tuple[list[dict], dict]:
    rows = [r for r in rows if clean(r)]
    by_paper: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_paper[r["source"]].append(r)

    out = [make_row(r["instruction"], r["input"], r["output"], r["source"], True, r["source"])
           for r in rows]

    n_unans = round(len(rows) * ratio / (1 - ratio)) if ratio < 1 else 0
    passages = [(r["source"], r["input"], words(r["input"])) for r in rows]
    # Optional distractor passages borrowed from the other split (small datasets only).
    if extra_pool:
        passages += [(r["source"], r["input"], words(r["input"]))
                     for r in extra_pool if clean(r)]
    if len({p[0] for p in passages}) < 2:
        sys.exit(
            f"{name}: need passages from at least 2 different papers to build "
            f"unanswerable examples, found {len(by_paper)}. Add more papers, raise "
            f"TEST_FRACTION in pipeline.py, or pass --cross-split-distractors."
        )
    made = 0
    overlaps: list[float] = []
    picks = rng.sample(rows, min(n_unans, len(rows))) if n_unans <= len(rows) else \
        [rng.choice(rows) for _ in range(n_unans)]
    for r in picks:
        pool = [p for p in passages if p[0] != r["source"]]
        cand = rng.sample(pool, min(candidates, len(pool)))
        src, passage, pw = min(cand, key=lambda p: overlap(r["instruction"], p[1], p[2]))
        overlaps.append(overlap(r["instruction"], passage, pw))
        out.append(make_row(r["instruction"], passage, rng.choice(ABSTAIN), r["source"],
                            False, src))
        made += 1

    rng.shuffle(out)
    stats = {
        "papers": len(by_paper),
        "answerable": len(rows),
        "unanswerable": made,
        "unanswerable_share": round(made / max(1, len(out)), 3),
        "mean_question_passage_overlap_unanswerable":
            round(sum(overlaps) / len(overlaps), 3) if overlaps else None,
    }
    return out, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, default=Path("output/final"))
    ap.add_argument("--out", type=Path, default=Path("output/grounded"))
    ap.add_argument("--unanswerable-ratio", type=float, default=0.25,
                    help="share of the final dataset that is unanswerable (default 0.25)")
    ap.add_argument("--candidates", type=int, default=8,
                    help="random passages considered per unanswerable pair (default 8)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--cross-split-distractors", action="store_true",
                    help="if a split has fewer than 2 papers, borrow unanswerable-example "
                         "passages from the other split (answerable rows are never shared)")
    args = ap.parse_args()

    if not 0 <= args.unanswerable_ratio < 1:
        sys.exit("--unanswerable-ratio must be in [0, 1)")
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    data = {}
    for split in ("train", "test"):
        src = args.data / f"{split}.jsonl"
        if not src.exists():
            sys.exit(f"missing {src}; run pipeline.py first")
        data[split] = read(src)

    all_stats = {}
    for split, other in (("train", "test"), ("test", "train")):
        n_papers = len({r.get("source") for r in data[split] if clean(r)})
        # Only borrow when a split is too small; train distractors must never come from test.
        borrow = args.cross_split_distractors and n_papers < 2 and split == "test"
        rows, stats = build_split(data[split], args.unanswerable_ratio, args.candidates,
                                  rng, split, extra_pool=data[other] if borrow else None)
        stats["borrowed_distractors_from_other_split"] = bool(borrow)
        with (args.out / f"{split}.jsonl").open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        all_stats[split] = stats
        print(f"{split}: {stats}")

    all_stats["settings"] = {"unanswerable_ratio": args.unanswerable_ratio,
                             "candidates": args.candidates, "seed": args.seed}
    (args.out / "stats.json").write_text(json.dumps(all_stats, indent=2), encoding="utf-8")
    print(f"wrote {args.out}/train.jsonl, test.jsonl, stats.json")


if __name__ == "__main__":
    main()
