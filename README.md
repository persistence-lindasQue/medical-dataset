# medical-dataset
New dataset of medical research 
# Grounded fertility Q&A: a guarded curation pipeline for teaching small models to answer only from the source

**One-line claim:** We built a guarded pipeline that turns open-access fertility papers into a grounded question-answering dataset with unanswerable examples, so a small model can be trained to cite the passage or say it doesn't know. Dataset and pipeline are complete; fine-tuning and evaluation were not finished before the deadline.

## Tracks and problem statements
- Track 1 (fine-tuning and data curation)
  - Problem statement 3: Grounded answers and fewer hallucinations
  - Related: problem statement 1 (healthcare as a regulated domain)

## Abstract
[Paste the abstract from SUBMISSION.md. Keep it under 500 words and keep the Limitations paragraph.]

## Headline before → after number
**None yet.** We did not complete fine-tuning or evaluation, so there is no before/after
result. The deliverable is the curated grounded dataset and the pipeline. See "With more credits".

## Base model + method in one line
Data: CuratorKIT curation with Cohere Aya Expanse for generation and judging. Planned
fine-tune: Tiny Aya with AlignTune on the grounded set (not completed).

## Problem statement and approach
Small models answer confidently even when the source does not support the answer. We
build a dataset where each row is (passage, question, answer) and about 25% of rows are
unanswerable, with an abstaining answer. A model trained on it should answer from the
passage or decline.

## Why this problem / real use case
Fertility is a high-stakes domain where unsupported claims about success rates or
treatments can mislead people. A model that only answers from vetted documents and says
when they don't cover a question is safer to deploy in patient-information settings.

## Hackathon repos (kits) used
- **CuratorKIT** (used): PDF parsing (MinerU backend), TextCleaner, SchemaGate,
  ExactDeduplicator, SecretsGate, ToxicityGate, QAGenerationTask, RewardGate. Filtering
  counts: [fill in from output/per_paper/*, e.g. generated / passed / rejected per paper].
- **AlignTune**: not used (not completed).
- **AuditKIT**: not used (not completed).
- AgentTune, SafeTune, CircuitKIT: not used.

## How the repos connect
PDFs -> our PDF guard (hidden-text and prompt-injection scan; flagged files quarantined)
-> CuratorKIT per paper -> merge and split by paper -> unanswerable-example builder ->
Hugging Face dataset. Planned next: AlignTune fine-tune -> AuditKIT before/after.

## Models and licenses
- Generation and judging: Cohere Aya Expanse ([32B / 8B]) via the Cohere API. [License:
  check the model card, CC-BY-NC for Aya Expanse weights; used here only through the API.]
- No model was fine-tuned.

## Datasets, benchmarks and licenses
- **Source data:** [3] open-access fertility papers: [title, link, license for each].
  Only derived Q&A is published, not the PDFs.
- **Languages:** English.
- **Splits:** train [67] rows ([50] answerable, [17] unanswerable, [2] papers); test [51]
  rows ([38] answerable, [13] unanswerable, [1] paper).
- **Leak check:** split by paper, so no paper appears on both sides. We did not run an
  embedding or MinHash check. Test unanswerable rows borrow distractor passages from the
  training papers (a stated leakage risk).
- **Benchmarks:** none run.

## Metrics, and why these
Planned (not run): accuracy on answerable rows (higher is better), abstention rate on
unanswerable rows (higher), unsupported-claim rate judged by an LLM (lower). Chosen to
match problem statement 3: fewer unsupported claims and honest abstention.

## Method / experiments
- Curation settings: [paste the CuratorConfig values from pipeline.py: models, thresholds,
  num_questions=3, concurrency 2].
- Hygiene: secrets gate and toxicity gate on; **PII pseudonymization off** (blocked by a
  Windows application-control policy on the build machine).
- Unanswerable construction: each unanswerable row pairs a question with a passage from a
  different paper, chosen as the lowest word overlap among 8 random candidates.

## Results
No model results. Dataset statistics are in the table above. [Optional: 3 sample rows,
one answerable and one unanswerable.]

## What didn't work
- The PDF guard first flagged 5 of 8 real papers (white or tiny text), which were false
  positives on normal paper layouts (coloured table headers, margin stamps). We loosened it.
- Setup consumed most of the time: Python 3.14 vs MinerU, a full codespace disk, and
  Windows application control blocking litellm and presidio native files.

## Limitations
- Very small corpus ([3] papers); no clinician review of generated answers.
- Unanswerable labels are a heuristic and not hand-verified: [spot-check result: __ read,
  __ wrong].
- PII step skipped. Test abstention examples may be slightly easier because of borrowed
  training passages. No fine-tune and no evaluation.

## With more credits
- **Planned approach:** more papers (50+), hand-verified unanswerable set, fine-tune Tiny
  Aya with AlignTune, evaluate base vs tuned with AuditKIT plus a judge for unsupported claims.
- **Expected before/after:** we expect abstention rate on unanswerable rows to rise and
  unsupported-claim rate to fall; we have no measurement to back a number.



## Documentation and reproduction
See the repo README for setup (Python 3.12) and commands:
`python pipeline.py --pdfs pdfs --out output` then
`python build_grounded_dataset.py --data output/final --out output/grounded [--cross-split-distractors]`.

## Declaration of work done outside the hackathon
["None."]

## Teammates tagged
["solo".]

## LLM usage statement
Substantial portions of the code and this write-up were produced with Claude's assistance,
then run and checked by the author. [Adjust to your honest estimate.]
