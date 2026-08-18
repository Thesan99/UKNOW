# Open items

Things worth resolving before results go in a writeup or the repo is presented.
None are fixed silently in the code — they are recorded here so the behavior of
record stays intact and the decisions are yours.

## 1. Class-conditional JPEG quality (highest priority)

`04_normalize_split.py` re-encodes real at quality 100 and AI at quality 70. That
is a systematic per-class encoding difference a detector can learn instead of any
real-vs-AI signal. It plausibly inflates the in-distribution accuracy (0.996) and
confounds the in-distribution → out-of-distribution (FLUX) gap that is the
headline finding.

Diagnostic (eval-only, no retraining): re-encode the test AI at quality 100 to
match real, everything else held, and re-run `07`/eval with the saved model.
- AI recall stays high → the head learned real features; ship the number.
- AI recall drops → the head was reading compression; rebuild both classes at
  identical quality and retrain.

## 2. Split is stratified by class, not topic

In `04`, sampling is global and `stratified_split` only balances real/ai across
splits. Topic proportions are whatever survived curation, so thin topics can
nearly vanish from the test set — which makes per-topic accuracy (the 2×2 cells)
noisy on those topics. If the per-topic read matters, sample and split per topic.

## 3. Normalize reads folders, not the curated CSVs

`04` walks `dataset_real` and `datasetV1/2/3` directly, not
`dataset_AI_final.csv` / `dataset_real_final.csv`. If curation lived only in those
CSVs (rather than by deleting files), eliminated images can re-enter the 1500
sample. Confirm the folders were physically pruned, or switch `04` to sample from
the CSV filepaths.

## 4. War-torn topic tagging mismatch (singular vs plural)

`03` saves war-torn files with the singular slug `war_torn_destroyed_building`;
`04`'s `guess_topic` matches on the plural `war_torn_destroyed_buildings`. The
plural is not a substring of the singular filename, so real war-torn images get
tagged topic `unknown`. Normalize the slug across both, or pull topic from the
manifest instead of re-deriving from filenames.

## 5. Information-integrity topics reuse the photojournalism scaffold

`documents`, `fake_news`, `memes` are in the 8-topic set but use the same
photojournalism prompt scaffold in `01`, and are marked WEAK on the Wikimedia
real side in `03`. They likely want their own prompt template and a different
real source. Decide whether they belong in the reported scope.

## 6. Training description vs run of record

`06` uses plain `CrossEntropyLoss` — no label smoothing, no temperature scaling —
and in the run of record early stopping never fired (it hit the 100-epoch
ceiling, so "best epoch" = last epoch). If the writeup mentions smoothing,
calibration, or early stopping, reconcile it with what actually ran, or add those
pieces and re-run.

## 7. Verdict calibration

The saturated, uncalibrated head pushes `real_score` to 0 or 100, so the middle
bands in `07` almost never fire. Temperature scaling would give the tiered
verdict a usable spread.

## 8. Unused VLM load

`07` had a Qwen2.5-VL load (~16GB) that the verdict never used. It is now behind
`USE_VLM=False` so a fresh clone doesn't OOM. Wire it in or remove it if the
geometric explanation is the final design.

## Canonical accuracy figure

Use **0.996 in-distribution** from the `06` run of record (fully traceable). The
stray 0.978 was an earlier config. Always label the figure "in-distribution".
