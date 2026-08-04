# Work log: BERT 1 (relation classifier) evaluation and presentation

Scope of this document: everything done on the **BERT 1** side (relation classifier, Chris)
between commits `21c420d` and `48abf28`.
It exists so that a new agent can take over without redoing anything.
BERT 2 (the QC filter, Hamed) is not covered here.

Everything that needs CUDA runs through the flake: `nix develop -c python <script>`.
`shell.nix` does not work and there is no system-wide CUDA.

---

## 1. The core problem that was fixed

The metrics originally stored in `finetuning/results.jsonl` under `"metrics"` were a plain
micro-F1 over all nine classes.
That metric also rewards every pair the model correctly rejects as `NoRelation`, and in
single-label classification micro-F1 is just accuracy, so the reported numbers were mostly
measuring the model's ability to say "no".

BioRED-style scoring ignores those rows entirely: a pair where gold and prediction are both
`NoRelation` produces no TP, no FP and no FN.
Every run was therefore re-scored with the correct metric.

Nothing that already existed in `results.jsonl` was overwritten.
The correct scores live alongside the old ones under the new key `"actual_metrics"`.

---

## 2. Scripts written (all in `finetuning/`)

| script | what it does | output |
|---|---|---|
| `reevaluate_biored_metric.py` | Re-scores every entry of `results.jsonl` with the BioRED metric | adds `actual_metrics` to `results.jsonl` |
| `epoch_curves.py` | Scores **all 40 checkpoints** of the four runs, epoch by epoch | `epoch_curves.csv`, `epoch_curves.json` |
| `plot_epoch_curves.py` | Draws the per-epoch figures from the CSV (no GPU needed) | 5 PNGs in `presentation/figures/` |
| `validate_with_eval_notebook.py` | Executes `eval.ipynb` headlessly on one of our checkpoints and compares | `eval_validation_run.ipynb` (gitignored) |
| `check_scoring_truth_table.py` | Verifies the scoring truth table on the slide against the real scorer | stdout only, seconds, no GPU |

Provenance is marked in the source: `build_samples` is copied from
`finetuning_pubmedbert.ipynb`, `compute_biored_f1` and `compute_negative_discrimination` are
copied from `eval.ipynb`, each behind a banner comment.
`epoch_curves.py` imports the scorer from `reevaluate_biored_metric.py` instead of copying it
again, so the curves and the table in the deck come from one code path.

### Gotchas encoded in those scripts (do not rediscover these)

- **Run directories hold no weights.** The trained model of a run was dumped by the notebook to
  `<run>_dump` (`model.save_pretrained`), and only `checkpoint-*` directories carry a tokenizer.
  `resolve_model_dir` / `resolve_tokenizer_dir` handle this.
- **Naming drift.** Some directories use `relations-bert_bioformers-...`, others
  `relations-bert-bioformers-...`. Any lookup must normalise `-` and `_` (`_normalise`,
  `resolve_run_dir`).
- **The gold column is `target_relation`, not `relation_type`.** On perturbated data
  `relation_type` holds the *fake* type of a `false_positive` row.
- **`results.jsonl` is not line-delimited JSON.** It holds pretty-printed multi-line objects;
  parse with `json.JSONDecoder().raw_decode` (`read_results`).
- **`finetuning.commons` cannot be imported** in the flake: it imports `yaml`, which is not in
  the devshell. Import from `dataset_preparation.perturbations` instead.
- **transformers 5.11 removed `Trainer(tokenizer=...)`.** It is `processing_class=` now.
  `eval.ipynb` still uses the old name and cannot run unmodified in this flake.

---

## 3. Results

### 3.1 The two test-set variants

`dataset_preparation/prepare_pure_biored.py` gained a `no_relation_ratio` parameter.
`None` keeps every co-mentioned unrelated pair and switches off the distance cap.

All evaluation uses `Data/BioRED/Test.PubTator` only (100 abstracts).
Train is read solely to *fit* the distance statistics used when sampling negatives.

| test set | rows | real relations | unrelated pairs |
|---|---|---|---|
| matched (1 sampled `NoRelation` per gold relation) | 2,326 | 1,163 | 1,163 |
| all co-mentioned pairs (`no_relation_ratio=None`) | 10,097 | 1,163 | 8,934 |

The positives are the *same rows* in both, only the negatives grow, which is why recall is
identical in both and only precision moves.
The 1,163 gold relations are Association 635, Positive_Correlation 325,
Negative_Correlation 171, Cotreatment 14, Bind 9, Comparison 6, Drug_Interaction 2,
Conversion 1.

Corpus shape, for reference (this is why ~10k candidate pairs come out of 100 abstracts):

| split | abstracts | entities | gold relations | all co-mentioned pairs |
|---|---|---|---|---|
| Train | 400 | 4,471 | 4,178 | 29,747 |
| Dev | 100 | 1,240 | 1,162 | 9,094 |
| Test | 100 | 1,264 | 1,163 | 9,791 |

(The builder reaches 10,097 rather than 9,791 because it splits BioRED's comma-joined
`mesh_id` annotations into their component concepts.)

### 3.2 Best-checkpoint scores

All four models, one test set (pure BioRED Test) and one selection rule (the checkpoint
`load_best_model_at_end` saved on Dev micro-F1):

| model | training negatives | epoch | micro-F1 (9 cls.) | BioRED F1 (matched) | BioRED F1 (all pairs) |
|---|---|---|---|---|---|
| PubMedBERT | distance-matched | 8 | 0.724 | **0.619** | 0.375 |
| Bioformer-8L | distance-matched | 9 | 0.675 | 0.530 | 0.312 |
| PubMedBERT | type-restricted | 8 | 0.672 | 0.575 | **0.391** |
| Bioformer-8L | type-restricted | 9 | 0.611 | 0.500 | 0.317 |

**The ordering of the two training sets flips between the last two columns, for both
encoders.** Distance-matched negatives win on the balanced test set, type-restricted
negatives win on the realistic one. This was invisible before, because the type-restricted
runs had only ever been scored on their own test split (990 gold / 972 negatives, no
all-pairs variant). The effect survives the alternative selection rule (select on Dev under
the BioRED metric): 0.397 vs 0.376 and 0.316 vs 0.314.

Superseded numbers, for reference, each model on its own test split and with two different
selection rules: 0.735 / 0.638 / 0.399, 0.678 / 0.548 / 0.314, 0.695 / 0.588 / --,
0.643 / 0.524 / --.

Encoders: `NeuML/pubmedbert-base-embeddings` (12 layers, hidden 768) and
`bioformers/bioformer-8L` (8 layers, hidden 512).

**The all-pairs column is the honest one.** 0.638 is inflated by the balanced test set:
precision is measured against 1,163 negatives instead of 8,934.
The `~44%` quoted from the supervisor's run is the all-pairs setting, so it must not be
compared against 0.638.

False-positive decomposition for PubMedBERT (from `negative_discrimination` in
`results.jsonl`):

- gold-negative rows claimed as a relation: 189 (matched) -> 1,574 (all pairs)
- gold rows given the wrong relation type: 218 -> 218 (same rows, same model)

So the model claims a relation on 1,574 of 8,934 unrelated pairs, **17.6%**.

### 3.3 Per-epoch curves

`finetuning/epoch_curves.csv`, BioRED F1 per epoch:

| run | test variant | e1 ... e10 | best |
|---|---|---|---|
| PubMedBERT / BioRED | matched | 0.492 0.587 0.539 0.592 **0.638** 0.627 0.631 0.619 0.599 0.609 | e5 |
| PubMedBERT / BioRED | all pairs | 0.314 0.358 0.353 0.368 **0.399** 0.385 0.389 0.375 0.376 0.379 | e5 |
| PubMedBERT / perturbed | matched | 0.501 0.553 0.498 0.609 0.571 0.583 0.587 0.588 **0.606** 0.595 | e9 |
| Bioformer-8L / perturbed | matched | 0.430 0.516 0.446 0.511 0.520 0.511 0.515 0.515 **0.524** 0.521 | e9 |
| Bioformer-8L / BioRED | matched | 0.491 0.533 0.531 **0.548** 0.525 0.508 0.524 0.513 0.530 0.524 | e4 |
| Bioformer-8L / BioRED | all pairs | 0.303 0.282 **0.325** 0.314 0.310 0.302 0.295 0.312 0.312 0.306 | e3 |

Observations: training is finished after 4-5 epochs, the rest is +-0.02 noise;
every run dips at epoch 3; Bioformer-8L on all pairs never improves at all.

---

## 4. Verification that was done

Three independent checks, all scripted and rerunnable:

1. **Old metric, new code.** `reevaluate_biored_metric.py` also recomputes the 9-class
   micro-F1 and reproduces the stored value of all seven entries bit for bit
   (e.g. `0.7351676698194325`). This proves the rebuilt test set and the model loading are
   identical to the original evaluation, so the metric is the only thing that changed.
2. **New metric, foreign code.** `validate_with_eval_notebook.py` executes `eval.ipynb` on
   `relations-bert-2026-07-01_18-38-41_BioRed_micro/checkpoint-2610` and prints
   TP=736, FP=1792, FN=427, P=0.2911, R=0.6328, F1=0.3988 -- identical to the stored
   `actual_metrics.full_norelation_test`. The notebook is never modified on disk; the
   in-memory copy gets the `/home/flabonte` paths, the checkpoint, and the
   `Trainer(tokenizer=)` -> `processing_class=` rename, each printed when it runs.
3. **The slide's truth table.** `check_scoring_truth_table.py` execs `compute_biored_f1` out
   of `eval.ipynb` and scores all 9x9 = 81 (gold, prediction) combinations, mapping each to
   the slide row that claims to cover it. All five rows pass and no case is left uncovered.
   Note: gold = relation, pred = *other* relation books **FN and FP**, so a wrong relation
   type is penalised twice.
4. **The per-epoch sweep re-derives everything.** All 11 values stored in `results.jsonl`
   were reproduced by `epoch_curves.py` to the last digit.

---

## 5. OPEN ITEM -- needs a decision before the deck is final

**Two of the four rows in the results table use test-selected checkpoints.**

The training notebook uses `load_best_model_at_end=True` with
`metric_for_best_model="eval_f1_micro"` and saves that Dev-selected model to `<run>_dump`.
The `_dump` is therefore the Dev-selected model. Two deck rows do not use it:

| row | checkpoint in deck | Dev-selected | deck value | Dev-selected value |
|---|---|---|---|---|
| PubMedBERT / BioRED | `checkpoint-2610` = **epoch 5** | epoch 8 | 0.638 / 0.399 | 0.6188 / 0.3750 |
| Bioformer-8L / BioRED | `checkpoint-2088` = **epoch 4** | epoch 9 | 0.548 / 0.314 | 0.5300 / 0.3120 |
| PubMedBERT / perturbed | `_dump` = epoch 8 | epoch 8 | 0.588 | 0.588 (ok) |
| Bioformer-8L / perturbed | `_dump` = epoch 9 | epoch 9 | 0.524 | 0.524 (ok) |

Epochs 5 and 4 are exactly the argmax of the *test* curve for those runs, so those two rows
are selected on the test set (worth about +2 points) while the other two are selected on Dev.
The table mixes two selection rules, which also makes the encoder comparison unfair, and it
contradicts the setup slide ("best one picked on Dev, then scored once on Test").

**Recommendation:** switch all four rows to the Dev-selected model.
The user had not decided this when the work log was written -- ask before changing numbers.

---

## 5b. OPEN ITEM -- the four table rows are not comparable

Two separate findings, both unresolved.

### The perturbed rows are scored on a different test set

`reevaluate_biored_metric.py` builds the test split from each entry's own `dataset` field,
so the BioRED-trained models are scored on the BioRED test set and the perturbed-trained
models on the perturbated test set:

| row | test set | rows | gold relations |
|---|---|---|---|
| BioRED-trained | BioRED test | 2,326 | 1,163 |
| perturbed-trained | perturbated test | 1,962 | 990 |

So 0.638 versus 0.588 compares two models on two different test sets with different gold
subsets. It is not a valid comparison.

The `--` in the "all rels." column is a smaller, separate issue: `no_relation_ratio=None`
belongs to the pure-BioRED builder, which enumerates co-mentioned pairs. The perturbated
builder samples roughly one negative per gold relation and has no "all pairs" mode.

**Proposed fix (not yet run):** score every checkpoint on the *same* BioRED test set,
both variants, regardless of what it was trained on. Well-defined, because all 9 classes
appear in both training sets, and it needs no retraining. Costs 4 forward passes for the
best checkpoints, ~40 for full curves. Caveat to state when reporting: a model trained on
type-restricted negatives meets entity-type combinations at test time that it never saw as
negatives.

### BUG: `perturbations.py` loses gold relations with comma-joined MeSH IDs

`dataset_preparation/perturbations.py:_build_entity_lookup` groups annotations by the raw
`mesh_id` string. BioRED sometimes stores one annotation under a comma-joined id such as
`"1508,836,841"` while the relations reference the individual components, so those lookups
miss and the gold relation is skipped at `if not info_a or not info_b: continue`.

`dataset_preparation/prepare_pure_biored.py:_build_entity_spans` already handles this by
indexing every component id separately. `perturbations.py` has a TODO about it but no fix.

Measured impact:

| split | pure BioRED gold | perturbated gold | lost |
|---|---|---|---|
| Train | 4,178 | 3,831 | 347 (8.3%) |
| Test | 1,163 | 990 | 173 (14.9%) |

The perturbated gold set is a strict subset (nothing extra). Verified: all 173 dropped Test
relations have an endpoint missing from the perturbated lookup, and in all 173 cases that
endpoint is a component of a comma-joined id (44 such annotations exist in Test). The loss
is not uniform: Association 109, Positive_Correlation 49, Negative_Correlation 11, Bind 4.

Consequence: the perturbed-trained models saw 8.3% fewer positives in training and were
scored on a test set missing 14.9% of the positives. "Perturbed is worse than original" is
therefore partly an artefact of this bug.

Fixing the evaluation side only needs the common-test-set change above. Fixing the training
side needs `perturbations.py` repaired and those two runs retrained. **Not done: that file
also feeds the BERT 2 / QC dataset, so it needs the owner's agreement first.**

### The "perturbed" condition contains no perturbations

`build_samples` keeps only `perturbation in {"gold", "false_positive"}` and therefore drops
every `label_flip`, `direction_swap` and `false_negative` row:

```
perturbated Train, all rows build_training_samples makes: 17,087
  gold 3,831 | label_flip 3,831 | false_negative 3,831 | false_positive 3,783 | direction_swap 1,811
build_samples() keeps 7,614 and drops 9,473
```

And in `dataset_preparation/perturbations.py:174`, `false_positive` means "two co-mentioned
entities with no annotated relation" (type-restricted), which is the same *kind* of negative
that `prepare_pure_biored.py` produces (distance-matched).

**The filtering is correct, not a bug.** BERT 1 is a 9-class single-label classifier, and a
`label_flip` row ("pair A-B, claimed type X, actually type Y, label 0") has no valid 9-class
target. Those rows exist for BERT 2's binary supported/unsupported task. The notebook's own
docstring says the samples are "restricted to gold and false_positive perturbations", so it
was deliberate. Only the *slide label* was wrong, because it implies the classifier consumed
corrupted labels.

### Reframing: the perturbed runs are a negative-sampling ablation

What the second condition actually varies is which negatives are sampled. Measured overlap of
the two training sets, on unordered `(pmid, id_1, id_2)` pairs of the Train split:

| training rows | distance-matched | type-restricted | shared | Jaccard |
|---|---|---|---|---|
| gold relations | 4,178 | 3,831 | 3,831 | 91.7% |
| `NoRelation` pairs | 4,162 | 3,163 unique | 632 | **9.4%** |

The two conditions share their positives and disagree on 91% of their negatives, which is
half of every training set and the half this project has shown to dominate the score. So the
two runs are an accidental but genuine ablation of the negative-sampling rule:

- **distance-matched** (`prepare_pure_biored.py`): keep pairs whose character gap resembles
  that of real relations, drop everything above the observed maximum, rank by closeness to
  the mean, cap at one per gold relation.
- **type-restricted** (`perturbations.py`): keep pairs whose entity-type combination occurs
  in some gold relation, roughly one per gold relation.

They are worth keeping and reporting. What they cannot say anything about is robustness to
corrupted labels; that claim belongs to BERT 2.

Two caveats to carry whenever a gap between the conditions is reported:

1. The type-restricted condition also has 8.3% fewer positives (the comma-joined-id bug
   above), so a gap is not cleanly attributable to the sampling rule.
2. Its 3,783 negative rows collapse to 3,163 unique pairs, so about 620 are duplicates.
   `sample_false_positive_pair` is called once per gold relation and can return the same pair
   for several relations of one abstract, so those pairs are effectively upweighted.

This is presented on the slide "Two ways to pick the negatives".

---

## 6. Presentation

`presentation/main.tex`, BERT 1 section (after the colleague's "The relation classifier"
frame). Compiles clean at 25 pages; the only box warning is a pre-existing 11.7 pt overfull
vbox on the colleague's "Abstract-level evaluation" slide (line 207), deliberately untouched.

Frames added: section divider, "The task and the prompt", "Building the training data",
"The class imbalance", "Models and training setup", "What BioRED counts, and what we
counted", "All runs, re-scored", "The same runs, epoch by epoch", "Same model, two test
sets", "Checking that the numbers hold", "What went wrong before this", "Next steps for
BERT 1".

**Interpretation is written into the source as `%` comments**, before the results, epoch-curve,
negatives, validation and next-steps frames. They are not shown on the slides. The results
frame also carries a provenance block naming, per row, which checkpoint it is and how it was
selected. Read those before changing any number.

Figures in `presentation/figures/`:
`biored_f1_per_epoch_all_runs.png` (on the slide), plus one per run:
`biored_f1_per_epoch_{pubmedbert,bioformer-8l}_{biored,perturbed-biored}.png`.

Style notes agreed with the user: plain human wording, short titles, no em dashes,
one sentence per line, key takeaway per slide in bold.

---

## 7. Next steps for BERT 1 (not started)

1. **Train on the realistic negative set** instead of the balanced one and report that number.
   This is first because it changes the baseline every later comparison is measured against.
   Evidence it matters: the supervisor's run trained on all negatives scores 0.4375 on the
   all-pairs test set, ours trained at 1:1 scores 0.399, and a supervisor run trained at 1:1
   scores ~0.32-0.34. So roughly 4-6 points. It costs only a longer epoch: 31,071 training
   rows instead of 8,340.
   (Derived from checkpoint step counts in `eval.ipynb`: 261 steps/epoch = 8,340 rows = 1:1;
   969 steps/epoch = ~31,000 rows = all negatives. Corroborated by the corpus table in 3.1:
   Train has 4,178 gold against 29,747 candidate pairs.)
2. Perturbed BioRED: retrain in the fixed pipeline and measure the gap to the baseline.
3. Synthetic data: train on the LLM-generated abstracts.
4. Filtered synthetic data: train on what the QC model (BERT 2) keeps.

Since training is finished after 4-5 epochs, future runs do not need 10.

---

## 8. Loose ends

- `dataset_preparation/perturbations.py:313` has a leftover `print(records[0])` in
  `samples_to_rels_like_df` that dumps a whole abstract on every call. Not removed because it
  is shared code; worth deleting.
- The supervisor's notebook builds 10,063 rows for the all-pairs test split where we build
  10,097 (0.3% apart, presumably the distance cap). Close enough to compare, not close enough
  to call identical.
- `eval.ipynb` cannot run as-is against the pinned transformers 5.11 (`Trainer(tokenizer=)`).
  Worth reporting upstream; the same call sits in `plot_checkpoints_comparison`.
