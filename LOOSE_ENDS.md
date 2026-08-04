# Loose ends

Living list of open work for the BERT 1 side.
Completed work is documented in `TASKS_DONE.md`, not here.
Update this file whenever an item is finished or a new one appears.

Status markers: `[ ]` open, `[~]` in flight, `[?]` blocked on a decision by Chris,
`[-]` consciously deferred.

---

## In flight

Nothing. The Dev sweep and the common-test-set sweep are both finished
(`finetuning/epoch_curves.csv`, 200 rows) and the subagent has reported.

## Done since the sweeps landed

- [x] **New numbers are in the LaTeX table.** All four rows now use the pure-BioRED test
  split and the checkpoint the training pipeline itself selected on Dev (epochs 8, 8, 9, 9,
  verified by matching each `_dump` score to its epoch exactly).
- [x] **Ordering claim rewritten.** "Perturbed stays behind original" was false on the
  all-pairs column. The slide now says PubMedBERT stays ahead of Bioformer-8L everywhere,
  but the ranking of the two training sets flips between the last two columns.
- [x] **Figures regenerated** from the complete data; the combined figure had been rendered
  mid-sweep and was missing two curves. The slide points at the common-test-set version.

## Waiting

- [ ] **Reclaim ~3 GB of unreachable git objects.** Commit `54b5754` accidentally added
  model weights and dataset caches via `git add finetuning/`; it was reset and recommitted
  as `9573c35` with only the intended files, and `.gitignore` now covers those paths.
  The objects are unreachable but still on disk, so `.git` is 3.1 GB.
  Needs `git reflog expire --expire-unreachable=now --all && git gc --prune=now`,
  which is irreversible, so it waits for Chris. Nothing was ever pushed.

- [ ] **Update `TASKS_DONE.md` section 3.2** with the final table (done) and close out 5b.

## Blocked on a decision

- [x] **Which checkpoint to quote. RESOLVED: the one the pipeline selects.**
  Every row now uses the checkpoint `load_best_model_at_end` saved to `<run>_dump`
  (Dev micro-F1): epochs 8, 8, 9, 9. This involves no post-hoc choice and matches the
  claim on the setup slide. It cost the two distance-matched rows about 2 points
  (0.638 -> 0.619 and 0.548 -> 0.530 matched).

  For the record, selecting on Dev under the *BioRED* metric instead would pick epochs
  8, 10, 4, 5 and is equally defensible; the flip finding below holds under either rule.
  One correction to an earlier claim in this file: the deck's old Bioformer-8L epoch 4 was
  reachable by Dev selection under the BioRED metric (Dev and Test both peak at epoch 4),
  so that row was not necessarily cherry-picked. PubMedBERT's epoch 5 is not reachable by
  any Dev rule tried.

  Dev is only mildly optimistic: over the 80 paired Dev/Test points, mean Dev minus Test is
  +0.006 (sd 0.019), and the selection regret ranges from 0.000 to 0.022.

- [?] **Rename the "perturbed BioRED" condition on the BERT 1 slides.**
  `build_samples` keeps only `perturbation in {gold, false_positive}`, dropping every
  `label_flip`, `direction_swap` and `false_negative` row (9,473 of 17,087 on Train).
  What is left is gold plus type-restricted co-mentioned negatives, so that condition
  tests a different negative-sampling strategy, not perturbation.

  **Resolution agreed with Chris: keep the label, add a footnote.**
  The "training data" column has to document provenance, and "perturbed BioRED" is the
  correct provenance: it names the pipeline the data came from and links to Hamed's
  dataset slide. Renaming it to "type-restricted negatives" would hide that.
  The label is only wrong about the *content*, so the correction belongs under the table:

  > `perturbed BioRED` = gold + negatives from the perturbation pipeline; the `label_flip`,
  > `direction_swap` and `false_negative` rows are filtered out by `build_samples`, so the
  > classifier never sees a perturbed relation. The two conditions therefore differ in how
  > negatives are sampled, not in whether labels were corrupted.

  What those two runs actually trained on: 3,831 gold relations, 3,783 type-restricted
  co-mentioned negatives labelled `NoRelation`, and zero perturbed rows (9,473 dropped).

  **The runs are still worth keeping.** Measured overlap against the pure-BioRED training
  set: gold 3,831 of 4,178 shared (91.7%), but negatives only 632 shared out of 4,162 pure
  and 3,163 unique perturbated (9.4% Jaccard). The two conditions share their positives and
  differ in 91% of their negatives, which is half of every training set and the half this
  project has shown to dominate the score. So they are an accidental but genuine ablation of
  the negative-sampling rule, distance-matched versus type-restricted. Reframe that part of
  the slide from "perturbed versus original" to "does the negative-sampling rule matter",
  which the common-test-set numbers can answer.

  Two caveats to carry: the perturbated condition also has 8.3% fewer positives (comma bug),
  so a gap is not cleanly attributable to the negative strategy; and its 3,783 negative rows
  collapse to 3,163 unique pairs, so ~620 are duplicates that are effectively upweighted.

  Leave `presentation/main.tex` lines 416 and 417 as they are. Lines 378, 425 and 448 are
  our source comments and the ordering bullet, still to be revisited with the new numbers.

  **Do not touch lines 64 and 642.** Those are Hamed's slides about the QC dataset for
  BERT 2, where the perturbations really are the training signal, so the phrase is correct
  there.

## Deferred by decision

- [-] **Comma-joined MeSH id bug in `dataset_preparation/perturbations.py`.**
  `_build_entity_lookup` groups by the raw `mesh_id` string, so relations referencing a
  component of a joined id like `"1508,836,841"` are silently dropped:
  347 of 4,178 gold on Train (8.3%), 173 of 1,163 on Test (14.9%).
  `prepare_pure_biored.py` already handles this correctly.
  Chris deferred the fix for time reasons, and nothing is being retrained.
  **Consequence to keep stating:** the perturbed models trained on 8.3% fewer positives,
  so part of any gap to the BioRED-trained models is this bug, not the data design.
  Fixing it also touches the BERT 2 / QC dataset, so it needs Hamed's agreement.

## Minor / housekeeping

- [ ] **Stray debug print.**
  `dataset_preparation/perturbations.py:313` has `print(records[0])` in
  `samples_to_rels_like_df`, which dumps a whole abstract on every call.
  Shared code, so not removed unilaterally.

- [ ] **`eval.ipynb` cannot run under the pinned transformers 5.11.**
  It uses `Trainer(tokenizer=...)`, removed in transformers 5.
  Our validation script patches an in-memory copy, but the notebook itself is still
  broken for anyone else in this flake.
  Worth reporting to the supervisor; the same call sits in `plot_checkpoints_comparison`.

- [ ] **Row-count discrepancy against the supervisor's notebook.**
  `eval.ipynb` builds 10,063 rows for the all-pairs test split where we build 10,097
  (0.3% apart, presumably the distance cap).
  Close enough to compare, not close enough to call identical.
  Not chased down.

- [-] **Pre-existing overfull vbox** at `presentation/main.tex:207`, on the colleague's
  "Abstract-level evaluation" slide (11.7 pt).
  Left untouched on purpose, it is not our slide.
