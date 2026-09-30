# QC model training and analysis scripts

Code behind the QC model numbers in the final report. The folder names are the dates the
scripts were written, same as in our working folder.

The scripts were run from a workspace where these dated folders sit in a `work/` folder next
to this repo (`work/` and `LLMs_for_NEL/` side by side). Paths assume that layout. Set
`NLP_LAB_WORK` to point somewhere else. Model checkpoints and the large intermediate CSVs are
not in the repo.

| Script | What it produces | Where in the report |
|---|---|---|
| `2026-07-08_meeting/qc_run2_executed.ipynb` | Trains the QC model (run 2, seed 42, Colab T4) | QC hyperparameter table |
| `2026-07-22_next_tasks/run2_infer_test.py` | Per-row predictions of the QC model on the perturbed test set | Precision 0.684, recall 0.730, F1 0.706, per-perturbation rates |
| `2026-07-22_next_tasks/task2_dynamic_threshold.py` | Abstract-level keep rate under the strict and the dynamic rule | 19% and 24% |
| `2026-07-22_next_tasks/task3_threshold_sweep.py` | Keep rate with a lowered decision cut-off | 23% |
| `2026-09-09_overnight/task1_driver.py` | Relation count against abstract length, each held fixed in turn | +0.15 to +0.24 against +0.07 to +0.12 |
| `2026-09-09_overnight/task1_control.py` | The same trend on the real BioRED text | 24.4 to 49.3% against 30.4 to 49.4% |
| `2026-09-09_overnight/task45_per_type.py` | QC results by claimed relation type and entity pair, on real and generated text | Per-type and matched tables |
| `2026-09-09_overnight/task5_matched.py` | The same claims scored on real and generated text | 841 triples, 40.9% against 43.1% |
| `2026-09-09_overnight/build_norelation_pairs.py`, `score_norelation.py`, `score_norelation_realtext.py`, `task3_norelation.py` | Implicit NoRelation pairs and their scores | 104,136 pairs, 14.6% against 14.7% |
| `2026-09-09_overnight/task2_dedup.py` | Abstracts kept against distinct papers kept | Paper and abstract counts in the ladder |
| `2026-09-10_entity_pair_labels/entity_pair_labels.py` | Label distribution per entity pair, and whether Chemical-Gene is hard on its own | Chemical-Gene discussion |
| `2026-09-10_norel_variants/norel_variants.py`, `norel_table.py` | NoRelation-aware filtering and its variants | Strict keep rate 30.5% to 17.3% |

The filtering itself (`decide.py`, `filter_synthetic.py`, `filter_norel_aware.py`) and the
rejection-rate figures are in `../qc_filtering/`. The perturbed dataset is built by
`../dataset_preparation/perturbations.py`.
