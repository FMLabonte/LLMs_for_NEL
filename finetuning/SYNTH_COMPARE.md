# Synthetic vs. real BioRED: PubMedBERT relation classifier

Comparison of two PubMedBERT relation classifiers that differ only in the source of the training abstracts.
Both are scored on the same held-out real BioRED Test split with the BioRED F1 metric.

## Methodology

Both runs use the identical 9-class setup (8 BioRED relation types plus `NoRelation`), the same prompt format (`Relation: A -> [MASK] -> B` + `Context: <abstract>`), the same distance-matched negative sampling (distance statistics fit on the real BioRED Train split), and the same held-out real BioRED Test split.
The BioRED F1 scorer is pooled over the 8 positive relation types and ignores rows where gold and prediction are both `NoRelation`.

Two evaluation variants are reported, exactly as in the presentation:

- **matched**: one distance-matched `NoRelation` example per gold relation (2,326 rows = 1,163 gold + 1,163 `NoRelation`).
- **all rels.**: every co-mentioned unrelated pair of the test abstracts (10,097 rows = 1,163 gold + 8,934 `NoRelation`).

### How the current run differs from the original

| Aspect | Previous (real BioRED) | Current (synthetic Qwen3-8B) |
|---|---|---|
| Abstract text | Real BioRED Train abstracts | LLM-generated abstracts (Qwen3-8B), entities/relations still from BioRED |
| Training rows | 1 abstract per paper | 3 generations per paper (context augmentation), 25,002 samples |
| Validation / selection | Real BioRED **Dev**, best checkpoint selected on Dev micro-F1 | **None** (no synthetic Dev split exists); no checkpoint selected |
| Between-epoch eval | On | Off (`ENABLE_EVAL_BETWEEN_EPOCHS = False`) |
| Epochs | 10 | 5 |
| Test split | Real BioRED Test | Real BioRED Test (unchanged, for comparability) |

Everything else is held constant, so the difference in scores is attributable to the abstract source (plus the 3x augmentation and the absence of checkpoint selection).
Because the current run has no synthetic Dev split, its epochs are reported un-selected, whereas the presentation quotes the previous model at the Dev-selected checkpoint (epoch 8: matched 0.619, all rels. 0.375).

## Epoch-wise comparison (BioRED F1, real BioRED Test)

Best value per column is in **bold**.

| Epoch | Prev. matched | Prev. all rels. | Synth. matched | Synth. all rels. |
|:-:|:-:|:-:|:-:|:-:|
| 1  | 0.492 | 0.314 | **0.573** | 0.348 |
| 2  | 0.587 | 0.358 | 0.541 | 0.381 |
| 3  | 0.539 | 0.353 | 0.546 | 0.362 |
| 4  | 0.592 | 0.368 | 0.509 | **0.389** |
| 5  | **0.638** | **0.399** | 0.502 | 0.366 |
| 6  | 0.627 | 0.385 | - | - |
| 7  | 0.631 | 0.389 | - | - |
| 8  | 0.619 | 0.375 | - | - |
| 9  | 0.599 | 0.376 | - | - |
| 10 | 0.609 | 0.379 | - | - |

Best per model:

| Metric | Previous (real BioRED) | Current (synthetic) |
|---|---|---|
| matched, best epoch | **0.638** (ep5) | 0.573 (ep1) |
| all rels., best epoch | **0.399** (ep5) | 0.389 (ep4) |

## Takeaways

On the **matched** metric the synthetic model trails the real-BioRED model (best 0.573 vs 0.638).
On the **all rels.** metric the two are close (best 0.389 vs 0.399), and the synthetic model even beats the previous model's Dev-selected value of 0.375.

The two models reach their all-rels. score differently.
The synthetic model is more conservative: at its best epoch it has higher precision (0.363) and lower recall (0.420), whereas the real-BioRED model has low precision (0.291) and high recall (0.633).
The synthetic model also peaks earlier (matched at epoch 1, all rels. at epoch 4) and then overfits the synthetic phrasing, so more epochs do not help.

Per-epoch source data: `epoch_biored_metric.json` (synthetic run dir) and `epoch_curves.csv` (previous runs).
