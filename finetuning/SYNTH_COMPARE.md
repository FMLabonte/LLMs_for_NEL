# Synthetic vs. real BioRED: PubMedBERT relation classifier

Both are scored on the same held-out real BioRED Test split with the BioRED F1 metric.

## Methodology

- Both runs use the same distance-matched negative sampling (distance statistics fit on the real BioRED Train split)
    - calculated individually for both datasets tho...

- evaluated on the same held-out real BioRED Test split.

### Evaluation metrics

- **matched**: one distance-matched `NoRelation` example per gold relation (2,326 rows = 1,163 gold + 1,163 `NoRelation`).

- **all rels.**: every co-mentioned unrelated pair of the test abstracts (10,097 rows = 1,163 gold + 8,934 `NoRelation`).

### Comparison

| Aspect | Previous (real BioRED) | Current (synthetic Qwen3-8B) |
|---|---|---|
| Abstract text | Real BioRED Train abstracts | LLM-generated abstracts (Qwen3-8B), entities/relations still from BioRED |
| Training rows | 1 abstract per paper | 3 generations per paper, 25,002 samples |
| Validation / selection | Real BioRED **Dev**, best checkpoint picked on micro-F1 | None, no synthetic Dev split exists |
| Between-epoch eval | On | Off |
| Epochs | 10 | 5 |
| Test split | Real BioRED Test | Real BioRED Test |


The synthetic epochs are reported unselected. The previous model is quoted at its Dev-picked checkpoint (epoch 8: matched 0.619, all rels. 0.375).

## Epoch-wise comparison (BioRED F1, real BioRED Test)

Best **BioRED F1** per column is in **bold**.

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

On the **matched** metric the synthetic model stay behinf the real one (0.573 vs 0.638 best).

On **all rels.** they are close (0.389 vs 0.399 best), and synthetic beats the previous model's Dev-picked 0.375.

The all-rels. score is reached differently.

Synthetic is more conservative: at its best epoch **precision** is **0.363** and **recall 0.420**, against **0.291** / **0.633** for the **real** one.

Synthetic also peaks early (matched at epoch 1, all rels. at epoch 4).

Per-epoch source data: `finetuning/relations-bert_NeuML-pubmedbert-base-embeddings_2026-08-04_21-41-09_BioRed_synthetic_micro/epoch_biored_metric` (synthetic run dir) and `./finetuning/epoch_curves.json` (original runs).

## Full per-epoch metrics

BioRED metric on the real BioRED Test split.
P = precision, R = recall; TP/FP/FN are the pooled positive-relation counts.
Best **BioRED F1** per column in **bold** (the `F1` column: pooled precision/recall/F1 over the 8 positive relation types, ignoring pairs where gold and prediction are both `NoRelation`).

### matched variant (2,326 rows)

#### real BioRED:

| Epoch | P | R | F1 | TP | FP | FN |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1  | 0.508 | 0.476 | 0.492 | 554 | 536 | 609 |
| 2  | 0.596 | 0.579 | 0.587 | 673 | 457 | 490 |
| 3  | 0.589 | 0.497 | 0.539 | 578 | 403 | 585 |
| 4  | 0.610 | 0.576 | 0.592 | 670 | 429 | 493 |
| 5  | 0.644 | 0.633 | **0.638** | 736 | 407 | 427 |
| 6  | 0.630 | 0.623 | 0.627 | 725 | 425 | 438 |
| 7  | 0.638 | 0.624 | 0.631 | 726 | 412 | 437 |
| 8  | 0.623 | 0.615 | 0.619 | 715 | 433 | 448 |
| 9  | 0.621 | 0.578 | 0.599 | 672 | 410 | 491 |
| 10 | 0.628 | 0.592 | 0.609 | 688 | 408 | 475 |

#### synthetic Qwen3-8B:

| Epoch | P | R | F1 | TP | FP | FN |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 0.595 | 0.552 | **0.573** | 642 | 437 | 521 |
| 2 | 0.635 | 0.472 | 0.541 | 549 | 316 | 614 |
| 3 | 0.592 | 0.506 | 0.546 | 588 | 405 | 575 |
| 4 | 0.647 | 0.420 | 0.509 | 488 | 266 | 675 |
| 5 | 0.618 | 0.422 | 0.502 | 491 | 304 | 672 |

### all rels. variant (10,097 rows)

#### real BioRED:

| Epoch | P | R | F1 | TP | FP | FN |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1  | 0.234 | 0.476 | 0.314 | 554 | 1814 | 609 |
| 2  | 0.259 | 0.579 | 0.358 | 673 | 1927 | 490 |
| 3  | 0.274 | 0.497 | 0.353 | 578 | 1532 | 585 |
| 4  | 0.271 | 0.576 | 0.368 | 670 | 1807 | 493 |
| 5  | 0.291 | 0.633 | **0.399** | 736 | 1792 | 427 |
| 6  | 0.279 | 0.623 | 0.385 | 725 | 1874 | 438 |
| 7  | 0.283 | 0.624 | 0.389 | 726 | 1842 | 437 |
| 8  | 0.270 | 0.615 | 0.375 | 715 | 1935 | 448 |
| 9  | 0.279 | 0.578 | 0.376 | 672 | 1736 | 491 |
| 10 | 0.279 | 0.592 | 0.379 | 688 | 1780 | 475 |

####  synthetic Qwen3-8B

| Epoch | P | R | F1 | TP | FP | FN |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 0.254 | 0.552 | 0.348 | 642 | 1887 | 521 |
| 2 | 0.320 | 0.472 | 0.381 | 549 | 1167 | 614 |
| 3 | 0.282 | 0.506 | 0.362 | 588 | 1501 | 575 |
| 4 | 0.363 | 0.420 | **0.389** | 488 | 856 | 675 |
| 5 | 0.323 | 0.422 | 0.366 | 491 | 1029 | 672 |

The precision/recall split is the clearest signal.
The **synthetic** model reaches its best all rels. F1 with **far fewer false positives** than the previous model (856 vs 1,792 at their best epochs), i.e. higher precision, at the cost of lower recall.
