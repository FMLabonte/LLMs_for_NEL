# Task 4, step 1: QC filtering of the synthetic abstracts

Run 2 QC model over 35,658 relation claims in 3,552 synthetic abstracts (2 generators x 3 splits x 3 generations per paper).

Rule shipped: **dynamic**, rare types **excluded**, cut-off 0.5.

## Kept per file

| model | split | abstracts | kept | kept % | papers surviving |
|---|---|---|---|---|---|
| qwen3_4b | dev | 294 | 78 | 26.5% | 32 / 98 |
| qwen3_4b | test | 300 | 83 | 27.7% | 35 / 100 |
| qwen3_4b | train | 1182 | 492 | 41.6% | 205 / 394 |
| qwen3_8b | dev | 294 | 71 | 24.1% | 28 / 98 |
| qwen3_8b | test | 300 | 82 | 27.3% | 34 / 100 |
| qwen3_8b | train | 1182 | 501 | 42.4% | 212 / 394 |

## All four rule combinations

| rule    | rares   |   kept |   total |   kept_pct |
|:--------|:--------|-------:|--------:|-----------:|
| strict  | include |   1001 |    3552 |       28.2 |
| strict  | exclude |   1089 |    3552 |       30.7 |
| dynamic | include |   1183 |    3552 |       33.3 |
| dynamic | exclude |   1307 |    3552 |       36.8 |

## Pass rate by relation count (the known bias)

| n_relations   |   size |   sum |   mean |
|:--------------|-------:|------:|-------:|
| 1-3           |    888 |   605 |   68.1 |
| 4-8           |   1182 |   382 |   32.3 |
| 9-12          |    522 |   177 |   33.9 |
| 13+           |    954 |   143 |   15   |

Column `mean` is the percentage kept. The 2026-07-22 profiling found the filter is relation-count biased rather than length biased, and that shows here too: abstracts asserting many relations rarely survive, because each claim is another chance to trip the rule.
