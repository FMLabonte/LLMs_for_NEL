# QC filtering that also judges the implicit NoRelation pairs

Meeting 10, Frederik's first item. Built 2026-09-10 with `filter_norel_aware.py`.

The files in `../filtered/` were filtered on the relations each generation prompt lists,
and nothing else. Every co-mentioned entity pair the prompt leaves out is implicitly a
NoRelation claim, and none of those were ever scored, so the filter could not catch a
relation the generator invented. These files fix that.

## Using it

Same schema as `../filtered/`, same loader, only the path changes:

```python
meta, anns, rels = build_synthetic_parsed(
    "qc_filtering/filtered_norel/results_qwen3_8b_train.json",
    "Data/BioRED/Train.PubTator",
)
```

## The rule

An abstract is kept when **no** claim fails at cut-off 0.5, counting both:

- the stated relations, rare types excluded because the QC model was never trained on them,
- the implicit NoRelation pairs, 104,136 of them across 3,252 abstracts.

That is the same strict rule as before. The only thing that changed is which claims the
filter looked at. Mean claims per abstract goes from 9.7 to 39.1.

## What it costs

| model | split | abstracts | kept, stated only | kept, NoRelation aware | papers |
|---|---|---|---|---|---|
| qwen3_4b | dev | 294 | 72 (24.5%) | 44 (15.0%) | 16 / 98 |
| qwen3_4b | test | 300 | 72 (24.0%) | 54 (18.0%) | 22 / 100 |
| qwen3_4b | train | 1179 | 392 (33.2%) | 204 (17.3%) | 80 / 394 |
| qwen3_8b | dev | 294 | 62 (21.1%) | 41 (13.9%) | 14 / 98 |
| qwen3_8b | test | 300 | 70 (23.3%) | 52 (17.3%) | 21 / 100 |
| qwen3_8b | train | 1179 | 415 (35.2%) | 220 (18.7%) | 87 / 394 |

Overall 1,083 abstracts kept becomes 615, so 30.5% becomes 17.3%. 468 abstracts that
passed before fail now, with a median of 3 failed implicit claims each.

## Two things to know before reading a training result

**1. The set is much smaller, and that is a confound on its own.** The 8B train file drops
from 501 abstracts over 212 papers to 220 over 87. Any difference against the old filtered
run mixes the effect of judging negatives with the effect of training on well under half
the data. A same-size random control drawn from the old set would separate the two.

**2. Most of the extra rejections are probably false alarms.** The implicit pairs were
scored twice, once against the generated abstract and once against the real BioRED
abstract for the same paper, where by construction nothing was invented. Real text is
flagged at 14.6%, generated at 14.7%. The QC model simply has a false-alarm rate of about
15% on negatives, and an abstract with 32 implicit pairs is very likely to collect at least
one. So this filter is stricter, but not obviously better informed, and a drop in
downstream performance would not be surprising.

Workings for that control: `work/2026-09-09_overnight/`, `task3_norelation.py`.
