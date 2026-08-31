# QC acceptance levels, the settings experiment

Meeting 8 task 4. Several filtered versions of the synthetic training data at
different levels of acceptance, so the relation classifier can be trained on each
to see whether any of them beats the unfiltered baseline.

Source: `synthetic_relation_scores.csv`, 35,658 claims scored by the run-2 QC
model over 3,546 synthetic abstracts. Decision cut-off 0.5. Claims of the
five rare BioRED types are dropped, not scored, because the QC model was never
trained on them.

An abstract is kept at level tau when the share of its claims that the QC model
rejects is at most tau. The levels nest, so a more permissive level keeps a superset
of a stricter one, and the only thing changing between two runs is how much
suspected noise is tolerated.

## The ladder

| level         | tolerated error rate   |   abstracts kept |   kept % |   train abstracts |   papers in train | note                                                  |
|:--------------|:-----------------------|-----------------:|---------:|------------------:|------------------:|:------------------------------------------------------|
| L0_strict     | 0%                     |             1083 |     30.5 |               807 |               190 | no detected error at all, the 2026-08-24 shipped rule |
| L1_rate10     | 10%                    |             1195 |     33.7 |               903 |               221 | up to 10% of the claims may be rejected               |
| L2_rate20     | 20%                    |             1620 |     45.7 |              1232 |               278 | up to 20%                                             |
| L3_rate33     | 33%                    |             1983 |     55.9 |              1488 |               317 | up to a third                                         |
| L4_rate50     | 50%                    |             2900 |     81.8 |              2053 |               371 | up to half                                            |
| L5_unfiltered | 100%                   |             3546 |    100   |              2358 |               393 | everything, the baseline Chris already has            |
| D_dynamic     | step fn                |             1260 |     35.5 |               952 |               232 | the step function from decide.py, kept for continuity |

Rate levels nest correctly: **True**.

## Kept per split

| level         |   dev |   test |   train |
|:--------------|------:|-------:|--------:|
| D_dynamic     |  24.8 |   27   |    40.4 |
| L0_strict     |  22.8 |   23.7 |    34.2 |
| L1_rate10     |  23.6 |   25.5 |    38.3 |
| L2_rate20     |  33.8 |   31.5 |    52.2 |
| L3_rate33     |  44.7 |   38.7 |    63.1 |
| L4_rate50     |  70.2 |   72.3 |    87.1 |
| L5_unfiltered | 100   |  100   |   100   |

Numbers are the percentage of abstracts kept.

## What to train

For each level, two runs:

1. the filtered set,
2. a same-size random control drawn from the unfiltered pool at seed 42.

The control is what separates a real filtering effect from a set-size effect. If
the filtered run beats its control the QC model is selecting useful data; if the
two match, all that happened was the set got smaller.

`L5_unfiltered` is the full set and is its own control, so it needs one run only.
That run is the baseline Chris already has from 2026-08-04.

## Getting the files

The JSONs are not committed: the ladder is roughly 100 MB of near-duplicate data
and it is fully determined by `acceptance_levels.csv`. Build any level locally:

```
python materialize.py --level L2_rate20          # filtered + random control
python materialize.py --level all --splits train dev
```

Output lands in `acceptance_levels/<level>/` in the source schema, so it drops
straight into `load_synthetic_abstracts` the same way the 2026-08-24 files did.
