# QC-filtered synthetic abstracts

Produced 2026-08-24 with the run-2 QC model (BERT 2). Every relation claim in
every synthetic abstract was scored, and the abstracts that survived the
rejection rule are collected in `filtered/`. This is the missing fourth row of the
comparison table in `finetuning/SYNTH_COMPARE.md`.

## Using it

The files keep Fred's schema exactly, so `load_synthetic_abstracts` and
`build_synthetic_parsed` in `dataset_preparation/synthetic_abstracts.py` read
them unchanged. Only the path changes:

```python
meta, anns, rels = build_synthetic_parsed(
    "qc_filtering/filtered/results_qwen3_8b_train.json",
    "Data/BioRED/Train.PubTator",
)
```

Papers whose generations were all rejected are absent from the file. Papers where
some generations survived carry only those, and since `build_synthetic_parsed`
skips missing generation keys, nothing downstream needs to change. All twelve
files were checked against that function before being written here.

## The files

| file | papers | abstracts | of original |
|---|---|---|---|
| `filtered/results_qwen3_4b_train.json` | 205 | 492 | 41.6% |
| `filtered/results_qwen3_4b_dev.json` | 32 | 78 | 26.5% |
| `filtered/results_qwen3_4b_test.json` | 35 | 83 | 27.7% |
| `filtered/results_qwen3_8b_train.json` | 212 | 501 | 42.4% |
| `filtered/results_qwen3_8b_dev.json` | 28 | 71 | 24.1% |
| `filtered/results_qwen3_8b_test.json` | 34 | 82 | 27.3% |

`filtered/random_qwen3_*.json` are **same-size random controls**, drawn from the same
papers with seed 42 and no reference to the QC scores. Fred asked for this arm
explicitly. Without it, a win for the filtered set cannot be told apart from an
effect of simply training on a smaller, less repetitive set.

`filtered/paper_ids_{train,dev,test}.txt` list the BioRED papers the synthetic data was
generated from, for the matched-real arm. Fred was explicit that the real side
must be matched by paper id rather than randomly sampled. The split mapping was
verified: synthetic train, dev and test sit entirely inside BioRED Train, Dev and
Test respectively, with every id resolving and no crossover.

## Two things to know before reading any result

**1. The filter is biased toward abstracts with few relations.** This is a
property of the filter, measured back in July and clearly visible in what
survived here:

| relations in the abstract | abstracts | kept |
|---|---|---|
| 1 to 3 | 888 | 68.1% |
| 4 to 8 | 1,182 | 32.3% |
| 9 to 12 | 522 | 33.9% |
| 13 or more | 954 | 15.0% |

Median relation count is 4 in the kept set against 9 in the rejected set. Length
barely differs, 175 words against 202. The cause is arithmetic rather than
anything semantic: per-claim recall is about 0.73, so an abstract with k claims
survives at roughly 0.73^k.

The consequence for the comparison is concrete. The filtered train file and its
random control hold the same number of abstracts, but the filtered one carries
about 30% fewer gold relations, because it selected simpler papers. A win for
the filtered arm therefore cannot be attributed to data quality alone without
saying this out loud.

**2. Rare relation types were excluded from the rejection count.** Fred generated
from the full BioRED relation set, so 18.8% of the synthetic abstracts assert at
least one Bind, Cotreatment, Comparison, Drug_Interaction or Conversion relation.
The June redesign dropped those five types, so the QC model has never been
trained on them and its verdict on such a claim is out of distribution. Letting
an unscoreable claim reject an otherwise sound abstract seemed worse than
ignoring it, so those claims are not counted as errors. This is a judgement call
and Fred has been asked to confirm it.

## Rebuilding it differently

The scoring and the decision are separate steps on purpose. Scoring cost 38
minutes; changing the rule costs seconds.

- `synthetic_relation_scores.csv` holds all 35,658 scored claims, one row per
  claim, with `prob_supported`.
- `abstract_decisions.csv` holds one row per synthetic abstract with its
  failed-claim count and minimum probability.
- `python decide.py --rule strict --rares include --cutoff 0.5` rewrites this
  folder under any other rule.

The rule used here is `dynamic` with rare types excluded at cut-off 0.5. Dynamic
means the tolerated number of failed claims scales with the relation count, 0 up
to 8 relations, 1 up to 12, then 2, which was the best step function found in the
July grid search. The four combinations give:

| rule | rare types | abstracts kept | share |
|---|---|---|---|
| strict | included | 1,001 | 28.2% |
| strict | excluded | 1,089 | 30.7% |
| dynamic | included | 1,183 | 33.3% |
| dynamic | excluded | 1,307 | 36.8% |

## Side finding

7.4% of the relation claims name an entity by a bare NCBI Gene ID rather than a
name, because the entity-name lookup in the generation pipeline falls back to the
identifier. In 45.7% of those cases the generator wrote the number into the
abstract as if it were a name, for instance "CTR9 binds to Rtf1 and an unknown
factor designated 54624". 49 papers are affected, and the unfiltered training
data already used them. Details and the caveat that cuts against over-claiming
this are in `GENERATOR_BUG.md`.
