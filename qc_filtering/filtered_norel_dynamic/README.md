# NoRelation-aware filtering under the dynamic rule

Built 2026-09-13 with `filter_norel_aware.py`. This is the `passed` column of
`../abstract_decisions_norel.csv`. Same scoring run as `../filtered_norel/`, different
decision rule.

**Use this one, not `../filtered_norel/`, when comparing against the `../filtered/` arm.**

## Why it exists

`../filtered_norel/` uses the strict rule: one failed claim rejects the abstract. That was
chosen on the assumption that Christoph's QC arm was also strict, so the only thing
changing between his old run and the new one would be which claims the filter looked at.

The assumption was wrong. His QC arm is the **dynamic** set, the 212-paper `../filtered/`
files, not the 173-paper strict one. So comparing his existing run against
`filtered_norel/` moves the rule **and** the claim set at once, and a difference in the
downstream numbers cannot be attributed to either.

These files hold the rule constant. Same step function `decide.py` applies, now judging
the implicit NoRelation pairs as well, so the only thing that moves is the claim set. That
is the drop-in Frederik asked for at meeting 10.

## `passed` means two different numbers. Always name the file.

| file | `passed` | rule | claims judged |
|---|---|---|---|
| `../abstract_decisions.csv` | 1,307 | dynamic | stated relations |
| `../abstract_decisions_norel.csv` | **779** | dynamic | stated + implicit |

Same rule, different claim set. That is the whole point, and it is also the easiest thing
in this folder to misquote. The repo has already lost time twice to a number that was
right in one file and wrong in another (3,552 against 3,546, and 212 against 501).

The norel CSV carries one other decision column, `passed_strict`, which is the strict rule
on the same claims, 615 abstracts. `../filtered_norel/` is built from it.

The stated-only verdicts are not duplicated here. The dynamic one is `passed` in
`../abstract_decisions.csv`; the strict one is `L0_strict` in
`../acceptance_levels/acceptance_levels.csv`, 1,083 abstracts. Both are still derivable
from the raw counts in this CSV (`failed_stated == 0` gives L0_strict exactly).

## Using it

Same schema as everything else in `qc_filtering/`, so it is a path change:

```python
meta, anns, rels = build_synthetic_parsed(
    "qc_filtering/filtered_norel_dynamic/results_qwen3_8b_train.json",
    "Data/BioRED/Train.PubTator",
)
```

Checked against `build_synthetic_parsed` before writing: 276 documents, 7,382 annotation
rows, 1,407 gold relations for the 8B train file.

## The rule

An abstract is kept when its failed claims do not exceed the allowance, counting both the
stated relations and the implicit NoRelation pairs. Cut-off 0.5 throughout. The allowance
is the 2026-07-22 step function: 0 failures up to 8 claims, 1 for 9 to 12, 2 for 13 or
more.

## What it costs

| model | split | abstracts | `passed_strict` | `passed` | papers |
|---|---|---|---|---|---|
| qwen3_4b | dev | 294 | 44 (15.0%) | 54 (18.4%) | 22 / 98 |
| qwen3_4b | test | 300 | 54 (18.0%) | 64 (21.3%) | 25 / 100 |
| qwen3_4b | train | 1179 | 204 (17.3%) | 276 (23.4%) | 112 / 394 |
| qwen3_8b | dev | 294 | 41 (13.9%) | 50 (17.0%) | 21 / 98 |
| qwen3_8b | test | 300 | 52 (17.3%) | 59 (19.7%) | 24 / 100 |
| qwen3_8b | train | 1179 | 220 (18.7%) | 276 (23.4%) | 112 / 394 |

Overall 615 becomes 779 of 3,546, so 17.3% becomes 22.0%. For the 8B train file the set
grows from 220 abstracts over 87 papers to 276 over 112.

## Read this before interpreting anything

**1. "Dynamic" does less work here than the name suggests.** The step function was tuned
in July on stated relations, where the median abstract carries 7 claims, and it caps at 2
failures. Folding in the implicit pairs moves the median to 15, the mean to 39 and the
maximum to 703. So 61% of abstracts now land in the top bucket and get exactly two
forgiven failures, whether they carry 13 claims or 703. It is closer to "strict plus two"
than to a genuinely size-scaled rule. It is still much nearer Christoph's arm than strict
is, which is the point, but it is not the July rule doing the July job.

**2. The size bias from `../filtered_norel/README.md` still applies**, just less sharply.
The QC model false-alarms on roughly 15% of negatives, real text and generated text alike
(14.6% against 14.7% on the same 3,390 pairs), so an abstract with many implicit pairs
collects a rejection almost regardless of how well it is written. Two forgiven failures
soften that but do not remove it. A drop in the downstream numbers on this set is still
not evidence that the synthetic data is bad.

**3. The set is still less than half the size of the arm it is compared against.** 276
abstracts over 112 papers against 501 over 212 for `../filtered/`. Any difference mixes
the effect of judging negatives with the effect of training on a smaller set. A random
control matched on both papers and abstracts would separate them.

**4. It is not a clean subset of `../filtered/`.** `decide.py` counts rare-type claims
toward the relation count when it computes the allowance and only stops counting them as
failures; this script drops all non-top-3 claims before it starts. So a rare claim that
can never fail still buys allowance in `decide.py` and not here. That is worth 41
abstracts, all in one direction, and it is why one file has 3,552 rows and the other
3,546. It changes no conclusion, but do not expect the two `passed` columns to nest.

Workings for the false-alarm control: `work/2026-09-09_overnight/`, `task3_norelation.py`.
