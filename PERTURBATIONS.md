# Perturbation taxonomy

Canonical list of perturbation types we apply to BioRED gold relations to produce training data for the quality-filter classifier. The names below are the exact strings used in the `perturbation` column of every built CSV; treat them as a stable vocabulary.

## Scope: which relation types we perturb

The dataset focuses on the three relation types that actually have enough examples to evaluate: `Association`, `Positive_Correlation`, `Negative_Correlation`. The other five (`Comparison`, `Cotreatment`, `Drug_Interaction`, `Bind`, `Conversion`) are dropped from the perturbed dataset; the CLI flag is `--rare-classes` (off by default; opt in to include them as an ablation).

This is the default the filter is trained and evaluated on. The motivation and per-class counts are in the [Class-imbalance versions](#class-imbalance-versions) section below.

## How to read the table

- **Family** groups perturbations that share a mechanism. Useful when slicing metrics later.
- **Mutates** says what part of the gold tuple `(entity_A, relation_type, entity_B)` is replaced.
- **Why** is the LLM error mode we hope this perturbation teaches the filter to catch.
- **Label** is the binary classifier target. `1` for the unmodified gold, `0` for everything else.
- **Applies to** says which relation types the perturbation is meaningful for. When a perturbation cannot be meaningfully applied to a gold relation, no sample is emitted for that (gold, perturbation) pair.

## Catalog

| Name | Family | Mutates | Why | Label | Applies to |
|---|---|---|---|---|---|
| `gold` | positive | nothing | establishes the positive class | 1 | all kept relation types |
| `label_flip` | label | relation type, swapped to a different one (one row per alternative kept type) | LLM picks the wrong relation type from a plausible menu | 0 | all kept relation types |
| `direction_swap` | direction | swap `entity_A` and `entity_B`, keep relation type | LLM reverses causality (e.g. "gene up-regulates compound" vs "compound up-regulates gene") | 0 | `Positive_Correlation`, `Negative_Correlation` only |
| `fp_co_related` | false positive | replace `entity_B` with another entity that is in the abstract AND has its own annotated relation (just not with `entity_A`) | hardest false-positive case: both entities are independently real in the text, the model has to decide *this specific pairing* is fabricated | 0 | all kept relation types |
| `fp_co_standalone` | false positive | replace `entity_B` with an entity that is in the abstract but has no annotated relation at all | medium false-positive: the entity is co-mentioned but bears no real relation in the abstract | 0 | all kept relation types |
| `fp_external` | false positive | replace `entity_B` with an entity that is not in the abstract at all (drawn from the corpus pool) | easiest false-positive: detectable from text presence alone, weak baseline | 0 | all kept relation types |
| `false_negative` | removal | replace the relation label with `NoRelation` for a triple that does have a real relation | LLM drops a real relation it should have kept | 0 | all kept relation types |

## Per-perturbation notes

### `label_flip` enumerates every alternative kept label per gold

For each gold relation we emit one `label_flip` row per alternative kept relation type. Under the reduced default (`Association`, `Positive_Correlation`, `Negative_Correlation`) that means every gold produces 2 `label_flip` rows: one swap to each of the other two types. With `--rare-classes` every gold produces 7 `label_flip` rows.

Reason: keeps the entity pair fixed across all label-swap directions, so per-direction difficulty (e.g. `Positive_Correlation → Association` vs `Positive_Correlation → Negative_Correlation`) can be measured without being confounded by different entity pairs. Decided with Frederik on 2026-05-21.

### `direction_swap` applies only to `Positive_Correlation` and `Negative_Correlation`

`Association` is **non-directional**: the relation only asserts that A and B are connected, with no claim about who acts on whom. If you swap the order of the arguments, you get back the same fact, so a "direction-swapped Association" sample would be labelled wrong while still being semantically correct, which is exactly the wrong training signal.

Only `Positive_Correlation` and `Negative_Correlation` carry direction (`A` goes up, `B` goes up vs `B` going up, `A` going up, and similarly for negative). Those are the only types `direction_swap` is applied to. Frederik made this call on 2026-05-08: Association says two things are connected but not in which direction, so direction swap only makes sense for the two correlation types.

The rare classes `Comparison`, `Cotreatment`, `Drug_Interaction`, `Bind`, `Conversion` are not in the perturbed dataset by default, so the question of direction-swapping them does not arise.

### `fp_*` keep the gold relation type

`fp_co_related`, `fp_co_standalone`, `fp_external` all replace `entity_B`. They **keep the gold relation's `relation_type`**. We deliberately do not randomize the relation type for fp samples, because that would overlap with `label_flip` (the model could then detect fp via the wrong-relation signal rather than via the entity swap). When we keep the gold relation type, the wrong-`entity_B` signal stays isolated, so per-type F1 actually measures what we claim.

### The `NoRelation -> real_relation` case is `fp_co_standalone`

On 2026-05-08 Frederik asked us to explicitly cover the case where two co-mentioned-but-unrelated entities get assigned one of the real relation labels. That case is already produced by `fp_co_standalone`, just framed from the other direction:

- `fp_co_standalone` starts from a real gold `(entity_A, R, entity_B)` triple
- replaces `entity_B` with an in-abstract entity that has **no annotated relations**
- keeps the gold's relation label `R`

The output is a sample `(entity_A, R, entity_B')` where the gold answer for that pair is `NoRelation`. That is exactly "take a NoRelation pair, claim a real relation `R` between them". No separate `no_rel_to_relation` label is added; `fp_co_standalone` already covers it. Frederik just wanted it called out, not given its own label.

### `false_negative` is the inverse case

`false_negative` is the other side: take a real relation and relabel it as `NoRelation` (so the gold answer is `R` and the claimed answer is `NoRelation`). Together with `fp_co_standalone` this gives both directions of the NoRelation boundary.

## Per-perturbation difficulty (to be measured)

We track macro-F1 per `perturbation` value, so the per-type difficulty ranking falls out of training once the filter is trained. Frederik mentioned on 2026-04-30 that in-abstract swaps will likely be harder for the model to spot than out-of-abstract ones (more semantic disambiguation needed), but we agreed to verify empirically before locking in any specific ordering.

## Sampling & balance rules

- `direction_swap` is restricted to `Positive_Correlation` and `Negative_Correlation` (see above).
- `label_flip` enumerates every alternative kept relation type per gold (one row per alternative). No random pick; the data set sees every label-swap direction on every entity pair.
- `fp_co_related` and `fp_co_standalone` collect ALL eligible in-abstract candidates per gold (the in-abstract entity pool is small, typically under 20). `fp_external` samples up to `FP_EXTERNAL_MAX_PER_GOLD` (default 5) candidates per gold from the corpus pool, since enumerating the full corpus would blow up the pool.
- **FP balance cap:** the per-split count of each FP type is capped at the per-split `label_flip` count. If the FP candidate pool exceeds the cap, we randomly downsample (seeded) to exactly the cap. This keeps all six perturbation types in the same order of magnitude. If a later experiment shows a particular FP type is the hardest to detect, we lift its cap by raising `FP_EXTERNAL_MAX_PER_GOLD` or by generating more candidates per gold.
- False-positive sampling is **type-restricted by default**: only `(entity_type_a, entity_type_b)` pairs that appear in real BioRED relations are eligible. This avoids implausible Species/CellLine pairings the model would learn to reject trivially. `--no-type-restrict` disables.
- **Reproducibility:** same seed → same CSV every run. The seeded `random.Random` is used both for the `fp_external` per-gold subsample and the per-split downsample to the cap.

## Class-imbalance versions

BioRED is dominated by 3 relation types. Per-class counts across the official splits:

| Relation type | train | dev | test |
|---|---:|---:|---:|
| `Association` | 2192 (52.5%) | 560 (48.2%) | 635 (54.6%) |
| `Positive_Correlation` | 1089 (26.1%) | 352 (30.3%) | 325 (27.9%) |
| `Negative_Correlation` | 763 (18.3%) | 216 (18.6%) | 171 (14.7%) |
| `Bind` | 61 (1.5%) | 19 (1.6%) | 9 (0.8%) |
| `Cotreatment` | 31 (0.7%) | 10 (0.9%) | 14 (1.2%) |
| `Comparison` | 28 (0.7%) | 5 (0.4%) | 6 (0.5%) |
| `Drug_Interaction` | 11 (0.3%) | **0** | 2 (0.2%) |
| `Conversion` | 3 (0.1%) | **0** | 1 (0.1%) |

The top 3 cover ~96-97% of every split. `Drug_Interaction` and `Conversion` have **zero examples in dev**, which means there is no evaluation signal for them no matter how the filter is trained. `Bind`, `Cotreatment`, `Comparison` are borderline at single-digit dev counts. We therefore build the dataset in two variants, both available from the same CLI:

- **reduced (default):** keep only `Association`, `Positive_Correlation`, `Negative_Correlation`. Built with `python cli.py build` (no flag) and used as the main training and evaluation set.
- **full:** every relation type included. Built with `python cli.py build --rare-classes` and used as an ablation to confirm that adding the rare classes does not help (and that they are not a stealthy source of label noise).

Both versions share the same perturbation logic; only the set of gold relations that survives into the perturbation loop changes.

## Adding `NoRelation` as a positive-class entity pair

`NoRelation` is the relation label used by the `false_negative` perturbation. To make `NoRelation` a first-class part of the training distribution (so the filter learns that "no relation" is a valid ground truth, not an absence), we also sample co-mentioned entity pairs that have no annotated relation and emit them as `gold` with `relation_type = NoRelation, label = 1`. Without this step `NoRelation` only ever appears with `label = 0`, which teaches the model the wrong thing.

This needs conscious balancing: in real BioRED, the count of co-mentioned-but-unrelated pairs is far larger than the count of annotated relations. Cap the per-abstract count to keep `NoRelation` from dominating. Tracked as issue #7; not yet implemented.

## One-thing-at-a-time spin-off datasets

For the per-type difficulty analysis, derive a separate spin-off CSV per perturbation type from a master sample pool. Each spin-off must differ from the others in **exactly one dimension**: the perturbation under test. Specifically:

- The set of gold relations covered is the same across spin-offs.
- For each gold, the entity selection used in the perturbation (which entity got picked as the swap target, etc.) is reused across spin-offs where applicable.
- Random seeds for variant selection are shared per gold.

Without this discipline, observed performance differences between perturbation types are confounded by random noise in entity sampling, not signal. Tracked as issue #3.

## Open design questions

- **Multi-ID umbrella entities** (e.g. `MODY → 3172,3651,6927`) currently get dropped (~8% of gold relations). Options: pick first ID, expand to multiple rows, skip the relation entirely. Default for now is skip; revisit before scaling up.
- **Combo perturbations** (e.g. `label_flip + direction_swap` on the same gold) are not in v1. Could be added later as a harder bucket if single-perturbation accuracy saturates.
- **`false_negative` entity-swap variant:** should we also generate false-negatives where `entity_B` is *replaced* (so the negative is about a different pair than the gold)? Current implementation only relabels the existing pair to `NoRelation`.
