# Perturbation taxonomy

Canonical list of perturbation types we apply to BioRED gold relations to produce training data for the quality-filter classifier. The names below are the exact strings used in the `perturbation` column of every built CSV; treat them as a stable vocabulary.

## How to read the table

- **Family** groups perturbations that share a mechanism. Useful when slicing metrics later.
- **Mutates** says what part of the gold tuple `(entity_A, relation_type, entity_B)` is replaced.
- **Why** is the LLM error mode we hope this perturbation teaches the filter to catch.
- **Label** is the binary classifier target. `1` for the unmodified gold, `0` for everything else.

## Catalog

| Name | Family | Mutates | Why | Label |
|---|---|---|---|---|
| `gold` | positive | nothing | establishes the positive class | 1 |
| `label_flip` | label | relation type, swapped to a different one (uniform random over the 7 alternatives) | LLM picks the wrong relation type from a plausible menu | 0 |
| `direction_swap` | direction | swap `entity_A` and `entity_B`, keep relation type | LLM reverses causality (gene treats disease vs disease causes gene) | 0 |
| `fp_co_related` | false positive | replace `entity_B` with another entity that is in the abstract AND has its own annotated relation (just not with `entity_A`) | hardest false-positive case: both entities are independently real in the text, the model has to decide *this specific pairing* is fabricated | 0 |
| `fp_co_standalone` | false positive | replace `entity_B` with an entity that is in the abstract but has no annotated relation | medium false-positive: the entity is co-mentioned but bears no real relation in the abstract | 0 |
| `fp_external` | false positive | replace `entity_B` with an entity that is not in the abstract at all (drawn from the corpus pool) | easiest false-positive: detectable from text presence alone, weak baseline | 0 |
| `false_negative` | removal | replace the relation label with `NoRelation` for a triple that does have a real relation | LLM drops a real relation it should have kept | 0 |

## Expected per-type difficulty (rough, to verify empirically)

`label_flip` ≈ `direction_swap` < `fp_external` < `fp_co_standalone` < `fp_co_related`

The intuition: detecting that an entity is missing from the abstract is essentially string lookup; detecting that two co-mentioned entities are *unrelated* requires real semantic understanding. Frederik flagged this directly in meeting 2: "swapping this one to something that is in the same abstract should be harder for the model to differentiate than something that isn't even in the abstract."

We track macro-F1 per `perturbation` value so this hypothesis is testable from the metrics we already log.

## Sampling & balance rules

- Direction swap is **skipped on symmetric relations** (`Association`, `Comparison`). Applying it would create a sample we label wrong but is semantically still correct.
- False-positive sampling is **type-restricted by default**: only `(entity_type_a, entity_type_b)` pairs that appear in real BioRED relations are eligible. This avoids implausible Species/CellLine pairings the model would learn to reject trivially. `--no-type-restrict` disables.
- Each gold relation produces at most one sample of each perturbation type. To average out random choices (which entity got swapped, which alt-label was picked), generate **N variants per gold per type** (default N=5; CLI: `--variants N`).

## Class-imbalance versions

BioRED is dominated by `Association`, `Negative_Correlation`, `Positive_Correlation`. The other five relation types (`Comparison`, `Cotreatment`, `Drug_Interaction`, `Bind`, `Conversion`) each have under 100 examples in train. We build the dataset in two variants:

- **full:** every relation type included.
- **reduced:** the five rare types are dropped before perturbation.

Train both, compare downstream filter performance.

## Adding `NoRelation` as a positive-class entity pair

`NoRelation` is the relation label used by the `false_negative` perturbation. To make `NoRelation` a first-class part of the training distribution (so the filter learns that "no relation" is a valid ground truth, not an absence), we also sample co-mentioned entity pairs that have no annotated relation and emit them as `gold` with `relation_type = NoRelation, label = 1`. Without this step `NoRelation` only ever appears with `label = 0`, which teaches the model the wrong thing.

This needs conscious balancing: in real BioRED, the count of co-mentioned-but-unrelated pairs is far larger than the count of annotated relations. Cap the per-abstract count to keep `NoRelation` from dominating.

## One-thing-at-a-time spin-off datasets

For the per-type difficulty analysis, derive a separate spin-off CSV per perturbation type from a master sample pool. Each spin-off must differ from the others in **exactly one dimension**: the perturbation under test. Specifically:

- The set of gold relations covered is the same across spin-offs.
- For each gold, the entity selection used in the perturbation (which entity got picked as the swap target, etc.) is reused across spin-offs where applicable.
- Random seeds for variant selection are shared per gold.

Without this discipline, observed performance differences between perturbation types are confounded by random noise in entity sampling, not signal.

## Open design questions

- **Multi-ID umbrella entities** (e.g. `MODY → 3172,3651,6927`) currently get dropped (~8% of gold relations). Options: pick first ID, expand to multiple rows, skip the relation entirely. Default for now is skip; revisit before scaling up.
- **Combo perturbations** (e.g. `label_flip + direction_swap` on the same gold) are not in v1. Could be added later as a harder bucket if single-perturbation accuracy saturates.
- **`false_negative` entity-swap variant:** should we also generate false-negatives where `entity_B` is *replaced* (so the negative is about a different pair than the gold)? Current implementation only relabels the existing pair to `NoRelation`.