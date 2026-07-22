# Perturbation taxonomy

Canonical list of perturbation types we apply to BioRED gold relations to produce training data for the quality-filter classifier. The names below are the exact strings used in the `perturbation` column of every built CSV; treat them as a stable vocabulary.

## Scope: which relation types we perturb

The label space is **four classes**: `Association`, `Positive_Correlation`, `Negative_Correlation`, and `NoRelation`. The first three come from BioRED relations (only these have enough examples to evaluate); the five rare types (`Comparison`, `Cotreatment`, `Drug_Interaction`, `Bind`, `Conversion`) are **always dropped** (the `--rare-classes` ablation was removed 2026-06-10). `NoRelation` is a genuine class sampled from co-mentioned but unrelated entity pairs, distance-matched to the real relations (see [Adding NoRelation golds](#adding-norelation-golds)).

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
| `gold` | positive | nothing | establishes the positive class | 1 | all four classes, incl. `NoRelation` |
| `label_flip` | label | relation type, swapped to another class over the full 4-class matrix (one row per alternative) | LLM picks the wrong relation type from a plausible menu (incl. claiming a relation where there is none, dropping a real one, or weakening a correlation to `Association`) | 0 | all four; every off-diagonal cell, no skip |
| `direction_swap` | direction | swap `entity_A` and `entity_B`, keep relation type | LLM reverses causality (e.g. "gene up-regulates compound" vs "compound up-regulates gene") | 0 | `Positive_Correlation`, `Negative_Correlation` only |
| `fp_external` | false positive | replace `entity_B` with an entity that is not in the abstract at all (drawn from the corpus pool) | the model should reject a relation to an entity the abstract never mentions | 0 | the three relation classes (not `NoRelation` golds) |

Two things were removed 2026-06-10 because the 4-class scheme made them redundant:

- **`false_negative`**: dropping a real relation (relation → `NoRelation`) is now just a `label_flip` cell in the 4-class matrix, alongside the new claim-a-relation cells (`NoRelation` → a real type).
- **`fp_co_related` and `fp_co_standalone`**: these replaced `entity_B` with an *in-abstract* entity, so the pair had no relation and was itself a `NoRelation` candidate. That made their rows duplicate the `NoRelation` → relation `label_flip` rows (measured: ~1.2k exact duplicates in train). `fp_external` is kept because its `entity_B` is *outside* the abstract, so it can never be a co-mention pair and never collides. The in-abstract false-positive signal is now carried by the `NoRelation` → relation flips.

## Per-perturbation notes

### `label_flip` walks the full 4-class matrix (every off-diagonal cell)

For each gold we emit one `label_flip` row per alternative class, over every off-diagonal cell of the 4-class matrix, with no skip. A positive/negative correlation does entail a generic `Association`, so on the gold text a specific → `Association` flip would be technically true; but the synthetic abstract is written by an LLM that may soften a correlation down to plain association-level language, making the weaker label a genuine mismatch the QC model must learn to catch. Keeping these cells also trains the model to separate `Association` from the specific correlations instead of collapsing them. So `IMPLIED_RELATIONS` in `perturbations.py` is empty: nothing is skipped.

The 4-class matrix (✓ = valid negative, `self` on the diagonal):

| gold ↓ \ to → | `Association` | `Positive_Correlation` | `Negative_Correlation` | `NoRelation` | # valid |
|---|:---:|:---:|:---:|:---:|:---:|
| `Association` | self | ✓ | ✓ | ✓ | 3 |
| `Positive_Correlation` | ✓ | self | ✓ | ✓ | 3 |
| `Negative_Correlation` | ✓ | ✓ | self | ✓ | 3 |
| `NoRelation` | ✓ | ✓ | ✓ | self | 3 |

So every gold produces 3 `label_flip` rows (the three other classes). The relation → `NoRelation` column is what used to be the separate `false_negative` perturbation; the `NoRelation` → relation row claims a relation where there is none; the specific → `Association` cells are the weakened-label negatives described above.

### `direction_swap` applies only to `Positive_Correlation` and `Negative_Correlation`

`Association` is **non-directional**: the relation only asserts that A and B are connected, with no claim about who acts on whom. If you swap the order of the arguments, you get back the same fact, so a "direction-swapped Association" sample would be labelled wrong while still being semantically correct, which is exactly the wrong training signal.

Only `Positive_Correlation` and `Negative_Correlation` carry direction (`A` goes up, `B` goes up vs `B` going up, `A` going up, and similarly for negative). Those are the only types `direction_swap` is applied to. Frederik made this call on 2026-05-08: Association says two things are connected but not in which direction, so direction swap only makes sense for the two correlation types.

`NoRelation` golds are not directional either, so they are never direction-swapped. The five rare relation types are no longer in the dataset, so the question of direction-swapping them does not arise.

### `fp_external` keeps the gold relation type

`fp_external` replaces `entity_B` with a corpus entity not in the abstract, and **keeps the gold relation's `relation_type`**. We deliberately do not randomize the relation type, because that would overlap with `label_flip` (the model could then detect the fp via the wrong-relation signal rather than via the entity swap). Keeping the gold relation type isolates the wrong-`entity_B` signal, so per-type F1 measures what we claim.

### Both directions of the NoRelation boundary are covered

On 2026-05-08 Frederik asked us to cover claiming a real relation between two co-mentioned-but-unrelated entities. That is now a **`NoRelation` gold flipped to a relation** (`label_flip`): a genuine non-relation pair `(entity_A, NoRelation, entity_B)` relabelled `(entity_A, R, entity_B)`: same pair, wrong claim. (It used to also be produced by `fp_co_standalone`, which is exactly why that family was dropped 2026-06-10: it duplicated these rows.)

The inverse direction, **relation → `NoRelation`** (dropping a real relation), is the relation-row `NoRelation` cell of the `label_flip` matrix (formerly the `false_negative` perturbation). Together these give both directions of the `NoRelation` boundary.

## Per-perturbation difficulty (to be measured)

We track macro-F1 per `perturbation` value, so the per-type difficulty ranking falls out of training once the filter is trained. Frederik mentioned on 2026-04-30 that in-abstract swaps will likely be harder for the model to spot than out-of-abstract ones (more semantic disambiguation needed), but we agreed to verify empirically before locking in any specific ordering.

## Sampling & balance rules

- `direction_swap` is restricted to `Positive_Correlation` and `Negative_Correlation` (see above).
- `label_flip` walks the full 4-class matrix: one row per alternative class per gold, every off-diagonal cell, no skip (see above). No random pick.
- **`NoRelation` gold sampling:** candidates are all co-mentioned entity pairs with no relation of any type. There are ~7x more of these than real relations, and they sit much farther apart in the text, so an unfiltered sample would teach the model "far apart = no relation". When `n_norelation_cap` is set we **distance-match**: each candidate is accepted with probability proportional to how common its in-abstract distance (sentence gap by default) is among the real related pairs, until the cap is hit (seeded; see `_norelation_gold_pairs`). Default cap `match_gold` = the per-split real-relation count, giving a 50/50 real-vs-`NoRelation` gold split.
- `fp_external` samples up to `FP_EXTERNAL_MAX_PER_GOLD` (default 5) candidates per relation gold from the corpus pool (entities not in the abstract), since enumerating the full corpus would blow up the pool. `NoRelation` golds get no FP samples.
- **FP balance cap:** the per-split count of each FP type is capped at the per-split `label_flip` count. If the FP candidate pool exceeds the cap, we randomly downsample (seeded) to exactly the cap; if it is smaller (common now that `NoRelation` golds inflate `label_flip`), every candidate is kept. If a later experiment shows a particular FP type is the hardest to detect, we lift its cap by raising `FP_EXTERNAL_MAX_PER_GOLD` or by generating more candidates per gold.
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

The top 3 cover ~96-97% of every split. `Drug_Interaction` and `Conversion` have **zero examples in dev**, so there is no evaluation signal for them no matter how the filter is trained; `Bind`, `Cotreatment`, `Comparison` are borderline at single-digit dev counts. We therefore **drop all five rare types** and keep only `Association`, `Positive_Correlation`, `Negative_Correlation` as the relation classes (plus `NoRelation`). Built with `python cli.py build`. The `--rare-classes` ablation that used to build a `_full` variant was removed 2026-06-10.

## Adding `NoRelation` golds

`NoRelation` is a first-class gold (label 1), not just a perturbation target. We sample co-mentioned entity pairs that have **no annotated relation of any type** and emit them as `gold` with `relation_type = NoRelation, label = 1`, so the filter learns that "no relation" is a valid ground truth, not an absence. These golds then take part in `label_flip` like any other class (flipped to the three real relations).

In real BioRED there are ~7x more such pairs than annotated relations (~26.7k vs ~4k in train), and they sit much farther apart in the text. We therefore distance-match and cap them: see the `NoRelation` gold sampling rule under [Sampling & balance rules](#sampling--balance-rules). Implemented 2026-06-10 (was issue #7). The cap size and the distance metric are configurable via `n_norelation_cap` and `norelation_distance_metric`; the pipeline defaults to the sentence gap with the `match_gold` cap (character gap is also supported; token is case-study-only).

## One-thing-at-a-time spin-off datasets

For the per-type difficulty analysis, derive a separate spin-off CSV per perturbation type from a master sample pool. Each spin-off must differ from the others in **exactly one dimension**: the perturbation under test. Specifically:

- The set of gold relations covered is the same across spin-offs.
- For each gold, the entity selection used in the perturbation (which entity got picked as the swap target, etc.) is reused across spin-offs where applicable.
- Random seeds for variant selection are shared per gold.

Without this discipline, observed performance differences between perturbation types are confounded by random noise in entity sampling, not signal. Tracked as issue #3.

## Open design questions

- **`NoRelation` cap size & distance metric**: how many `NoRelation` golds to keep per split (pipeline default `match_gold`, a 50/50 split) and which distance to match on (character, token, or sentence gap; pipeline default character).
- **Combo perturbations** (e.g. `label_flip + direction_swap` on the same gold) are not in v1. Could be added later as a harder bucket if single-perturbation accuracy saturates.
