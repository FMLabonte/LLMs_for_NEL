"""
Phase 1 dataset builder.

Public entry point is `build_perturbed_dataframe`. The CLI in cli.py calls it;
notebooks can import it directly and skip the CSV round-trip:

    from dataset_preparation.build_dataset import build_perturbed_dataframe
    df = build_perturbed_dataframe("train")

Reproducibility: the same seed always produces the same DataFrame (rows,
ordering, label distribution). Verified post `label_flip` rework on 2026-05-21.
"""

from __future__ import annotations

import pandas as pd

from data_loader import load_biored_split
from perturbations import (
    BIORED_RELATION_TYPES,
    TOP_RELATION_TYPES,
    build_training_samples,
)


def assert_known_relation_types(rels: pd.DataFrame, split: str) -> None:
    """Fail loudly if BioRED contains a relation type we didn't list.

    Catches the case where the BioRED maintainers add a new relation type and
    our perturbation logic silently doesn't cover it.
    """
    observed = set(rels["relation_type"].unique())
    unknown = observed - set(BIORED_RELATION_TYPES)
    if unknown:
        raise ValueError(
            f"[{split}] Unknown relation types in BioRED: {sorted(unknown)}. "
            f"Add them to BIORED_RELATION_TYPES in perturbations.py."
        )


def build_perturbed_dataframe(
    split: str,
    seed: int = 42,
    type_restricted_false_positives: bool = True,
    n_norelation_cap: int | str | None = "match_gold",
    norelation_distance_metric: str = "sentence",
) -> pd.DataFrame:
    """Build the perturbed dataset for one split, return as DataFrame.

    The label space is the 4 kept classes: Association, Positive_Correlation,
    Negative_Correlation, and NoRelation. The 5 rare BioRED relation types
    (Comparison, Cotreatment, Drug_Interaction, Bind, Conversion) are always
    dropped (the --rare-classes ablation was removed 2026-06-10).

    Parameters
    ----------
    split
        "train", "dev", or "test" (matches the keys in data_loader.SPLIT_FILES).
    seed
        Random seed for the perturbation sampler. Same seed gives the same
        DataFrame every call (same rows, same ordering, same FP downsampling).
    type_restricted_false_positives
        When True (default), FP sampling is restricted to entity-type pairs
        actually observed in BioRED relations. Avoids implausible pairings
        like Species/CellLine that LLMs would never plausibly hallucinate.
    n_norelation_cap
        How many distance-matched NoRelation golds to keep per split. An int sets
        an absolute cap; "match_gold" (default) caps to the number of real top-3
        relations in the split (a 50/50 relation-vs-NoRelation gold split); None
        keeps every candidate (~7x the positives, unbalanced).
    norelation_distance_metric
        Distance used to match NoRelation golds to the real related pairs.
        "sentence" (default, Frederik's preference) counts sentence terminators
        between the closest mentions; "char" is the in-abstract character gap.

    Returns
    -------
    pandas.DataFrame
        One row per training sample. Columns match `TrainingSample.to_dict()`:
        pmid, abstract, entity_a_id/text/type, relation_type,
        entity_b_id/text/type, label, perturbation.
    """
    meta, anns, rels = load_biored_split(split)
    assert_known_relation_types(rels, split)

    gold_relation_types = list(TOP_RELATION_TYPES)
    cap = n_norelation_cap
    if cap == "match_gold":
        cap = int(rels["relation_type"].isin(set(gold_relation_types)).sum())

    samples = build_training_samples(
        meta,
        anns,
        rels,
        seed=seed,
        type_restricted_false_positives=type_restricted_false_positives,
        gold_relation_types=gold_relation_types,
        n_norelation_cap=cap,
        norelation_distance_metric=norelation_distance_metric,
    )
    return pd.DataFrame([s.to_dict() for s in samples])