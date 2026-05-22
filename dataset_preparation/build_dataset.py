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
    keep_rare_classes: bool = False,
) -> pd.DataFrame:
    """Build the perturbed dataset for one split, return as DataFrame.

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
    keep_rare_classes
        False (default) drops the 5 rare BioRED relation types (Comparison,
        Cotreatment, Drug_Interaction, Bind, Conversion) before perturbation,
        leaving only Association, Positive_Correlation, Negative_Correlation.
        True keeps the full 8 as an ablation. See PERTURBATIONS.md,
        "Class-imbalance versions".

    Returns
    -------
    pandas.DataFrame
        One row per training sample. Columns match `TrainingSample.to_dict()`:
        pmid, abstract, entity_a_id/text/type, relation_type,
        entity_b_id/text/type, label, perturbation.
    """
    meta, anns, rels = load_biored_split(split)
    assert_known_relation_types(rels, split)

    kept_relation_types = (
        list(BIORED_RELATION_TYPES) if keep_rare_classes else list(TOP_RELATION_TYPES)
    )

    samples = build_training_samples(
        meta,
        anns,
        rels,
        seed=seed,
        type_restricted_false_positives=type_restricted_false_positives,
        kept_relation_types=kept_relation_types,
    )
    return pd.DataFrame([s.to_dict() for s in samples])