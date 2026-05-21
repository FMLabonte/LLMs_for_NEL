"""
Phase 1 dataset builder (library).

Functions used by cli.py to turn raw BioRED into perturbed training samples.
No CLI in this file; run via `python cli.py build`.
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


def build_split(
    split: str,
    seed: int = 42,
    type_restricted_false_positives: bool = True,
    keep_rare_classes: bool = False,
) -> pd.DataFrame:
    """Build the perturbed dataset for one split, return as DataFrame.

    keep_rare_classes=False (default) drops the 5 rare BioRED relation types
    (Comparison, Cotreatment, Drug_Interaction, Bind, Conversion) before
    perturbation, leaving only Association, Positive_Correlation,
    Negative_Correlation. keep_rare_classes=True keeps the full set as an
    ablation. See PERTURBATIONS.md, "Class-imbalance versions".
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
