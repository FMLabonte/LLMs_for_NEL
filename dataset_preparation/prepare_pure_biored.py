"""
Prepare the original (non-perturbated) BioRED dataset for relation classification.

Unlike :mod:`dataset_preparation.perturbations`, which synthesises negative
examples by corrupting gold relations, this module keeps every gold relation as
it is and *adds* explicit ``NoRelation`` examples sampled from entity pairs that
are co-mentioned in an abstract but never linked by a gold relation.

The sampling of ``NoRelation`` pairs is guided by the textual distance (in
characters) between the two entities:

  1. We first measure the mean and max distance between *related* entities over
     the whole dataset (:func:`compute_relation_distance_stats`).
  2. For every abstract we then collect all unrelated co-mentioned pairs, drop
     any whose distance exceeds the observed maximum, and keep the ones whose
     distance is closest to the observed mean.
  3. Per abstract we add at most as many ``NoRelation`` examples as there are
     gold relations. Fewer is fine when not enough suitable pairs exist.

The resulting DataFrame has the exact same columns as
:func:`dataset_preparation.perturbations.samples_to_rels_like_df`, so the
downstream notebook can treat ``BioRed`` and ``BioRedPerturbated`` identically:
gold relations carry ``perturbation == "gold"`` while ``NoRelation`` examples
carry ``perturbation == "false_positive"``.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

from dataset_preparation.perturbations import (
    BIORED_RELATION_TYPES,
    NO_RELATION_LABEL,
    TrainingSample,
    _real_pairs_per_paper,
    samples_to_rels_like_df,
)
from pubtator_parser import parse_pubtator

Span = tuple[int, int]


def _build_entity_spans(anns: pd.DataFrame) -> dict[tuple[str, str], dict]:
    """Build a per-paper entity index keyed by ``(pmid, mesh_id)``.

    Each value holds the canonical (longest) mention, the entity type and every
    character span ``(start, end)`` at which the entity is mentioned in the
    document. Spans are required to compute inter-entity distances.

    BioRED occasionally stores a single annotation under a comma-joined
    ``mesh_id`` (e.g. ``"1508,836,841"``) while the gold relations reference the
    individual component IDs. To keep those relations resolvable, each component
    ID is indexed separately and inherits the joint annotation's span, mention
    and entity type. Non-joined IDs are unaffected (they are their own single
    component).
    """
    aggregated: dict[tuple[str, str], dict] = {}
    for row in anns.itertuples(index=False):
        pmid: str = row.pmid
        mention: str = "" if pd.isna(row.mention) else str(row.mention)
        entity_type: str = str(row.entity_type)
        span: Span | None = None
        if not (pd.isna(row.start) or pd.isna(row.end)):
            span = (int(row.start), int(row.end))

        for component in str(row.mesh_id).split(","):
            component = component.strip()
            if not component:
                continue
            entry = aggregated.setdefault(
                (pmid, component),
                {"mentions": [], "entity_type": entity_type, "spans": []},
            )
            if mention:
                entry["mentions"].append(mention)
            if span is not None:
                entry["spans"].append(span)

    index: dict[tuple[str, str], dict] = {}
    for key, entry in aggregated.items():
        mentions: list[str] = entry["mentions"]
        canonical: str = max(mentions, key=len) if mentions else ""
        index[key] = {
            "mention": canonical,
            "entity_type": entry["entity_type"],
            "spans": entry["spans"],
        }
    return index


def _span_gap(span_a: Span, span_b: Span) -> int:
    """Return the number of characters between two spans (0 if they overlap)."""
    start_a, end_a = span_a
    start_b, end_b = span_b
    return max(0, max(start_a, start_b) - min(end_a, end_b))


def entity_pair_distance(spans_a: list[Span], spans_b: list[Span]) -> float | None:
    """Smallest character gap between any mention of two entities.

    Returns ``None`` when either entity has no usable span, in which case the
    pair cannot participate in distance-based sampling.
    """
    if not spans_a or not spans_b:
        return None
    return float(min(_span_gap(a, b) for a in spans_a for b in spans_b))


def compute_relation_distance_stats(
    meta: pd.DataFrame,
    anns: pd.DataFrame,
    rels: pd.DataFrame,
    relation_types: list[str] = BIORED_RELATION_TYPES,
    verbose: bool = True,
) -> dict[str, float]:
    """Explore the textual distance between *related* entities in BioRED.

    Iterates over every gold relation whose type is in ``relation_types`` and
    whose two entities can be located in the annotations, computes the character
    distance between them and aggregates summary statistics.

    Returns:
        A dict with ``count``, ``mean``, ``max``, ``min``, ``median`` and
        ``std`` of the related-entity distances.
    """
    entity_index = _build_entity_spans(anns)
    valid_pmids = set(meta["pmid"])
    relation_type_set = set(relation_types)

    distances: list[float] = []
    for _, row in rels.iterrows():
        if row["relation_type"] not in relation_type_set:
            continue
        pmid = row["pmid"]
        if pmid not in valid_pmids:
            continue
        info_a = entity_index.get((pmid, row["id_1"]))
        info_b = entity_index.get((pmid, row["id_2"]))
        if not info_a or not info_b:
            continue
        distance = entity_pair_distance(info_a["spans"], info_b["spans"])
        if distance is not None:
            distances.append(distance)

    if not distances:
        raise ValueError("No related entity pairs with resolvable spans were found.")

    arr = np.asarray(distances, dtype=float)
    stats: dict[str, float] = {
        "count": float(arr.size),
        "mean": float(arr.mean()),
        "max": float(arr.max()),
        "min": float(arr.min()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
    }

    if verbose:
        print("Related-entity distance statistics (characters):")
        print(f"  pairs  : {int(stats['count'])}")
        print(f"  mean   : {stats['mean']:.1f}")
        print(f"  median : {stats['median']:.1f}")
        print(f"  std    : {stats['std']:.1f}")
        print(f"  min    : {stats['min']:.1f}")
        print(f"  max    : {stats['max']:.1f}")

    return stats


def build_pure_biored_samples(
    meta: pd.DataFrame,
    anns: pd.DataFrame,
    rels: pd.DataFrame,
    distance_stats: dict[str, float] | None = None,
    relation_types: list[str] = BIORED_RELATION_TYPES,
    seed: int = 42,
    no_relation_ratio: float | None = 1.0,
    verbose: bool = True,
) -> list[TrainingSample]:
    """Build gold + distance-matched ``NoRelation`` samples for pure BioRED.

    For every abstract all gold relations are kept and, on top of them, up to
    ``no_relation_ratio`` times as many ``NoRelation`` examples are added.
    Candidate unrelated pairs are restricted to a distance no larger than the
    observed maximum and ranked by closeness to the observed mean related-entity
    distance.

    Args:
        distance_stats: Pre-computed output of
            :func:`compute_relation_distance_stats`. Computed on the fly when
            omitted.
        seed: Seed used only to break ties between equally mean-close candidates.
        no_relation_ratio: How many ``NoRelation`` examples to keep per gold
            relation of an abstract. ``None`` keeps *every* co-mentioned unrelated
            pair and switches off the distance cap, which is the realistic
            evaluation setting (all candidate pairs of a document are judged).
    """
    rng = random.Random(seed)
    if distance_stats is None:
        distance_stats = compute_relation_distance_stats(meta, anns, rels, relation_types, verbose=verbose)
    mean_distance: float = distance_stats["mean"]
    max_distance: float = distance_stats["max"]

    abstract_by_pmid: dict[str, str] = dict(zip(meta["pmid"], meta["abstract"]))
    entity_index = _build_entity_spans(anns)
    real_pairs_by_pmid = _real_pairs_per_paper(rels)
    relation_type_set = set(relation_types)

    entities_by_pmid: dict[str, list[str]] = {}
    for pmid, mesh_id in entity_index:
        entities_by_pmid.setdefault(pmid, []).append(mesh_id)

    samples: list[TrainingSample] = []
    gold_count_by_pmid: dict[str, int] = {}

    for _, row in rels.iterrows():
        rel_type = row["relation_type"]
        if rel_type not in relation_type_set:
            continue
        pmid = row["pmid"]
        if pmid not in abstract_by_pmid:
            continue
        a_id, b_id = row["id_1"], row["id_2"]
        info_a = entity_index.get((pmid, a_id))
        info_b = entity_index.get((pmid, b_id))
        if not info_a or not info_b:
            continue
        samples.append(
            TrainingSample(
                pmid=pmid,
                abstract=abstract_by_pmid[pmid],
                entity_a_id=a_id,
                entity_a_text=info_a["mention"],
                entity_a_type=info_a["entity_type"],
                relation_type=rel_type,
                entity_b_id=b_id,
                entity_b_text=info_b["mention"],
                entity_b_type=info_b["entity_type"],
                label=1,
                perturbation="gold",
            )
        )
        gold_count_by_pmid[pmid] = gold_count_by_pmid.get(pmid, 0) + 1

    no_relation_count = 0
    for pmid, n_gold in gold_count_by_pmid.items():
        real_pairs = real_pairs_by_pmid.get(pmid, set())
        ent_ids = entities_by_pmid.get(pmid, [])

        candidates: list[tuple[str, str, dict, dict, float]] = []
        for i, a_id in enumerate(ent_ids):
            for b_id in ent_ids[i + 1:]:
                if a_id == b_id or (a_id, b_id) in real_pairs:
                    continue
                info_a = entity_index[(pmid, a_id)]
                info_b = entity_index[(pmid, b_id)]
                distance = entity_pair_distance(info_a["spans"], info_b["spans"])
                if distance is None:
                    continue
                if no_relation_ratio is not None and distance > max_distance:
                    continue
                candidates.append((a_id, b_id, info_a, info_b, distance))

        if no_relation_ratio is None:
            kept = candidates
        else:
            rng.shuffle(candidates)
            candidates.sort(key=lambda candidate: abs(candidate[4] - mean_distance))
            kept = candidates[: int(round(no_relation_ratio * n_gold))]

        for a_id, b_id, info_a, info_b, _ in kept:
            samples.append(
                TrainingSample(
                    pmid=pmid,
                    abstract=abstract_by_pmid[pmid],
                    entity_a_id=a_id,
                    entity_a_text=info_a["mention"],
                    entity_a_type=info_a["entity_type"],
                    relation_type=NO_RELATION_LABEL,
                    entity_b_id=b_id,
                    entity_b_text=info_b["mention"],
                    entity_b_type=info_b["entity_type"],
                    label=0,
                    perturbation="false_positive",
                )
            )
            no_relation_count += 1

    if verbose:
        gold_total = sum(gold_count_by_pmid.values())
        print(f"Built {gold_total} gold relations and {no_relation_count} NoRelation examples.")

    return samples


def build_pure_biored_dataframe(
    pubtator_file: str | Path,
    relation_types: list[str] = BIORED_RELATION_TYPES,
    seed: int = 42,
    no_relation_ratio: float | None = 1.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Parse a BioRED PubTator file and return a rels-like DataFrame.

    The output mirrors :func:`dataset_preparation.perturbations.samples_to_rels_like_df`,
    so it is a drop-in replacement for the perturbated pipeline's ``samples``.
    """
    meta_df, anns_df, rels_df = parse_pubtator(pubtator_file)
    samples = build_pure_biored_samples(
        meta_df,
        anns_df,
        rels_df,
        relation_types=relation_types,
        seed=seed,
        no_relation_ratio=no_relation_ratio,
        verbose=verbose,
    )
    return samples_to_rels_like_df(samples)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python prepare_pure_biored.py <path_to_pubtator_file>")
        sys.exit(1)

    meta, anns, rels = parse_pubtator(sys.argv[1])
    stats = compute_relation_distance_stats(meta, anns, rels)
    samples = build_pure_biored_samples(meta, anns, rels, distance_stats=stats)
    df = samples_to_rels_like_df(samples)

    print("\n=== pure BioRED dataframe ===")
    print(df["perturbation"].value_counts())
    print(df.head(10).to_string(index=False))
