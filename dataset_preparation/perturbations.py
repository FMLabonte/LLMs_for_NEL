"""
Phase 1: Perturbation generators for the quality-filter classifier.

Given a gold BioRED relation, produce perturbed variants used as negative
training examples. Four perturbation types:

  1. label_flip     - swap the relation type for a different one
  2. direction_swap - swap entity_A and entity_B (only for directional rels)
  3. false_positive - invent a relation between two co-mentioned, unrelated entities
  4. false_negative - claim NoRelation for a relation that actually exists

All perturbations operate on the (entity_pair, relation_label) tuple ONLY.
The abstract text is never modified.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Literal

import pandas as pd

PerturbationType = Literal[
    "gold",
    "label_flip",
    "direction_swap",
    "false_positive",
    "false_negative",
]

# Relation types observed in BioRED gold. Keep this list authoritative;
# build_dataset.py asserts that every observed relation type is in here.
BIORED_RELATION_TYPES: list[str] = [
    "Positive_Correlation",
    "Negative_Correlation",
    "Association",
    "Comparison",
    "Cotreatment",
    "Drug_Interaction",
    "Bind",
    "Conversion",
]

# Symmetric relations: direction-swap is not a meaningful perturbation
# (A Association B is identical to B Association A), so we skip them.
SYMMETRIC_RELATIONS: set[str] = {"Association", "Comparison"}

NO_RELATION_LABEL = "NoRelation"


@dataclass(frozen=True)
class TrainingSample:
    """One training example for the quality-filter classifier."""
    pmid:           str
    abstract:       str
    entity_a_id:    str
    entity_a_text:  str
    entity_a_type:  str
    relation_type:  str          # may be NoRelation for false-negative perturbations
    entity_b_id:    str
    entity_b_text:  str
    entity_b_type:  str
    label:          int          # 1 = correct, 0 = incorrect
    perturbation:   PerturbationType

    def to_dict(self) -> dict:
        return asdict(self)

    def to_pandas_record(self):
        record = {}
        record["pmid"] = self.pmid
        record["relation_type"] = self.relation_type
        record["id_1"] = self.entity_a_id
        record["id_2"] = self.entity_b_id
        record["perturbation"] = self.perturbation
        record["label"] = self.label

        return record


def _build_entity_lookup(
    anns: pd.DataFrame,
) -> dict[tuple[str, str], dict]:
    """(pmid, mesh_id) -> {mention, entity_type}.

    Picks the longest mention seen for that ID in that paper as the canonical
    surface form. Other choices (most-frequent, first-occurring) are TODOs.
    """
    lookup: dict[tuple[str, str], dict] = {}
    for (pmid, mesh_id), grp in anns.groupby(["pmid", "mesh_id"]):
        mentions = grp["mention"].drop_duplicates().tolist()
        canonical = max(mentions, key=len) if mentions else ""
        lookup[(pmid, mesh_id)] = {
            "mention":     canonical,
            "entity_type": grp["entity_type"].iloc[0],
        }
    return lookup


def _real_pairs_per_paper(
    rels: pd.DataFrame,
) -> dict[str, set[tuple[str, str]]]:
    """For each pmid, the set of entity-id pairs that DO have a gold relation
    (stored in both orders so membership tests are direction-agnostic)."""
    out: dict[str, set[tuple[str, str]]] = {}
    for pmid, grp in rels.groupby("pmid"):
        pairs: set[tuple[str, str]] = set()
        for _, r in grp.iterrows():
            pairs.add((r["id_1"], r["id_2"]))
            pairs.add((r["id_2"], r["id_1"]))
        out[pmid] = pairs
    return out


def _plausible_type_pairs(
    rels: pd.DataFrame,
    entity_info: dict[tuple[str, str], dict],
) -> set[tuple[str, str]]:
    """(entity_type_a, entity_type_b) tuples actually observed in gold relations.

    Used to constrain false-positive sampling so we don't invent pairs like
    Species/CellLine that LLMs would never plausibly hallucinate a relation for.
    Stored in both orders.
    """
    pairs: set[tuple[str, str]] = set()
    for _, r in rels.iterrows():
        info_a = entity_info.get((r["pmid"], r["id_1"]))
        info_b = entity_info.get((r["pmid"], r["id_2"]))
        if info_a and info_b:
            ta, tb = info_a["entity_type"], info_b["entity_type"]
            pairs.add((ta, tb))
            pairs.add((tb, ta))
    return pairs


# ---------------------------------------------------------------------------
# Perturbation #1: Label flip
# ---------------------------------------------------------------------------

def perturb_label_flip(
    relation_type: str,
    rng: random.Random,
) -> str:
    """Pick a different relation type, uniform over the remaining options."""
    candidates = [r for r in BIORED_RELATION_TYPES if r != relation_type]
    return rng.choice(candidates)


# ---------------------------------------------------------------------------
# Perturbation #2: Direction swap
# ---------------------------------------------------------------------------

def direction_swap_is_meaningful(relation_type: str) -> bool:
    """True for directional relations only. Skip Association / Comparison."""
    return relation_type not in SYMMETRIC_RELATIONS


# ---------------------------------------------------------------------------
# Perturbation #3: False positive  (invent a relation that doesn't exist)
# ---------------------------------------------------------------------------

def sample_false_positive_pair(
    pmid: str,
    pmid_anns: pd.DataFrame,
    real_pairs: set[tuple[str, str]],
    plausible_type_pairs: set[tuple[str, str]],
    rng: random.Random,
    type_restricted: bool = True,
) -> tuple[str, str, dict, dict] | None:
    """Pick two co-mentioned entities in `pmid` with NO annotated relation.

    Returns (id_a, id_b, info_a, info_b) or None if no such pair exists.

    If `type_restricted=True`, only entity-type pairs that appear somewhere in
    the gold relations are considered (to avoid implausible Species/CellLine
    style false positives).
    """
    # Unique entities in this paper, with their types
    ent_rows = (
        pmid_anns[["mesh_id", "entity_type", "mention"]]
        .drop_duplicates(subset=["mesh_id"])
    )
    ents = ent_rows.to_dict("records")
    if len(ents) < 2:
        return None

    candidates: list[tuple[str, str, dict, dict]] = []
    for i, a in enumerate(ents):
        for b in ents[i + 1:]:
            if a["mesh_id"] == b["mesh_id"]:
                continue
            if (a["mesh_id"], b["mesh_id"]) in real_pairs:
                continue
            if type_restricted and (
                (a["entity_type"], b["entity_type"]) not in plausible_type_pairs
            ):
                continue
            info_a = {"mention": a["mention"], "entity_type": a["entity_type"]}
            info_b = {"mention": b["mention"], "entity_type": b["entity_type"]}
            candidates.append((a["mesh_id"], b["mesh_id"], info_a, info_b))

    if not candidates:
        return None
    return rng.choice(candidates)


# ---------------------------------------------------------------------------
# Perturbation #4: False negative  (claim NoRelation for a real relation)
# ---------------------------------------------------------------------------

def perturb_false_negative_label() -> str:
    return NO_RELATION_LABEL


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_training_samples(
    meta: pd.DataFrame,
    anns: pd.DataFrame,
    rels: pd.DataFrame,
    seed: int = 42,
    type_restricted_false_positives: bool = True,
) -> list[TrainingSample]:
    """Build the full training set: gold + 4 perturbations per gold relation."""
    rng = random.Random(seed)
    abstract_by_pmid = dict(zip(meta["pmid"], meta["abstract"]))

    entity_info = _build_entity_lookup(anns)
    real_pairs_by_pmid = _real_pairs_per_paper(rels)
    plausible_type_pairs = _plausible_type_pairs(rels, entity_info)

    # Group annotations by pmid once for cheap lookup inside the loop
    anns_by_pmid = {pmid: grp for pmid, grp in anns.groupby("pmid")}

    samples: list[TrainingSample] = []

    def _make(
        pmid: str,
        a_id: str, info_a: dict,
        rel_type: str,
        b_id: str, info_b: dict,
        label: int,
        perturbation: PerturbationType,
    ) -> TrainingSample:
        return TrainingSample(
            pmid=pmid,
            abstract=abstract_by_pmid[pmid],
            entity_a_id=a_id,
            entity_a_text=info_a["mention"],
            entity_a_type=info_a["entity_type"],
            relation_type=rel_type,
            entity_b_id=b_id,
            entity_b_text=info_b["mention"],
            entity_b_type=info_b["entity_type"],
            label=label,
            perturbation=perturbation,
        )

    for _, row in rels.iterrows():
        pmid = row["pmid"]
        a_id, b_id = row["id_1"], row["id_2"]
        rel_type = row["relation_type"]

        if pmid not in abstract_by_pmid:
            continue
        info_a = entity_info.get((pmid, a_id))
        info_b = entity_info.get((pmid, b_id))
        if not info_a or not info_b:
            continue

        # ----- gold -----
        samples.append(_make(pmid, a_id, info_a, rel_type, b_id, info_b, 1, "gold"))

        # ----- #1 label flip -----
        flipped = perturb_label_flip(rel_type, rng)
        samples.append(_make(pmid, a_id, info_a, flipped, b_id, info_b, 0, "label_flip"))

        # ----- #2 direction swap (only for directional relations) -----
        if direction_swap_is_meaningful(rel_type):
            samples.append(_make(pmid, b_id, info_b, rel_type, a_id, info_a, 0, "direction_swap"))

        # ----- #3 false positive -----
        fp = sample_false_positive_pair(
            pmid=pmid,
            pmid_anns=anns_by_pmid.get(pmid, anns.iloc[0:0]),
            real_pairs=real_pairs_by_pmid.get(pmid, set()),
            plausible_type_pairs=plausible_type_pairs,
            rng=rng,
            type_restricted=type_restricted_false_positives,
        )
        if fp is not None:
            fp_a_id, fp_b_id, fp_info_a, fp_info_b = fp
            fake_rel = rng.choice(BIORED_RELATION_TYPES)
            samples.append(_make(pmid, fp_a_id, fp_info_a, fake_rel,
                                 fp_b_id, fp_info_b, 0, "false_positive"))

        # ----- #4 false negative -----
        samples.append(_make(pmid, a_id, info_a, perturb_false_negative_label(),
                             b_id, info_b, 0, "false_negative"))

    return samples


def samples_to_rels_like_df(samples: list[TrainingSample]) -> pd.DataFrame:
    """Convert a list of TrainingSamples to DataFrame similar to 'rels' DF"""
    records = [sample.to_pandas_record() for sample in samples]
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# TODOs (Phase 1 follow-ups)
# ---------------------------------------------------------------------------
#
# - Multi-ID entities: BioRED occasionally puts comma-separated MeSH IDs in
#   one annotation row (e.g. "3172,3651,6927" for the umbrella term MODY).
#   Decide: pick first, expand to multiple rows, or skip.
#
# - Canonical mention selection: currently "longest". Consider "most frequent"
#   or "first-occurring" and ablate.
#
# - Class balance: false_positive can produce 0-many samples per gold relation
#   while the others produce 0-1. Decide how to balance / cap.
#
# - Direction-swap collision: after swapping, the (b_id, a_id) pair might
#   itself be a gold relation in the same paper (different rel type). Currently
#   we don't filter for that; should we?
#
# - Sentence-level vs document-level: the filter sees the full abstract per
#   sample. Some prior RE work uses just the entity-spanning sentence.
