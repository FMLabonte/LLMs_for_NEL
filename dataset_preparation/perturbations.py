"""
Phase 1: Perturbation generators for the quality-filter classifier.

Given a gold BioRED relation, produce perturbed variants used as negative
training examples. Seven perturbation labels (one positive, six negative):

  gold              - unchanged BioRED relation (positive)
  label_flip        - swap relation type for a different one
  direction_swap    - swap entity_A and entity_B (only for directional rels)
  fp_co_related     - replace entity_B with another in-abstract entity that
                      has its own annotated relation elsewhere
  fp_co_standalone  - replace entity_B with another in-abstract entity that
                      has no annotated relations at all
  fp_external       - replace entity_B with an entity not in the abstract
  false_negative    - claim NoRelation for a triple that does have a relation

All perturbations operate on the (entity_pair, relation_label) tuple ONLY.
The abstract text is never modified.

See PERTURBATIONS.md (root of repo) for the full taxonomy with rationale,
worked examples, and design decisions.
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
    "fp_co_related",
    "fp_co_standalone",
    "fp_external",
    "false_negative",
]

FpSubType = Literal["co_related", "co_standalone", "external"]

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

    def to_pandas_record(self) -> dict:
        """Flat record matching the 'rels' DataFrame schema (plus label/perturbation).

        Drops the abstract text and entity surface text/type. Downstream pipelines
        join back to the abstract via pmid -> meta lookup.
        """
        return {
            "pmid":          self.pmid,
            "relation_type": self.relation_type,
            "id_1":          self.entity_a_id,
            "id_2":          self.entity_b_id,
            "perturbation":  self.perturbation,
            "label":         self.label,
        }


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


def _entities_with_relations_per_paper(
    rels: pd.DataFrame,
) -> dict[str, set[str]]:
    """For each pmid, the set of entity IDs that participate in at least one
    annotated relation. Used to distinguish fp_co_related from fp_co_standalone."""
    out: dict[str, set[str]] = {}
    for pmid, grp in rels.groupby("pmid"):
        ids = set(grp["id_1"]).union(set(grp["id_2"]))
        out[pmid] = ids
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


def _build_corpus_pool(
    entity_info: dict[tuple[str, str], dict],
) -> dict[str, dict]:
    """mesh_id -> {mention, entity_type}, deduped across all abstracts.

    Used as the candidate pool for fp_external sampling (entities drawn from
    the corpus that are NOT in the current abstract). First occurrence wins.
    """
    pool: dict[str, dict] = {}
    for (_pmid, mesh_id), info in entity_info.items():
        if mesh_id not in pool:
            pool[mesh_id] = info
    return pool


# ---------------------------------------------------------------------------
# Perturbation: Label flip
# ---------------------------------------------------------------------------

def perturb_label_flip(
    relation_type: str,
    rng: random.Random,
) -> str:
    """Pick a different relation type, uniform over the remaining options."""
    candidates = [r for r in BIORED_RELATION_TYPES if r != relation_type]
    return rng.choice(candidates)


# ---------------------------------------------------------------------------
# Perturbation: Direction swap
# ---------------------------------------------------------------------------

def direction_swap_is_meaningful(relation_type: str) -> bool:
    """True for directional relations only. Skip Association / Comparison."""
    return relation_type not in SYMMETRIC_RELATIONS


# ---------------------------------------------------------------------------
# Perturbation: False positive (3 sub-types)
# ---------------------------------------------------------------------------

def sample_fp_entity_b(
    pmid: str,
    entity_a_id: str,
    entity_a_info: dict,
    pmid_anns: pd.DataFrame,
    real_pairs: set[tuple[str, str]],
    entities_with_relations: set[str],
    plausible_type_pairs: set[tuple[str, str]],
    corpus_pool: dict[str, dict],
    rng: random.Random,
    sub_type: FpSubType,
    type_restricted: bool = True,
) -> tuple[str, dict] | None:
    """Pick an entity_B for a false-positive pair, anchored on a fixed entity_A.

    sub_type controls where entity_B is drawn from:

      co_related    : in this abstract AND has its own annotated relation
                      elsewhere (just not with entity_A).
      co_standalone : in this abstract AND has no annotated relations at all.
      external      : NOT in this abstract; drawn from the corpus pool.

    type_restricted=True (default) further filters to entity-type pairs that
    actually appear in real BioRED relations, avoiding implausible
    Species/CellLine combinations LLMs would never plausibly hallucinate.

    Returns (b_id, b_info) or None if no candidate exists.
    """
    if sub_type == "external":
        in_abstract_ids = (
            set(pmid_anns["mesh_id"].unique())
            if pmid_anns is not None and len(pmid_anns) > 0
            else set()
        )
        candidates: list[tuple[str, dict]] = []
        for b_id, b_info in corpus_pool.items():
            if b_id == entity_a_id:
                continue
            if b_id in in_abstract_ids:
                continue
            if type_restricted and (
                (entity_a_info["entity_type"], b_info["entity_type"])
                not in plausible_type_pairs
            ):
                continue
            candidates.append((b_id, b_info))
    else:
        # co_related or co_standalone: candidates from this abstract
        if pmid_anns is None or len(pmid_anns) == 0:
            return None
        ent_rows = (
            pmid_anns[["mesh_id", "entity_type", "mention"]]
            .drop_duplicates(subset=["mesh_id"])
        )
        candidates = []
        for _, e in ent_rows.iterrows():
            b_id = e["mesh_id"]
            if b_id == entity_a_id:
                continue
            if (entity_a_id, b_id) in real_pairs:
                continue  # this pair already has a real relation; not a false positive
            if type_restricted and (
                (entity_a_info["entity_type"], e["entity_type"])
                not in plausible_type_pairs
            ):
                continue
            has_rels = b_id in entities_with_relations
            if sub_type == "co_related" and not has_rels:
                continue
            if sub_type == "co_standalone" and has_rels:
                continue
            b_info = {"mention": e["mention"], "entity_type": e["entity_type"]}
            candidates.append((b_id, b_info))

    if not candidates:
        return None
    return rng.choice(candidates)


# ---------------------------------------------------------------------------
# Perturbation: False negative
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
    """Build the full training set: one gold sample plus the perturbed
    negatives for each gold relation.

    Per gold, emits up to:
      1 gold + 1 label_flip + 1 direction_swap (if directional)
      + 1 fp_co_related + 1 fp_co_standalone + 1 fp_external
      + 1 false_negative
    Some sub-types may emit 0 samples if no candidate exists in this abstract.
    """
    rng = random.Random(seed)
    abstract_by_pmid = dict(zip(meta["pmid"], meta["abstract"]))

    entity_info = _build_entity_lookup(anns)
    real_pairs_by_pmid = _real_pairs_per_paper(rels)
    entities_with_rels_by_pmid = _entities_with_relations_per_paper(rels)
    plausible_type_pairs = _plausible_type_pairs(rels, entity_info)
    corpus_pool = _build_corpus_pool(entity_info)

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

    fp_sub_types: tuple[FpSubType, ...] = ("co_related", "co_standalone", "external")

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

        # ----- label flip -----
        flipped = perturb_label_flip(rel_type, rng)
        samples.append(_make(pmid, a_id, info_a, flipped, b_id, info_b, 0, "label_flip"))

        # ----- direction swap (only for directional relations) -----
        if direction_swap_is_meaningful(rel_type):
            samples.append(_make(pmid, b_id, info_b, rel_type, a_id, info_a, 0, "direction_swap"))

        # ----- false positive (3 sub-types, each anchored on entity_a from gold) -----
        for sub in fp_sub_types:
            picked = sample_fp_entity_b(
                pmid=pmid,
                entity_a_id=a_id,
                entity_a_info=info_a,
                pmid_anns=anns_by_pmid.get(pmid),
                real_pairs=real_pairs_by_pmid.get(pmid, set()),
                entities_with_relations=entities_with_rels_by_pmid.get(pmid, set()),
                plausible_type_pairs=plausible_type_pairs,
                corpus_pool=corpus_pool,
                rng=rng,
                sub_type=sub,
                type_restricted=type_restricted_false_positives,
            )
            if picked is not None:
                fp_b_id, fp_b_info = picked
                fake_rel = rng.choice(BIORED_RELATION_TYPES)
                samples.append(_make(pmid, a_id, info_a, fake_rel,
                                     fp_b_id, fp_b_info, 0, f"fp_{sub}"))

        # ----- false negative -----
        samples.append(_make(pmid, a_id, info_a, perturb_false_negative_label(),
                             b_id, info_b, 0, "false_negative"))

    return samples


def samples_to_rels_like_df(samples: list[TrainingSample]) -> pd.DataFrame:
    """Convert a list of TrainingSamples to a DataFrame with the same shape
    as the 'rels' DataFrame produced by pubtator_parser, plus the
    perturbation/label columns."""
    records = [sample.to_pandas_record() for sample in samples]
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# TODOs (open issues)
# ---------------------------------------------------------------------------
#
# - Multi-ID umbrella entities: BioRED occasionally puts comma-separated MeSH IDs
#   in one annotation row (e.g. "3172,3651,6927" for MODY). Currently dropped
#   silently (about 8% of relations). See issue tracker.
#
# - Canonical mention selection: currently "longest". Consider "most frequent"
#   or "first-occurring" and ablate.
#
# - Multi-variant generation per perturbation per gold (issue #5).
#
# - Two dataset versions for relation-class imbalance (full vs reduced) (issue #6).
#
# - NoRelation as an explicit gold-positive class, not just the false_negative
#   label (issue #7).
#
# - Per-perturbation spin-off datasets for one-thing-at-a-time analysis
#   (issue #3).