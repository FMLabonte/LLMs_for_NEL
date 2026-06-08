"""
Phase 1: Perturbation generators for the quality-filter classifier.

Given a gold BioRED relation, produce perturbed variants used as negative
training examples. Seven perturbation labels (one positive, six negative):

  gold              - unchanged BioRED relation (positive)
  label_flip        - swap relation type for a different one (every alternative
                      kept type, except a type the gold already implies, e.g.
                      Pos/Neg_Correlation are not flipped to the weaker Association)
  direction_swap    - swap entity_A and entity_B (only for directional rels)
  fp_co_related     - replace entity_B with another in-abstract entity that
                      has its own annotated relation elsewhere
  fp_co_standalone  - replace entity_B with another in-abstract entity that
                      has no annotated relations
  fp_external       - replace entity_B with an entity not in the abstract
  false_negative    - claim NoRelation for a triple that does have a relation

All perturbations operate on the (entity_pair, relation_label) tuple ONLY.
The abstract text is never modified.

Balance rule: the per-split count of each FP type is capped at the per-split
label_flip count, so all six perturbation types end up in roughly the same
order of magnitude. Decided 2026-05-21.

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

# The 3 BioRED types with enough examples to evaluate. The dataset defaults
# to these; --rare-classes opts the rest back in as an ablation.
TOP_RELATION_TYPES: list[str] = [
    "Association",
    "Positive_Correlation",
    "Negative_Correlation",
]

# The only relation types where swapping (entity_A, entity_B) actually
# changes the meaning. Association is non-directional; the rare classes
# are dropped from the default dataset and not meaningful to direction-swap
# either (per Frederik on 2026-05-08).
DIRECTIONAL_RELATIONS: set[str] = {
    "Positive_Correlation",
    "Negative_Correlation",
}

# A label_flip is only a usable NEGATIVE when the swapped relation type is
# actually wrong for the pair. Positive_Correlation and Negative_Correlation both
# still imply a generic Association (a positive/negative correlation IS an
# association), so flipping either of them to Association yields a claim that is
# still true, just less specific. That is label noise, not a negative, so we skip
# it. Mapping: gold relation type -> set of weaker types it implies (and must not
# be flipped to). The reverse direction (Association -> a specific correlation) is
# left in place: it asserts a direction the annotators did not, which is
# defensible as a negative.
IMPLIED_RELATIONS: dict[str, set[str]] = {
    "Positive_Correlation": {"Association"},
    "Negative_Correlation": {"Association"},
}

NO_RELATION_LABEL = "NoRelation"

# fp_external would otherwise produce thousands of candidates per gold (the
# whole corpus pool, type-filtered). Cap the per-gold count to keep the FP
# pool tractable before the per-split downsample to label_flip-count.
FP_EXTERNAL_MAX_PER_GOLD = 5


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
    (stored in both orders so membership tests are direction-agnostic).

    Built from the full rels DataFrame, including rare-class relations, so an
    fp candidate is never an entity pair that BioRED annotated with ANY label.
    """
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
    annotated relation. Used to distinguish fp_co_related from fp_co_standalone.

    Built from the full rels DataFrame so a rare-class-only entity is still
    correctly classified as having relations.
    """
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
    Stored in both orders. Built from the full rels DataFrame so type
    plausibility tracks the whole BioRED, not just the kept relation types.
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
# Perturbation: Direction swap
# ---------------------------------------------------------------------------

def direction_swap_is_meaningful(relation_type: str) -> bool:
    """True for the only directional relations: Positive/Negative_Correlation.

    Association is non-directional (only asserts a link, not who acts on whom);
    the rare classes are dropped from the default dataset anyway. Decision per
    Frederik on 2026-05-08.
    """
    return relation_type in DIRECTIONAL_RELATIONS


# ---------------------------------------------------------------------------
# Perturbation: False positive (3 sub-types)
# ---------------------------------------------------------------------------

def sample_fp_candidates(
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
    external_max_per_gold: int = FP_EXTERNAL_MAX_PER_GOLD,
) -> list[tuple[str, dict]]:
    """Return all eligible entity_B candidates for a false-positive pair,
    anchored on a fixed entity_A.

    For co_related and co_standalone the in-abstract pool is small (typically
    < 20 entities) so we return everything. For external the corpus pool is
    huge; we draw at most external_max_per_gold candidates so the global FP
    pool stays tractable before the per-split downsample.
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
        if len(candidates) > external_max_per_gold:
            candidates = rng.sample(candidates, external_max_per_gold)
        return candidates

    # co_related or co_standalone: candidates from this abstract
    if pmid_anns is None or len(pmid_anns) == 0:
        return []
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
            continue
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
        candidates.append((b_id, {"mention": e["mention"], "entity_type": e["entity_type"]}))
    return candidates


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
    kept_relation_types: list[str] | None = None,
) -> list[TrainingSample]:
    """Build the full training set for one split.

    Per gold relation we emit:
      - 1 gold sample
      - up to (|kept_relation_types| - 1) label_flip samples (every alternative
        kept type, minus any type the gold implies, e.g. a specific correlation
        is not flipped to the weaker Association)
      - 1 direction_swap sample (only for Pos/Neg_Correlation)
      - 1 false_negative sample
      - up to LF/N_gold fp_co_related, fp_co_standalone, fp_external samples
        each, where LF is the per-split label_flip count and the per-gold pool
        is generated then globally downsampled to LF rows per FP type.

    kept_relation_types restricts which gold relations are perturbed and which
    target types label_flip can pick. Defaults to BIORED_RELATION_TYPES (the
    full set). FP helper structures (real_pairs, entities_with_relations,
    plausible_type_pairs) are still built from the full rels DataFrame so a
    pair annotated only as a rare class is correctly excluded from FP candidates.
    """
    if kept_relation_types is None:
        kept_relation_types = list(BIORED_RELATION_TYPES)
    kept_set = set(kept_relation_types)

    rng = random.Random(seed)
    abstract_by_pmid = dict(zip(meta["pmid"], meta["abstract"]))

    entity_info = _build_entity_lookup(anns)
    real_pairs_by_pmid = _real_pairs_per_paper(rels)
    entities_with_rels_by_pmid = _entities_with_relations_per_paper(rels)
    plausible_type_pairs = _plausible_type_pairs(rels, entity_info)
    corpus_pool = _build_corpus_pool(entity_info)

    anns_by_pmid = {pmid: grp for pmid, grp in anns.groupby("pmid")}
    kept_rels = rels[rels["relation_type"].isin(kept_set)]

    samples: list[TrainingSample] = []
    fp_sub_types: tuple[FpSubType, ...] = ("co_related", "co_standalone", "external")
    fp_pools: dict[FpSubType, list[TrainingSample]] = {s: [] for s in fp_sub_types}
    label_flip_count = 0

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

    # Phase 1: per-gold emission for gold/label_flip/direction_swap/false_negative,
    # and collection of FP candidates into per-subtype pools.
    for _, row in kept_rels.iterrows():
        pmid = row["pmid"]
        a_id, b_id = row["id_1"], row["id_2"]
        rel_type = row["relation_type"]

        if pmid not in abstract_by_pmid:
            continue
        info_a = entity_info.get((pmid, a_id))
        info_b = entity_info.get((pmid, b_id))
        if not info_a or not info_b:
            continue

        samples.append(_make(pmid, a_id, info_a, rel_type, b_id, info_b, 1, "gold"))

        # label_flip: one row per alternative kept type, skipping any type the
        # gold relation already implies (e.g. Positive_Correlation -> Association
        # is still true, so it is not a valid negative). See IMPLIED_RELATIONS.
        implied = IMPLIED_RELATIONS.get(rel_type, frozenset())
        for alt in kept_relation_types:
            if alt == rel_type or alt in implied:
                continue
            samples.append(_make(pmid, a_id, info_a, alt, b_id, info_b, 0, "label_flip"))
            label_flip_count += 1

        if direction_swap_is_meaningful(rel_type):
            samples.append(_make(pmid, b_id, info_b, rel_type, a_id, info_a, 0, "direction_swap"))

        # FP: collect all eligible candidates per subtype, downsample globally below
        for sub in fp_sub_types:
            candidates = sample_fp_candidates(
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
            for fp_b_id, fp_b_info in candidates:
                # Keep gold's relation_type so this perturbation isolates the
                # wrong-entity_B signal (otherwise it overlaps with label_flip).
                fp_pools[sub].append(_make(
                    pmid, a_id, info_a, rel_type, fp_b_id, fp_b_info, 0, f"fp_{sub}",
                ))

        samples.append(_make(pmid, a_id, info_a, perturb_false_negative_label(),
                             b_id, info_b, 0, "false_negative"))

    # Phase 2: cap each FP pool at the per-split label_flip count
    for sub, pool in fp_pools.items():
        if len(pool) > label_flip_count:
            pool = rng.sample(pool, label_flip_count)
        samples.extend(pool)

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
# - NoRelation as an explicit gold-positive class, not just the false_negative
#   label (issue #7).
#
# - Per-perturbation spin-off datasets for one-thing-at-a-time analysis
#   (issue #3).