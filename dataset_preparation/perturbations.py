"""
Phase 1: Perturbation generators for the quality-filter classifier.

Four gold classes (label 1): Association, Positive_Correlation,
Negative_Correlation (from BioRED relations) and NoRelation (genuinely-unrelated
co-mention pairs, distance-matched and capped). From each gold we produce
perturbed variants used as negative training examples (label 0):

  gold              - a true (entity_pair, relation) example, incl. NoRelation
  label_flip        - swap the relation type for another valid class over the
                      4-class matrix. The only skip is specific -> Association
                      (still true). Covers relation -> NoRelation (formerly the
                      separate false_negative) and NoRelation -> relation.
  direction_swap    - swap entity_A and entity_B (only Pos/Neg_Correlation)
  fp_external       - replace entity_B with an entity not in the abstract

The 5 rare BioRED relation types are always dropped (the --rare-classes ablation
was removed 2026-06-10). All perturbations operate on the (entity_pair,
relation_label) tuple ONLY; the abstract text is never modified.

Balance rule: the per-split count of each FP type is capped at the per-split
label_flip count. Decided 2026-05-21.

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
    "fp_external",
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
# actually wrong for the pair. Every SPECIFIC BioRED relation type implies the
# generic Association: a Bind, Cotreatment, Drug_Interaction, Conversion, or a
# positive/negative correlation between two entities still means they are
# associated. So flipping any specific type to Association produces a claim that is
# still true, not a usable negative. We therefore never use Association as a
# label_flip target. Mapping: gold relation type -> set of weaker types it implies
# (and must not be flipped to).
#
# Association itself is NOT a key: flipping it to a specific type asserts a
# direction/mechanism the annotators did not, which is a defensible negative, so we
# keep those. The only entailment in BioRED's otherwise flat taxonomy is
# "specific type -> Association"; there is no specific -> specific entailment.
# The mapping is keyed over all BioRED relation types for completeness, although
# only the top-3 are used (the rare classes are dropped from the build).
IMPLIED_RELATIONS: dict[str, set[str]] = {
    rel: {"Association"} for rel in BIORED_RELATION_TYPES if rel != "Association"
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


def _split_multi_id_annotations(anns: pd.DataFrame) -> pd.DataFrame:
    """Split BioRED annotations whose ``mesh_id`` is a comma-separated ID list.

    BioRED tags a single text span that refers to several normalized concepts with
    all of their IDs in one annotation, e.g. ``mesh_id = "D002289,D055752"`` for the
    span "squamous- and small-cell lung cancer". Relations always reference the
    individual IDs (verified: no relation endpoint contains a comma), so without
    splitting, those individual IDs never match an annotation and about 8% of gold
    relations get silently dropped. We explode such annotations into one row per ID,
    keeping the shared mention / entity_type / offsets.

    Comma is the only multi-ID separator. Other punctuation that appears in a single
    id (e.g. ``|`` in a SequenceVariant id like ``c|DEL|1314_1328|``) is left
    untouched.
    """
    if anns.empty or "mesh_id" not in anns.columns:
        return anns
    out = anns.copy()
    out["mesh_id"] = out["mesh_id"].apply(
        lambda v: [s.strip() for s in str(v).split(",")] if pd.notna(v) else [v]
    )
    return out.explode("mesh_id", ignore_index=True)


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

    Built from the full rels DataFrame, including rare-class relations, so a
    NoRelation candidate is never an entity pair that BioRED annotated with ANY label.
    """
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
# NoRelation gold pairs (genuine negatives, distance-matched)
# ---------------------------------------------------------------------------

def _char_gap(spans_a: list, spans_b: list) -> int | None:
    """Smallest character gap between the closest mentions of two entities.

    Each entity may be mentioned several times; we take the minimum gap over all
    mention pairs (0 if any two mentions overlap). Offsets share one coordinate
    system per abstract, so raw start/end are enough.
    """
    best = None
    for sa, ea in spans_a:
        for sb, eb in spans_b:
            if ea <= sb:
                g = sb - ea
            elif eb <= sa:
                g = sa - eb
            else:
                g = 0
            if best is None or g < best:
                best = g
    return best


def _norelation_gold_pairs(
    anns: pd.DataFrame,
    rels: pd.DataFrame,
    gold_set: set[str],
    rng: random.Random,
    cap: int | None = None,
    metric: str = "char",
) -> list[tuple[str, str, str]]:
    """Genuine NoRelation golds: unordered entity pairs co-mentioned in an
    abstract that have NO relation of ANY type (implicit negatives).

    There are ~7x more of these than positives, and they sit much farther apart
    in the text than real related pairs, so an unfiltered sample would teach the
    model "far apart = no relation". When ``cap`` is set we therefore distance-
    match: each candidate is accepted with probability proportional to how common
    its character gap is among the real related pairs (the "mimic the distance"
    rule). ``metric`` is char only in the pipeline; the token/sentence variants
    live in a separate case study.

    Returns [(pmid, id_a, id_b), ...]. Deterministic for a fixed ``rng``.
    """
    if metric != "char":
        raise NotImplementedError(
            f"NoRelation distance metric {metric!r} not supported in the pipeline "
            f"(char only; token/sentence are case-study-only)."
        )
    from collections import Counter, defaultdict
    from itertools import combinations

    spans: dict[tuple[str, str], list] = defaultdict(list)
    for r in anns.itertuples(index=False):
        spans[(r.pmid, r.mesh_id)].append((int(r.start), int(r.end)))

    ents_by_pmid: dict[str, list] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for (pmid, mid) in spans:
        if (pmid, mid) not in seen:
            seen.add((pmid, mid))
            ents_by_pmid[pmid].append(mid)

    real = _real_pairs_per_paper(rels)  # both orders, ANY relation type

    target_gaps: list[int] = []
    for r in rels.itertuples(index=False):
        if r.relation_type in gold_set:
            sa, sb = spans.get((r.pmid, r.id_1)), spans.get((r.pmid, r.id_2))
            if sa and sb:
                g = _char_gap(sa, sb)
                if g is not None:
                    target_gaps.append(g)

    candidates: list[tuple[str, str, str, int]] = []
    for pmid, ents in ents_by_pmid.items():
        rp = real.get(pmid, set())
        for a, b in combinations(sorted(set(ents)), 2):
            if (a, b) in rp:
                continue
            g = _char_gap(spans[(pmid, a)], spans[(pmid, b)])
            if g is not None:
                candidates.append((pmid, a, b, g))

    if cap is None or len(candidates) <= cap:
        return [(p, a, b) for (p, a, b, _g) in candidates]

    binw = 10
    tc = Counter(g // binw for g in target_gaps) if target_gaps else Counter()
    maxc = max(tc.values()) if tc else 1
    order = candidates[:]
    rng.shuffle(order)
    chosen: list[tuple[str, str, str]] = []
    for (pmid, a, b, gap) in order:
        if len(chosen) >= cap:
            break
        if rng.random() < tc.get(gap // binw, 0) / maxc:
            chosen.append((pmid, a, b))
    if len(chosen) < cap:  # top up with the remaining closest-to-target candidates
        taken = set(chosen)
        rest = [(p, a, b, gp) for (p, a, b, gp) in order if (p, a, b) not in taken]
        rest.sort(key=lambda x: tc.get(x[3] // binw, 0), reverse=True)
        for (p, a, b, _gp) in rest:
            if len(chosen) >= cap:
                break
            chosen.append((p, a, b))
    return chosen[:cap]


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
# Perturbation: False positive (external only)
# ---------------------------------------------------------------------------

def sample_external_fp_candidates(
    entity_a_id: str,
    entity_a_info: dict,
    pmid_anns: pd.DataFrame,
    plausible_type_pairs: set[tuple[str, str]],
    corpus_pool: dict[str, dict],
    rng: random.Random,
    type_restricted: bool = True,
    external_max_per_gold: int = FP_EXTERNAL_MAX_PER_GOLD,
) -> list[tuple[str, dict]]:
    """Eligible entity_B candidates for an external false positive, anchored on a
    fixed entity_A: entities drawn from the corpus that are NOT in this abstract.

    The corpus pool is huge, so we draw at most external_max_per_gold candidates
    per gold to keep the global FP pool tractable before the per-split downsample.
    (Only fp_external survives. The in-abstract families fp_co_related and
    fp_co_standalone were dropped 2026-06-10: their entity_B is co-mentioned, so the
    pair has no relation and is itself a NoRelation candidate, which made those rows
    duplicate the NoRelation -> relation label_flips. fp_external's entity_B is not in
    the abstract, so it can never collide.)
    """
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


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_training_samples(
    meta: pd.DataFrame,
    anns: pd.DataFrame,
    rels: pd.DataFrame,
    seed: int = 42,
    type_restricted_false_positives: bool = True,
    gold_relation_types: list[str] | None = None,
    n_norelation_cap: int | None = None,
    norelation_distance_metric: str = "char",
) -> list[TrainingSample]:
    """Build the full training set for one split (4-class scheme).

    Gold classes: Association, Positive_Correlation, Negative_Correlation (from
    BioRED relations) plus NoRelation (genuinely-unrelated co-mention pairs,
    distance-matched and capped). Per gold we emit:
      - 1 gold sample (label 1)
      - one label_flip per valid alternative class (label 0). NoRelation is a full
        participant: relation golds may flip TO NoRelation (what used to be the
        separate false_negative perturbation), and NoRelation golds flip to the
        three relations. The only skip is specific -> Association (still true; see
        IMPLIED_RELATIONS).
      - 1 direction_swap (only Pos/Neg_Correlation; never NoRelation)
      - fp_external (entity_B drawn from outside the abstract), globally downsampled
        to the per-split label_flip count. NoRelation golds get NO fp samples. The
        in-abstract fp families were dropped 2026-06-10 (they overlapped the
        NoRelation -> relation label_flips).

    gold_relation_types selects which BioRED relations become golds (default the
    top-3). n_norelation_cap caps the NoRelation golds (None = all candidates);
    they are distance-matched via norelation_distance_metric ("char"). FP helper
    structures are built from the full rels DataFrame so a pair annotated with ANY
    relation is excluded from FP and NoRelation candidates.
    """
    if gold_relation_types is None:
        gold_relation_types = list(TOP_RELATION_TYPES)
    gold_set = set(gold_relation_types)
    flip_targets = list(gold_relation_types) + [NO_RELATION_LABEL]

    rng = random.Random(seed)
    nr_rng = random.Random(seed + 1)  # isolated stream for NoRelation sampling
    abstract_by_pmid = dict(zip(meta["pmid"], meta["abstract"]))

    # Split comma-separated multi-ID annotations so relations that reference an
    # individual id resolve. Without this, ~8% of gold relations are dropped because
    # the relation's id does not exact-match the composite annotation id.
    anns = _split_multi_id_annotations(anns)

    entity_info = _build_entity_lookup(anns)
    plausible_type_pairs = _plausible_type_pairs(rels, entity_info)
    corpus_pool = _build_corpus_pool(entity_info)

    anns_by_pmid = {pmid: grp for pmid, grp in anns.groupby("pmid")}
    kept_rels = rels[rels["relation_type"].isin(gold_set)]

    # NoRelation golds: genuine negatives (co-mention pairs with no relation),
    # distance-matched to the real related pairs and capped.
    norel_pairs = _norelation_gold_pairs(
        anns, rels, gold_set, nr_rng,
        cap=n_norelation_cap, metric=norelation_distance_metric,
    )

    samples: list[TrainingSample] = []
    fp_external_pool: list[TrainingSample] = []
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

    # Combined gold list: real-relation golds (get fp) then NoRelation golds (no fp).
    gold_specs: list[tuple[str, str, str, str, bool]] = [
        (row["pmid"], row["id_1"], row["id_2"], row["relation_type"], True)
        for _, row in kept_rels.iterrows()
    ]
    gold_specs += [(pmid, a, b, NO_RELATION_LABEL, False) for (pmid, a, b) in norel_pairs]

    # Phase 1: per-gold emission for gold/label_flip/direction_swap, and collection
    # of FP candidates into per-subtype pools.
    for pmid, a_id, b_id, rel_type, do_fp in gold_specs:
        if pmid not in abstract_by_pmid:
            continue
        info_a = entity_info.get((pmid, a_id))
        info_b = entity_info.get((pmid, b_id))
        if not info_a or not info_b:
            continue

        samples.append(_make(pmid, a_id, info_a, rel_type, b_id, info_b, 1, "gold"))

        # label_flip over the 4-class matrix: one row per valid alternative class,
        # skipping any type the gold already implies (specific -> Association is
        # still true, so not a valid negative). See IMPLIED_RELATIONS. This also
        # covers relation -> NoRelation (the old false_negative) and the new
        # NoRelation -> {Association, Pos, Neg} flips.
        implied = IMPLIED_RELATIONS.get(rel_type, frozenset())
        for alt in flip_targets:
            if alt == rel_type or alt in implied:
                continue
            samples.append(_make(pmid, a_id, info_a, alt, b_id, info_b, 0, "label_flip"))
            label_flip_count += 1

        if direction_swap_is_meaningful(rel_type):
            samples.append(_make(pmid, b_id, info_b, rel_type, a_id, info_a, 0, "direction_swap"))

        # fp_external only, and only for real-relation golds (NoRelation golds get
        # none: swapping entity_B on a non-relation just yields another non-relation).
        if do_fp:
            candidates = sample_external_fp_candidates(
                entity_a_id=a_id,
                entity_a_info=info_a,
                pmid_anns=anns_by_pmid.get(pmid),
                plausible_type_pairs=plausible_type_pairs,
                corpus_pool=corpus_pool,
                rng=rng,
                type_restricted=type_restricted_false_positives,
            )
            for fp_b_id, fp_b_info in candidates:
                # Keep gold's relation_type so this perturbation isolates the
                # wrong-entity_B signal (otherwise it overlaps with label_flip).
                fp_external_pool.append(_make(
                    pmid, a_id, info_a, rel_type, fp_b_id, fp_b_info, 0, "fp_external",
                ))

    # Phase 2: cap the fp_external pool at the per-split label_flip count
    if len(fp_external_pool) > label_flip_count:
        fp_external_pool = rng.sample(fp_external_pool, label_flip_count)
    samples.extend(fp_external_pool)

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
# - Canonical mention selection: currently "longest". Consider "most frequent"
#   or "first-occurring" and ablate.
#
# - NoRelation distance metric: pipeline uses the character gap; the token and
#   sentence variants are computed in a separate case study. Pick one, then wire
#   the chosen metric in here.
#
# - Per-perturbation spin-off datasets for one-thing-at-a-time analysis
#   (issue #3).