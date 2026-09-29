"""Canonical-recipe (C5) trainer for the recipe-matched SYNTH_COMPARE comparison.

Report ``finetuning/12_CANONICAL_COMPARE.md`` puts the whole SYNTH_COMPARE report
series (real / synthetic-8B / synthetic-4B / combined, negative-ratio and QC-subset
variants) on ONE frozen recipe so a single real baseline is comparable across all of
them. This script trains one such model.

The recipe (C5) is exactly the scaling-test recipe of ``scaling_train.py`` (reports
10/11): PubMedBERT, the 394 seed-42 matched papers, distance-matched negatives fit on
the full real BioRED Train, 5 epochs, and every epoch checkpoint scored on the full
real BioRED Test (matched + all rels.) with the BioRED F1 metric. The only knobs this
script adds over ``scaling_train.py`` are the abstract *source*, the number of
*generations*, and the training *negative ratio* -- the axes the SYNTH_COMPARE reports
vary. Evaluation is reused verbatim from ``scaling_train.py`` so numbers stay comparable
to reports 10/11.

Sources:

  * ``biored``   -- real BioRED Train abstracts (the canonical real baseline).
  * ``synth8b``  -- Qwen3-8B synthetic abstracts.
  * ``synth4b``  -- Qwen3-4B synthetic abstracts.
  * ``combined`` -- real BioRED Train concatenated with the Qwen3-8B synthetic abstracts
    (one synthetic paraphrase per real paper at ``--generations 1``). Has no legitimate
    Dev split, so it trains through with no checkpoint selection (``--dev-source none``).

Usage (CUDA lives in the flake)::

    nix develop . -c python finetuning/canonical_train.py --source synth4b --n-papers all
    nix develop . -c python finetuning/canonical_train.py --source combined --n-papers all --dev-source none
    nix develop . -c python finetuning/canonical_train.py --source synth8b --n-papers all --generations 3
    nix develop . -c python finetuning/canonical_train.py --source synth8b --n-papers all --neg-ratio all
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset_preparation.perturbations import NO_RELATION_LABEL
from dataset_preparation.prepare_pure_biored import (
    build_pure_biored_samples,
    compute_relation_distance_stats,
)
from dataset_preparation.synthetic_abstracts import (
    GENERATION_KEY,
    build_synthetic_parsed,
    clean_markdown_text,
    load_synthetic_abstracts,
)
from pubtator_parser import parse_pubtator
from finetuning.qc_abstract_selection import (
    load_decisions,
    load_norel_decisions,
    qc_allowed_generations,
    verify_allowed_against_synthetic,
)
from finetuning.train_06_1_controls import (
    passing_paper_ids,
    restrict_parsed,
    select_failed_one_per_paper,
    select_random_one_per_paper,
)
from finetuning.scaling_train import (
    BATCH_SIZE,
    ID2LABEL,
    LABEL2ID,
    MASK_TOKEN,
    PUBTATOR_FILE_DEV,
    PUBTATOR_FILE_TEST,
    PUBTATOR_FILE_TRAIN,
    RELATION_LABELS,
    SAMPLE_SEED,
    SYNTH_DIR,
    _filter_parsed,
    compute_metrics,
    covered_paper_ids,
    find_checkpoints,
    hardware_args,
    score_all_epochs,
    score_checkpoints,
    select_paper_ids,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "finetuning" / "canonical_runs"
SYNTH_TRAIN_JSON = {
    "synth8b": SYNTH_DIR / "results_qwen3_8b_train.json",
    "synth4b": SYNTH_DIR / "results_qwen3_4b_train.json",
}
SYNTH_DEV_JSON = {
    "synth8b": SYNTH_DIR / "results_qwen3_8b_dev.json",
    "synth4b": SYNTH_DIR / "results_qwen3_4b_dev.json",
}
# What the matched-paper set is defined against (real papers that also have an 8B abstract).
MATCHED_SYNTH_JSON = SYNTH_DIR / "results_qwen3_8b_train.json"
# optimizer/scheduler/rng state we never resume from; stripped after scoring to save disk.
PRUNE_AFTER_SCORING = ("optimizer.pt", "scheduler.pt", "scaler.pt", "rng_state.pth")

# QC-subset study (report 06/06_1) constants, kept identical to train_06_1_controls.py.
QC_CSV = REPO_ROOT / "qc_filtering" / "abstract_decisions.csv"
QC_MODEL = "qwen3_8b"
QC_SEED_TRAIN = 42
QC_SEED_DEV = 142
QC_GENERATIONS = (1, 2, 3)  # allowed-set picks the specific generation; this only bounds the range

# QC acceptance-level ladder (report 09) constants.
ACCEPTANCE_CSV = REPO_ROOT / "qc_filtering" / "acceptance_levels" / "acceptance_levels.csv"
ACCEPTANCE_LEVELS = ("L0_strict", "L1_rate10", "L2_rate20", "L3_rate33", "L4_rate50", "L5_unfiltered")

# Auto Dev split per source: synthetic selects on its own synthetic Dev, real on real Dev,
# combined has no legitimate Dev split (train through, no selection).
DEFAULT_DEV_SOURCE = {
    "biored": "biored",
    "synth8b": "synth8b",
    "synth4b": "synth4b",
    "combined": "none",
}


def rich_samples(
    parsed: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    distance_stats: dict[str, float],
    no_relation_ratio: float | None,
) -> pd.DataFrame:
    """Build the notebook's rich per-pair frame at a given training negative ratio.

    Mirrors ``scaling_train._rich_samples`` but exposes ``no_relation_ratio`` (1.0 = 1:1,
    2.0 = 1:2, ``None`` = every co-mentioned unrelated pair) so the negative-ratio study
    (report 05) can be reproduced under the same recipe.
    """
    samples = build_pure_biored_samples(
        *parsed, distance_stats=distance_stats, no_relation_ratio=no_relation_ratio, verbose=False
    )
    frame = pd.DataFrame([sample.to_dict() for sample in samples])
    frame = frame[frame["perturbation"].isin(["gold", "false_positive"])].copy()
    frame["prompt"] = frame.apply(
        lambda row: (
            f"Relation: {row['entity_a_text']} -> {MASK_TOKEN} -> {row['entity_b_text']}\n"
            f"Context: {row['abstract']}"
        ),
        axis=1,
    )
    frame["target_relation"] = np.where(
        frame["perturbation"] == "false_positive", NO_RELATION_LABEL, frame["relation_type"]
    )
    frame["label"] = frame["target_relation"].map(LABEL2ID)
    return frame


def build_source_parsed(
    source: str, pubtator_file: Path, synth_json: Path | None, generations: tuple[int, ...]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return an unfiltered ``(meta, anns, rels)`` triple for one abstract source."""
    if source == "biored":
        return parse_pubtator(pubtator_file)
    return build_synthetic_parsed(synth_json, pubtator_file, generations)


def build_train_df(
    source: str,
    selected: set[str],
    distance_stats: dict[str, float],
    generations: tuple[int, ...],
    no_relation_ratio: float | None,
) -> pd.DataFrame:
    """Build the sub-sampled training frame for the chosen source (394 matched papers)."""
    if source == "combined":
        real = _filter_parsed(parse_pubtator(PUBTATOR_FILE_TRAIN), selected, synthetic=False)
        synth = _filter_parsed(
            build_synthetic_parsed(SYNTH_TRAIN_JSON["synth8b"], PUBTATOR_FILE_TRAIN, generations),
            selected,
            synthetic=True,
        )
        return pd.concat(
            [rich_samples(real, distance_stats, no_relation_ratio), rich_samples(synth, distance_stats, no_relation_ratio)],
            ignore_index=True,
        )
    synth_json = SYNTH_TRAIN_JSON.get(source)
    parsed = build_source_parsed(source, PUBTATOR_FILE_TRAIN, synth_json, generations)
    parsed = _filter_parsed(parsed, selected, synthetic=source != "biored")
    return rich_samples(parsed, distance_stats, no_relation_ratio)


def cleaned_gen1_map(synth_json: Path) -> dict[str, str]:
    """Map ``paper_id -> Markdown-cleaned generation_1 text`` (papers with a non-empty gen_1).

    The Markdown wrapper Qwen3-8B adds to its abstracts is a deterministic domain cue a
    classifier can shortcut on (report 07); :func:`clean_markdown_text` strips it. The source
    JSON is never modified.
    """
    out: dict[str, str] = {}
    for pid, gens in load_synthetic_abstracts(synth_json).items():
        text = gens.get("generation_1")
        if text:
            out[str(pid)] = clean_markdown_text(text)
    return out


def build_swap_df(
    swap_n: int,
    selected: set[str],
    distance_stats: dict[str, float],
    no_relation_ratio: float | None,
) -> pd.DataFrame:
    """Real BioRED 394-paper baseline with ``swap_n`` papers' abstracts swapped for synth8b gen1.

    Report ``13_DATA_REPLACEMENT.md``: start from the C5 real-BioRED baseline (the 394 matched
    papers, 1:1 distance-matched negatives fit on the full real Train) and replace the abstract
    *text* of a seed-42 nested sub-sample of ``swap_n`` papers with the Markdown-cleaned Qwen3-8B
    ``generation_1`` abstract. The pair set is unchanged by the swap (golds + distance-matched
    negatives are a function of the real annotations and seed only), so only the Context text of
    the swapped papers differs from the real baseline. Nesting: ``select_paper_ids`` uses one
    seed-42 shuffle, so swap100 ⊂ swap200 ⊂ swap300 ⊂ the 394 selected papers.
    """
    covered = covered_paper_ids(PUBTATOR_FILE_TRAIN, MATCHED_SYNTH_JSON)
    swap_ids = set(select_paper_ids(covered, swap_n))
    meta, anns, rels = _filter_parsed(parse_pubtator(PUBTATOR_FILE_TRAIN), selected, synthetic=False)
    gen1 = cleaned_gen1_map(SYNTH_TRAIN_JSON["synth8b"])
    missing = swap_ids - set(gen1)
    if missing:
        raise ValueError(f"{len(missing)} swap papers lack a synthetic generation_1 abstract")
    meta = meta.copy()
    mask = meta["pmid"].astype(str).isin(swap_ids)
    if int(mask.sum()) != swap_n:
        raise ValueError(f"expected {swap_n} papers to swap, matched {int(mask.sum())} in meta")
    meta.loc[mask, "abstract"] = meta.loc[mask, "pmid"].astype(str).map(gen1)
    return rich_samples((meta, anns, rels), distance_stats, no_relation_ratio)


def build_qc_swap_df(
    arm: str,
    swap_frac: int,
    distance_stats: dict[str, float],
    no_relation_ratio: float | None,
) -> tuple[pd.DataFrame, set[str], set[str]]:
    """Data-replacement swap on a QC subset (report 21 extension): the report-13 swap, but on the
    QC arms instead of the full 394 set.

    Start from the ``real_subset`` / ``real_subset_norel`` endpoint (real BioRED text on the arm's
    QC-passing papers) and replace ``swap_frac`` percent of them with the arm's QC-passing synthetic
    generation. So the 0%% endpoint is the base-table real-subset row and the 100%% endpoint is the
    base-table QC synthetic row (``dedup`` / ``dedup_norel``).

    To make the 100%% endpoint identical to the QC synthetic base row, the swapped-in text is the
    *exact* generation that arm trains on -- the highest-``mean_prob`` passing generation per paper
    (not gen1), taken *raw* from the synth8b JSON (the QC arms apply no Markdown cleanup, unlike the
    report-13 swap). Only the Context text of the swapped papers changes; the pair set (golds +
    distance-matched negatives on the real annotations) is invariant across the whole gradient.

    Nesting: the swap subset is a single seed-42 (``SAMPLE_SEED``) nested draw over the sorted QC
    paper ids, so 25%% ⊂ 50%% ⊂ 75%%; which papers are swapped is fixed across training seeds, matching
    report-13. Returns ``(train_df, keep_ids, swap_ids)``.

    Arms:
      * ``dedup``       -- the 212 QC-passing papers (stated-only dynamic QC).
      * ``dedup_norel`` -- the 112 papers the NoRelation-aware dynamic QC keeps.
    """
    if arm == "dedup_norel":
        decisions = load_norel_decisions()
    else:
        decisions = load_decisions(QC_CSV, expected_rule="dynamic")
    train_allowed = qc_allowed_generations(decisions, QC_MODEL, "train", deduplicate=True)
    verify_allowed_against_synthetic(train_allowed, SYNTH_TRAIN_JSON["synth8b"], QC_MODEL, "train")

    keep = {pid for pid, _gen in train_allowed}
    synth = load_synthetic_abstracts(SYNTH_TRAIN_JSON["synth8b"])
    gen_text: dict[str, str] = {}
    for pid, gen in train_allowed:
        text = synth[pid].get(GENERATION_KEY.format(index=gen))
        if not text:
            raise ValueError(f"paper {pid} lacks QC-selected generation_{gen} text in synth8b train JSON")
        gen_text[pid] = text  # raw (no Markdown cleanup), to match the QC arm's endpoint text

    swap_n = round(swap_frac / 100 * len(keep))
    swap_ids = set(select_paper_ids(sorted(keep), swap_n))
    meta, anns, rels = restrict_parsed(parse_pubtator(PUBTATOR_FILE_TRAIN), keep)
    meta = meta.copy()
    mask = meta["pmid"].astype(str).isin(swap_ids)
    if int(mask.sum()) != swap_n:
        raise ValueError(f"expected {swap_n} papers to swap, matched {int(mask.sum())} in meta")
    meta.loc[mask, "abstract"] = meta.loc[mask, "pmid"].astype(str).map(gen_text)
    return rich_samples((meta, anns, rels), distance_stats, no_relation_ratio), keep, swap_ids


def build_dev_df(
    dev_source: str, distance_stats: dict[str, float], no_relation_ratio: float | None
) -> pd.DataFrame:
    """Build the full Dev frame used for checkpoint selection (never sub-sampled)."""
    if dev_source == "biored":
        parsed = parse_pubtator(PUBTATOR_FILE_DEV)
    else:
        parsed = build_synthetic_parsed(SYNTH_DEV_JSON[dev_source], PUBTATOR_FILE_DEV, (1,))
    return rich_samples(parsed, distance_stats, no_relation_ratio)


def build_qc_arm(
    qc_arm: str, distance_stats: dict[str, float], no_relation_ratio: float | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (train, dev) frames for a QC-subset arm, reusing report-06 selections verbatim.

    Arms (all at the QC-dedup one-abstract-per-paper budget):
      * ``dedup``       -- synthetic, highest-mean_prob *passing* gen per paper (212 papers).
      * ``dedup_norel`` -- same recipe as ``dedup`` but the ``passed`` decision comes from the
        NoRelation-aware CSV (dynamic rule, also judges the implicit NoRelation pairs), which
        leaves 112 train / 21 dev papers. mean_prob for the per-paper dedup is joined from the
        stated-only CSV. Dev selection on the synthetic 8B Dev split.
      * ``allpass``     -- synthetic, every *passing* generation (212 papers, up to 3 gens each).
      * ``real_subset`` -- real BioRED text on the same 212 QC-passing papers.
      * ``random212``   -- synthetic, N random papers (=#passing), one random gen each (seed 42).
      * ``qc_failed``   -- synthetic, the 182 no-pass papers, best failing gen each.
    """
    if qc_arm == "dedup_norel":
        norel = load_norel_decisions()
        train_allowed = qc_allowed_generations(norel, QC_MODEL, "train", deduplicate=True)
        dev_allowed = qc_allowed_generations(norel, QC_MODEL, "dev", deduplicate=True)
        verify_allowed_against_synthetic(train_allowed, SYNTH_TRAIN_JSON["synth8b"], QC_MODEL, "train")
        verify_allowed_against_synthetic(dev_allowed, SYNTH_DEV_JSON["synth8b"], QC_MODEL, "dev")
        train_parsed = build_synthetic_parsed(SYNTH_TRAIN_JSON["synth8b"], PUBTATOR_FILE_TRAIN, QC_GENERATIONS, allowed=train_allowed)
        dev_parsed = build_synthetic_parsed(SYNTH_DEV_JSON["synth8b"], PUBTATOR_FILE_DEV, QC_GENERATIONS, allowed=dev_allowed)
        return (
            rich_samples(train_parsed, distance_stats, no_relation_ratio),
            rich_samples(dev_parsed, distance_stats, no_relation_ratio),
        )

    if qc_arm == "real_subset_norel":
        # Real BioRED text on the same papers the NoRel-aware dedup keeps (112 train papers) -
        # the real-text counterpart to the ``dedup_norel`` synthetic arm. Dev = full real Dev.
        norel = load_norel_decisions()
        train_allowed = qc_allowed_generations(norel, QC_MODEL, "train", deduplicate=True)
        keep = {pid for pid, _gen in train_allowed}
        train_parsed = restrict_parsed(parse_pubtator(PUBTATOR_FILE_TRAIN), keep)
        dev_parsed = parse_pubtator(PUBTATOR_FILE_DEV)
        return (
            rich_samples(train_parsed, distance_stats, no_relation_ratio),
            rich_samples(dev_parsed, distance_stats, no_relation_ratio),
        )

    qc = load_decisions(QC_CSV, expected_rule="dynamic")

    if qc_arm == "real_subset":
        keep = passing_paper_ids(qc, "train")
        train_parsed = restrict_parsed(parse_pubtator(PUBTATOR_FILE_TRAIN), keep)
        dev_parsed = parse_pubtator(PUBTATOR_FILE_DEV)
        return (
            rich_samples(train_parsed, distance_stats, no_relation_ratio),
            rich_samples(dev_parsed, distance_stats, no_relation_ratio),
        )

    if qc_arm in ("dedup", "allpass"):
        dedup = qc_arm == "dedup"
        train_allowed = qc_allowed_generations(qc, QC_MODEL, "train", deduplicate=dedup)
        dev_allowed = qc_allowed_generations(qc, QC_MODEL, "dev", deduplicate=dedup)
    elif qc_arm == "random212":
        train_allowed = select_random_one_per_paper(qc, "train", len(passing_paper_ids(qc, "train")), QC_SEED_TRAIN)
        dev_allowed = select_random_one_per_paper(qc, "dev", len(passing_paper_ids(qc, "dev")), QC_SEED_DEV)
    elif qc_arm == "qc_failed":
        train_allowed = select_failed_one_per_paper(qc, "train")
        dev_allowed = select_failed_one_per_paper(qc, "dev")
    else:
        raise ValueError(f"unknown qc arm: {qc_arm!r}")

    verify_allowed_against_synthetic(train_allowed, SYNTH_TRAIN_JSON["synth8b"], QC_MODEL, "train")
    verify_allowed_against_synthetic(dev_allowed, SYNTH_DEV_JSON["synth8b"], QC_MODEL, "dev")
    train_parsed = build_synthetic_parsed(SYNTH_TRAIN_JSON["synth8b"], PUBTATOR_FILE_TRAIN, QC_GENERATIONS, allowed=train_allowed)
    dev_parsed = build_synthetic_parsed(SYNTH_DEV_JSON["synth8b"], PUBTATOR_FILE_DEV, QC_GENERATIONS, allowed=dev_allowed)
    return (
        rich_samples(train_parsed, distance_stats, no_relation_ratio),
        rich_samples(dev_parsed, distance_stats, no_relation_ratio),
    )


def _acceptance_frame(df: pd.DataFrame, split: str, level: str) -> pd.DataFrame:
    """8B rows KEPT at acceptance ``level`` for one split, with normalised ids.

    Kept means the boolean level column is True; ``paper_id`` -> str and the
    ``generation_<n>`` label -> its 1-based int index, matching the adapter's key space.
    """
    frame = df[(df["model"] == QC_MODEL) & (df["split"] == split) & (df[level])].copy()
    frame["paper_id"] = frame["paper_id"].astype(str)
    frame["gen_index"] = frame["generation"].astype(str).str.split("_").str[-1].astype(int)
    return frame


def acceptance_dedup_allowed(df: pd.DataFrame, split: str, level: str) -> set[tuple[str, int]]:
    """Highest-``mean_prob`` KEPT generation per paper at ``level`` -> (paper_id, gen_index) set."""
    frame = _acceptance_frame(df, split, level)
    idx = frame.groupby("paper_id")["mean_prob"].idxmax()
    chosen = frame.loc[idx]
    return set(zip(chosen["paper_id"], chosen["gen_index"]))


def acceptance_papers(df: pd.DataFrame, split: str, level: str) -> set[str]:
    """Papers with >=1 KEPT 8B generation at ``level`` in ``split``."""
    return set(_acceptance_frame(df, split, level)["paper_id"])


def build_acceptance_arm(
    level: str, arm: str, distance_stats: dict[str, float], no_relation_ratio: float | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (train, dev) frames for one acceptance-level arm (report 09).

    Arms (qwen3_8b only, deduped to one abstract per paper):
      * ``dedup``       -- synthetic, highest-mean_prob KEPT gen per paper at ``level``;
        Dev selection on the synthetic 8B Dev split, kept+deduped at the same level.
      * ``real_subset`` -- real BioRED text on the papers kept at ``level`` (>=1 kept
        8B/train abstract); Dev selection on the FULL real BioRED Dev split.
    """
    df = pd.read_csv(ACCEPTANCE_CSV)

    if arm == "real_subset":
        keep = acceptance_papers(df, "train", level)
        train_parsed = restrict_parsed(parse_pubtator(PUBTATOR_FILE_TRAIN), keep)
        dev_parsed = parse_pubtator(PUBTATOR_FILE_DEV)
        return (
            rich_samples(train_parsed, distance_stats, no_relation_ratio),
            rich_samples(dev_parsed, distance_stats, no_relation_ratio),
        )

    if arm != "dedup":
        raise ValueError(f"unknown acceptance arm: {arm!r}")

    train_allowed = acceptance_dedup_allowed(df, "train", level)
    dev_allowed = acceptance_dedup_allowed(df, "dev", level)
    verify_allowed_against_synthetic(train_allowed, SYNTH_TRAIN_JSON["synth8b"], QC_MODEL, "train")
    verify_allowed_against_synthetic(dev_allowed, SYNTH_DEV_JSON["synth8b"], QC_MODEL, "dev")
    train_parsed = build_synthetic_parsed(SYNTH_TRAIN_JSON["synth8b"], PUBTATOR_FILE_TRAIN, QC_GENERATIONS, allowed=train_allowed)
    dev_parsed = build_synthetic_parsed(SYNTH_DEV_JSON["synth8b"], PUBTATOR_FILE_DEV, QC_GENERATIONS, allowed=dev_allowed)
    return (
        rich_samples(train_parsed, distance_stats, no_relation_ratio),
        rich_samples(dev_parsed, distance_stats, no_relation_ratio),
    )


def parse_neg_ratio(value: str) -> float | None:
    """Map the ``--neg-ratio`` CLI value to ``no_relation_ratio`` (``all`` -> ``None``)."""
    return None if value == "all" else float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["biored", "synth8b", "synth4b", "combined"], default="synth8b")
    parser.add_argument(
        "--qc-arm",
        choices=["none", "dedup", "dedup_norel", "allpass", "real_subset", "real_subset_norel", "random212", "qc_failed"],
        default="none",
        help="QC-subset study arm (report 06/06_1); overrides --source/--n-papers when set",
    )
    parser.add_argument(
        "--acceptance-level",
        choices=list(ACCEPTANCE_LEVELS),
        default=None,
        help="QC acceptance-level ladder (report 09); overrides --source/--qc-arm when set",
    )
    parser.add_argument(
        "--acceptance-arm",
        choices=["dedup", "real_subset"],
        default="dedup",
        help="acceptance-level arm: synthetic dedup, or real-text matching subset",
    )
    parser.add_argument(
        "--swap-n",
        type=int,
        default=None,
        help="data-replacement study (report 13): real 394-paper baseline with N papers' "
        "abstracts swapped for Markdown-cleaned synth8b gen1; overrides --source. Dev on real BioRED.",
    )
    parser.add_argument(
        "--qc-swap-arm",
        choices=["dedup", "dedup_norel"],
        default=None,
        help="data-replacement swap on a QC subset (report 21 extension): start from real text on "
        "the QC arm's papers and swap --swap-frac%% of them for that arm's QC-passing synthetic gen; "
        "Dev on real BioRED. Overrides --source/--qc-arm/--swap-n.",
    )
    parser.add_argument(
        "--swap-frac",
        type=float,
        choices=[25, 50, 75, 87.5, 93.75, 96.875],
        default=None,
        help="percent of the QC arm's papers to swap (paired with --qc-swap-arm)",
    )
    parser.add_argument(
        "--include-dev-in-train",
        action="store_true",
        help="train+dev experiment (report 21): fold the source's own Dev split into training; "
        "no Dev selection, so pair with --fixed-epoch. Only for --source biored/synth8b/synth4b.",
    )
    parser.add_argument(
        "--fixed-epoch",
        type=int,
        default=None,
        help="score the checkpoint at this epoch on Test and record it as the selected epoch "
        "(instead of Dev selection); used with --include-dev-in-train.",
    )
    parser.add_argument("--n-papers", default="all", help="number of papers, or 'all' for every matched paper")
    parser.add_argument(
        "--seed",
        type=int,
        default=SAMPLE_SEED,
        help="training seed: seeds the head init and HF data shuffling (report 21 multi-seed means)",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--generations", type=int, default=1, help="synthetic generations per paper (1 = matched volume)")
    parser.add_argument("--neg-ratio", default="1", choices=["1", "2", "3", "all"], help="training NoRelation ratio")
    parser.add_argument(
        "--dev-source",
        choices=["biored", "synth8b", "synth4b", "none"],
        default=None,
        help="Dev split for checkpoint selection; default is per-source (combined -> none)",
    )
    parser.add_argument("--model", default="NeuML/pubmedbert-base-embeddings")
    parser.add_argument("--dry-run", action="store_true", help="build data and print sizes, do not train")
    parser.add_argument(
        "--prune-checkpoints",
        action="store_true",
        help="delete checkpoint dirs after scoring (keep only canonical_result.json) to save disk",
    )
    parser.add_argument(
        "--score-dev-only",
        action="store_true",
        help="score only the Dev-selected checkpoint on Test (fast); per_epoch holds just that epoch",
    )
    args = parser.parse_args()

    generations = tuple(range(1, args.generations + 1))
    no_relation_ratio = parse_neg_ratio(args.neg_ratio)
    train_distance_stats = compute_relation_distance_stats(*parse_pubtator(PUBTATOR_FILE_TRAIN), verbose=False)

    if args.qc_swap_arm is not None:
        # Data-replacement swap on a QC subset (report 21 extension): overrides --source/--qc-arm.
        # Real text on the QC arm's papers with swap_frac%% swapped for the arm's QC-passing synth gen;
        # select on the full real BioRED Dev (matches the real_subset endpoint's selection axis).
        if args.swap_frac is None:
            parser.error("--qc-swap-arm requires --swap-frac")
        # Whole-number fractions keep the 2-digit tag (qcswap-dedup-25); fractional ones (87.5) use %g.
        frac_tag = f"{int(args.swap_frac):02d}" if args.swap_frac == int(args.swap_frac) else f"{args.swap_frac:g}"
        condition = f"qcswap-{args.qc_swap_arm}-{frac_tag}"
        args.source = "biored"  # real annotations + mixed text; keeps it out of result4_run()
        dev_source = "biored"
        train_df, keep_ids, swap_ids = build_qc_swap_df(
            args.qc_swap_arm, args.swap_frac, train_distance_stats, no_relation_ratio
        )
        selected = keep_ids
        n_papers = len(keep_ids)
        dev_df = build_dev_df(dev_source, train_distance_stats, no_relation_ratio)
        print(f"swapped papers    : {len(swap_ids)} / {n_papers} ({args.swap_frac}%)")
    elif args.swap_n is not None:
        # Data-replacement study (report 13): overrides --source. Real 394-paper baseline with
        # swap_n papers' abstracts replaced by cleaned synth8b gen1; select on real BioRED Dev.
        condition = f"swap{args.swap_n:03d}"
        args.source = "biored"  # real annotations + mixed text; keeps it out of result4_run()
        dev_source = "biored"
        covered = covered_paper_ids(PUBTATOR_FILE_TRAIN, MATCHED_SYNTH_JSON)
        selected = set(covered)  # the full 394-paper C5 baseline paper universe
        n_papers = len(selected)
        train_df = build_swap_df(args.swap_n, selected, train_distance_stats, no_relation_ratio)
        dev_df = build_dev_df(dev_source, train_distance_stats, no_relation_ratio)
    elif args.acceptance_level is not None:
        # Acceptance-level ladder (report 09): overrides --source/--qc-arm.
        condition = f"acc-{args.acceptance_level}-{args.acceptance_arm}"
        dev_source = "biored" if args.acceptance_arm == "real_subset" else "synth8b"
        n_papers = 0  # paper universe is level-defined, recorded in the result JSON
        selected: set[str] = set()
        train_df, dev_df = build_acceptance_arm(
            args.acceptance_level, args.acceptance_arm, train_distance_stats, no_relation_ratio
        )
    elif args.qc_arm != "none":
        # QC-subset arm: overrides --source/--n-papers; all four arms are Dev-selected.
        synth_arm = args.qc_arm not in ("real_subset", "real_subset_norel")
        default_dev = "synth8b" if synth_arm else "biored"
        dev_source = args.dev_source or default_dev
        # A synthetic QC arm selected on real BioRED Dev (via --dev-source biored) is a distinct
        # experiment (report 21 real-Dev endpoints): same synthetic training text, but the
        # checkpoint is picked on the full real Dev instead of the synthetic Dev. Tag it "-realdev"
        # so it does not collide with the synth-Dev run of the same arm.
        realdev = synth_arm and dev_source == "biored"
        condition = f"qc-{args.qc_arm}" + ("-realdev" if realdev else "")
        n_papers = 0  # paper universe is arm-defined (212/182), recorded via the run tag
        selected = set()
        train_df, dev_df = build_qc_arm(args.qc_arm, train_distance_stats, no_relation_ratio)
        if realdev:
            dev_df = build_dev_df("biored", train_distance_stats, no_relation_ratio)
    else:
        condition = args.source + ("-traindev" if args.include_dev_in_train else "")
        dev_source = args.dev_source or DEFAULT_DEV_SOURCE[args.source]
        covered = covered_paper_ids(PUBTATOR_FILE_TRAIN, MATCHED_SYNTH_JSON)
        n_papers = len(covered) if args.n_papers == "all" else min(int(args.n_papers), len(covered))
        selected = set(select_paper_ids(covered, n_papers))
        train_df = build_train_df(args.source, selected, train_distance_stats, generations, no_relation_ratio)
        if args.include_dev_in_train:
            # Train+Dev experiment (report 21 extension): fold this source's own Dev split into the
            # training set. There is then no Dev to select on, so the eval checkpoint is fixed via
            # --fixed-epoch to that seed's baseline-394 Dev-selected epoch.
            dev_train = build_dev_df(dev_source, train_distance_stats, no_relation_ratio)
            train_df = pd.concat([train_df, dev_train], ignore_index=True)
            dev_df = None
            dev_source = "none"
        else:
            dev_df = None if dev_source == "none" else build_dev_df(dev_source, train_distance_stats, no_relation_ratio)

    print(f"condition         : {condition}  (generations={generations}, neg_ratio={args.neg_ratio})")
    print(f"dev source        : {dev_source}")
    print(f"selected papers   : {len(selected)} (n_papers={n_papers})")
    print(f"train rows        : {len(train_df)}  (gold+neg)")
    print(f"dev rows          : {0 if dev_df is None else len(dev_df)}")
    print(train_df["target_relation"].value_counts().to_dict())

    if args.dry_run:
        print("dry-run: stopping before training")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def to_tokenized(frame: pd.DataFrame) -> Dataset:
        dataset = Dataset.from_pandas(frame.reset_index(drop=True))
        return dataset.map(
            lambda batch: tokenizer(batch["prompt"], truncation=True, padding=True, max_length=512),
            batched=True,
            batch_size=BATCH_SIZE,
        )

    tokenized_train = to_tokenized(train_df)
    tokenized_dev = None if dev_df is None else to_tokenized(dev_df)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    tag = (
        f"canonical_{condition}_g{args.generations}_neg{args.neg_ratio}_n{n_papers:03d}_s{args.seed}_"
        f"{args.model.replace('/', '-')}_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    )
    out_dir = RUNS_DIR / tag

    # Seed BEFORE model creation so the freshly added classifier head is initialised
    # deterministically and independent of import order (report 14 reproducibility artefact).
    set_seed(args.seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(RELATION_LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    )
    if torch.cuda.is_available():
        model = model.to("cuda")

    select = tokenized_dev is not None
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch" if select else "no",
        save_strategy="epoch",
        load_best_model_at_end=select,
        metric_for_best_model="eval_f1_micro" if select else None,
        greater_is_better=True if select else None,
        save_total_limit=args.epochs,
        report_to=[],
        seed=args.seed,
        **hardware_args(),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_dev,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics if select else None,
    )
    trainer.train()

    dev_f1_by_epoch: dict[int, float] = {}
    for entry in trainer.state.log_history:
        if "eval_f1_micro" in entry and "epoch" in entry:
            dev_f1_by_epoch[int(round(entry["epoch"]))] = float(entry["eval_f1_micro"])

    test_parsed = parse_pubtator(PUBTATOR_FILE_TEST)

    checkpoints = find_checkpoints(out_dir)
    if args.fixed_epoch is not None:
        # Train+dev experiment: no Dev to select on; the eval checkpoint is fixed to that seed's
        # baseline-394 Dev-selected epoch (report 21). Record it as the selected epoch.
        if not 1 <= args.fixed_epoch <= len(checkpoints):
            raise ValueError(f"--fixed-epoch {args.fixed_epoch} out of range (1..{len(checkpoints)})")
        selected_ckpt = checkpoints[args.fixed_epoch - 1].name
        selected_epoch = args.fixed_epoch
    elif select and trainer.state.best_model_checkpoint:
        selected_ckpt = Path(trainer.state.best_model_checkpoint).name
        selected_epoch = checkpoints.index(out_dir / selected_ckpt) + 1
    else:
        selected_ckpt = None
        selected_epoch = None

    # Dev-only scoring: score just the Dev-selected (or fixed) checkpoint on Test (fast). The score
    # is identical to full-epoch scoring; only the non-selected epochs are skipped.
    if (args.score_dev_only or args.fixed_epoch is not None) and selected_ckpt is not None:
        to_score = [out_dir / selected_ckpt]
    else:
        to_score = checkpoints
    epoch_scores = score_checkpoints(to_score, test_parsed, train_distance_stats)

    per_epoch = []
    for epoch, checkpoint in enumerate(checkpoints, start=1):
        if checkpoint.name not in epoch_scores:
            continue  # dev-only: non-selected epochs are not scored on Test
        scores = epoch_scores[checkpoint.name]
        per_epoch.append(
            {
                "epoch": epoch,
                "checkpoint": checkpoint.name,
                "dev_f1_micro": dev_f1_by_epoch.get(epoch),
                "matched": scores["matched"],
                "all_pairs": scores["all_pairs"],
            }
        )

    result = {
        "condition": condition,
        "source": args.source,
        "qc_arm": args.qc_arm,
        "swap_n": args.swap_n,
        "qc_swap_arm": args.qc_swap_arm,
        "swap_frac": args.swap_frac,
        "include_dev_in_train": args.include_dev_in_train,
        "fixed_epoch": args.fixed_epoch,
        "acceptance_level": args.acceptance_level,
        "acceptance_arm": args.acceptance_arm if args.acceptance_level is not None else None,
        "model": args.model,
        "generations": args.generations,
        "neg_ratio": args.neg_ratio,
        "dev_source": dev_source,
        "n_papers": n_papers,
        "seed": SAMPLE_SEED,
        "run_seed": args.seed,  # training seed (head init + shuffling); report 21 multi-seed means
        "seeded_init": True,  # set_seed(args.seed) before from_pretrained (report 14 fix)
        "epochs": args.epochs,
        "selected_pmids": sorted(selected),
        "run_dir": str(out_dir),
        "dev_selected_epoch": selected_epoch,
        "dev_selected_checkpoint": selected_ckpt,
        "per_epoch": per_epoch,
    }
    (out_dir / "canonical_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Drop optimizer/scheduler/rng state (never resumed from) to keep the sweep's disk bounded.
    for checkpoint in checkpoints:
        for name in PRUNE_AFTER_SCORING:
            (checkpoint / name).unlink(missing_ok=True)

    # Optionally drop the checkpoint weights entirely once scored; the result JSON keeps every
    # number, so the run is fully preserved for the report at ~kB instead of ~GB.
    if args.prune_checkpoints:
        for checkpoint in checkpoints:
            shutil.rmtree(checkpoint, ignore_errors=True)
        print(f"pruned {len(checkpoints)} checkpoints (kept canonical_result.json)")

    if selected_epoch is not None:
        # find by epoch number (per_epoch may hold only the selected epoch under --score-dev-only)
        sel = next((e for e in per_epoch if e["epoch"] == selected_epoch), per_epoch[0])
        print(f"\nDev-selected epoch {selected_epoch} ({selected_ckpt})")
        print(f"  matched   BioRED F1 = {sel['matched']['biored_f1']:.4f}")
        print(f"  all_pairs BioRED F1 = {sel['all_pairs']['biored_f1']:.4f}")
    else:
        best = max(per_epoch, key=lambda e: e["all_pairs"]["biored_f1"])
        print(f"\nNo Dev selection (dev_source=none). Best all_pairs epoch {best['epoch']} = {best['all_pairs']['biored_f1']:.4f}")
    print(f"Wrote {out_dir / 'canonical_result.json'}")


if __name__ == "__main__":
    main()
