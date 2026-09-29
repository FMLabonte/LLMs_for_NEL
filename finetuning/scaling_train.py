"""Data-scaling experiment: train one BioRED relation classifier at a fixed training-set size.

This is a standalone, parametrised copy of the training path in
``finetuning/finetuning_pubmedbert.ipynb``, specialised for the data-scaling study
(``finetuning/10_SCALING_TEST.md``).
The notebook is left untouched; the breaking change here is that the training set is
sub-sampled to ``--n-papers`` BioRED papers instead of always using all of them.

For one invocation it:

  1. Draws a nested seed-42 sub-sample of ``--n-papers`` papers from the papers that have
     a synthetic abstract (so a real run and a synthetic run at the same size see the exact
     same paper ids).
  2. Builds the training samples from either the real BioRED abstracts (``--source biored``)
     or the first synthetic Qwen3 generation (``--source synthetic``); entities, gold
     relations and distance-matched negatives always come from the real BioRED annotations,
     the ``NoRelation`` distance statistics are always fit on the *full* real BioRED Train.
  3. Trains ``--epochs`` epochs, evaluating on the full Dev split every epoch (real BioRED
     Dev for ``biored``, the synthetic Qwen3 Dev for ``synthetic``) and keeping every epoch
     checkpoint. The Dev-selected checkpoint is the one with the best Dev ``f1_micro``.
  4. Scores every epoch checkpoint on the full real BioRED Test with the BioRED F1 metric,
     both variants (``matched`` and ``all_pairs``), reusing the project's scoring code
     verbatim, and writes ``scaling_result.json`` into the run directory.

Only the amount of training data varies between sizes; Dev and Test stay at full size.

Usage (CUDA lives in the flake)::

    nix develop . -c python finetuning/scaling_train.py --source synthetic --n-papers 100
    nix develop . -c python finetuning/scaling_train.py --source biored    --n-papers all --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(REPO_ROOT))

from dataset_preparation.perturbations import BIORED_RELATION_TYPES, NO_RELATION_LABEL  # noqa: E402
from dataset_preparation.prepare_pure_biored import (  # noqa: E402
    build_pure_biored_samples,
    compute_relation_distance_stats,
)
from dataset_preparation.synthetic_abstracts import build_synthetic_parsed  # noqa: E402
from pubtator_parser import parse_pubtator  # noqa: E402
from finetuning.eval_biored_metric_single import build_eval_frame  # noqa: E402
from finetuning.reevaluate_biored_metric import (  # noqa: E402
    compute_biored_f1,
    predict,
    resolve_tokenizer_dir,
    tokenize,
)

# --------------------------------------------------------------------------------------
# Fixed configuration (shared across every run of the sweep).
# --------------------------------------------------------------------------------------
DATA_DIR = REPO_ROOT / "Data"
PUBTATOR_FILE_TRAIN = DATA_DIR / "BioRED/Train.PubTator"
PUBTATOR_FILE_DEV = DATA_DIR / "BioRED/Dev.PubTator"
PUBTATOR_FILE_TEST = DATA_DIR / "BioRED/Test.PubTator"
SYNTH_DIR = DATA_DIR / "Synthetic abstracts"
SYNTH_TRAIN_JSON = SYNTH_DIR / "results_qwen3_8b_train.json"
SYNTH_DEV_JSON = SYNTH_DIR / "results_qwen3_8b_dev.json"

RELATION_LABELS: list[str] = BIORED_RELATION_TYPES + [NO_RELATION_LABEL]
LABEL2ID: dict[str, int] = {name: idx for idx, name in enumerate(RELATION_LABELS)}
ID2LABEL: dict[int, str] = {idx: name for name, idx in LABEL2ID.items()}
MASK_TOKEN = "[MASK]"

SAMPLE_SEED = 42  # nested sub-sample of papers
BATCH_SIZE = 32
LR = 5e-5
RUNS_DIR = REPO_ROOT / "finetuning" / "scaling_runs"
EVAL_VARIANTS: tuple[tuple[str, float | None], ...] = (("matched", 1.0), ("all_pairs", None))


def covered_paper_ids(train_pubtator: Path, synth_json: Path) -> list[str]:
    """Return real BioRED Train pmids that also have a synthetic abstract, sorted.

    Sorting makes the subsequent seed-42 shuffle reproducible and independent of file order.
    """
    meta, _, _ = parse_pubtator(train_pubtator)
    train_pmids = set(meta["pmid"].astype(str))
    synth_pmids = {str(record["paper_id"]) for record in json.loads(synth_json.read_text("utf-8"))}
    return sorted(train_pmids & synth_pmids)


def select_paper_ids(covered: list[str], n_papers: int, seed: int = SAMPLE_SEED) -> list[str]:
    """Draw a nested seed-42 sub-sample of ``n_papers`` ids from ``covered``.

    A single shuffle with a fixed seed makes the sizes nested (the 100-set is a subset of the
    200-set, etc.), so scaling adds data rather than resampling it.
    """
    shuffled = list(covered)
    random.Random(seed).shuffle(shuffled)
    return sorted(shuffled[:n_papers])


def _rich_samples(parsed: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], distance_stats: dict[str, float]) -> pd.DataFrame:
    """Build the notebook's rich per-pair training frame (prompt/target_relation/label).

    Mirrors ``build_samples`` (rich variant) in ``finetuning_pubmedbert.ipynb``: gold
    relations plus distance-matched ``NoRelation`` pairs, with the prompt built from the
    entity surface strings and the abstract text.
    """
    samples = build_pure_biored_samples(*parsed, distance_stats=distance_stats, verbose=False)
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


def _filter_parsed(
    parsed: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], pmids: set[str], synthetic: bool
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Restrict a ``(meta, anns, rels)`` triple to the given base pmids.

    For synthetic virtual documents the pmid is ``"<pmid>#g<n>"``; the base id before the
    ``#`` is matched against ``pmids``.
    """
    meta, anns, rels = parsed

    def base(series: pd.Series) -> pd.Series:
        ids = series.astype(str)
        return ids.str.split("#").str[0] if synthetic else ids

    return (
        meta[base(meta["pmid"]).isin(pmids)].copy(),
        anns[base(anns["pmid"]).isin(pmids)].copy(),
        rels[base(rels["pmid"]).isin(pmids)].copy(),
    )


def build_train_df(source: str, selected: list[str], distance_stats: dict[str, float]) -> pd.DataFrame:
    """Build the sub-sampled training frame for the chosen abstract source."""
    selected_set = set(selected)
    if source == "biored":
        parsed = _filter_parsed(parse_pubtator(PUBTATOR_FILE_TRAIN), selected_set, synthetic=False)
    elif source == "synthetic":
        # First generation only, so one synthetic abstract matches one real abstract per paper.
        parsed_full = build_synthetic_parsed(SYNTH_TRAIN_JSON, PUBTATOR_FILE_TRAIN, (1,))
        parsed = _filter_parsed(parsed_full, selected_set, synthetic=True)
    else:
        raise ValueError(f"Unknown source: {source!r}")
    return _rich_samples(parsed, distance_stats)


def build_dev_df(source: str, distance_stats: dict[str, float]) -> pd.DataFrame:
    """Build the full Dev frame used for checkpoint selection (never sub-sampled)."""
    if source == "biored":
        parsed = parse_pubtator(PUBTATOR_FILE_DEV)
    else:
        parsed = build_synthetic_parsed(SYNTH_DEV_JSON, PUBTATOR_FILE_DEV, (1,))
    return _rich_samples(parsed, distance_stats)


def hardware_args() -> dict[str, object]:
    """Return device-specific TrainingArguments, halving batch/LR on <=8.5 GB GPUs.

    Set ``C5_PIN_HALF=1`` to force the halved values (batch 16 / LR 2.5e-5) regardless of VRAM,
    so runs on a larger GPU (e.g. the 12 GB RTX 3060) stay comparable to the 8 GB machine's C5
    numbers (report 20 cross-machine parallel training).
    """
    if not torch.cuda.is_available():
        return {"per_device_train_batch_size": BATCH_SIZE, "per_device_eval_batch_size": BATCH_SIZE, "learning_rate": LR}
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    pin_half = os.environ.get("C5_PIN_HALF") == "1"
    div = 2 if (total_vram_gb <= 8.5 or pin_half) else 1
    return {
        "fp16": True,
        "per_device_train_batch_size": BATCH_SIZE // div,
        "per_device_eval_batch_size": BATCH_SIZE // div,
        "learning_rate": LR / div,
    }


def compute_metrics(eval_pred) -> dict[str, float]:
    """9-class micro-F1 (the checkpoint-selection signal), matching the notebook."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"f1_micro": float(f1_score(labels, predictions, average="micro"))}


def find_checkpoints(run_dir: Path) -> list[Path]:
    """Return ``checkpoint-*`` dirs of a run sorted by global step (i.e. by epoch)."""
    return sorted(
        (path for path in run_dir.glob("checkpoint-*") if path.is_dir()),
        key=lambda path: int(path.name.split("-")[-1]),
    )


def score_checkpoints(
    checkpoints: list[Path], test_parsed, train_distance_stats
) -> dict[str, dict[str, dict[str, float]]]:
    """Score a given list of checkpoints on the full real BioRED Test with the BioRED metric.

    Reuses ``build_eval_frame`` / ``predict`` / ``compute_biored_f1`` so the numbers match
    ``eval_all_epochs_biored_metric.py`` exactly. Returns ``{checkpoint: {variant: scores}}``.
    Passing a single checkpoint (e.g. the Dev-selected one) is the fast, dev-only scoring path.
    """
    reference_tokenizer = AutoTokenizer.from_pretrained(resolve_tokenizer_dir(checkpoints[-1]))
    frames: dict[str, object] = {}
    for name, ratio in EVAL_VARIANTS:
        frame = build_eval_frame(test_parsed, train_distance_stats, ratio)
        frames[name] = (frame, tokenize(frame, reference_tokenizer))

    results: dict[str, dict[str, dict[str, float]]] = {}
    for checkpoint in checkpoints:
        tokenizer_dir = resolve_tokenizer_dir(checkpoint)
        per_variant: dict[str, dict[str, float]] = {}
        for name, (frame, tokenized) in frames.items():
            predicted = predict(checkpoint, tokenizer_dir, tokenized)
            predicted["target_relation"] = frame["target_relation"].values
            overall = compute_biored_f1(
                predicted, pred_col="predicted_relation", gold_col="target_relation", negative_label=NO_RELATION_LABEL
            )["overall"]
            per_variant[name] = {
                "rows": len(frame),
                "biored_f1": overall["f1"],
                "biored_precision": overall["precision"],
                "biored_recall": overall["recall"],
                "tp": overall["tp"],
                "fp": overall["fp"],
                "fn": overall["fn"],
            }
        results[checkpoint.name] = per_variant
    return results


def score_all_epochs(run_dir: Path, test_parsed, train_distance_stats) -> dict[str, dict[str, dict[str, float]]]:
    """Score every epoch checkpoint in ``run_dir`` (thin wrapper over :func:`score_checkpoints`)."""
    return score_checkpoints(find_checkpoints(run_dir), test_parsed, train_distance_stats)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["biored", "synthetic"], required=True)
    parser.add_argument("--n-papers", required=True, help="number of papers, or 'all' for every covered paper")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--model", default="NeuML/pubmedbert-base-embeddings")
    parser.add_argument("--dry-run", action="store_true", help="build data and print sizes, do not train")
    args = parser.parse_args()

    covered = covered_paper_ids(PUBTATOR_FILE_TRAIN, SYNTH_TRAIN_JSON)
    n_papers = len(covered) if args.n_papers == "all" else int(args.n_papers)
    n_papers = min(n_papers, len(covered))
    selected = select_paper_ids(covered, n_papers)

    # NoRelation distance statistics are always fit on the FULL real BioRED Train.
    train_distance_stats = compute_relation_distance_stats(*parse_pubtator(PUBTATOR_FILE_TRAIN), verbose=False)

    train_df = build_train_df(args.source, selected, train_distance_stats)
    dev_df = build_dev_df(args.source, train_distance_stats)

    print(f"source            : {args.source}")
    print(f"covered papers    : {len(covered)}")
    print(f"selected papers   : {len(selected)} (n_papers={n_papers})")
    print(f"train rows        : {len(train_df)}  (gold+neg)")
    print(f"dev rows          : {len(dev_df)}")
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
    tokenized_dev = to_tokenized(dev_df)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"scaling_{args.source}_n{n_papers:03d}_{args.model.replace('/', '-')}_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    out_dir = RUNS_DIR / tag

    # Seed BEFORE model creation so the classifier head init is deterministic and
    # independent of import order (report 14 reproducibility artefact).
    set_seed(SAMPLE_SEED)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(RELATION_LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    )
    if torch.cuda.is_available():
        model = model.to("cuda")

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_micro",
        greater_is_better=True,
        save_total_limit=args.epochs,
        report_to=[],
        **hardware_args(),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_dev,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # Per-epoch Dev f1_micro (selection signal) from the training log.
    dev_f1_by_epoch: dict[int, float] = {}
    for entry in trainer.state.log_history:
        if "eval_f1_micro" in entry and "epoch" in entry:
            dev_f1_by_epoch[int(round(entry["epoch"]))] = float(entry["eval_f1_micro"])

    test_parsed = parse_pubtator(PUBTATOR_FILE_TEST)
    epoch_scores = score_all_epochs(out_dir, test_parsed, train_distance_stats)

    checkpoints = find_checkpoints(out_dir)
    # Dev-selected checkpoint: the one Trainer kept as best (max Dev f1_micro).
    selected_ckpt = Path(trainer.state.best_model_checkpoint).name if trainer.state.best_model_checkpoint else checkpoints[-1].name
    selected_epoch = checkpoints.index(out_dir / selected_ckpt) + 1

    per_epoch = []
    for epoch, checkpoint in enumerate(checkpoints, start=1):
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
        "source": args.source,
        "model": args.model,
        "n_papers": n_papers,
        "seed": SAMPLE_SEED,
        "epochs": args.epochs,
        "selected_pmids": selected,
        "run_dir": str(out_dir),
        "dev_selected_epoch": selected_epoch,
        "dev_selected_checkpoint": selected_ckpt,
        "per_epoch": per_epoch,
    }
    result_path = out_dir / "scaling_result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    sel = per_epoch[selected_epoch - 1]
    print(f"\nDev-selected epoch {selected_epoch} ({selected_ckpt})")
    print(f"  matched   BioRED F1 = {sel['matched']['biored_f1']:.4f}")
    print(f"  all_pairs BioRED F1 = {sel['all_pairs']['biored_f1']:.4f}")
    print(f"Wrote {result_path}")


if __name__ == "__main__":
    main()
