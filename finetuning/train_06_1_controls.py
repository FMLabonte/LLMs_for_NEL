"""Control training runs for the QC-filtering study (report 06_1_SYNTH_FILTERED.md).

Three additional PubMedBERT relation-classifier runs, each at a ~one-abstract-per-paper
budget so they sit alongside the existing ``QC dedup`` row (212 QC-passing papers, one
abstract each) from ``06_SYNTH_FILTERED.md``:

- ``real_subset``: the *real* BioRED abstracts, but restricted to the 212 papers that have
  at least one QC-passing Qwen3-8B synthetic abstract. Isolates the real-text ceiling on
  exactly the papers QC kept.
- ``random212``: synthetic, 212 *random* papers (drawn from all 394 generated papers,
  seed 42), one *random* generation each. The random-selection control for QC dedup at an
  identical budget.
- ``qc_failed``: synthetic, the 182 papers where *no* generation passed QC, one abstract
  each (the highest ``mean_prob`` failing generation per paper, mirroring dedup's
  best-per-paper rule). Trains on the rejects.

The recipe (encoder, LR, epochs, distance-matched negatives fit on real BioRED Train,
prompt, trainer args) is copied verbatim from ``finetuning/finetuning_pubmedbert.ipynb`` so
the numbers stay directly comparable to the runs already in the report. Every epoch
checkpoint is kept; score them afterwards with
``eval_all_epochs_biored_metric.py <run_dir>`` on the held-out real BioRED Test.

Usage (CUDA lives in the flake)::

    RUN_MODE=real_subset nix develop -c python finetuning/train_06_1_controls.py
    RUN_MODE=random212   nix develop -c python finetuning/train_06_1_controls.py
    RUN_MODE=qc_failed   nix develop -c python finetuning/train_06_1_controls.py

Set ``BUILD_ONLY=1`` to construct and report the datasets without training (dry run).
"""

from __future__ import annotations

import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

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
)
import evaluate

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from finetuning.commons import parse_pubtator  # noqa: E402
from dataset_preparation.perturbations import BIORED_RELATION_TYPES, NO_RELATION_LABEL  # noqa: E402
from dataset_preparation.prepare_pure_biored import (  # noqa: E402
    build_pure_biored_samples,
    compute_relation_distance_stats,
)
from dataset_preparation.synthetic_abstracts import build_synthetic_parsed  # noqa: E402
from finetuning.qc_abstract_selection import (  # noqa: E402
    load_decisions,
    verify_allowed_against_synthetic,
)

# --- Recipe constants (must match finetuning_pubmedbert.ipynb) ------------------------
MODEL: str = "NeuML/pubmedbert-base-embeddings"
DATASET_NAME: Literal["BioRed"] = "BioRed"
F1_AVERAGE: str = "micro"
BATCH_SIZE: int = 32
NUM_PROCS: int = 32
LR: float = 0.00005
EPOCHS: int = 5
MAX_LENGTH: int = 512
MASK_TOKEN: str = "[MASK]"
QC_MODEL: str = "qwen3_8b"
RANDOM_SEED_TRAIN: int = 42
RANDOM_SEED_DEV: int = 142

DATA_DIR: Path = REPO_ROOT / "Data"
PUBTATOR_FILE_TRAIN: Path = DATA_DIR / "BioRED" / "Train.PubTator"
PUBTATOR_FILE_DEV: Path = DATA_DIR / "BioRED" / "Dev.PubTator"
PUBTATOR_FILE_TEST: Path = DATA_DIR / "BioRED" / "Test.PubTator"
SYNTH_DIR: Path = DATA_DIR / "Synthetic abstracts"
SYNTH_TRAIN_JSON: Path = SYNTH_DIR / "results_qwen3_8b_train.json"
SYNTH_DEV_JSON: Path = SYNTH_DIR / "results_qwen3_8b_dev.json"
QC_CSV: Path = REPO_ROOT / "qc_filtering" / "abstract_decisions.csv"

SYNTH_GENERATIONS: tuple[int, ...] = (1, 2, 3)

RELATION_LABELS: list[str] = BIORED_RELATION_TYPES + [NO_RELATION_LABEL]
label2id: dict[str, int] = {name: idx for idx, name in enumerate(RELATION_LABELS)}
id2label: dict[int, str] = {idx: name for name, idx in label2id.items()}


def build_samples(
    pubtator_file: Path,
    parsed: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None,
    distance_stats: dict[str, float] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build the relation-classification samples frame (copied from the notebook).

    Gold relations + distance-matched ``NoRelation`` examples, one prompt per pair.
    ``parsed`` decouples the abstract source from this function (real or synthetic triple).
    """
    if parsed is not None:
        meta_df, anns_df, rels_df = parsed
    else:
        meta_df, anns_df, rels_df = parse_pubtator(pubtator_file)

    samples = build_pure_biored_samples(
        meta_df, anns_df, rels_df, distance_stats=distance_stats, verbose=verbose
    )
    samples = pd.DataFrame([sample.to_dict() for sample in samples])
    samples = samples[samples["perturbation"].isin(["gold", "false_positive"])].copy()
    samples["prompt"] = samples.apply(
        lambda row: (
            f"Relation: {row['entity_a_text']} -> {MASK_TOKEN} -> {row['entity_b_text']}\n"
            f"Context: {row['abstract']}"
        ),
        axis=1,
    )
    samples["target_relation"] = np.where(
        samples["perturbation"] == "false_positive",
        NO_RELATION_LABEL,
        samples["relation_type"],
    )
    samples["label"] = samples["target_relation"].map(label2id)
    return samples


def _qc_frame(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """8B rows for one split, ``paper_id`` and ``generation`` normalised."""
    frame = df[(df["model"] == QC_MODEL) & (df["split"] == split)].copy()
    frame["paper_id"] = frame["paper_id"].astype(str)
    frame["gen_index"] = frame["generation"].astype(str).str.split("_").str[-1].astype(int)
    return frame


def passing_paper_ids(df: pd.DataFrame, split: str) -> set[str]:
    """Papers with >=1 QC-passing 8B generation in ``split``."""
    frame = _qc_frame(df, split)
    return set(frame.loc[frame["passed"], "paper_id"])


def select_random_one_per_paper(
    df: pd.DataFrame, split: str, n_papers: int, seed: int
) -> set[tuple[str, int]]:
    """``n_papers`` random papers (from all generated), one random generation each."""
    frame = _qc_frame(df, split)
    rng = random.Random(seed)
    papers = sorted(frame["paper_id"].unique())
    if n_papers > len(papers):
        n_papers = len(papers)
    chosen = rng.sample(papers, n_papers)
    allowed: set[tuple[str, int]] = set()
    for paper in chosen:
        gens = sorted(frame.loc[frame["paper_id"] == paper, "gen_index"].tolist())
        allowed.add((paper, rng.choice(gens)))
    return allowed


def select_failed_one_per_paper(df: pd.DataFrame, split: str) -> set[tuple[str, int]]:
    """Papers with *no* passing generation; best (highest mean_prob) failing gen each."""
    frame = _qc_frame(df, split)
    passed = passing_paper_ids(df, split)
    nopass = frame[~frame["paper_id"].isin(passed)]
    idx = nopass.groupby("paper_id")["mean_prob"].idxmax()
    chosen = nopass.loc[idx]
    return set(zip(chosen["paper_id"], chosen["gen_index"]))


def restrict_parsed(
    parsed: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], paper_ids: set[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Restrict a (meta, anns, rels) triple to a set of ``pmid`` strings."""
    meta, anns, rels = parsed
    ids = {str(pid) for pid in paper_ids}
    return (
        meta[meta["pmid"].astype(str).isin(ids)].copy(),
        anns[anns["pmid"].astype(str).isin(ids)].copy(),
        rels[rels["pmid"].astype(str).isin(ids)].copy(),
    )


accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")


def compute_metrics(eval_pred) -> dict[str, float]:
    """Accuracy + micro/macro/weighted F1 for the 9-way relation head."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        **accuracy_metric.compute(predictions=predictions, references=labels),
        **precision_metric.compute(predictions=predictions, references=labels, average="macro", zero_division=0),
        **recall_metric.compute(predictions=predictions, references=labels, average="macro", zero_division=0),
        "f1_micro": f1_metric.compute(predictions=predictions, references=labels, average="micro")["f1"],
        "f1_macro": f1_metric.compute(predictions=predictions, references=labels, average="macro")["f1"],
        "f1_weighted": f1_metric.compute(predictions=predictions, references=labels, average="weighted")["f1"],
    }


def main() -> None:
    """Build the datasets for ``RUN_MODE`` and train (unless ``BUILD_ONLY``)."""
    run_mode = os.environ.get("RUN_MODE", "")
    if run_mode not in {"real_subset", "random212", "qc_failed"}:
        raise SystemExit(f"RUN_MODE must be one of real_subset|random212|qc_failed, got {run_mode!r}")
    build_only = os.environ.get("BUILD_ONLY", "0") == "1"

    # Distance stats are always fit on the full real BioRED Train, as in the notebook.
    train_parsed = parse_pubtator(PUBTATOR_FILE_TRAIN)
    train_distance_stats = compute_relation_distance_stats(*train_parsed)

    test_df = build_samples(PUBTATOR_FILE_TEST, distance_stats=train_distance_stats, verbose=False)
    qc = load_decisions(QC_CSV, expected_rule="dynamic")

    tag: str
    if run_mode == "real_subset":
        tag = "real_qc-subset"
        keep = passing_paper_ids(qc, "train")
        print(f"[real_subset] real BioRED restricted to {len(keep)} QC-passing papers")
        train_df = build_samples(
            PUBTATOR_FILE_TRAIN,
            parsed=restrict_parsed(train_parsed, keep),
            distance_stats=train_distance_stats,
        )
        # Validation: full real BioRED Dev (standard selection signal, disjoint from Test).
        val_df = build_samples(PUBTATOR_FILE_DEV, distance_stats=train_distance_stats, verbose=False)

    else:
        if run_mode == "random212":
            tag = "synthetic_random212"
            n_pass = len(passing_paper_ids(qc, "train"))
            train_allowed = select_random_one_per_paper(qc, "train", n_pass, RANDOM_SEED_TRAIN)
            dev_allowed = select_random_one_per_paper(
                qc, "dev", len(passing_paper_ids(qc, "dev")), RANDOM_SEED_DEV
            )
        else:  # qc_failed
            tag = "synthetic_qc-failed"
            train_allowed = select_failed_one_per_paper(qc, "train")
            dev_allowed = select_failed_one_per_paper(qc, "dev")

        print(f"[{run_mode}] train abstracts={len(train_allowed)} dev abstracts={len(dev_allowed)}")
        verify_allowed_against_synthetic(train_allowed, SYNTH_TRAIN_JSON, QC_MODEL, "train")
        verify_allowed_against_synthetic(dev_allowed, SYNTH_DEV_JSON, QC_MODEL, "dev")

        train_parsed_synth = build_synthetic_parsed(
            SYNTH_TRAIN_JSON, PUBTATOR_FILE_TRAIN, SYNTH_GENERATIONS, allowed=train_allowed
        )
        train_df = build_samples(
            PUBTATOR_FILE_TRAIN, parsed=train_parsed_synth, distance_stats=train_distance_stats
        )
        dev_parsed_synth = build_synthetic_parsed(
            SYNTH_DEV_JSON, PUBTATOR_FILE_DEV, SYNTH_GENERATIONS, allowed=dev_allowed
        )
        val_df = build_samples(
            PUBTATOR_FILE_DEV, parsed=dev_parsed_synth, distance_stats=train_distance_stats, verbose=False
        )

    out_dir = (
        f"finetuning/relations-bert_{MODEL.replace('/', '-')}_"
        f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{DATASET_NAME}_{tag}_{F1_AVERAGE}"
    )
    n_train_papers = train_df["pmid"].nunique() if "pmid" in train_df.columns else -1
    print(
        f"RUN_MODE={run_mode} tag={tag}\n"
        f"  train rows={len(train_df)} (papers={n_train_papers}) | "
        f"val rows={len(val_df)} | test rows={len(test_df)}"
    )
    print("  train label dist:\n" + train_df["target_relation"].value_counts().to_string())
    if build_only:
        print("BUILD_ONLY=1 -> stopping before training.")
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    def preprocess(examples: dict) -> dict:
        return tokenizer(examples["prompt"], truncation=True, padding=True, max_length=MAX_LENGTH)

    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True)).map(
        preprocess, batched=True, batch_size=BATCH_SIZE, num_proc=NUM_PROCS
    )
    val_ds = Dataset.from_pandas(val_df.reset_index(drop=True)).map(
        preprocess, batched=True, batch_size=BATCH_SIZE, num_proc=NUM_PROCS
    )

    # Hardware args mirror the notebook: fp16 on CUDA, batch/LR halved on <=8.5 GB cards.
    hw: dict = {}
    if torch.cuda.is_available():
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        div = 2 if total_vram_gb <= 8.5 else 1
        hw = {
            "fp16": True,
            "per_device_train_batch_size": BATCH_SIZE // div,
            "per_device_eval_batch_size": BATCH_SIZE // div,
            "learning_rate": LR / div,
        }
        print(f"CUDA VRAM {total_vram_gb:.1f} GB -> batch {BATCH_SIZE // div}, lr {LR / div}")
    else:
        raise SystemExit("No CUDA device available; refusing to train on CPU.")

    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_micro",
        greater_is_better=True,
        save_total_limit=EPOCHS,
        report_to="none",
        **hw,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=len(RELATION_LABELS), id2label=id2label, label2id=label2id
    ).to("cuda")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    test_ds = Dataset.from_pandas(test_df.reset_index(drop=True)).map(
        preprocess, batched=True, batch_size=BATCH_SIZE, num_proc=NUM_PROCS
    )
    print("=== held-out real BioRED Test (9-class) ===")
    print(trainer.evaluate(test_ds))

    model.save_pretrained(out_dir + "_dump")
    (REPO_ROOT / "finetuning" / f"LAST_RUN_DIR_{run_mode}.txt").write_text(out_dir + "\n")
    print(f"RUN_DIR={out_dir}")


if __name__ == "__main__":
    main()
