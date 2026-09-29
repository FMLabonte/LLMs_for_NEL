"""Re-score every run listed in ``results.jsonl`` with the BioRED-style metric.

The metrics stored in ``results.jsonl`` were produced with a plain micro-F1 over all
nine classes, which also rewards correctly predicted ``NoRelation`` pairs.
The official BioRED scoring ignores those rows entirely, so the stored numbers are
inflated and not comparable to published results.

This script loads every model referenced in ``results.jsonl``, re-runs it on the test
split of the dataset it was trained on, and writes the correct scores back into the
same entry under the new key ``actual_metrics``.
Nothing that is already in the file is modified.

Usage (CUDA lives in the flake, not on the system)::

    nix develop -c python finetuning/reevaluate_biored_metric.py

Provenance of the code below is marked with banner comments:
``build_samples`` is copied from ``finetuning/finetuning_pubmedbert.ipynb``,
``compute_biored_f1`` and ``compute_negative_discrimination`` are copied from
``finetuning/eval.ipynb``.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, Optional

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
)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dataset_preparation.perturbations import (  # noqa: E402
    BIORED_RELATION_TYPES,
    NO_RELATION_LABEL,
    build_training_samples,
    samples_to_rels_like_df,
)
from dataset_preparation.prepare_pure_biored import (  # noqa: E402
    build_pure_biored_samples,
    compute_relation_distance_stats,
)
from pubtator_parser import parse_pubtator  # noqa: E402

FINETUNING_DIR = REPO_ROOT / "finetuning"
RESULTS_FILE = FINETUNING_DIR / "results.jsonl"
PUBTATOR_FILE_TRAIN = REPO_ROOT / "Data/BioRED/Train.PubTator"
PUBTATOR_FILE_DEV = REPO_ROOT / "Data/BioRED/Dev.PubTator"
PUBTATOR_FILE_TEST = REPO_ROOT / "Data/BioRED/Test.PubTator"

RELATION_LABELS: list[str] = BIORED_RELATION_TYPES + [NO_RELATION_LABEL]
label2id: dict[str, int] = {name: idx for idx, name in enumerate(RELATION_LABELS)}
MASK_TOKEN = "[MASK]"

BATCH_SIZE = 32
GOLD_COLUMN = "target_relation"


# ---------------------------------------------------------------------------
# COPIED from finetuning/finetuning_pubmedbert.ipynb (data-building cell).
# Only the default for ``dataset_name`` was made explicit; the prompt format and
# the label mapping are unchanged so the test set matches the one used at training.
# ---------------------------------------------------------------------------
def build_samples(
    pubtator_file: Path,
    dataset_name: Literal["BioRed", "BioRedPerturbated"],
    label_map: dict[str, int] = label2id,
    mask_token: str = MASK_TOKEN,
    no_relation_label: str = NO_RELATION_LABEL,
    parsed: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None,
    distance_stats: dict[str, float] | None = None,
    no_relation_ratio: float | None = 1.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Parse a PubTator file and build the relation-classification samples DataFrame.

    Args:
        pubtator_file: Path to the PubTator file to parse.
        dataset_name: Which sample-building strategy to use. ``"BioRed"`` uses gold
            relations plus distance-matched NoRelation examples, ``"BioRedPerturbated"``
            uses the perturbated training samples.
        label_map: Mapping from relation label name to integer class id.
        mask_token: Token used as the relation placeholder in the prompt.
        no_relation_label: Label assigned to ``false_positive`` (unrelated) pairs.
        parsed: Optional pre-parsed ``(meta, anns, rels)`` tuple to avoid re-reading the file.
        distance_stats: Train-fit distance thresholds for NoRelation sampling on dev/test.
        no_relation_ratio: NoRelation examples per gold relation; ``None`` keeps all
            co-mentioned unrelated pairs (added here, not part of the copied cell).
        verbose: Whether to print NoRelation sampling diagnostics.

    Returns:
        A DataFrame with ``prompt``, ``target_relation`` and ``label`` columns,
        restricted to ``gold`` and ``false_positive`` perturbations.
    """
    if parsed is not None:
        meta_df, anns_df, rels_df = parsed
    else:
        meta_df, anns_df, rels_df = parse_pubtator(pubtator_file)

    if dataset_name == "BioRedPerturbated":
        samples = build_training_samples(meta_df, anns_df, rels_df)

    elif dataset_name == "BioRed":
        # Original BioRED: gold relations + distance-matched NoRelation examples.
        samples = build_pure_biored_samples(
            meta_df,
            anns_df,
            rels_df,
            distance_stats=distance_stats,
            no_relation_ratio=no_relation_ratio,
            verbose=verbose,
        )

    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name!r}")

    samples = samples_to_rels_like_df(samples)

    # Gold relations (8 types) + unrelated entity pairs (NoRelation).
    samples = samples[samples["perturbation"].isin(["gold", "false_positive"])].copy()

    samples["prompt"] = samples.apply(
        lambda row: (
            f"Relation: {row['entity_a_text']} -> {mask_token} -> {row['entity_b_text']}\n"
            f"Context: {row['abstract']}"
        ),
        axis=1,
    )
    samples["target_relation"] = np.where(
        samples["perturbation"] == "false_positive",
        no_relation_label,
        samples["relation_type"],
    )
    samples["label"] = samples["target_relation"].map(label_map)
    return samples


# ---------------------------------------------------------------------------
# COPIED from finetuning/eval.ipynb (the "Eval code" cell), unchanged apart from
# type hints and the removal of the plotting helper.
#
# The scoring logic mirrors run_biored_eval.py's `eval()` function:
#     - A row where gold == negative_label AND pred == negative_label contributes NOTHING
#       (no TP, no FP, no FN) - correctly predicting "no relation" is not rewarded.
#     - gold != negative_label, pred == gold           -> TP
#     - gold != negative_label, pred != gold           -> FN and, if pred is a positive
#                                                        class, FP as well
#     - gold == negative_label, pred != negative_label -> FP (hallucinated a relation)
# ---------------------------------------------------------------------------
def compute_biored_f1(
    df: pd.DataFrame,
    pred_col: str = "predicted_relation",
    gold_col: str = "relation_type",
    negative_label: str = "no_relation",
) -> dict[str, Any]:
    """Pooled and per-class BioRED/BioCreative-VIII-style precision, recall and F1.

    Assumes one row per candidate relation pair, so each row is scored independently.
    ``negative_label`` can never count as a true positive.
    """
    preds = df[pred_col].tolist()
    golds = df[gold_col].tolist()

    tp_count = 0
    fp_count = 0
    fn_count = 0

    typed_tp: dict[str, int] = defaultdict(int)
    typed_fp: dict[str, int] = defaultdict(int)
    typed_fn: dict[str, int] = defaultdict(int)
    typed_support: dict[str, int] = defaultdict(int)

    for pred, gold in zip(preds, golds):
        gold_is_negative = gold == negative_label
        pred_is_negative = pred == negative_label

        if gold_is_negative and pred_is_negative:
            # correctly predicted "no relation" -> not scored at all
            continue

        if not gold_is_negative:
            typed_support[gold] += 1
            if pred == gold:
                tp_count += 1
                typed_tp[gold] += 1
            else:
                # missed or mistyped the true relation
                fn_count += 1
                typed_fn[gold] += 1

        if not pred_is_negative and pred != gold:
            # predicted a positive relation that doesn't match gold
            fp_count += 1
            typed_fp[pred] += 1

    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    per_class: dict[str, dict[str, float]] = {}
    all_labels = set(typed_tp) | set(typed_fp) | set(typed_fn) | set(typed_support)
    for label in sorted(all_labels, key=str):
        tp = typed_tp[label]
        fp = typed_fp[label]
        fn = typed_fn[label]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        per_class[label] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": p,
            "recall": r,
            "f1": f,
            "support": typed_support[label],
        }

    return {
        "overall": {
            "tp": tp_count,
            "fp": fp_count,
            "fn": fn_count,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "per_class": per_class,
    }


def compute_negative_discrimination(
    df: pd.DataFrame,
    pred_col: str = "predicted_relation",
    gold_col: str = "relation_type",
    negative_label: str = "no_relation",
) -> dict[str, float]:
    """Binary "is there a relation at all" score, ignoring wrong-type errors.

    Every row counts here, unlike :func:`compute_biored_f1`, which skips true negatives.
    """
    preds = df[pred_col].tolist()
    golds = df[gold_col].tolist()

    tp = fp = fn = tn = 0

    for pred, gold in zip(preds, golds):
        gold_is_positive = gold != negative_label
        pred_is_positive = pred != negative_label

        if gold_is_positive and pred_is_positive:
            tp += 1
        elif gold_is_positive and not pred_is_positive:
            fn += 1
        elif not gold_is_positive and pred_is_positive:
            fp += 1
        else:
            tn += 1

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ---------------------------------------------------------------------------
# Own code from here on.
# ---------------------------------------------------------------------------
def read_results(results_file: Path) -> list[dict[str, Any]]:
    """Read ``results.jsonl``, which holds pretty-printed (multi-line) JSON objects."""
    text = results_file.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    entries: list[dict[str, Any]] = []
    position = 0
    while position < len(text):
        while position < len(text) and text[position].isspace():
            position += 1
        if position >= len(text):
            break
        entry, position = decoder.raw_decode(text, position)
        entries.append(entry)
    return entries


def write_results(results_file: Path, entries: list[dict[str, Any]]) -> None:
    """Write the entries back in the original format: one indented object per block."""
    results_file.write_text(
        "".join(json.dumps(entry, indent=4) + "\n" for entry in entries), encoding="utf-8"
    )


def _normalise(name: str) -> str:
    """Lower-case a directory name and treat ``-`` and ``_`` as the same character."""
    return re.sub(r"[-_]", "-", name.lower())


def resolve_model_dir(reference: str, search_dir: Path = FINETUNING_DIR) -> Path | None:
    """Resolve a ``results.jsonl`` model reference to a directory holding model weights.

    Entries reference either a concrete ``checkpoint-*`` directory or a run directory.
    A run directory contains no weights itself: the best model of that run was dumped
    to ``<run>_dump`` by the training notebook, so that dump is used instead.
    Directory names in the file are not byte-exact (``-`` vs ``_`` after ``relations-bert``),
    hence the normalised fallback search.
    """
    reference = reference.strip().strip("/")
    candidates = [search_dir / reference, search_dir / f"{reference}_dump"]
    for candidate in candidates:
        if (candidate / "config.json").exists():
            return candidate

    wanted = [_normalise(part) for part in Path(reference).parts]
    for directory in sorted(search_dir.iterdir()):
        if not directory.is_dir():
            continue
        for suffix in ("", "_dump"):
            if _normalise(directory.name) != _normalise(f"{wanted[0]}{suffix}"):
                continue
            resolved = directory.joinpath(*Path(reference).parts[1:])
            if (resolved / "config.json").exists():
                return resolved
    return None


def resolve_tokenizer_dir(model_dir: Path) -> Path:
    """Return a directory that carries a tokenizer for ``model_dir``.

    ``_dump`` directories were written with ``model.save_pretrained`` only, so the
    tokenizer is taken from the newest checkpoint of the corresponding run.
    """
    if (model_dir / "tokenizer_config.json").exists():
        return model_dir

    run_dir_name = model_dir.name.removesuffix("_dump")
    for directory in sorted(model_dir.parent.iterdir()):
        if not directory.is_dir() or _normalise(directory.name) != _normalise(run_dir_name):
            continue
        checkpoints = sorted(
            (path for path in directory.glob("checkpoint-*") if (path / "tokenizer_config.json").exists()),
            key=lambda path: int(path.name.split("-")[-1]),
        )
        if checkpoints:
            return checkpoints[-1]
    raise FileNotFoundError(f"No tokenizer found for {model_dir}")


def describe_encoder(model_dir: Path) -> str:
    """Identify the base encoder of a checkpoint from its architecture."""
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    layers = config.get("num_hidden_layers")
    hidden = config.get("hidden_size")
    if (layers, hidden) == (12, 768):
        return "NeuML/pubmedbert-base-embeddings"
    if (layers, hidden) == (8, 512):
        return "bioformers/bioformer-8L"
    return f"unknown ({layers} layers, hidden {hidden})"


def build_test_dataset(
    dataset_name: str,
    distance_stats: dict[str, float],
    no_relation_ratio: float | None,
    pubtator_file: Path = PUBTATOR_FILE_TEST,
) -> pd.DataFrame:
    """Build an evaluation split for one dataset variant, exactly as the training notebook does.

    ``pubtator_file`` selects the split (Test by default, Dev for the selection curves);
    ``distance_stats`` must stay the Train-fit statistics in either case.
    """
    return build_samples(
        pubtator_file,
        dataset_name=dataset_name,
        distance_stats=distance_stats,
        no_relation_ratio=no_relation_ratio,
        verbose=False,
    )


def tokenize(test_df: pd.DataFrame, tokenizer: AutoTokenizer) -> Dataset:
    """Tokenize the prompt column with the checkpoint's own tokenizer."""
    dataset = Dataset.from_pandas(test_df.reset_index(drop=True))
    return dataset.map(
        lambda batch: tokenizer(batch["prompt"], truncation=True, padding=True, max_length=512),
        batched=True,
        batch_size=BATCH_SIZE,
    )


def predict(model_dir: Path, tokenizer_dir: Path, dataset: Dataset) -> pd.DataFrame:
    """Run a checkpoint over a tokenized dataset and attach the predicted label.

    Adapted from ``predict_with_model`` in ``finetuning/eval.ipynb``.
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    if torch.cuda.is_available():
        model = model.to("cuda")

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="/tmp/reevaluate-biored",
            per_device_eval_batch_size=BATCH_SIZE,
            report_to=[],
        ),
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    logits = trainer.predict(dataset).predictions
    predicted_ids = np.argmax(logits, axis=1)
    id2label: dict[int, str] = {int(key): value for key, value in model.config.id2label.items()}

    frame = dataset.to_pandas()
    frame["predicted_relation"] = [id2label[int(index)] for index in predicted_ids]

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return frame


def score(
    frame: pd.DataFrame,
    model_dir: Path,
    dataset_name: str,
    variant: str,
    eval_split: str = "Test.PubTator",
) -> dict[str, Any]:
    """Assemble the ``actual_metrics`` payload for one checkpoint on one evaluation variant.

    ``eval_split`` only names the split the frame came from; it does not change the scoring.
    """
    biored = compute_biored_f1(
        frame,
        pred_col="predicted_relation",
        gold_col=GOLD_COLUMN,
        negative_label=NO_RELATION_LABEL,
    )
    discrimination = compute_negative_discrimination(
        frame,
        pred_col="predicted_relation",
        gold_col=GOLD_COLUMN,
        negative_label=NO_RELATION_LABEL,
    )
    overall = biored["overall"]
    return {
        "metric": "biored_f1",
        "description": (
            "Pooled over the 8 positive relation types; rows where gold and prediction "
            "are both NoRelation are not scored at all."
        ),
        "checkpoint": str(model_dir.relative_to(FINETUNING_DIR)),
        "base_model": describe_encoder(model_dir),
        "eval_split": eval_split,
        "eval_dataset": dataset_name,
        "eval_variant": variant,
        "eval_rows": int(len(frame)),
        "biored_precision": overall["precision"],
        "biored_recall": overall["recall"],
        "biored_f1": overall["f1"],
        "tp": overall["tp"],
        "fp": overall["fp"],
        "fn": overall["fn"],
        "micro_f1_all_classes": float(
            f1_score(frame[GOLD_COLUMN], frame["predicted_relation"], average="micro", zero_division=0)
        ),
        "macro_f1_all_classes": float(
            f1_score(frame[GOLD_COLUMN], frame["predicted_relation"], average="macro", zero_division=0)
        ),
        "negative_discrimination": discrimination,
        "per_class": biored["per_class"],
    }


def main() -> None:
    """Re-score every entry of ``results.jsonl`` and add the ``actual_metrics`` key."""
    entries = read_results(RESULTS_FILE)
    print(f"Read {len(entries)} entries from {RESULTS_FILE}")

    train_parsed = parse_pubtator(PUBTATOR_FILE_TRAIN)
    train_distance_stats = compute_relation_distance_stats(*train_parsed, verbose=False)

    test_frames: dict[tuple[str, str], pd.DataFrame] = {}
    tokenized: dict[tuple[str, str, str], Dataset] = {}

    for index, entry in enumerate(entries):
        reference: str = entry.get("model") or entry.get("model_name", "")
        dataset_name: str = entry.get("dataset", "BioRed")
        print(f"\n[{index + 1}/{len(entries)}] {reference}  ({dataset_name})")

        model_dir = resolve_model_dir(reference)
        if model_dir is None:
            print(f"    SKIPPED: no weights found for {reference!r}")
            entry["actual_metrics"] = {"error": f"no weights found for {reference!r}"}
            continue

        tokenizer_dir = resolve_tokenizer_dir(model_dir)
        print(f"    weights   : {model_dir.relative_to(FINETUNING_DIR)}")
        print(f"    tokenizer : {tokenizer_dir.relative_to(FINETUNING_DIR)}")
        print(f"    encoder   : {describe_encoder(model_dir)}")

        # The perturbated builder creates its own negatives, so the all-NoRelation
        # variant only exists for the original BioRED test split.
        variants: list[tuple[str, float | None]] = [("matched_norelation", 1.0)]
        if dataset_name == "BioRed":
            variants.append(("full_norelation", None))

        payloads: dict[str, dict[str, Any]] = {}
        for variant, ratio in variants:
            frame_key = (dataset_name, variant)
            if frame_key not in test_frames:
                test_frames[frame_key] = build_test_dataset(dataset_name, train_distance_stats, ratio)
                print(f"    built {dataset_name}/{variant} test split: {len(test_frames[frame_key])} rows")

            cache_key = (dataset_name, variant, str(tokenizer_dir))
            if cache_key not in tokenized:
                tokenized[cache_key] = tokenize(
                    test_frames[frame_key], AutoTokenizer.from_pretrained(tokenizer_dir)
                )

            frame = predict(model_dir, tokenizer_dir, tokenized[cache_key])
            payloads[variant] = score(frame, model_dir, dataset_name, variant)

        actual_metrics = payloads["matched_norelation"]
        if "full_norelation" in payloads:
            full = payloads["full_norelation"]
            actual_metrics["full_norelation_test"] = {
                key: value
                for key, value in full.items()
                if key not in {"metric", "description", "checkpoint", "base_model", "eval_dataset"}
            }
        entry["actual_metrics"] = actual_metrics

        old_f1 = entry.get("metrics", {}).get("f1_micro")
        comparison = f" vs. stored micro-F1 = {old_f1:.4f}" if old_f1 is not None else ""
        print(
            f"    BioRED-style F1 = {actual_metrics['biored_f1']:.4f} "
            f"(P={actual_metrics['biored_precision']:.4f}, R={actual_metrics['biored_recall']:.4f})"
            f"{comparison}"
        )
        if "full_norelation_test" in actual_metrics:
            full = actual_metrics["full_norelation_test"]
            print(
                f"    all NoRelation pairs ({full['eval_rows']} rows): "
                f"F1 = {full['biored_f1']:.4f} "
                f"(P={full['biored_precision']:.4f}, R={full['biored_recall']:.4f})"
            )

    backup = RESULTS_FILE.with_suffix(".jsonl.bak")
    if not backup.exists():
        shutil.copy2(RESULTS_FILE, backup)
        print(f"\nBackup written to {backup}")
    write_results(RESULTS_FILE, entries)
    print(f"Updated {RESULTS_FILE} with 'actual_metrics'")

    print("\n=== Summary ===")
    print(
        f"{'checkpoint':<70}{'dataset':<20}{'old micro-F1':>13}"
        f"{'BioRED F1':>11}{'BioRED F1 (all neg.)':>22}"
    )
    for entry in entries:
        metrics = entry.get("actual_metrics", {})
        if "biored_f1" not in metrics:
            continue
        full = metrics.get("full_norelation_test")
        full_f1 = f"{full['biored_f1']:.4f}" if full else "-"
        print(
            f"{metrics['checkpoint']:<70}{metrics['eval_dataset']:<20}"
            f"{entry.get('metrics', {}).get('f1_micro', float('nan')):>13.4f}"
            f"{metrics['biored_f1']:>11.4f}{full_f1:>22}"
        )


if __name__ == "__main__":
    main()
