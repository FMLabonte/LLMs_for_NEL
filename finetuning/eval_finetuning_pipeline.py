from __future__ import annotations

import sys
from pathlib import Path
from typing import Union

from finetuning.commons import prepare_data

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments


def evaluate_finetuning(
    val_ds: Dataset,
    val_labels: list[int],
    base_model_name: str,
    finetuned_model_path: Union[str, Path],
    tokenizer: AutoTokenizer,
    device: torch.device,
    id2label: dict[int, str],
    label2id: dict[str, int],
) -> None:
    """
    Compare the base model against the fine-tuned model on the validation set.

    Prints a summary table with accuracy, precision, recall, F1 (weighted)
    and Matthews Correlation Coefficient, followed by per-class classification
    reports for both models.
    """

    def _get_predictions(model_path_or_name: Union[str, Path]) -> np.ndarray:
        """Load a model and return argmax predictions on val_ds."""
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path_or_name),
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        ).to(device)
        model.eval()

        eval_args = TrainingArguments(
            output_dir="./eval_tmp",
            per_device_eval_batch_size=32,
            fp16=False,
            dataloader_num_workers=0,
            report_to="none",
        )

        trainer = Trainer(model=model, args=eval_args)
        output = trainer.predict(val_ds)
        return np.argmax(output.predictions, axis=-1)

    def _compute_metrics(preds: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        """Compute scalar evaluation metrics for a set of predictions."""
        return {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, average="weighted", zero_division=0),
            "recall": recall_score(labels, preds, average="weighted", zero_division=0),
            "f1": f1_score(labels, preds, average="weighted", zero_division=0),
            "mcc": matthews_corrcoef(labels, preds),
        }

    labels = np.array(val_labels)
    target_names = [id2label[i] for i in sorted(id2label)]

    print("\nRunning inference on base model …")
    base_preds = _get_predictions(base_model_name)

    print("Running inference on fine-tuned model …")
    ft_preds = _get_predictions(finetuned_model_path)

    base_metrics = _compute_metrics(base_preds, labels)
    ft_metrics = _compute_metrics(ft_preds, labels)

    col_w = 14
    metric_names = ["accuracy", "precision", "recall", "f1", "mcc"]
    sep = "=" * (14 + col_w * 3 + 6)

    print()
    print(sep)
    print("  EVALUATION SUMMARY: Base vs. Fine-tuned Model")
    print(sep)
    print(f"{'Metric':<14} {'Base Model':>{col_w}} {'Fine-tuned':>{col_w}} {'Δ (ft − base)':>{col_w}}")
    print("-" * (14 + col_w * 3 + 6))
    for m in metric_names:
        base_val = base_metrics[m]
        ft_val = ft_metrics[m]
        delta = ft_val - base_val
        print(f"{m:<14} {base_val:>{col_w}.4f} {ft_val:>{col_w}.4f} {delta:>+{col_w}.4f}")
    print(sep)

    all_label_ids = sorted(id2label.keys())

    print("\n--- Classification Report: Base Model ---")
    print(classification_report(labels, base_preds, labels=all_label_ids, target_names=target_names, zero_division=0))

    print("--- Classification Report: Fine-tuned Model ---")
    print(classification_report(labels, ft_preds, labels=all_label_ids, target_names=target_names, zero_division=0))


if __name__ == "__main__":
    _MODEL_NAME = "bioformers/bioformer-8L"
    _PUBTATOR_FILE = Path("/Users/chris/git/LLMs_for_NEL/Data/BioRED/Dev.PubTator")
    _FINETUNED_PATH = Path(__file__).resolve().parent.parent / "bioformer-finetuned-final"
    _PROMPT_TEMPLATE_FILE = Path(__file__).parent / "prompting" / "bioformer-prompting.yaml"

    data = prepare_data(_PUBTATOR_FILE, _MODEL_NAME, _PROMPT_TEMPLATE_FILE)
    evaluate_finetuning(
        val_ds=data.val_ds,
        val_labels=data.val_df["label"].tolist(),
        base_model_name=_MODEL_NAME,
        finetuned_model_path=_FINETUNED_PATH,
        tokenizer=data.tokenizer,
        device=data.device,
        id2label=data.id2label,
        label2id=data.label2id,
    )
