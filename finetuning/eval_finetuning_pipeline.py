from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Union

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jinja2
import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from pubtator_parser import parse_pubtator


@dataclass
class PipelineData:
    """Container for all data and configuration produced by prepare_data."""

    train_ds: Dataset
    val_ds: Dataset
    val_df: pd.DataFrame
    tokenizer: AutoTokenizer
    device: torch.device
    id2label: dict[int, str]
    label2id: dict[str, int]


def transform_dataset(df: pd.DataFrame, template: jinja2.Template) -> pd.DataFrame:
    """
    Transform a merged metadata+relations DataFrame into a model-ready DataFrame.

    Expects columns: pmid, abstract, relation_type, id_1, id_2.
    Groups by pmid, renders the Jinja2 prompt template, and assigns label 'true'
    to every row. Returns a DataFrame with columns 'text' and 'label'.
    """
    records = []
    for (_, abstract), group in df.groupby(["pmid", "abstract"], sort=False):
        relations_str = "\n".join(
            f"{row.relation_type}: '{row.id_1}',  '{row.id_2}'" for row in group.itertuples()
        )
        text = template.render(abstract=abstract, relations=relations_str)
        records.append({"text": text, "label": "true"})
    return pd.DataFrame(records, columns=["text", "label"])


def prepare_data(
    pubtator_file: Path,
    model_name: str,
    prompt_template_file: Path,
    test_size: float = 0.1,
    random_state: int = 42,
) -> PipelineData:
    """
    Load and prepare all pipeline inputs from a PubTator file.

    Handles device detection, prompt template loading, dataset transformation,
    label encoding, train/val split, and tokenization.
    Returns a PipelineData instance ready for training or evaluation.
    """
    device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    if device_name == "cpu":
        print("#" * 50)
        print("WARNING: CPU is USED")
        print("#" * 50)
    device = torch.device(device_name)
    print(f"Using device: {device}")

    prompt_cfg: dict = yaml.safe_load(prompt_template_file.read_text(encoding="utf-8"))
    template = jinja2.Template(prompt_cfg["template"])

    label2id: dict[str, int] = {"false": 0, "true": 1}
    id2label: dict[int, str] = {0: "false", 1: "true"}

    metadata_df, _, relations_df = parse_pubtator(pubtator_file)
    merged_df = metadata_df.merge(relations_df, on="pmid")
    df = transform_dataset(merged_df, template)
    df["label"] = df["label"].map(label2id)

    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df["label"])
    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _tokenize(batch: dict) -> dict:
        """Tokenize a batch of texts."""
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)

    train_ds = train_ds.map(_tokenize, batched=True)
    val_ds = val_ds.map(_tokenize, batched=True)

    return PipelineData(
        train_ds=train_ds,
        val_ds=val_ds,
        val_df=val_df,
        tokenizer=tokenizer,
        device=device,
        id2label=id2label,
        label2id=label2id,
    )


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
