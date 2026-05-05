from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jinja2
import pandas as pd
import torch
import yaml
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from code.perturbations import build_training_samples, samples_to_rels_like_df
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

    meta_df, anns_df, rels_df = parse_pubtator(pubtator_file)

    samples = build_training_samples(meta_df, anns_df, rels_df)

    rels_df = samples_to_rels_like_df(samples)

    merged_df = meta_df.merge(rels_df, on="pmid")
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
