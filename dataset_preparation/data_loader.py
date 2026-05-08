"""
Phase 1: BioRED data loader.

Wraps the project's pubtator_parser to load BioRED's three splits
(Train / Dev / Test) into (metadata, annotations, relations) DataFrames.

Usage
-----
    from data_loader import load_biored_split, load_all_biored
    meta, anns, rels = load_biored_split("train")

CLI
---
    python data_loader.py
    (prints per-split row counts as a sanity check)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pubtator_parser import parse_pubtator  # noqa: E402

BIORED_DIR = _REPO_ROOT / "Data" / "BioRED"

SPLIT_FILES: dict[str, str] = {
    "train": "Train.PubTator",
    "dev":   "Dev.PubTator",
    "test":  "Test.PubTator",
}


def load_biored_split(
    split: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load (metadata, annotations, relations) DataFrames for one BioRED split.

    Parameters
    ----------
    split : 'train' | 'dev' | 'test'
    """
    if split not in SPLIT_FILES:
        raise ValueError(
            f"Unknown split {split!r}; expected one of {list(SPLIT_FILES)}"
        )
    path = BIORED_DIR / SPLIT_FILES[split]
    if not path.exists():
        raise FileNotFoundError(f"BioRED split file not found: {path}")
    return parse_pubtator(path)


def load_all_biored() -> dict[
    str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
]:
    """Load all 3 BioRED splits as a dict {split_name: (meta, anns, rels)}."""
    return {split: load_biored_split(split) for split in SPLIT_FILES}


def summarize(
    meta: pd.DataFrame,
    anns: pd.DataFrame,
    rels: pd.DataFrame,
) -> dict:
    """One-line stats for sanity-checking a split."""
    return {
        "papers":            len(meta),
        "papers_with_rels":  rels["pmid"].nunique(),
        "entities":          len(anns),
        "relations":         len(rels),
        "entity_types":      sorted(anns["entity_type"].unique().tolist()),
        "relation_types":    sorted(rels["relation_type"].unique().tolist()),
    }


if __name__ == "__main__":
    for split_name in SPLIT_FILES:
        meta, anns, rels = load_biored_split(split_name)
        s = summarize(meta, anns, rels)
        print(
            f"[{split_name:5s}] "
            f"papers={s['papers']:4d} (with rels: {s['papers_with_rels']:4d})  "
            f"entities={s['entities']:5d}  relations={s['relations']:4d}"
        )
        if split_name == "train":
            print(f"          entity types:   {s['entity_types']}")
            print(f"          relation types: {s['relation_types']}")
