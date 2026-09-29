"""Score one fine-tuned checkpoint on the real BioRED Test split with the BioRED metric.

This is the same scoring path as :mod:`finetuning.reevaluate_biored_metric` (which
rewrites ``results.jsonl`` for every stored run), reduced to a single model so a freshly
trained checkpoint can be placed next to the documented baselines without touching the
registry. The BioRED F1 (pooled over the 8 positive relation types, ignoring rows where
gold and prediction are both ``NoRelation``) and the negative-discrimination score are
reused verbatim from that module.

Two evaluation variants are reported, exactly as in the presentation table:
  * ``matched``   - one distance-matched ``NoRelation`` example per gold relation.
  * ``all_pairs`` - every co-mentioned unrelated pair of the document (no distance cap).

Usage (CUDA lives in the flake)::

    nix develop -c python finetuning/eval_biored_metric_single.py <model_dir> [tokenizer_dir]

``model_dir`` may be a ``checkpoint-*`` directory or a ``<run>_dump`` (weights only);
in the latter case the tokenizer is resolved from the run's newest checkpoint unless a
``tokenizer_dir`` is given explicitly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dataset_preparation.perturbations import BIORED_RELATION_TYPES, NO_RELATION_LABEL  # noqa: E402
from dataset_preparation.prepare_pure_biored import (  # noqa: E402
    build_pure_biored_samples,
    compute_relation_distance_stats,
)
from pubtator_parser import parse_pubtator  # noqa: E402
from finetuning.reevaluate_biored_metric import (  # noqa: E402
    compute_biored_f1,
    compute_negative_discrimination,
    predict,
    resolve_tokenizer_dir,
    tokenize,
)

PUBTATOR_FILE_TRAIN = REPO_ROOT / "Data/BioRED/Train.PubTator"
PUBTATOR_FILE_TEST = REPO_ROOT / "Data/BioRED/Test.PubTator"

RELATION_LABELS = BIORED_RELATION_TYPES + [NO_RELATION_LABEL]
label2id = {name: idx for idx, name in enumerate(RELATION_LABELS)}
MASK_TOKEN = "[MASK]"


def build_eval_frame(
    parsed: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    distance_stats: dict[str, float],
    no_relation_ratio: float | None,
) -> pd.DataFrame:
    """Build the real BioRED Test frame with the notebook's prompt/label columns.

    Mirrors ``build_samples`` in ``finetuning_pubmedbert.ipynb`` (rich frame variant),
    so the prompt fed to the model is byte-identical to the one seen at training time.
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
    frame["label"] = frame["target_relation"].map(label2id)
    return frame


def score_variant(frame: pd.DataFrame, name: str) -> dict[str, float]:
    """Print and return BioRED F1 + negative discrimination for one evaluation frame."""
    biored = compute_biored_f1(
        frame, pred_col="predicted_relation", gold_col="target_relation", negative_label=NO_RELATION_LABEL
    )
    discrimination = compute_negative_discrimination(
        frame, pred_col="predicted_relation", gold_col="target_relation", negative_label=NO_RELATION_LABEL
    )
    overall = biored["overall"]
    print(
        f"  [{name:9s}] rows={len(frame):5d}  "
        f"BioRED F1={overall['f1']:.4f}  P={overall['precision']:.4f}  R={overall['recall']:.4f}  "
        f"(TP={overall['tp']} FP={overall['fp']} FN={overall['fn']})  "
        f"| any-relation F1={discrimination['f1']:.4f}"
    )
    return {"biored": biored, "discrimination": discrimination}


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python eval_biored_metric_single.py <model_dir> [tokenizer_dir]")
        sys.exit(1)

    model_dir = Path(sys.argv[1]).resolve()
    tokenizer_dir = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else resolve_tokenizer_dir(model_dir)
    print(f"model     : {model_dir}")
    print(f"tokenizer : {tokenizer_dir}")

    train_parsed = parse_pubtator(PUBTATOR_FILE_TRAIN)
    train_distance_stats = compute_relation_distance_stats(*train_parsed, verbose=False)
    test_parsed = parse_pubtator(PUBTATOR_FILE_TEST)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    print("\nReal BioRED Test (held out), BioRED-style scoring:")
    for name, ratio in (("matched", 1.0), ("all_pairs", None)):
        frame = build_eval_frame(test_parsed, train_distance_stats, ratio)
        tokenized = tokenize(frame, tokenizer)
        predicted = predict(model_dir, tokenizer_dir, tokenized)
        # keep the gold column that scoring expects (order is preserved by tokenize/predict)
        predicted["target_relation"] = frame["target_relation"].values
        score_variant(predicted, name)


if __name__ == "__main__":
    main()
