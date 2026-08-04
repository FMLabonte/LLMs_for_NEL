"""Score every epoch checkpoint of a run on the real BioRED Test with the BioRED metric.

This is the batch counterpart to :mod:`finetuning.eval_biored_metric_single`: instead of
one model it walks every ``checkpoint-*`` directory of a training run and reports the
same two variants used in the presentation table,

  * ``matched``   - one distance-matched ``NoRelation`` example per gold relation,
  * ``all_pairs`` - every co-mentioned unrelated pair of the document (no distance cap),

so the synthetic-abstract run can be compared epoch-by-epoch against the documented
baselines without selecting a checkpoint on the test set. The scoring code (BioRED F1 and
negative discrimination) and the evaluation-frame construction are imported verbatim from
:mod:`finetuning.eval_biored_metric_single` / :mod:`finetuning.reevaluate_biored_metric`,
so the numbers are directly comparable to the stored ``actual_metrics``.

Usage (CUDA lives in the flake)::

    nix develop -c python finetuning/eval_all_epochs_biored_metric.py <run_dir> [out.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pubtator_parser import parse_pubtator  # noqa: E402
from dataset_preparation.prepare_pure_biored import compute_relation_distance_stats  # noqa: E402
from finetuning.eval_biored_metric_single import (  # noqa: E402
    PUBTATOR_FILE_TEST,
    PUBTATOR_FILE_TRAIN,
    build_eval_frame,
)
from finetuning.reevaluate_biored_metric import (  # noqa: E402
    NO_RELATION_LABEL,
    compute_biored_f1,
    compute_negative_discrimination,
    predict,
    resolve_tokenizer_dir,
    tokenize,
)

EVAL_VARIANTS: tuple[tuple[str, float | None], ...] = (("matched", 1.0), ("all_pairs", None))


def find_checkpoints(run_dir: Path) -> list[Path]:
    """Return all ``checkpoint-*`` directories in ``run_dir`` sorted by global step."""
    return sorted(
        (path for path in run_dir.glob("checkpoint-*") if path.is_dir()),
        key=lambda path: int(path.name.split("-")[-1]),
    )


def score_checkpoint(
    checkpoint_dir: Path,
    eval_frames: dict[str, "object"],
) -> dict[str, dict[str, float]]:
    """Score one checkpoint on every pre-built evaluation frame.

    Args:
        checkpoint_dir: A ``checkpoint-*`` directory carrying model weights and tokenizer.
        eval_frames: Mapping ``variant -> (frame, tokenized_dataset)`` reused across
            checkpoints so the test frame is tokenized only once per variant.

    Returns:
        Mapping ``variant -> {biored_f1, precision, recall, tp, fp, fn, any_relation_f1}``.
    """
    tokenizer_dir = resolve_tokenizer_dir(checkpoint_dir)
    results: dict[str, dict[str, float]] = {}
    for name, (frame, tokenized) in eval_frames.items():
        predicted = predict(checkpoint_dir, tokenizer_dir, tokenized)
        predicted["target_relation"] = frame["target_relation"].values
        biored = compute_biored_f1(
            predicted, pred_col="predicted_relation", gold_col="target_relation", negative_label=NO_RELATION_LABEL
        )["overall"]
        discrimination = compute_negative_discrimination(
            predicted, pred_col="predicted_relation", gold_col="target_relation", negative_label=NO_RELATION_LABEL
        )
        results[name] = {
            "rows": len(frame),
            "biored_f1": biored["f1"],
            "biored_precision": biored["precision"],
            "biored_recall": biored["recall"],
            "tp": biored["tp"],
            "fp": biored["fp"],
            "fn": biored["fn"],
            "any_relation_f1": discrimination["f1"],
        }
    return results


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python eval_all_epochs_biored_metric.py <run_dir> [out.json]")
        sys.exit(1)

    run_dir = Path(sys.argv[1]).resolve()
    out_path = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else run_dir / "epoch_biored_metric.json"

    checkpoints = find_checkpoints(run_dir)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint-* directories found under {run_dir}")

    train_parsed = parse_pubtator(PUBTATOR_FILE_TRAIN)
    train_distance_stats = compute_relation_distance_stats(*train_parsed, verbose=False)
    test_parsed = parse_pubtator(PUBTATOR_FILE_TEST)

    # Build and tokenize each evaluation frame once; tokenizer is shared across checkpoints
    # of a run (same base encoder), so tokenization is identical for all of them.
    reference_tokenizer = AutoTokenizer.from_pretrained(resolve_tokenizer_dir(checkpoints[-1]))
    eval_frames: dict[str, object] = {}
    for name, ratio in EVAL_VARIANTS:
        frame = build_eval_frame(test_parsed, train_distance_stats, ratio)
        eval_frames[name] = (frame, tokenize(frame, reference_tokenizer))

    print(f"run       : {run_dir.name}")
    print(f"checkpoints: {[c.name for c in checkpoints]}")
    print("\nReal BioRED Test (held out), BioRED-style scoring per epoch:\n")

    all_results: dict[str, dict[str, dict[str, float]]] = {}
    for checkpoint in checkpoints:
        scores = score_checkpoint(checkpoint, eval_frames)
        all_results[checkpoint.name] = scores
        matched = scores["matched"]
        all_pairs = scores["all_pairs"]
        print(
            f"  {checkpoint.name:16s} | "
            f"matched  F1={matched['biored_f1']:.4f} P={matched['biored_precision']:.4f} R={matched['biored_recall']:.4f} "
            f"|| all_pairs F1={all_pairs['biored_f1']:.4f} P={all_pairs['biored_precision']:.4f} R={all_pairs['biored_recall']:.4f}"
        )

    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nWrote per-epoch scores to {out_path}")


if __name__ == "__main__":
    main()
