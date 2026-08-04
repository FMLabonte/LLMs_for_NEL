"""Score every checkpoint of every run with the BioRED-style metric, epoch by epoch.

``reevaluate_biored_metric.py`` scores one checkpoint per run (the one referenced in
``results.jsonl``). This script walks *all* ``checkpoint-*`` directories of a run, so the
metric can be plotted over training epochs. The scoring itself is imported from
``reevaluate_biored_metric``, not re-implemented, so the curves and the table in the
presentation come from the same code path.

Every checkpoint is scored on both the **Dev** and the **Test** split, so checkpoint
selection on Dev can be compared against the Test curve. The negative-sampling distance
statistics are fitted on **Train** for both splits, exactly as the training notebook does.

Every checkpoint is also scored on one common evaluation set, the pure BioRED one, so the
runs are comparable regardless of what they were trained on. A perturbated-trained run is
additionally scored on the perturbated evaluation set, which is kept as a diagnostic: the
difference between the two says how much of its score comes from its own, differently
sampled distribution. ``train_dataset`` and ``eval_dataset`` therefore differ per row.

Results are written to a tidy CSV (one row per run/epoch/split/eval set/variant) plus a JSON
holding the full payload including per-class scores, so the plots can be varied later
without re-running the models. Already-scored rows are skipped, so the script can be
interrupted and resumed.

The last checkpoint scored for a run is compared against the value stored in
``results.jsonl`` under ``actual_metrics``; divergences are reported at the end.
That cross-check only looks at the Test rows, because ``results.jsonl`` holds Test scores.

Usage (CUDA lives in the flake, not on the system)::

    nix develop -c python finetuning/epoch_curves.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from finetuning.reevaluate_biored_metric import (  # noqa: E402
    FINETUNING_DIR,
    _normalise,
    PUBTATOR_FILE_DEV,
    PUBTATOR_FILE_TEST,
    PUBTATOR_FILE_TRAIN,
    RESULTS_FILE,
    build_test_dataset,
    compute_relation_distance_stats,
    describe_encoder,
    parse_pubtator,
    predict,
    read_results,
    resolve_model_dir,
    resolve_tokenizer_dir,
    score,
    tokenize,
)

CSV_FILE = FINETUNING_DIR / "epoch_curves.csv"
JSON_FILE = FINETUNING_DIR / "epoch_curves.json"

# Both splits are scored per checkpoint. Test is first so that a fresh run reproduces the
# original Test-only sweep before it spends GPU time on Dev.
SPLITS: list[tuple[str, Path]] = [("Test", PUBTATOR_FILE_TEST), ("Dev", PUBTATOR_FILE_DEV)]
# Rows written before the Dev sweep existed carry no split column; they are all Test rows.
DEFAULT_SPLIT = "Test"

# Every run is scored on the pure BioRED evaluation set, so the four runs are comparable.
# A perturbated-trained run is additionally scored on the perturbated evaluation set it was
# trained against; that pair of numbers shows how much of its score is due to its own,
# differently sampled distribution.
COMMON_EVAL_DATASET = "BioRed"
# Perturbated data brings its own negatives, so the all-negatives variant only exists for
# the pure BioRED builder.
EVAL_VARIANTS: dict[str, list[tuple[str, float | None]]] = {
    "BioRed": [("matched_norelation", 1.0), ("full_norelation", None)],
    "BioRedPerturbated": [("matched_norelation", 1.0)],
}

CSV_COLUMNS = [
    "run",
    "encoder",
    "train_dataset",
    "eval_dataset",
    "split",
    "variant",
    "epoch",
    "step",
    "checkpoint",
    "eval_rows",
    "tp",
    "fp",
    "fn",
    "biored_precision",
    "biored_recall",
    "biored_f1",
    "micro_f1_all_classes",
    "macro_f1_all_classes",
    "negative_discrimination_f1",
]


def resolve_run_dir(name: str, search_dir: Path = FINETUNING_DIR) -> Path | None:
    """Find the directory holding the per-epoch checkpoints of a run.

    The dump of a run and the run itself do not always spell their name the same way
    (``relations-bert_bioformers-...`` versus ``relations-bert-bioformers-...``), so the
    lookup normalises ``-`` and ``_`` and requires the directory to contain checkpoints.
    """
    wanted = _normalise(name.removesuffix("_dump"))
    for directory in sorted(search_dir.iterdir()):
        if not directory.is_dir() or _normalise(directory.name) != wanted:
            continue
        if list(directory.glob("checkpoint-*")):
            return directory
    return None


def collect_runs(entries: list[dict[str, Any]]) -> dict[Path, str]:
    """Map every run directory referenced in ``results.jsonl`` to its dataset name.

    Entries point either at a run directory, at its ``_dump`` (the best model of the
    run, saved without checkpoints) or at a single ``checkpoint-*``. All three resolve
    to the same run directory, which is the one holding the per-epoch checkpoints.
    """
    runs: dict[Path, str] = {}
    for entry in entries:
        reference: str = entry.get("model") or entry.get("model_name", "")
        dataset: str = entry.get("dataset", "BioRed")
        model_dir = resolve_model_dir(reference)
        if model_dir is None:
            print(f"  no weights for {reference!r}, skipped")
            continue

        # checkpoint-* -> parent run dir; <run>_dump -> the run dir it was dumped from.
        candidate = model_dir.parent if model_dir.name.startswith("checkpoint-") else model_dir
        run_dir = resolve_run_dir(candidate.name)
        if run_dir is None:
            print(f"  no checkpoint directory found for {candidate.name}, skipped")
            continue
        runs.setdefault(run_dir, dataset)
    return runs


def checkpoints_of(run_dir: Path) -> list[tuple[int, int, Path]]:
    """Return ``(epoch, step, path)`` for every checkpoint of a run, ordered by step."""
    paths = sorted(
        (path for path in run_dir.glob("checkpoint-*") if (path / "config.json").exists()),
        key=lambda path: int(path.name.split("-")[-1]),
    )
    return [(epoch, int(path.name.split("-")[-1]), path) for epoch, path in enumerate(paths, start=1)]


def _migrate(record: dict[str, Any]) -> dict[str, Any]:
    """Bring a stored row or payload up to the current schema, in place.

    Rows written before the Dev sweep have no ``split``; they are all Test rows.
    Rows written before training and evaluation data were separated have a single
    ``dataset`` column, which meant both at once.
    """
    split = record.get("split")
    if split is None or (isinstance(split, float) and pd.isna(split)):
        record["split"] = DEFAULT_SPLIT
    dataset = record.get("dataset")
    if dataset is not None and not (isinstance(dataset, float) and pd.isna(dataset)):
        record.setdefault("train_dataset", dataset)
        record.setdefault("eval_dataset", dataset)
    return record


def load_done() -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[tuple[str, str, str, str]]]:
    """Load previously scored rows so an interrupted sweep can be resumed."""
    rows: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    if CSV_FILE.exists():
        rows = [_migrate(row) for row in pd.read_csv(CSV_FILE).to_dict("records")]
    if JSON_FILE.exists():
        payloads = [_migrate(payload) for payload in json.loads(JSON_FILE.read_text(encoding="utf-8"))]
    done = {
        (str(row["split"]), str(row["eval_dataset"]), str(row["checkpoint"]), str(row["variant"]))
        for row in rows
    }
    return rows, payloads, done


def flush(rows: list[dict[str, Any]], payloads: list[dict[str, Any]]) -> None:
    """Write the CSV and the JSON after every checkpoint, so nothing is lost on a crash."""
    pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(CSV_FILE, index=False)
    JSON_FILE.write_text(json.dumps(payloads, indent=2), encoding="utf-8")


def crosscheck(rows: list[dict[str, Any]], entries: list[dict[str, Any]]) -> list[str]:
    """Compare the swept checkpoints against the values stored in ``results.jsonl``.

    Entries that name a concrete checkpoint must match to the last digit. Entries that
    name a ``_dump`` hold the best-on-Dev model, which is a copy of one of the
    checkpoints, so the stored value has to occur somewhere in that run's curve.
    Only the Test rows are compared, since ``results.jsonl`` stores Test scores, and only
    rows whose evaluation set is the one the stored entry was scored on.
    """
    # results.jsonl holds Test scores only, so the Dev rows take no part in this check.
    rows = [row for row in rows if str(row.get("split", DEFAULT_SPLIT)) == "Test"]
    by_key = {
        (str(row["checkpoint"]), str(row["variant"]), str(row["eval_dataset"])): row for row in rows
    }
    messages: list[str] = []

    for entry in entries:
        metrics = entry.get("actual_metrics", {})
        if "biored_f1" not in metrics:
            continue
        stored_checkpoint = str(metrics["checkpoint"])
        # The stored runs were always scored on the split of their own training data.
        eval_dataset = str(entry.get("dataset", COMMON_EVAL_DATASET))
        variants = [("matched_norelation", metrics["biored_f1"])]
        if "full_norelation_test" in metrics:
            variants.append(("full_norelation", metrics["full_norelation_test"]["biored_f1"]))

        for variant, stored_f1 in variants:
            row = by_key.get((stored_checkpoint, variant, eval_dataset))
            if row is not None:
                delta = abs(float(row["biored_f1"]) - stored_f1)
                verdict = "match" if delta < 1e-9 else f"DIVERGENCE (delta {delta:.6f})"
                messages.append(
                    f"{verdict:<28}{stored_checkpoint} [{variant}]: "
                    f"stored {stored_f1:.4f}, swept {float(row['biored_f1']):.4f}"
                )
                continue

            # A _dump entry: look for the epoch whose score equals the stored one.
            run_name = stored_checkpoint.removesuffix("_dump").split("/")[0]
            curve = [
                candidate
                for candidate in rows
                if str(candidate["variant"]) == variant
                and str(candidate["eval_dataset"]) == eval_dataset
                and str(candidate["run"]).replace("_", "-") == run_name.replace("_", "-")
            ]
            hits = [c for c in curve if abs(float(c["biored_f1"]) - stored_f1) < 1e-9]
            if hits:
                messages.append(
                    f"{'match (dump = epoch ' + str(hits[0]['epoch']) + ')':<28}"
                    f"{stored_checkpoint} [{variant}]: stored {stored_f1:.4f}"
                )
            elif curve:
                closest = min(curve, key=lambda c: abs(float(c["biored_f1"]) - stored_f1))
                messages.append(
                    f"{'NO EPOCH MATCHES':<28}{stored_checkpoint} [{variant}]: "
                    f"stored {stored_f1:.4f}, closest epoch {closest['epoch']} "
                    f"at {float(closest['biored_f1']):.4f}"
                )
    return messages


def main() -> int:
    """Score all checkpoints of all runs and cross-check against ``results.jsonl``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="ignore existing rows and score every checkpoint again",
    )
    arguments = parser.parse_args()

    entries = read_results(RESULTS_FILE)
    runs = collect_runs(entries)
    print(f"{len(runs)} run(s) with checkpoints:")
    for run_dir, train_dataset in runs.items():
        print(f"  {run_dir.name}  ({train_dataset}, {len(checkpoints_of(run_dir))} checkpoints)")

    rows, payloads, done = ([], [], set()) if arguments.rescore else load_done()
    if done:
        print(
            f"\n{len(done)} split/eval-set/checkpoint/variant combination(s) already scored, "
            "skipping those"
        )

    # The distance statistics used for NoRelation sampling are fitted on Train for both
    # evaluation splits, exactly as the training notebook does.
    train_distance_stats = compute_relation_distance_stats(*parse_pubtator(PUBTATOR_FILE_TRAIN), verbose=False)
    eval_frames: dict[tuple[str, str, str], pd.DataFrame] = {}
    tokenized: dict[tuple[str, str, str, str], Any] = {}

    for run_dir, train_dataset in runs.items():
        checkpoints = checkpoints_of(run_dir)
        encoder = describe_encoder(checkpoints[0][2])
        # The common BioRED evaluation set first, then the run's own set where it differs.
        eval_datasets = [COMMON_EVAL_DATASET]
        if train_dataset != COMMON_EVAL_DATASET:
            eval_datasets.append(train_dataset)

        print(f"\n=== {run_dir.name} ({encoder}, trained on {train_dataset}) ===")
        for epoch, step, checkpoint in checkpoints:
            tokenizer_dir = resolve_tokenizer_dir(checkpoint)
            for split, pubtator_file in SPLITS:
                for eval_dataset in eval_datasets:
                    for variant, ratio in EVAL_VARIANTS[eval_dataset]:
                        key = (
                            split,
                            eval_dataset,
                            str(checkpoint.relative_to(FINETUNING_DIR)),
                            variant,
                        )
                        if key in done:
                            continue

                        frame_key = (split, eval_dataset, variant)
                        if frame_key not in eval_frames:
                            eval_frames[frame_key] = build_test_dataset(
                                eval_dataset,
                                train_distance_stats,
                                ratio,
                                pubtator_file=pubtator_file,
                            )
                            print(
                                f"  built {split}/{eval_dataset}/{variant} split: "
                                f"{len(eval_frames[frame_key])} rows"
                            )

                        cache_key = (split, eval_dataset, variant, str(tokenizer_dir))
                        if cache_key not in tokenized:
                            tokenized[cache_key] = tokenize(
                                eval_frames[frame_key], AutoTokenizer.from_pretrained(tokenizer_dir)
                            )

                        frame = predict(checkpoint, tokenizer_dir, tokenized[cache_key])
                        payload = score(
                            frame, checkpoint, eval_dataset, variant, eval_split=f"{split}.PubTator"
                        )
                        payload.update(
                            {
                                "run": run_dir.name,
                                "encoder": encoder,
                                "train_dataset": train_dataset,
                                "split": split,
                                "epoch": epoch,
                                "step": step,
                            }
                        )
                        payloads.append(payload)
                        rows.append(
                            {
                                "run": run_dir.name,
                                "encoder": encoder,
                                "train_dataset": train_dataset,
                                "eval_dataset": eval_dataset,
                                "split": split,
                                "variant": variant,
                                "epoch": epoch,
                                "step": step,
                                "checkpoint": key[2],
                                "eval_rows": payload["eval_rows"],
                                "tp": payload["tp"],
                                "fp": payload["fp"],
                                "fn": payload["fn"],
                                "biored_precision": payload["biored_precision"],
                                "biored_recall": payload["biored_recall"],
                                "biored_f1": payload["biored_f1"],
                                "micro_f1_all_classes": payload["micro_f1_all_classes"],
                                "macro_f1_all_classes": payload["macro_f1_all_classes"],
                                "negative_discrimination_f1": payload["negative_discrimination"]["f1"],
                            }
                        )
                        done.add(key)
                        flush(rows, payloads)
                        print(
                            f"  epoch {epoch:>2} (step {step:>5}) [{split:<4}] "
                            f"[eval {eval_dataset:<18}] [{variant:<18}] "
                            f"BioRED F1 = {payload['biored_f1']:.4f}  "
                            f"P = {payload['biored_precision']:.4f}  R = {payload['biored_recall']:.4f}"
                        )

    print(f"\nWrote {len(rows)} rows to {CSV_FILE} and {JSON_FILE}")

    print("\n=== cross-check against results.jsonl ===")
    messages = crosscheck(rows, entries)
    for message in messages:
        print(f"  {message}")
    diverged = [message for message in messages if "DIVERGENCE" in message or "NO EPOCH" in message]
    if diverged:
        print(f"\n{len(diverged)} divergence(s) found.")
        return 1
    print("\nAll stored values reproduced by the sweep.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
