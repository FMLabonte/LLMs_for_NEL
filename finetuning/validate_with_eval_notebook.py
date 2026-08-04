"""Cross-check ``reevaluate_biored_metric.py`` against ``eval.ipynb`` itself.

The re-scoring script re-implements the driver around the notebook's scorer: it builds
the test split, runs the model and calls ``compute_biored_f1``.
This script removes the doubt that the re-implementation drifted, by executing the
notebook headlessly on one of our own checkpoints and comparing the F1 the notebook
prints with the value stored in ``results.jsonl`` under ``actual_metrics``.

The notebook is not modified on disk. A copy is patched in memory:

    * the absolute ``/home/flabonte/LLMs_for_NEL`` paths become this repository,
    * the hard-coded checkpoint becomes ``--checkpoint``,
    * cells after the evaluation cell (plotting over all checkpoints) are dropped,
    * ``Trainer(tokenizer=...)`` becomes ``Trainer(processing_class=...)``, because the
      argument was removed in transformers 5.x, which the flake pins. This is a pure
      API rename and does not touch the scoring.

Every patch is printed, so it is visible that the scoring code itself is untouched.
The notebook evaluates ``tokenized_test_3``, the test split with *all* NoRelation
pairs, so the comparison target is ``actual_metrics.full_norelation_test``.

Usage::

    nix develop -c python finetuning/validate_with_eval_notebook.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import nbformat
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parent.parent
FINETUNING_DIR = REPO_ROOT / "finetuning"
NOTEBOOK = FINETUNING_DIR / "eval.ipynb"
RESULTS_FILE = FINETUNING_DIR / "results.jsonl"

FOREIGN_ROOT = "/home/flabonte/LLMs_for_NEL"
DEFAULT_CHECKPOINT = "relations-bert-2026-07-01_18-38-41_BioRed_micro/checkpoint-2610"

# The notebook cell that prints the BioRED-style report; later cells only plot.
# Anchored to the line start so the commented-out example usage does not match.
LAST_CELL_MARKER = re.compile(r"^print_biored_f1_report\(", re.MULTILINE)
TOLERANCE = 1e-9


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


def rename_trainer_tokenizer_kwarg(source: str) -> tuple[str, int]:
    """Rename the ``tokenizer`` keyword to ``processing_class`` inside ``Trainer(...)`` calls.

    Returns the patched source and the number of call sites that were changed.
    Other constructors keep their ``tokenizer`` argument.
    """
    hits = 0
    for match in reversed(list(re.finditer(r"\bTrainer\(", source))):
        start = match.end()
        depth = 1
        position = start
        while position < len(source) and depth:
            depth += {"(": 1, ")": -1}.get(source[position], 0)
            position += 1
        arguments, count = re.subn(r"\btokenizer\s*=", "processing_class=", source[start : position - 1])
        if count:
            hits += 1
            source = source[:start] + arguments + source[position - 1 :]
    return source, hits


def patch_notebook(notebook: nbformat.NotebookNode, checkpoint: Path) -> nbformat.NotebookNode:
    """Rewrite foreign paths, point the notebook at ``checkpoint`` and cut the plotting cells."""
    cells = []
    for cell in notebook.cells:
        cells.append(cell)
        if cell.cell_type == "code" and LAST_CELL_MARKER.search(cell.source):
            break
    else:
        raise RuntimeError(f"{NOTEBOOK} has no cell calling print_biored_f1_report()")

    dropped = len(notebook.cells) - len(cells)
    print(f"patch: keeping {len(cells)} cells, dropping {dropped} trailing cell(s) (plotting)")

    checkpoint_pattern = re.compile(rf'"{re.escape(FOREIGN_ROOT)}/finetuning/[^"]+checkpoint-\d+"')
    for index, cell in enumerate(cells):
        if cell.cell_type != "code":
            continue
        source = cell.source

        patched, hits = checkpoint_pattern.subn(f'"{checkpoint}"', source)
        if hits:
            print(f"patch: cell {index}: {hits} checkpoint path(s) -> {checkpoint}")
        source = patched

        if FOREIGN_ROOT in source:
            count = source.count(FOREIGN_ROOT)
            source = source.replace(FOREIGN_ROOT, str(REPO_ROOT))
            print(f"patch: cell {index}: {count} data path(s) -> {REPO_ROOT}")

        # transformers 5.x dropped Trainer(tokenizer=...); the replacement is a rename.
        # Only Trainer calls are touched: DataCollatorWithPadding still takes ``tokenizer``.
        source, hits = rename_trainer_tokenizer_kwarg(source)
        if hits:
            print(f"patch: cell {index}: {hits} Trainer(tokenizer=) -> Trainer(processing_class=)")

        cell.source = source

    notebook.cells = cells
    return notebook


def extract_report(notebook: nbformat.NotebookNode) -> dict[str, float]:
    """Pull TP/FP/FN and P/R/F1 out of the report the notebook printed."""
    pattern = re.compile(
        r"TP=(\d+)\s+FP=(\d+)\s+FN=(\d+)\s+"
        r"Precision=([\d.]+)\s+Recall=([\d.]+)\s+F1=([\d.]+)"
    )
    for cell in reversed(notebook.cells):
        for output in cell.get("outputs", []):
            text = output.get("text") or output.get("data", {}).get("text/plain", "")
            match = pattern.search(text if isinstance(text, str) else "".join(text))
            if match:
                return {
                    "tp": int(match.group(1)),
                    "fp": int(match.group(2)),
                    "fn": int(match.group(3)),
                    "precision": float(match.group(4)),
                    "recall": float(match.group(5)),
                    "f1": float(match.group(6)),
                }
    raise RuntimeError("The notebook did not print a BioRED F1 report.")


def find_expected(entries: list[dict[str, Any]], checkpoint: str) -> dict[str, Any]:
    """Return the stored all-negatives metrics of ``checkpoint`` from ``results.jsonl``."""
    for entry in entries:
        metrics = entry.get("actual_metrics", {})
        if metrics.get("checkpoint") == checkpoint and "full_norelation_test" in metrics:
            return metrics["full_norelation_test"]
    raise SystemExit(
        f"No entry with actual_metrics.full_norelation_test for {checkpoint!r}. "
        "Run finetuning/reevaluate_biored_metric.py first."
    )


def main() -> int:
    """Execute the notebook on one checkpoint and compare it to the stored metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="checkpoint directory relative to finetuning/ (default: %(default)s)",
    )
    parser.add_argument(
        "--executed-notebook",
        type=Path,
        default=FINETUNING_DIR / "eval_validation_run.ipynb",
        help="where to store the executed notebook copy (default: %(default)s)",
    )
    arguments = parser.parse_args()

    checkpoint_dir = FINETUNING_DIR / arguments.checkpoint
    if not (checkpoint_dir / "config.json").exists():
        raise SystemExit(f"No checkpoint at {checkpoint_dir}")

    expected = find_expected(read_results(RESULTS_FILE), arguments.checkpoint)
    print(f"checkpoint : {arguments.checkpoint}")
    print(f"expected   : F1={expected['biored_f1']:.4f} on {expected['eval_rows']} rows "
          f"(from results.jsonl, actual_metrics.full_norelation_test)\n")

    notebook = patch_notebook(nbformat.read(NOTEBOOK, as_version=4), checkpoint_dir)

    print(f"\nexecuting {NOTEBOOK.name} ...")
    client = NotebookClient(notebook, timeout=7200, kernel_name="python3", resources={"metadata": {"path": str(FINETUNING_DIR)}})
    client.execute()
    nbformat.write(notebook, arguments.executed_notebook)
    print(f"executed notebook written to {arguments.executed_notebook}")

    report = extract_report(notebook)
    print("\n=== comparison ===")
    print(f"{'':<12}{'notebook':>12}{'script':>12}")
    rows = [
        ("TP", report["tp"], expected["tp"]),
        ("FP", report["fp"], expected["fp"]),
        ("FN", report["fn"], expected["fn"]),
        ("precision", report["precision"], expected["biored_precision"]),
        ("recall", report["recall"], expected["biored_recall"]),
        ("F1", report["f1"], expected["biored_f1"]),
    ]
    for name, notebook_value, script_value in rows:
        formatted = f"{notebook_value:>12}{script_value:>12.4f}" if isinstance(notebook_value, float) else f"{notebook_value:>12}{script_value:>12}"
        print(f"{name:<12}{formatted}")

    # The notebook prints four decimals, so the counts must be identical and the
    # rates equal within rounding.
    mismatches = [
        name
        for name, notebook_value, script_value in rows
        if abs(notebook_value - script_value) > (5e-5 if isinstance(notebook_value, float) else TOLERANCE)
    ]
    if mismatches:
        print(f"\nFAILED: {', '.join(mismatches)} differ")
        return 1
    print("\nOK: the notebook reproduces the stored actual_metrics exactly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
