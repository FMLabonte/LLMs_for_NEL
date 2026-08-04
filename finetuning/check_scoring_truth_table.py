"""Verify the scoring truth table used in the presentation against ``eval.ipynb``.

The slide "The evaluation pitfall: what actually counts" claims, per (gold, prediction)
combination, what the BioRED-style scorer books and how plain 9-class micro-F1 judges
the same row. This script checks that claim against the real scorer instead of against
a reading of it: ``compute_biored_f1`` is extracted from ``eval.ipynb`` and executed
verbatim, never re-implemented.

The check is exhaustive. Every one of the 9x9 (gold, prediction) label combinations is
scored on its own, mapped to the slide row that claims to cover it, and compared. That
verifies two things at once: each row books what the slide says, and the five rows
leave no case uncovered.

No GPU and no model are involved, so this runs in seconds:

    nix develop -c python finetuning/check_scoring_truth_table.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from sklearn.metrics import f1_score

NOTEBOOK = Path(__file__).resolve().parent / "eval.ipynb"
NEGATIVE = "NoRelation"
POSITIVE_LABELS = [
    "Association",
    "Positive_Correlation",
    "Negative_Correlation",
    "Bind",
    "Cotreatment",
    "Comparison",
    "Drug_Interaction",
    "Conversion",
]
ALL_LABELS = POSITIVE_LABELS + [NEGATIVE]

# The five rows of the slide table: gold pattern, prediction pattern, what the BioRED
# scorer is claimed to book, and how our 9-class micro-F1 is claimed to judge the row.
SLIDE_ROWS: list[tuple[str, str, set[str], str]] = [
    ("relation", "same relation", {"TP"}, "correct"),
    ("relation", "other relation", {"FN", "FP"}, "wrong"),
    ("relation", NEGATIVE, {"FN"}, "wrong"),
    (NEGATIVE, "relation", {"FP"}, "wrong"),
    (NEGATIVE, NEGATIVE, set(), "correct"),
]


def load_scorer(notebook_file: Path = NOTEBOOK) -> Callable[..., dict[str, Any]]:
    """Exec the ``eval.ipynb`` cell that defines ``compute_biored_f1`` and return it."""
    notebook = json.loads(notebook_file.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if "def compute_biored_f1" in source:
            namespace: dict[str, Any] = {}
            exec(compile(source, str(notebook_file), "exec"), namespace)
            return namespace["compute_biored_f1"]
    raise RuntimeError(f"compute_biored_f1 not found in {notebook_file}")


def slide_row_for(gold: str, prediction: str) -> int:
    """Index of the slide row that claims to cover this (gold, prediction) pair."""
    if gold != NEGATIVE and prediction == gold:
        return 0
    if gold != NEGATIVE and prediction != NEGATIVE:
        return 1
    if gold != NEGATIVE:
        return 2
    if prediction != NEGATIVE:
        return 3
    return 4


def format_bookings(bookings: set[frozenset[str]]) -> str:
    """Render the set of observed TP/FP/FN bookings of one slide row."""
    return " / ".join(sorted("+".join(sorted(booking)) or "ignored" for booking in bookings))


def main() -> int:
    """Score all label combinations and compare them to the slide table."""
    compute_biored_f1 = load_scorer()

    bookings_per_row: dict[int, set[frozenset[str]]] = {index: set() for index in range(len(SLIDE_ROWS))}
    micro_per_row: dict[int, set[str]] = {index: set() for index in range(len(SLIDE_ROWS))}
    combinations = 0

    for gold in ALL_LABELS:
        for prediction in ALL_LABELS:
            combinations += 1
            frame = pd.DataFrame([{"relation_type": gold, "predicted_relation": prediction}])
            overall = compute_biored_f1(frame, negative_label=NEGATIVE)["overall"]
            booked = frozenset(name.upper() for name in ("tp", "fp", "fn") if overall[name])
            # Single-label micro-F1 is 1.0 exactly when the row is classified correctly.
            micro = f1_score([gold], [prediction], average="micro", zero_division=0)

            row = slide_row_for(gold, prediction)
            bookings_per_row[row].add(booked)
            micro_per_row[row].add("correct" if micro == 1.0 else "wrong")

    print(f"exhaustive over {combinations} (gold, prediction) combinations of {len(ALL_LABELS)} labels\n")
    print(
        f"{'gold':<12}{'prediction':<16}{'slide: scorer':<16}{'observed':<16}"
        f"{'slide: micro':<14}{'observed':<12}{'ok'}"
    )

    all_ok = True
    for index, (gold_pattern, prediction_pattern, claimed, claimed_micro) in enumerate(SLIDE_ROWS):
        bookings = bookings_per_row[index]
        micros = micro_per_row[index]
        # The row only holds if every combination it covers books the same thing.
        ok = bookings == {frozenset(claimed)} and micros == {claimed_micro}
        all_ok &= ok
        print(
            f"{gold_pattern:<12}{prediction_pattern:<16}"
            f"{('+'.join(sorted(claimed)) or 'ignored'):<16}{format_bookings(bookings):<16}"
            f"{claimed_micro:<14}{' / '.join(sorted(micros)):<12}{'yes' if ok else 'NO'}"
        )

    if not all_ok:
        print("\nMISMATCH: the slide table does not describe the scorer.")
        return 1
    print("\nOK: every combination is covered by exactly one slide row, and every row is right.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
