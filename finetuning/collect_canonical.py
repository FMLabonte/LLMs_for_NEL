"""Aggregate the canonical-recipe (C5) runs into the tables behind ``12_CANONICAL_COMPARE.md``.

Reads every ``canonical_result.json`` under ``finetuning/canonical_runs/`` plus the two
recipe-identical baselines from the 5-epoch scaling sweep (real and synthetic-8B at 394
papers, ``finetuning/scaling_runs/``), and prints the report's Markdown tables so the
numbers are reproducible rather than hand-transcribed.

Conditions are keyed by ``(condition, generations, neg_ratio)`` because several share the
``synth8b`` label (the 1-gen baseline, the 3-gen augmentation row, and the negative-ratio
variants).

Usage::

    nix develop . -c python finetuning/collect_canonical.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CANON_DIR = REPO_ROOT / "finetuning" / "canonical_runs"
SCALING_DIR = REPO_ROOT / "finetuning" / "scaling_runs"


def load_runs() -> dict[tuple[str, int, str], dict]:
    """Return ``{(condition, generations, neg_ratio): result_dict}`` for every C5 run."""
    runs: dict[tuple[str, int, str], dict] = {}

    # Recipe-identical baselines from the scaling sweep (1 generation, 1:1 negatives).
    for path in glob.glob(str(SCALING_DIR / "scaling_biored_n394_*2026-09-04*/scaling_result.json")):
        runs[("real", 1, "1")] = json.loads(Path(path).read_text())
    for path in glob.glob(str(SCALING_DIR / "scaling_synthetic_n394_*2026-09-04*/scaling_result.json")):
        runs[("synth8b", 1, "1")] = json.loads(Path(path).read_text())

    for path in glob.glob(str(CANON_DIR / "*/canonical_result.json")):
        d = json.loads(Path(path).read_text())
        condition = d.get("condition", d.get("source"))
        # The Result-4 selection experiment (synthetic source, but selected on the real
        # BioRED Dev split) shares its (condition, gen, neg) key with the synth8b baseline;
        # keep it out of the standard tables so it does not clobber the baseline. It is
        # reported separately via ``result4_run()``.
        if d.get("source", "").startswith("synth") and d.get("dev_source") == "biored":
            continue
        # "biored" (real text) shares the "real" label with the scaling baseline.
        if condition == "biored":
            condition = "real"
        key = (condition, int(d.get("generations", 1)), str(d.get("neg_ratio", "1")))
        runs[key] = d
    return runs


def result4_run() -> dict | None:
    """Return the Result-4 run dict (synthetic source selected on the real BioRED Dev split)."""
    for path in glob.glob(str(CANON_DIR / "*/canonical_result.json")):
        d = json.loads(Path(path).read_text())
        if d.get("source", "").startswith("synth") and d.get("dev_source") == "biored":
            return d
    return None


def dev_row(d: dict) -> dict | None:
    """Return the Dev-selected epoch's row, or ``None`` when the run was unselected.

    Matches by ``epoch`` number rather than list index so it also works for dev-only-scored
    runs whose ``per_epoch`` holds just the selected epoch.
    """
    epoch = d.get("dev_selected_epoch")
    if not epoch:
        return None
    for row in d["per_epoch"]:
        if row.get("epoch") == epoch:
            return row
    return d["per_epoch"][epoch - 1] if epoch - 1 < len(d["per_epoch"]) else None


def best_row(d: dict, variant: str) -> dict:
    """Return the epoch row with the highest BioRED F1 for ``variant``."""
    return max(d["per_epoch"], key=lambda e: e[variant]["biored_f1"])


def fmt_dev(d: dict) -> str:
    """Format the Dev-selected all/matched F1 (or ``unselected``)."""
    row = dev_row(d)
    if row is None:
        return "unselected"
    return f"{row['all_pairs']['biored_f1']:.3f} / {row['matched']['biored_f1']:.3f} (ep{d['dev_selected_epoch']})"


def fmt_best(d: dict, variant: str) -> str:
    """Format the best-epoch BioRED F1 for one variant with its epoch."""
    row = best_row(d, variant)
    return f"{row[variant]['biored_f1']:.3f} (ep{row['epoch']})"


def table(title: str, rows: list[tuple[str, tuple[str, int, str]]], runs: dict) -> None:
    """Print one Markdown comparison table for the given labelled condition keys."""
    print(f"\n### {title}\n")
    print("| Condition | Dev-selected all / matched | Best-epoch all rels. | Best-epoch matched |")
    print("|---|:-:|:-:|:-:|")
    for label, key in rows:
        if key not in runs:
            print(f"| {label} | _pending_ | _pending_ | _pending_ |")
            continue
        d = runs[key]
        print(f"| {label} | {fmt_dev(d)} | {fmt_best(d, 'all_pairs')} | {fmt_best(d, 'matched')} |")


def appendix(runs: dict) -> None:
    """Print a per-epoch BioRED F1 table (both variants) for every condition."""
    print("\n## Full per-epoch metrics (real BioRED Test)\n")
    print("`*` marks the Dev-selected epoch; P/R are precision/recall; TP/FP/FN pooled positives.\n")
    label_of = {
        ("real", 1, "1"): "Real BioRED (394, 1 gen, 1:1)",
        ("synth8b", 1, "1"): "Synthetic Qwen3-8B (394, 1 gen, 1:1)",
        ("synth4b", 1, "1"): "Synthetic Qwen3-4B (394, 1 gen, 1:1)",
        ("combined", 1, "1"): "Combined real+8B (394, 1 gen, 1:1, unselected)",
        ("synth8b", 3, "1"): "Synthetic Qwen3-8B (394, 3 gens, 1:1)",
        ("qc-dedup", 1, "1"): "QC dedup (212)",
        ("qc-dedup_norel", 1, "1"): "QC dedup NoRelation-aware (112)",
        ("qc-real_subset", 1, "1"): "Real subset (212 QC papers)",
        ("qc-random212", 1, "1"): "Random 212",
        ("qc-qc_failed", 1, "1"): "QC-failed (182)",
        ("real", 1, "2"): "Real 1:2",
        ("real", 1, "3"): "Real 1:3",
        ("real", 1, "all"): "Real 1:all",
        ("synth8b", 1, "2"): "Synthetic 8B 1:2",
        ("synth8b", 1, "3"): "Synthetic 8B 1:3",
        ("synth8b", 1, "all"): "Synthetic 8B 1:all",
    }
    extra = result4_run()
    items = list(label_of.items())
    if extra is not None:
        items.append((None, "Synthetic Qwen3-8B (394, 1 gen, 1:1), selected on BioRED Dev (Result 4)"))
    for key, label in items:
        d = extra if key is None else (runs[key] if key in runs else None)
        if d is None:
            continue
        sel = d.get("dev_selected_epoch")
        print(f"\n#### {label}\n")
        print("| Epoch | Dev f1_micro | matched F1 | mP | mR | matched TP/FP/FN | all rels. F1 | aP | aR | all rels. TP/FP/FN |")
        print("|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|")
        for e in d["per_epoch"]:
            m, a = e["matched"], e["all_pairs"]
            star = "*" if e["epoch"] == sel else ""
            dev = f"{e['dev_f1_micro']:.4f}" if e.get("dev_f1_micro") is not None else "-"
            print(
                f"| {e['epoch']}{star} | {dev} | {m['biored_f1']:.3f} | {m['biored_precision']:.3f} | "
                f"{m['biored_recall']:.3f} | {m['tp']}/{m['fp']}/{m['fn']} | {a['biored_f1']:.3f} | "
                f"{a['biored_precision']:.3f} | {a['biored_recall']:.3f} | {a['tp']}/{a['fp']}/{a['fn']} |"
            )


def main() -> None:
    import sys as _sys

    runs = load_runs()
    if "--appendix" in _sys.argv:
        appendix(runs)
        return
    print("Loaded conditions:", sorted(runs.keys()))

    table(
        "Source comparison (394 papers, 1 generation, 1:1 negatives)",
        [
            ("Real BioRED", ("real", 1, "1")),
            ("Synthetic Qwen3-8B", ("synth8b", 1, "1")),
            ("Synthetic Qwen3-4B", ("synth4b", 1, "1")),
            ("Combined real+8B (unselected)", ("combined", 1, "1")),
            ("Synthetic 8B, 3 generations (augmentation)", ("synth8b", 3, "1")),
        ],
        runs,
    )

    table(
        "QC-subset study (212/182-paper budget, 1:1 negatives)",
        [
            ("Real BioRED (full 394, ref)", ("real", 1, "1")),
            ("QC dedup (212, best passing gen)", ("qc-dedup", 1, "1")),
            ("QC dedup NoRelation-aware (112, best passing gen)", ("qc-dedup_norel", 1, "1")),
            ("Real subset (212 QC papers)", ("qc-real_subset", 1, "1")),
            ("Random 212 (no selection)", ("qc-random212", 1, "1")),
            ("QC-failed (182 reject papers)", ("qc-qc_failed", 1, "1")),
        ],
        runs,
    )

    table(
        "Negative-ratio sweep (394 papers, 1 generation)",
        [
            ("Real 1:1", ("real", 1, "1")),
            ("Real 1:2", ("real", 1, "2")),
            ("Real 1:3", ("real", 1, "3")),
            ("Real 1:all", ("real", 1, "all")),
            ("Synth 8B 1:1", ("synth8b", 1, "1")),
            ("Synth 8B 1:2", ("synth8b", 1, "2")),
            ("Synth 8B 1:3", ("synth8b", 1, "3")),
            ("Synth 8B 1:all", ("synth8b", 1, "all")),
        ],
        runs,
    )


if __name__ == "__main__":
    main()
