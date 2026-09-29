"""Aggregate the multi-seed runs into the mean/variance tables behind ``21_MEAN.md``.

Experiments are grouped (base sources/QC arms, the data-replacement swap, the negative-ratio
sweep). Each experiment is measured under several seeds: the report-12 initial run (unknown seed),
the seeded initial run (seed 42), and the multi-seed sweep (seeds 69, 21, 16 for the base group;
42/69/21 for swap and negative-ratio). One value is taken per (experiment, seed): where an
experiment has two runs at the same seed (e.g. Real BioRED has two seed-42 runs) the extra is
disregarded. Reports mean/variance across the available seeds for Dev-selected all-rels. F1 (plus
precision and recall) and Dev-selected matched F1.

Usage::

    nix develop . -c python finetuning/collect_mean.py
"""

from __future__ import annotations

import glob
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ROOTS = [REPO_ROOT / "finetuning" / "canonical_runs", REPO_ROOT / "finetuning" / "repro_check"]

sys.path.insert(0, str(REPO_ROOT))
from finetuning.collect_fixed import _is_seeded, _key, load  # noqa: E402
from finetuning.collect_canonical import dev_row  # noqa: E402

# Experiments grouped into report sections: (group title, [(config key, label), ...]).
GROUPS: list[tuple[str, list[tuple[tuple, str]]]] = [
    ("Base sources and QC arms", [
        (("real", 394, "1"), "Real BioRED (394)"),
        (("synth8b", 394, "1"), "Synthetic Qwen3-8B (394)"),
        (("synth4b", 394, "1"), "Synthetic Qwen3-4B (394)"),
        (("qc", "dedup", "1"), "QC dedup (212, synth-Dev, old)"),
        (("qc", "dedup_realdev", "1"), "QC dedup (212, real-Dev)"),
        (("qc", "dedup_norel", "1"), "QC dedup NoRel-aware (112, synth-Dev, old)"),
        (("qc", "dedup_norel_realdev", "1"), "QC dedup NoRel-aware (112, real-Dev)"),
        (("qc", "random212", "1"), "Random 212"),
        (("qc", "real_subset", "1"), "Real subset (212 QC papers)"),
        (("qc", "real_subset_norel", "1"), "Real subset NoRel (112 papers)"),
    ]),
    ("Data replacement / abstract swap (report 13); swap 0 = Real BioRED, swap 394 = Synthetic 8B", [
        (("swap", 100, "1"), "Swap 100"),
        (("swap", 200, "1"), "Swap 200"),
        (("swap", 300, "1"), "Swap 300"),
        (("swap", 345, "1"), "Swap 345"),
        (("swap", 369, "1"), "Swap 369"),
        (("swap", 382, "1"), "Swap 382"),
    ]),
    ("Data replacement / abstract swap on QC subsets (report 21 extension); swap 0% = Real subset row, 100% = QC synthetic row", [
        (("qcswap", "dedup", 25), "QC dedup swap 25%"),
        (("qcswap", "dedup", 50), "QC dedup swap 50%"),
        (("qcswap", "dedup", 75), "QC dedup swap 75%"),
        (("qcswap", "dedup", 87.5), "QC dedup swap 87.5%"),
        (("qcswap", "dedup", 93.75), "QC dedup swap 93.75%"),
        (("qcswap", "dedup", 96.875), "QC dedup swap 96.875%"),
        (("qcswap", "dedup_norel", 25), "QC dedup NoRel swap 25%"),
        (("qcswap", "dedup_norel", 50), "QC dedup NoRel swap 50%"),
        (("qcswap", "dedup_norel", 75), "QC dedup NoRel swap 75%"),
    ]),
    ("Negative-ratio sweep (report 12 Result 3); 1:1 = Real / Synthetic 8B base rows", [
        (("real", 394, "2"), "Real 1:2"),
        (("real", 394, "3"), "Real 1:3"),
        (("real", 394, "all"), "Real 1:all"),
        (("synth8b", 394, "2"), "Synth 8B 1:2"),
        (("synth8b", 394, "3"), "Synth 8B 1:3"),
        (("synth8b", 394, "all"), "Synth 8B 1:all"),
    ]),
    ("Negative-ratio sweep on QC subsets; 1:1 = the QC-arm base rows above", [
        (("qc", "dedup", "2"), "QC dedup 1:2"),
        (("qc", "dedup", "3"), "QC dedup 1:3"),
        (("qc", "dedup", "all"), "QC dedup 1:all"),
        (("qc", "dedup_norel", "2"), "QC dedup NoRel 1:2"),
        (("qc", "dedup_norel", "3"), "QC dedup NoRel 1:3"),
        (("qc", "dedup_norel", "all"), "QC dedup NoRel 1:all"),
        (("qc", "real_subset", "2"), "Real subset 1:2"),
        (("qc", "real_subset", "3"), "Real subset 1:3"),
        (("qc", "real_subset", "all"), "Real subset 1:all"),
        (("qc", "real_subset_norel", "2"), "Real subset NoRel 1:2"),
        (("qc", "real_subset_norel", "3"), "Real subset NoRel 1:3"),
        (("qc", "real_subset_norel", "all"), "Real subset NoRel 1:all"),
    ]),
    ("Dataset scaling (report 11); paper-count, 1:1; n394 = the Real / Synthetic 8B base rows", [
        (("real", 100, "1"), "Real n100"),
        (("real", 200, "1"), "Real n200"),
        (("real", 300, "1"), "Real n300"),
        (("synth8b", 100, "1"), "Synth 8B n100"),
        (("synth8b", 200, "1"), "Synth 8B n200"),
        (("synth8b", 300, "1"), "Synth 8B n300"),
        (("real_traindev", 394, "1"), "Real 394 + Dev"),
        (("synth8b_traindev", 394, "1"), "Synth 8B 394 + Dev"),
    ]),
]
# Column order and headers. "old" = report-12 unseeded run (seed unknown); 42 = seeded initial;
# 69/21/16 = the sweep (swap/neg-ratio use 42/69/21, so their seed-16 cell stays blank).
# The 5th "random/reference" column: report-12 unknown-seed run for existing experiments; for the
# brand-new real_subset_norel experiment (no report-12 run) it holds the seed-1509 "random" run.
COLUMNS: list[tuple[object, str]] = [
    (69, "seed 69"), (21, "seed 21"), (16, "seed 16"), (42, "seed 42"), ("old", "random (prior report / s1509)"),
]


def _all_keys() -> set[tuple]:
    return {k for _, rows in GROUPS for k, _ in rows}


def gather() -> dict[tuple, dict[object, dict]]:
    """Return ``key -> {seed_label: devsel_result_dict}`` with one run kept per (key, seed)."""
    wanted = _all_keys()
    out: dict[tuple, dict[object, dict]] = {k: {} for k in wanted}
    for root in ROOTS:
        for p in sorted(glob.glob(str(root / "*/canonical_result.json"))):
            path = Path(p)
            d = json.loads(path.read_text())
            if not _is_seeded(path, d):
                continue
            key = _key(d)
            if key in wanted:
                seed = int(d.get("run_seed", 42))
                out[key].setdefault(seed, d)  # first run wins; extras at same seed disregarded
    _, old = load()
    for key in wanted:
        if 1509 in out[key]:
            # Prefer the seed-1509 run for the 5th "random / s1509" column. Where a seed-1509 run
            # exists (all baseline rows, plus the QC-subset side experiments) it supersedes the old
            # unseeded prior-report run, which stays on disk but is dropped from the report.
            out[key]["old"] = out[key].pop(1509)
        elif key in old:
            # Side experiments without a seed-1509 run fall back to their prior-report random run.
            out[key].setdefault("old", old[key])
    return out


def _val(runs: dict[object, dict], seed: object, variant: str, field: str = "biored_f1") -> float | None:
    """Dev-selected epoch's ``field`` (biored_f1 / biored_precision / biored_recall) for a seed."""
    d = runs.get(seed)
    if d is None:
        return None
    row = dev_row(d)
    return None if row is None else float(row[variant][field])


def _cell(v: float | None) -> str:
    return "-" if v is None else f"{v:.3f}"


def table(data: dict[tuple, dict[object, dict]], rows: list[tuple[tuple, str]], variant: str, title: str, field: str) -> None:
    print(f"\n### {title}\n")
    head = " | ".join(h for _, h in COLUMNS)
    print(f"| Experiment | {head} | n | mean | variance | std |")
    print("|---|" + ":-:|" * (len(COLUMNS) + 4))
    for key, label in rows:
        runs = data.get(key, {})
        vals = [v for s, _ in COLUMNS if (v := _val(runs, s, variant, field)) is not None]
        cells = " | ".join(_cell(_val(runs, s, variant, field)) for s, _ in COLUMNS)
        if vals:
            mean = f"{statistics.fmean(vals):.3f}"
            var = f"{statistics.pvariance(vals):.5f}" if len(vals) > 1 else "0.00000"
            std = f"{statistics.pstdev(vals):.3f}" if len(vals) > 1 else "0.000"
        else:
            mean = var = std = "-"
        print(f"| {label} | {cells} | {len(vals)} | {mean} | {var} | {std} |")


def main() -> None:
    data = gather()
    for group_title, rows in GROUPS:
        print(f"\n## {group_title}")
        table(data, rows, "all_pairs", "Dev-selected all-relations BioRED F1 (headline)", "biored_f1")
        table(data, rows, "all_pairs", "Dev-selected all-relations precision", "biored_precision")
        table(data, rows, "all_pairs", "Dev-selected all-relations recall", "biored_recall")
        table(data, rows, "matched", "Dev-selected matched BioRED F1 (secondary)", "biored_f1")


if __name__ == "__main__":
    main()
