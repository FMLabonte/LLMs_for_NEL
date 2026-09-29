"""Aggregate the reseeded C5 runs into the tables behind ``20_Canonical_Fixed.md``.

Every run is classified as *new* (seeded: ``seeded_init == True``) or *old* (the unseeded
runs whose weights were pruned but whose result JSONs were kept). Runs are keyed by config so
the new number can be shown next to the old one, quantifying the impact of seeding the head.

Group-1 baselines launched before the ``seeded_init`` field was added; pass their run dirs are
detected as seeded by timestamp (``>= SEEDED_TS``) as a fallback.

Usage::

    nix develop . -c python finetuning/collect_fixed.py            # comparison tables
    nix develop . -c python finetuning/collect_fixed.py --appendix # per-epoch, seeded only
"""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CANON_DIR = REPO_ROOT / "finetuning" / "canonical_runs"
SCALING_DIR = REPO_ROOT / "finetuning" / "scaling_runs"
SEEDED_TS = "2026-09-14_15-00"  # group-1 baselines (no seeded_init field) started after this

sys.path.insert(0, str(REPO_ROOT))
from finetuning.collect_canonical import best_row, dev_row  # noqa: E402

_TS = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2})")


def _source_kind(d: dict) -> str:
    """Map a run's source label to the report kind."""
    src = d.get("source") or d.get("condition") or ""
    return {"biored": "real", "synthetic": "synth8b"}.get(src, src)


def _key(d: dict) -> tuple:
    """Config key shared by an old run and its seeded replacement."""
    if d.get("include_dev_in_train"):
        # Train+dev experiment (report 21): source's own Dev folded into training, fixed eval epoch.
        return (f"{_source_kind(d)}_traindev", int(d.get("n_papers") or 0), str(d.get("neg_ratio", "1")))
    if d.get("qc_swap_arm") is not None:
        f = d["swap_frac"]
        fk = int(f) if float(f) == int(f) else f  # 25.0 -> 25, 87.5 -> 87.5
        return ("qcswap", d["qc_swap_arm"], fk)
    if d.get("swap_n") is not None:
        return ("swap", int(d["swap_n"]), "1")
    if d.get("qc_arm") not in (None, "none"):
        arm = d["qc_arm"]
        # A synthetic QC arm selected on real BioRED Dev is a distinct experiment (report 21
        # real-Dev endpoints); key it apart from the synth-Dev run of the same arm.
        if arm not in ("real_subset", "real_subset_norel") and d.get("dev_source") == "biored":
            arm = f"{arm}_realdev"
        return ("qc", arm, str(d.get("neg_ratio", "1")))
    return (_source_kind(d), int(d.get("n_papers") or 0), str(d.get("neg_ratio", "1")))


def _is_seeded(path: Path, d: dict) -> bool:
    if d.get("seeded_init") is True:
        return True
    m = _TS.search(path.parent.name)
    return bool(m and m.group(1) >= SEEDED_TS)


def load() -> tuple[dict[tuple, dict], dict[tuple, dict]]:
    """Return ``(new, old)`` maps ``key -> result_dict`` (latest run wins per key)."""
    new: dict[tuple, dict] = {}
    old: dict[tuple, dict] = {}
    for pattern in (CANON_DIR / "*/canonical_result.json", SCALING_DIR / "*/scaling_result.json"):
        for p in sorted(glob.glob(str(pattern))):
            path = Path(p)
            d = json.loads(path.read_text())
            (new if _is_seeded(path, d) else old).setdefault(_key(d), d)
            (new if _is_seeded(path, d) else old)[_key(d)] = d
    return new, old


def _best(d: dict | None, variant: str) -> float | None:
    return None if d is None else float(best_row(d, variant)[variant]["biored_f1"])


def _devsel(d: dict | None, variant: str) -> float | None:
    """Dev-selected epoch's F1 for a variant (the headline number)."""
    if d is None:
        return None
    row = dev_row(d)
    return None if row is None else float(row[variant]["biored_f1"])


def _cell(v: float | None) -> str:
    return "-" if v is None else f"{v:.3f}"


def _delta(new: dict | None, old: dict | None) -> str:
    """Δ on the headline Dev-selected all-rels."""
    a, b = _devsel(new, "all_pairs"), _devsel(old, "all_pairs")
    return "-" if a is None or b is None else f"{a - b:+.3f}"


APPENDIX_ORDER: list[tuple[tuple, str]] = [
    (("real", 394, "1"), "Real BioRED (394, 1 gen, 1:1)"),
    (("synth8b", 394, "1"), "Synthetic Qwen3-8B (394, 1 gen, 1:1)"),
    (("qc", "dedup", "1"), "QC dedup (212)"),
    (("qc", "allpass", "1"), "QC allpass (212)"),
    (("qc", "dedup_norel", "1"), "QC dedup NoRelation-aware (112)"),
    (("qc", "random212", "1"), "Random 212"),
    (("qc", "qc_failed", "1"), "QC-failed (182)"),
    (("qc", "real_subset", "1"), "Real subset (212 QC papers)"),
    (("real", 394, "2"), "Real 1:2"), (("real", 394, "3"), "Real 1:3"), (("real", 394, "all"), "Real 1:all"),
    (("synth8b", 394, "2"), "Synth 8B 1:2"), (("synth8b", 394, "3"), "Synth 8B 1:3"), (("synth8b", 394, "all"), "Synth 8B 1:all"),
    (("real", 100, "1"), "Real n100"), (("real", 200, "1"), "Real n200"), (("real", 300, "1"), "Real n300"),
    (("synth8b", 100, "1"), "Synth 8B n100"), (("synth8b", 200, "1"), "Synth 8B n200"), (("synth8b", 300, "1"), "Synth 8B n300"),
    (("swap", 100, "1"), "Swap 100"), (("swap", 200, "1"), "Swap 200"), (("swap", 300, "1"), "Swap 300"),
]


def appendix(new: dict[tuple, dict]) -> None:
    """Full per-epoch tables (report-12 format) for every seeded run present."""
    print("`*` marks the Dev-selected epoch; P/R are precision/recall; TP/FP/FN pooled positives.")
    for key, label in APPENDIX_ORDER:
        d = new.get(key)
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


def table(title: str, rows: list[tuple[str, tuple]], new: dict, old: dict) -> None:
    print(f"\n### {title}\n")
    print("| Condition | old Dev-sel all | new Dev-sel all | Δ | new Dev-sel matched | new best all (side) |")
    print("|---|:-:|:-:|:-:|:-:|:-:|")
    for label, key in rows:
        n, o = new.get(key), old.get(key)
        status = "" if n else " _(pending)_"
        print(
            f"| {label}{status} | {_cell(_devsel(o, 'all_pairs'))} | {_cell(_devsel(n, 'all_pairs'))} | "
            f"{_delta(n, o)} | {_cell(_devsel(n, 'matched'))} | {_cell(_best(n, 'all_pairs'))} |"
        )


def main() -> None:
    new, old = load()
    if "--appendix" in sys.argv:
        appendix(new)
        return

    table("Result 1: source baselines (394, 1 gen, 1:1)",
          [("Real BioRED", ("real", 394, "1")), ("Synthetic Qwen3-8B", ("synth8b", 394, "1"))], new, old)
    table("Result 2: QC-subset study (1:1)",
          [("QC dedup (212)", ("qc", "dedup", "1")),
           ("QC allpass (212)", ("qc", "allpass", "1")),
           ("QC dedup NoRelation-aware (112)", ("qc", "dedup_norel", "1")),
           ("Random 212", ("qc", "random212", "1")),
           ("QC-failed (182)", ("qc", "qc_failed", "1")),
           ("Real subset (212 QC papers)", ("qc", "real_subset", "1"))], new, old)
    table("Result 3: negative-ratio sweep (394, 1 gen)",
          [("Real 1:1", ("real", 394, "1")), ("Real 1:2", ("real", 394, "2")),
           ("Real 1:3", ("real", 394, "3")), ("Real 1:all", ("real", 394, "all")),
           ("Synth 8B 1:1", ("synth8b", 394, "1")), ("Synth 8B 1:2", ("synth8b", 394, "2")),
           ("Synth 8B 1:3", ("synth8b", 394, "3")), ("Synth 8B 1:all", ("synth8b", 394, "all"))], new, old)
    table("Report 11: dataset scaling (paper count, 1:1)",
          [(f"Real n{n}", ("real", n, "1")) for n in (100, 200, 300, 394)]
          + [(f"Synth 8B n{n}", ("synth8b", n, "1")) for n in (100, 200, 300, 394)], new, old)
    table("Report 13: data replacement (swap of 394)",
          [("Swap 0 (real)", ("real", 394, "1"))]
          + [(f"Swap {n}", ("swap", n, "1")) for n in (100, 200, 300)]
          + [("Swap 394 (synthetic)", ("synth8b", 394, "1"))], new, old)


if __name__ == "__main__":
    main()
