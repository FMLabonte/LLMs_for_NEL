"""Build the actual synthetic-abstract JSONs for one or more acceptance levels.

Kept separate from acceptance_levels.py because the decision table is 400 kB and
belongs in git, while the JSONs it describes are roughly 100 MB of near-duplicate
data and do not. Anyone with the repo and the synthetic data can rebuild any
level in a few seconds.

For every level and split it writes two files per generator:

    results_qwen3_*.json   the abstracts that passed at this level
    random_qwen3_*.json    the same NUMBER of abstracts, drawn at random from the
                           unfiltered pool at seed 42

The random control is the point. Comparing a filtered run against the unfiltered
baseline confounds two things, the filtering and the smaller training set. The
control holds size fixed so only the selection differs.

Run:
    python materialize.py --level L2_rate20
    python materialize.py --level all --splits train dev
"""
import argparse
import json
import random
import re
import zipfile
from pathlib import Path

import pandas as pd

OUT_DIR = Path(__file__).resolve().parent
LEVEL_DIR = OUT_DIR / "acceptance_levels"
TABLE = LEVEL_DIR / "acceptance_levels.csv"
SYN_DIR = OUT_DIR.parent / "Data/Synthetic abstracts"
SEED = 42


def _ensure_synthetic_dir() -> Path:
    if not SYN_DIR.exists():
        with zipfile.ZipFile(OUT_DIR.parent / "Data/Synthetic abstracts.zip") as z:
            z.extractall(OUT_DIR.parent / "Data")
    return SYN_DIR


def materialize(df: pd.DataFrame, level: str, splits: list[str]) -> list[dict]:
    dest = LEVEL_DIR / level
    dest.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    report = []

    for path in sorted(_ensure_synthetic_dir().glob("results_qwen3_*.json")):
        model, split = re.match(r"results_(qwen3_\d+b)_(\w+)\.json", path.name).groups()
        if splits and split not in splits:
            continue

        papers = json.loads(path.read_text())
        sub = df[(df.model == model) & (df.split == split)]
        keep = set(zip(sub[sub[level]].paper_id, sub[sub[level]].generation))
        all_pairs = sorted(zip(sub.paper_id, sub.generation))

        kept_papers, n_kept = [], 0
        for p in papers:
            pid = str(p["paper_id"])
            gens = {k: v for k, v in p["synthetic_abstracts"].items() if (pid, k) in keep}
            if gens:
                kept_papers.append({**p, "synthetic_abstracts": gens})
                n_kept += len(gens)
        (dest / path.name).write_text(json.dumps(kept_papers, indent=1))

        picked = set(rng.sample(all_pairs, n_kept)) if n_kept <= len(all_pairs) else set(all_pairs)
        rnd_papers = []
        for p in papers:
            pid = str(p["paper_id"])
            gens = {k: v for k, v in p["synthetic_abstracts"].items() if (pid, k) in picked}
            if gens:
                rnd_papers.append({**p, "synthetic_abstracts": gens})
        (dest / path.name.replace("results_", "random_")).write_text(
            json.dumps(rnd_papers, indent=1))

        report.append({"level": level, "model": model, "split": split,
                       "abstracts": len(all_pairs), "kept": n_kept,
                       "kept_pct": round(n_kept / len(all_pairs) * 100, 1),
                       "papers": len(kept_papers)})
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="all",
                    help="a level name from LEVELS.md, or 'all'")
    ap.add_argument("--splits", nargs="*", default=["train", "dev", "test"])
    args = ap.parse_args()

    if not TABLE.exists():
        raise SystemExit(f"{TABLE} missing, run acceptance_levels.py first")
    df = pd.read_csv(TABLE, dtype={"paper_id": str})

    meta = json.loads((LEVEL_DIR / "levels.json").read_text())
    known = [lv["name"] for lv in meta["levels"]]
    levels = known if args.level == "all" else [args.level]
    unknown = [lv for lv in levels if lv not in known]
    if unknown:
        raise SystemExit(f"unknown level(s) {unknown}, known: {known}")

    report = []
    for lv in levels:
        report.extend(materialize(df, lv, args.splits))
        print(f"wrote {LEVEL_DIR / lv}")

    rep = pd.DataFrame(report)
    print("\n" + rep.to_string(index=False))
    (LEVEL_DIR / "materialized.csv").write_text(rep.to_csv(index=False))


if __name__ == "__main__":
    main()
