"""QC filtering that also scores the implicit NoRelation pairs.

The shipped filter (decide.py) only ever judged the relations a generation prompt lists.
Every co-mentioned entity pair that the prompt does not put in a relation is implicitly a
NoRelation claim, and none of those were scored, so a relation the generator invented
could not be caught. Frederik asked at meeting 10 for at least one run that includes them.

An abstract is kept here when NONE of its claims fails, counting both sides:

  * the stated relations, from synthetic_relation_scores.csv, rare types excluded
    because the QC model was never trained on them,
  * the implicit NoRelation pairs, from the overnight scoring run.

That is the same strict rule Christoph's "QC dedup" arm already uses, so the only thing
that changes between his existing run and this one is which claims the filter looked at.

Output mirrors filtered/ exactly, same schema, so it is a path change on his side:

  filtered_norel/results_qwen3_*_{train,dev,test}.json

Run:
    python filter_norel_aware.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SCORES = HERE / "synthetic_relation_scores.csv"
NOREL = REPO.parent / "work" / "2026-09-09_overnight" / "norelation_scores.csv"
SYN_DIR = REPO / "Data" / "Synthetic abstracts"
OUT = HERE / "filtered_norel"

CUTOFF = 0.5
TOP3 = {"Association", "Positive_Correlation", "Negative_Correlation"}
KEY = ["model", "split", "paper_id", "generation"]


def failures_per_abstract() -> pd.DataFrame:
    stated = pd.read_csv(SCORES, dtype={"paper_id": str})
    stated = stated[stated.relation_type.isin(TOP3)].copy()
    stated["failed"] = stated.prob_supported < CUTOFF
    s = stated.groupby(KEY).failed.agg(["sum", "count"])
    s.columns = ["failed_stated", "n_stated"]

    implicit = pd.read_csv(NOREL, dtype={"paper_id": str})
    implicit["failed"] = implicit.prob_supported < CUTOFF
    i = implicit.groupby(KEY).failed.agg(["sum", "count"])
    i.columns = ["failed_implicit", "n_implicit"]

    d = s.join(i, how="left").fillna(0).astype(int).reset_index()
    d["failed_total"] = d.failed_stated + d.failed_implicit
    d["keep_stated_only"] = d.failed_stated == 0
    d["keep_norel_aware"] = d.failed_total == 0
    return d


def write_filtered(d: pd.DataFrame) -> list[str]:
    OUT.mkdir(exist_ok=True)
    lines = []
    for path in sorted(SYN_DIR.glob("results_qwen3_*.json")):
        model, split = path.stem.replace("results_", "").rsplit("_", 1)
        papers = json.loads(path.read_text())
        sub = d[(d.model == model) & (d.split == split)]
        keep = set(zip(sub[sub.keep_norel_aware].paper_id, sub[sub.keep_norel_aware].generation))

        kept_papers, n_kept = [], 0
        for p in papers:
            pid = str(p["paper_id"])
            gens = {k: v for k, v in p["synthetic_abstracts"].items() if (pid, k) in keep}
            if gens:
                kept_papers.append({**p, "synthetic_abstracts": gens})
                n_kept += len(gens)
        (OUT / path.name).write_text(json.dumps(kept_papers, indent=1))

        before = int(sub.keep_stated_only.sum())
        total = len(sub)
        lines.append(
            f"| {model} | {split} | {total} | {before} ({before/total*100:.1f}%) | "
            f"{n_kept} ({n_kept/total*100:.1f}%) | {len(kept_papers)} / {papers.__len__()} |")
    return lines


def main():
    d = failures_per_abstract()
    covered = (d.n_implicit > 0).sum()
    print(f"abstracts scored: {len(d):,}   with implicit pairs available: {covered:,}")
    print(f"implicit claims folded in: {d.n_implicit.sum():,}")
    print(f"mean claims per abstract: stated {d.n_stated.mean():.1f} -> "
          f"{(d.n_stated + d.n_implicit).mean():.1f} with implicit\n")

    a, b = int(d.keep_stated_only.sum()), int(d.keep_norel_aware.sum())
    print(f"kept, stated relations only : {a:,} of {len(d):,} ({a/len(d)*100:.1f}%)")
    print(f"kept, NoRelation aware      : {b:,} of {len(d):,} ({b/len(d)*100:.1f}%)")

    lost = d[d.keep_stated_only & ~d.keep_norel_aware]
    print(f"\nabstracts that passed before and fail now: {len(lost):,}")
    print(f"  of those, median implicit failures: {lost.failed_implicit.median():.0f}")

    lines = write_filtered(d)
    print("\n| model | split | abstracts | kept, stated only | kept, NoRelation aware | papers |")
    print("|---|---|---|---|---|---|")
    for l in lines:
        print(l)

    d.to_csv(HERE / "abstract_decisions_norel.csv", index=False)
    print(f"\nwritten: {OUT}/  and abstract_decisions_norel.csv")


if __name__ == "__main__":
    main()
