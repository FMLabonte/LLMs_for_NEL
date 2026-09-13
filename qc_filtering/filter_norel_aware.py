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
OUT_DYN = HERE / "filtered_norel_dynamic"

CUTOFF = 0.5
TOP3 = {"Association", "Positive_Correlation", "Negative_Correlation"}
KEY = ["model", "split", "paper_id", "generation"]


def allowed_errors(n_claims: int) -> int:
    """The 2026-07-22 step function, same one decide.py uses for `passed`."""
    if n_claims <= 8:
        return 0
    if n_claims <= 12:
        return 1
    return 2


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
    d["n_total"] = d.n_stated + d.n_implicit

    # Strict: no claim may fail. The two columns shipped on 2026-09-10.
    d["keep_stated_only"] = d.failed_stated == 0
    d["keep_norel_aware"] = d.failed_total == 0

    # Dynamic: the same step function `passed` uses, now judging both claim sets.
    # Added 2026-09-13, because Christoph's QC arm is the dynamic set and not the
    # strict one, so a strict-only norel file changes the rule and the claim set at
    # the same time and his before/after comparison measures two things at once.
    #
    # Two readings of "the same rule", both kept because they cost the same:
    #   full    allowance scales with everything judged, so the extra false-alarm
    #           exposure from ~29 implicit pairs is compensated. Literal reading.
    #   stated  allowance stays on the stated count, so the July budget is untouched
    #           and the only change is that more claims can spend it. Conservative.
    d["keep_norel_dynamic"] = d.failed_total <= d.n_total.map(allowed_errors)
    d["keep_norel_dyn_stated"] = d.failed_total <= d.n_stated.map(allowed_errors)
    return d


def write_filtered(d: pd.DataFrame, column: str, out_dir: Path,
                   before_col: str = "keep_stated_only") -> list[str]:
    """Write Fred-schema JSONs holding only the generations `column` keeps."""
    out_dir.mkdir(exist_ok=True)
    lines = []
    for path in sorted(SYN_DIR.glob("results_qwen3_*.json")):
        model, split = path.stem.replace("results_", "").rsplit("_", 1)
        papers = json.loads(path.read_text())
        sub = d[(d.model == model) & (d.split == split)]
        keep = set(zip(sub[sub[column]].paper_id, sub[sub[column]].generation))

        kept_papers, n_kept = [], 0
        for p in papers:
            pid = str(p["paper_id"])
            gens = {k: v for k, v in p["synthetic_abstracts"].items() if (pid, k) in keep}
            if gens:
                kept_papers.append({**p, "synthetic_abstracts": gens})
                n_kept += len(gens)
        (out_dir / path.name).write_text(json.dumps(kept_papers, indent=1))

        before = int(sub[before_col].sum())
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

    c, e = int(d.keep_norel_dynamic.sum()), int(d.keep_norel_dyn_stated.sum())
    print(f"kept, NoRelation aware, dynamic (full)   : {c:,} ({c/len(d)*100:.1f}%)")
    print(f"kept, NoRelation aware, dynamic (stated) : {e:,} ({e/len(d)*100:.1f}%)")

    lines = write_filtered(d, "keep_norel_aware", OUT)
    print("\nstrict rule -> filtered_norel/")
    print("| model | split | abstracts | kept, stated only | kept, NoRelation aware | papers |")
    print("|---|---|---|---|---|---|")
    for l in lines:
        print(l)

    dyn = write_filtered(d, "keep_norel_dynamic", OUT_DYN, before_col="keep_norel_aware")
    print("\ndynamic rule -> filtered_norel_dynamic/")
    print("| model | split | abstracts | kept, strict | kept, dynamic | papers |")
    print("|---|---|---|---|---|---|")
    for l in dyn:
        print(l)

    d.to_csv(HERE / "abstract_decisions_norel.csv", index=False)
    print(f"\nwritten: {OUT}/, {OUT_DYN}/ and abstract_decisions_norel.csv")


if __name__ == "__main__":
    main()
