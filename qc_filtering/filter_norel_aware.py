"""QC filtering that also scores the implicit NoRelation pairs.

The shipped filter (decide.py) only ever judged the relations a generation prompt lists.
Every co-mentioned entity pair that the prompt does not put in a relation is implicitly a
NoRelation claim, and none of those were scored, so a relation the generator invented
could not be caught. Frederik asked at meeting 10 for at least one run that includes them.

Both claim sets are judged:

  * the stated relations, from synthetic_relation_scores.csv, rare types excluded
    because the QC model was never trained on them,
  * the implicit NoRelation pairs, from the overnight scoring run.

Two decision columns, two output folders:

  passed         the dynamic step function, same rule decide.py applies in
                 abstract_decisions.csv  ->  filtered_norel_dynamic/
  passed_strict  no claim may fail at all  ->  filtered_norel/

`passed` is the one to use. Christoph's QC arm is the dynamic set, so judging the
implicit pairs under the strict rule would change the rule and the claim set at the
same time and his before/after would measure two things at once. Under `passed` the
rule is held constant and only the claim set moves, which is the comparison Frederik
asked for at meeting 10. `passed_strict` stays because filtered_norel/ is built from
it and he has already been given its 30.5% -> 17.3% figure.

NOTE: `passed` means 1,307 in abstract_decisions.csv and 779 here. Same rule, different
claims. Never quote the number without naming the file.

Both folders mirror filtered/ exactly, same schema, so it is a path change downstream:

  filtered_norel_dynamic/results_qwen3_*_{train,dev,test}.json

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

    # `passed` means the same thing here as in abstract_decisions.csv: the dynamic
    # step function decided it. The only difference between the two files is which
    # claims were judged, stated relations there, stated plus implicit NoRelation
    # pairs here. That is deliberate, so the before/after comparison holds the rule
    # constant. It also means the number differs between the files, 1,307 there and
    # 779 here, so never quote `passed` without saying which file it came from.
    d["passed"] = d.failed_total <= d.n_total.map(allowed_errors)

    # The strict rule, kept because ../filtered_norel/ is built from it and Frederik
    # has already been given its 30.5% -> 17.3% figure.
    d["passed_strict"] = d.failed_total == 0
    return d


def write_filtered(d: pd.DataFrame, column: str, out_dir: Path) -> dict[tuple, tuple]:
    """Write Fred-schema JSONs holding only the generations `column` keeps.

    Returns {(model, split): (abstracts_total, kept, papers_kept, papers_total)}.
    """
    out_dir.mkdir(exist_ok=True)
    counts = {}
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
        counts[(model, split)] = (len(sub), n_kept, len(kept_papers), len(papers))
    return counts


def main():
    d = failures_per_abstract()
    covered = (d.n_implicit > 0).sum()
    print(f"abstracts scored: {len(d):,}   with implicit pairs available: {covered:,}")
    print(f"implicit claims folded in: {d.n_implicit.sum():,}")
    print(f"mean claims per abstract: stated {d.n_stated.mean():.1f} -> "
          f"{(d.n_stated + d.n_implicit).mean():.1f} with implicit\n")

    # Derived, not shipped as columns: the stated-only verdicts live in
    # abstract_decisions.csv (dynamic) and acceptance_levels.csv (L0_strict).
    stated_strict = int((d.failed_stated == 0).sum())
    p, s = int(d.passed.sum()), int(d.passed_strict.sum())
    print(f"kept, strict, stated only  : {stated_strict:,} of {len(d):,} "
          f"({stated_strict/len(d)*100:.1f}%)   [= L0_strict]")
    print(f"kept, strict  (passed_strict): {s:,} ({s/len(d)*100:.1f}%)")
    print(f"kept, dynamic (passed)       : {p:,} ({p/len(d)*100:.1f}%)")

    lost = d[(d.failed_stated == 0) & ~d.passed_strict]
    print(f"\nabstracts the strict stated-only rule kept and passed_strict drops: {len(lost):,}")
    print(f"  of those, median implicit failures: {lost.failed_implicit.median():.0f}")

    strict = write_filtered(d, "passed_strict", OUT)
    dyn = write_filtered(d, "passed", OUT_DYN)

    print("\n| model | split | abstracts | passed_strict | passed (dynamic) | papers, dynamic |")
    print("|---|---|---|---|---|---|")
    for key in sorted(dyn):
        total, n_dyn, pap_dyn, pap_all = dyn[key]
        _, n_str, _, _ = strict[key]
        print(f"| {key[0]} | {key[1]} | {total} | {n_str} ({n_str/total*100:.1f}%) | "
              f"{n_dyn} ({n_dyn/total*100:.1f}%) | {pap_dyn} / {pap_all} |")

    d.to_csv(HERE / "abstract_decisions_norel.csv", index=False)
    print(f"\nwritten: {OUT}/, {OUT_DYN}/ and abstract_decisions_norel.csv")


if __name__ == "__main__":
    main()
