"""Task 2: report deduplicated abstracts kept next to total abstracts kept.

Fred at meeting 9: "some abstracts are duplicates, because I produced them multiple
times ... so you have abstracts kept and you can have deduplicated abstracts kept."

Two things are done here rather than one, because the request rests on a premise that
should be checked before it is used:

  1. Are the repeated generations actually near-duplicates? Measured, not assumed,
     with word-level Jaccard overlap between the generations of one paper, against a
     control of pairs drawn from different papers.
  2. The acceptance ladder recomputed from the scored claims, with abstracts kept and
     deduplicated abstracts kept side by side. Nothing is read from LEVELS.md.

Run:
    python task2_dedup.py
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent / "LLMs_for_NEL"
SCORES = REPO / "qc_filtering" / "synthetic_relation_scores.csv"
SYN_DIR = REPO / "Data" / "Synthetic abstracts"

FOUR_CLASS = {"Association", "Positive_Correlation", "Negative_Correlation", "NoRelation"}
CUTOFF = 0.5
LEVELS = [("L0 strict", 0.0), ("L1 10%", 0.10), ("L2 20%", 0.20),
          ("L3 33%", 0.33), ("L4 50%", 0.50), ("L5 unfiltered", 1.0)]


def clean(text: str) -> str:
    text = re.sub(r"^\s*\**\s*Abstract:?\s*\**\s*", "", text.strip(), flags=re.I)
    return re.sub(r"\*\*", "", text).strip()


def duplicate_check() -> dict:
    """How much text do two generations of the same paper share?"""
    texts: dict[tuple[str, str, str], list[str]] = {}
    for path in sorted(SYN_DIR.glob("results_qwen3_*.json")):
        model, split = re.match(r"results_(qwen3_\d+b)_(\w+)\.json", path.name).groups()
        for p in json.load(open(path)):
            key = (model, split, str(p["paper_id"]))
            texts[key] = [clean(v) for v in p["synthetic_abstracts"].values()]
    print(f"{len(texts)} paper-files, "
          f"{sum(len(v) for v in texts.values())} synthetic abstracts")

    def jac(a: str, b: str) -> float:
        sa, sb = set(a.lower().split()), set(b.lower().split())
        return len(sa & sb) / max(len(sa | sb), 1)

    same, exact = [], 0
    for v in texts.values():
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                same.append(jac(v[i], v[j]))
                exact += v[i] == v[j]
    rng = random.Random(42)
    keys = list(texts)
    diff = []
    for _ in range(len(same)):
        k1, k2 = rng.sample(keys, 2)
        diff.append(jac(rng.choice(texts[k1]), rng.choice(texts[k2])))

    s, d = pd.Series(same), pd.Series(diff)
    print(f"\nword overlap (Jaccard) between two generations of the SAME paper: "
          f"median {s.median():.3f}, mean {s.mean():.3f}, n={len(s):,}")
    print(f"word overlap between abstracts of DIFFERENT papers (control):     "
          f"median {d.median():.3f}, mean {d.mean():.3f}, n={len(d):,}")
    print(f"byte-identical generation pairs: {exact}")
    return {"same_paper_jaccard_median": round(float(s.median()), 4),
            "same_paper_jaccard_mean": round(float(s.mean()), 4),
            "different_paper_jaccard_median": round(float(d.median()), 4),
            "different_paper_jaccard_mean": round(float(d.mean()), 4),
            "n_pairs": int(len(s)), "byte_identical_pairs": int(exact)}


def ladder() -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(SCORES, dtype={"paper_id": str})
    print(f"\nloaded {len(raw):,} scored claims")
    pool = raw.drop_duplicates(["model", "split", "paper_id", "generation"])
    print(f"pool: {len(pool):,} synthetic abstracts, "
          f"{pool.paper_id.nunique()} distinct BioRED papers, "
          f"{pool.groupby(['model','paper_id']).ngroups} paper-generator combinations")

    d = raw[raw.relation_type.isin(FOUR_CLASS)].copy()
    d["failed"] = d.prob_supported < CUTOFF
    key = ["model", "split", "paper_id", "generation"]
    a = (d.groupby(key)
           .agg(n_relations=("rel_idx", "count"), n_failed=("failed", "sum"))
           .reset_index())
    a["rate"] = a.n_failed / a.n_relations
    print(f"scoreable abstracts (at least one non-rare claim): {len(a):,} "
          f"of {len(pool):,}")
    assert len(a) > 3000

    rows = []
    for name, tau in LEVELS:
        k = a[a.rate <= tau + 1e-12]
        rows.append({
            "level": name,
            "tolerated error rate": f"{tau:.0%}",
            "abstracts kept": len(k),
            "abstracts %": round(100 * len(k) / len(a), 1),
            "papers kept": k.paper_id.nunique(),
            "papers %": round(100 * k.paper_id.nunique() / a.paper_id.nunique(), 1),
            "paper-generator pairs kept": k.groupby(["model", "paper_id"]).ngroups,
            "abstracts per kept paper": round(len(k) / max(k.paper_id.nunique(), 1), 2),
        })
    t = pd.DataFrame(rows)
    print("\nAcceptance ladder, total and deduplicated:")
    print(t.to_string(index=False))

    # Same thing restricted to the train split, which is what Chris trains on.
    tr = a[a.split == "train"]
    rows = []
    for name, tau in LEVELS:
        k = tr[tr.rate <= tau + 1e-12]
        rows.append({"level": name, "train abstracts kept": len(k),
                     "train papers kept": k.paper_id.nunique()})
    t_train = pd.DataFrame(rows)
    print("\nTrain split only (what the relation classifier is trained on):")
    print(t_train.to_string(index=False))

    return t, {"pool_abstracts": int(len(pool)),
               "pool_papers": int(pool.paper_id.nunique()),
               "scoreable_abstracts": int(len(a)),
               "ladder": t.to_dict("records"),
               "ladder_train": t_train.to_dict("records")}


if __name__ == "__main__":
    dup = duplicate_check()
    tbl, lad = ladder()
    out = {"duplicate_check": dup, **lad}
    (HERE / "task2_stats.json").write_text(json.dumps(out, indent=2) + "\n")
    tbl.to_csv(HERE / "task2_ladder.csv", index=False)
    print(f"\nwrote task2_stats.json, task2_ladder.csv")
