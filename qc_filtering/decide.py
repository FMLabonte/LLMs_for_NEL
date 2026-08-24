"""Task 4, step 2: turn per-relation QC scores into kept/rejected synthetic abstracts.

Separate from filter_synthetic.py on purpose: the scoring costs 46 minutes, the
decision rule costs seconds, and the rule is still being argued about.

Two rules, from the 2026-07-22 threshold work:
  strict   reject the abstract if ANY relation scores below the cut-off
           (Fred's "1 detected error rejects the abstract", 100% of faulty
           abstracts caught, 19% of correct ones kept)
  dynamic  allowed errors scale with the relation count (<=8 -> 0, 9-12 -> 1,
           >12 -> 2), the "free lunch" step function, 24% kept at 99% caught

Two treatments of the five rare relation types (Bind, Cotreatment, Comparison,
Drug_Interaction, Conversion), which the 4-class redesign dropped in June and
which the QC model has therefore never been trained on:
  include  score them like any other claim (they are out of distribution)
  exclude  ignore them when counting errors

Outputs, into filtered/:
  results_qwen3_*_{train,dev,test}.json   Fred's schema, generations pruned to
                                          the ones that passed. Drops straight
                                          into Chris's load_synthetic_abstracts.
  random_qwen3_*_{train,dev,test}.json    same-size random control, seed 42, so
                                          "QC picked well" can be told apart
                                          from "the set got smaller"
  abstract_decisions.csv                  one row per synthetic abstract
  SUMMARY.md                              the numbers, all four rule combos
"""
import argparse, json, random, re, zipfile
from pathlib import Path

import pandas as pd

OUT_DIR = Path(__file__).resolve().parent
SCORES = OUT_DIR / "synthetic_relation_scores.csv"
SYN_DIR = OUT_DIR.parent / "Data/Synthetic abstracts"
FILTERED = OUT_DIR / "filtered"

FOUR_CLASS = {"Association", "Positive_Correlation", "Negative_Correlation", "NoRelation"}
SEED = 42


def allowed_errors(n_relations: int, rule: str) -> int:
    if rule == "strict":
        return 0
    if n_relations <= 8:
        return 0
    if n_relations <= 12:
        return 1
    return 2


def decide(df: pd.DataFrame, rule: str, rares: str, cutoff: float) -> pd.DataFrame:
    """One row per synthetic abstract, with the pass/fail verdict."""
    d = df.copy()
    d["is_rare"] = ~d.relation_type.isin(FOUR_CLASS)
    d["failed"] = d.prob_supported < cutoff
    if rares == "exclude":
        d.loc[d.is_rare, "failed"] = False        # not counted either way

    key = ["model", "split", "paper_id", "generation"]
    out = d.groupby(key).agg(
        n_relations=("rel_idx", "count"),
        n_rare=("is_rare", "sum"),
        n_failed=("failed", "sum"),
        min_prob=("prob_supported", "min"),
        mean_prob=("prob_supported", "mean"),
        num_relations_meta=("num_relations", "first"),
        abstract_words=("abstract_words", "first"),
    ).reset_index()
    out["allowed"] = out.n_relations.map(lambda n: allowed_errors(n, rule))
    out["passed"] = out.n_failed <= out.allowed
    return out


def _ensure_synthetic_dir() -> Path:
    """Unpack Data/Synthetic abstracts.zip next to itself if it has not been."""
    if not SYN_DIR.exists():
        with zipfile.ZipFile(OUT_DIR.parent / "Data/Synthetic abstracts.zip") as z:
            z.extractall(OUT_DIR.parent / "Data")
    return SYN_DIR


def export(decisions: pd.DataFrame, tag: str) -> list[str]:
    """Write Fred-schema JSONs holding only the kept generations, plus a
    same-size random control drawn from the same papers."""
    FILTERED.mkdir(exist_ok=True)
    rng = random.Random(SEED)
    lines = []

    for path in sorted(_ensure_synthetic_dir().glob("results_qwen3_*.json")):
        model, split = re.match(r"results_(qwen3_\d+b)_(\w+)\.json", path.name).groups()
        papers = json.loads(path.read_text())
        sub = decisions[(decisions.model == model) & (decisions.split == split)]
        keep = set(zip(sub[sub.passed].paper_id, sub[sub.passed].generation))
        all_pairs = sorted(zip(sub.paper_id, sub.generation))

        kept_papers, n_kept = [], 0
        for p in papers:
            pid = str(p["paper_id"])
            gens = {k: v for k, v in p["synthetic_abstracts"].items() if (pid, k) in keep}
            if gens:
                kept_papers.append({**p, "synthetic_abstracts": gens})
                n_kept += len(gens)
        (FILTERED / path.name).write_text(json.dumps(kept_papers, indent=1))

        # Random control: same number of abstracts, drawn without regard to QC.
        picked = set(rng.sample(all_pairs, n_kept)) if n_kept <= len(all_pairs) else set(all_pairs)
        rnd_papers = []
        for p in papers:
            pid = str(p["paper_id"])
            gens = {k: v for k, v in p["synthetic_abstracts"].items() if (pid, k) in picked}
            if gens:
                rnd_papers.append({**p, "synthetic_abstracts": gens})
        (FILTERED / path.name.replace("results_", "random_")).write_text(json.dumps(rnd_papers, indent=1))

        total = len(all_pairs)
        lines.append(f"| {model} | {split} | {total} | {n_kept} | {n_kept/total*100:.1f}% | "
                     f"{len(kept_papers)} / {len(papers)} |")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", choices=["strict", "dynamic"], default="dynamic")
    ap.add_argument("--rares", choices=["include", "exclude"], default="exclude")
    ap.add_argument("--cutoff", type=float, default=0.5)
    args = ap.parse_args()

    df = pd.read_csv(SCORES, dtype={"paper_id": str})
    print(f"{len(df):,} scored relation claims\n")

    # All four combinations, so the choice is made on numbers rather than taste.
    grid = []
    for rule in ("strict", "dynamic"):
        for rares in ("include", "exclude"):
            d = decide(df, rule, rares, args.cutoff)
            grid.append({"rule": rule, "rares": rares,
                         "kept": int(d.passed.sum()), "total": len(d),
                         "kept_pct": round(d.passed.mean() * 100, 1)})
    grid_df = pd.DataFrame(grid)
    print(grid_df.to_string(index=False))

    chosen = decide(df, args.rule, args.rares, args.cutoff)
    chosen.to_csv(OUT_DIR / "abstract_decisions.csv", index=False)

    per_split = (chosen.groupby(["model", "split"])
                 .agg(abstracts=("passed", "size"), kept=("passed", "sum"))
                 .assign(kept_pct=lambda x: (x.kept / x.abstracts * 100).round(1)))
    print(f"\nchosen rule: {args.rule} / rares={args.rares} / cutoff={args.cutoff}")
    print(per_split.to_string())

    rows = export(chosen, args.rule)

    bias = chosen.groupby(pd.cut(chosen.n_relations, [0, 3, 8, 12, 100],
                                 labels=["1-3", "4-8", "9-12", "13+"]),
                          observed=True).passed.agg(["size", "sum", "mean"])
    bias["mean"] = (bias["mean"] * 100).round(1)

    summary = [
        "# Task 4, step 1: QC filtering of the synthetic abstracts",
        "",
        f"Run 2 QC model over {len(df):,} relation claims in {len(chosen):,} synthetic "
        f"abstracts (2 generators x 3 splits x 3 generations per paper).",
        "",
        f"Rule shipped: **{args.rule}**, rare types **{args.rares}d**, cut-off {args.cutoff}.",
        "",
        "## Kept per file",
        "",
        "| model | split | abstracts | kept | kept % | papers surviving |",
        "|---|---|---|---|---|---|",
        *rows,
        "",
        "## All four rule combinations",
        "",
        grid_df.to_markdown(index=False),
        "",
        "## Pass rate by relation count (the known bias)",
        "",
        bias.to_markdown(),
        "",
        "Column `mean` is the percentage kept. The 2026-07-22 profiling found the "
        "filter is relation-count biased rather than length biased, and that shows "
        "here too: abstracts asserting many relations rarely survive, because each "
        "claim is another chance to trip the rule.",
    ]
    (OUT_DIR / "SUMMARY.md").write_text("\n".join(summary) + "\n")
    print(f"\nwrote {FILTERED}/ (filtered + random control), abstract_decisions.csv, SUMMARY.md")


if __name__ == "__main__":
    main()
