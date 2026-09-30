"""Is the climb in the error rate the GENERATOR degrading, or the QC MODEL degrading?

task1_driver.py established that the QC model's rejection rate rises with the number of
relations an abstract was asked to express. That measurement is taken on generated text
only, so it cannot separate two explanations:

  a) the generator really does express relations worse when asked for many of them,
  b) the QC model itself gets less reliable on relation-dense papers, whoever wrote
     the text.

The control is the same one task5_matched.py uses. Take relation triples that exist on
both sides, score them against the real BioRED abstract and against the generated
abstract, and bucket by the paper's relation count. If the real-text rejection rate also
climbs, the climb is a property of the papers or of the QC model. If only the generated
side climbs, it is the generator.

Reads the same precomputed score files as task5_matched.py, so no GPU is needed.

Run:
    python task1_control.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent / "LLMs_for_NEL"
WORK = HERE.parent
sys.path.insert(0, str(REPO))
from pubtator_parser import parse_pubtator  # noqa: E402

SYN_POS = REPO / "qc_filtering" / "synthetic_relation_scores.csv"
REAL_PRED = WORK / "2026-07-22_next_tasks" / "qc_run2_test_predictions.csv"
REAL_ROWS = WORK / "2026-07-08_meeting" / "data" / "qc_test.csv"

CUTOFF = 0.5
FOUR = ["Association", "Positive_Correlation", "Negative_Correlation", "NoRelation"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def name_maps() -> dict[str, dict[str, str]]:
    m: dict[str, dict[str, str]] = {}
    for f in ["Train.PubTator", "Dev.PubTator", "Test.PubTator"]:
        _, anns, _ = parse_pubtator(REPO / "Data" / "BioRED" / f)
        for pmid, g in anns.groupby("pmid"):
            d = m.setdefault(str(pmid), {})
            for _, r in g.iterrows():
                d.setdefault(str(r["mention"]).strip().lower(), str(r["mesh_id"]))
    return m


def main():
    maps = name_maps()

    pred = pd.read_csv(REAL_PRED)
    rows = pd.read_csv(REAL_ROWS)
    assert len(pred) == len(rows) == 15199
    real = rows.copy()
    real["prob"] = pred["prob"].values
    real["rejected"] = real.prob < CUTOFF
    gold = real[real.label == 1].copy()
    gold["key"] = [f"{p}|{'|'.join(sorted([str(a), str(b)]))}|{r}"
                   for p, a, b, r in zip(gold.pmid, gold.entity_a_id,
                                         gold.entity_b_id, gold.relation_type)]

    syn = pd.read_csv(SYN_POS, dtype={"paper_id": str})
    syn = syn[(syn.split == "test") & (syn.relation_type.isin(FOUR))].copy()

    def to_id(pid, name):
        return maps.get(str(pid), {}).get(str(name).strip().lower())

    syn["id_a"] = [to_id(p, n) for p, n in zip(syn.paper_id, syn.entity_a)]
    syn["id_b"] = [to_id(p, n) for p, n in zip(syn.paper_id, syn.entity_b)]
    syn = syn[~(syn.id_a.isna() | syn.id_b.isna())].copy()
    syn["key"] = [f"{p}|{'|'.join(sorted([a, b]))}|{r}"
                  for p, a, b, r in zip(syn.paper_id, syn.id_a, syn.id_b,
                                        syn.relation_type)]
    syn["rejected"] = syn.prob_supported < CUTOFF

    shared = set(gold.key) & set(syn.key)
    g = gold[gold.key.isin(shared)].copy()
    s = syn[syn.key.isin(shared)].copy()
    print(f"matched triples: {len(shared):,}  real rows {len(g):,}  "
          f"synthetic rows {len(s):,}  papers {g.pmid.nunique()}")

    # The relation count is a property of the paper's generation prompt.
    relcount = s.groupby("paper_id").num_relations.first()
    key2paper = dict(zip(s.key, s.paper_id))
    g["paper_id"] = g.key.map(key2paper)
    g["num_relations"] = g.paper_id.map(relcount)
    s["num_relations"] = s.paper_id.map(relcount)
    g = g[g.num_relations.notna()].copy()

    edges = [0, 3, 7, 13, 10_000]
    labels = ["1-3", "4-7", "8-13", "14+"]
    for df in (g, s):
        df["bucket"] = pd.cut(df.num_relations, bins=edges, labels=labels)

    print("\nSame triples, bucketed by how many relations the paper was asked for.")
    print(f"{'relations':>10} | {'real text':>22} | {'generated text':>22} | {'diff':>7}")
    print("-" * 70)
    out = []
    for b in labels:
        gb, sb = g[g.bucket == b], s[s.bucket == b]
        if len(gb) == 0 or len(sb) == 0:
            continue
        gr, sr = gb.rejected.mean(), sb.rejected.mean()
        glo, ghi = wilson(gb.rejected.sum(), len(gb))
        slo, shi = wilson(sb.rejected.sum(), len(sb))
        print(f"{b:>10} | {gr:6.3f} [{glo:.3f},{ghi:.3f}] n={len(gb):<4} | "
              f"{sr:6.3f} [{slo:.3f},{shi:.3f}] n={len(sb):<5} | {sr-gr:+7.3f}")
        out.append({"bucket": b, "real_n": len(gb), "real_rate": gr,
                    "syn_n": len(sb), "syn_rate": sr, "diff": sr - gr})

    t = pd.DataFrame(out)
    t.to_csv(HERE / "task1_control.csv", index=False)

    swing_real = t.real_rate.max() - t.real_rate.min()
    swing_syn = t.syn_rate.max() - t.syn_rate.min()
    print("-" * 70)
    print(f"swing across buckets, real text      : {swing_real:+.3f}")
    print(f"swing across buckets, generated text : {swing_syn:+.3f}")
    print(f"\ncorrelation with relation count, real text     : "
          f"{np.corrcoef(g.num_relations, g.rejected.astype(float))[0,1]:+.3f}")
    print(f"correlation with relation count, generated text: "
          f"{np.corrcoef(s.num_relations, s.rejected.astype(float))[0,1]:+.3f}")


if __name__ == "__main__":
    main()
