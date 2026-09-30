"""Is the NoRelation-aware filter worth a training run, and is there a fairer variant?

`qc_filtering/filter_norel_aware.py` folds the implicit NoRelation pairs into the
rejection rule and keeps an abstract only when no claim fails. That drops the keep rate
from 30.5% to 17.3%, and every one of the 468 newly failing abstracts fails purely on
NoRelation. The README there argues the set is close to a size filter. This script tests
that argument properly and prices the alternatives, because the answer decides whether
Christoph spends a training run on the set.

All numbers here are QC APPLICATION: synthetic abstracts, no labels, so a claim that
fails is a rejection and never an error.

Four questions:

  1. are the per-abstract implicit failure counts just binomial noise at the model's own
     false-alarm rate, as the README's 0.85^k argument assumes?
  2. if not, does the surviving signal live in the generated TEXT or in the PAPER?
  3. does the implicit signal agree with the stated-claim signal, which is the only
     proxy for generation quality we have?
  4. what do the fairer variants keep, and do any of them break the size bias?

Run:
    /opt/homebrew/Caskroom/miniconda/base/bin/python norel_variants.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

HERE = Path(__file__).resolve().parent
LAB = HERE.parent.parent
QC = LAB / "LLMs_for_NEL" / "qc_filtering"
OVERNIGHT = LAB / "work" / "2026-09-09_overnight"

CUTOFF = 0.5
KEY = ["model", "split", "paper_id", "generation"]


def load():
    d = pd.read_csv(QC / "abstract_decisions_norel.csv", dtype={"paper_id": str})
    # The CSV shipped these as columns until 2026-09-13, when it was cut back to
    # `passed` (dynamic) and `passed_strict`. Both are still exactly derivable from
    # the raw counts, so this analysis is unchanged.
    d["keep_stated_only"] = d.failed_stated == 0
    d["keep_norel_aware"] = d.passed_strict
    neg = pd.read_csv(OVERNIGHT / "norelation_scores.csv", dtype={"paper_id": str})
    neg["failed"] = neg.prob_supported < CUTOFF
    real = pd.read_csv(OVERNIGHT / "norelation_realtext_scores.csv", dtype={"paper_id": str})
    real["failed_real"] = real.prob_supported_real < CUTOFF
    for c in ("entity_a_id", "entity_b_id"):
        neg[c] = neg[c].astype(str)
        real[c] = real[c].astype(str)
    return d, neg, real


def q1_overdispersion(d, out):
    print("=" * 78)
    print("1. Is the implicit failure count binomial noise at the model's own rate?")
    print("=" * 78)
    g = d[d.n_implicit > 0]
    obs, n_i = g.failed_implicit.values, g.n_implicit.values
    p = g.failed_implicit.sum() / g.n_implicit.sum()
    exp = n_i * p
    var = n_i * p * (1 - p)
    chi2 = ((obs - exp) ** 2 / np.maximum(var, 1e-9)).sum()
    dof = len(g) - 1
    rho_num = ((obs - exp) ** 2).sum() - var.sum()
    rho = rho_num / ((n_i * (n_i - 1)).sum() * p * (1 - p))

    print(f"pooled implicit rejection rate p = {p:.4f} over {g.n_implicit.sum():,} pairs")
    print(f"dispersion chi2/dof = {chi2/dof:.2f}   (1.00 would be pure binomial noise)")
    print(f"beta-binomial rho   = {rho:.4f}   (0 would be no between-abstract variation)")
    print("\nsurvival by size bucket, observed against the binomial prediction:")
    print(f"{'k implicit':>12} {'n abs':>7} {'survived':>9} {'observed':>9} {'binomial':>9}")
    for lo, hi in [(1, 1), (2, 3), (4, 6), (7, 10), (11, 16), (17, 25), (26, 40), (41, 10**9)]:
        m = g[(g.n_implicit >= lo) & (g.n_implicit <= hi)]
        if m.empty:
            continue
        surv = int((m.failed_implicit == 0).sum())
        lab = f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"
        print(f"{lab:>12} {len(m):>7} {surv:>9} {surv/len(m):>9.3f} "
              f"{((1 - p) ** m.n_implicit).mean():>9.3f}")
    print("\nREADING: chi2/dof well above 1 and large abstracts surviving far more often")
    print("than the binomial predicts. The failures are NOT pure coin flipping, so the")
    print("README's 0.85^k argument is not the whole mechanism. Question 2 asks where")
    print("that real signal actually lives.")
    out["pooled_implicit_rejection_rate"] = round(float(p), 4)
    out["dispersion_chi2_per_dof"] = round(float(chi2 / dof), 2)
    out["betabinomial_rho"] = round(float(rho), 4)


def q2_signal_location(d, neg, real, out):
    print("\n" + "=" * 78)
    print("2. Does the signal live in the generated text or in the paper?")
    print("=" * 78)

    # (a) the three generations of one paper share a prompt and an entity set but are
    # different texts. Agreement between them is agreement about the paper.
    g = neg.groupby(KEY).failed.agg(["sum", "size"]).reset_index()
    g.columns = KEY + ["fail", "n"]
    g["rate"] = g.fail / g.n
    trio = g.groupby(["model", "split", "paper_id"]).filter(lambda x: len(x) == 3)
    grp = trio.groupby(["model", "split", "paper_id"])
    ms_b = grp.rate.mean().var(ddof=1) * 3
    within = grp.rate.transform(lambda x: x - x.mean())
    ms_w = (within ** 2).sum() / (len(trio) - grp.ngroups)
    icc = (ms_b - ms_w) / (ms_b + 2 * ms_w)
    print(f"papers with all three generations scored: {grp.ngroups:,}")
    print(f"ICC of the implicit rejection rate across the three generations: {icc:.3f}")
    print("  1.0 would mean entirely a property of the paper, 0 of the generated text.")

    # (b) the same pairs scored against the real BioRED abstract, where by construction
    # the generator invented nothing.
    pk = KEY + ["entity_a_id", "entity_b_id"]
    m = real.merge(neg[pk + ["failed"]].rename(columns={"failed": "failed_gen"}),
                   on=pk, how="inner")
    b = int((m.failed_gen & ~m.failed_real).sum())
    c = int((~m.failed_gen & m.failed_real).sum())
    mcnemar = (abs(b - c) - 1) ** 2 / (b + c) if b + c else float("nan")
    print(f"\npaired control: {len(m):,} pairs over {m.groupby(KEY).ngroups} abstracts, test split")
    print(f"  rejected against REAL BioRED text : {m.failed_real.mean():.4f}")
    print(f"  rejected against GENERATED text   : {m.failed_gen.mean():.4f}")
    print(f"  discordant: generated-only {b}, real-only {c}")
    print(f"  McNemar chi2 = {mcnemar:.2f}   (3.84 would be p = 0.05)")
    both = int((m.failed_gen & m.failed_real).sum())
    print(f"  {both} of {int(m.failed_gen.sum())} generated-text flags "
          f"({both/int(m.failed_gen.sum())*100:.1f}%) also fire on the real abstract")

    ab = m.groupby(KEY).agg(fg=("failed_gen", "sum"), fr=("failed_real", "sum")).reset_index()
    sr, sg = float((ab.fr == 0).mean()), float((ab.fg == 0).mean())
    print(f"\nunder the shipped zero-failure rule, on those {len(ab)} abstracts:")
    print(f"  the REAL BioRED abstract would survive : {int((ab.fr==0).sum())} ({sr*100:.1f}%)")
    print(f"  the generated abstract survives        : {int((ab.fg==0).sum())} ({sg*100:.1f}%)")
    print("\nREADING: the rule rejects text that cannot contain an invented relation at")
    print("the same rate as text that can. Whatever it selects on, it is not whether the")
    print("generator hallucinated.")
    out["icc_across_generations"] = round(float(icc), 3)
    out["paired_real_reject_rate"] = round(float(m.failed_real.mean()), 4)
    out["paired_gen_reject_rate"] = round(float(m.failed_gen.mean()), 4)
    out["mcnemar_chi2"] = round(float(mcnemar), 2)
    out["paired_survival_real"] = round(sr, 3)
    out["paired_survival_gen"] = round(sg, 3)
    return m


def q3_agreement(d, out):
    print("\n" + "=" * 78)
    print("3. Does the implicit signal agree with the stated-claim signal?")
    print("=" * 78)
    m = d[(d.n_implicit > 0) & (d.n_stated > 0)].copy()
    m["rate_i"] = m.failed_implicit / m.n_implicit
    m["rate_s"] = m.failed_stated / m.n_stated
    r, p = st.spearmanr(m.rate_i, m.rate_s)

    def resid(y, x):
        return y - np.poly1d(np.polyfit(x, y, 1))(x)

    rp, pp = st.spearmanr(resid(m.rate_i, np.log1p(m.n_implicit)),
                          resid(m.rate_s, np.log1p(m.n_stated)))
    print(f"spearman(implicit rate, stated rate), {len(m):,} abstracts : r={r:+.3f} p={p:.2g}")
    print(f"same, both residualised on log size                       : r={rp:+.3f} p={pp:.2g}")
    print("\nREADING: an abstract that overstates the relations it was asked for should")
    print("also be the one that invents relations it was not asked for. There is no such")
    print("relationship, which is what we would expect if the implicit flags are noise")
    print("plus paper identity.")
    out["spearman_implicit_vs_stated"] = round(float(r), 3)
    out["spearman_implicit_vs_stated_partial"] = round(float(rp), 3)


def q4_variants(d, paired, out):
    print("\n" + "=" * 78)
    print("4. The variants")
    print("=" * 78)
    p = d[d.n_implicit > 0].failed_implicit.sum() / d[d.n_implicit > 0].n_implicit.sum()
    d = d.copy()
    d["rate_implicit"] = np.where(d.n_implicit > 0, d.failed_implicit / d.n_implicit, 0.0)
    d["binom_tail"] = [1 - st.binom.cdf(f - 1, n, p) if n > 0 and f > 0 else 1.0
                       for f, n in zip(d.failed_implicit, d.n_implicit)]

    rows = []

    def add(name, mask):
        r = st.pointbiserialr(mask.astype(int), np.log1p(d.n_implicit))[0]
        kept = d[mask]
        rows.append({
            "rule": name, "kept": int(mask.sum()), "pct": round(mask.mean() * 100, 1),
            "median_k": int(kept.n_implicit.median()),
            "k_zero": int((kept.n_implicit == 0).sum()),
            "qwen8b_train": int(((kept.model == "qwen3_8b") & (kept.split == "train")).sum()),
            "r_keep_logk": round(float(r), 3),
        })

    add("A stated only (status quo, ../filtered)", d.keep_stated_only)
    add("B zero implicit failures (built)", d.keep_norel_aware)
    for t in (0.20, 0.30, 0.40, 0.50):
        add(f"C proportional, implicit rate <= {t:.2f}",
            d.keep_stated_only & (d.rate_implicit <= t))
    for a in (0.05, 0.10, 0.20):
        add(f"D binomial tail p > {a:.2f}", d.keep_stated_only & (d.binom_tail > a))
    for per in (10, 7, 5):
        add(f"E allowance, failures <= ceil(k/{per})",
            d.keep_stated_only & (d.failed_implicit <= np.ceil(d.n_implicit / per)))

    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    print("\nREADING: r(keep, log k) is the size bias. The status-quo filter already sits")
    print("at -0.47, so the built rule at -0.52 sharpens a bias it did not create. No")
    print("proportional, tail or allowance variant moves it: they all still score a")
    print("quantity dominated by how many entity pairs the paper happens to have.")

    # The one construction that changes what is being measured: a pair counts against
    # the generation only when the real abstract for the same paper passes it.
    print("\n" + "-" * 78)
    print("F the paired rule: a pair vetoes only when generated fails and real passes")
    print("-" * 78)
    paired["gen_only"] = paired.failed_gen & ~paired.failed_real
    ab = paired.groupby(KEY).gen_only.sum().rename("gen_only").reset_index()
    sub = d.merge(ab, on=KEY, how="inner")
    fair = sub.keep_stated_only & (sub.gen_only == 0)
    veto = float(paired.gen_only.mean())
    print(f"on the {len(sub)} paired abstracts:")
    for name, mask in [("stated only", sub.keep_stated_only),
                       ("built rule", sub.keep_norel_aware),
                       ("paired rule", fair)]:
        r = st.pointbiserialr(mask.astype(int), np.log1p(sub.n_implicit))[0]
        print(f"  {name:<14} keeps {int(mask.sum()):>3} ({mask.mean()*100:>4.1f}%)  "
              f"median k {sub[mask].n_implicit.median():>3.0f}  r(keep,log k) {r:+.3f}")
    keep_st = int(sub.keep_stated_only.sum())
    print(f"\npass-through of the {keep_st} stated-only survivors:")
    print(f"  built rule  {int((sub.keep_norel_aware & sub.keep_stated_only).sum()):>3} "
          f"({(sub.keep_norel_aware & sub.keep_stated_only).sum()/keep_st*100:.0f}%)")
    print(f"  paired rule {int(fair.sum()):>3} ({fair.sum()/keep_st*100:.0f}%)")
    print(f"\nper-pair veto rate: built {paired.failed_gen.mean():.4f} -> paired {veto:.4f}")
    print(f"implied survival: built {1-paired.failed_gen.mean():.3f}^k -> paired {1-veto:.3f}^k")
    for k in (1, 5, 10, 20, 32):
        print(f"    k={k:>3}: built {(1-paired.failed_gen.mean())**k:.3f}   paired {(1-veto)**k:.3f}")

    st_only = d[d.keep_stated_only]
    proj = float(((1 - veto) ** st_only.n_implicit).sum())
    print(f"\ncorpus-wide projection, independent-Bernoulli estimate at the paired rate:")
    print(f"  stated-only survivors     : {len(st_only):,}")
    print(f"  built rule keeps          : {int(d.keep_norel_aware.sum()):,}")
    print(f"  paired rule, expected     : ~{proj:,.0f} (overdispersion pushes this higher)")
    for mdl in ("qwen3_8b", "qwen3_4b"):
        s = st_only[(st_only.model == mdl) & (st_only.split == "train")]
        built = int(d[(d.model == mdl) & (d.split == "train")].keep_norel_aware.sum())
        print(f"  {mdl} train: {len(s)} stated-only -> "
              f"~{((1-veto)**s.n_implicit).sum():.0f} paired vs {built} built")
    print("\nREADING: the paired rule is the only variant that changes what is measured")
    print("rather than how much of it is tolerated. It costs 12% of the stated-only")
    print("survivors instead of 29%. It needs the real-text scoring run extended from")
    print("3,390 pairs to all 104,136, which is one more overnight run of")
    print("work/2026-09-09_overnight/score_norelation_realtext.py.")

    out["variants"] = rows
    out["paired_veto_rate"] = round(veto, 4)
    out["paired_rule_projected_keep"] = int(round(proj))
    return t


def main():
    d, neg, real = load()
    out: dict = {"stage": "QC application", "cutoff": CUTOFF, "abstracts": int(len(d))}
    print(f"abstracts: {len(d):,}   stated-only keeps {int(d.keep_stated_only.sum()):,}   "
          f"built norel rule keeps {int(d.keep_norel_aware.sum()):,}\n")
    q1_overdispersion(d, out)
    paired = q2_signal_location(d, neg, real, out)
    q3_agreement(d, out)
    table = q4_variants(d, paired, out)
    table.to_csv(HERE / "variants.csv", index=False)
    (HERE / "stats.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwritten: {HERE/'variants.csv'} and {HERE/'stats.json'}")


if __name__ == "__main__":
    main()
