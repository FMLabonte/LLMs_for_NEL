"""Task 2 (Fred's #2): length-scaled dynamic rejection threshold + grid search.

Fixed rule: reject an abstract if >= N of its claims are flagged, with N fixed.
Dynamic rule: N scales with how many relations the abstract presents. Short lists stay
strict (0 errors allowed), longer lists tolerate 1 then 2. Rationale: more relations =
more chances for a single false rejection, so leniency is affordable there.

Step function over the number of claimed relations c:
    c <= t1        -> allow 0  (reject if >= 1)
    t1 < c <= t2   -> allow 1  (reject if >= 2)
    c >  t2        -> allow 2  (reject if >= 3)
Grid-search (t1, t2). Goal: correct-kept UP while faulty-caught stays ~100%.
No retraining ('free lunch'): purely a decision rule over the same predictions.
"""
import pandas as pd, numpy as np, random, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.environ.get("NLP_LAB_WORK", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRED = f"{BASE}/2026-07-22_next_tasks/qc_run2_test_predictions.csv"
FIGDIR = f"{BASE}/2026-07-22_next_tasks/figures"
REAL = ["Association", "Positive_Correlation", "Negative_Correlation"]
THR = 0.5
os.makedirs(FIGDIR, exist_ok=True)

p = pd.read_csv(PRED)
p["pred"] = (p["prob"] >= THR).astype(int)

# correct abstracts: real-relation golds per pmid (matches run 2)
gold = p[(p["label"] == 1) & (p["relation_type"].isin(REAL))]
per = gold.groupby("pmid")["pred"].agg(n="size", detected=lambda s: int((s == 0).sum()))

# faulty abstracts: correct + 1 injected error (label==0 row, same pmid), seed 42
rng = random.Random(42)
neg_by_pmid = {pmid: g.index.tolist() for pmid, g in p[p["label"] == 0].groupby("pmid")}
faulty = []  # (n_claims, detected)
for pmid, g in gold.groupby("pmid"):
    if pmid not in neg_by_pmid:
        continue
    inj = rng.choice(neg_by_pmid[pmid])
    det = int((g["pred"] == 0).sum()) + int(p.loc[inj, "pred"] == 0)
    faulty.append((len(g) + 1, det))
faulty = pd.DataFrame(faulty, columns=["n", "detected"])

def allowed(c, t1, t2):           # errors tolerated for an abstract with c claims
    return 0 if c <= t1 else 1 if c <= t2 else 2

def evaluate(t1, t2):
    ck = np.mean([d < 1 + allowed(c, t1, t2) for c, d in zip(per["n"], per["detected"])])
    fc = np.mean([d >= 1 + allowed(c, t1, t2) for c, d in zip(faulty["n"], faulty["detected"])])
    return ck, fc

# baseline: fixed reject-if->=1 everywhere (t1,t2 huge)
base_ck, base_fc = evaluate(10**9, 10**9)
print(f"baseline (fixed reject-if->=1): correct-kept={base_ck:.3f}  faulty-caught={base_fc:.3f}")
print(f"  (run 2 reported 0.190 / 1.000)\n")

# grid search
rows = []
for t1 in range(2, 12):
    for t2 in range(t1, 16):
        ck, fc = evaluate(t1, t2)
        rows.append((t1, t2, ck, fc))
grid = pd.DataFrame(rows, columns=["t1", "t2", "correct_kept", "faulty_caught"])

for floor in [1.00, 0.99, 0.98, 0.95]:
    ok = grid[grid["faulty_caught"] >= floor]
    if len(ok):
        b = ok.sort_values("correct_kept", ascending=False).iloc[0]
        print(f"best with faulty-caught >= {floor:.2f}: "
              f"t1={int(b.t1)} t2={int(b.t2)}  correct-kept={b.correct_kept:.3f}  "
              f"faulty-caught={b.faulty_caught:.3f}  (+{100*(b.correct_kept-base_ck):.0f} pts kept)")

# figure: baseline vs a chosen dynamic operating point (>=99% faulty caught)
ok99 = grid[grid["faulty_caught"] >= 0.99].sort_values("correct_kept", ascending=False)
chosen = ok99.iloc[0] if len(ok99) else grid.sort_values("correct_kept", ascending=False).iloc[0]
labels = ["correct abstracts kept", "faulty abstracts caught"]
base_v = [base_ck * 100, base_fc * 100]
dyn_v = [chosen.correct_kept * 100, chosen.faulty_caught * 100]
x = np.arange(2); w = 0.38
fig, ax = plt.subplots(figsize=(6.4, 3.6))
b1 = ax.bar(x - w/2, base_v, w, label="fixed (reject-if->=1)", color="#9AA7B4")
b2 = ax.bar(x + w/2, dyn_v, w, label=f"dynamic (t1={int(chosen.t1)}, t2={int(chosen.t2)})", color="#2A4D7A")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.5, f"{b.get_height():.0f}%",
                ha="center", va="bottom", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylim(0, 112); ax.set_ylabel("percent (higher = better, both)")
ax.set_title("Dynamic threshold: more correct kept, faulty still caught")
ax.legend(loc="lower center", fontsize=8, framealpha=0.9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(f"{FIGDIR}/task2_dynamic_vs_fixed.png", dpi=150)
print(f"\nwrote {FIGDIR}/task2_dynamic_vs_fixed.png")
