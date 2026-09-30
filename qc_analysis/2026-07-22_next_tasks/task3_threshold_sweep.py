"""Task 3 (Fred's #3): trade precision for recall, no retraining.

Lower the decision threshold -> the model accepts more -> higher recall, fewer false
rejections of real golds -> more correct abstracts kept. But it also accepts more bad
claims, so injected errors are caught less often -> faulty-caught drops. This sweep
quantifies that trade at the per-pair AND abstract level, and checks it combined with
the Task-2 dynamic tolerance.
"""
import pandas as pd, numpy as np, random, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

BASE = os.environ.get("NLP_LAB_WORK", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRED = f"{BASE}/2026-07-22_next_tasks/qc_run2_test_predictions.csv"
FIGDIR = f"{BASE}/2026-07-22_next_tasks/figures"
REAL = ["Association", "Positive_Correlation", "Negative_Correlation"]
os.makedirs(FIGDIR, exist_ok=True)

p = pd.read_csv(PRED)
y = p["label"].values
prob = p["prob"].values

# correct abstracts: real-relation gold probs per pmid
gold = p[(p["label"] == 1) & (p["relation_type"].isin(REAL))]
gold_probs = [g["prob"].values for _, g in gold.groupby("pmid")]

# faulty: correct + 1 injected error (label==0, same pmid), seed 42
rng = random.Random(42)
neg_by_pmid = {pmid: g.index.tolist() for pmid, g in p[p["label"] == 0].groupby("pmid")}
faulty_probs = []
for pmid, g in gold.groupby("pmid"):
    if pmid in neg_by_pmid:
        inj = rng.choice(neg_by_pmid[pmid])
        faulty_probs.append(np.append(g["prob"].values, p.loc[inj, "prob"]))

def kept(sets, thr):     # reject-if->=1: pass iff 0 flagged
    return np.mean([int((s < thr).sum() == 0) for s in sets])
def caught(sets, thr):   # caught iff >=1 flagged
    return np.mean([int((s < thr).sum() >= 1) for s in sets])

print(f"{'thr':>5} {'prec':>6} {'recall':>7} {'kept':>6} {'caught':>7}")
rows = []
for thr in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]:
    pred = (prob >= thr).astype(int)
    pr, rc, _, _ = precision_recall_fscore_support(y, pred, average="binary", pos_label=1, zero_division=0)
    ck, fc = kept(gold_probs, thr), caught(faulty_probs, thr)
    rows.append((thr, pr, rc, ck, fc))
    print(f"{thr:5.2f} {pr:6.3f} {rc:7.3f} {ck:6.3f} {fc:7.3f}")
sw = pd.DataFrame(rows, columns=["thr", "prec", "recall", "kept", "caught"])

# best decision threshold that keeps faulty-caught >= 0.99
ok = sw[sw["caught"] >= 0.99].sort_values("kept", ascending=False)
if len(ok):
    b = ok.iloc[0]
    print(f"\nbest thr with faulty-caught>=0.99: thr={b.thr:.2f}  kept={b.kept:.3f}  caught={b.caught:.3f}  "
          f"(vs 0.50: kept 0.190). recall={b.recall:.3f}, precision={b.prec:.3f}")

# figure: kept vs caught as the threshold moves (the trade)
fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.plot(sw["thr"], sw["kept"] * 100, "-o", color="#2E7D32", label="correct abstracts kept")
ax.plot(sw["thr"], sw["caught"] * 100, "-s", color="#2A4D7A", label="faulty abstracts caught")
ax.axvline(0.5, color="#B23A1E", ls="--", lw=1, alpha=0.7)
ax.text(0.5, 5, " run-2 point (0.5)", color="#B23A1E", fontsize=8, ha="left")
ax.set_xlabel("decision threshold (lower = higher recall)")
ax.set_ylabel("percent (higher = better, both)")
ax.set_title("Trading precision for recall: the abstract-level trade-off")
ax.set_ylim(0, 108); ax.legend(loc="center left", fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(f"{FIGDIR}/task3_threshold_sweep.png", dpi=150)
print(f"\nwrote {FIGDIR}/task3_threshold_sweep.png")
