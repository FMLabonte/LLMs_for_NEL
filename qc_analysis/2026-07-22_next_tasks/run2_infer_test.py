"""Local CPU inference: run-2 QC model over the test set -> per-row supported prob.
Reproduces the published run-2 test numbers as a sanity check, then writes a small
predictions CSV that tasks 1-3 consume. No Colab, no retraining."""
import os, time, argparse
import pandas as pd, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

BASE      = os.environ.get("NLP_LAB_WORK", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = f"{BASE}/qc_model_run2/qc-pubmedbert-final"
TEST_CSV  = f"{BASE}/2026-07-08_meeting/data/qc_test.csv"
OUT_CSV   = f"{BASE}/2026-07-22_next_tasks/qc_run2_test_predictions.csv"
MAXLEN    = 512   # MUST match the run-2 notebook (MAX_LEN=512); 256 over-truncates long abstracts

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0)   # 0 = full test set
ap.add_argument("--batch", type=int, default=32)
args = ap.parse_args()

torch.set_num_threads(os.cpu_count() or 4)
df = pd.read_csv(TEST_CSV).reset_index(drop=True)
if args.limit:
    df = df.iloc[:args.limit].copy()

def make_prompt(r):
    return (f"Relation: {r.entity_a_text} -> {r.relation_type} -> {r.entity_b_text}\n"
            f"Context: {r.abstract}")
df["text"] = df.apply(make_prompt, axis=1)

tok   = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

probs, t0 = [], time.time()
with torch.no_grad():
    for i in range(0, len(df), args.batch):
        batch = df["text"].iloc[i:i+args.batch].tolist()
        enc = tok(batch, truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")
        logits = model(**enc).logits
        probs.extend(torch.softmax(logits, dim=-1)[:, 1].numpy().tolist())
        if (i // args.batch) % 20 == 0:
            el = time.time() - t0; done = i + len(batch)
            print(f"{done}/{len(df)}  {done/max(el,1e-9):.1f} ex/s", flush=True)
df["prob"] = probs

pred = (df["prob"] >= 0.5).astype(int)
pr, rc, f1, _ = precision_recall_fscore_support(df["label"], pred, average="binary", pos_label=1, zero_division=0)
print(f"\nTEST @0.5  P={pr:.4f} R={rc:.4f} F1={f1:.4f}   (published run2: 0.684/0.730/0.706)")
print("confusion [[TN,FP],[FN,TP]] =", confusion_matrix(df['label'], pred).tolist())

cols = ["pmid", "relation_type", "entity_a_id", "entity_b_id", "label", "perturbation", "prob"]
df[cols].to_csv(OUT_CSV, index=False)
print(f"wrote {OUT_CSV}  rows={len(df)}  elapsed={time.time()-t0:.0f}s")
