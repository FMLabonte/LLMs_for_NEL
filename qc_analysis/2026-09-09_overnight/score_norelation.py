"""Task 3, step 2: score the implicit NoRelation claims with the run-2 QC model.

Same checkpoint, same prompt shape and same max_length as qc_filtering/filter_synthetic.py,
so the NoRelation scores sit in the same space as the positive-claim scores already in
synthetic_relation_scores.csv.

Device: MPS. Checked against CPU on 414 claims before this was run, max absolute
probability difference 3.6e-05 and 100% agreement at the 0.5 cut-off, and the CPU path
itself reproduces the shipped synthetic_relation_scores.csv to 2.4e-06.

Run:
    python score_norelation.py
    python score_norelation.py --limit 2000     # smoke test
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent / "qc_model_run2" / "qc-pubmedbert-final"
PAIRS = HERE / "norelation_pairs.parquet"
OUT = HERE / "norelation_scores.csv"
MAX_LEN = 512

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--device", default="mps")
args = ap.parse_args()

df = pd.read_parquet(PAIRS)
if args.limit:
    df = df.iloc[: args.limit].copy()
print(f"{len(df):,} implicit NoRelation claims, "
      f"{df.drop_duplicates(['model','split','paper_id','generation']).shape[0]:,} abstracts",
      flush=True)
assert len(df) > 0, "empty pair file"

texts = ("Relation: " + df.entity_a + " -> NoRelation -> " + df.entity_b
         + "\nContext: " + df.abstract).tolist()
df = df.drop(columns=["abstract"])

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(args.device).eval()

# Sort by length so a batch is not padded out to the longest abstract in the corpus.
# The order is restored before writing.
#
# Padding then goes to a BUCKET, the next multiple of 64, not to the longest member of
# the batch. With per-batch padding every batch has its own tensor shape, and the MPS
# caching allocator keeps a separate cached buffer for each one. On the first attempt
# that grew to 17 GB and the machine started swapping, which is what made the run get
# slower and slower rather than faster. Eight fixed shapes cost a little wasted compute
# and keep memory flat.
BUCKET = 64
probs = np.empty(len(texts), dtype=np.float64)
order = np.argsort([len(t) for t in texts])
t0 = time.time()
with torch.no_grad():
    for i in range(0, len(order), args.batch):
        idx = order[i:i + args.batch]
        enc = tok([texts[j] for j in idx], truncation=True, max_length=MAX_LEN,
                  padding=True, return_tensors="pt")
        n = enc["input_ids"].shape[1]
        pad_to = min(MAX_LEN, ((n + BUCKET - 1) // BUCKET) * BUCKET)
        if pad_to > n:
            enc = tok([texts[j] for j in idx], truncation=True, max_length=pad_to,
                      padding="max_length", return_tensors="pt")
        enc = {k: v.to(args.device) for k, v in enc.items()}
        p = torch.softmax(model(**enc).logits.float(), dim=-1)[:, 1].cpu().numpy()
        probs[idx] = p
        if (i // args.batch) % 50 == 0:
            el = time.time() - t0
            done = i + len(idx)
            rate = done / max(el, 1e-9)
            print(f"{done:,}/{len(texts):,}  seq={pad_to}  {rate:.1f} ex/s  "
                  f"~{(len(texts)-done)/max(rate,1e-9)/60:.0f} min left", flush=True)

df["prob_supported"] = probs
df.to_csv(OUT, index=False)
print(f"\nwrote {OUT}  rows={len(df):,}  elapsed={(time.time()-t0)/60:.1f} min")
print(f"rejected (prob<0.5, i.e. QC says a relation IS asserted): "
      f"{(df.prob_supported < 0.5).mean():.4f}")
