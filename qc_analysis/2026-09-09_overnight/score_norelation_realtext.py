"""Task 3, the control run: score the SAME implicit NoRelation pairs against the REAL
BioRED abstract.

The perturbed test set only contains 1,131 NoRelation golds, because the builder caps
them to match the number of real relations, and only 380 of those also appear as
implicit pairs in a generation prompt. 380 pairs is a thin control for the central
claim of task 3.

So the control is built directly instead. Every distinct (paper, entity pair) among the
implicit NoRelation pairs of the BioRED TEST split is scored a second time, with the
real abstract of that paper as the context instead of the generated one. Same model,
same prompt shape, same pairs, same ground truth. The only thing that differs is who
wrote the text, which is exactly the comparison Fred asked for.

The test split is used rather than train because the QC model was fine-tuned on BioRED
train, so a real train abstract would be seen data.

Run:
    python score_norelation_realtext.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent / "LLMs_for_NEL"
sys.path.insert(0, str(REPO))
from pubtator_parser import parse_pubtator  # noqa: E402

MODEL_DIR = HERE.parent / "qc_model_run2" / "qc-pubmedbert-final"
PAIRS = HERE / "norelation_pairs.parquet"
OUT = HERE / "norelation_realtext_scores.csv"
MAX_LEN, BUCKET, BATCH, DEVICE = 512, 64, 64, "mps"

meta, _, _ = parse_pubtator(REPO / "Data" / "BioRED" / "Test.PubTator")
real_text = {str(r.pmid): f"{r.title} {r.abstract}".strip()
             for r in meta.itertuples()}
print(f"real BioRED test abstracts: {len(real_text)}")

df = pd.read_parquet(PAIRS)
df = df[df.split == "test"].copy()
# One row per (paper, entity pair). The generation column is irrelevant here: the
# context is the real abstract, which does not depend on which generation it was.
df = df.drop_duplicates(["paper_id", "entity_a_id", "entity_b_id"]).reset_index(drop=True)
print(f"distinct implicit NoRelation pairs on the test split: {len(df):,} "
      f"over {df.paper_id.nunique()} papers")
missing = ~df.paper_id.isin(real_text)
assert not missing.any(), f"{int(missing.sum())} papers have no real abstract"

texts = [f"Relation: {a} -> NoRelation -> {b}\nContext: {real_text[p]}"
         for a, b, p in zip(df.entity_a, df.entity_b, df.paper_id)]
df = df.drop(columns=["abstract"])
df["real_abstract_words"] = [len(real_text[p].split()) for p in df.paper_id]

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE).eval()

order = np.argsort([len(t) for t in texts])
probs = np.empty(len(texts), dtype=np.float64)
t0 = time.time()
with torch.no_grad():
    for i in range(0, len(order), BATCH):
        idx = order[i:i + BATCH]
        enc = tok([texts[j] for j in idx], truncation=True, max_length=MAX_LEN,
                  padding=True, return_tensors="pt")
        n = enc["input_ids"].shape[1]
        pad_to = min(MAX_LEN, ((n + BUCKET - 1) // BUCKET) * BUCKET)
        if pad_to > n:
            enc = tok([texts[j] for j in idx], truncation=True, max_length=pad_to,
                      padding="max_length", return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        probs[idx] = torch.softmax(model(**enc).logits.float(), -1)[:, 1].cpu().numpy()
        if (i // BATCH) % 10 == 0:
            el = time.time() - t0
            print(f"{i+len(idx):,}/{len(texts):,}  {(i+len(idx))/max(el,1e-9):.1f} ex/s",
                  flush=True)

df["prob_supported_real"] = probs
lower = {p: t.lower() for p, t in real_text.items()}
df["both_mentioned_real"] = [
    (str(a).lower() in lower[p]) and (str(b).lower() in lower[p])
    for a, b, p in zip(df.entity_a, df.entity_b, df.paper_id)]
df.to_csv(OUT, index=False)
print(f"\nwrote {OUT}  rows={len(df):,}  elapsed={(time.time()-t0)/60:.1f} min")
print(f"rejected against the real abstract: {(df.prob_supported_real < 0.5).mean():.4f}")
