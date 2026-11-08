import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, ConfusionMatrixDisplay

TEXT, LABEL = "utterance", "intentClass"
TOXIC = {"E", "I"}

tr = pd.read_csv("train.csv")
te = pd.read_csv("valid.csv")  # held-out test
for d in (tr, te):
    d[TEXT] = d[TEXT].astype(str)

gss = GroupShuffleSplit(n_splits=1, test_size=0.12, random_state=42)
tr_idx, dev_idx = next(gss.split(tr, groups=tr["matchId"]))
train, dev = tr.iloc[tr_idx], tr.iloc[dev_idx]
print("train/dev/test sizes:", len(train), len(dev), len(te))

# ---------- Exploration ----------
print("\nClass distribution (train):")
print(train[LABEL].value_counts(normalize=True).round(3))
lengths = train[TEXT].str.split().str.len()
print(f"\nWords per message: mean {lengths.mean():.1f}, median {lengths.median():.0f}, max {lengths.max()}")

# ---------- Features ----------
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
X_train = vec.fit_transform(train[TEXT])

# ---------- 4-class model ----------
clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(X_train, train[LABEL])

# ---------- Binary model ----------
clf_b = LogisticRegression(max_iter=1000, class_weight="balanced")
clf_b.fit(X_train, train[LABEL].isin(TOXIC))

for tag, df in [("dev", dev), ("valid", te)]:
    X = vec.transform(df[TEXT])
    pred = clf.predict(X)
    print(f"\n=== 4-class, {tag} ===")
    print(classification_report(df[LABEL], pred, digits=3))
    print("macro F1:", round(f1_score(df[LABEL], pred, average="macro"), 3))
    ConfusionMatrixDisplay.from_predictions(df[LABEL], pred, xticks_rotation=45)
    plt.tight_layout(); plt.savefig(f"cm_4class_{tag}.png", dpi=200); plt.close()

    pred_b = clf_b.predict(X)
    yb = df[LABEL].isin(TOXIC)
    print(f"\n=== Binary toxic vs non-toxic, {tag} ===")
    print(classification_report(yb, pred_b, target_names=["non-toxic", "toxic"], digits=3))
    ConfusionMatrixDisplay.from_predictions(yb, pred_b, display_labels=["non-toxic", "toxic"])
    plt.tight_layout(); plt.savefig(f"cm_binary_{tag}.png", dpi=200); plt.close()

# ---------- Error analysis (dev set) ----------
pred_dev = clf.predict(vec.transform(dev[TEXT]))
errs = dev.assign(pred=pred_dev)
errs = errs[errs[LABEL] != errs["pred"]]
print(f"\n{len(errs)} errors on dev. Random sample of 25:")
for _, r in errs.sample(25, random_state=0).iterrows():
    print(f"true={r[LABEL]} pred={r['pred']} | {r[TEXT]}")

print("\nImplicit (I) misclassified, sample:")
sub = errs[errs[LABEL] == "I"]
for _, r in sub.sample(min(15, len(sub)), random_state=0).iterrows():
    print(f"pred={r['pred']} | {r[TEXT]}")
