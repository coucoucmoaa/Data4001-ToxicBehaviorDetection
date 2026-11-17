import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

TEXT, LABEL = "utterance", "intentClass"
TOXIC = {"E", "I"}
SEEDS = [42, 0, 1, 2, 3]
N_BOOT = 1000

tr = pd.read_csv("train.csv")
te = pd.read_csv("valid.csv")  # held-out test
for d in (tr, te):
    d[TEXT] = d[TEXT].fillna("nan").astype(str)  # same as astype(str) in starter.py


def split(seed):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.12, random_state=seed)
    tr_idx, dev_idx = next(gss.split(tr, groups=tr["matchId"]))
    return tr.iloc[tr_idx], tr.iloc[dev_idx]


def train_baseline(train):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X = vec.fit_transform(train[TEXT])
    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced").fit(X, train[LABEL])
    clf_b = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced").fit(X, train[LABEL].isin(TOXIC))
    return vec, clf, clf_b


def subsets(train, dev):
    seen = te["matchId"].isin(set(train["matchId"]))
    return {"dev": dev, "valid": te, "valid seen": te[seen], "valid unseen": te[~seen]}


def evaluate(model, df):
    vec, clf, clf_b = model
    X = vec.transform(df[TEXT])
    y, yb = df[LABEL].to_numpy(), df[LABEL].isin(TOXIC).to_numpy()
    return y, clf.predict(X), yb, clf_b.predict(X)


def metrics(y, p, yb, pb):
    return {
        "acc": accuracy_score(y, p),
        "macroF1": f1_score(y, p, average="macro"),
        "F1_I": f1_score(y, p, labels=["I"], average="macro"),
        "toxicF1": f1_score(yb, pb),
    }


def bootstrap_ci(df, preds, rng):
    # resample whole matches: messages of the same match are not independent
    rows_of = df.reset_index().groupby("matchId").indices
    matches = list(rows_of)
    boot = []
    for _ in range(N_BOOT):
        pick = rng.choice(len(matches), size=len(matches), replace=True)
        ix = np.concatenate([rows_of[matches[i]] for i in pick])
        boot.append(metrics(*(a[ix] for a in preds)))
    boot = pd.DataFrame(boot)
    return {k: np.percentile(boot[k], [2.5, 97.5]) for k in ("macroF1", "toxicF1")}


# Seed 42 (same split as starter.py) with bootstrap CIs
train, dev = split(42)
model = train_baseline(train)
print(f"training set: {len(train)} messages, {train['matchId'].nunique()} matches")
print(f"dev: {len(dev)} messages, valid: {len(te)} messages")
print("valid matches absent from train.csv:", len(set(te["matchId"]) - set(tr["matchId"])))

rng = np.random.default_rng(42)
print("\n=== Seed 42, 95% CI (match-level bootstrap) ===")
for name, df in subsets(train, dev).items():
    preds = evaluate(model, df)
    m = metrics(*preds)
    ci = bootstrap_ci(df, preds, rng)
    print(f"{name:13s} n={len(df):5d} ({df['matchId'].nunique()} matches)  "
          f"acc {m['acc']:.3f}  macro F1 {m['macroF1']:.3f} [{ci['macroF1'][0]:.3f}, {ci['macroF1'][1]:.3f}]  "
          f"I F1 {m['F1_I']:.3f}  toxic F1 {m['toxicF1']:.3f} [{ci['toxicF1'][0]:.3f}, {ci['toxicF1'][1]:.3f}]")

# Same pipeline with 5 split seeds
runs = []
for seed in SEEDS:
    train_s, dev_s = split(seed)
    model_s = train_baseline(train_s)
    for name, df in subsets(train_s, dev_s).items():
        runs.append({"seed": seed, "subset": name, **metrics(*evaluate(model_s, df))})
runs = pd.DataFrame(runs)

print("\n=== 5 seeds, mean +/- std ===")
print(runs.groupby("subset")[["macroF1", "F1_I", "toxicF1"]].agg(["mean", "std"]).round(3))

gap = runs.pivot(index="seed", columns="subset", values="macroF1")
gap = gap["valid seen"] - gap["valid unseen"]
print(f"\nmacro F1 seen - unseen per seed: {gap.round(3).tolist()}, mean {gap.mean():.3f}")
