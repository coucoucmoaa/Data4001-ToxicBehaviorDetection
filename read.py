import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

tr = pd.read_csv("train.csv")
te = pd.read_csv("valid.csv")   # our held-out test set

# check whether the authors already split by match
print("matchId overlap train/valid:", len(set(tr["matchId"]) & set(te["matchId"])))

gss = GroupShuffleSplit(n_splits=1, test_size=0.12, random_state=42)
tr_idx, dev_idx = next(gss.split(tr, groups=tr["matchId"]))
train, dev = tr.iloc[tr_idx], tr.iloc[dev_idx]

print(len(train), len(dev), len(te))
print(train["intentClass"].value_counts(normalize=True).round(3))
print(dev["intentClass"].value_counts(normalize=True).round(3))
