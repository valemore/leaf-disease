import pandas as pd

theirs_2019 = pd.read_csv("data/theirs/soft_targets_2019.csv")
theirs_2020 = pd.read_csv("data/theirs/soft_targets_2020.csv")

ours_2019 = pd.read_csv("data/2019/labels.csv")
ours_2020 = pd.read_csv("data/images/labels.csv")

soft_2019 = ours_2019.merge(theirs_2019, how="left", on="image_id")

missing_idxs = soft_2019.loc[soft_2019["p0"].isna(), :].index

for idx in missing_idxs:
    label = f'p{soft_2019.loc[idx, "label"]}'
    soft_2019.loc[idx, label] = 1.0

soft_2019 = soft_2019.fillna(0.0)

soft_2020 = ours_2020.merge(theirs_2020, how="left", on="image_id")

soft_2019.to_csv("data/theirs_soft_2019.csv", index=False)
soft_2020.to_csv("data/theirs_soft_2020.csv", index=False)
