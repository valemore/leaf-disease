# Run this once with original train_images
import math
from pathlib import Path
import random
import shutil

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

img_dir = Path("./data/stratified/train_images/")
val_dir = Path("./data/stratified/val_images")
val_dir.mkdir(exist_ok=True)

df = pd.read_csv("./data/stratified/train.csv")

print(df.shape)
print(df["label"].value_counts(normalize=True))

random_seed  = 5293
random.seed(random_seed)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=random_seed)

for train_idxs, val_idxs in sss.split(np.zeros(df.shape[0]), y=df["label"]):
    print("Split!")

assert len([idx for idx in train_idxs if idx in val_idxs]) == 0
assert len([idx for idx in val_idxs if idx in train_idxs]) == 0

print("Train  label distribution")
print(df.loc[train_idxs, "label"].value_counts(normalize=True))
print("Val label distribution")
print(df.loc[val_idxs, "label"].value_counts(normalize=True))

for img in df.loc[val_idxs, "image_id"]:
    shutil.move(img_dir / img, val_dir / img)

train_imgs = list(img_dir.glob("*"))
val_imgs = list(val_dir.glob("*"))

assert len(train_imgs) == df.loc[train_idxs, :].shape[0]
assert len(val_imgs) == df.loc[val_idxs, :].shape[0]

df.loc[train_idxs, :].reset_index(drop=True).to_csv(img_dir / "labels.csv", index=False)
df.loc[val_idxs, :].reset_index(drop=True).to_csv(val_dir / "labels.csv", index=False)
