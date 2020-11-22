# Run this once with original train_images
import math
from pathlib import Path
import random
import shutil

import pandas as pd

img_dir = Path("./data/train_images/")
val_dir = Path("./data/val_images")
val_dir.mkdir(exist_ok=True)
n_images = 21397

img_list = list(img_dir.glob("*"))

random.seed(52109)
random.shuffle(img_list)

val_ratio = 0.1
n_train = math.floor(len(img_list) * (1-val_ratio))

for img in img_list[n_train:]:
    shutil.move(img, val_dir / img.name)

df = pd.read_csv("./data/train.csv")
train_imgs = list(img_dir.glob("*"))

train_ids_list = []
for img in train_imgs:
    train_ids_list.append(img.name)

df[df["image_id"].isin(train_ids_list)].to_csv("./data/train_images/labels.csv", index=False)

val_imgs = list(val_dir.glob("*"))
val_ids_list = []
for img in val_imgs:
    val_ids_list.append(img.name)

df[df["image_id"].isin(val_ids_list)].to_csv("./data/val_images/labels.csv", index=False)
