import pandas as pd
from pathlib import Path
import json
import shutil
from tqdm import tqdm

dir_2019 = Path("data/2019/")
dir_2020 = Path("data/images/")

dir_duplicates_df = Path("theirs/2019-2020-duplicate_images/")

labels_2019_df_fname = dir_2019 / "labels.csv"

labels_2019_df = pd.read_csv(labels_2019_df_fname)

def remove_duplicates(df, prefix):
    new_df = df.copy()
    duplicates_df_fname = dir_duplicates_df / f"{prefix}.csv"
    duplicates_df = pd.read_csv(duplicates_df_fname)

    for img_id in duplicates_df["2019_id"]:
        new_df = new_df.loc[new_df["image_id"] != img_id]
    return new_df.reset_index(drop=True)


deduplicated_df = labels_2019_df.copy()
for prefix in ["cbb", "cbsd", "cgm", "cmd", "healthy"]:
    deduplicated_df = remove_duplicates(deduplicated_df, prefix)

files_to_remove = labels_2019_df.loc[~labels_2019_df["image_id"].isin(deduplicated_df["image_id"]), "image_id"]

for img_fname in files_to_remove:
    (dir_2019 / img_fname).unlink()

deduplicated_df.to_csv(dir_2019 / "labels.csv", index=False)
