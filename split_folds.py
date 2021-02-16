import pandas as pd
import shutil
from pathlib import Path

from leaf.dta import LeafDataset, get_leaf_splits

if __name__ == "__main__":
    dset_2020 = LeafDataset("data/images", "data/images/labels.csv")

    num_splits = 5
    folds = get_leaf_splits("./data/images/labels.csv", num_splits, random_seed=5293)

    for fold, (train_idxs, val_idxs) in enumerate(folds):
        if fold != 4:
           continue
        test_dset = LeafDataset.from_leaf_dataset(dset_2020, val_idxs)
        test_df = pd.DataFrame({"image_id": test_dset.fnames, "label": test_dset.labels})
        test_df.to_csv("data/test_df.csv", index=False)

    for fname in test_df["image_id"]:
        src = Path("data/images") / fname
        dst = Path("data/mock_test") / fname
        shutil.copy(src, dst)
