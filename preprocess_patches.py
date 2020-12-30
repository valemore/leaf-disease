from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd

from torchvision import transforms
from torchvision.utils import save_image

from leaf.dta import LeafDataset, GetPatches


# Transforms with normalizations for imagenet
transforms = transforms.Compose([
    transforms.ToTensor(),
    GetPatches(800, 600, 224)
])

dset = LeafDataset("./data/val_images", "./data/val_images/labels.csv")
output_dir = Path("./data/patches_val/")


def preprocess_image(idx):
    fname = dset.fnames[idx]
    img, label = dset[idx]
    imgs = transforms(img)
    for i, img in enumerate(imgs):
        save_image(img, output_dir / f"{fname}-{i:03}.jpg")
    return idx, fname, label, len(imgs)


if __name__ == "__main__":
    output_dir.mkdir(exist_ok=True)

    result_df = pd.DataFrame(columns=["idx", "fname", "label", "n_patches"])

    with Pool(processes=4) as pool:
        idx_list, fname_list, label_list, n_patches_list = zip(*tqdm(pool.imap_unordered(preprocess_image, range(len(dset)))))

    result_df = pd.DataFrame({
        "idx": idx_list, "fname": fname_list, "label": label_list, "n_patches": n_patches_list
    })
    result_df.to_csv(output_dir / "labels.csv", index=False)
