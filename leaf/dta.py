import numpy as np
import pandas as pd
from pathlib import Path
from skimage import io
from time import sleep

import torch
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop


class LeafDataset(Dataset):
    """Cassava Leaf Disease Classification dataset."""

    def __init__(self, img_dir, labels_csv=None, transform=None):
        self.img_dir = img_dir if isinstance(img_dir, Path) else Path(img_dir)
        self.transform = transform
        if labels_csv:
            df = pd.read_csv(labels_csv)
            self.fnames = df["image_id"].values
            self.labels = df["label"].values
        else:
            self.fnames = np.array([img.name for img in img_dir.glob("*.jpg")])
            self.labels = None
        self.dataset_len = len(self.fnames)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = io.imread(self.img_dir / self.fnames[idx])
        if self.transform:
            img = self.transform(img)
        if self.labels is not None:
            label = self.labels[idx]
        else:
            label = None
        return img, label


def test_greenness(img, green_tol=0.1, ratio = 0.5):
    green_mod = (img[1, :, :] + green_tol).reshape(-1)
    n_pixels = len(green_mod)
    return ((green_mod > img[0, :, :].reshape(-1)) & (green_mod > img[2, :, :].reshape(-1))).sum() / n_pixels > ratio


class GetPatches(object):
    def __init__(self, img_width, img_height, x_patches, y_patches):
        assert isinstance(img_width, (int, tuple))
        assert isinstance(img_height, (int, tuple))
        assert isinstance(x_patches, (int, tuple))
        assert isinstance(y_patches, (int, tuple))
        assert img_width % x_patches == 0 and img_height % y_patches == 0
        self.x_patches = x_patches
        self.y_patches = y_patches
        self.img_width = img_width
        self.img_height = img_height
        self.patch_width = int(img_width / x_patches)
        self.patch_height = int(img_height / y_patches)

    def __call__(self, sample):
        assert tuple(sample.shape[1:]) == (self.img_width, self.img_height)
        patches = []
        for x in range(self.x_patches):
            for y in range(self.y_patches):
                patch = sample[:, (x * self.patch_width):((x + 1) * self.patch_width), (y * self.patch_height):((y + 1) * self.patch_height)]
                patches.append(patch)
        return patches


class TransformPatches(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, patches):
        transformed_patches = []
        for patch in patches:
            for t in self.transforms:
                patch = t(patch)
            transformed_patches.append(patch)
        return transformed_patches


class RandomGreen(object):
    def __init__(self, size_x, size_y, green_tol=0.1, green_ratio=0.5):
        self.size_x = size_x
        self.size_y = size_y
        self.green_tol = green_tol
        self.green_ratio = green_ratio
        self.random_crop = RandomCrop(size=(size_x, size_y), pad_if_needed=True, fill=-1)

    def __call__(self, img):
        cand = self.random_crop(img)
        while(not ((cand > -1).all() & test_greenness(cand, self.green_tol, self.green_ratio))):
            #nongreen_imgs.append(cand)
            cand = self.random_crop(img)
        #green_imgs.append(cand)
        return cand
