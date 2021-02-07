import numpy as np
import pandas as pd
from pathlib import Path
from skimage import io
from math import ceil
from sklearn.model_selection import StratifiedKFold

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info, DataLoader

import cv2

TINY_SIZE = 100
NUM_CLASSES = 5


def get_leaf_splits(labels_csv, num_splits, random_seed):
    labels = pd.read_csv(labels_csv)["label"].values
    folds = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=random_seed).split(
        np.zeros(len(labels)), labels)
    return folds


class LeafDataset(Dataset):
    """Cassava Leaf Disease Classification dataset."""

    def __init__(self, img_dir, labels_csv=None, transform=None, tiny=False):
        self.img_dir = img_dir if isinstance(img_dir, Path) else Path(img_dir)
        self.transform = transform
        self.tiny = tiny
        if labels_csv:
            df = pd.read_csv(labels_csv)
            if self.tiny:
                df = df.iloc[:TINY_SIZE, :]
            self.fnames = df["image_id"].values
            self.labels = df["label"].values
        else:
            self.fnames = np.array([img.name for img in img_dir.glob("*.jpg")])
            if self.tiny:
                self.fnames = self.fnames[:TINY_SIZE]
            self.labels = None
        self.dataset_len = len(self.fnames)

    @classmethod
    def from_leaf_dataset(cls, leaf_dataset, subset, transform=None):
        self = cls.__new__(cls)
        self.img_dir = leaf_dataset.img_dir
        self.transform = leaf_dataset.transform if transform is None else transform
        self.tiny = leaf_dataset.tiny
        self.fnames = leaf_dataset.fnames[subset]
        self.labels = leaf_dataset.labels[subset]
        self.dataset_len = len(subset)

        if isinstance(leaf_dataset, DistillationDataSet):
            self.soft_labels = leaf_dataset.soft_labels[subset]
            self.soft_ratio = leaf_dataset.soft_ratio
            self.hard_labels = leaf_dataset.hard_labels[subset]

        return self

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        # img = io.imread(self.img_dir / self.fnames[idx])
        img = cv2.imread(str(self.img_dir / self.fnames[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)["image"]
        if self.labels is not None:
            label = self.labels[idx]
        else:
            label = None
        return img, label, idx


class UnionDataSet(Dataset):
    def __init__(self, one, two, transform=None):
        assert (one.labels is None and two.labels is None) or (one.labels is not None and two.labels is not None)
        self.one = one
        self.two = two
        self.transform = transform
        self.dataset_len = len(one) + len(two)
        self.one_len  = len(one)
        self.two_len = len(two)
        self.one_max  = len(one) - 1
        self.two_max = len(two) - 1

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        if idx > self.one_max:
            img, label, idx = self.two[idx - self.one_len]
            img = self.transform(image=img)["image"]
            return img, label, (1, idx)
        else:
            img, label, idx = self.one[idx]
            img = self.transform(image=img)["image"]
            return img, label, (0, idx)


class LeafDataLoader(DataLoader):
    def __init__(self, dset, batch_size, shuffle, sampler=None, num_workers=4):
        super().__init__(dset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers, pin_memory=True)
        self.dataset_len = len(dset)
        self.dataloader_len = ceil(len(dset) / batch_size)

    def __len__(self):
        return self.dataloader_len


class DistillationDataSet(Dataset):
    def __init__(self, dset, soft_targets_csv, soft_ratio=0.3):
        self.dset = dset
        df = pd.read_csv(soft_targets_csv)
        self.soft_labels = df.iloc[:, 2:].values
        self.soft_ratio = soft_ratio

        self.img_dir = dset.img_dir
        self.fnames = dset.fnames
        self.hard_labels = dset.labels
        self.dataset_len = len(self.dset)

        if self.dset.tiny:
            self.soft_labels = self.soft_labels[:len(self.dset), :]

        assert len(self.soft_labels) == len(self.fnames)

        hard_labels_oh = np.zeros((len(self.hard_labels), NUM_CLASSES))
        for i, l in enumerate(self.hard_labels):
            hard_labels_oh[i, l] = 1.0

        self.labels = (1 - self.soft_ratio) * hard_labels_oh + self.soft_ratio * self.soft_labels

        self.transform = None
        self.tiny = self.dset.tiny

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        img, _, idx = self.dset[idx]
        label = self.labels[idx]
        return img, label, idx
