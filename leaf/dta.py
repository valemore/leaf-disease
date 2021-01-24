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


def get_img_id(img_fname):
    if img_fname == "":
        return -1
    return int(img_fname[:-4])


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
        self.img_ids = np.array([get_img_id(fname) for fname in self.fnames])
        self.dataset_len = len(self.fnames)

    @classmethod
    def from_leaf_dataset(cls, leaf_dataset, subset, transform=None):
        self = cls.__new__(cls)
        self.img_dir = leaf_dataset.img_dir
        self.transform = leaf_dataset.transform if transform is None else transform
        self.tiny = leaf_dataset.tiny
        self.fnames = leaf_dataset.fnames[subset]
        self.labels = leaf_dataset.labels[subset]
        self.img_ids = leaf_dataset.img_ids[subset]
        self.dataset_len = len(subset)

        return self


    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        img = io.imread(self.img_dir / self.fnames[idx])
        # img = cv2.imread(str(self.img_dir / self.fnames[idx]))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)["image"]
        if self.labels is not None:
            label = self.labels[idx]
        else:
            label = None
        return img, label, idx


class LeafIterableDataset(IterableDataset):
    def __init__(self, img_dir, labels_csv=None, transform=None, tiny=False):
        super().__init__()
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
        self.img_ids = [get_img_id(fname) for fname in self.fnames]
        self.dataset_len = len(self.fnames)

    def __len__(self):
        return self.dataset_len

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = self.dataset_len
        else:
            per_worker = ceil(self.dataset_len / worker_info.num_workers)
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(self.dataset_len, iter_start + per_worker)
        for idx in range(iter_start, iter_end):
            img = io.imread(self.img_dir / self.fnames[idx])
            # img = cv2.imread(str(self.img_dir / self.fnames[idx]))
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform:
                img = self.transform(image=img)["image"]
            if self.labels is not None:
                label = self.labels[idx]
            else:
                label = None
            yield img, label, idx


class LeafDataLoader(DataLoader):
    def __init__(self, dset, batch_size, shuffle, num_workers=4):
        if shuffle is not None:
            assert isinstance(dset, LeafDataset), "Setting shuffling to True or False only makes sense for map-style dataset!"
            super().__init__(dset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
        else:
            assert isinstance(dset, LeafIterableDataset), "Setting shuffling to none only makes sense for iterable dataset!"
            super().__init__(dset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        self.dataset_len = len(dset)
        self.dataloader_len = ceil(len(dset) / batch_size)

    def __len__(self):
        return self.dataloader_len
