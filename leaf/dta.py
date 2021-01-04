import colorsys

import numpy as np
import pandas as pd
from pathlib import Path
from skimage import io
from time import sleep
from math import ceil, floor

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info, DataLoader
from torchvision.transforms import RandomCrop, RandomResizedCrop, CenterCrop, Resize
from torchvision.transforms.functional import crop


TINY_SIZE = 10


class LeafIterableDataset(IterableDataset):
    """
    Cassava Leaf Disease Classification dataset.
    Use better with RandomGreen and GetPatches transforms.
    """
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
            if self.transform:
                img = self.transform(img) # may be a single image or a list
            if self.labels is not None:
                label = self.labels[idx]
            else:
                label = None
            yield img, label


class LeafCollate(object):
    def __init__(self, max_samples_per_image):
        self.max_samples_per_image = max_samples_per_image

    def __call__(self, batch):
        bs = len(batch)
        _, c, h, w = batch[0][0].shape
        batch_imgs = torch.zeros((bs * self.max_samples_per_image, c, h, w))
        batch_labels = torch.full((bs * self.max_samples_per_image,), -1)
        for i, (imgs, labels) in enumerate(batch):
            n_pad = self.max_samples_per_image - imgs.shape[0]
            batch_imgs[(i*self.max_samples_per_image):(i*self.max_samples_per_image + imgs.shape[0]), :, :, :] = imgs
            batch_labels[(i*self.max_samples_per_image):(i*self.max_samples_per_image + imgs.shape[0])] = labels
        #_, n_samples, c, h, w = imgs.shape
        #n_pad = len(batch) * self.max_samples_per_image - n_samples
        # imgs = torch.cat([imgs,
        #                   torch.zeros((n_pad, c, h, w))], dim=0)
        # labels = torch.LongTensor(labels + n_pad * [-1])
        return batch_imgs, batch_labels


class LeafDataLoader(DataLoader):
    def __init__(self, dset, batch_size, shuffle, num_workers=4, max_samples_per_image=1):
        collate_fn = None if max_samples_per_image == 1 else LeafCollate(max_samples_per_image)
        super().__init__(dset, collate_fn=collate_fn, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
        self.dataloader_len = ceil(len(dset) / batch_size)
        self.max_samples_per_image = max_samples_per_image
        self.num_padded_samples = self.dataloader_len * self.batch_size * max_samples_per_image

    def __len__(self):
        return self.dataloader_len


class LeafDataset(Dataset):
    """Cassava Leaf Disease Classification dataset."""

    def __init__(self, img_dir, labels_csv=None, extended_labels=False, transform=None, tiny=False):
        assert not (extended_labels and labels_csv is None)
        self.img_dir = img_dir if isinstance(img_dir, Path) else Path(img_dir)
        self.transform = transform
        self.tiny = tiny
        if labels_csv:
            df = pd.read_csv(labels_csv)
            if self.tiny:
                df = df.iloc[:TINY_SIZE, :]
            if extended_labels:
                self.fnames = np.array([f"{fname}-{patch_idx:03}.jpg" for fname, n_patches in zip(df["fname"], df["n_patches"]) for patch_idx in range(n_patches)])
                self.labels = np.array([label for label, n_patches in zip(df["label"], df["n_patches"]) for _ in range(n_patches)])
                self.original_fnames = np.array([fname for fname, n_patches in zip(df["fname"], df["n_patches"]) for _ in range(n_patches)])
                self.n_patches = np.array([n_patches for n_patches in df["n_patches"] for _ in range(n_patches)])
            else:
                self.fnames = df["image_id"].values
                self.labels = df["label"].values
                self.original_fnames = None
                self.n_patches = None
        else:
            self.fnames = np.array([img.name for img in img_dir.glob("*.jpg")])
            if self.tiny:
                self.fnames = self.fnames[:TINY_SIZE]
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
        # if self.original_fnames is not None:
        #     original_fname = self.original_fnames[idx]
        #     n_patches = self.n_patches[idx]
        # else:
        #     original_fname = None
        #     n_patches = None
        return img, label, idx


def convert_to_hsv(a):
    return np.array(colorsys.rgb_to_hsv(*a.tolist()))


def test_colors(img, min_ratio=0.3, min_value=0.0, min_hue=40/360, max_hue=170/360):
    """Heuristic to test for 'leafness' of the IMG given as a channel-first float tensor rescaled to the range [0, 1]."""
    img_hsv = np.apply_along_axis(convert_to_hsv, axis=0, arr=img.reshape(3, -1))
    n_pixels = img_hsv.shape[1]
    return ((img_hsv[0, :] >= min_hue) & (img_hsv[0, :] <= max_hue) & (img_hsv[2, :] >= min_value)).sum() / n_pixels >= min_ratio


class GetPatches(object):
    def __init__(self, img_width, img_height, patch_size, min_ratio=0.3, min_value=0.0, min_hue=40/360, max_hue=170/360, include_whole=True):
        assert isinstance(img_width, (int, tuple))
        assert isinstance(img_height, (int, tuple))
        assert isinstance(patch_size, (int, tuple))
        assert (min_ratio is None and min_value is None and min_hue is None and max_hue is None) or (min_ratio is not None and min_value is not None and min_hue is not None and max_hue is not None)
        self.img_width = img_width
        self.img_height = img_height
        self.min_ratio = min_ratio
        self.min_value = min_value
        self.min_hue = min_hue
        self.max_hue = max_hue
        self.include_whole = include_whole
        if self.include_whole:
            self.resize = Resize(256)
            self.center_crop = CenterCrop(224)
        self.patch_size = patch_size
        assert self.patch_size == 224 # TODO
        self.x_patches = ceil(img_width / patch_size)
        self.y_patches = ceil(img_height / patch_size)
        self.step_x = floor((self.img_width - self.patch_size) / (self.x_patches - 1))
        self.step_y = floor((self.img_height - self.patch_size) / (self.y_patches - 1))

    def __call__(self, sample):
        assert tuple(sample.shape[1:]) == (self.img_height, self.img_width)
        patches = []
        if self.include_whole:
            patches.append(self.center_crop(self.resize(sample)))

        for patch_y in np.arange(self.y_patches) * self.step_y:
            for patch_x in np.arange(self.x_patches) * self.step_x:
                patch = crop(sample, patch_y, patch_x, self.patch_size, self.patch_size)
                if self.min_ratio:
                    if test_colors(patch, self.min_ratio, self.min_value, self.min_hue, self.max_hue):
                        patches.append(patch)
                else:
                    patches.append(patch)
        assert len(patches) > 0 # TODO: Do we process batches gracefully in this case?
        return patches


class TransformPatches(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, patches):
        return [self.transform(patch) for patch in patches]


class RandomGreen(object):
    def __init__(self, size_x, size_y, min_ratio=0.3, min_value=0.0, min_hue=40/360, max_hue=170/360, keep_aspect_ratio=True):
        assert (min_ratio is None and min_value is None and min_hue is None and max_hue is None) or (
                    min_ratio is not None and min_value is not None and min_hue is not None and max_hue is not None)
        self.size_x = size_x
        self.size_y = size_y
        self.min_ratio = min_ratio
        self.min_value = min_value
        self.min_hue = min_hue
        self.max_hue = max_hue
        if keep_aspect_ratio:
            self.random_crop = RandomCrop(size=(size_y, size_x), pad_if_needed=True, fill=-1)
        else:
            self.random_crop = RandomResizedCrop(size=(size_y, size_x))
        self.resize = Resize(256)
        self.center_crop = CenterCrop(224)
        assert size_x == size_y == 224 # TODO

    def __call__(self, img, max_tries=10):
        cand = self.random_crop(img)
        tries = 0
        while(not ((cand > -1).all() and test_colors(cand, self.min_ratio, self.min_value, self.min_hue, self.max_hue))):
            if tries == max_tries:
                cand = self.resize(cand)
                cand = self.center_crop(cand)
                return cand
            cand = self.random_crop(img)
            tries += 1
        return cand
