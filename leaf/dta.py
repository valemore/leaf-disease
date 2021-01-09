import colorsys

import numpy as np
import pandas as pd
from pathlib import Path
from skimage import io
from time import sleep
from math import ceil, floor
from sklearn.model_selection import StratifiedKFold

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info, DataLoader
from torchvision.transforms import Compose, RandomCrop, RandomResizedCrop, CenterCrop, Resize
from torchvision.transforms.functional import crop


TINY_SIZE = 100


def get_img_id(img_fname):
    if img_fname == "":
        return -1
    return int(img_fname[:-4])


def get_patch_id(patch_fname):
    if patch_fname == "":
        return -1
    return int(patch_fname[-7:-4])


def get_leaf_splits(labels_csv, num_splits, random_seed):
    labels = pd.read_csv(labels_csv)["label"].values
    folds = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=random_seed).split(
        np.zeros(len(labels)), labels)
    return folds


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
            if self.transform:
                img = self.transform(img) # may be a single image or a list
            if self.labels is not None:
                label = self.labels[idx]
            else:
                label = None
            yield img, label, idx


class LeafCollate(object):
    def __init__(self, max_samples_per_image):
        self.max_samples_per_image = max_samples_per_image

    def __call__(self, batch):
        bs = len(batch)
        _, c, h, w = batch[0][0].shape
        batch_imgs = torch.zeros((bs * self.max_samples_per_image, c, h, w))
        batch_labels = torch.full((bs * self.max_samples_per_image,), -1)
        batch_idxs = torch.full((bs * self.max_samples_per_image,), -1)
        for i, (imgs, labels, idxs) in enumerate(batch):
            batch_imgs[(i*self.max_samples_per_image):(i*self.max_samples_per_image + imgs.shape[0]), :, :, :] = imgs
            if labels is None:
                batch_labels = None
            else:
                batch_labels[(i*self.max_samples_per_image):(i*self.max_samples_per_image + imgs.shape[0])] = labels
            batch_idxs[(i * self.max_samples_per_image):(i * self.max_samples_per_image + imgs.shape[0])] = idxs
        #_, n_samples, c, h, w = imgs.shape
        #n_pad = len(batch) * self.max_samples_per_image - n_samples
        # imgs = torch.cat([imgs,
        #                   torch.zeros((n_pad, c, h, w))], dim=0)
        # labels = torch.LongTensor(labels + n_pad * [-1])
        return batch_imgs, batch_labels, batch_idxs


class LeafDataLoader(DataLoader):
    def __init__(self, dset, batch_size, shuffle, num_workers=4, max_samples_per_image=1):
        collate_fn = LeafCollate(max_samples_per_image) if max_samples_per_image > 1 else None
        if shuffle is not None:
            assert isinstance(dset, LeafDataset)
            super().__init__(dset, collate_fn=collate_fn, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
        else:
            assert isinstance(dset, LeafIterableDataset)
            super().__init__(dset, collate_fn=collate_fn, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        self.max_samples_per_image = max_samples_per_image
        self.num_padded_samples = len(dset) * max_samples_per_image
        self.dataloader_len = ceil(len(dset) / batch_size)

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
        if leaf_dataset.original_fnames is not None:
            self.original_fnames = leaf_dataset.original_fnames[subset]
            self.n_patches = leaf_dataset.n_patches[subset]
        else:
            self.original_fnames = None
            self.n_patches = None
        self.img_ids = leaf_dataset.img_ids[subset]
        self.dataset_len = len(subset)

        return self


    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
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
    def __init__(self, img_width, img_height, patch_size, test_colors=False, min_ratio=0.3, min_value=0.0, min_hue=40/360, max_hue=170/360, include_center=False):
        assert isinstance(img_width, (int, tuple))
        assert isinstance(img_height, (int, tuple))
        assert isinstance(patch_size, (int, tuple))
        #assert (min_ratio is None and min_value is None and min_hue is None and max_hue is None) or (min_ratio is not None and min_value is not None and min_hue is not None and max_hue is not None)
        self.test_colors = test_colors
        self.img_width = img_width
        self.img_height = img_height
        self.min_ratio = min_ratio
        self.min_value = min_value
        self.min_hue = min_hue
        self.max_hue = max_hue
        self.include_center = include_center
        resize_size = 256 if patch_size == 224 else 436 if patch_size == 380 else "Error"
        self.resize = Resize(resize_size)
        self.center_crop = CenterCrop(patch_size)
        self.patch_size = patch_size
        assert self.patch_size in [224, 380]
        self.x_patches = ceil(img_width / patch_size)
        self.y_patches = ceil(img_height / patch_size)
        self.step_x = floor((self.img_width - self.patch_size) / (self.x_patches - 1))
        self.step_y = floor((self.img_height - self.patch_size) / (self.y_patches - 1))

    def __call__(self, sample):
        assert tuple(sample.shape[1:]) == (self.img_height, self.img_width)
        patches = []
        if self.include_center:
            patches.append(self.center_crop(self.resize(sample)))

        for patch_y in np.arange(self.y_patches) * self.step_y:
            for patch_x in np.arange(self.x_patches) * self.step_x:
                patch = crop(sample, patch_y, patch_x, self.patch_size, self.patch_size)
                if self.test_colors:
                    if test_colors(patch, self.min_ratio, self.min_value, self.min_hue, self.max_hue):
                        patches.append(patch)
                else:
                    patches.append(patch)
        if len(patches) == 0:
            print("Image with no valid patches!")
            patches.append(self.center_crop(self.resize(sample)))
        #assert len(patches) > 0 # TODO: Do we process batches gracefully in this case?
        return patches


class TransformPatches(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, patches):
        return [self.transform(patch) for patch in patches]


class RandomGreenBase(object):
    def __init__(self, patch_size, min_ratio=0.3, min_value=0.0, min_hue=40 / 360, max_hue=170 / 360, keep_aspect_ratio=True):
        assert (min_ratio is None and min_value is None and min_hue is None and max_hue is None) or (
                    min_ratio is not None and min_value is not None and min_hue is not None and max_hue is not None)
        self.size_x = patch_size
        self.min_ratio = min_ratio
        self.min_value = min_value
        self.min_hue = min_hue
        self.max_hue = max_hue
        if keep_aspect_ratio:
            self.random_crop = RandomCrop(patch_size, pad_if_needed=True, fill=-1)
        else:
            self.random_crop = RandomResizedCrop(patch_size)

    def __call__(self, img):
        cand = self.random_crop(img)
        if (cand > -1).all() and test_colors(cand, self.min_ratio, self.min_value, self.min_hue, self.max_hue):
            return cand
        else:
            return None


class RandomInnerCrop(object):
    def __init__(self, **params):
        self.random_crop = RandomCrop(**params, pad_if_needed=True, fill=-1)

    def __call__(self, img):
        cand = self.random_crop(img)
        while not ((cand > -1).all()):
            cand = self.random_crop(img)
        return cand


class RandomGreen(object):
    def __init__(self, patch_size, min_ratio=0.3, min_value=0.0, min_hue=40 / 360, max_hue=170 / 360, keep_aspect_ratio=True, default_center=False):
        assert (min_ratio is None and min_value is None and min_hue is None and max_hue is None) or (
                    min_ratio is not None and min_value is not None and min_hue is not None and max_hue is not None)
        self.random_green_base = RandomGreenBase(patch_size, min_ratio, min_value, min_hue, max_hue, keep_aspect_ratio)

        if default_center:
            resize_size = 256 if patch_size == 224 else 436 if patch_size == 380 else "Error"
            self.default_crop = Compose([Resize(resize_size), CenterCrop(patch_size)])
        else:
            if keep_aspect_ratio:
                self.default_crop = RandomInnerCrop(patch_size)
            else:
                self.default_crop = RandomResizedCrop(patch_size)

    def __call__(self, img, max_tries=10):
        cand = self.random_green_base(img)
        tries = 0
        while cand is None:
            if tries == max_tries:
                return self.default_crop(img)
            cand = self.random_green_base(img)
            tries += 1
        return cand


class GetRandomResizedCrops(object):
    def __init__(self, num_crops, **params):
        self.num_crops = num_crops
        self.random_resized_crop = RandomResizedCrop(**params)

    def __call__(self, img):
        patches = []
        for _ in range(self.num_crops):
            patch = self.random_resized_crop(img)
            patches.append(patch)
        return patches


class GetRandomCrops(object):
    def __init__(self, num_crops, **params):
        self.num_crops = num_crops
        self.random_crop = RandomCrop(**params, pad_if_needed=True, fill=-1)

    def __call__(self, img):
        patches = []
        for _ in range(self.num_crops):
            cand = self.random_crop(img)
            while (not ((cand > -1).all())):
                cand = self.random_crop(img)
            patches.append(cand)
        return patches


class RandomGreenCrops(object):
    def __init__(self, num_crops, **params):
        self.num_crops = num_crops
        self.ramdom_green_base = RandomGreenBase(**params)

    def __call__(self, img, max_tries=10):
        patches = []
        for _ in range(self.num_crops):
            cand = self.ramdom_green_base(img)
            while cand is None:
                tries = 0
                if tries == max_tries:
                    get_random_crops = GetRandomCrops(self.num_crops - len(patches))
                    patches.extend(get_random_crops(img))
                    return patches
                cand = self.random_crop(img)
                tries += 1
            patches.append(cand)
        return patches
