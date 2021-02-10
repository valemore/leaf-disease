from leaf.dta import LeafDataset, LeafDataLoader, get_leaf_splits, UnionDataSet, TINY_SIZE
from leaf.model import LeafModel, train_one_epoch, validate_one_epoch
from leaf.sched import get_warmup_scheduler, LinearLR, fix_optimizer
from leaf.cutmix import CutMix
from leaf.cutmix_utils import CutMixCrossEntropyLoss
from leaf.mosaic import Mosaic, MosaicTransform
from leaf.utils import seed_everything

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import albumentations as A

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(386 * 386 * 3, out_features=5)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        return x

original_transforms = A.Compose([A.Resize(386, 386)])
dset_2020 = LeafDataset("data/images", "data/images/labels.csv", transform=original_transforms)
img_idxs = np.random.randint(0, len(dset_2020), 2) # 228, 18003

scales = [(0.0, 0.0), (0.17, 0.17), (0.33, 0.33), (0.5, 0.5)]

transforms = A.Compose([
    A.Resize(386, 386),
    # A.OneOf([A.ShiftScaleRotate(shift_limit=0.0, scale_limit=scale, rotate_limit=60) for scale in scales]),
    # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
    # A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=10, p=1.0),
    # A.IAAPiecewiseAffine(),
    # A.RGBShift(10, 10, 10, p=1.0),
    # A.GaussNoise(p=1.0),
    # A.HorizontalFlip(p=0.5),
    # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)
])

original_dset = LeafDataset.from_leaf_dataset(dset_2020, img_idxs.tolist(), transform=original_transforms)
pre_dset = LeafDataset.from_leaf_dataset(dset_2020, img_idxs.tolist(), transform=transforms)
mosaic_dset = MosaicTransform(pre_dset, 5, beta=2.0, prob=1.0, transform=None)

img, lab, _ = mosaic_dset[0]
original_img, original_label, _ = original_dset[0]

fig, axs = plt.subplots(2, 1, figsize=(5, 20))
fig.tight_layout()
axs[0].imshow(original_img)
axs[1].imshow(img)
plt.show()


fig, axs = plt.subplots(5, 1, figsize=(5, 20))
fig.set_size_inches(5, 20)
fig.tight_layout()
for ax, img in zip(axs, original_dset):
    ax.imshow(img[0])
    ax.axis('off')
    ax.set_title(img[1])
plt.show()
