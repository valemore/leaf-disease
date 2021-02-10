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
img_idx = np.random.randint(len(dset_2020)) # 13955

transforms = A.Compose([
    A.Resize(386, 386),
    # A.OneOf(A.ShiftScaleRotate(0, (0.5), 0)),
    A.ShiftScaleRotate(shift_limit=0.0, scale_limit=(2.0, 2.0), rotate_limit=0, p=1.0),
    # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
    # A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=20, val_shift_limit=10, p=1.0),
    # A.RGBShift(10, 10, 10, p=1.0),
    A.GaussNoise(p=1.0),
    # A.Transpose(p=0.5),
    # A.HorizontalFlip(p=0.5),
    # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)
])

original_singleton_dset = LeafDataset.from_leaf_dataset(dset_2020, [img_idx], transform=original_transforms)
singleton_dset = LeafDataset.from_leaf_dataset(dset_2020, [img_idx], transform=transforms)

img, img_lab, _ = singleton_dset[0]
original_img, original_label, _ = original_singleton_dset[0]

plt.rcParams["figure.figsize"] = (10, 10)
fig, axs =plt.subplots(2, 1)
axs[0].imshow(original_img)
axs[1].imshow(img)
plt.show()


iter_train_dset = iter(train_dataloader)
img, lab, idx = next(iter_train_dset)
data_aug, target = fmix(data, target, alpha=1., decay_power=10., shape=(128,128))

for i in range(3):
    f, axarr = plt.subplots(1,4)
    for p in range(0,3,2):
        idx = np.random.randint(0, len(data))
        img_org = data[idx]
        new_img = data_aug[idx]
        axarr[p].imshow(img_org.permute(1,2,0))
        axarr[p+1].imshow(new_img.permute(1,2,0))
        axarr[p].set_title('original')
        axarr[p+1].set_title('fmix image')
        axarr[p].axis('off')
        axarr[p+1].axis('off')