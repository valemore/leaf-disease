from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import os
import random

import numpy as np

from torch.optim import SGD

from leaf.dta import LeafDataset, LeafIterableDataset, LeafDataLoader, LeafDataLoader, get_leaf_splits
from leaf.model import LeafModel, train_one_epoch, validate_one_epoch
from leaf.sched import get_warmup_scheduler
from leaf.cutmix import CutMix

from torch.optim.lr_scheduler import OneCycleLR

import neptune

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


@dataclass
class CFG:
    num_classes: int = 5
    img_size: int = 380
    arch: str = "tf_efficientnet_b4_ns"
    loss_fn: str = "CrossEntropyLoss"
    smoothing: float = 0.00
    crop1_size: int = 560
    crop2_range: int = 40

    def __repr__(self):
        return json.dumps(self.__dict__)
cfg = CFG()

output_dir = Path("/mnt/hdd/leaf-disease-outputs")
output_dir.mkdir(exist_ok=True)

num_workers = 1

batch_size = 12
val_batch_size = 24

log_steps = 200

min_lr = 8.055822378718028e-4
#max_lr = 0.015
max_lr = 0.06190499161193587
weight_decay = 0.0
momentum = 0.95

grad_norm = None

num_epochs = 7

train_transforms = A.Compose([
    A.SmallestMaxSize(cfg.crop1_size),
    A.RandomSizedCrop(min_max_height=(CFG.img_size - CFG.crop2_range, CFG.img_size + CFG.crop2_range), height=CFG.img_size, width=CFG.img_size),
    # A.RandomCrop(CFG.img_size, CFG.img_size),
    #A.SmallestMaxSize(CFG.img_size),
    # A.RandomBrightnessContrast(brightness_limit=0.07, contrast_limit=0.07, p=1.0),
    # A.RGBShift(p=1.0),
    A.GaussNoise(p=1.0),
    A.HueSaturationValue(hue_shift_limit=2, sat_shift_limit=20, val_shift_limit=20, p=1.0),
    A.ISONoise(p=1.0),
    # Grid !!!
    A.HorizontalFlip(p=0.5),
    # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
    ToTensorV2()
])

post_cutmix_transforms = A.Compose([
    ToTensorV2()
])

val_transforms = A.Compose([
    A.SmallestMaxSize(cfg.crop1_size),
    A.CenterCrop(CFG.img_size, CFG.img_size),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
    ToTensorV2()
])

pre_cutmix_train_dset = LeafDataset("data/stratified/train_images", "data/stratified/train_images/labels.csv", transform=train_transforms)
train_dset = CutMix(pre_cutmix_train_dset, num_class=5, beta=1.0, prob=0.5, num_mix=3, transform=None)

import matplotlib.pyplot as plt

for i in range(3):
    f, axarr = plt.subplots(1,4)
    for p in range(0,3,2):
        idx = np.random.randint(0, len(pre_cutmix_train_dset))
        img_org, _, _ = pre_cutmix_train_dset[idx]
        new_img, _ = train_dset[idx]
        axarr[p].imshow(img_org.numpy().transpose(1, 2, 0))
        axarr[p+1].imshow(new_img.numpy().transpose(1, 2, 0))
        axarr[p].set_title('original')
        axarr[p+1].set_title('cutmix image')
        axarr[p].axis('off')
        axarr[p+1].axis('off')

plt.show()
plt.close()

