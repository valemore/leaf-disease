from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import os
import random

import numpy as np

from torch.optim import SGD

from leaf.dta import LeafDataset, LeafDataLoader, get_leaf_splits, UnionDataSet
from leaf.model import LeafModel, train_one_epoch, validate_one_epoch
from leaf.cutmix import CutMix
from leaf.cutmix_utils import CutMixCrossEntropyLoss

import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CyclicLR, OneCycleLR

import neptune

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

if __name__ == "__main__":
    on_gcp = os.getcwd() == "/home/jupyter/leaf-disease"
    print(f"Running lr finder on {'GCP' if on_gcp else 'local'} machine!")

    @dataclass
    class CFG:
        description: str = "simple-lr-finder"
        num_classes: int = 5
        img_size: int = 380
        arch: str = "tf_efficientnet_b4_ns"
        loss_fn: str = "CrossEntropyLoss"

        def __repr__(self):
            return json.dumps(self.__dict__)
    cfg = CFG()

    output_dir = Path("/home/jupyter/outputs") if on_gcp else Path("/mnt/hdd/leaf-disease-outputs")
    output_dir.mkdir(exist_ok=True)

    num_workers = 4

    batch_size = 36 if on_gcp else 12
    val_batch_size = 72 if on_gcp else 24

    max_steps = 100 if on_gcp else 300
    log_steps = 10 if on_gcp else 30
    num_runs = 20

    grad_norm = None

    num_epochs = 1

    min_lr = 0.01
    max_lr = 1.0
    weight_decay_list = [0.0]
    momentum_list = [0.9]

    train_transforms = A.Compose([
        A.Resize(CFG.img_size, CFG.img_size),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, p =1.0),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
        # A.OneOf([A.MotionBlur(blur_limit=3), A.MedianBlur(blur_limit=3), A.GaussianBlur(blur_limit=3)], p=1.0,),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
        ToTensorV2()
    ])

    post_cutmix_transforms = None

    val_transforms = A.Compose([
        A.Resize(CFG.img_size, CFG.img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
        ToTensorV2()
    ])

    # folds = get_leaf_splits("./data/images/labels.csv", num_splits, random_seed=5293)

    dset_2020 = LeafDataset("data/images", "data/images/labels.csv", transform=None)
    dset_2019 = LeafDataset("data/2019", "data/2019/labels.csv", transform=None)

    num_splits = 5
    folds = get_leaf_splits("./data/images/labels.csv", num_splits, random_seed=5293)

    for fold, (train_idxs, val_idxs) in enumerate(folds):
        if fold != 0:
            continue
        # fold_dset = LeafDataset.from_leaf_dataset(dset_2020, train_idxs, transform=None)
        # pre_cutmix_train_dset = UnionDataSet(fold_dset, dset_2019, transform=train_transforms)
        train_dset = LeafDataset.from_leaf_dataset(dset_2020, train_idxs, transform=train_transforms)
        val_dset = LeafDataset.from_leaf_dataset(dset_2020, val_idxs, transform=val_transforms)

        train_dataloader = LeafDataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_dataloader = LeafDataLoader(val_dset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)

        model_prefix = f"{cfg.arch}_{cfg.description}_fold{fold}.{datetime.now().strftime('%b%d_%H-%M-%S')}"

        neptune.init(project_qualified_name='vmorelli/leaf')
        global_params_dict = {
                param: eval(param) for param in ["cfg", "train_transforms", "post_cutmix_transforms", "val_transforms", "batch_size", "num_epochs", "min_lr", "max_lr", "weight_decay_list", "momentum_list", "grad_norm"]
        }
        neptune_tags = ["lr"]
        neptune_tags.extend((["gcp"] if on_gcp else []))
        neptune.create_experiment(name=model_prefix, params=global_params_dict, upload_source_files=['*.py', 'leaf/*.py', 'environment.yml'],
                                      description="Learning rate finder",
                                      tags=neptune_tags)

        for lr in np.linspace(min_lr, max_lr, num=num_runs):
            momentum = random.choice(momentum_list)
            weight_decay = random.choice(weight_decay_list)
            run_params = {param: eval(param) for param in ["lr", "momentum", "weight_decay"]}

            leaf_model = LeafModel(cfg, model_prefix=model_prefix, output_dir=None)

            # optimizer = Adam(leaf_model.model.parameters(), lr=min_lr)
            optimizer = SGD(leaf_model.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
            # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=len(train_dataloader))
            scheduler = None
            leaf_model.update_optimizer_scheduler(optimizer, scheduler)

            str_run_params = {p: str(pv) for p, pv in run_params.items()}
            print(str_run_params)
            neptune.log_text("params", y=f"{json.dumps(run_params)}", x=lr)

            train_losses, _ = train_one_epoch(leaf_model, train_dataloader, max_steps=max_steps, log_steps=log_steps, epoch_name=f"lr-finder_{json.dumps(run_params)}", grad_norm=grad_norm)
            neptune.log_text("loss-history", y=f"{json.dumps(train_losses)}", x=lr)
            # neptune.log_text("acc-history", y=f"{json.dumps(train_accs)}", x=lr)
            log_loss = 1000 if np.isnan(train_losses[-1]) else train_losses[-1]
            neptune.log_metric("loss/train", y=log_loss, x=lr)
            # neptune.log_metric("acc/train", y=train_accs[-1], x=lr)

            val_loss, val_acc = validate_one_epoch(leaf_model, val_dataloader)
            val_loss = 1000 if torch.isnan(val_loss) else val_loss
            neptune.log_metric("loss/val", y=val_loss, x=lr)
            neptune.log_metric("acc/val", y=val_acc, x=lr)
            print(f"Validation loss {val_loss}, acc {val_acc}")

        neptune.stop()
