from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import os
import random

import numpy as np

from torch.optim import SGD, Adam

from leaf.dta import LeafDataset, LeafDataLoader, get_leaf_splits, UnionDataSet
from leaf.model import LeafModel, train_one_epoch, validate_one_epoch
from leaf.sched import get_warmup_scheduler
from leaf.cutmix import CutMix
from leaf.cutmix_utils import CutMixCrossEntropyLoss

from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts

import neptune

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


if __name__ == "__main__":
    on_gcp = os.getcwd() == "/home/jupyter/leaf-disease"
    print(f"Running on {'GCP' if on_gcp else 'local'} machine!")

    @dataclass
    class CFG:
        description: str = "cutmix-folds"
        num_classes: int = 5
        img_size: int = 380
        arch: str = "tf_efficientnet_b4_ns"
        loss_fn: str = "CutMixCrossEntropyLoss"
        cutmix_prob: float = 0.5
        cutmix_num_mix: int = 2

        def __repr__(self):
            return json.dumps(self.__dict__)
    cfg = CFG()

    output_dir = Path("/home/jupyter/outputs") if on_gcp else Path("/mnt/hdd/leaf-disease-outputs")
    output_dir.mkdir(exist_ok=True)

    num_workers = 4

    batch_size = 36 if on_gcp else 12
    val_batch_size = 72 if on_gcp else 24

    log_steps = 50 if on_gcp else 200

    #max_lr = 0.015
    max_lr = 0.0375
    max_lr = 3 * max_lr if on_gcp else max_lr
    weight_decay = 0.0
    momentum = 0.9

    grad_norm = None
    
    num_epochs = 7

    train_transforms = A.Compose([
        A.Resize(CFG.img_size, CFG.img_size),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.5, rotate_limit=20),
        A.RandomBrightnessContrast(brightness_limit=0.07, contrast_limit=0.07, p=1.0),
        A.RGBShift(p=1.0),
        A.GaussNoise(p=1.0),
        A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=20, val_shift_limit=(-10, 20), p=1.0),
        A.HorizontalFlip(p=0.5),
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

    num_splits = 7
    folds = get_leaf_splits("./data/images/labels.csv", num_splits, random_seed=5293)

    for fold, (train_idxs, val_idxs) in enumerate(folds):
        if fold != 0:
            continue
        fold_dset = LeafDataset.from_leaf_dataset(dset_2020, train_idxs, transform=None)
        pre_cutmix_train_dset = UnionDataSet(fold_dset, dset_2019, transform=train_transforms)
        train_dset = CutMix(pre_cutmix_train_dset, num_class=5, beta=1.0, prob=CFG.cutmix_prob, num_mix=CFG.cutmix_num_mix, transform=post_cutmix_transforms)
        val_dset = LeafDataset.from_leaf_dataset(dset_2020, val_idxs, transform=val_transforms)

        train_dataloader = LeafDataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_dataloader = LeafDataLoader(val_dset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)

        model_prefix = f"{cfg.arch}_{cfg.description}_fold{fold}.{datetime.now().strftime('%b%d_%H-%M-%S')}"
        leaf_model = LeafModel(cfg, model_prefix=model_prefix, output_dir=output_dir)

        # optimizer = Adam(leaf_model.model.parameters(), lr=min_lr)
        optimizer = SGD(leaf_model.model.parameters(), lr=max_lr, momentum=momentum, weight_decay=weight_decay)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=len(train_dataloader), T_mult=2)
        leaf_model.update_optimizer_scheduler(optimizer, scheduler)

        neptune.init(project_qualified_name='vmorelli/leaf')
        params_dict = {
            param: eval(param) for param in ["cfg", "train_transforms", "post_cutmix_transforms", "val_transforms", "batch_size", "num_epochs", "max_lr", "weight_decay", "optimizer", "scheduler", "grad_norm"]
        }
        neptune.create_experiment(name=model_prefix, params=params_dict, upload_source_files=['*.py', 'leaf/*.py', 'environment.yml'],
                                  description="Local run with new scheduler",
                                  tags=[].extend(["gcp"] if on_gcp else []))
        str_params_dict = {p: str(pv) for p, pv in params_dict.items()}
        neptune.log_text("params", f"{json.dumps(str_params_dict)}")

        decay_epoch = None
        for epoch in range(1, num_epochs+1):
            epoch_name = f"{model_prefix}-{epoch}"

            if epoch == decay_epoch:
                for pg in leaf_model.optimizer.param_groups:
                    pg["lr"] /= 2
                    pg["initial_lr"] /= 2
                leaf_model.scheduler.base_lrs = [base_lr / 2 for base_lr in leaf_model.scheduler.base_lrs]
                decay_epoch *= leaf_model.scheduler.T_mult

            train_one_epoch(leaf_model, train_dataloader, log_steps=log_steps, epoch_name=epoch_name, epoch=epoch, neptune=neptune, grad_norm=grad_norm)
            val_loss, val_acc = validate_one_epoch(leaf_model, val_dataloader)
            print(f"Validation after epoch {epoch}: loss {val_loss}, acc {val_acc}")
            val_step = len(train_dataloader) * epoch
            neptune.log_metric("loss/val", y=val_loss, x=val_step)
            neptune.log_metric("acc/val", y=val_acc, x=val_step)
            leaf_model.save_checkpoint(f"{epoch_name}", epoch=epoch)

        neptune.stop()
