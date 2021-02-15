from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import os
import random

import numpy as np

import torch
from torch.optim import SGD, Adam

from leaf.dta import LeafDataset, LeafDataLoader, get_leaf_splits, UnionDataSet, TINY_SIZE
from leaf.model import LeafModel, train_one_epoch, validate_one_epoch
from leaf.sched import get_warmup_scheduler, LinearLR, fix_optimizer, reset_initial_lr, get_one_cycle
from leaf.cutmix import CutMix
from leaf.cutmix_utils import CutMixCrossEntropyLoss
from leaf.mosaic import Mosaic
from leaf.utils import seed_everything

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CyclicLR, LambdaLR, CosineAnnealingLR

import neptune

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def get_params_dict(cfg):
    params_dict = {}
    for param in ["train_transforms", "post_cutmix_transforms", "post_mosaic_transforms", "val_transforms", "batch_size", "num_epochs", "max_lr", "min_lr", "start_lr", "final_lr", "mid_lr", "leaf_model.optimizer", "leaf_model.scheduler", "grad_norm"]:
        try:
            params_dict[param] = eval(param)
        except:
            pass
    params_dict.update(cfg.__dict__)
    return params_dict


on_gcp = os.getcwd() == "/home/jupyter/leaf-disease"
output_dir = Path("/home/jupyter/outputs") if on_gcp else Path("/mnt/hdd/leaf-disease-outputs")
output_dir.mkdir(exist_ok=True)
print(f"Running on {'GCP' if on_gcp else 'local'} machine!")

seed_everything(5293)

@dataclass
class CFG:
    description: str = "gcp adam mosaic"
    model_file: str = "b4-adam-mosaic"
    num_classes: int = 5
    img_size: int = 380
    arch: str = "tf_efficientnet_b4_ns"
    loss_fn: str = "CutMixCrossEntropyLoss"
    mosaic_prob: float = 0.5
    mosaic_beta: float = 2.0

    def __repr__(self):
        return json.dumps(self.__dict__)
cfg = CFG()

num_workers = 4
use_fp16 = True

batch_size = 36
val_batch_size = 72

debug = False
if debug:
    batch_size = int(batch_size / 2)
    val_batch_size = int(batch_size / 2)

log_steps = 50 if on_gcp else 200

max_lr = 3e-4
start_lr = 1e-6
final_lr = 1e-6
mid_lr = 3e-5

weight_decay = 0.0

grad_norm = None

num_epochs = 10

train_transforms = A.Compose([
    A.Resize(CFG.img_size, CFG.img_size),
    A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=90, p=1.0),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
    A.RGBShift(p=1.0),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
    ToTensorV2()
])

post_mosaic_transforms = None

val_transforms = A.Compose([
    A.Resize(CFG.img_size, CFG.img_size),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
    ToTensorV2()
])

dset_2020 = LeafDataset("data/images", "data/images/labels.csv", transform=None)
dset_2019 = LeafDataset("data/2019", "data/2019/labels.csv", transform=None, tiny=debug)

num_splits = 5
folds = get_leaf_splits("./data/images/labels.csv", num_splits, random_seed=5293)

class Trainer(object):
    def __init__(self, leaf_model, train_dataloader, val_dataloader, log_steps, steps_offset=0, epoch=0, neptune=None, fp16=True, grad_norm=None):
        self.leaf_model = leaf_model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.log_steps = log_steps
        self.steps_offset = 0
        self.epoch = 0
        self.neptune = neptune
        self.fp16 = fp16
        self.grad_norm = grad_norm

    def train_epochs(self, epochs):
        for epoch in epochs:
            epoch_name = f"{self.leaf_model.model_prefix}-{self.epoch}"
            train_one_epoch(self.leaf_model, self.train_dataloader, log_steps=self.log_steps, epoch_name=epoch_name, steps_offset=self.steps_offset, neptune=self.neptune, grad_norm=self.grad_norm, fp16=self.fp16)
            self.steps_offset += len(self.train_dataloader)
            val_loss, val_acc = validate_one_epoch(self.leaf_model, self.val_dataloader)
            print(f"Validation after step {self.steps_offset}: loss {val_loss}, acc {val_acc}")
            if self.neptune:
                neptune.log_metric("loss/val", y=val_loss, x=self.steps_offset)
                neptune.log_metric("acc/val", y=val_acc, x=self.steps_offset)
            self.leaf_model.save_checkpoint(f"{epoch_name}", epoch_name=f"{epoch_name}")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    for fold, (train_idxs, val_idxs) in enumerate(folds):
        if fold != 0:
            continue
        if debug:
            train_idxs = train_idxs[:TINY_SIZE]
            val_idxs = val_idxs[:TINY_SIZE]

        fold_dset = LeafDataset.from_leaf_dataset(dset_2020, train_idxs, transform=None)
        pre_mosaic_train_dset = UnionDataSet(fold_dset, dset_2019, transform=train_transforms)
        # train_dset = pre_mosaic_train_dset
        train_dset = Mosaic(pre_mosaic_train_dset, num_class=5, beta=cfg.mosaic_beta, prob=cfg.mosaic_prob)
        val_dset = LeafDataset.from_leaf_dataset(dset_2020, val_idxs, transform=val_transforms)

        train_dataloader = LeafDataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_dataloader = LeafDataLoader(val_dset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)

        model_prefix = f"{cfg.model_file}_fold{fold}.{datetime.now().strftime('%b%d_%H-%M-%S')}"
        leaf_model = LeafModel(cfg, model_prefix=model_prefix, output_dir=output_dir)


    #     leaf_model.scheduler = get_one_cycle(leaf_model.optimizer, start_lr, max_lr, final_lr, num_epochs, steps_per_epoch=len(train_dataloader))

        neptune.init(project_qualified_name='vmorelli/cassava')

        neptune_tags = []
        neptune_tags.extend((["gcp"] if on_gcp else []) + (["dbg"] if debug else []))
        neptune.create_experiment(name=model_prefix, params=get_params_dict(cfg), upload_source_files=['*.py', 'leaf/*.py', 'environment.yml', "*.ipynb"],
                                  description=cfg.description,
                                  tags=neptune_tags)

        trainer = Trainer(leaf_model, train_dataloader, val_dataloader, log_steps, neptune=neptune, fp16=use_fp16, grad_norm=grad_norm)

        # Warmup
        leaf_model.optimizer = Adam(leaf_model.model.parameters(), lr=start_lr, weight_decay=weight_decay)
        leaf_model.scheduler = LinearLR(leaf_model.optimizer, start_lr, max_lr, len(train_dataloader))
        trainer.train_epochs([1])

        # 5 Cosine annealing
        reset_initial_lr(leaf_model.optimizer)
        cos_epochs = 5
        leaf_model.scheduler = CosineAnnealingWarmRestarts(leaf_model.optimizer, T_0=cos_epochs*len(train_dataloader)+1, eta_min=final_lr)
        trainer.train_epochs([2, 3, 4, 5, 6])

        # Warmup
        reset_initial_lr(leaf_model.optimizer)
        leaf_model.scheduler = LinearLR(leaf_model.optimizer, start_lr, mid_lr, len(train_dataloader))
        trainer.train_epochs([7])

        # 5 Cosine annealing
        reset_initial_lr(leaf_model.optimizer)
        cos_epochs = 5
        leaf_model.scheduler = CosineAnnealingWarmRestarts(leaf_model.optimizer, T_0=cos_epochs*len(train_dataloader)+1, eta_min=final_lr)
        trainer.train_epochs([8, 9, 10, 11, 12])

        neptune.stop()