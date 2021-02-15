from tqdm import tqdm
from pathlib import Path

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from leaf.dta import LeafDataset, get_leaf_splits

import timm

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

arch = "tf_efficientnet_b4_ns"
val_img_size = 480
tta_img_size = val_img_size
batch_size = 32

checkpoint_dir = Path("b4-folds")
checkpoints = {
    0: "b4-446_fold0.Feb14_20-26-07/b4-446_fold0.Feb14_20-26-07-10",
    1: "b4-446_fold1.Feb14_22-09-33/b4-446_fold1.Feb14_22-09-33-11",
    2: "b4-446_fold2.Feb14_23-51-32/b4-446_fold2.Feb14_23-51-32-8",
    3: "b4-446_fold3.Feb15_01-32-45/b4-446_fold3.Feb15_01-32-45-11",
    4: "b4-446_fold4.Feb15_03-15-34/b4-446_fold4.Feb15_03-15-34-9"
}

transforms = A.Compose([
    A.Resize(val_img_size, val_img_size),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
    ToTensorV2()
])

num_tta = 15

tta_transforms = A.Compose([
    A.Resize(tta_img_size, tta_img_size),
    A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=90, p=1.0),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
    A.RGBShift(p=1.0),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
    ToTensorV2()
])

dset_2020 = LeafDataset("data/images", "data/images/labels.csv", transform=None)
num_splits = 5
folds = get_leaf_splits("./data/images/labels.csv", num_splits, random_seed=5293)

model = timm.create_model(model_name=arch, num_classes=5, pretrained=False)
device = torch.device("cuda")

torch.cuda.empty_cache()
fold_accs = np.zeros(5)
for fold, (train_idxs, val_idxs) in enumerate(folds):
    # if fold != 0:
    #    continue

    checkpoint = checkpoint_dir / checkpoints[fold]
    checkpoint_dict = torch.load(checkpoint)
    model.load_state_dict(checkpoint_dict['model_state_dict'])

    model = model.to(device)
    model.eval()

    dset = LeafDataset.from_leaf_dataset(dset_2020, val_idxs, transform=transforms)

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    with torch.no_grad():
        logits_all = torch.zeros((len(dset), 5), dtype=float, device=device)
        labels_all = torch.zeros((len(dset),), dtype=int, device=device)
        idxs_all = torch.zeros((len(dset),), dtype=int, device=device)
        i = 0
        for imgs, labels, idxs in tqdm(dataloader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            bs = imgs.shape[0]
            logits_all[i:(i + bs), :] = model.forward(imgs)
            labels_all[i:(i + bs)] = labels
            idxs_all[i:(i + bs)] = idxs
            i += bs

        logits_all  = logits_all[:i]
        labels_all = labels_all[:i]
        idxs_all = idxs_all[:i]

        preds_all = logits_all.argmax(axis=-1)
        fold_accs[fold] = (labels_all == preds_all).sum().item() / i

print(np.mean(fold_accs))
# 256 0.8544657783174949
# 380 0.8867597241909522
# 412 0.8897974022229842
# 446 0.8912930678623925
# 480
# 512 0.8915734307736326
# 600 0.8919006973854053
# 650 0.8840482971608792
# 554 0.8807766357324294


# np.all(idxs_all.cpu().numpy()  == np.arange(len(idxs_all)))