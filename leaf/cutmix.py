
import numpy as np
import random
from torch.utils.data.dataset import Dataset

from leaf.cutmix_utils import onehot, rand_bbox
from leaf.dta import UnionDataSet


def len_robust(x):
    if hasattr(x, "len"):
        return len(x)
    else:
        return 1

class CutMix(Dataset):
    def __init__(self, dataset, num_class, num_mix=1, beta=1., prob=1.0, transform=None):
        self.dataset = dataset
        self.num_class = num_class
        self.num_mix = num_mix
        self.beta = beta
        self.prob = prob
        self.transform = transform
        if len_robust(dataset[0][1]) > 1:
            print("soft targets")
            self.soft_targets = True
        else:
            print("hard targets")
            self.soft_targets = False

    def __getitem__(self, index):
        img, lb, _ = self.dataset[index]
        if self.soft_targets:
            lb_onehot = lb
        else:
            lb_onehot = onehot(self.num_class, lb)

        for _ in range(self.num_mix):
            r = np.random.rand(1)
            if r > self.prob:
                continue

            # generate mixed sample
            lam = np.random.beta(self.beta, self.beta)
            rand_index = np.random.randint(0, len(self))

            img2, lb2, _ = self.dataset[rand_index]
            if self.soft_targets:
                lb2_onehot = lb2
            else:
                lb2_onehot = onehot(self.num_class, lb2)

            bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
            img[:, bbx1:bbx2, bby1:bby2] = img2[:, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))
            lb_onehot = lb_onehot * lam + lb2_onehot * (1. - lam)

            if self.transform:
                img = self.transform(image=img)["image"]

        return img, lb_onehot, index

    def __len__(self):
        return len(self.dataset)