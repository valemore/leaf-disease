
import numpy as np
import random
from torch.utils.data.dataset import Dataset

from leaf.cutmix_utils import onehot
from leaf.dta import UnionDataSet


class Mixup(Dataset):
    def __init__(self, dataset, num_class, beta=1., prob=1.0):
        self.dataset = dataset
        self.num_class = num_class
        self.beta = beta
        self.prob = prob
        if isinstance(dataset, UnionDataSet):
            assert len(dataset.one.labels.shape) == len(dataset.two.labels.shape)
            self.soft_targets = len(dataset.one.labels.shape) > 1
        else:
            self.soft_targets = len(dataset.labels.shape) > 1

    def __to_oh(self, lb):
        if self.soft_targets:
            return lb
        return onehot(self.num_class, lb)

    def __getitem__(self, index):
        img, lb, _ = self.dataset[index]
        lb = self.__to_oh(lb)

        r = np.random.rand(1)
        if r > self.prob:
            return img, lb, index

        lam = np.random.beta(self.beta, self.beta)

        rand_idx = np.random.randint(0, len(self))

        img2, lb2, _ = self.dataset[rand_idx]
        lb2 = self.__to_oh(lb2)

        img = lam * img + (1 - lam) * img2
        lb = lam * lb + (1 - lam) * lb2

        return img, lb, index

    def __len__(self):
        return len(self.dataset)


class MixupTransform(Dataset):
    def __init__(self, dataset, num_class, beta=1., prob=1.0):
        self.dataset = dataset
        self.num_class = num_class
        self.beta = beta
        self.prob = prob
        if isinstance(dataset, UnionDataSet):
            assert len(dataset.one.labels.shape) == len(dataset.two.labels.shape)
            self.soft_targets = len(dataset.one.labels.shape) > 1
        else:
            self.soft_targets = len(dataset.labels.shape) > 1

    def __to_oh(self, lb):
        if self.soft_targets:
            return lb
        return onehot(self.num_class, lb)

    def __getitem__(self, index):
        img, lb, _ = self.dataset[index]
        lb = self.__to_oh(lb)

        r = np.random.rand(1)
        if r > self.prob:
            if self.transform:
                img = self.transform(image=img)["image"]
            return img, lb, index

        lam = np.random.beta(self.beta, self.beta)

        rand_idx = np.random.randint(0, len(self))

        img2, lb2, _ = self.dataset[rand_idx]
        lb2 = self.__to_oh(lb2)

        img = lam * img + (1 - lam) * img2
        lb = lam * lb + (1 - lam) * lb2

        if self.transform:
            img = self.transform(image=img)["image"]

        return img, lb, index

    def __len__(self):
        return len(self.dataset)
