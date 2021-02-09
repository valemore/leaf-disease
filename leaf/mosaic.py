
import numpy as np
import random
from torch.utils.data.dataset import Dataset

from leaf.cutmix_utils import onehot
from leaf.dta import UnionDataSet


class Mosaic(Dataset):
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
        gam = np.random.beta(self.beta, self.beta)

        split_x = int(lam * img.shape[2])
        split_y = int(gam * img.shape[1])

        rand_idxs = np.random.randint(0, len(self), 3)

        lb = lam * gam * lb

        img2, lb2, _ = self.dataset[rand_idxs[0]]
        lb2 = self.__to_oh(lb2)
        img[:, :split_y, split_x:] = img2[:, :split_y, split_x:]
        lb += (1 - lam) * gam * lb2

        img2, lb2, _ = self.dataset[rand_idxs[1]]
        lb2 = self.__to_oh(lb2)
        img[:, split_y:, :split_x] = img2[:, split_y:, :split_x]
        lb += lam * (1-gam) * lb2

        img2, lb2, _ = self.dataset[rand_idxs[2]]
        lb2 = self.__to_oh(lb2)
        img[:, split_y:, split_x:] = img2[:, split_y:, split_x:]
        lb += (1-lam) * (1-gam) * lb2

        return img, lb, index

    def __len__(self):
        return len(self.dataset)


class MosaicTransform(Dataset):
    def __init__(self, dataset, num_class, beta=1., prob=1.0, transform=None):
        self.dataset = dataset
        self.num_class = num_class
        self.beta = beta
        self.prob = prob
        self.transform = transform
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
        gam = np.random.beta(self.beta, self.beta)

        split_x = int(lam * img.shape[1])
        split_y = int(gam * img.shape[0])

        rand_idxs = np.random.randint(0, len(self), 3)

        lb = lam * gam * lb

        img2, lb2, _ = self.dataset[rand_idxs[0]]
        lb2 = self.__to_oh(lb2)
        img[:split_y, split_x:, :] = img2[:split_y, split_x:, :]
        lb += (1 - lam) * gam * lb2

        img2, lb2, _ = self.dataset[rand_idxs[1]]
        lb2 = self.__to_oh(lb2)
        img[split_y:, :split_x, :] = img2[split_y:, :split_x, :]
        lb += lam * (1-gam) * lb2

        img2, lb2, _ = self.dataset[rand_idxs[2]]
        lb2 = self.__to_oh(lb2)
        img[split_y:, split_x:, :] = img2[split_y:, split_x:, :]
        lb += (1-lam) * (1-gam) * lb2

        if self.transform:
            img = self.transform(image=img)["image"]

        return img, lb, index

    def __len__(self):
        return len(self.dataset)
