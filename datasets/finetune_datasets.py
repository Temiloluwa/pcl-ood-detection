import numpy as np
import random
import os
import pandas as pd
from torchvision import datasets, transforms
from .constrastive_datasets import means, stds
from utils import load_pickle
from PIL import ImageFilter, Image
from collections import Counter



class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

normalize = lambda x: transforms.Normalize(mean=means[x],
                                    std=stds[x])

finetune_augmentation = lambda x: transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize(x)
    ])


def finetune_no_augmentation(x, center_crop=True): 

    if center_crop:
        return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize(x)
            ])
    else:
        print("retrieving aug without center-crop")
        return transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                normalize(x)
            ])



class CIFAR10TrainSet(datasets.CIFAR10):
    """
    CIFAR10TrainSet
    Original idx seed = 5
    """
    def __init__(self, root, transform, train=True, percent=0.1, new_idx=False):
        super().__init__(root=root, train=train, download=False, transform=transform)
        data = self.data
        targets = np.array(self.targets)
        num_classes = targets.max() + 1
        samples_p_class = int(len(data) * percent / num_classes) 
        label = np.array(self.targets)
        idx_classes = [np.where(label==i)[0] for i in range(num_classes)]
        idx = load_pickle("/data/datasets/percent_idx", f"CIFAR10_{int(percent*100)}_percent")
        if new_idx:
            np.random.seed(6, 100)
            idx = [np.random.choice(i, samples_p_class) for i in idx_classes]

        self.data = np.vstack([data[i] for i in idx])
        self.targets = []
        for i in range(num_classes):
            self.targets += list((np.zeros(samples_p_class) + i).astype(int))

        idx_collapsed = [i for arr in idx for i in arr]
        assert np.sum(np.array(targets[idx_collapsed]) - np.array(self.targets)) == 0, "error"
        
        idx = np.arange(len(idx_collapsed))
        np.random.shuffle(idx)
        self.targets = list(np.array(self.targets)[idx])
        self.data = self.data[idx]


class CIFAR100TrainSet(datasets.CIFAR100):
    """
    CIFAR100TrainSet
    """
    def __init__(self, root, transform, train=True, percent=0.1, new_idx=False):
        super().__init__(root=root, train=train, download=False, transform=transform)

        data = self.data
        targets = np.array(self.targets)
        num_classes = targets.max() + 1
        
        label = np.array(self.targets)
        idx_classes = [np.where(label==i)[0] for i in range(num_classes)]
        idx = load_pickle("/data/datasets/percent_idx", f"CIFAR100_{int(percent*100)}_percent")
        #samples_p_class = int(len(data) * percent / num_classes)
        samples_p_class = len(idx[0])
        if new_idx:
            idx = [np.random.choice(i, samples_p_class) for i in idx_classes]

        self.data = np.vstack([data[i] for i in idx])
        self.targets = []
        for i in range(num_classes):
            self.targets += list((np.zeros(samples_p_class) + i).astype(int))

        idx_collapsed = [i for arr in idx for i in arr]
        assert np.sum(np.array(targets[idx_collapsed]) - np.array(self.targets)) == 0, "error"
        
        idx = np.arange(len(idx_collapsed))
        np.random.shuffle(idx)
        self.targets = list(np.array(self.targets)[idx])
        self.data = self.data[idx]



class Imagenet30TrainSet(datasets.ImageFolder):
    """
    Imagenet30 train set
    """
    def __init__(self, root, transform=None, data_type="train", percent=0.1, new_idx=False):
        super().__init__(root=root, transform=transform)

        data_type = root.split(os.sep)[-1]
        if data_type == "train":
            self.gs_list = pd.read_pickle("/data/datasets/imgnet30_train_grayscale.pickle")
        else:
            self.gs_list = pd.read_pickle("/data/datasets/imgnet30_val_grayscale.pickle")
        
        self.imgs = list(filter (lambda x:x[0] not in self.gs_list, self.samples))
        self.samples = list(filter (lambda x:x[0] not in self.gs_list, self.samples))
        self.targets = [v for k,v in self.imgs]
        
        num_classes = len(np.unique(self.targets))
        sample_p_class = int(Counter(self.targets).most_common()[0][1] * percent)
        self.idx_classes = load_pickle("/data/datasets/percent_idx", f"Imagenet30_{int(percent*100)}_percent")
       
        if new_idx:
            print("Using new idx")
            self.idx_classes = \
            [np.random.choice(np.where(np.array(self.targets)==i)[0], sample_p_class)\
                            for i in range(num_classes)]
            self.idx_classes = [list(i) for i in self.idx_classes]
            
        new_imgs = []
        new_samples = []
        new_targets = []
        for i in self.idx_classes:
            new_imgs.extend(list(np.array(self.imgs)[i]))
            new_samples.extend(list(np.array(self.samples)[i]))
            new_targets.extend(list(np.array(self.targets)[i]))
        
        self.imgs = [(k, int(v)) for k,v in new_imgs]
        self.samples = [(k, int(v)) for k,v in new_samples]
        self.targets = [int(i) for i in new_targets]
