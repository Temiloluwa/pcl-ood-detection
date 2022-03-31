import random
import torch
import os
import pandas as pd
import torchvision.datasets as datasets
from PIL import ImageFilter, Image
from torchvision.datasets.cifar import CIFAR10, CIFAR100
from torchvision.datasets import SVHN, STL10
from torchvision.transforms import transforms
import numpy as np


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class TwoCropsTransform2:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    

class ImageFolderInstance(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)           
        return sample, index


class CIFAR10Instance(CIFAR10):
    def __getitem__(self, index):
        sample = self.data[index]
        sample = Image.fromarray(sample)
        if self.transform is not None:
            sample = self.transform(sample)           
        return sample, index


class CIFAR100Instance(CIFAR100):
    def __getitem__(self, index):
        sample = self.data[index]
        sample = Image.fromarray(sample)
        if self.transform is not None:
            sample = self.transform(sample)           
        return sample, index


class SVHNInstance(SVHN):
    def __getitem__(self, index):
        sample = self.data[index]
        sample = Image.fromarray(sample.transpose(1,2,0))
        if self.transform is not None:
            sample = self.transform(sample)           
        return sample, index


class STL10Instance(STL10):
    def __getitem__(self, index):
        sample = self.data[index]
        sample = Image.fromarray(sample.transpose(1,2,0))
        if self.transform is not None:
            sample = self.transform(sample)           
        return sample, index


class LSUNCrop(datasets.ImageFolder):

    def __getitem__(self, index):
        sample = self.samples[index][0]
        sample = Image.open(sample)
        if self.transform is not None:
            sample = self.transform(sample)     
        return sample, index


class LSUNResize(datasets.ImageFolder):

    def __getitem__(self, index):
        sample = self.samples[index][0]
        sample = Image.open(sample)
        if self.transform is not None:
            sample = self.transform(sample)           
        return sample, index


class ImageNet30(datasets.ImageFolder):
    def __init__(self, **kwargs):
        super(ImageNet30, self).__init__(**kwargs)
        
        data_type = kwargs['root'].split(os.sep)[-1]
        if data_type == "train":
            self.gs_list = pd.read_pickle("/data/datasets/imgnet30_train_grayscale.pickle")
        else:
            self.gs_list = pd.read_pickle("/data/datasets/imgnet30_val_grayscale.pickle")
        
        self.imgs = list(filter (lambda x:x[0] not in self.gs_list, self.samples))
        self.targets = [v for k,v in self.imgs]
        self.samples = list(filter (lambda x:x[0] not in self.gs_list, self.samples))

    def __getitem__(self, index):
        sample = self.samples[index][0]
        sample = Image.open(sample)
        if self.transform is not None:
            sample = self.transform(sample)           
        return sample, index

class Rot90:
    """"Rotate Image if height > width"""
    def __call__(self, y):
        return y.transpose(Image.ROTATE_90) if y.size[0] < y.size[1] else y
    