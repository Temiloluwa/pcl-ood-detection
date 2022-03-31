import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
from .constrastive_datasets import means, stds
import os
import pickle

def load_pickle(root, filename):
    filename = os.path.join(root, 
                     f'{filename}.pickle')
    with open(filename, 'rb') as f:
        return pickle.load(f)

distill_train_transform_dict = lambda x: {
    "default": transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(means[x], stds[x])]),

    "mocov1" : transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(means[x], stds[x])]),

    "mocov1_eval" : transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(means[x], stds[x])
    ])
}

distill_test_transform_dict =  lambda x: {
    "default": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(means[x], stds[x]),
    ]),
    "mocov1": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(means[x], stds[x])
        ]),

    "mocov1_eval" : transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(means[x], stds[x])
    ])
}



def get_distill_trainset(dataset):
    dataset_class = datasets.CIFAR10 if dataset == "CIFAR10" else datasets.CIFAR100
    num_classes = 10 if dataset == "CIFAR10" else 100

    class CIFARInstanceSample(dataset_class):
        """
        CIFAR10Instance+Sample Dataset
        """
        def __init__(self, root, train=True,
                    transform=None, target_transform=None,
                    download=False, k=4096, mode='exact', is_sample=True, percent=1.0):
            super().__init__(root=root, train=train, download=download,
                            transform=transform, target_transform=target_transform)
            self.k = k
            self.mode = mode
            self.is_sample = is_sample
            self.train_data = self.data
            self.train_labels = self.targets
            if self.train:
                num_samples = len(self.train_data)
                label = self.train_labels
            else:
                num_samples = len(self.test_data)
                label = self.test_labels

            self.cls_positive = [[] for i in range(num_classes)]
            for i in range(num_samples):
                self.cls_positive[label[i]].append(i)

            self.cls_negative = [[] for i in range(num_classes)]
            for i in range(num_classes):
                for j in range(num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
            self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

            if 0 < percent < 1:
                n = int(len(self.cls_negative[0]) * percent)
                self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                    for i in range(num_classes)]

            self.cls_positive = np.asarray(self.cls_positive)
            self.cls_negative = np.asarray(self.cls_negative)


        def __getitem__(self, index):
            if self.train:
                img, target = self.train_data[index], self.train_labels[index]
            else:
                img, target = self.test_data[index], self.test_labels[index]

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            if not self.is_sample:
                # directly return
                return img, target, index
            else:
                # sample contrastive examples
                if self.mode == 'exact':
                    pos_idx = index
                elif self.mode == 'relax':
                    pos_idx = np.random.choice(self.cls_positive[target], 1)
                    pos_idx = pos_idx[0]
                else:
                    raise NotImplementedError(self.mode)
                replace = True if self.k > len(self.cls_negative[target]) else False
                neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
                sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
                return img, target, index, sample_idx


    return CIFARInstanceSample



def get_distill_testset(dataset):
    return datasets.CIFAR10 if dataset == "CIFAR10" else datasets.CIFAR100