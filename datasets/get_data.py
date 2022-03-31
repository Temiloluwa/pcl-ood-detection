import torch
import os
import pandas as pd
import numpy as np
from torch.utils import data
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets
from collections import Counter
from functools import partial

from .constrastive_datasets import train_dataset_dict, test_dataset_dict, target_train_dataset, target_test_dataset, ImageNet30, StanfordDogs, CUB
from .finetune_datasets import CIFAR10TrainSet, CIFAR100TrainSet, Imagenet30TrainSet, finetune_augmentation, finetune_no_augmentation
from .distill_datasets import get_distill_trainset, get_distill_testset, distill_train_transform_dict, distill_test_transform_dict 


def get_contrastive_loaders(args, dataset):
    if args.model_type != "distilled":
        batch_size = args.batch_size * 4
    else:
        batch_size = args.batch_size

    train_dataset = train_dataset_dict(args.data, args.aug_num, args.rot_90)[dataset]
    test_dataset = test_dataset_dict(args.data, args.aug_num, args.rot_90)[dataset]

    train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True)
    print(f"Train dataset:{dataset}, transform : {train_dataset.transform}")    

    test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True)

    print(f"Test dataset:{dataset} transform : {test_dataset.transform}")    

    print(f"train images {len(train_dataset)}, test images {len(test_dataset)}")
    return train_loader, test_loader



def get_contrastive_targets(dataset):
    train_ds = target_train_dataset[dataset]
    test_ds = target_test_dataset[dataset]
    train_target = torch.zeros(len(train_ds))
    test_target = torch.zeros(len(test_ds))
    
    for idx,(_, targ) in enumerate(train_ds):
        train_target[idx] = targ

    for idx,(_, targ) in enumerate(test_ds):
        test_target[idx] = targ
    print(f"Train target count {Counter(train_target.detach().numpy())}")
    print(f"Test target count {Counter(test_target.detach().numpy())}")
    return train_target, test_target



def get_finetune_train_loaders(dataset, percent, batch_size, augmentation):
    train_transform = finetune_augmentation if augmentation else finetune_no_augmentation
    if augmentation:
        print("Using Data Augmentation")
        
    if dataset == "CIFAR10":
        train_set = CIFAR10TrainSet(root="/data/datasets/CIFAR10", \
                        transform=train_transform("CIFAR10"), train=True, percent=percent)
        test_set = datasets.CIFAR10(root="/data/datasets/CIFAR10", transform=finetune_no_augmentation("CIFAR10"), train=False)
    elif dataset == "CIFAR100":
        train_set = CIFAR100TrainSet(root="/data/datasets/CIFAR100", \
                        transform=train_transform("CIFAR100"), train=True, percent=percent)
        test_set = datasets.CIFAR100(root="/data/datasets/CIFAR100", transform=finetune_no_augmentation("CIFAR100"), train=False)
    
    elif dataset == "Imagenet30":
        train_set = Imagenet30TrainSet(root="/data/datasets/ImageNet30/train", \
                        transform=train_transform("Imagenet30"), percent=percent)
        test_set = ImageNet30(root="/data/datasets/ImageNet30/val", transform=finetune_no_augmentation("Imagenet30"))
         

    train_loader =  DataLoader(train_set,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=2,
                            pin_memory=True)

    test_loader = DataLoader(test_set,
                            batch_size=batch_size * 2,
                            shuffle=False,
                            num_workers=2,
                            pin_memory=True)

        
    return train_loader, test_loader



def get_ensemble_eval_loaders(dataset, aug_dataset, batch_size, center_crop=True, shuffle=False):
    global finetune_no_augmentation
    finetune_no_augmentation = partial(finetune_no_augmentation, center_crop=center_crop)
    if dataset == "CIFAR10":
        train_set = datasets.CIFAR10(root="/data/datasets/CIFAR10",\
                transform=finetune_no_augmentation(aug_dataset), train=True)
        test_set = datasets.CIFAR10(root="/data/datasets/CIFAR10",\
                transform=finetune_no_augmentation(aug_dataset), train=False)
    
    elif dataset == "CIFAR100":
        train_set = datasets.CIFAR100(root="/data/datasets/CIFAR100",\
                transform=finetune_no_augmentation(aug_dataset), train=True)
        test_set = datasets.CIFAR100(root="/data/datasets/CIFAR100",\
                transform=finetune_no_augmentation(aug_dataset), train=False)

    elif dataset == "SVHN":
        train_set = datasets.SVHN(root="/data/datasets/svhn-data",\
                transform=finetune_no_augmentation(aug_dataset), split="train")
        test_set = datasets.SVHN(root="/data/datasets/svhn-data",\
                transform=finetune_no_augmentation(aug_dataset), split="test")

    elif dataset == "LSUNCrop":
        train_set = datasets.ImageFolder(root="/data/datasets/LSUN_datasets/LSUN",\
                transform=finetune_no_augmentation(aug_dataset))
        test_set = datasets.ImageFolder(root="/data/datasets/LSUN_datasets/LSUN",\
                transform=finetune_no_augmentation(aug_dataset))
    
    elif dataset == "LSUNResize":
        train_set = datasets.ImageFolder(root="/data/datasets/LSUN_datasets/LSUN_resize",\
                transform=finetune_no_augmentation(aug_dataset))
        test_set = datasets.ImageFolder(root="/data/datasets/LSUN_datasets/LSUN_resize",\
                transform=finetune_no_augmentation(aug_dataset))
    elif dataset == "Imagenet30":
        train_set = ImageNet30(root=os.path.join("/data/datasets", 'ImageNet30', 'train'),\
                transform=finetune_no_augmentation(aug_dataset))
        test_set = ImageNet30(root=os.path.join("/data/datasets", 'ImageNet30', 'val'),\
                transform=finetune_no_augmentation(aug_dataset))
    elif dataset == "SD":
        train_set =  StanfordDogs(root="/data/datasets/stanford_dogs",\
                transform=finetune_no_augmentation(aug_dataset), reduced_data=False)
        test_set =  StanfordDogs(root="/data/datasets/stanford_dogs",\
                transform=finetune_no_augmentation(aug_dataset), reduced_data=False)
    elif dataset == "CUB":
        train_set = CUB(root="/data/datasets/cub200/CUB_200_2011/images",\
                transform=finetune_no_augmentation(aug_dataset), reduced_data=False)
        test_set = CUB(root="/data/datasets/cub200/CUB_200_2011/images",\
                transform=finetune_no_augmentation(aug_dataset), reduced_data=False)

        
       


    train_loader =  DataLoader(train_set,
                            batch_size=batch_size * 2,
                            shuffle=shuffle,
                            num_workers=2,
                            pin_memory=True)

    test_loader = DataLoader(test_set,
                            batch_size=batch_size * 2,
                            shuffle=False,
                            num_workers=2,
                            pin_memory=True)

        
    return train_loader, test_loader



def get_distill_loaders(args):
    """
    cifar 10
    """

    data_folder = os.path.join("/data/datasets", args.data)
    train_transform = distill_train_transform_dict(args.data)["mocov1_eval"]
    test_transform = distill_test_transform_dict(args.data)["mocov1_eval"]
    
    train_set = get_distill_trainset(args.data)
    test_set = get_distill_testset(args.data)

    train_set = train_set(root=data_folder,
                            download=True,
                            train=True,
                            transform=train_transform,
                            k=args.nce_k,
                            mode=args.nce_mode,
                            is_sample=args.is_sample,
                            percent=args.percent)


    n_data = len(train_set)
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers)

    test_set = test_set(root=data_folder,
                        download=True,
                        train=False,
                        transform=test_transform)
                        
    test_loader = DataLoader(test_set,
                             batch_size=int(args.batch_size/2),
                             shuffle=False,
                             num_workers=int(args.num_workers/2))

    return train_loader, test_loader, n_data



def get_distill_loaders_2(args):
    """
    cifar 10
    """
    aug = "mocov1" if args.augmentation else "mocov1_eval"
    data_folder = os.path.join("/data/datasets", args.data)
    train_transform = distill_train_transform_dict(args.data)[aug]
    test_transform = distill_test_transform_dict(args.data)[aug]
    
    test_set = get_distill_testset(args.data)

    train_set = test_set(root=data_folder,
                        download=True,
                        train=True,
                        transform=train_transform)

    test_set = test_set(root=data_folder,
                        download=True,
                        train=False,
                        transform=test_transform)
    
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers)

    test_loader = DataLoader(test_set,
                             batch_size=int(args.batch_size *2),
                             shuffle=False,
                             num_workers=int(args.num_workers *2))

    return train_loader, test_loader

