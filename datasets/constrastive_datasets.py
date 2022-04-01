import os
from contrastive.pcl.loader import *
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

# paths
CIFAR10DATA = "/data/datasets/CIFAR10"
CIFAR100DATA = "/data/datasets/CIFAR100"
SVHNDATA = "/data/datasets/svhn-data"
datadir  = "/data/datasets"
cifar_10_dir = os.path.join(datadir, "CIFAR10")
cifar_100_dir = os.path.join(datadir, "CIFAR100")
svhn_dir = os.path.join(datadir, "svhn-data")
lsun_crop_dir = os.path.join(datadir, 'LSUN_datasets', 'LSUN')
lsun_resize_dir = os.path.join(datadir, 'LSUN_datasets', "LSUN_resize")
imagenet30_dir = os.path.join(datadir, 'ImageNet30')

# image means and stds
means = {"CIFAR10":[0.491, 0.482, 0.447],
        "CIFAR100": [0.5071, 0.4865, 0.4409],
        "STL10":[0.4914, 0.4822, 0.4465],
         "SVHN": [0.4377, 0.4438, 0.4728],
         "LSUNCrop":[0.5, 0.5, 0.5],
         "LSUNResize": [0.5, 0.5, 0.5],
         "Imagenet30": [0.485, 0.456, 0.406]
    } 

stds = {"CIFAR10":[0.247, 0.243, 0.262],
        "CIFAR100": [0.2673, 0.2564, 0.2762],
        "STL10":[0.2471, 0.2435, 0.2616],
        "SVHN":[0.1980, 0.2010, 0.1970],
        "LSUNCrop": [0.5, 0.5, 0.5],
        "LSUNResize": [0.5, 0.5, 0.5],
        "Imagenet30": [0.229, 0.224, 0.225]
} 

def normalize(x):
    return transforms.Normalize(mean=means[x], std=stds[x])

eval_augmentation = lambda x: transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize(x)
    ])

# dataset dicts
train_dataset_dict = lambda x: {
    "CIFAR10": CIFAR10Instance(cifar_10_dir,
        train=True,
        transform=eval_augmentation(x)),

    "CIFAR100": CIFAR100Instance(cifar_100_dir,
        train=True,
        transform=eval_augmentation(x)),

    "SVHN": SVHNInstance(svhn_dir,
        split='train',
        transform=eval_augmentation(x)),

    "LSUNCrop": LSUNCrop(lsun_crop_dir,
                         transform=eval_augmentation(x)), 

    "LSUNResize": LSUNResize(lsun_resize_dir,
                         transform=eval_augmentation(x)),

    "Imagenet30": ImageNet30(root=os.path.join(imagenet30_dir, 'train'),
                         transform=eval_augmentation(x))
}


test_dataset_dict = lambda x: {
    "CIFAR10": CIFAR10Instance(cifar_10_dir,
        train=False,
        transform=eval_augmentation(x)),
    
    "CIFAR100": CIFAR100Instance(cifar_100_dir,
        train=False,
        transform=eval_augmentation(x)),
    
    "SVHN": SVHNInstance(svhn_dir,
        split='test',
        transform=eval_augmentation(x)),

    "LSUNCrop": LSUNCrop(lsun_crop_dir,
                         transform=eval_augmentation(x)),

    "LSUNResize": LSUNResize(lsun_resize_dir,
                         transform=eval_augmentation(x)),
    
    "Imagenet30": ImageNet30(root=os.path.join(imagenet30_dir, 'val'),
                        transform=eval_augmentation(x))
}

target_train_dataset = {"CIFAR10":CIFAR10(root=CIFAR10DATA, train=True, download=True), 
                       "CIFAR100":CIFAR100(root=CIFAR100DATA, train=True, download=True), 
                        "SVHN": SVHN(root=SVHNDATA, split="train", download=True),
                        "LSUNCrop": CIFAR10(root=CIFAR10DATA, train=False), 
                        "LSUNResize": CIFAR10(root=CIFAR10DATA, train=False),
                        "Imagenet30": ImageNet30(root=os.path.join(datadir, 'ImageNet30', 'train'))
                        }

target_test_dataset = {"CIFAR10":CIFAR10(root=CIFAR10DATA, train=False), 
                       "CIFAR100":CIFAR100(root=CIFAR100DATA, train=False),
                        "SVHN": SVHN(root=SVHNDATA, split="test", download=True),
                        "LSUNCrop": CIFAR10(root=CIFAR10DATA, train=False), 
                        "LSUNResize": CIFAR10(root=CIFAR10DATA, train=False),
                        "Imagenet30": ImageNet30(root=os.path.join(datadir, 'ImageNet30', 'val'))
                        }
