import os
from contrastive.pcl.loader import *
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

CIFAR10DATA = "/data/datasets/CIFAR10"
CIFAR100DATA = "/data/datasets/CIFAR100"
SVHNDATA = "/data/datasets/svhn-data"
datadir  = "/data/datasets"

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


class Rot90:
    """"Rotate Image if height > width"""
    def __call__(self, y):
        return y.transpose(Image.ROTATE_90) if y.size[0] < y.size[1] else y


class ImageNet30(datasets.ImageFolder):
    def __init__(self, **kwargs):
        super(ImageNet30, self).__init__(**kwargs)
        
        data_type = kwargs['root'].split(os.sep)[-1]
        if data_type == "train":
            self.gs_list = pd.read_pickle("/data/datasets/imgnet30_train_grayscale.pickle")
        else:
            self.gs_list = pd.read_pickle("/data/datasets/imgnet30_val_grayscale.pickle")
        
        self.imgs = list(filter (lambda x:x[0] not in self.gs_list, self.samples))
        self.samples = list(filter (lambda x:x[0] not in self.gs_list, self.samples))
        self.targets = [v for k,v in self.imgs]


class CUB(datasets.ImageFolder):
    def __init__(self, *args, new_idx=False, reduced_data=True, **kwargs):
        super().__init__(*args, **kwargs)
        if reduced_data:
            num_classes = len(np.unique(self.targets))
            sample_p_class = 15
            
            self.idx_classes = pd.read_pickle("/data/datasets/percent_idx/cub.pickle")
            
            if new_idx:
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
            self.targets = new_targets


class StanfordDogs(datasets.ImageFolder):
    def __init__(self, *args, new_idx=False, reduced_data=True, **kwargs):
        super().__init__(*args, **kwargs)
        if reduced_data:
            num_classes = len(np.unique(self.targets))
            sample_p_class = 25
            
            
            self.idx_classes = pd.read_pickle("/data/datasets/percent_idx/sd.pickle")
            
            if new_idx:
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
            self.targets = new_targets

   

"""
augmentation used to get presentation results
eval_augmentation = lambda x, y: transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize(x)
    ])
"""
"""eval_augmentation = lambda x, y: transforms.Compose([
    #transforms.Resize(y),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize(x)
])
"""


def eval_augmentation(x, y, z):
    aug = [ 
            # 0 - exp 3 - imagenet30 (rot90)
            [transforms.Resize((375, 500)),
            transforms.ToTensor(),
            normalize(x)],

            # 1 - exp 4 - imagenet30 (rot90)
            [transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize(x)],

            # 2 - exp 5 - cifar10
            [transforms.Resize(96),
            transforms.ToTensor(),
            normalize(x)],
            
            # 3 - default - exp 6 - cifar10 und exp7 - imagenet30 (no rot90)
            [transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize(x)],
        ]

    aug = aug[y]

    if z:
        aug = [Rot90()] + aug
    
    aug = transforms.Compose(aug)
    return aug 
    


train_dataset_dict = lambda x, y, z: {
    "CIFAR10": CIFAR10Instance(os.path.join(datadir, "CIFAR10"),
        train=True,
        transform=eval_augmentation(x, y, z)),

    "CIFAR100": CIFAR100Instance(os.path.join(datadir, "CIFAR100"),
        train=True,
        transform=eval_augmentation(x, y, z)),

    "SVHN": SVHNInstance(os.path.join(datadir, "svhn-data"),
        split='train',
        transform=eval_augmentation(x, y, z)),

    "LSUNCrop": LSUNCrop(os.path.join(datadir, 'LSUN_datasets', 'LSUN'),
                         transform=eval_augmentation(x, y, z)), 

    "LSUNResize": LSUNResize(os.path.join(datadir, 'LSUN_datasets', "LSUN_resize"),
                         transform=eval_augmentation(x, y, z)),

    "Imagenet30": ImageNet30(root=os.path.join(datadir, 'ImageNet30', 'train'),
                         transform=eval_augmentation(x, y, z)),
    "CUB":CUB(root="/data/datasets/cub200/CUB_200_2011/images", transform=eval_augmentation(x, y, z)),
    "SD": StanfordDogs(root="/data/datasets/stanford_dogs", transform=eval_augmentation(x, y, z))
}


test_dataset_dict = lambda x, y, z: {
    "CIFAR10": CIFAR10Instance(os.path.join(datadir, "CIFAR10"),
        train=False,
        transform=eval_augmentation(x, y, z)),
    
    "CIFAR100": CIFAR100Instance(os.path.join(datadir, "CIFAR100"),
        train=False,
        transform=eval_augmentation(x, y, z)),
    
    "SVHN": SVHNInstance(os.path.join(datadir, "svhn-data"),
        split='test',
        transform=eval_augmentation(x, y, z)),

    "LSUNCrop": LSUNCrop(os.path.join(datadir, 'LSUN_datasets', 'LSUN'),
                         transform=eval_augmentation(x, y, z)),

    "LSUNResize": LSUNResize(os.path.join(datadir, 'LSUN_datasets', "LSUN_resize"),
                         transform=eval_augmentation(x, y, z)),
    
    "Imagenet30": ImageNet30(root=os.path.join(datadir, 'ImageNet30', 'val'),
                         transform=eval_augmentation(x, y, z)),
    "CUB":CUB(root="/data/datasets/cub200/CUB_200_2011/images", transform=eval_augmentation(x, y, z)),
    "SD": StanfordDogs(root="/data/datasets/stanford_dogs", transform=eval_augmentation(x, y, z))
}


target_train_dataset = {"CIFAR10":CIFAR10(root=CIFAR10DATA, train=True), 
                       "CIFAR100":CIFAR100(root=CIFAR100DATA, train=True), 
                        "SVHN": SVHN(root=SVHNDATA, split="train"),
                        "LSUNCrop": CIFAR10(root=CIFAR10DATA, train=False), 
                        "LSUNResize": CIFAR10(root=CIFAR10DATA, train=False),
                        "Imagenet30": ImageNet30(root=os.path.join(datadir, 'ImageNet30', 'train')),
                        "CUB": CUB(root="/data/datasets/cub200/CUB_200_2011/images"),
                        "SD": StanfordDogs(root="/data/datasets/stanford_dogs")
                        }

target_test_dataset = {"CIFAR10":CIFAR10(root=CIFAR10DATA, train=False), 
                       "CIFAR100":CIFAR100(root=CIFAR100DATA, train=False),
                        "SVHN": SVHN(root=SVHNDATA, split="test"),
                        "LSUNCrop": CIFAR10(root=CIFAR10DATA, train=False), 
                        "LSUNResize": CIFAR10(root=CIFAR10DATA, train=False),
                        "Imagenet30": ImageNet30(root=os.path.join(datadir, 'ImageNet30', 'val')),
                        "CUB":CUB(root="/data/datasets/cub200/CUB_200_2011/images"),
                        "SD": StanfordDogs(root="/data/datasets/stanford_dogs")
                        }
