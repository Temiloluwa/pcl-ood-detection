import time
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from . import calMetric as m
from . import calData as d
from utils import load_finetuned_model
from distilled.models import ResNet18

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

normalize = lambda x: transforms.Normalize(mean=means[x],
                                     std=stds[x])

root_path = "/data/datasets"
criterion = nn.CrossEntropyLoss()
transform = lambda x: transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize(x)
])

start = time.time()

def test(args, ckpt_path, ood_dataset):
    if args.model_type == "finetuned":
        net1 = load_finetuned_model(args, ckpt_path)
    else:
        if args.pytorch_model:
            net1 = torchvision.models.__dict__['resnet18'](num_classes=args.num_classes)
        else:
            net1 = ResNet18(num_classes=args.num_classes)
        net1.load_state_dict(torch.load(ckpt_path)["model"])
        net1.cuda(args.eval_gpu)

    optimizer1 = optim.SGD(net1.parameters(), lr = 0, momentum = 0)
    net1.cuda(args.eval_gpu)
    net1.eval()

    if args.data == "CIFAR10":
        testset = torchvision.datasets.CIFAR10(root=root_path, train=False, download=True, transform=transform("CIFAR10"))
    else:
        testset = torchvision.datasets.CIFAR100(root=root_path, train=False, download=True, transform=transform("CIFAR100"))
    testloaderIn = torch.utils.data.DataLoader(testset, batch_size=1,
                                            shuffle=False, num_workers=2)
    if ood_dataset == "CIFAR10":
        testsetout = torchvision.datasets.CIFAR10(root=root_path, train=False, transform=transform(args.data))
    elif ood_dataset == "CIFAR100":
        testsetout = torchvision.datasets.CIFAR100(root=root_path, train=False, transform=transform(args.data))

    elif ood_dataset == "SVHN":
        testsetout = torchvision.datasets.SVHN(root="/data/datasets/svhn-data",\
                    transform=transform(args.data), split="test")
    elif ood_dataset == "LSUNCrop":
        testsetout = torchvision.datasets.ImageFolder(root="/data/datasets/LSUN_datasets/LSUN",\
                    transform=transform(args.data))
        
    elif ood_dataset == "LSUNResize":
        testsetout = torchvision.datasets.ImageFolder(root="/data/datasets/LSUN_datasets/LSUN_resize",\
                    transform=transform(args.data))
        
    testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=1,
                                         shuffle=False, num_workers=2)

    d.testData(net1, criterion, args.eval_gpu, testloaderIn, testloaderOut, args.magnitude, args.temp) 
    return m.metric(args.data, ood_dataset, args.arch)

    